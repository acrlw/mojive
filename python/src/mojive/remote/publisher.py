"""Remote: publisher."""

from __future__ import annotations

import contextlib
import pickle
import queue
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, Listener
from typing import Any

from mojive.adapters.base import (
    SceneFrame,
)
from mojive.commands import CommandResult

from .protocol import (
    AUTHKEY,
    DEFAULT_PORT,
    RemoteFrame,
    RemoteStructure,
    _close_connection,
)


class _LatestSender:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self._reliable: deque[bytes] = deque()
        self._latest: bytes | None = None
        self._condition = threading.Condition()
        self.closed = False
        threading.Thread(target=self._run, name="mojive-remote-send", daemon=True).start()

    def reliable(self, payload: bytes, *, clear_latest: bool = False) -> None:
        with self._condition:
            if clear_latest:
                self._latest = None
            self._reliable.append(payload)
            self._condition.notify()

    def latest(self, payload: bytes) -> None:
        with self._condition:
            self._latest = payload
            self._condition.notify()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.closed or self._reliable or self._latest is not None
                    )
                    if self.closed:
                        return
                    if self._reliable:
                        payload = self._reliable.popleft()
                    else:
                        payload = self._latest
                        self._latest = None
                self.connection.send_bytes(payload)
        except (EOFError, OSError):
            pass
        finally:
            self.close()

    def close(self) -> None:
        with self._condition:
            if self.closed:
                return
            self.closed = True
            self._condition.notify_all()
        _close_connection(self.connection)


@dataclass
class _CommandRequest:
    payload: dict
    result: Future = field(default_factory=Future)


class SnapshotPublisher:
    """Transport reliable scene structure and latest-only frames."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        command_timeout: float = 10.0,
    ) -> None:
        self.host, self.port = host, int(port)
        self.command_port = self.port + 1
        self._state_listener = Listener((host, self.port), authkey=AUTHKEY)
        try:
            self._command_listener = Listener((host, self.command_port), authkey=AUTHKEY)
        except Exception:
            self._state_listener.close()
            raise
        self._clients: list[_LatestSender] = []
        self._clients_lock = threading.Lock()
        self._command_clients: set[Connection] = set()
        self._command_clients_lock = threading.Lock()
        self._commands: queue.SimpleQueue[_CommandRequest] = queue.SimpleQueue()
        self._structure: bytes | None = None
        self._structure_revision = -1
        self._frame: bytes | None = None
        self._frame_sequence = 0
        self._command_timeout = max(0.0, float(command_timeout))
        self._closed = False
        threading.Thread(target=self._accept_state, name="mojive-remote-state", daemon=True).start()
        threading.Thread(
            target=self._accept_commands, name="mojive-remote-command", daemon=True
        ).start()

    def publish_structure(self, structure: RemoteStructure) -> None:
        """Reliably publish stable structure to current and future clients."""
        payload = pickle.dumps(structure, protocol=pickle.HIGHEST_PROTOCOL)
        self._structure = payload
        self._structure_revision = int(structure.structure_revision)
        self._frame = None
        with self._clients_lock:
            self._clients = [client for client in self._clients if not client.closed]
            clients = tuple(self._clients)
        for client in clients:
            client.reliable(payload, clear_latest=True)

    def publish_frame(self, frame: SceneFrame, debug_commands=None) -> int:
        """Publish a latest-only frame and return its sequence number."""
        self._frame_sequence += 1
        with self._clients_lock:
            self._clients = [client for client in self._clients if not client.closed]
            clients = tuple(self._clients)
        # Retain one bootstrap frame for a future viewer, but do not serialize
        # every training step while nobody is connected.
        if not clients and self._frame is not None:
            return self._frame_sequence
        commands = frame.debug_commands if debug_commands is None else debug_commands
        packet = RemoteFrame(
            frame_sequence=self._frame_sequence,
            frame=frame,
            debug_commands=tuple(commands or ()),
            structure_revision=self._structure_revision,
        )
        payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        self._frame = payload
        for client in clients:
            client.latest(payload)
        return self._frame_sequence

    def pump_commands(self, handler: Callable[[dict], Any], budget: int = 256) -> int:
        """Handle queued viewer commands on the publisher thread."""
        count = 0
        for _ in range(max(0, int(budget))):
            try:
                request = self._commands.get_nowait()
            except queue.Empty:
                break
            if not request.result.set_running_or_notify_cancel():
                count += 1
                continue
            try:
                result = handler(request.payload)
            except Exception as exc:
                result = CommandResult.bad(f"remote command failed: {exc}")
            request.result.set_result(result)
            count += 1
        return count

    def _accept_state(self) -> None:
        while not self._closed:
            try:
                connection = self._state_listener.accept()
            except (EOFError, OSError):
                return
            client = _LatestSender(connection)
            with self._clients_lock:
                self._clients.append(client)
            if self._structure is not None:
                client.reliable(self._structure)
            if self._frame is not None:
                client.latest(self._frame)

    def _accept_commands(self) -> None:
        while not self._closed:
            try:
                connection = self._command_listener.accept()
            except (EOFError, OSError):
                return
            with self._command_clients_lock:
                if self._closed:
                    connection.close()
                    return
                self._command_clients.add(connection)
            threading.Thread(
                target=self._command_client,
                args=(connection,),
                name="mojive-remote-command-client",
                daemon=True,
            ).start()

    def _command_client(self, connection: Connection) -> None:
        try:
            while not self._closed:
                request = _CommandRequest(connection.recv())
                self._commands.put(request)
                try:
                    result = request.result.result(timeout=self._command_timeout)
                except TimeoutError:
                    cancelled = request.result.cancel()
                    result = CommandResult.bad(
                        "remote command timed out before execution"
                        if cancelled
                        else "remote command timed out; completion unknown; inspect before retrying"
                    )
                connection.send(result)
        except (EOFError, OSError):
            pass
        finally:
            with self._command_clients_lock:
                self._command_clients.discard(connection)
            with contextlib.suppress(OSError):
                connection.close()

    def close(self) -> None:
        """Close listeners and all connected clients."""
        if self._closed:
            return
        self._closed = True
        self._state_listener.close()
        self._command_listener.close()
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.close()
        with self._command_clients_lock:
            command_clients, self._command_clients = self._command_clients, set()
        for connection in command_clients:
            with contextlib.suppress(OSError):
                _close_connection(connection)
