from __future__ import annotations

import json
import select
import socket
import socketserver
import stat
import threading
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path

from .protocol import _request_deadline, _response
from .service import ControlService
from .waiting import wait_response


class _RequestHandler(socketserver.StreamRequestHandler):
    def _disconnected(self):
        try:
            ready, _, _ = select.select([self.request], [], [], 0)
            return bool(ready) and not self.request.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def handle(self) -> None:
        while True:
            try:
                line = self.rfile.readline(self.server.limits.max_request_bytes + 1)
            except OSError:
                return
            if not line:
                return
            if len(line) > self.server.limits.max_request_bytes:
                self.server.stats.increment("oversize_requests")
                self._write(
                    _response(
                        None,
                        error={
                            "code": "request_too_large",
                            "message": "Request exceeds the configured byte limit",
                        },
                    )
                )
                return
            if not line.endswith(b"\n"):
                return
            try:
                request = json.loads(line)
                response = self.server.service.submit(
                    request,
                    request_bytes=len(line),
                    disconnected=self._disconnected,
                )
                if isinstance(response, Future):
                    response = wait_response(
                        response,
                        request.get("id"),
                        _request_deadline(request),
                        self.server.stats,
                        disconnected=self._disconnected,
                    )
            except Exception as exc:
                response = _response(None, error={"code": "invalid_request", "message": str(exc)})
            if response is None or not self._write(response):
                return

    def _write(self, response):
        encoded = self.server.encode_response(response)
        if encoded is None:
            return False
        try:
            self.wfile.write(encoded)
            self.wfile.flush()
            return True
        except OSError:
            return False


class ControlServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Concurrent newline-delimited JSON server over an AF_UNIX socket."""

    daemon_threads = True

    def __init__(self, socket_path: Path, service: ControlService) -> None:
        path = Path(socket_path).expanduser()
        # Resolve the parent, retaining a final symlink so it can be rejected.
        self.socket_path = path.parent.resolve() / path.name
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._socket_identity: tuple[int, int] | None = None
        self._remove_stale_socket()
        self.service = service
        self.limits = service.limits
        self.stats = service.stats
        # Unix connect() can fail immediately when the listen backlog is full.
        # Admit a configured burst at the socket boundary before applying slots.
        self.request_queue_size = self.limits.max_connections
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        self._closed = False
        super().__init__(str(self.socket_path), _RequestHandler)
        info = self.socket_path.lstat()
        self._socket_identity = (info.st_dev, info.st_ino)
        self.socket_path.chmod(0o600)

    def process_request(self, request, client_address):
        with self._connections_lock:
            admitted = not self._closed and len(self._connections) < self.limits.max_connections
            if admitted:
                self._connections.add(request)
                self.stats.connection(1)
        if not admitted:
            self.stats.increment("connections_rejected")
            reply = self.encode_response(
                _response(
                    None,
                    error={
                        "code": "busy",
                        "message": "RPC connection limit reached; request was not started",
                    },
                )
            )
            try:
                request.settimeout(0.1)
                if reply is not None:
                    request.sendall(reply)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_connection(request)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_connection(request)

    def _release_connection(self, request):
        with self._connections_lock:
            self._connections.remove(request)
            self.stats.connection(-1)

    def encode_response(self, response):
        """Validate the complete wire response before writing any bytes.

        ASCII escaping makes character and byte lengths identical. Keep the C
        encoder fast path; decoded response objects and its temporary string are
        application-owned allocations, outside the wire byte budget.
        """
        encoded = json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n"
        if len(encoded) > self.limits.max_response_bytes:
            self.stats.increment("oversize_responses")
            response = _response(
                response.get("id"),
                error={
                    "code": "response_too_large",
                    "message": "Response exceeds the configured byte limit; the operation may have completed",
                },
            )
            encoded = json.dumps(response, separators=(",", ":")) + "\n"
        return encoded.encode() if len(encoded) <= self.limits.max_response_bytes else None

    def _remove_stale_socket(self) -> None:
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode):
            raise FileExistsError(f"RPC path exists and is not a socket: {self.socket_path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                probe.connect(str(self.socket_path))
            except ConnectionRefusedError:
                current = self.socket_path.lstat()
                if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                    self.socket_path.unlink()
                    return
            except FileNotFoundError:
                return
        raise FileExistsError(f"RPC socket is already in use: {self.socket_path}")

    def server_close(self) -> None:
        with self._connections_lock:
            self._closed = True
            connections = tuple(self._connections)
        for connection in connections:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
        super().server_close()
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (info.st_dev, info.st_ino) == self._socket_identity:
            self.socket_path.unlink()
        self._socket_identity = None
