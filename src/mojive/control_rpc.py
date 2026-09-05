"""Versioned local RPC for headless scene control and capture."""

from __future__ import annotations

import json
import math
import queue
import socket
import socketserver
import stat
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .control_errors import ControlError as RpcError

PROTOCOL_VERSION = 1
DEFAULT_SOCKET = Path("output/mojive.sock")


class ControlService:
    """Expose one transport-independent ControlApplication through protocol version 1."""

    def __init__(self, adapter=None, asset_path=None, *, session=None, camera=None, app=None):
        from .control import ControlApplication

        self.application = ControlApplication(
            adapter, asset_path, session=session, camera=camera, app=app
        )

    @property
    def session(self):
        """Return the caller-visible application session."""
        return self.application.session

    @property
    def camera(self):
        """Return the legacy capture-camera controller."""
        return self.application.camera

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """Call an application operation directly, using the same validation as RPC."""
        return self.application.dispatch(method, params)

    def close(self) -> None:
        """Close application-owned resources."""
        self.application.close()

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | Future:
        """Validate one protocol request and return a serializable response."""
        if not isinstance(request, dict):
            return _response(
                None, error={"code": "invalid_request", "message": "request must be a JSON object"}
            )
        request_id = request.get("id")
        version = request.get("version")
        if version != PROTOCOL_VERSION:
            return _response(
                request_id,
                error={
                    "code": "version_mismatch",
                    "message": f"Expected protocol version {PROTOCOL_VERSION}; received {version}",
                },
            )
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str) or not isinstance(params, dict):
            return _response(
                request_id,
                error={"code": "invalid_request", "message": "method and params are required"},
            )
        try:
            deadline = _request_deadline(request)
            with self.application.lock:
                if deadline is not None and time.monotonic() >= deadline:
                    raise RpcError("deadline_exceeded", "Request expired before execution")
                result = self.dispatch(method, params)
        except RpcError as exc:
            return _response(request_id, error=exc.payload())
        except Exception as exc:
            return _response(
                request_id,
                error={"code": "internal_error", "message": str(exc)},
            )
        if isinstance(result, Future):
            response = Future()

            def complete(source):
                try:
                    response.set_result(_response(request_id, result=source.result()))
                except RpcError as exc:
                    response.set_result(_response(request_id, error=exc.payload()))
                except Exception as exc:
                    response.set_result(
                        _response(request_id, error={"code": "capture_failed", "message": str(exc)})
                    )

            result.add_done_callback(complete)
            return response
        return _response(request_id, result=result)


@dataclass
class _PendingRequest:
    request: dict[str, Any]
    result: Future = field(default_factory=Future)


class ViewerControlService:
    """Marshal RPC requests onto an interactive viewer's UI thread."""

    def __init__(self, viewer) -> None:
        self._core = ControlService(
            session=viewer.session,
            camera=viewer.app.camera,
            app=viewer.app,
        )
        self._pending: queue.Queue[_PendingRequest] = queue.Queue()
        self._queue_lock = threading.Lock()
        self._closed = False

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            return self._core.handle(request)
        try:
            deadline = _request_deadline(request)
        except RpcError as exc:
            return _response(request.get("id"), error={"code": exc.code, "message": str(exc)})
        pending = _PendingRequest(request)
        with self._queue_lock:
            if self._closed:
                return _response(
                    request.get("id"),
                    error={"code": "unavailable", "message": "Viewer RPC service is closed"},
                )
            self._pending.put(pending)
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            return pending.result.result(timeout=timeout)
        except TimeoutError:
            cancelled = pending.result.cancel()
            return _response(
                request.get("id"),
                error={
                    "code": "deadline_exceeded" if cancelled else "completion_unknown",
                    "message": "Request expired before execution"
                    if cancelled
                    else "Request started before its deadline and may still complete",
                },
            )

    def pump(self, limit: int = 64) -> int:
        """Execute queued requests on the caller's viewer thread."""

        handled = 0
        while handled < max(1, int(limit)):
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                break
            if pending.result.set_running_or_notify_cancel():
                response = self._core.handle(pending.request)
                if isinstance(response, Future):
                    response.add_done_callback(
                        lambda value, result=pending.result: result.set_result(value.result())
                    )
                else:
                    pending.result.set_result(response)
            handled += 1
        return handled

    def close(self) -> None:
        with self._queue_lock:
            self._closed = True
            while True:
                try:
                    pending = self._pending.get_nowait()
                except queue.Empty:
                    break
                if pending.result.set_running_or_notify_cancel():
                    pending.result.set_result(
                        _response(
                            pending.request.get("id"),
                            error={
                                "code": "unavailable",
                                "message": "Viewer RPC service is closed",
                            },
                        )
                    )
        self._core.close()


class ViewerRpcServer:
    """Background socket transport attached to one interactive viewer."""

    def __init__(self, viewer, socket_path: Path = DEFAULT_SOCKET) -> None:
        self.service = ViewerControlService(viewer)
        self.server = ControlServer(socket_path, self.service)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="mojive-viewer-rpc",
            daemon=True,
        )
        viewer.app._rpc_service = self.service
        self._viewer = viewer
        self._closed = False
        self.thread.start()

    @property
    def socket_path(self) -> Path:
        return self.server.socket_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.server.server_close()
        self.service.close()
        if getattr(self._viewer.app, "_rpc_service", None) is self.service:
            self._viewer.app._rpc_service = None
        self.thread.join(timeout=2.0)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                line = self.rfile.readline()
            except OSError:
                return
            if not line:
                return
            try:
                request = json.loads(line)
                response = self.server.service.handle(request)
                if isinstance(response, Future):
                    deadline = _request_deadline(request)
                    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                    try:
                        response = response.result(timeout=timeout)
                    except TimeoutError:
                        response = _response(
                            request.get("id"),
                            error={
                                "code": "completion_unknown",
                                "message": "Request started before its deadline and may still complete",
                            },
                        )
            except Exception as exc:
                response = _response(None, error={"code": "invalid_request", "message": str(exc)})
            try:
                self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                self.wfile.flush()
            except OSError:
                return


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
        super().__init__(str(self.socket_path), _RequestHandler)
        info = self.socket_path.lstat()
        self._socket_identity = (info.st_dev, info.st_ino)
        self.socket_path.chmod(0o600)

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
        super().server_close()
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (info.st_dev, info.st_ino) == self._socket_identity:
            self.socket_path.unlink()
        self._socket_identity = None


class RpcClient:
    """Persistent local control client with correlation, timeouts, and recovery."""

    def __init__(self, socket_path: Path = DEFAULT_SOCKET, timeout: float = 5.0) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.timeout = float(timeout)
        self._next_id = 1
        self._client: socket.socket | None = None
        self._lock = threading.Lock()

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send one request, validate correlation metadata, and return its result."""
        if params is not None and not isinstance(params, dict):
            raise RpcError("invalid_params", "Parameters must be a JSON object")
        with self._lock:
            if not math.isfinite(self.timeout) or self.timeout <= 0.0:
                raise ValueError("RPC timeout must be finite and positive")
            deadline = time.monotonic() + self.timeout
            request_id = self._next_id
            self._next_id += 1
            request = {
                "version": PROTOCOL_VERSION,
                "id": request_id,
                "method": method,
                "params": params or {},
                "deadline": deadline,
            }
            try:
                client = self._connect()
                client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                response = _read_response(client, deadline=deadline)
            except TimeoutError as exc:
                self.close()
                raise RpcError(
                    "timeout",
                    f"RPC request timed out after {self.timeout:g} seconds; "
                    "execution may have started. Check state before retrying a mutation.",
                ) from exc
            except RpcError:
                self.close()
                raise
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.close()
                raise RpcError("connection_failed", str(exc)) from exc
            if response.get("version") != PROTOCOL_VERSION:
                self.close()
                raise RpcError("invalid_response", "RPC response version is incompatible")
            if response.get("id") != request_id:
                self.close()
                raise RpcError("invalid_response", "RPC response ID does not match the request")
            if response.get("error"):
                error = response["error"]
                raise RpcError(error["code"], error["message"], details=error.get("details"))
            return response.get("result")

    def hello(self) -> dict[str, Any]:
        return self.call("hello")

    def describe_operations(self, *, name=None, scope=None, available_only=False) -> dict[str, Any]:
        """Read live operation schemas and availability, optionally filtering the catalog."""
        params = {"available_only": available_only}
        if name is not None:
            params["name"] = name
        if scope is not None:
            params["scope"] = scope
        return self.call("describe_operations", params)

    def get_state(self) -> dict[str, Any]:
        return self.call("get_state")

    def set_ctrl(self, values) -> dict[str, Any]:
        return self.call("set_ctrl", {"values": list(values)})

    def step(self, count: int = 1, *, ctrl=None, observe: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {"count": int(count), "observe": bool(observe)}
        if ctrl is not None:
            params["ctrl"] = list(ctrl)
        return self.call("step", params)

    def _connect(self) -> socket.socket:
        if self._client is None:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(self.timeout)
            try:
                client.connect(str(self.socket_path))
            except Exception:
                client.close()
                raise
            self._client = client
        else:
            self._client.settimeout(self.timeout)
        return self._client

    def close(self) -> None:
        """Close the persistent connection; the next call reconnects."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> RpcClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _read_response(client: socket.socket, *, deadline: float | None = None) -> dict[str, Any]:
    data = bytearray()
    while not data.endswith(b"\n"):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("RPC response deadline expired")
            client.settimeout(remaining)
        chunk = client.recv(65536)
        if not chunk:
            break
        data.extend(chunk)
    if not data:
        raise RpcError("invalid_response", "RPC server closed without a response")
    return json.loads(data)


def _request_deadline(request: dict[str, Any]) -> float | None:
    """Read an optional deadline on the host's shared monotonic clock."""
    value = request.get("deadline")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RpcError("invalid_request", "deadline must be finite monotonic seconds")
    return float(value)


def _response(request_id, *, result=None, error=None) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "id": request_id,
        "result": result if error is None else None,
        "error": error,
    }
