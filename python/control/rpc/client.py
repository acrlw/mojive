from __future__ import annotations

import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any

from mojive.control.contracts import DEFAULT_RPC_LIMITS, RpcLimits
from mojive.control.errors import ControlError as RpcError

from .protocol import DEFAULT_SOCKET, PROTOCOL_VERSION, _validate_response


class RpcClient:
    """Persistent local control client with correlation, timeouts, and recovery."""

    def __init__(
        self,
        socket_path: Path = DEFAULT_SOCKET,
        timeout: float = 5.0,
        *,
        limits: RpcLimits = DEFAULT_RPC_LIMITS,
    ) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.timeout = float(timeout)
        self.limits = limits
        self._next_id = 1
        self._client: socket.socket | None = None
        self._lock = threading.Lock()

    def call(
        self, method: str, params: dict[str, Any] | None = None, *, operation_version: int = 1
    ) -> Any:
        """Send one request, validate correlation metadata, and return its result."""
        if not isinstance(method, str) or not method:
            raise RpcError("invalid_params", "Method must be a nonempty string")
        if type(operation_version) is not int or operation_version < 1:
            raise RpcError("invalid_params", "Operation revision must be a positive integer")
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
            if operation_version != 1:
                request["operation_version"] = operation_version
            try:
                encoded = (
                    json.dumps(request, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                )
            except (TypeError, ValueError) as exc:
                raise RpcError(
                    "invalid_params", f"Parameters must contain JSON values: {exc}"
                ) from exc
            if len(encoded) > self.limits.max_request_bytes:
                raise RpcError("request_too_large", "Request exceeds the configured byte limit")
            try:
                client = self._connect()
                write_error = None
                try:
                    client.sendall(encoded)
                except OSError as error:
                    # A budget rejection may arrive while a large write is still
                    # in progress. Read that response; never resend the request.
                    write_error = error
                try:
                    response = _read_response(
                        client, deadline=deadline, max_bytes=self.limits.max_response_bytes
                    )
                except (OSError, RpcError) as read_error:
                    if write_error is not None:
                        raise write_error from read_error
                    raise
                finally:
                    if write_error is not None:
                        self.close()
                _validate_response(response, request_id)
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
            if response.get("error") is not None:
                error = response["error"]
                if response.get("id") is None:
                    self.close()
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

    def get_state(self, *, observations: bool = True) -> dict[str, Any]:
        """Read physics state and optional sensor, contact, and actuator measurements."""
        return self.call("get_state", {"observations": observations})

    def capture_array(self, *, mode="rgb", width=640, height=480, encoding="raw"):
        """Capture the session scene directly to a NumPy array without disk I/O."""
        from mojive.capture import decode_image

        return decode_image(
            self.call(
                "capture",
                {
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "transport": "base64",
                    "encoding": encoding,
                },
            )
        )

    def capture_into(self, buffer, *, mode="rgb", surface=None) -> dict[str, Any]:
        """Write into a reusable SharedImage; return metadata without image serialization.

        The completed response is the write boundary: read ``buffer.array``
        afterwards and finish consuming it before the next write. On timeout,
        completion is unknown; discard this buffer instead of reusing it.
        ``surface='viewport'`` or ``'window'`` captures the presented UI and
        requires a buffer matching its current pixel dimensions.
        """
        descriptor = buffer.descriptor
        params = {"transport": "shared_memory", "buffer": descriptor}
        if surface is None:
            params.update(mode=mode, height=descriptor["shape"][0], width=descriptor["shape"][1])
            return self.call("capture", params)
        if mode != "rgb":
            raise ValueError("Presented capture requires RGB mode")
        params["surface"] = surface
        return self.call("capture_viewport", params)

    def set_ctrl(self, values) -> dict[str, Any]:
        return self.call("set_ctrl", {"values": list(values)})

    def step(self, count: int = 1, *, ctrl=None, observe: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {"count": count, "observe": observe}
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


def _read_response(
    client: socket.socket,
    *,
    deadline: float | None = None,
    max_bytes: int = DEFAULT_RPC_LIMITS.max_response_bytes,
) -> dict[str, Any]:
    data = bytearray()
    while not data.endswith(b"\n"):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("RPC response deadline expired")
            client.settimeout(remaining)
        chunk = client.recv(min(65536, max_bytes + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise RpcError("response_too_large", "RPC response exceeds the configured byte limit")
    if not data:
        raise RpcError("invalid_response", "RPC server closed without a response")
    if not data.endswith(b"\n"):
        raise RpcError("invalid_response", "RPC server closed before completing its response")
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise RpcError("invalid_response", "RPC response is not valid JSON") from exc
