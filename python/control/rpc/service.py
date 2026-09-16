from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any

from mojive.control.contracts import DEFAULT_RPC_LIMITS, RpcLimits
from mojive.control.errors import ControlError as RpcError

from .protocol import PROTOCOL_VERSION, _request_deadline, _response
from .stats import RpcStats


class ControlService:
    """Expose one transport-independent ControlApplication through protocol version 1."""

    def __init__(
        self,
        adapter=None,
        asset_path=None,
        *,
        session=None,
        camera=None,
        app=None,
        limits: RpcLimits = DEFAULT_RPC_LIMITS,
        stats: RpcStats | None = None,
    ):
        from mojive.control import ControlApplication

        self.application = ControlApplication(
            adapter, asset_path, session=session, camera=camera, app=app
        )
        self.limits = limits
        self.stats = stats or RpcStats(limits, "headless")
        self.application.rpc_stats = self.stats.snapshot
        self._slots = threading.BoundedSemaphore(limits.max_inflight_requests)

    def submit(self, request, *, request_bytes: int = 0, disconnected=None):
        """Dispatch a transport request; asynchronous handlers retain completion ownership."""
        if not self._slots.acquire(blocking=False):
            self.stats.increment("queue_rejected")
            return _response(
                request.get("id") if isinstance(request, dict) else None,
                error={
                    "code": "busy",
                    "message": "RPC has too many unfinished requests; request was not started",
                },
            )
        self.stats.increment("accepted")
        self.stats.inflight(1)
        started = self.stats.start()
        try:
            response = self._handle(request, disconnected=disconnected)
        except BaseException as error:
            self._complete(
                _response(None, error={"code": "internal_error", "message": str(error)}), started
            )
            raise
        if isinstance(response, Future):
            response.add_done_callback(lambda result: self._complete(result.result(), started))
        else:
            self._complete(response, started)
        return response

    def _complete(self, response, started):
        self.stats.finish(response, started)
        self.stats.inflight(-1)
        self._slots.release()

    def handle(self, request):
        """Validate and dispatch one request under the service's unfinished-work budget."""
        return self.submit(request)

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

    def _handle(self, request: dict[str, Any], *, disconnected=None) -> dict[str, Any] | Future:
        """Validate one protocol request and return a serializable response."""
        if not isinstance(request, dict):
            return _response(
                None, error={"code": "invalid_request", "message": "request must be a JSON object"}
            )
        request_id = request.get("id")
        version = request.get("version")
        if type(version) is not int or version != PROTOCOL_VERSION:
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
        from mojive.control.operations import OPERATIONS

        operation_version = request.get("operation_version", 1)
        operation = OPERATIONS.get(method)
        if type(operation_version) is not int or (
            operation is not None and operation_version != operation.version
        ):
            return _response(
                request_id,
                error={
                    "code": "operation_version_mismatch",
                    "message": f"Unsupported revision {operation_version!r} of {method}",
                },
            )
        try:
            deadline = _request_deadline(request)
            with self.application.lock:
                if disconnected is not None and disconnected():
                    self.stats.increment("cancelled")
                    raise RpcError("cancelled", "Peer disconnected before execution")
                if deadline is not None and time.monotonic() >= deadline:
                    self.stats.increment("expired")
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
            response.set_running_or_notify_cancel()

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
