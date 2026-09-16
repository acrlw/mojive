"""Bounded RPC scheduling on the existing Viewer owner thread."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mojive.control.contracts import DEFAULT_RPC_LIMITS, RpcLimits
from mojive.control.errors import ControlError as RpcError

from .protocol import DEFAULT_SOCKET, _request_deadline, _response
from .server import ControlServer
from .service import ControlService
from .stats import RpcStats
from .waiting import wait_response


@dataclass
class _PendingRequest:
    request: dict[str, Any]
    size: int
    deadline: float | None
    disconnected: Callable[[], bool] | None
    queued_at: float
    result: Future = field(default_factory=Future)


class ViewerControlService:
    """Queue bounded requests; only pump executes scene operations on the Viewer thread."""

    def __init__(self, viewer, *, limits: RpcLimits = DEFAULT_RPC_LIMITS) -> None:
        self.limits = limits
        self.stats = RpcStats(limits, "viewer")
        self._core = ControlService(
            session=viewer.session,
            camera=viewer.app.camera,
            app=viewer.app,
            limits=limits,
            stats=self.stats,
        )
        self._pending: queue.Queue[_PendingRequest] = queue.Queue(limits.max_pending_requests)
        self._queue_lock = threading.Lock()
        self._queued_bytes = 0
        self._inflight = 0
        self._closed = False
        self._active_response: Future | None = None

    def submit(self, request, *, request_bytes: int | None = None, disconnected=None):
        """Admit a request without waiting; cancellation remains possible until pump starts it."""
        if not isinstance(request, dict):
            return self._core._handle(request)
        request_id = request.get("id")
        try:
            deadline = _request_deadline(request)
            if deadline is not None and time.monotonic() >= deadline:
                self.stats.increment("expired")
                return _response(
                    request_id,
                    error={
                        "code": "deadline_exceeded",
                        "message": "Request expired before execution",
                    },
                )
            size = (
                request_bytes
                if request_bytes is not None
                else len(json.dumps(request, separators=(",", ":"), allow_nan=False).encode()) + 1
            )
        except (RpcError, TypeError, ValueError) as error:
            return _response(request_id, error={"code": "invalid_request", "message": str(error)})
        if size > self.limits.max_request_bytes:
            self.stats.increment("oversize_requests")
            return _response(
                request_id,
                error={
                    "code": "request_too_large",
                    "message": "Request exceeds the configured byte limit",
                },
            )
        with self._queue_lock:
            if self._closed:
                return _response(
                    request_id,
                    error={
                        "code": "unavailable",
                        "message": "Viewer RPC service is closed",
                    },
                )
            if (
                self._pending.full()
                or self._queued_bytes + size > self.limits.max_pending_bytes
                or self._inflight >= self.limits.max_inflight_requests
            ):
                self.stats.increment("queue_rejected")
                return _response(
                    request_id,
                    error={
                        "code": "busy",
                        "message": "Viewer RPC queue is full; request was not started",
                    },
                )
            pending = _PendingRequest(request, size, deadline, disconnected, time.monotonic())
            self._queued_bytes += size
            self._inflight += 1
            self.stats.inflight(1)
            self.stats.enqueue(size)
            pending.result.add_done_callback(self._release_request)
            self._pending.put_nowait(pending)
        return pending.result

    def _release_request(self, result):
        with self._queue_lock:
            self._inflight -= 1
        self.stats.inflight(-1)
        if result.cancelled():
            self.stats.increment("cancelled")

    def handle(self, request):
        """Preserve the synchronous service interface used by embedded callers."""
        result = self.submit(request)
        if not isinstance(result, Future):
            return result
        return wait_response(result, request.get("id"), _request_deadline(request), self.stats)

    def _take(self):
        with self._queue_lock:
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                return None
            self._queued_bytes -= pending.size
            self.stats.dequeue(pending.size)
            return pending

    def pump(self, limit: int | None = None, *, budget_ms: float | None = None) -> int:
        """Execute ordered requests up to count/time limits, checked between handlers.

        At least one queued item can advance per call. A running handler cannot be
        preempted; remaining valid requests stay queued for a subsequent frame.
        """
        count = self.limits.requests_per_pump if limit is None else limit
        budget = self.limits.pump_budget_ms if budget_ms is None else budget_ms
        if type(count) is not int or count <= 0:
            raise ValueError("limit must be a positive integer")
        if type(budget) not in (int, float) or not 0 < budget < float("inf"):
            raise ValueError("budget_ms must be finite and positive")
        if self._active_response is not None:
            if not self._active_response.done():
                return 0
            self._active_response = None
        if self._pending.empty():
            return 0
        started = time.monotonic()
        handled = 0
        while handled < count:
            if handled and (time.monotonic() - started) * 1000 >= budget:
                break
            pending = self._take()
            if pending is None:
                break
            handled += 1
            if pending.result.cancelled():
                continue
            if pending.disconnected is not None and pending.disconnected():
                pending.result.cancel()
                continue
            if pending.deadline is not None and time.monotonic() >= pending.deadline:
                if pending.result.cancel():
                    self.stats.increment("expired")
                continue
            if not pending.result.set_running_or_notify_cancel():
                continue
            request_started = self.stats.start(pending.queued_at)
            response = self._core._handle(pending.request)
            if isinstance(response, Future):
                # Deferred document writes own Session until their UI-thread
                # completion. Later requests retain ordering and preconditions.
                from mojive.control.operations import OPERATIONS

                operation = OPERATIONS.get(pending.request.get("method"))
                if operation is not None and operation.writes_document:
                    self._active_response = response
                response.add_done_callback(
                    lambda value, target=pending.result, begin=request_started: self._complete(
                        target, value.result(), begin
                    )
                )
                if self._active_response is response:
                    break
            else:
                self._complete(pending.result, response, request_started)
        if handled:
            exhausted = handled >= count or (time.monotonic() - started) * 1000 >= budget
            self.stats.pump(started, exhausted and not self._pending.empty())
        return handled

    def _complete(self, target, response, started):
        self.stats.finish(response, started)
        target.set_result(response)

    def close(self) -> None:
        with self._queue_lock:
            self._closed = True
        while (pending := self._take()) is not None:
            if pending.result.set_running_or_notify_cancel():
                self.stats.increment("cancelled")
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

    def __init__(
        self, viewer, socket_path: Path = DEFAULT_SOCKET, *, limits: RpcLimits = DEFAULT_RPC_LIMITS
    ) -> None:
        self.service = ViewerControlService(viewer, limits=limits)
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
