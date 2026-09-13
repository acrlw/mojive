"""Bounded request timing samples and transport counters, independent of scene state."""

import threading
import time
from collections import deque
from dataclasses import asdict

from mojive.control.contracts import RPC_COUNTERS, RPC_TIMINGS, RpcLimits

SAMPLE_LIMIT = 256


class RpcStats:
    """Keep fixed-size rolling timing windows; snapshots never retain request contents."""

    def __init__(self, limits: RpcLimits, mode: str):
        self.limits = limits
        self.mode = mode
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(RPC_COUNTERS, 0)
        self._timings = {name: deque(maxlen=SAMPLE_LIMIT) for name in RPC_TIMINGS}

    def increment(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    def inflight(self, delta: int) -> None:
        with self._lock:
            self._counts["inflight_requests"] += delta

    def connection(self, delta: int) -> None:
        with self._lock:
            self._counts["connections"] += delta
            self._counts["peak_connections"] = max(
                self._counts["peak_connections"], self._counts["connections"]
            )

    def enqueue(self, size: int) -> None:
        with self._lock:
            self._counts["accepted"] += 1
            self._counts["queued_requests"] += 1
            self._counts["queued_bytes"] += size
            self._counts["peak_queued_requests"] = max(
                self._counts["peak_queued_requests"], self._counts["queued_requests"]
            )

    def dequeue(self, size: int) -> None:
        with self._lock:
            self._counts["queued_requests"] -= 1
            self._counts["queued_bytes"] -= size

    def start(self, queued_at: float | None = None) -> float:
        now = time.monotonic()
        with self._lock:
            self._counts["started"] += 1
            self._counts["active_requests"] += 1
            if queued_at is not None:
                self._timings["queue_wait_ms"].append((now - queued_at) * 1000)
        return now

    def finish(self, response: dict, started: float) -> None:
        with self._lock:
            self._counts["active_requests"] -= 1
            self._counts["completed"] += 1
            self._counts["failed"] += response.get("error") is not None
            self._timings["handler_ms"].append((time.monotonic() - started) * 1000)

    def pump(self, started: float, exhausted: bool) -> None:
        with self._lock:
            self._timings["pump_ms"].append((time.monotonic() - started) * 1000)
            self._counts["pump_budget_exhausted"] += exhausted

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            timings = {name: tuple(values) for name, values in self._timings.items()}
        return {
            "mode": self.mode,
            "limits": asdict(self.limits),
            "sample_limit": SAMPLE_LIMIT,
            "counters": counts,
            "timings": {name: _summary(values) for name, values in timings.items()},
        }


def _summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}

    def quantile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "maximum": ordered[-1],
    }
