"""Control budgets and diagnostic field names, independent of transport implementations."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RpcLimits:
    """Byte limits include the terminating newline; pump time is checked between requests."""

    max_request_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 256 * 1024 * 1024
    max_connections: int = 16
    max_inflight_requests: int = 128
    max_pending_requests: int = 128
    max_pending_bytes: int = 64 * 1024 * 1024
    requests_per_pump: int = 8
    pump_budget_ms: float = 2.0

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "pump_budget_ms":
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ValueError(f"{item.name} must be finite and positive")


DEFAULT_RPC_LIMITS = RpcLimits()

RPC_COUNTERS = (
    "accepted",
    "started",
    "completed",
    "failed",
    "cancelled",
    "expired",
    "completion_unknown",
    "queue_rejected",
    "connections_rejected",
    "oversize_requests",
    "oversize_responses",
    "queued_requests",
    "queued_bytes",
    "active_requests",
    "connections",
    "peak_connections",
    "peak_queued_requests",
    "pump_budget_exhausted",
    "inflight_requests",
)
RPC_TIMINGS = ("queue_wait_ms", "handler_ms", "pump_ms")
