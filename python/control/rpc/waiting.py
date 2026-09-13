"""Wait for protocol completion without confusing queued cancellation with execution."""

import time
from concurrent.futures import CancelledError

from .protocol import _response


def wait_response(future, request_id, deadline, stats, *, disconnected=None):
    """Cancel unstarted work on timeout/disconnect; never retry an active mutation."""
    while True:
        if disconnected is not None and disconnected():
            future.cancel()
            return None
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        timeout = remaining
        if disconnected is not None:
            timeout = 0.02 if remaining is None else min(remaining, 0.02)
        try:
            return future.result(timeout=timeout)
        except CancelledError:
            return _response(
                request_id,
                error={
                    "code": "deadline_exceeded",
                    "message": "Request cancelled before execution",
                },
            )
        except TimeoutError:
            if deadline is None or time.monotonic() < deadline:
                continue
            cancelled = future.cancel()
            stats.increment("expired" if cancelled else "completion_unknown")
            return _response(
                request_id,
                error={
                    "code": "deadline_exceeded" if cancelled else "completion_unknown",
                    "message": "Request expired before execution"
                    if cancelled
                    else "Request started before its deadline and may still complete",
                },
            )
