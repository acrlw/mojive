"""Measured simulation throughput from an authoritative step counter."""

from __future__ import annotations

import time


class StepRate:
    """Measure completed physics steps per wall second, allowing episode resets."""

    def __init__(self):
        self.value: float | None = None
        self._count: int | None = None
        self._source_time = 0.0
        self._started = 0.0
        self._steps = 0

    def update(self, count: int, source_time: float, *, now: float | None = None) -> float:
        """Return a rate averaged over at least 250 ms; a reset starts a new interval."""
        now = time.monotonic() if now is None else now
        if self._count is None or count < self._count or source_time < self._source_time:
            self._started, self._steps, self.value = now, 0, 0.0
        else:
            self._steps += count - self._count
            elapsed = now - self._started
            if elapsed >= 0.25:
                self.value = self._steps / elapsed
                self._started, self._steps = now, 0
        self._count, self._source_time = count, source_time
        return self.value
