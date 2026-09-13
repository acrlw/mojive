"""Simulation ownership and bounded snapshot publication contracts."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol


class SimulationDriver(Protocol):
    """Advance owned physics while the caller reads a stable displayed state.

    Only the owning Session calls these methods. suspend() fences all native
    work and binds the authoritative state for synchronous edits. poll() binds
    a completed snapshot without waiting. Both return newly observed step counts.
    """

    def start(self, speed: float, *, reset_clock: bool = True) -> None: ...
    def poll(self) -> int: ...
    def suspend(self) -> int: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Snapshot:
    slot: int
    step: int
    ready_ns: int


class SnapshotPool:
    """Keep published/leased storage immutable until its consumer releases it."""

    def __init__(self, size=3, *, latest=False):
        if size < 2:
            raise ValueError("At least two snapshot slots are required")
        self._condition = threading.Condition()
        self._free = deque(range(size))
        self._leased = set()
        self._ready = deque()
        self._latest = latest
        self._done = False
        self._closed = False
        self._error = None
        self.dropped = 0

    def reserve(self):
        with self._condition:
            self._condition.wait_for(lambda: self._free or self._closed)
            if self._closed:
                return None
            slot = self._free.popleft()
            self._leased.add(slot)
            return slot

    def publish(self, snapshot):
        with self._condition:
            if self._latest:
                while self._ready:
                    old = self._ready.popleft()
                    self._leased.remove(old.slot)
                    self._free.append(old.slot)
                    self.dropped += 1
            self._ready.append(snapshot)
            self._condition.notify_all()

    def acquire(self, *, wait=True):
        with self._condition:
            if wait:
                self._condition.wait_for(lambda: self._ready or self._done or self._closed)
            if self._error is not None:
                raise RuntimeError("Physics worker failed") from self._error
            return self._ready.popleft() if self._ready else None

    def release(self, slot):
        with self._condition:
            self._leased.remove(slot)
            self._free.append(slot)
            self._condition.notify_all()

    def finish(self, error=None):
        with self._condition:
            self._error = error
            self._done = True
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
