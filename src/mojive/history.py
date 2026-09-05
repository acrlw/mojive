"""Document history storage with shared-resource memory accounting."""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import Generic, TypeVar

import numpy as np

State = TypeVar("State")


@dataclass(frozen=True)
class EditRecord(Generic[State]):
    """One committed document edit and its revision endpoints."""

    label: str
    before: State
    after: State
    before_revision: int
    after_revision: int


class EditHistory(Generic[State]):
    """Retain a contiguous Undo/Redo chain within record and Python memory limits.

    Shared arrays and their backing storage count once across the whole history.
    Opaque native allocations are not included in the Python byte estimate; the
    record limit also bounds the number of retained native adapter snapshots.
    """

    def __init__(self, *, record_limit: int = 100, byte_limit: int = 256 * 1024 * 1024) -> None:
        if record_limit <= 0 or byte_limit <= 0:
            raise ValueError("History limits must be positive")
        self.record_limit = int(record_limit)
        self.byte_limit = int(byte_limit)
        self.undo: list[EditRecord[State]] = []
        self.redo: list[EditRecord[State]] = []
        self.bytes = 0
        self._allocations: dict[int, dict[int, int]] = {}
        self._references: dict[int, int] = {}

    def append(self, record: EditRecord[State]) -> bool:
        """Append an edit, pruning oldest records; return whether it was retained."""
        self.clear_redo()
        allocations = _snapshot_allocations(record)
        self._allocations[id(record)] = allocations
        for identity, size in allocations.items():
            count = self._references.get(identity, 0)
            if not count:
                self.bytes += size
            self._references[identity] = count + 1
        self.undo.append(record)
        while self.undo and (len(self.undo) > self.record_limit or self.bytes > self.byte_limit):
            self._discard(self.undo.pop(0))
        return bool(self.undo and self.undo[-1] is record)

    def clear_redo(self) -> None:
        """Discard the alternate edit branch after a new mutation."""
        for record in self.redo:
            self._discard(record)
        self.redo.clear()

    def clear(self) -> None:
        """Release both history branches and their accounting."""
        self.undo.clear()
        self.redo.clear()
        self._allocations.clear()
        self._references.clear()
        self.bytes = 0

    def _discard(self, record: EditRecord[State]) -> None:
        for identity, size in self._allocations.pop(id(record)).items():
            count = self._references[identity] - 1
            if count:
                self._references[identity] = count
            else:
                del self._references[identity]
                self.bytes -= size


def _snapshot_allocations(root: object) -> dict[int, int]:
    """Visit retained Python objects once, including NumPy backing storage."""
    sizes: dict[int, int] = {}
    pending = [root]
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in sizes:
            continue
        sizes[identity] = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            if value.base is not None:
                pending.append(value.base)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif not isinstance(value, type) and hasattr(value, "__dict__"):
            pending.append(vars(value))
    return sizes
