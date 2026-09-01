"""Bounded in-editor output history and transient viewport status messages."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class OutputMessage:
    sequence: int
    timestamp: str
    level: str
    text: str


@dataclass(frozen=True)
class ViewportStatus:
    message: OutputMessage
    expires_at: float | None


class OutputBuffer:
    """Keep diagnostic history without making the render loop own log allocations."""

    def __init__(self, capacity: int = 1000) -> None:
        self._entries: deque[OutputMessage] = deque(maxlen=max(1, int(capacity)))
        self._lock = threading.Lock()
        self._sequence = 0
        self._status: ViewportStatus | None = None

    def write(
        self,
        text: str,
        *,
        level: str = "info",
        timestamp: str | None = None,
    ) -> OutputMessage | None:
        value = str(text).strip()
        if not value:
            return None
        with self._lock:
            self._sequence += 1
            message = OutputMessage(
                self._sequence,
                timestamp or datetime.now().strftime("%H:%M:%S"),
                str(level).lower(),
                value,
            )
            self._entries.append(message)
            return message

    def publish(
        self,
        text: str,
        *,
        level: str = "info",
        duration: float | None = 5.0,
    ) -> OutputMessage | None:
        message = self.write(text, level=level)
        if message is None:
            return None
        expires_at = None if duration is None else time.monotonic() + max(0.0, float(duration))
        with self._lock:
            self._status = ViewportStatus(message, expires_at)
        return message

    def entries(self) -> tuple[OutputMessage, ...]:
        with self._lock:
            return tuple(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def active_status(self, now: float | None = None) -> OutputMessage | None:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            status = self._status
            if status is None:
                return None
            if status.expires_at is not None and current >= status.expires_at:
                self._status = None
                return None
            return status.message

    def loguru_sink(self, formatted: Any) -> None:
        """Accept one Loguru message while retaining the component and severity."""

        record = getattr(formatted, "record", {})
        extra = record.get("extra", {})
        component = extra.get("component", "runtime")
        message = str(record.get("message", str(formatted))).strip()
        level = str(getattr(record.get("level"), "name", "info")).lower()
        timestamp = record.get("time")
        stamp = timestamp.strftime("%H:%M:%S") if timestamp is not None else None
        self.write(f"[mojive/{component}] {message}", level=level, timestamp=stamp)

    def copy_text(self, entries: Iterable[OutputMessage] | None = None) -> str:
        selected = self.entries() if entries is None else entries
        return "\n".join(
            f"{entry.timestamp} [{entry.level.upper()}] {entry.text}" for entry in selected
        )
