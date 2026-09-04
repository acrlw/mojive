"""Capture and interactive recording contracts shared by viewer frontends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CaptureSurface(StrEnum):
    """Choose which composed image a capture or recording contains."""

    SCENE = "scene"
    VIEWPORT = "viewport"
    WINDOW = "window"


class RecordingPhase(StrEnum):
    """Lifecycle state of an interactive viewer recording."""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


@dataclass(frozen=True)
class RecordingInfo:
    """Read-only snapshot of the interactive recorder state."""

    phase: RecordingPhase = RecordingPhase.IDLE
    surface: CaptureSurface = CaptureSurface.SCENE
    path: Path | None = None
    fps: float = 30.0
    frames: int = 0
    duration: float = 0.0

    @property
    def active(self) -> bool:
        return self.phase is not RecordingPhase.IDLE


__all__ = ["CaptureSurface", "RecordingInfo", "RecordingPhase"]
