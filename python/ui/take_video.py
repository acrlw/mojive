"""Pace a one-shot take replay against encoded frames, independently of display rate."""

from dataclasses import dataclass


@dataclass
class TakeVideo:
    frame_count: int
    structure_generation: int
    end_hold_frames: int
    cursor: int = 0
    playing: bool = False
    started: bool = False
    frame_due: bool = False
    elapsed: float = 0.0
    tail_frames: int | None = None

    def prepare_frame(self, dt: float, fps: float) -> float:
        """Return replay time for this display frame; encode at most one new image."""
        period = 1.0 / fps
        if not self.started:
            self.started = True
            self.frame_due = True
            return 0.0
        # Encoder or display stalls may slow the export in wall time. Advancing
        # by encoded time preserves the complete motion and its video duration.
        self.elapsed = min(period * 2, self.elapsed + max(0.0, dt))
        self.frame_due = self.elapsed + 1e-12 >= period
        if self.frame_due:
            self.elapsed = max(0.0, self.elapsed - period)
        return period if self.frame_due else 0.0

    def captured(self, *, at_end: bool) -> bool:
        """Include the final pose, then count any additional end-hold images."""
        if at_end:
            self.tail_frames = 0 if self.tail_frames is None else self.tail_frames + 1
            return self.tail_frames >= self.end_hold_frames
        return False
