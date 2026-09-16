"""Transient timeline gestures and edit decisions, independent of UI and document owners."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .timeline import TimelineMarkerIndex, nearest_take_frame


@dataclass
class TimelineEditorState:
    """One panel's selection and gesture state; methods never mutate a scene or take."""

    selected_id: int = -1
    selected_keyframes: set[int] = field(default_factory=set)
    edit_lane: str = "model"
    take_selection: tuple[int, int] | None = None
    selected_snapshot: int = -1
    selection_anchor_time: float = 0.0
    selection_base: set[int] = field(default_factory=set)
    drag_id: int = -1
    drag_start_x: float = 0.0
    drag_offset_x: float = 0.0
    drag_preview_time: float = 0.0
    drag_moved: bool = False
    pointer_mode: str = ""
    scrub_resume: bool = False
    range_anchor: int = -1
    range_preview: tuple[int, int] | None = None

    def select_snapshot(self, snapshot_id: int) -> None:
        self.edit_lane = "snapshots"
        self.selected_snapshot = snapshot_id
        self.selected_id = -1
        self.selected_keyframes.clear()

    def select_key(
        self,
        key_id: int,
        time: float,
        *,
        additive: bool,
        editable: bool,
        mouse_x: float,
        marker_x: float,
    ) -> None:
        self.edit_lane = "model"
        self.selected_snapshot = -1
        if additive:
            self.selected_keyframes.symmetric_difference_update((key_id,))
        else:
            self.selected_keyframes = {key_id}
        self.selected_id = key_id if key_id in self.selected_keyframes else -1
        if editable and not additive:
            self.pointer_mode = "key"
            self.drag_id = key_id
            self.drag_start_x = mouse_x
            self.drag_offset_x = marker_x - mouse_x
            self.drag_preview_time = time
            self.drag_moved = False

    def begin_empty(
        self, lane: str, *, on_track: bool, playing: bool, time: float, additive: bool
    ) -> None:
        self.pointer_mode = "select" if on_track and lane in ("model", "take") else "scrub"
        self.scrub_resume = playing and self.pointer_mode == "scrub"
        self.selected_id = -1
        if self.pointer_mode == "select":
            self.selection_anchor_time = time
            self.selection_base = set(self.selected_keyframes) if additive else set()

    def select_to(
        self, time: float, markers: TimelineMarkerIndex, take_times: Sequence[float]
    ) -> None:
        start, end = sorted((self.selection_anchor_time, time))
        if self.edit_lane == "take" and take_times:
            self.take_selection = (
                nearest_take_frame(take_times, start),
                nearest_take_frame(take_times, end),
            )
        elif self.edit_lane == "model":
            self.selected_keyframes = self.selection_base | set(markers.ids_between(start, end))

    def preview_key(self, mouse_x: float, time: float, threshold: float) -> float | None:
        if self.drag_id < 0:
            return None
        self.drag_moved = self.drag_moved or abs(mouse_x - self.drag_start_x) > threshold
        if self.drag_moved:
            self.drag_preview_time = time
            return time
        return None

    def cancel_key(self) -> None:
        """Discard a held preview so releasing after Escape cannot commit it."""
        if self.drag_id >= 0:
            self.drag_id = -1
            self.drag_moved = False
            self.pointer_mode = "cancelled"

    def release(self, *, editable: bool) -> tuple[tuple[int, float] | None, bool]:
        """Return one retiming and/or replay decision, then consume the gesture."""
        retime = (
            (self.drag_id, self.drag_preview_time)
            if self.drag_id >= 0 and self.drag_moved and editable
            else None
        )
        self.drag_id = -1
        self.drag_moved = False
        resume = self.pointer_mode == "scrub" and self.scrub_resume
        if self.pointer_mode in ("key", "scrub", "select"):
            self.pointer_mode = ""
            self.scrub_resume = False
        return retime, resume

    def clear_model_selection(self) -> None:
        self.selected_id = -1
        self.selected_keyframes.clear()
        self.drag_id = -1
        self.drag_moved = False
