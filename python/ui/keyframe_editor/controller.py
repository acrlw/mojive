"""Timeline selection, transport and edit gestures, independent of UI rendering.

Session owns documents and playback. TimelineEditor owns only transient editor
state and sorted query caches; writes use the narrow TimelineCommands boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mojive import commands as cmd
from mojive.adapters.base import KeyframeInfo, KeyframeProperties
from mojive.interaction.input import InputClaim
from mojive.ui.pointer_bindings import PointerFrame
from mojive.ui.timeline import (
    keyframe_range,
    nearest_take_frame,
    neighboring_sorted_keyframe,
    timeline_x_to_time,
)

if TYPE_CHECKING:
    from mojive.session import Session

_MIN_TIMELINE_SPAN = 1e-6


class TimelineCommands(Protocol):
    session: Session
    take_video_active: bool
    style_scale: float
    input_claim: InputClaim | None
    queue_model_edit: (
        Callable[[cmd.Command, Callable[[cmd.CommandResult], None] | None], None] | None
    )

    def submit(self, command: cmd.Command) -> cmd.CommandResult: ...
    def submit_model_edit(
        self, command: cmd.Command, completed: Callable[[cmd.CommandResult], None] | None = None
    ) -> None: ...
    def tr(self, text: str) -> str: ...


@dataclass(frozen=True)
class TimelineHit:
    """One frame's hit-test geometry, never retained as document or gesture state."""

    position: tuple[float, float]
    time_bounds: tuple[float, float]
    lane: str
    on_track: bool
    keyframe_id: int
    snapshot_id: int
    marker_positions: dict[int, float]


class TimelineEditor:
    """One panel instance's transient selection, viewport and edit state."""

    def __init__(self) -> None:
        self.model_id = -1
        self.selected_id = -1
        self.selected_keyframes: set[int] = set()
        self.edit_lane = "model"
        self.take_selection: tuple[int, int] | None = None
        self.selection_anchor_time = 0.0
        self.selection_base: set[int] = set()
        self.seen_take_id = -2
        self.seen_take_length = 0
        self.selected_snapshot = -1
        self.selection_generation = -1
        self.properties: KeyframeProperties | None = None
        self.name = ""
        self.time = 0.0
        self.error = ""

        self.view_model_id = -1
        self.view_start = 0.0
        self.view_end = 1.0
        self.view_needs_fit = True
        self.playhead = 0.0
        self.seen_active_id = -2
        self.seen_take_cursor = -2
        self.seen_take_playhead: float | None = None
        self.seen_frame_time: float | None = None
        self.pan_moved = False
        self.drag_id = -1
        self.drag_start_x = 0.0
        self.drag_offset_x = 0.0
        self.drag_preview_time = 0.0
        self.drag_moved = False
        self.pointer_mode = ""
        self.pointer_chord = None
        self.scrub_resume = False
        self.range_anchor = -1
        self.range_preview: tuple[int, int] | None = None
        self.follow_mode = "page"
        self.locked_fraction = 0.15
        self.last_followed_playhead: float | None = None
        self.keyframe_cache_key: tuple[int, int] | None = None
        self.keyframe_cache: tuple[KeyframeInfo, ...] = ()
        self.keyframe_by_id: dict[int, KeyframeInfo] = {}

    def toggle_recording(self, ctx: TimelineCommands):
        """Share playhead recording and completed-range selection across both toolbars."""
        take_times = ctx.session.state_take_times
        self.sync_selection(ctx, self.keyframe_by_id, take_times)
        recording = ctx.session.state_take_recording
        result = ctx.submit(
            cmd.StopStateTakeRecording()
            if recording
            else cmd.StartStateTakeRecording(
                new_take=False,
                frame_index=nearest_take_frame(take_times, self.playhead) if take_times else None,
            )
        )
        self.error = "" if result.ok else result.message
        if result.ok:
            self.view_needs_fit = not recording
            self.edit_lane = "take"
            self.take_selection = None
            if recording:
                first = ctx.session.state_take_recording_start_frame
                self.sync_selection(ctx, self.keyframe_by_id, ctx.session.state_take_times)
                result = ctx.submit(cmd.SeekStateTake(first))
                if result.ok:
                    self.take_selection = (first, len(ctx.session.state_take_times) - 1)
                    self.playhead = ctx.session.state_take_times[first]
                self.error = "" if result.ok else result.message
        return result

    def set_follow_mode(self, mode: str) -> None:
        if mode == "locked":
            fraction = (self.playhead - self.view_start) / max(
                self.view_end - self.view_start, _MIN_TIMELINE_SPAN
            )
            self.locked_fraction = fraction if 0.05 <= fraction <= 0.95 else 0.15
        self.follow_mode = mode
        self.last_followed_playhead = None

    def seek_time(self, ctx: TimelineCommands, take_times: Sequence[float], time: float) -> None:
        if ctx.session.state_take_recording or ctx.take_video_active:
            return
        if take_times:
            index = nearest_take_frame(take_times, time)
            if index != ctx.session.state_take_cursor or ctx.session.state_take_playing:
                result = ctx.submit(cmd.SeekStateTake(index))
                self.error = "" if result.ok else result.message
                if not result.ok:
                    return
            self.playhead = take_times[index]
            self.seen_take_cursor = index
        else:
            self.playhead = time

    def begin_timeline_edit(
        self,
        ctx: TimelineCommands,
        keyframe_by_id: dict[int, KeyframeInfo],
        take_times: Sequence[float],
        hit: TimelineHit,
        *,
        editable: bool,
        load: bool,
        additive: bool,
    ) -> None:
        """Resolve a press into one editing gesture; Session remains the state owner."""
        mouse_xy = hit.position
        time_lo, time_hi = hit.time_bounds
        lane = hit.lane
        hit_id, snapshot_hit = hit.keyframe_id, hit.snapshot_id
        marker_positions = hit.marker_positions
        if snapshot_hit >= 0:
            self.edit_lane = "snapshots"
            self.selected_snapshot = snapshot_hit
            self.selected_id = -1
            self.selected_keyframes.clear()
            if (
                load
                and ctx.session.paused
                and not ctx.session.state_take_playing
                and not ctx.session.state_take_recording
            ):
                result = ctx.submit(cmd.RestoreSceneSnapshot(snapshot_hit))
                self.error = "" if result.ok else result.message
                if result.ok:
                    self.playhead = next(
                        item.time
                        for item in ctx.session.scene_snapshots
                        if item.snapshot_id == snapshot_hit
                    )
        elif hit_id >= 0:
            self.edit_lane = "model"
            self.selected_snapshot = -1
            if additive:
                if hit_id in self.selected_keyframes:
                    self.selected_keyframes.remove(hit_id)
                else:
                    self.selected_keyframes.add(hit_id)
            else:
                self.selected_keyframes = {hit_id}
            self.selected_id = hit_id if hit_id in self.selected_keyframes else -1
            self.selection_generation = -1
            marker = keyframe_by_id[hit_id]
            self.playhead = marker.time
            if editable and not additive:
                self.pointer_mode = "key"
                self.drag_id = hit_id
                self.drag_start_x = mouse_xy[0]
                self.drag_offset_x = marker_positions[hit_id] - mouse_xy[0]
                self.drag_preview_time = marker.time
                self.drag_moved = False
            if editable and load:
                self.load_keyframe(ctx, marker)
        else:
            if not ctx.session.state_take_recording:
                on_track = hit.on_track
                self.pointer_mode = "select" if on_track and lane in ("model", "take") else "scrub"
                self.scrub_resume = ctx.session.state_take_playing and self.pointer_mode == "scrub"
                self.selected_id = -1
                self.selection_generation = -1
                if self.pointer_mode == "select":
                    self.selection_anchor_time = timeline_x_to_time(
                        mouse_xy[0], self.view_start, self.view_end, time_lo, time_hi
                    )
                    self.selection_base = set(self.selected_keyframes) if additive else set()
                    if lane == "take":
                        self.seek_time(ctx, take_times, self.selection_anchor_time)

    def update_timeline_drag(
        self,
        ctx: TimelineCommands,
        keyframes: tuple[KeyframeInfo, ...],
        take_times: Sequence[float],
        hit: TimelineHit,
        pointer: PointerFrame,
        *,
        editable: bool,
        clear_range: bool,
        escape: bool,
    ) -> bool:
        """Update held gestures and finalize key edits/scrub restoration on release."""
        mouse_xy = hit.position
        time_lo, time_hi = hit.time_bounds
        scale = ctx.style_scale
        claim = ctx.input_claim or InputClaim()
        pointer_claimed = self.pointer_chord is not None and any(
            claim.claims_button(button) for button in self.pointer_chord.buttons
        )
        if pointer_claimed or (self.drag_id >= 0 and escape):
            original = next((key for key in keyframes if key.keyframe_id == self.drag_id), None)
            if original is not None:
                self.playhead = original.time
            self.drag_id = -1
            self.drag_moved = False
            self.range_preview = None
            self.scrub_resume = False
            self.pointer_mode = "cancelled"
            # Cancelling a preview must not become a key edit on release or
            # clear a pre-existing selection/range in the same Escape frame.
            escape = False
        if self.pointer_mode == "select" and pointer.held(self.pointer_chord):
            time = timeline_x_to_time(
                min(time_hi, max(time_lo, mouse_xy[0])),
                self.view_start,
                self.view_end,
                time_lo,
                time_hi,
            )
            start, end = sorted((self.selection_anchor_time, time))
            if self.edit_lane == "take" and take_times:
                self.take_selection = (
                    nearest_take_frame(take_times, start),
                    nearest_take_frame(take_times, end),
                )
            elif self.edit_lane == "model":
                self.selected_keyframes = self.selection_base | {
                    keyframes[index].keyframe_id for index in keyframe_range(keyframes, start, end)
                }

        if self.pointer_mode == "scrub" and pointer.held(self.pointer_chord):
            x = min(time_hi, max(time_lo, mouse_xy[0]))
            self.seek_time(
                ctx,
                take_times,
                timeline_x_to_time(x, self.view_start, self.view_end, time_lo, time_hi),
            )

        if self.pointer_mode == "range" and pointer.held(self.pointer_chord):
            x = min(time_hi, max(time_lo, mouse_xy[0]))
            time = timeline_x_to_time(x, self.view_start, self.view_end, time_lo, time_hi)
            index = nearest_take_frame(take_times, time)
            self.range_preview = (min(index, self.range_anchor), max(index, self.range_anchor))

        if clear_range or escape:
            if self.pointer_mode == "range":
                self.range_preview = None
                self.pointer_mode = "cancelled"
            if ctx.session.state_take_range is not None:
                ctx.submit(cmd.SetStateTakeRange())
            elif (
                escape
                and self.pointer_mode != "cancelled"
                and (self.take_selection is not None or self.selected_keyframes)
            ):
                self.take_selection = None
                self.selected_keyframes.clear()
                self.selected_id = -1
                if self.pointer_mode == "select":
                    self.pointer_mode = "cancelled"

        if self.drag_id >= 0 and pointer.held(self.pointer_chord):
            self.drag_moved = self.drag_moved or abs(mouse_xy[0] - self.drag_start_x) > 3.0 * scale
            if self.drag_moved:
                drag_x = min(time_hi, max(time_lo, mouse_xy[0] + self.drag_offset_x))
                self.drag_preview_time = timeline_x_to_time(
                    drag_x, self.view_start, self.view_end, time_lo, time_hi
                )
                self.playhead = self.drag_preview_time
        released = self.pointer_chord is not None and not pointer.held(self.pointer_chord)
        if self.drag_id >= 0 and released:
            if self.drag_moved and editable:
                self.retime_keyframe(ctx, self.drag_id, self.drag_preview_time)
            self.drag_id = -1
            self.drag_moved = False

        if released and self.pointer_mode in (
            "key",
            "scrub",
            "select",
        ):
            if self.pointer_mode == "scrub" and self.scrub_resume:
                result = ctx.submit(cmd.PlayStateTake())
                self.error = "" if result.ok else result.message
            self.pointer_mode = ""
            self.scrub_resume = False
        return released

    def finish_timeline_range(
        self, ctx: TimelineCommands, hit: TimelineHit, *, hovered: bool
    ) -> bool:
        """Commit a released loop range or expose the stationary pan context menu."""
        if self.pointer_mode not in ("range", "pan", "cancelled"):
            return False
        open_menu = self.pointer_mode == "pan" and not self.pan_moved and hovered
        if open_menu and hit.on_track:
            self.edit_lane = hit.lane
        if self.pointer_mode == "range" and self.range_preview is not None:
            first, last = self.range_preview
            command = cmd.SetStateTakeLoop(first, last) if first < last else cmd.SetStateTakeRange()
            result = ctx.submit(command)
            self.error = "" if result.ok else result.message
        self.pointer_mode = ""
        self.range_preview = None

        return open_menu

    def delete_selection(self, ctx, editable):
        if ctx.session.state_take_recording or ctx.take_video_active:
            return
        if self.edit_lane == "take" and self.take_selection is not None:
            result = ctx.submit(cmd.DeleteStateTakeFrames(*self.take_selection))
            self.error = "" if result.ok else result.message
            if result.ok:
                self.take_selection = None
        elif self.edit_lane == "model" and editable and self.selected_keyframes:
            commands = tuple(
                cmd.RemoveModelKeyframe(key)
                for key in sorted(self.selected_keyframes, reverse=True)
            )
            if ctx.queue_model_edit is not None:
                for command in commands:
                    ctx.submit_model_edit(command, self.snapshot_removed)
            else:
                self.snapshot_removed(
                    ctx.session.apply_edits(commands, label="Delete model keyframes").result
                )

    def selection_count(self):
        if self.edit_lane == "take" and self.take_selection is not None:
            return self.take_selection[1] - self.take_selection[0] + 1
        return len(self.selected_keyframes) if self.edit_lane == "model" else 0

    def status_detail(self, translate) -> str:
        count = self.selection_count()
        return f"{translate('Selected frames')}: {count}" if count else ""

    def sync_selection(
        self,
        ctx: TimelineCommands,
        keyframe_by_id: dict[int, KeyframeInfo],
        take_times: Sequence[float],
    ) -> None:
        take_id = ctx.session.active_state_take_id
        if take_id != self.seen_take_id:
            self.seen_take_id = take_id
            self.seen_take_cursor = -2
            self.take_selection = None
            self.range_preview = None
            self.pointer_mode = ""
            if take_id >= 0:
                self.edit_lane = "take"
                self.playhead = (
                    take_times[max(0, ctx.session.state_take_cursor)] if take_times else 0.0
                )
            self.view_needs_fit = True
        if len(take_times) != self.seen_take_length:
            self.take_selection = None
            self.seen_take_length = len(take_times)
        active = ctx.session.active_keyframe
        if active != self.seen_active_id:
            self.seen_active_id = active
            if active in keyframe_by_id:
                self.playhead = keyframe_by_id[active].time
        take_cursor = ctx.session.state_take_cursor
        if take_cursor != self.seen_take_cursor:
            self.seen_take_cursor = take_cursor
            if 0 <= take_cursor < len(take_times):
                self.playhead = take_times[take_cursor]
        take_playhead = ctx.session.state_take_playhead
        if take_playhead is not None and (
            ctx.session.state_take_playing or take_playhead != self.seen_take_playhead
        ):
            self.playhead = take_playhead
        self.seen_take_playhead = take_playhead
        frame_time = float(ctx.session.frame.time)
        if math.isfinite(frame_time):
            if not ctx.session.state_take_recording and (
                not ctx.session.paused
                or (
                    take_cursor < 0
                    and self.seen_frame_time is not None
                    and frame_time != self.seen_frame_time
                )
            ):
                self.playhead = frame_time
            self.seen_frame_time = frame_time

    def keyframes(
        self, ctx: TimelineCommands
    ) -> tuple[tuple[KeyframeInfo, ...], dict[int, KeyframeInfo]]:
        cache_key = (ctx.session.structure_generation, self.model_id)
        if cache_key != self.keyframe_cache_key:
            self.keyframe_cache = tuple(
                sorted(
                    (key for key in ctx.session.keyframes if key.model_id == self.model_id),
                    key=lambda key: (key.time, key.keyframe_id),
                )
            )
            self.keyframe_by_id = {key.keyframe_id: key for key in self.keyframe_cache}
            self.keyframe_cache_key = cache_key
            self.selected_keyframes.intersection_update(self.keyframe_by_id)
            if self.selected_id >= 0 and self.selected_id not in self.keyframe_by_id:
                self.clear_selection()
        return self.keyframe_cache, self.keyframe_by_id

    def load_neighbor(
        self, ctx: TimelineCommands, keyframes: tuple[KeyframeInfo, ...], direction: int
    ) -> None:
        marker = neighboring_sorted_keyframe(
            keyframes,
            self.keyframe_by_id.get(self.selected_id),
            self.playhead,
            direction,
        )
        if marker is not None:
            self.selected_id = marker.keyframe_id
            self.selection_generation = -1
            self.load_keyframe(ctx, marker)

    def load_keyframe(self, ctx: TimelineCommands, keyframe: KeyframeInfo) -> None:
        result = ctx.submit(cmd.LoadKeyframe(keyframe.keyframe_id))
        if result.ok:
            self.playhead = keyframe.time
            self.error = ""
        else:
            self.error = result.message

    def snapshot_created(self, result) -> None:
        if result.ok:
            self.selected_id = result.entity_id
            self.selected_keyframes = {result.entity_id}
            self.selection_generation = -1
            self.view_needs_fit = True
            self.error = ""
        else:
            self.error = result.message

    def snapshot_updated(self, result, time: float) -> None:
        if result.ok:
            self.selection_generation = -1
            self.playhead = float(time)
            self.error = ""
        else:
            self.error = result.message

    def snapshot_removed(self, result) -> None:
        if result.ok:
            self.clear_selection()
        else:
            self.error = result.message

    def retime_keyframe(self, ctx: TimelineCommands, keyframe_id: int, time: float) -> None:
        properties = ctx.session.keyframe_properties(keyframe_id)
        if properties is None:
            self.error = ctx.tr("Keyframe state is no longer available")
            return
        ctx.submit_model_edit(
            _set_keyframe_command(properties, properties.name, time),
            lambda result: self.snapshot_updated(result, time),
        )

    def set_model(self, model_id: int) -> None:
        if model_id == self.model_id:
            return
        self.model_id = model_id
        self.keyframe_cache_key = None
        self.keyframe_cache = ()
        self.keyframe_by_id = {}
        self.view_needs_fit = True
        self.view_model_id = -1
        self.seen_active_id = -2
        self.seen_take_cursor = -2
        self.pointer_mode = ""
        self.range_preview = None
        self.last_followed_playhead = None
        self.clear_selection()

    def clear_selection(self) -> None:
        self.selected_id = -1
        self.selected_keyframes.clear()
        self.selection_generation = -1
        self.properties = None
        self.name = ""
        self.time = 0.0
        self.error = ""
        self.drag_id = -1
        self.drag_moved = False


def _set_keyframe_command(
    properties: KeyframeProperties, name: str, time: float
) -> cmd.SetModelKeyframe:
    return cmd.SetModelKeyframe(
        properties.keyframe_id,
        properties.model_id,
        name,
        float(time),
        properties.qpos,
        properties.qvel,
        properties.act,
        properties.ctrl,
        properties.mocap_position,
        properties.mocap_quaternion,
    )
