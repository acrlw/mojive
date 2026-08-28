"""A compact Dope Sheet for model-local MuJoCo state keyframes."""

from __future__ import annotations

import math

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, KeyframeInfo, KeyframeProperties
from ..draw2d import ImguiDraw2D
from ..theme import with_alpha
from . import Panel, PanelContext, begin_kv_table

_MIN_TIMELINE_SPAN = 1e-6


def unique_keyframe_name(existing: set[str]) -> str:
    index = 1
    name = f"key{index}"
    while name in existing:
        index += 1
        name = f"key{index}"
    return name


def fitted_timeline_range(times: tuple[float, ...], fallback: float = 0.0) -> tuple[float, float]:
    """Return a padded, finite range with enough context around isolated keys."""

    finite = tuple(float(value) for value in times if math.isfinite(value))
    if not finite:
        center = float(fallback) if math.isfinite(fallback) else 0.0
        return center - 0.5, center + 0.5
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-9:
        context = max(1.0, abs(lo) * 0.1)
        return lo - context * 0.5, hi + context * 0.5
    padding = (hi - lo) * 0.08
    return lo - padding, hi + padding


def nice_timeline_step(span: float, pixel_width: float, target_pixels: float = 90.0) -> float:
    """Choose a stable 1/2/5 ruler step for the visible time range."""

    if not math.isfinite(span) or not math.isfinite(pixel_width) or pixel_width <= 0.0:
        return 1.0
    raw = max(float(span), _MIN_TIMELINE_SPAN) * target_pixels / pixel_width
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10.0**exponent)
    nice = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0 if fraction <= 5.0 else 10.0
    return nice * (10.0**exponent)


def zoom_timeline_range(
    start: float, end: float, anchor: float, wheel: float
) -> tuple[float, float]:
    """Zoom around ``anchor`` while keeping it at the same screen position."""

    span = max(float(end) - float(start), _MIN_TIMELINE_SPAN)
    ratio = min(1.0, max(0.0, (float(anchor) - float(start)) / span))
    new_span = min(1e12, max(_MIN_TIMELINE_SPAN, span * math.exp(-float(wheel) * 0.18)))
    new_start = float(anchor) - ratio * new_span
    return new_start, new_start + new_span


def timeline_time_to_x(time: float, start: float, end: float, lo: float, hi: float) -> float:
    span = max(float(end) - float(start), _MIN_TIMELINE_SPAN)
    return float(lo) + (float(time) - float(start)) * (float(hi) - float(lo)) / span


def timeline_x_to_time(x: float, start: float, end: float, lo: float, hi: float) -> float:
    width = max(float(hi) - float(lo), 1e-9)
    return float(start) + (float(x) - float(lo)) * (float(end) - float(start)) / width


def neighboring_keyframe(
    markers: tuple[tuple[int, float], ...],
    selected_id: int,
    playhead: float,
    direction: int,
) -> int:
    """Return the adjacent marker ID, using the playhead when none is selected."""

    if not markers or direction == 0:
        return -1
    ordered = sorted(markers, key=lambda marker: (marker[1], marker[0]))
    for slot, marker in enumerate(ordered):
        if marker[0] == selected_id:
            adjacent = slot + (1 if direction > 0 else -1)
            return ordered[min(len(ordered) - 1, max(0, adjacent))][0]
    if direction > 0:
        return next((key_id for key_id, time in ordered if time > playhead), ordered[-1][0])
    return next((key_id for key_id, time in reversed(ordered) if time < playhead), ordered[0][0])


class KeyframesPanel(Panel):
    """Edit whole-model state snapshots on a Blender-style time ruler."""

    name = "Keyframes"
    default_open = False
    shortcut = ""
    dock_with = "Output"

    def __init__(self) -> None:
        super().__init__()
        self._model_id = -1
        self._selected_id = -1
        self._selection_generation = -1
        self._properties: KeyframeProperties | None = None
        self._name = ""
        self._time = 0.0
        self._error = ""

        self._view_model_id = -1
        self._view_start = -0.5
        self._view_end = 0.5
        self._view_needs_fit = True
        self._playhead = 0.0
        self._seen_active_id = -2
        self._drag_id = -1
        self._drag_start_x = 0.0
        self._drag_offset_x = 0.0
        self._drag_preview_time = 0.0
        self._drag_moved = False

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        models = tuple(ctx.session.scene_models)
        if not models:
            imgui.text_disabled("no editable model keyframes")
            return

        model_ids = tuple(model.model_id for model in models)
        selected = ctx.session.selected_node
        if self._model_id not in model_ids:
            preferred = selected.model_id if selected is not None else model_ids[0]
            self._set_model(preferred if preferred in model_ids else model_ids[0])

        if begin_kv_table("keyframe_model"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("model")
            imgui.table_next_column()
            slot = model_ids.index(self._model_id)
            imgui.set_next_item_width(-1.0)
            changed, slot = imgui.combo(
                "##keyframe-model", slot, tuple(model.name for model in models)
            )
            if changed:
                self._set_model(model_ids[slot])
            imgui.end_table()

        keyframes = tuple(
            sorted(
                (key for key in ctx.session.keyframes if key.model_id == self._model_id),
                key=lambda key: (key.time, key.keyframe_id),
            )
        )
        editable = bool(ctx.session.paused)
        self._sync_selection(ctx, keyframes)
        self._draw_toolbar(ctx, keyframes, editable)
        self._draw_dope_sheet(ctx, keyframes, editable)

        if not keyframes:
            imgui.text_disabled("Capture the current state to create the first keyframe.")
        self._draw_selected(ctx, editable)

    def _draw_toolbar(
        self, ctx: PanelContext, keyframes: tuple[KeyframeInfo, ...], editable: bool
    ) -> None:
        if not editable:
            imgui.begin_disabled()
        if imgui.button("Capture"):
            name = unique_keyframe_name({key.name for key in keyframes})
            result = ctx.submit(cmd.AddModelKeyframe(self._model_id, name))
            if result.ok:
                self._selected_id = result.entity_id
                self._selection_generation = -1
                self._view_needs_fit = True
                self._error = ""
            else:
                self._error = result.message
        if not editable:
            imgui.end_disabled()
            imgui.set_item_tooltip("Pause the simulation before capturing a keyframe")

        imgui.same_line()
        if not keyframes or not editable:
            imgui.begin_disabled()
        if imgui.button("Previous"):
            self._load_neighbor(ctx, keyframes, -1)
        imgui.same_line()
        if imgui.button("Next"):
            self._load_neighbor(ctx, keyframes, 1)
        if not keyframes or not editable:
            imgui.end_disabled()

        imgui.same_line()
        if imgui.button("View All"):
            self._view_needs_fit = True
        imgui.same_line()
        imgui.text_disabled(f"{self._playhead:g} s  ·  {len(keyframes)} snapshot(s)")

    def _draw_dope_sheet(
        self, ctx: PanelContext, keyframes: tuple[KeyframeInfo, ...], editable: bool
    ) -> None:
        scale = ctx.style_scale
        available = max(260.0 * scale, float(imgui.get_content_region_avail().x))
        height = 116.0 * scale
        channel_width = min(150.0 * scale, max(92.0 * scale, available * 0.25))
        ruler_height = 27.0 * scale
        lo_vec = imgui.get_cursor_screen_pos()
        lo = (float(lo_vec.x), float(lo_vec.y))
        hi = (lo[0] + available, lo[1] + height)
        time_lo = lo[0] + channel_width
        time_hi = hi[0]
        time_width = max(1.0, time_hi - time_lo)

        flags = (
            imgui.ButtonFlags_.mouse_button_left.value
            | imgui.ButtonFlags_.mouse_button_middle.value
        )
        imgui.invisible_button("##keyframe-dope-sheet", imgui.ImVec2(available, height), flags)
        hovered = imgui.is_item_hovered()
        mouse = imgui.get_mouse_pos()
        mouse_xy = (float(mouse.x), float(mouse.y))
        over_timeline = hovered and mouse_xy[0] >= time_lo

        if self._view_needs_fit or self._view_model_id != self._model_id:
            self._view_start, self._view_end = fitted_timeline_range(
                tuple(key.time for key in keyframes), self._playhead
            )
            self._view_model_id = self._model_id
            self._view_needs_fit = False

        io = imgui.get_io()
        if over_timeline and io.mouse_wheel:
            anchor = timeline_x_to_time(
                mouse_xy[0], self._view_start, self._view_end, time_lo, time_hi
            )
            self._view_start, self._view_end = zoom_timeline_range(
                self._view_start, self._view_end, anchor, float(io.mouse_wheel)
            )
        if over_timeline and imgui.is_mouse_dragging(imgui.MouseButton_.middle):
            shift = -float(io.mouse_delta.x) * (self._view_end - self._view_start) / time_width
            self._view_start += shift
            self._view_end += shift

        marker_y = lo[1] + ruler_height + (height - ruler_height) * 0.5
        marker_radius = 7.0 * scale
        marker_positions = {
            key.keyframe_id: timeline_time_to_x(
                self._drag_preview_time
                if self._drag_id == key.keyframe_id and self._drag_moved
                else key.time,
                self._view_start,
                self._view_end,
                time_lo,
                time_hi,
            )
            for key in keyframes
        }
        hit_id = (
            self._hit_marker(marker_positions, marker_y, marker_radius, mouse_xy) if hovered else -1
        )

        if over_timeline and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            if hit_id >= 0:
                self._selected_id = hit_id
                self._selection_generation = -1
                marker = next(key for key in keyframes if key.keyframe_id == hit_id)
                self._playhead = marker.time
                if editable:
                    self._drag_id = hit_id
                    self._drag_start_x = mouse_xy[0]
                    self._drag_offset_x = marker_positions[hit_id] - mouse_xy[0]
                    self._drag_preview_time = marker.time
                    self._drag_moved = False
                if editable and imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
                    self._load_keyframe(ctx, marker)
            else:
                self._playhead = timeline_x_to_time(
                    mouse_xy[0], self._view_start, self._view_end, time_lo, time_hi
                )
                self._selected_id = -1
                self._selection_generation = -1

        if self._drag_id >= 0 and imgui.is_mouse_down(imgui.MouseButton_.left):
            self._drag_moved = (
                self._drag_moved or abs(mouse_xy[0] - self._drag_start_x) > 3.0 * scale
            )
            if self._drag_moved:
                drag_x = min(time_hi, max(time_lo, mouse_xy[0] + self._drag_offset_x))
                self._drag_preview_time = timeline_x_to_time(
                    drag_x, self._view_start, self._view_end, time_lo, time_hi
                )
                self._playhead = self._drag_preview_time
        if self._drag_id >= 0 and imgui.is_mouse_released(imgui.MouseButton_.left):
            if self._drag_moved and editable:
                self._retime_keyframe(ctx, self._drag_id, self._drag_preview_time)
            self._drag_id = -1
            self._drag_moved = False

        self._paint_dope_sheet(
            ctx,
            keyframes,
            lo,
            hi,
            time_lo,
            time_hi,
            ruler_height,
            marker_y,
            marker_radius,
            marker_positions,
            hit_id,
        )
        if hit_id >= 0 and hovered:
            key = next(key for key in keyframes if key.keyframe_id == hit_id)
            imgui.set_tooltip(f"{key.name or 'key'}  ·  {key.time:g} s\nDouble-click to load")
        elif over_timeline and hovered:
            imgui.set_item_tooltip("Left-click: move playhead · Wheel: zoom · Middle-drag: pan")

    def _paint_dope_sheet(
        self,
        ctx: PanelContext,
        keyframes: tuple[KeyframeInfo, ...],
        lo: tuple[float, float],
        hi: tuple[float, float],
        time_lo: float,
        time_hi: float,
        ruler_height: float,
        marker_y: float,
        marker_radius: float,
        marker_positions: dict[int, float],
        hit_id: int,
    ) -> None:
        overlay = ImguiDraw2D()
        theme = ctx.theme
        ruler_bottom = lo[1] + ruler_height
        overlay.rect_filled(lo, hi, theme.bg_child, rounding=3.0 * ctx.style_scale)
        overlay.rect_filled(lo, (time_lo, hi[1]), theme.bg_header)
        overlay.rect_filled((time_lo, lo[1]), (time_hi, ruler_bottom), theme.bg_frame)
        overlay.line((time_lo, lo[1]), (time_lo, hi[1]), theme.border, 1.0)
        overlay.line((lo[0], ruler_bottom), (hi[0], ruler_bottom), theme.border, 1.0)

        step = nice_timeline_step(self._view_end - self._view_start, time_hi - time_lo)
        first = math.ceil(self._view_start / step) * step
        tick = first
        iterations = 0
        while tick <= self._view_end + step * 1e-7 and iterations < 1000:
            x = timeline_time_to_x(tick, self._view_start, self._view_end, time_lo, time_hi)
            overlay.line((x, ruler_bottom), (x, hi[1]), with_alpha(theme.border, 0.65), 1.0)
            label = _format_tick(tick, step)
            label_width, _ = overlay.text_size(label)
            label_x = min(time_hi - label_width - 3.0, max(time_lo + 3.0, x + 4.0))
            overlay.text((label_x, lo[1] + 5.0), theme.text_disabled, label)
            tick += step
            iterations += 1

        overlay.text(
            (lo[0] + 10.0 * ctx.style_scale, marker_y - imgui.get_font_size() * 0.5),
            theme.text,
            "Model State",
        )
        overlay.text(
            (lo[0] + 10.0 * ctx.style_scale, hi[1] - imgui.get_font_size() - 5.0),
            theme.text_disabled,
            "snapshot channel",
        )

        playhead_x = timeline_time_to_x(
            self._playhead, self._view_start, self._view_end, time_lo, time_hi
        )
        if time_lo <= playhead_x <= time_hi:
            overlay.line((playhead_x, lo[1]), (playhead_x, hi[1]), theme.danger, 1.5)
            overlay.convex_fill(
                (
                    (playhead_x - 5.0, lo[1]),
                    (playhead_x + 5.0, lo[1]),
                    (playhead_x, lo[1] + 7.0),
                ),
                theme.danger,
            )

        for key in keyframes:
            x = marker_positions[key.keyframe_id]
            if x < time_lo - marker_radius or x > time_hi + marker_radius:
                continue
            selected = key.keyframe_id == self._selected_id
            hovered = key.keyframe_id == hit_id
            fill = (
                (0.98, 0.67, 0.24, 1.0)
                if hovered
                else theme.warning
                if selected
                else theme.text_disabled
            )
            points = (
                (x, marker_y - marker_radius),
                (x + marker_radius, marker_y),
                (x, marker_y + marker_radius),
                (x - marker_radius, marker_y),
            )
            overlay.convex_fill(points, fill)
            overlay.polyline(points, theme.bg_window, 1.0, closed=True)
            if key.keyframe_id == ctx.session.active_keyframe:
                overlay.circle_filled((x, marker_y), 2.25 * ctx.style_scale, theme.primary_bright)

        if self._selected_id in marker_positions:
            key = next(key for key in keyframes if key.keyframe_id == self._selected_id)
            x = marker_positions[key.keyframe_id]
            if time_lo <= x <= time_hi:
                value = (
                    self._drag_preview_time
                    if self._drag_id == key.keyframe_id and self._drag_moved
                    else key.time
                )
                label = f"{key.name or 'key'}  {value:g} s"
                text_width, _ = overlay.text_size(label)
                label_x = min(time_hi - text_width - 6.0, max(time_lo + 6.0, x + 10.0))
                overlay.text((label_x, marker_y + 13.0 * ctx.style_scale), theme.warning, label)

        overlay.rect(lo, hi, theme.border, 1.0, rounding=3.0 * ctx.style_scale)

    @staticmethod
    def _hit_marker(
        positions: dict[int, float],
        marker_y: float,
        radius: float,
        mouse: tuple[float, float],
    ) -> int:
        limit = radius + 4.0
        hits = (
            (abs(x - mouse[0]) + abs(marker_y - mouse[1]), key_id)
            for key_id, x in positions.items()
            if abs(x - mouse[0]) <= limit and abs(marker_y - mouse[1]) <= limit
        )
        return min(hits, default=(math.inf, -1))[1]

    def _sync_selection(self, ctx: PanelContext, keyframes: tuple[KeyframeInfo, ...]) -> None:
        ids = {key.keyframe_id for key in keyframes}
        if self._selected_id >= 0 and self._selected_id not in ids:
            self._clear_selection()
        active = ctx.session.active_keyframe
        if active != self._seen_active_id:
            self._seen_active_id = active
            if active in ids:
                marker = next(key for key in keyframes if key.keyframe_id == active)
                self._playhead = marker.time
        if not ctx.session.paused:
            self._playhead = float(ctx.session.frame.time)

    def _load_neighbor(
        self, ctx: PanelContext, keyframes: tuple[KeyframeInfo, ...], direction: int
    ) -> None:
        key_id = neighboring_keyframe(
            tuple((key.keyframe_id, key.time) for key in keyframes),
            self._selected_id,
            self._playhead,
            direction,
        )
        marker = next((key for key in keyframes if key.keyframe_id == key_id), None)
        if marker is not None:
            self._selected_id = marker.keyframe_id
            self._selection_generation = -1
            self._load_keyframe(ctx, marker)

    def _load_keyframe(self, ctx: PanelContext, keyframe: KeyframeInfo) -> None:
        result = ctx.submit(cmd.LoadKeyframe(keyframe.keyframe_id))
        if result.ok:
            self._playhead = keyframe.time
            self._error = ""
        else:
            self._error = result.message

    def _retime_keyframe(self, ctx: PanelContext, keyframe_id: int, time: float) -> None:
        properties = ctx.session.keyframe_properties(keyframe_id)
        if properties is None:
            self._error = "Keyframe state is no longer available"
            return
        result = ctx.submit(_set_keyframe_command(properties, properties.name, time))
        if result.ok:
            self._selection_generation = -1
            self._playhead = float(time)
            self._error = ""
        else:
            self._error = result.message

    def _draw_selected(self, ctx: PanelContext, editable: bool) -> None:
        if self._selected_id < 0:
            imgui.text_disabled(
                "MuJoCo keyframes are exact model-state snapshots; no interpolation is implied."
            )
            self._draw_error(ctx)
            return
        generation = ctx.session.structure_generation
        if self._selection_generation != generation:
            self._selection_generation = generation
            self._properties = ctx.session.keyframe_properties(self._selected_id)
            if self._properties is not None:
                self._name = self._properties.name
                self._time = self._properties.time
        properties = self._properties
        if properties is None:
            self._draw_error(ctx)
            return

        imgui.separator()
        imgui.text_disabled("selected snapshot")
        if begin_kv_table("keyframe_properties"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("name")
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._name = imgui.input_text("##keyframe-name", self._name)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("time")
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._time = imgui.input_double(
                "##keyframe-time", self._time, 0.0, 0.0, "%.9g"
            )
            imgui.end_table()

        dirty = self._name.strip() != properties.name or self._time != properties.time
        if not editable or not dirty or not self._name.strip():
            imgui.begin_disabled()
        if imgui.button("Apply"):
            result = ctx.submit(
                _set_keyframe_command(properties, self._name.strip(), float(self._time))
            )
            if result.ok:
                self._selection_generation = -1
                self._playhead = float(self._time)
                self._error = ""
            else:
                self._error = result.message
        if not editable or not dirty or not self._name.strip():
            imgui.end_disabled()
        imgui.same_line()
        if not editable:
            imgui.begin_disabled()
        if imgui.button("Load"):
            keyframe = KeyframeInfo(
                properties.keyframe_id, properties.name, properties.time, properties.model_id
            )
            self._load_keyframe(ctx, keyframe)
        imgui.same_line()
        if imgui.button("Delete"):
            result = ctx.submit(cmd.RemoveModelKeyframe(properties.keyframe_id))
            if result.ok:
                self._clear_selection()
            else:
                self._error = result.message
        if not editable:
            imgui.end_disabled()
            imgui.set_item_tooltip("Pause the simulation before editing keyframes")

        imgui.text_disabled("Exact state snapshot · double-click a marker to load · drag to retime")
        self._draw_error(ctx)

    def _draw_error(self, ctx: PanelContext) -> None:
        if self._error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.danger), self._error)
            if imgui.small_button("Copy error##keyframes"):
                imgui.set_clipboard_text(self._error)

    def _set_model(self, model_id: int) -> None:
        self._model_id = model_id
        self._view_needs_fit = True
        self._view_model_id = -1
        self._seen_active_id = -2
        self._clear_selection()

    def _clear_selection(self) -> None:
        self._selected_id = -1
        self._selection_generation = -1
        self._properties = None
        self._name = ""
        self._time = 0.0
        self._error = ""
        self._drag_id = -1
        self._drag_moved = False


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


def _format_tick(value: float, step: float) -> str:
    if abs(value) < step * 1e-9:
        value = 0.0
    decimals = max(0, min(9, -math.floor(math.log10(step)))) if step < 1.0 else 0
    return f"{value:.{decimals}f}"


__all__ = [
    "KeyframesPanel",
    "fitted_timeline_range",
    "neighboring_keyframe",
    "nice_timeline_step",
    "timeline_time_to_x",
    "timeline_x_to_time",
    "unique_keyframe_name",
    "zoom_timeline_range",
]
