"""A compact Dope Sheet for model-local MuJoCo state keyframes."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, KeyframeInfo, KeyframeProperties
from ...gizmo import _rounded_polygon_corners
from ..draw2d import ImguiDraw2D
from ..theme import with_alpha
from ..viewport_widgets import ToolHint
from . import Panel, PanelContext, begin_kv_table, button_row_layout, button_width

_MIN_TIMELINE_SPAN = 1e-6
_COMMAND_HEIGHT_PT = 28.0
_COMMAND_ICON_PT = 16.0
_MARKER_SPACING_FACTOR = 1.5


def timeline_status_hints(translate) -> tuple[ToolHint, ...]:
    """Return the compact interaction grammar for a hovered dope sheet."""

    return (
        ToolHint("mouse", "left", translate("Move playhead"), hint_id="keyframes.playhead"),
        ToolHint("mouse", "wheel", translate("Zoom"), hint_id="keyframes.zoom"),
        ToolHint("mouse", "middle", translate("Pan"), hint_id="keyframes.pan"),
    )


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


def nearest_take_frame(times: Sequence[float], time: float) -> int:
    """Return the nearest chronological take frame without scanning the full recording."""

    if not times:
        return -1
    slot = bisect.bisect_left(times, float(time))
    if slot <= 0:
        return 0
    if slot >= len(times):
        return len(times) - 1
    return slot - 1 if abs(times[slot - 1] - time) <= abs(times[slot] - time) else slot


def decimated_marker_ids(
    markers: Sequence[tuple[int, float]],
    lo: float,
    hi: float,
    min_spacing: float,
    priority_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Bound overlapping marker draws while retaining interactive priority markers."""

    spacing = max(float(min_spacing), 1.0)
    visible = tuple((keyframe_id, x) for keyframe_id, x in markers if lo <= x <= hi)
    capacity = max(1, int(max(0.0, hi - lo) // spacing) + 1)
    if len(visible) <= capacity:
        return tuple(keyframe_id for keyframe_id, _x in visible)
    priority = set(priority_ids)
    buckets: dict[int, int] = {}
    priority_visible: list[int] = []
    for keyframe_id, x in visible:
        if keyframe_id in priority:
            priority_visible.append(keyframe_id)
            continue
        buckets.setdefault(int((x - lo) // spacing), keyframe_id)
    return (*buckets.values(), *dict.fromkeys(priority_visible))


def _draw_command_icon(draw, center, kind: str, color, scale: float) -> None:
    """Draw one 16 pt transport or keyframe glyph."""

    x, y = (float(center[0]), float(center[1]))
    s = float(scale)

    def rounded_fill(points, *, radius: float = 0.75) -> None:
        path = _rounded_polygon_corners(points, radius * s, tuple(range(len(points))), segments=5)
        draw.fringed_concave_fill(tuple(map(tuple, path)), color)

    if kind == "record":
        draw.circle_filled((x, y), 4.5 * s, color, segments=20)
    elif kind == "play":
        rounded_fill(((x - 4.5 * s, y - 6.0 * s), (x + 6.0 * s, y), (x - 4.5 * s, y + 6.0 * s)))
    elif kind == "pause":
        draw.rect_filled(
            (x - 5.0 * s, y - 6.0 * s),
            (x - 1.5 * s, y + 6.0 * s),
            color,
            rounding=0.75 * s,
        )
        draw.rect_filled(
            (x + 1.5 * s, y - 6.0 * s),
            (x + 5.0 * s, y + 6.0 * s),
            color,
            rounding=0.75 * s,
        )
    elif kind == "stop":
        draw.rect_filled(
            (x - 5.5 * s, y - 5.5 * s),
            (x + 5.5 * s, y + 5.5 * s),
            color,
            rounding=1.0 * s,
        )
    elif kind in ("first", "previous"):
        rounded_fill(((x - 5.5 * s, y), (x + 3.5 * s, y - 6.0 * s), (x + 3.5 * s, y + 6.0 * s)))
        if kind == "first":
            draw.rect_filled(
                (x - 7.0 * s, y - 6.0 * s),
                (x - 5.2 * s, y + 6.0 * s),
                color,
                rounding=0.65 * s,
            )
    elif kind in ("next", "last"):
        rounded_fill(((x + 5.5 * s, y), (x - 3.5 * s, y - 6.0 * s), (x - 3.5 * s, y + 6.0 * s)))
        if kind == "last":
            draw.rect_filled(
                (x + 5.2 * s, y - 6.0 * s),
                (x + 7.0 * s, y + 6.0 * s),
                color,
                rounding=0.65 * s,
            )
    elif kind == "clear":
        draw.line((x - 5.0 * s, y - 5.0 * s), (x + 5.0 * s, y + 5.0 * s), color, 1.8 * s)
        draw.line((x + 5.0 * s, y - 5.0 * s), (x - 5.0 * s, y + 5.0 * s), color, 1.8 * s)
    elif kind in ("key-previous", "key-next"):
        direction = -1.0 if kind == "key-previous" else 1.0
        diamond_x = x - direction * 2.5 * s
        rounded_fill(
            (
                (diamond_x, y - 4.5 * s),
                (diamond_x + 4.5 * s, y),
                (diamond_x, y + 4.5 * s),
                (diamond_x - 4.5 * s, y),
            ),
            radius=0.55,
        )
        rounded_fill(
            (
                (x + direction * 7.0 * s, y),
                (x + direction * 3.5 * s, y - 3.5 * s),
                (x + direction * 3.5 * s, y + 3.5 * s),
            )
        )
    elif kind == "view":
        draw.rect(
            (x - 7.0 * s, y - 5.0 * s), (x + 7.0 * s, y + 5.0 * s), color, 1.5 * s, rounding=1.5 * s
        )
        draw.circle_filled((x, y), 2.0 * s, color, segments=16)
    else:
        draw.convex_fill(
            ((x, y - 6.0 * s), (x + 6.0 * s, y), (x, y + 6.0 * s), (x - 6.0 * s, y)),
            color,
        )


def _command_button(
    item_id: str,
    kind: str,
    tooltip: str,
    theme,
    scale: float,
    *,
    label: str = "",
    enabled: bool = True,
    selected: bool = False,
) -> bool:
    """Render a compact 28 pt semantic button with a vector glyph."""

    scale = float(scale)
    height = _COMMAND_HEIGHT_PT * scale
    width = _command_button_width(label, scale)
    if not enabled:
        imgui.begin_disabled()
    origin = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(item_id, imgui.ImVec2(width, height))
    hovered = enabled and imgui.is_item_hovered()
    active = enabled and imgui.is_item_active()
    if not enabled:
        imgui.end_disabled()

    background = (
        theme.bg_frame_active
        if selected or active
        else theme.bg_frame_hovered
        if hovered
        else theme.bg_frame
    )
    foreground = (
        theme.text_disabled
        if not enabled
        else theme.primary_bright
        if hovered or selected
        else theme.text
    )
    draw = ImguiDraw2D()
    lo = (float(origin.x), float(origin.y))
    hi = (lo[0] + width, lo[1] + height)
    draw.rect_filled(lo, hi, background, rounding=3.0 * scale)
    icon_center = (lo[0] + 14.0 * scale, lo[1] + height * 0.5)
    icon_color = theme.danger if kind == "record" and enabled else foreground
    _draw_command_icon(draw, icon_center, kind, icon_color, scale)
    if label:
        text_height = float(imgui.calc_text_size(label).y)
        draw.text(
            (lo[0] + 31.0 * scale, lo[1] + (height - text_height) * 0.5),
            foreground,
            label,
        )
    imgui.set_item_tooltip(tooltip)
    return bool(clicked and enabled)


def _command_button_width(label: str, scale: float) -> float:
    height = _COMMAND_HEIGHT_PT * float(scale)
    if not label:
        return height
    text_width = float(imgui.calc_text_size(label).x)
    return max(92.0 * scale, text_width + _COMMAND_ICON_PT * scale + 24.0 * scale)


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
        self._seen_take_cursor = -2
        self._drag_id = -1
        self._drag_start_x = 0.0
        self._drag_offset_x = 0.0
        self._drag_preview_time = 0.0
        self._drag_moved = False
        self._keyframe_cache_key: tuple[int, int] | None = None
        self._keyframe_cache: tuple[KeyframeInfo, ...] = ()
        self._keyframe_by_id: dict[int, KeyframeInfo] = {}

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        models = tuple(ctx.session.scene_models)
        if not models:
            imgui.text_disabled(ctx.tr("no editable model keyframes"))
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
            imgui.text_disabled(ctx.tr("model"))
            imgui.table_next_column()
            slot = model_ids.index(self._model_id)
            imgui.set_next_item_width(-1.0)
            changed, slot = imgui.combo(
                "##keyframe-model", slot, tuple(model.name for model in models)
            )
            if changed:
                self._set_model(model_ids[slot])
            imgui.end_table()

        keyframes, keyframe_by_id = self._keyframes(ctx)
        take_times = ctx.session.state_take_times
        editable = bool(ctx.session.paused)
        self._sync_selection(ctx, keyframe_by_id, take_times)
        self._draw_take_transport(ctx, take_times)
        self._draw_keyframe_toolbar(ctx, keyframes, editable)
        self._draw_dope_sheet(ctx, keyframes, keyframe_by_id, take_times, editable)

        self._draw_selected(ctx, editable)

    def _draw_take_transport(self, ctx: PanelContext, take_times: Sequence[float]) -> None:
        session = ctx.session
        supported = bool(session.adapter.caps.simulation and session.adapter.caps.state_snapshots)
        recording = session.state_take_recording
        playing = session.state_take_playing
        cursor = session.state_take_cursor

        scale = ctx.style_scale
        record_label = ctx.tr("Stop Recording" if recording else "Record New Take")
        status = self._take_status(ctx, take_times)
        widths = (
            _command_button_width(record_label, scale),
            *(_command_button_width("", scale) for _index in range(7)),
            float(imgui.calc_text_size(status).x),
        )
        inline = button_row_layout(
            widths,
            imgui.get_content_region_avail().x,
            3.0 * scale,
        )
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(3.0 * scale, 4.0 * scale),
        )
        if _command_button(
            "##take-record",
            "stop" if recording else "record",
            ctx.tr("Stop recording" if recording else "Record new take"),
            ctx.theme,
            scale,
            label=record_label,
            enabled=supported,
            selected=recording,
        ):
            result = ctx.submit(
                cmd.StopStateTakeRecording() if recording else cmd.StartStateTakeRecording()
            )
            self._error = "" if result.ok else result.message
            if result.ok:
                self._view_needs_fit = not recording
        transport_enabled = bool(supported and take_times and not recording)
        item_index = 1
        for kind, target, tooltip in (
            ("first", 0, ctx.tr("First frame")),
            ("previous", cursor - 1, ctx.tr("Previous frame")),
        ):
            if inline[item_index]:
                imgui.same_line()
            if _command_button(
                f"##take-{kind}",
                kind,
                tooltip,
                ctx.theme,
                scale,
                enabled=transport_enabled,
            ):
                result = ctx.submit(cmd.SeekStateTake(target))
                self._error = "" if result.ok else result.message
            item_index += 1
        if inline[item_index]:
            imgui.same_line()
        if _command_button(
            "##take-play-pause",
            "pause" if playing else "play",
            ctx.tr("Pause" if playing else "Replay"),
            ctx.theme,
            scale,
            enabled=transport_enabled,
            selected=playing,
        ):
            result = ctx.submit(cmd.PauseStateTake() if playing else cmd.PlayStateTake())
            self._error = "" if result.ok else result.message
        item_index += 1
        if inline[item_index]:
            imgui.same_line()
        if _command_button(
            "##take-stop",
            "stop",
            ctx.tr("Stop"),
            ctx.theme,
            scale,
            enabled=transport_enabled,
        ):
            result = ctx.submit(cmd.PauseStateTake())
            if result.ok and take_times:
                result = ctx.submit(cmd.SeekStateTake(0))
            self._error = "" if result.ok else result.message
        item_index += 1
        for kind, target, tooltip in (
            ("next", cursor + 1, ctx.tr("Next frame")),
            ("last", len(take_times) - 1, ctx.tr("Last frame")),
        ):
            if inline[item_index]:
                imgui.same_line()
            if _command_button(
                f"##take-{kind}",
                kind,
                tooltip,
                ctx.theme,
                scale,
                enabled=transport_enabled,
            ):
                result = ctx.submit(cmd.SeekStateTake(target))
                self._error = "" if result.ok else result.message
            item_index += 1
        if inline[item_index]:
            imgui.same_line()
        if _command_button(
            "##take-clear",
            "clear",
            ctx.tr("Clear take"),
            ctx.theme,
            scale,
            enabled=transport_enabled,
        ):
            result = ctx.submit(cmd.ClearStateTake())
            self._error = "" if result.ok else result.message
            if result.ok:
                self._view_needs_fit = True
        item_index += 1

        recording = session.state_take_recording
        playing = session.state_take_playing
        cursor = session.state_take_cursor
        if 0 <= cursor < len(take_times):
            self._playhead = take_times[cursor]
            self._seen_take_cursor = cursor
        if inline[item_index]:
            imgui.same_line()
        status = self._take_status(ctx, take_times)
        if recording:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.danger), status)
        elif playing:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.primary_bright),
                status,
            )
        else:
            imgui.text_disabled(status)
        imgui.pop_style_var()

    @staticmethod
    def _take_status(ctx: PanelContext, take_times: Sequence[float]) -> str:
        session = ctx.session
        cursor = session.state_take_cursor
        if session.state_take_recording:
            return f"{ctx.tr('REC')}  {len(take_times)} {ctx.tr('frames')}"
        if session.state_take_playing:
            return f"{ctx.tr('PLAY')}  {cursor + 1}/{len(take_times)}"
        if take_times:
            if cursor >= 0:
                return f"{ctx.tr('frame')} {cursor + 1}/{len(take_times)}"
            return f"{len(take_times)} {ctx.tr('recorded frames')}"
        supported = bool(session.adapter.caps.simulation and session.adapter.caps.state_snapshots)
        return ctx.tr("no recorded take" if supported else "state recording unavailable")

    def _draw_keyframe_toolbar(
        self, ctx: PanelContext, keyframes: tuple[KeyframeInfo, ...], editable: bool
    ) -> None:
        scale = ctx.style_scale
        capture_label = ctx.tr("Capture Snapshot")
        status = f"{self._playhead:g} s  ·  {len(keyframes)} {ctx.tr('snapshots')}"
        widths = (
            _command_button_width(capture_label, scale),
            *(_command_button_width("", scale) for _index in range(3)),
            float(imgui.calc_text_size(status).x),
        )
        inline = button_row_layout(
            widths,
            imgui.get_content_region_avail().x,
            3.0 * scale,
        )
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(3.0 * scale, 4.0 * scale),
        )
        if _command_button(
            "##capture-snapshot",
            "key",
            ctx.tr("Capture snapshot" if editable else "Pause to edit (Space)"),
            ctx.theme,
            scale,
            label=capture_label,
            enabled=editable,
        ):
            name = unique_keyframe_name({key.name for key in keyframes})
            result = ctx.submit(cmd.AddModelKeyframe(self._model_id, name))
            if result.ok:
                self._selected_id = result.entity_id
                self._selection_generation = -1
                self._view_needs_fit = True
                self._error = ""
            else:
                self._error = result.message
        navigation_enabled = bool(keyframes and editable)
        item_index = 1
        for kind, direction, tooltip in (
            ("key-previous", -1, ctx.tr("Previous key")),
            ("key-next", 1, ctx.tr("Next key")),
        ):
            if inline[item_index]:
                imgui.same_line()
            if _command_button(
                f"##{kind}",
                kind,
                tooltip if editable else ctx.tr("Pause to edit (Space)"),
                ctx.theme,
                scale,
                enabled=navigation_enabled,
            ):
                self._load_neighbor(ctx, keyframes, direction)
            item_index += 1
        if inline[item_index]:
            imgui.same_line()
        if _command_button(
            "##key-view-all",
            "view",
            ctx.tr("View all"),
            ctx.theme,
            scale,
        ):
            self._view_needs_fit = True
        item_index += 1
        if inline[item_index]:
            imgui.same_line()
        imgui.text_disabled(status)
        imgui.pop_style_var()

    def _draw_dope_sheet(
        self,
        ctx: PanelContext,
        keyframes: tuple[KeyframeInfo, ...],
        keyframe_by_id: dict[int, KeyframeInfo],
        take_times: Sequence[float],
        editable: bool,
    ) -> None:
        scale = ctx.style_scale
        available = max(260.0 * scale, float(imgui.get_content_region_avail().x))
        height = 154.0 * scale
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
        if over_timeline:
            # The dope sheet uses the wheel for zoom. Owning the wheel here
            # prevents the docked Keyframes window from scrolling as well.
            imgui.set_item_key_owner(imgui.Key.mouse_wheel_y)
            ctx.status_hints = timeline_status_hints(ctx.tr)

        if self._view_needs_fit or self._view_model_id != self._model_id:
            self._view_start, self._view_end = fitted_timeline_range(
                tuple(key.time for key in keyframes) + tuple(take_times), self._playhead
            )
            self._view_model_id = self._model_id
            self._view_needs_fit = False
        if ctx.session.state_take_recording and self._playhead > self._view_end:
            span = self._view_end - self._view_start
            shift = self._playhead - self._view_end + span * 0.08
            self._view_start += shift
            self._view_end += shift

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

        row_height = (height - ruler_height) * 0.5
        marker_y = lo[1] + ruler_height + row_height * 0.5
        take_y = marker_y + row_height
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
                marker = keyframe_by_id[hit_id]
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
                time = timeline_x_to_time(
                    mouse_xy[0], self._view_start, self._view_end, time_lo, time_hi
                )
                if (
                    take_times
                    and not ctx.session.state_take_recording
                    and abs(mouse_xy[1] - take_y) <= row_height * 0.5
                ):
                    index = nearest_take_frame(take_times, time)
                    result = ctx.submit(cmd.SeekStateTake(index))
                    if result.ok:
                        self._playhead = take_times[index]
                        self._error = ""
                    else:
                        self._error = result.message
                else:
                    self._playhead = time
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
            take_y,
            marker_radius,
            marker_positions,
            keyframe_by_id,
            take_times,
            hit_id,
        )
        if hit_id >= 0 and hovered:
            key = keyframe_by_id[hit_id]
            imgui.set_tooltip(
                f"{key.name or ctx.tr('keyframe')}  ·  {key.time:g} s\n{ctx.tr('Double-click to load')}"
            )

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
        take_y: float,
        marker_radius: float,
        marker_positions: dict[int, float],
        keyframe_by_id: dict[int, KeyframeInfo],
        take_times: Sequence[float],
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
        row_divider = (marker_y + take_y) * 0.5
        overlay.line((lo[0], row_divider), (hi[0], row_divider), theme.border, 1.0)

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
            ctx.tr("Model Keyframes"),
        )
        overlay.text(
            (lo[0] + 10.0 * ctx.style_scale, take_y - imgui.get_font_size() * 0.5),
            theme.text,
            ctx.tr("Recorded Take"),
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

        marker_ids = decimated_marker_ids(
            tuple(marker_positions.items()),
            time_lo - marker_radius,
            time_hi + marker_radius,
            marker_radius * _MARKER_SPACING_FACTOR,
            (self._selected_id, ctx.session.active_keyframe, hit_id),
        )
        for keyframe_id in marker_ids:
            key = keyframe_by_id[keyframe_id]
            x = marker_positions[keyframe_id]
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

        cursor = ctx.session.state_take_cursor
        drawn_columns: set[int] = set()
        first_take = bisect.bisect_left(take_times, self._view_start)
        last_take = bisect.bisect_right(take_times, self._view_end)
        visible_count = max(0, last_take - first_take)
        sample_stride = max(1, visible_count // max(1, round(time_hi - time_lo)))
        for index in range(first_take, last_take, sample_stride):
            if index == cursor:
                continue
            time = take_times[index]
            x = timeline_time_to_x(time, self._view_start, self._view_end, time_lo, time_hi)
            column = round(x)
            if column in drawn_columns or not time_lo <= x <= time_hi:
                continue
            drawn_columns.add(column)
            length = 8.0 * ctx.style_scale
            overlay.line(
                (x, take_y - length),
                (x, take_y + length),
                with_alpha(theme.primary, 0.72),
                2.0,
            )
        if 0 <= cursor < len(take_times):
            x = timeline_time_to_x(
                take_times[cursor], self._view_start, self._view_end, time_lo, time_hi
            )
            if time_lo <= x <= time_hi:
                length = 13.0 * ctx.style_scale
                overlay.line((x, take_y - length), (x, take_y + length), theme.danger, 2.5)

        if self._selected_id in marker_positions:
            key = keyframe_by_id[self._selected_id]
            x = marker_positions[key.keyframe_id]
            if time_lo <= x <= time_hi:
                value = (
                    self._drag_preview_time
                    if self._drag_id == key.keyframe_id and self._drag_moved
                    else key.time
                )
                label = f"{key.name or ctx.tr('keyframe')}  {value:g} s"
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

    def _sync_selection(
        self,
        ctx: PanelContext,
        keyframe_by_id: dict[int, KeyframeInfo],
        take_times: Sequence[float],
    ) -> None:
        if self._selected_id >= 0 and self._selected_id not in keyframe_by_id:
            self._clear_selection()
        active = ctx.session.active_keyframe
        if active != self._seen_active_id:
            self._seen_active_id = active
            if active in keyframe_by_id:
                self._playhead = keyframe_by_id[active].time
        take_cursor = ctx.session.state_take_cursor
        if take_cursor != self._seen_take_cursor:
            self._seen_take_cursor = take_cursor
            if 0 <= take_cursor < len(take_times):
                self._playhead = take_times[take_cursor]
        if not ctx.session.paused:
            self._playhead = float(ctx.session.frame.time)

    def _keyframes(
        self, ctx: PanelContext
    ) -> tuple[tuple[KeyframeInfo, ...], dict[int, KeyframeInfo]]:
        cache_key = (ctx.session.structure_generation, self._model_id)
        if cache_key != self._keyframe_cache_key:
            self._keyframe_cache = tuple(
                sorted(
                    (key for key in ctx.session.keyframes if key.model_id == self._model_id),
                    key=lambda key: (key.time, key.keyframe_id),
                )
            )
            self._keyframe_by_id = {key.keyframe_id: key for key in self._keyframe_cache}
            self._keyframe_cache_key = cache_key
        return self._keyframe_cache, self._keyframe_by_id

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
            self._error = ctx.tr("Keyframe state is no longer available")
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
        imgui.text_disabled(ctx.tr("selected snapshot"))
        if begin_kv_table("keyframe_properties"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("name"))
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._name = imgui.input_text("##keyframe-name", self._name)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("time"))
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._time = imgui.input_double(
                "##keyframe-time", self._time, 0.0, 0.0, "%.9g"
            )
            imgui.end_table()

        dirty = self._name.strip() != properties.name or self._time != properties.time
        if not editable or not dirty or not self._name.strip():
            imgui.begin_disabled()
        action_labels = (ctx.tr("Apply"), ctx.tr("Load"), ctx.tr("Delete"))
        inline = button_row_layout(
            tuple(button_width(label) for label in action_labels),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        if imgui.button(action_labels[0]):
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
        if inline[1]:
            imgui.same_line()
        if not editable:
            imgui.begin_disabled()
        if imgui.button(action_labels[1]):
            keyframe = KeyframeInfo(
                properties.keyframe_id, properties.name, properties.time, properties.model_id
            )
            self._load_keyframe(ctx, keyframe)
        if inline[2]:
            imgui.same_line()
        if imgui.button(action_labels[2]):
            result = ctx.submit(cmd.RemoveModelKeyframe(properties.keyframe_id))
            if result.ok:
                self._clear_selection()
            else:
                self._error = result.message
        if not editable:
            imgui.end_disabled()
            imgui.set_item_tooltip(ctx.tr("Pause the simulation before editing keyframes"))
        self._draw_error(ctx)

    def _draw_error(self, ctx: PanelContext) -> None:
        if self._error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.danger), self._error)
            if imgui.small_button(f"{ctx.tr('Copy error')}##keyframes"):
                imgui.set_clipboard_text(self._error)

    def _set_model(self, model_id: int) -> None:
        self._model_id = model_id
        self._keyframe_cache_key = None
        self._keyframe_cache = ()
        self._keyframe_by_id = {}
        self._view_needs_fit = True
        self._view_model_id = -1
        self._seen_active_id = -2
        self._seen_take_cursor = -2
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
    "nearest_take_frame",
    "neighboring_keyframe",
    "nice_timeline_step",
    "timeline_time_to_x",
    "timeline_x_to_time",
    "unique_keyframe_name",
    "zoom_timeline_range",
]
