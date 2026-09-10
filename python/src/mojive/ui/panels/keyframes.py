"""A compact Dope Sheet for model-local MuJoCo state keyframes."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, KeyframeInfo, KeyframeProperties
from ...curves2d import CORNER_SMOOTHING, capped_polyline_points
from ...gizmo import _rounded_polygon_corners
from ...input import InputClaim
from ..draw2d import ImguiDraw2D, fit_text, text_line_y
from ..input_bindings import DEFAULT_INPUT_BINDINGS
from ..pointer_bindings import PointerAction
from ..theme import with_alpha
from ..viewport_widgets import (
    PLAYBACK_HALF_HEIGHT_PT,
    PLAYBACK_RESET_SCALE,
    ToolHint,
    draw_expand_glyph,
    draw_playback_glyph,
    draw_reset_glyph,
    pointer_tool_hint,
)
from . import Panel, PanelContext, button_row_layout, button_width, segmented_control

_MIN_TIMELINE_SPAN = 1e-6
_COMMAND_HEIGHT_PT = 28.0
_COMMAND_ICON_PT = 16.0
_MARKER_SPACING_FACTOR = 1.5
_LOOP_COLOR = (0.98, 0.52, 0.18, 1.0)
_COMMAND_PLAYBACK_KINDS = {
    "previous": "previous",
    "next": "step",
    "play": "play",
    "pause": "pause",
    "stop": "stop",
}


@lru_cache(maxsize=256)
def _rounded_command_icon_path(
    points: tuple[tuple[float, float], ...],
    radius: float,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        map(
            tuple,
            _rounded_polygon_corners(
                points, radius, tuple(range(len(points))), smoothing=smoothing
            ),
        )
    )


@lru_cache(maxsize=64)
def _fit_corner_path(scale: float, sx: int, sy: int, smoothing: float):
    return capped_polyline_points(
        (
            (sx * 2.75 * scale, sy * 6 * scale),
            (sx * 6 * scale, sy * 6 * scale),
            (sx * 6 * scale, sy * 2.75 * scale),
        ),
        1.5 * scale,
        round_start=True,
        round_end=True,
        smoothing=smoothing,
    )


@lru_cache(maxsize=128)
def _command_stroke(points, width, scale, smoothing):
    return capped_polyline_points(
        tuple((x * scale, y * scale) for x, y in points),
        width * scale,
        round_start=True,
        round_end=True,
        smoothing=smoothing,
    )


def timeline_status_hints(
    translate, *, has_range: bool = False, bindings=DEFAULT_INPUT_BINDINGS
) -> tuple[ToolHint, ...]:
    """Prioritize the range gesture when a narrow status bar can fit few hints."""

    hints = tuple(
        hint
        for hint in (
            pointer_tool_hint(
                PointerAction.TIMELINE_RANGE,
                bindings,
                translate("Select loop range"),
                hint_id="keyframes.range",
            ),
            pointer_tool_hint(
                PointerAction.TIMELINE_SCRUB,
                bindings,
                translate("Move playhead"),
                hint_id="keyframes.playhead",
            ),
            pointer_tool_hint(
                PointerAction.TIMELINE_ZOOM, bindings, translate("Zoom"), hint_id="keyframes.zoom"
            ),
            pointer_tool_hint(
                PointerAction.TIMELINE_PAN, bindings, translate("Pan"), hint_id="keyframes.pan"
            ),
        )
        if hint is not None
    )
    if has_range:
        hints += (
            ToolHint("key", "Esc", translate("Clear range"), hint_id="keyframes.clear_range"),
        )
    return hints


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
        return min(0.0, center), max(1.0, center + 0.5)
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-9:
        context = max(1.0, abs(lo) * 0.1)
        return max(0.0, lo - context * 0.5) if lo >= 0 else lo - context * 0.5, hi + context * 0.5
    padding = (hi - lo) * 0.08
    return max(0.0, lo - padding) if lo >= 0 else lo - padding, hi + padding


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


def timeline_channel_width(available: float, scale: float) -> float:
    return min(150 * scale, available * 0.35)


def timeline_time_to_x(time: float, start: float, end: float, lo: float, hi: float) -> float:
    span = max(float(end) - float(start), _MIN_TIMELINE_SPAN)
    return float(lo) + (float(time) - float(start)) * (float(hi) - float(lo)) / span


def timeline_x_to_time(x: float, start: float, end: float, lo: float, hi: float) -> float:
    width = max(float(hi) - float(lo), 1e-9)
    return float(start) + (float(x) - float(lo)) * (float(end) - float(start)) / width


def follow_timeline_range(
    start: float, end: float, playhead: float, mode: str, locked_fraction: float = 0.15
) -> tuple[float, float]:
    """Page an escaped playhead forward, or hold its relative screen position."""
    span = max(end - start, _MIN_TIMELINE_SPAN)
    if mode == "locked":
        start = playhead - min(0.95, max(0.05, locked_fraction)) * span
    elif mode == "page" and not start + span * 0.01 <= playhead <= end - span * 0.01:
        start = playhead - span * 0.08
    else:
        return start, end
    return start, start + span


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


def recorded_take_spans(times: Sequence[float], start: float, end: float, lo: float, hi: float):
    """Merge overlapping sample marks with work bounded by visible pixel columns."""
    first = bisect.bisect_left(times, start)
    last = bisect.bisect_right(times, end)
    if first == last:
        return
    stride = max(1, (last - first) // max(1, round(hi - lo)))
    factor = (hi - lo) / max(end - start, _MIN_TIMELINE_SPAN)
    left = right = None
    for index in range(first, last + stride - 1, stride):
        x = lo + (times[min(index, last - 1)] - start) * factor
        mark_left, mark_right = max(lo, x - 1.0), min(hi, x + 1.0)
        if right is not None and mark_left > right:
            yield left, right
            left = None
        if left is None:
            left = mark_left
        right = mark_right
    yield left, right


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


def _draw_command_icon(
    draw, center, kind: str, color, scale: float, *, smoothing: float = CORNER_SMOOTHING
) -> None:
    """Draw one 16 pt transport or keyframe glyph."""

    x, y = (float(center[0]), float(center[1]))
    s = float(scale)

    def rounded_fill(points, *, radius: float = 0.75) -> None:
        draw.fringed_concave_fill(
            _rounded_command_icon_path(
                tuple((px * s, py * s) for px, py in points), radius * s, smoothing
            ),
            color,
            origin=(x, y),
        )

    def stroke(points, width):
        draw.fringed_concave_fill(
            _command_stroke(points, width, s, smoothing), color, origin=(x, y)
        )

    if kind in ("first", "last"):
        direction = -1.0 if kind == "first" else 1.0
        for offset in (-2.0, 2.5):
            rounded_fill(
                tuple(
                    (direction * (px + offset), py)
                    for px, py in ((-2.5, -4.5), (2.0, 0.0), (-2.5, 4.5))
                ),
                radius=0.45,
            )
        stroke(((direction * 6, -4.5), (direction * 6, 4.5)), 1.4)
    elif kind in _COMMAND_PLAYBACK_KINDS:
        draw_playback_glyph(
            draw, center, color, s * 0.85, _COMMAND_PLAYBACK_KINDS[kind], smoothing=smoothing
        )
    elif kind in ("loop", "reset"):
        draw_reset_glyph(draw, center, color, s * 0.72)
    elif kind == "options":
        draw_expand_glyph(draw, center, color, s)
    elif kind == "record":
        draw.circle_filled((x, y), 4.5 * s, color)
    elif kind == "add":
        stroke(((-5, 0), (5, 0)), 1.6)
        stroke(((0, -5), (0, 5)), 1.6)
    elif kind == "clear":
        stroke(((-5, -5), (5, 5)), 1.8)
        stroke(((5, -5), (-5, 5)), 1.8)
    elif kind in ("key-previous", "key-next"):
        direction = -1.0 if kind == "key-previous" else 1.0
        diamond_x = -direction * 2.5
        rounded_fill(
            (
                (diamond_x, -4.5),
                (diamond_x + 4.5, 0.0),
                (diamond_x, 4.5),
                (diamond_x - 4.5, 0.0),
            ),
            radius=0.55,
        )
        rounded_fill(
            (
                (direction * 7.0, 0.0),
                (direction * 3.5, -3.5),
                (direction * 3.5, 3.5),
            )
        )
    elif kind == "fit":
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            draw.fringed_concave_fill(_fit_corner_path(s, sx, sy, smoothing), color, origin=(x, y))
    elif kind == "follow":
        stroke(((-6, 0), (2, 0)), 1.5)
        rounded_fill(((0, -3), (4, 0), (0, 3)), radius=0.4)
        stroke(((6, -5), (6, 5)), 1.5)
    elif kind == "view":
        draw.rect(
            (x - 7.0 * s, y - 5.0 * s),
            (x + 7.0 * s, y + 5.0 * s),
            color,
            1.5 * s,
            rounding=1.5 * s,
            smoothing=smoothing,
        )
        draw.circle_filled((x, y), 2.0 * s, color)
    else:
        rounded_fill(
            ((0.0, -6.0), (6.0, 0.0), (0.0, 6.0), (-6.0, 0.0)),
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
    smoothing: float = CORNER_SMOOTHING,
    width: float | None = None,
    layouts: dict | None = None,
) -> bool:
    """Render a compact 28 pt semantic button with a vector glyph."""

    scale = float(scale)
    height = _COMMAND_HEIGHT_PT * scale
    width = _command_button_width(label, scale) if width is None else width
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
    draw = ImguiDraw2D(corner_smoothing=smoothing)
    lo = (float(origin.x), float(origin.y))
    hi = (lo[0] + width, lo[1] + height)
    draw.rect_filled(lo, hi, background, rounding=float(imgui.get_style().frame_rounding))
    center_y = lo[1] + height * 0.5
    icon_center = (lo[0] + width * 0.5, center_y)
    if label:
        key = (imgui.get_font(), imgui.get_font_size(), kind, label, width, scale)
        cached = layouts.get(item_id) if layouts is not None else None
        if cached is None or cached[0] != key:
            icon_width = {
                "record": 9.0,
                "stop": 2 * PLAYBACK_HALF_HEIGHT_PT * PLAYBACK_RESET_SCALE * 0.85,
                "view": 15.5,
                "key": 12.0,
            }.get(kind, _COMMAND_ICON_PT) * scale
            gap = 7.0 * scale
            label = fit_text(draw, label, max(1.0, width - 16 * scale - icon_width - gap))
            ink = draw.text_ink_bounds(label) or (0.0, 0.0, *draw.text_size(label))
            left = (width - icon_width - gap - (ink[2] - ink[0])) * 0.5
            cached = (
                key,
                label,
                left + icon_width * 0.5,
                left + icon_width + gap - ink[0],
                -(ink[1] + ink[3]) * 0.5,
            )
            if layouts is not None:
                layouts[item_id] = cached
        _, label, icon_x, text_x, text_y = cached
        icon_center = (lo[0] + icon_x, center_y)
        draw.text(
            (lo[0] + text_x, center_y + text_y),
            foreground,
            label,
            pixel_snap=False,
        )
    icon_color = theme.danger if kind == "record" and enabled else foreground
    _draw_command_icon(draw, icon_center, kind, icon_color, scale, smoothing=smoothing)
    imgui.set_item_tooltip(tooltip)
    return bool(clicked and enabled)


def _toolbar_status(text: str, color, scale: float, *, width: float | None = None) -> None:
    draw = ImguiDraw2D()
    origin = imgui.get_cursor_screen_pos()
    height = _COMMAND_HEIGHT_PT * scale
    imgui.dummy((draw.text_size(text)[0] if width is None else width, height))
    draw.text((origin.x, text_line_y(draw, origin.y + height * 0.5)), color, text)


def _command_button_width(label: str, scale: float) -> float:
    height = _COMMAND_HEIGHT_PT * float(scale)
    if not label:
        return height
    text_width = float(imgui.calc_text_size(label).x)
    return max(92.0 * scale, text_width + _COMMAND_ICON_PT * scale + 24.0 * scale)


class KeyframesPanel(Panel):
    """Edit whole-model state snapshots on a Blender-style time ruler."""

    id = "keyframes"
    name = "Keyframes"
    default_open = True
    shortcut = ""
    dock_with = "Output"

    def __init__(self) -> None:
        super().__init__()
        self._model_id = -1
        self._command_layouts: dict = {}
        self._selected_id = -1
        self._selected_snapshot = -1
        self._selection_generation = -1
        self._properties: KeyframeProperties | None = None
        self._name = ""
        self._time = 0.0
        self._error = ""

        self._view_model_id = -1
        self._view_start = 0.0
        self._view_end = 1.0
        self._view_needs_fit = True
        self._playhead = 0.0
        self._seen_active_id = -2
        self._seen_take_cursor = -2
        self._drag_id = -1
        self._drag_start_x = 0.0
        self._drag_offset_x = 0.0
        self._drag_preview_time = 0.0
        self._drag_moved = False
        self._pointer_mode = ""
        self._pointer_chord = None
        self._scrub_resume = False
        self._range_anchor = -1
        self._range_preview: tuple[int, int] | None = None
        self._follow_mode = "page"
        self._locked_fraction = 0.15
        self._last_followed_playhead: float | None = None
        self._keyframe_cache_key: tuple[int, int] | None = None
        self._keyframe_cache: tuple[KeyframeInfo, ...] = ()
        self._keyframe_by_id: dict[int, KeyframeInfo] = {}

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        models = tuple(ctx.session.scene_models)
        model_ids = tuple(model.model_id for model in models)
        selected = ctx.session.selected_node
        if self._model_id not in model_ids:
            preferred = selected.model_id if selected is not None else -1
            self._set_model(
                preferred if preferred in model_ids else model_ids[0] if model_ids else -1
            )
        keyframes, keyframe_by_id = self._keyframes(ctx)
        take_times = ctx.session.state_take_times
        editable = bool(
            ctx.session.paused
            and not ctx.take_video_active
            and ctx.session.adapter.caps.supports("model.keyframe_edit")
        )
        self._sync_selection(ctx, keyframe_by_id, take_times)
        self._draw_compact_toolbar(ctx, take_times)
        self._draw_dope_sheet(ctx, models, keyframes, keyframe_by_id, take_times, editable)
        if self._selected_snapshot >= 0:
            snapshot = next(
                (
                    item
                    for item in ctx.session.scene_snapshots
                    if item.snapshot_id == self._selected_snapshot
                ),
                None,
            )
            if snapshot is None:
                self._selected_snapshot = -1
            else:
                imgui.text_disabled(f"{snapshot.name}  {snapshot.time:.3f} {ctx.tr('s')}")
                if imgui.button(ctx.tr("Remove snapshot") + "##remove-scene-snapshot"):
                    ctx.submit(cmd.RemoveSceneSnapshot(snapshot.snapshot_id))
                    self._selected_snapshot = -1
        self._draw_selected(ctx, editable)
        self._draw_error(ctx)

    def _draw_model_header(self, ctx, models, keyframes, editable, width):
        scale = ctx.style_scale
        gap = 5 * scale
        icon_width = _command_button_width("", scale)
        imgui.set_next_item_width(max(1, width - icon_width - gap))
        model_ids = tuple(model.model_id for model in models)
        slot = model_ids.index(self._model_id) if self._model_id in model_ids else 0
        imgui.begin_disabled(not models)
        changed, slot = imgui.combo(
            "##keyframe-model", slot, tuple(model.name for model in models) or (ctx.tr("Scene"),)
        )
        imgui.end_disabled()
        imgui.set_item_tooltip(models[slot].name if models else ctx.tr("Scene"))
        if changed:
            self._set_model(model_ids[slot])
        imgui.same_line()
        if _command_button(
            "##add-model-keyframe",
            "add",
            ctx.tr("Add Model Keyframe"),
            ctx.theme,
            scale,
            enabled=editable and self._model_id >= 0,
        ):
            name = unique_keyframe_name({key.name for key in keyframes})
            ctx.submit_model_edit(
                cmd.AddModelKeyframe(self._model_id, name), self._snapshot_created
            )

    def _draw_compact_toolbar(self, ctx, take_times):
        scale = ctx.style_scale
        gap = 5 * scale
        width = max(1.0, imgui.get_content_region_avail().x)
        height = _COMMAND_HEIGHT_PT * scale
        labels = (
            ctx.tr("Stop Recording" if ctx.session.state_take_recording else "Record Take"),
            ctx.tr("Stop Video" if ctx.take_video_active else "Export Video"),
            ctx.tr("Capture Snapshot"),
        )
        icon_width = _command_button_width("", scale)
        first, last = (
            (take_times[0], take_times[-1]) if take_times else (self._view_start, self._view_end)
        )
        time_reference = f"{max(99, abs(first), abs(last)):.3f} s"
        time_width = imgui.calc_text_size(("-" if first < 0 else "") + time_reference).x
        transport_width = 5 * icon_width + time_width + 5 * gap
        field_width = max(
            56 * scale, imgui.calc_text_size(f"{max(abs(first), abs(last)):.2f}").x + 16 * scale
        )
        take_width = max(
            84 * scale,
            imgui.calc_text_size(ctx.tr("Take 1" if take_times else "No take")).x + 44 * scale,
        )
        range_widths = (take_width, field_width, field_width, icon_width)
        range_width = sum(range_widths) + 3 * gap
        follow_labels = (ctx.tr("Off"), ctx.tr("Page"), ctx.tr("Locked"))
        follow_width = 3 * (
            max(imgui.calc_text_size(label).x for label in follow_labels) + 16 * scale
        )
        follow_label_width = imgui.calc_text_size(ctx.tr("Follow")).x
        view_width = follow_label_width + gap + follow_width + 2 * (icon_width + gap)
        separator_width = 13 * scale
        fixed_width = transport_width + range_width + view_width + 3 * separator_width
        command_minimum = 3 * icon_width + 2 * gap
        if fixed_width + command_minimum <= width:
            caption_budget = width - fixed_width
        elif command_minimum + transport_width + separator_width <= width:
            caption_budget = width - transport_width - separator_width
        else:
            caption_budget = width
        shown = list(labels)
        for index in (1, 2, 0):
            total = sum(_command_button_width(label, scale) for label in shown) + 2 * gap
            if total <= caption_budget:
                break
            shown[index] = ""
        widths = [_command_button_width(label, scale) for label in shown]
        right = imgui.get_cursor_screen_pos().x + width
        first_group = True

        def group(group_width):
            nonlocal first_group
            if (
                not first_group
                and imgui.get_item_rect_max().x + separator_width + group_width <= right
            ):
                imgui.same_line(0, 6 * scale)
                pos = imgui.get_cursor_screen_pos()
                imgui.dummy((scale, height))
                ImguiDraw2D().line(
                    (pos.x, pos.y + 5 * scale),
                    (pos.x, pos.y + height - 5 * scale),
                    (*ctx.theme.text_disabled[:3], 0.25),
                    scale,
                )
                imgui.same_line(0, 6 * scale)
            first_group = False
            imgui.begin_group()

        imgui.push_style_var(imgui.StyleVar_.item_spacing, (gap, 6 * scale))
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (8 * scale, max(0, (height - imgui.get_font_size()) * 0.5)),
        )
        group(sum(widths) + 2 * gap)
        recording = ctx.session.state_take_recording
        supported = (
            ctx.session.adapter.caps.simulation
            and ctx.session.adapter.caps.state_snapshots
            and ctx.session.adapter.caps.clock_control
        )
        if _command_button(
            "##take-record",
            "stop" if recording else "record",
            labels[0],
            ctx.theme,
            scale,
            label=shown[0],
            width=widths[0],
            layouts=self._command_layouts,
            enabled=supported and not ctx.take_video_active,
            selected=recording,
        ):
            result = ctx.submit(
                cmd.StopStateTakeRecording() if recording else cmd.StartStateTakeRecording()
            )
            self._error = "" if result.ok else result.message
            if result.ok:
                self._view_needs_fit = not recording
        imgui.same_line()
        if _command_button(
            "##take-video",
            "view",
            labels[1],
            ctx.theme,
            scale,
            label=shown[1],
            width=widths[1],
            layouts=self._command_layouts,
            enabled=ctx.start_take_video is not None
            and (ctx.take_video_active or (bool(take_times) and not recording)),
        ):
            try:
                ctx.stop_recording() if ctx.take_video_active else ctx.start_take_video()
                self._error = ""
            except (RuntimeError, ValueError) as exc:
                self._error = str(exc)
        imgui.same_line()
        if _command_button(
            "##capture-snapshot",
            "key",
            ctx.tr("Capture complete scene state"),
            ctx.theme,
            scale,
            label=shown[2],
            width=widths[2],
            layouts=self._command_layouts,
            enabled=ctx.session.adapter.caps.state_snapshots and not ctx.take_video_active,
        ):
            result = ctx.submit(cmd.CaptureSceneSnapshot())
            if result.ok:
                self._selected_snapshot = result.entity_id
                self._selected_id = -1
                self._view_needs_fit = True
            self._error = "" if result.ok else result.message
        imgui.end_group()
        group(transport_width)
        self._draw_transport_header(ctx, take_times, time_width)
        imgui.end_group()
        group(range_width)
        self._draw_take_range(ctx, take_times, range_widths)
        imgui.end_group()
        group(view_width)
        follow_inline = button_row_layout(
            (follow_label_width, follow_width, icon_width, icon_width),
            imgui.get_content_region_avail().x,
            gap,
        )
        _toolbar_status(ctx.tr("Follow"), ctx.theme.text_disabled, scale, width=follow_label_width)
        if follow_inline[1]:
            imgui.same_line()
        mode = segmented_control(
            "timeline-follow",
            follow_labels,
            ("off", "page", "locked").index(self._follow_mode),
            width=min(follow_width, imgui.get_content_region_avail().x),
            theme=ctx.theme,
        )
        if ("off", "page", "locked")[mode] != self._follow_mode:
            self._set_follow_mode(("off", "page", "locked")[mode])
        imgui.set_item_tooltip(
            ctx.tr(
                "Page at the edge, or lock the playhead in place. Right-drag turns following off."
            )
        )
        if follow_inline[2]:
            imgui.same_line()
        self._draw_view_controls(ctx)
        if follow_inline[3]:
            imgui.same_line(0, gap)
        if _command_button(
            "##timeline-options", "options", ctx.tr("Timeline settings"), ctx.theme, scale
        ):
            imgui.open_popup("timeline-options")
        imgui.push_style_var(imgui.StyleVar_.window_padding, (10 * scale, 8 * scale))
        if imgui.begin_popup("timeline-options"):
            if ctx.recording_config is not None:
                titles = (ctx.tr("Start delay (s)"), ctx.tr("End hold (s)"))
                label_width = max(imgui.calc_text_size(title).x for title in titles) + 8 * scale
                flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
                if imgui.begin_table(
                    "timeline_recording_settings", 2, flags, (label_width + 140 * scale, 0)
                ):
                    imgui.table_setup_column(
                        "label", imgui.TableColumnFlags_.width_fixed, label_width
                    )
                    imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
                    for field, title in zip(("countdown", "end_hold"), titles, strict=True):
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.align_text_to_frame_padding()
                        imgui.text_disabled(title)
                        imgui.table_next_column()
                        imgui.set_next_item_width(-1)
                        changed, value = imgui.input_float(
                            "##timeline-" + field,
                            getattr(ctx.recording_config, field),
                            0.5,
                            5.0,
                            "%.1f",
                        )
                        if changed:
                            ctx.set_recording_config(
                                replace(ctx.recording_config, **{field: value})
                            )
                    imgui.end_table()
            imgui.end_popup()
        imgui.end_group()
        imgui.pop_style_var(3)

    def _draw_take_range(self, ctx, take_times, widths):
        scale = ctx.style_scale
        gap = 5 * scale
        recording = ctx.session.state_take_recording
        loop = ctx.session.state_take_loop
        endpoints = (
            [take_times[loop[0]], take_times[loop[1]]]
            if loop
            else [take_times[0], take_times[-1]]
            if take_times
            else [self._view_start, self._view_end]
        )
        inline = button_row_layout(widths, imgui.get_content_region_avail().x, gap)
        imgui.set_next_item_width(widths[0])
        if imgui.begin_combo("##timeline-take", ctx.tr("Take 1" if take_times else "No take")):
            imgui.text_disabled(self._take_status(ctx, take_times))
            if take_times and not recording and imgui.selectable(ctx.tr("Clear take"), False)[0]:
                ctx.submit(cmd.ClearStateTake())
            imgui.end_combo()
        values, changed = [], False
        for i, (name, title, value) in enumerate(
            zip(("start", "end"), ("Range start (s)", "Range end (s)"), endpoints, strict=True),
            start=1,
        ):
            if inline[i]:
                imgui.same_line()
            imgui.set_next_item_width(widths[i])
            edited, value = imgui.drag_float("##timeline-range-" + name, value, 0.01, format="%.2f")
            imgui.set_item_tooltip(ctx.tr(title))
            changed |= edited
            values.append(value)
        start, end = values
        if changed and math.isfinite(start) and math.isfinite(end) and end > start:
            if len(take_times) > 1:
                first, last = (
                    nearest_take_frame(take_times, start),
                    nearest_take_frame(take_times, end),
                )
                if first < last:
                    ctx.submit(cmd.SetStateTakeLoop(first, last))
            else:
                self._view_start, self._view_end, self._view_needs_fit = start, end, False
        if inline[3]:
            imgui.same_line()
        if _command_button(
            "##timeline-loop",
            "loop",
            ctx.tr("Clear range" if loop is not None else "Loop"),
            ctx.theme,
            scale,
            selected=loop is not None,
            enabled=len(take_times) > 1 and not recording,
        ):
            ctx.submit(
                cmd.SetStateTakeLoop() if loop else cmd.SetStateTakeLoop(0, len(take_times) - 1)
            )

    def _draw_transport_header(self, ctx, take_times, time_width):
        scale = ctx.style_scale
        width = imgui.get_content_region_avail().x
        button_width = _command_button_width("", scale)
        enabled = (
            bool(take_times) and not ctx.session.state_take_recording and not ctx.take_video_active
        )
        playing = ctx.session.state_take_playing
        actions = (
            ("first", "first", cmd.SeekStateTake(0), "First frame"),
            (
                "previous",
                "previous",
                cmd.SeekStateTake(ctx.session.state_take_cursor - 1),
                "Previous frame",
            ),
            (
                "play-pause",
                "pause" if playing else "play",
                cmd.PauseStateTake() if playing else cmd.PlayStateTake(),
                "Pause" if playing else "Replay",
            ),
            ("next", "next", cmd.SeekStateTake(ctx.session.state_take_cursor + 1), "Next frame"),
            ("last", "last", cmd.SeekStateTake(max(0, len(take_times) - 1)), "Last frame"),
        )
        widths = (button_width,) * len(actions) + (time_width,)
        inline = button_row_layout(widths, width, imgui.get_style().item_spacing.x)
        for i, (name, icon, command, label) in enumerate(actions):
            if inline[i]:
                imgui.same_line()
            if _command_button(
                f"##take-{name}",
                icon,
                ctx.tr(label),
                ctx.theme,
                scale,
                enabled=enabled,
                selected=name == "play-pause" and playing,
            ):
                result = ctx.submit(command)
                self._error = "" if result.ok else result.message
        if inline[len(actions)]:
            imgui.same_line()
        _toolbar_status(f"{self._playhead:.3f} s", ctx.theme.text_disabled, scale, width=time_width)

    def _draw_view_controls(self, ctx):
        if _command_button("##key-view", "fit", ctx.tr("View all"), ctx.theme, ctx.style_scale):
            self._view_needs_fit = True

    def _set_follow_mode(self, mode: str) -> None:
        if mode == "locked":
            fraction = (self._playhead - self._view_start) / max(
                self._view_end - self._view_start, _MIN_TIMELINE_SPAN
            )
            self._locked_fraction = fraction if 0.05 <= fraction <= 0.95 else 0.15
        self._follow_mode = mode
        self._last_followed_playhead = None

    def _seek_time(self, ctx: PanelContext, take_times: Sequence[float], time: float) -> None:
        if ctx.session.state_take_recording or ctx.take_video_active:
            return
        if take_times:
            index = nearest_take_frame(take_times, time)
            if index != ctx.session.state_take_cursor or ctx.session.state_take_playing:
                result = ctx.submit(cmd.SeekStateTake(index))
                self._error = "" if result.ok else result.message
                if not result.ok:
                    return
            self._playhead = take_times[index]
            self._seen_take_cursor = index
        else:
            self._playhead = time

    @staticmethod
    def _take_status(ctx: PanelContext, take_times: Sequence[float]) -> str:
        session = ctx.session
        cursor = session.state_take_cursor
        if session.state_take_recording:
            return f"{ctx.tr('REC')} · {ctx.tr('Recorded frames')}: {len(take_times)}"
        if session.state_take_playing:
            return f"{ctx.tr('PLAY')} · {ctx.tr('frame')}: {cursor + 1}/{len(take_times)}"
        if take_times:
            if cursor >= 0:
                return f"{ctx.tr('frame')}: {cursor + 1}/{len(take_times)}"
            return f"{ctx.tr('Recorded frames')}: {len(take_times)}"
        supported = bool(session.adapter.caps.simulation and session.adapter.caps.state_snapshots)
        return ctx.tr("no recorded take" if supported else "state recording unavailable")

    def _draw_dope_sheet(
        self,
        ctx: PanelContext,
        models,
        keyframes: tuple[KeyframeInfo, ...],
        keyframe_by_id: dict[int, KeyframeInfo],
        take_times: Sequence[float],
        editable: bool,
    ) -> None:
        scale = ctx.style_scale
        available = max(1.0, float(imgui.get_content_region_avail().x))
        ruler_height = (_COMMAND_HEIGHT_PT + 4) * scale
        tracks = 3 if take_times else 2
        height = max(
            ruler_height + tracks * 20 * scale,
            min(ruler_height + tracks * 30 * scale, imgui.get_content_region_avail().y - 8 * scale),
        )
        channel_width = timeline_channel_width(available, scale)
        lo_vec = imgui.get_cursor_screen_pos()
        lo = (float(lo_vec.x), float(lo_vec.y))
        hi = (lo[0] + available, lo[1] + height)
        time_lo = lo[0] + channel_width
        time_hi = hi[0]
        time_width = max(1.0, time_hi - time_lo)

        draw_list = imgui.get_window_draw_list()
        splitter = imgui.ImDrawListSplitter()
        splitter.split(draw_list, 2)
        splitter.set_current_channel(draw_list, 1)
        imgui.set_cursor_screen_pos((lo[0] + 2 * scale, lo[1] + 2 * scale))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (5 * scale, 0))
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (8 * scale, max(0, (_COMMAND_HEIGHT_PT * scale - imgui.get_font_size()) * 0.5)),
        )
        self._draw_model_header(ctx, models, keyframes, editable, channel_width - 4 * scale)
        imgui.pop_style_var(2)
        imgui.set_cursor_screen_pos(lo)
        splitter.set_current_channel(draw_list, 0)

        flags = (
            imgui.ButtonFlags_.mouse_button_left.value
            | imgui.ButtonFlags_.mouse_button_right.value
            | imgui.ButtonFlags_.mouse_button_middle.value
        )
        imgui.invisible_button("##keyframe-dope-sheet", imgui.ImVec2(available, height), flags)
        timeline_id = imgui.get_item_id()
        hovered = imgui.is_item_hovered()
        mouse = imgui.get_mouse_pos()
        mouse_xy = (float(mouse.x), float(mouse.y))
        over_timeline = hovered and time_lo <= mouse_xy[0] <= time_hi
        ctx.status_hints = timeline_status_hints(
            ctx.tr,
            has_range=ctx.session.state_take_loop is not None,
            bindings=ctx.input_bindings or DEFAULT_INPUT_BINDINGS,
        )
        if over_timeline:
            # The dope sheet uses the wheel for zoom. Owning the wheel here
            # prevents the docked Keyframes window from scrolling as well.
            imgui.set_item_key_owner(imgui.Key.mouse_wheel_y)

        owns_escape = (
            imgui.is_window_focused()
            and not ctx.popup_owned_frame
            and (self._pointer_mode == "range" or ctx.session.state_take_loop is not None)
        )
        if owns_escape:
            imgui.internal.set_key_owner(
                imgui.Key.escape, timeline_id, imgui.internal.InputFlagsPrivate_.lock_until_release
            )

        fitted = self._view_needs_fit or self._view_model_id != self._model_id
        if fitted:
            self._view_start, self._view_end = fitted_timeline_range(
                tuple(key.time for key in keyframes)
                + tuple(take_times)
                + tuple(item.time for item in ctx.session.scene_snapshots)
                + (self._playhead,),
                self._playhead,
            )
            self._view_model_id = self._model_id
            self._view_needs_fit = False

        io = imgui.get_io()
        bindings = ctx.input_bindings or DEFAULT_INPUT_BINDINGS
        pointer = bindings.pointer_frame(ctx.input_claim or InputClaim())
        pan_press = bindings.pointer_match(PointerAction.TIMELINE_PAN, pointer, press=True)
        range_press = bindings.pointer_match(PointerAction.TIMELINE_RANGE, pointer, press=True)
        select_press = bindings.pointer_match(PointerAction.TIMELINE_SCRUB, pointer, press=True)
        load_press = bindings.pointer_match(PointerAction.TIMELINE_LOAD, pointer, press=True)
        if over_timeline and (pan_press or range_press) and not self._pointer_mode:
            self._drag_start_x = mouse_xy[0]
            self._pointer_chord = range_press or pan_press
            if range_press:
                if len(take_times) > 1 and not ctx.session.state_take_recording:
                    self._pointer_mode = "range"
                    time = timeline_x_to_time(
                        mouse_xy[0], self._view_start, self._view_end, time_lo, time_hi
                    )
                    self._range_anchor = nearest_take_frame(take_times, time)
                    self._range_preview = (self._range_anchor, self._range_anchor)
            else:
                self._pointer_mode = "pan"

        zooming = bool(
            over_timeline
            and bindings.pointer_match(PointerAction.TIMELINE_ZOOM, pointer)
            and not self._pointer_mode
        )
        if zooming:
            anchor = timeline_x_to_time(
                mouse_xy[0], self._view_start, self._view_end, time_lo, time_hi
            )
            if self._follow_mode == "locked":
                anchor = self._playhead
            self._view_start, self._view_end = zoom_timeline_range(
                self._view_start, self._view_end, anchor, float(io.mouse_wheel)
            )
        if (
            self._pointer_mode == "pan"
            and pointer.held(self._pointer_chord)
            and abs(mouse_xy[0] - self._drag_start_x) >= float(io.mouse_drag_threshold)
        ):
            self._follow_mode = "off"
            shift = -float(io.mouse_delta.x) * (self._view_end - self._view_start) / time_width
            self._view_start += shift
            self._view_end += shift

        if (
            not self._pointer_mode
            and not zooming
            and not fitted
            and (
                ctx.session.state_take_playing
                or ctx.session.state_take_recording
                or self._playhead != self._last_followed_playhead
            )
        ):
            self._view_start, self._view_end = follow_timeline_range(
                self._view_start,
                self._view_end,
                self._playhead,
                self._follow_mode,
                self._locked_fraction,
            )
        self._last_followed_playhead = self._playhead

        row_height = (height - ruler_height) / (3 if take_times else 2)
        marker_y = lo[1] + ruler_height + row_height * 0.5
        take_y = marker_y + row_height
        snapshot_y = take_y + row_height if take_times else take_y
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
        snapshot_positions = {
            item.snapshot_id: timeline_time_to_x(
                item.time, self._view_start, self._view_end, time_lo, time_hi
            )
            for item in ctx.session.scene_snapshots
        }
        snapshot_hit = (
            self._hit_marker(snapshot_positions, snapshot_y, marker_radius, mouse_xy)
            if hovered
            else -1
        )
        hit_id = (
            self._hit_marker(marker_positions, marker_y, marker_radius, mouse_xy) if hovered else -1
        )

        if (
            over_timeline
            and (select_press or load_press)
            and not self._pointer_mode
            and not ctx.take_video_active
        ):
            self._pointer_chord = load_press or select_press
            if snapshot_hit >= 0:
                self._selected_snapshot = snapshot_hit
                self._selected_id = -1
                if (
                    load_press
                    and ctx.session.paused
                    and not ctx.session.state_take_playing
                    and not ctx.session.state_take_recording
                ):
                    result = ctx.submit(cmd.RestoreSceneSnapshot(snapshot_hit))
                    self._error = "" if result.ok else result.message
                    if result.ok:
                        self._playhead = next(
                            item.time
                            for item in ctx.session.scene_snapshots
                            if item.snapshot_id == snapshot_hit
                        )
            elif hit_id >= 0:
                self._selected_snapshot = -1
                self._selected_id = hit_id
                self._selection_generation = -1
                marker = keyframe_by_id[hit_id]
                self._playhead = marker.time
                if editable:
                    self._pointer_mode = "key"
                    self._drag_id = hit_id
                    self._drag_start_x = mouse_xy[0]
                    self._drag_offset_x = marker_positions[hit_id] - mouse_xy[0]
                    self._drag_preview_time = marker.time
                    self._drag_moved = False
                if editable and load_press:
                    self._load_keyframe(ctx, marker)
            else:
                if not ctx.session.state_take_recording:
                    self._pointer_mode = "scrub"
                    self._scrub_resume = ctx.session.state_take_playing
                    self._selected_id = -1
                    self._selection_generation = -1

        if self._pointer_mode == "scrub" and pointer.held(self._pointer_chord):
            x = min(time_hi, max(time_lo, mouse_xy[0]))
            self._seek_time(
                ctx,
                take_times,
                timeline_x_to_time(x, self._view_start, self._view_end, time_lo, time_hi),
            )

        if self._pointer_mode == "range" and pointer.held(self._pointer_chord):
            x = min(time_hi, max(time_lo, mouse_xy[0]))
            time = timeline_x_to_time(x, self._view_start, self._view_end, time_lo, time_hi)
            index = nearest_take_frame(take_times, time)
            self._range_preview = (min(index, self._range_anchor), max(index, self._range_anchor))

        if (
            owns_escape
            and not io.want_text_input
            and imgui.internal.is_key_pressed(imgui.Key.escape, 0, timeline_id)
        ):
            if self._pointer_mode == "range":
                self._range_preview = None
                self._pointer_mode = "cancelled"
            elif ctx.session.state_take_loop is not None:
                ctx.submit(cmd.SetStateTakeLoop())

        if self._drag_id >= 0 and pointer.held(self._pointer_chord):
            self._drag_moved = (
                self._drag_moved or abs(mouse_xy[0] - self._drag_start_x) > 3.0 * scale
            )
            if self._drag_moved:
                drag_x = min(time_hi, max(time_lo, mouse_xy[0] + self._drag_offset_x))
                self._drag_preview_time = timeline_x_to_time(
                    drag_x, self._view_start, self._view_end, time_lo, time_hi
                )
                self._playhead = self._drag_preview_time
        released = self._pointer_chord is not None and not pointer.held(self._pointer_chord)
        if self._drag_id >= 0 and released:
            if self._drag_moved and editable:
                self._retime_keyframe(ctx, self._drag_id, self._drag_preview_time)
            self._drag_id = -1
            self._drag_moved = False

        if released and self._pointer_mode in (
            "key",
            "scrub",
        ):
            if self._pointer_mode == "scrub" and self._scrub_resume:
                result = ctx.submit(cmd.PlayStateTake())
                self._error = "" if result.ok else result.message
            self._pointer_mode = ""
            self._scrub_resume = False
        if released and self._pointer_mode in (
            "range",
            "pan",
            "cancelled",
        ):
            if self._pointer_mode == "range" and self._range_preview is not None:
                first, last = self._range_preview
                command = (
                    cmd.SetStateTakeLoop(first, last) if first < last else cmd.SetStateTakeLoop()
                )
                result = ctx.submit(command)
                self._error = "" if result.ok else result.message
            self._pointer_mode = ""
            self._range_preview = None

        self._paint_dope_sheet(
            ctx,
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
        overlay = ImguiDraw2D()
        imgui.push_clip_rect((time_lo, lo[1] + ruler_height), (time_hi, hi[1]), True)
        for snapshot_id in decimated_marker_ids(
            tuple(snapshot_positions.items()),
            time_lo,
            time_hi,
            marker_radius * 1.5,
            (self._selected_snapshot, snapshot_hit),
        ):
            x = snapshot_positions[snapshot_id]
            radius = marker_radius * 0.7
            _draw_command_icon(
                overlay,
                (x, snapshot_y),
                "key",
                ctx.theme.info if snapshot_id != self._selected_snapshot else ctx.theme.primary,
                radius / 6,
            )
        imgui.pop_clip_rect()
        if snapshot_hit >= 0 and hovered:
            snapshot = next(
                item for item in ctx.session.scene_snapshots if item.snapshot_id == snapshot_hit
            )
            imgui.set_tooltip(
                f"{snapshot.name}  {snapshot.time:g} s\n{ctx.tr('Double-click to load')}"
            )
        elif hit_id >= 0 and hovered:
            key = keyframe_by_id[hit_id]
            imgui.set_tooltip(
                f"{key.name or ctx.tr('keyframe')}  ·  {key.time:g} s\n{ctx.tr('Double-click to load')}"
            )
        splitter.merge(draw_list)

    def _paint_dope_sheet(
        self,
        ctx: PanelContext,
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

        loop = self._range_preview or ctx.session.state_take_loop
        if loop is not None:
            start_x, end_x = (
                timeline_time_to_x(
                    take_times[index], self._view_start, self._view_end, time_lo, time_hi
                )
                for index in loop
            )
            left, right = max(time_lo, start_x), min(time_hi, end_x)
            if left <= right:
                overlay.rect_filled(
                    (left, ruler_bottom), (right, hi[1]), with_alpha(_LOOP_COLOR, 0.12)
                )
                overlay.rect_filled(
                    (left, lo[1]), (right, ruler_bottom), with_alpha(_LOOP_COLOR, 0.3)
                )
                for x in (start_x, end_x):
                    if time_lo <= x <= time_hi:
                        overlay.line((x, lo[1]), (x, hi[1]), _LOOP_COLOR, 1.5 * ctx.style_scale)

        step = nice_timeline_step(
            self._view_end - self._view_start, time_hi - time_lo, 90 * ctx.style_scale
        )
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

        inset = 10.0 * ctx.style_scale
        label_width = max(1.0, time_lo - lo[0] - 2.0 * inset)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect((lo[0], ruler_bottom), (time_lo, hi[1]), True)
        rows = [(marker_y, "Model Keyframes")]
        if take_times:
            rows.append((take_y, "Recorded Take"))
        rows.append((take_y + (take_y - marker_y) if take_times else take_y, "Snapshots"))
        for center_y, label in rows:
            text = ctx.tr(label)
            size = imgui.calc_text_size(text, wrap_width=label_width)
            draw_list.add_text(
                imgui.get_font(),
                imgui.get_font_size(),
                (lo[0] + inset, center_y - size.y * 0.5),
                imgui.get_color_u32(imgui.Col_.text),
                text,
                wrap_width=label_width,
            )
        draw_list.pop_clip_rect()
        draw_list.push_clip_rect((time_lo, lo[1]), (time_hi, hi[1]), True)

        playhead_x = timeline_time_to_x(
            self._playhead, self._view_start, self._view_end, time_lo, time_hi
        )
        if time_lo <= playhead_x <= time_hi:
            overlay.line(
                (playhead_x, lo[1]), (playhead_x, hi[1]), theme.danger, 1.5 * ctx.style_scale
            )
            overlay.convex_fill(
                tuple(
                    (playhead_x + px * ctx.style_scale, lo[1] + py * ctx.style_scale)
                    for px, py in _rounded_command_icon_path(
                        ((-5.0, 0.0), (5.0, 0.0), (0.0, 7.0)),
                        0.6,
                    )
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
                (0.98, 0.67, 0.24, 1.0) if hovered else theme.warning if selected else theme.primary
            )
            points = tuple(
                (x + px, marker_y + py)
                for px, py in _rounded_command_icon_path(
                    (
                        (0.0, -marker_radius),
                        (marker_radius, 0.0),
                        (0.0, marker_radius),
                        (-marker_radius, 0.0),
                    ),
                    0.6 * ctx.style_scale,
                )
            )
            overlay.convex_fill(points, fill)
            overlay.polyline(points, theme.bg_window, 1.0, closed=True)
            if key.keyframe_id == ctx.session.active_keyframe:
                overlay.circle_filled((x, marker_y), 2.25 * ctx.style_scale, theme.primary_bright)

        cursor = ctx.session.state_take_cursor
        length = 8.0 * ctx.style_scale
        color = with_alpha(theme.primary, 0.72)
        for left, right in recorded_take_spans(
            take_times, self._view_start, self._view_end, time_lo, time_hi
        ):
            if right - left <= 2.0:
                x = (left + right) * 0.5
                overlay.line((x, take_y - length), (x, take_y + length), color, 2.0)
            else:
                overlay.rect_filled((left, take_y - length), (right, take_y + length), color)
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

        draw_list.pop_clip_rect()
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

    def _snapshot_created(self, result) -> None:
        if result.ok:
            self._selected_id = result.entity_id
            self._selection_generation = -1
            self._view_needs_fit = True
            self._error = ""
        else:
            self._error = result.message

    def _snapshot_updated(self, result, time: float) -> None:
        if result.ok:
            self._selection_generation = -1
            self._playhead = float(time)
            self._error = ""
        else:
            self._error = result.message

    def _snapshot_removed(self, result) -> None:
        if result.ok:
            self._clear_selection()
        else:
            self._error = result.message

    def _retime_keyframe(self, ctx: PanelContext, keyframe_id: int, time: float) -> None:
        properties = ctx.session.keyframe_properties(keyframe_id)
        if properties is None:
            self._error = ctx.tr("Keyframe state is no longer available")
            return
        ctx.submit_model_edit(
            _set_keyframe_command(properties, properties.name, time),
            lambda result: self._snapshot_updated(result, time),
        )

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
        scale = ctx.style_scale
        available = imgui.get_content_region_avail().x
        gap = imgui.get_style().item_spacing.x
        time_width = min(available, 100 * scale)
        fields_inline = available >= 220 * scale
        name_width = min(280 * scale, available - time_width - gap if fields_inline else available)
        imgui.set_next_item_width(max(1, name_width))
        _changed, self._name = imgui.input_text("##keyframe-name", self._name)
        imgui.set_item_tooltip(ctx.tr("name"))
        if fields_inline:
            imgui.same_line()
        imgui.set_next_item_width(time_width)
        _changed, self._time = imgui.input_double("##keyframe-time", self._time, 0.0, 0.0, "%.9g s")
        imgui.set_item_tooltip(ctx.tr("time"))

        dirty = self._name.strip() != properties.name or self._time != properties.time
        if not editable or not dirty or not self._name.strip():
            imgui.begin_disabled()
        action_labels = (ctx.tr("Apply"), ctx.tr("Load"), ctx.tr("Delete"))
        action_widths = tuple(button_width(label) for label in action_labels)
        actions_inline = (
            fields_inline and available >= name_width + time_width + sum(action_widths) + 5 * gap
        )
        if actions_inline:
            imgui.same_line()
        inline = button_row_layout(
            action_widths,
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        if imgui.button(action_labels[0]):
            time = float(self._time)
            ctx.submit_model_edit(
                _set_keyframe_command(properties, self._name.strip(), time),
                lambda result: self._snapshot_updated(result, time),
            )
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
            ctx.submit_model_edit(
                cmd.RemoveModelKeyframe(properties.keyframe_id), self._snapshot_removed
            )
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
        if model_id == self._model_id:
            return
        self._model_id = model_id
        self._keyframe_cache_key = None
        self._keyframe_cache = ()
        self._keyframe_by_id = {}
        self._view_needs_fit = True
        self._view_model_id = -1
        self._seen_active_id = -2
        self._seen_take_cursor = -2
        self._pointer_mode = ""
        self._range_preview = None
        self._last_followed_playhead = None
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
