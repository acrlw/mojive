"""Timeline controls and presentation helpers."""

from __future__ import annotations

import math
import sys
from functools import lru_cache

from imgui_bundle import imgui

from mojive.geometry2d.curves import CORNER_SMOOTHING
from mojive.interaction.gizmo import _rounded_polygon_corners
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.paint_protocol import Draw2D
from mojive.ui.text_layout import fit_text, text_line_y

from ..icons import ICON_GRID, draw_icon, production_icon_metrics
from ..input_bindings import DEFAULT_INPUT_BINDINGS
from ..pointer_bindings import PointerAction
from ..viewport_widgets import (
    ToolHint,
    pointer_tool_hint,
)

_MIN_TIMELINE_SPAN = 1e-6
_COMMAND_HEIGHT_PT = 28.0
_COMMAND_ICON_PT = 16.0
_MARKER_SPACING_FACTOR = 1.5
_LOOP_COLOR = (0.98, 0.52, 0.18, 1.0)
FOLLOW_MODE_TOOLTIPS = (
    "Follow off: keep the timeline view fixed.",
    "Follow page: advance the view when the playhead reaches an edge.",
    "Follow locked: keep the playhead at its current screen position.",
)
_COMMAND_ICON_NAMES = {
    "first": "transport-first",
    "last": "transport-last",
    "previous": "transport-previous",
    "next": "transport-next",
    "play": "transport-play",
    "pause": "transport-pause",
    "stop": "transport-stop",
    "loop": "transport-loop",
    "options": "transport-more",
    "record": "transport-record",
    "add": "key-add",
    "clear": "key-clear",
    "key-previous": "key-previous",
    "key-next": "key-next",
    "fit": "key-fit",
    "follow": "key-follow",
    "view": "key-view",
    "key": "key-snapshot",
    "key-keyframe": "key-keyframe",
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


def timeline_status_hints(
    translate,
    *,
    has_range: bool = False,
    edit_lane: str = "",
    over_ruler: bool = False,
    has_selection: bool = False,
    bindings=DEFAULT_INPUT_BINDINGS,
) -> tuple[ToolHint, ...]:
    """Describe the hovered track or ruler through the shared status surface."""

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
    if edit_lane in ("model", "take"):
        selection_hint = pointer_tool_hint(
            PointerAction.TIMELINE_SCRUB,
            bindings,
            translate("Move playhead" if over_ruler else "Select"),
            hint_id="keyframes.playhead" if over_ruler else "keyframes.select",
        )
        hints = (
            *((selection_hint,) if selection_hint is not None else ()),
            ToolHint(
                "key",
                "Cmd+A" if sys.platform == "darwin" else "Ctrl+A",
                translate("Select track"),
                hint_id="keyframes.select_all",
            ),
            *(
                hint
                for hint in hints
                if hint.hint_id not in ("keyframes.range", "keyframes.playhead")
            ),
            *(hint for hint in hints if hint.hint_id == "keyframes.range"),
        )
    if has_selection:
        hints = (
            ToolHint("key", "Delete", translate("Delete selection"), hint_id="keyframes.delete"),
            *(
                (
                    ToolHint(
                        "key",
                        "Esc",
                        translate("Clear selection"),
                        hint_id="keyframes.clear_selection",
                    ),
                )
                if not has_range
                else ()
            ),
            *hints,
        )
    if has_range:
        clear_hint = pointer_tool_hint(
            PointerAction.TIMELINE_CLEAR_RANGE,
            bindings,
            translate("Clear loop range"),
            hint_id="keyframes.clear_range_pointer",
        )
        hints = (
            *((clear_hint,) if clear_hint is not None else ()),
            ToolHint("key", "Esc", translate("Clear range"), hint_id="keyframes.clear_range"),
            *hints,
        )
    return hints


def unique_keyframe_name(existing: set[str]) -> str:
    index = 1
    name = f"key{index}"
    while name in existing:
        index += 1
        name = f"key{index}"
    return name


def _draw_command_icon(
    draw, center, kind: str, color, scale: float, *, smoothing: float = CORNER_SMOOTHING
) -> None:
    """Draw one production transport or keyframe glyph in its 16 pt slot."""

    del smoothing
    try:
        name = _COMMAND_ICON_NAMES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown keyframe command icon: {kind!r}") from exc
    draw_icon(draw, center, _COMMAND_ICON_PT * float(scale), name, color)


def _command_icon_visible_width(kind: str, scale: float) -> float:
    """Return the rendered glyph width used to center icon-label pairs."""

    try:
        name = _COMMAND_ICON_NAMES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown keyframe command icon: {kind!r}") from exc
    x0, _y0, x1, _y1 = production_icon_metrics(name).bounds
    return (x1 - x0) * _COMMAND_ICON_PT * float(scale) / ICON_GRID


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
    draw: Draw2D | None = None,
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
    draw = (ImguiDraw2D() if draw is None else draw).with_corner_smoothing(smoothing)
    lo = (float(origin.x), float(origin.y))
    hi = (lo[0] + width, lo[1] + height)
    draw.rect_filled(lo, hi, background, rounding=float(imgui.get_style().frame_rounding))
    center_y = lo[1] + height * 0.5
    icon_center = (lo[0] + width * 0.5, center_y)
    if label:
        key = (imgui.get_font(), imgui.get_font_size(), kind, label, width, scale)
        cached = layouts.get(item_id) if layouts is not None else None
        if cached is None or cached[0] != key:
            icon_width = _command_icon_visible_width(kind, scale)
            gap = 7.0 * scale
            label = fit_text(draw, label, max(1.0, width - 16 * scale - icon_width - gap))
            ink = draw.text_ink_bounds(label) or (0.0, 0.0, *draw.text_size(label))
            left = (width - icon_width - gap - (ink[2] - ink[0])) * 0.5
            cached = (
                key,
                label,
                left + icon_width * 0.5,
                left + icon_width + gap - ink[0],
                text_line_y(draw, 0.0),
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


def _format_tick(value: float, step: float) -> str:
    if abs(value) < step * 1e-9:
        value = 0.0
    decimals = max(0, min(9, -math.floor(math.log10(step)))) if step < 1.0 else 0
    return f"{value:.{decimals}f}"
