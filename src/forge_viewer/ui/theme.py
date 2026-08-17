from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import NodeKind

RGBA = tuple[float, float, float, float]


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb8(r: int, g: int, b: int, a: float = 1.0) -> RGBA:

    return (r / 255.0, g / 255.0, b / 255.0, a)


def with_alpha(color: RGBA, alpha: float) -> RGBA:
    return (color[0], color[1], color[2], alpha)


def relative_luminance(color: RGBA) -> float:

    r, g, b = (_to_linear(c) for c in color[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def lstar(color: RGBA) -> float:

    y = relative_luminance(color)
    return 116.0 * y ** (1.0 / 3.0) - 16.0 if y > 0.008856 else 903.3 * y


def luma601(color: RGBA) -> float:

    return 255.0 * (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])


def hsl_saturation(color: RGBA) -> float:
    hi, lo = max(color[:3]), min(color[:3])
    if hi == lo:
        return 0.0
    mid = (hi + lo) / 2.0
    return (hi - lo) / (2.0 - hi - lo) if mid > 0.5 else (hi - lo) / (hi + lo)


def hsl_hue(color: RGBA) -> float:

    r, g, b = color[:3]
    hi, lo = max(r, g, b), min(r, g, b)
    d = hi - lo
    if d == 0:
        return 0.0
    if hi == r:
        h = ((g - b) / d) % 6.0
    elif hi == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return h * 60.0


def chroma(color: RGBA) -> float:

    r, g, b = (_to_linear(c) for c in color[:3])
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = f(x / 0.95047), f(y / 1.0), f(z / 1.08883)
    return math.hypot(500.0 * (fx - fy), 200.0 * (fy - fz))


def pack_u32(color: RGBA) -> int:

    r, g, b, a = (max(0, min(255, round(c * 255.0))) for c in color)
    return r | (g << 8) | (b << 16) | (a << 24)


PRIMARY: RGBA = rgb8(159, 174, 137)
PRIMARY_BRIGHT: RGBA = rgb8(184, 198, 163)
PRIMARY_DIM: RGBA = rgb8(104, 116, 88)


DANGER: RGBA = rgb8(208, 103, 68)
WARNING: RGBA = rgb8(201, 161, 92)

BG_WINDOW: RGBA = rgb8(30, 33, 37)
BG_CHILD: RGBA = rgb8(26, 29, 32)
BG_POPUP: RGBA = rgb8(24, 27, 30)
BG_FRAME: RGBA = rgb8(43, 47, 52)
BG_FRAME_HOVERED: RGBA = rgb8(54, 59, 65)
BG_FRAME_ACTIVE: RGBA = rgb8(64, 70, 77)
BG_HEADER: RGBA = rgb8(48, 53, 58)
BORDER: RGBA = rgb8(58, 63, 69)
TEXT: RGBA = rgb8(220, 223, 227)
TEXT_DISABLED: RGBA = rgb8(123, 129, 137)


NODE_COLORS: dict[NodeKind, RGBA] = {
    NodeKind.FLEX: rgb8(46, 53, 62),
    NodeKind.WORLD: rgb8(66, 81, 103),
    NodeKind.JOINT: rgb8(123, 84, 131),
    NodeKind.SITE: rgb8(162, 97, 124),
    NodeKind.GEOM: rgb8(172, 125, 101),
    NodeKind.LINK: rgb8(154, 150, 198),
    NodeKind.CAMERA: rgb8(138, 183, 192),
    NodeKind.ENVIRONMENT: rgb8(35, 71, 80),
    NodeKind.LIGHT: rgb8(202, 198, 152),
    NodeKind.ROBOT: rgb8(197, 222, 212),
    NodeKind.SKIN: rgb8(239, 243, 241),
}


def node_color(kind: NodeKind | str) -> RGBA:

    try:
        return NODE_COLORS[NodeKind(kind)]
    except (KeyError, ValueError):
        return rgb8(140, 145, 150)


AXIS_COLORS: dict[str, RGBA] = {
    "x": rgb8(239, 110, 106),
    "y": rgb8(84, 168, 83),
    "z": rgb8(105, 147, 246),
}
AXIS_ORDER: tuple[str, str, str] = ("x", "y", "z")


PERTURB_COMMANDED: RGBA = AXIS_COLORS["y"]

PERTURB_ACTUAL: RGBA = AXIS_COLORS["x"]


@dataclass(frozen=True)
class Theme:
    primary: RGBA = PRIMARY
    primary_bright: RGBA = PRIMARY_BRIGHT
    primary_dim: RGBA = PRIMARY_DIM
    danger: RGBA = DANGER
    warning: RGBA = WARNING
    text: RGBA = TEXT
    text_disabled: RGBA = TEXT_DISABLED
    bg_window: RGBA = BG_WINDOW
    bg_child: RGBA = BG_CHILD
    bg_popup: RGBA = BG_POPUP
    bg_frame: RGBA = BG_FRAME
    bg_frame_hovered: RGBA = BG_FRAME_HOVERED
    bg_frame_active: RGBA = BG_FRAME_ACTIVE
    bg_header: RGBA = BG_HEADER
    border: RGBA = BORDER
    node_colors: dict[NodeKind, RGBA] = field(default_factory=lambda: dict(NODE_COLORS))
    axis_colors: dict[str, RGBA] = field(default_factory=lambda: dict(AXIS_COLORS))

    def node_color(self, kind: NodeKind | str) -> RGBA:
        try:
            return self.node_colors[NodeKind(kind)]
        except (KeyError, ValueError):
            return rgb8(140, 145, 150)

    def axis_color(self, axis: int | str) -> RGBA:
        key = AXIS_ORDER[axis] if isinstance(axis, int) else axis
        return self.axis_colors[key]


THEME = Theme()


def apply(imgui: Any, theme: Theme = THEME, ui_scale: float = 1.0) -> None:

    style = imgui.get_style()
    col = imgui.Col_
    v4 = imgui.ImVec4

    def put(idx: Any, c: RGBA) -> None:
        style.set_color_(int(idx), v4(*c))

    put(col.text, theme.text)
    put(col.text_disabled, theme.text_disabled)
    put(col.window_bg, theme.bg_window)
    put(col.child_bg, theme.bg_child)
    put(col.popup_bg, theme.bg_popup)
    put(col.border, theme.border)
    put(col.border_shadow, (0.0, 0.0, 0.0, 0.0))
    put(col.frame_bg, theme.bg_frame)
    put(col.frame_bg_hovered, theme.bg_frame_hovered)
    put(col.frame_bg_active, theme.bg_frame_active)
    put(col.title_bg, theme.bg_child)
    put(col.title_bg_active, theme.bg_header)
    put(col.title_bg_collapsed, theme.bg_child)
    put(col.menu_bar_bg, theme.bg_child)
    put(col.scrollbar_bg, theme.bg_child)
    put(col.scrollbar_grab, theme.bg_frame_active)
    put(col.scrollbar_grab_hovered, theme.primary_dim)
    put(col.scrollbar_grab_active, theme.primary)
    put(col.check_mark, theme.primary_bright)
    put(col.slider_grab, theme.primary_dim)
    put(col.slider_grab_active, theme.primary)
    put(col.button, theme.bg_frame)
    put(col.button_hovered, theme.bg_frame_hovered)
    put(col.button_active, theme.primary_dim)
    put(col.header, theme.bg_header)
    put(col.header_hovered, theme.bg_frame_active)
    put(col.header_active, theme.primary_dim)
    put(col.separator, theme.border)
    put(col.separator_hovered, theme.primary_dim)
    put(col.separator_active, theme.primary)
    put(col.resize_grip, theme.bg_frame)
    put(col.resize_grip_hovered, theme.primary_dim)
    put(col.resize_grip_active, theme.primary)
    put(col.tab, theme.bg_child)
    put(col.tab_hovered, theme.primary_dim)
    put(col.tab_selected, theme.bg_header)
    put(col.tab_selected_overline, theme.primary)
    put(col.tab_dimmed, theme.bg_child)
    put(col.tab_dimmed_selected, theme.bg_frame)
    put(col.docking_preview, with_alpha(theme.primary, 0.55))
    put(col.docking_empty_bg, theme.bg_child)
    put(col.plot_lines, theme.primary)
    put(col.plot_lines_hovered, theme.primary_bright)
    put(col.plot_histogram, theme.primary_dim)
    put(col.plot_histogram_hovered, theme.primary)
    put(col.table_header_bg, theme.bg_header)
    put(col.table_border_strong, theme.border)
    put(col.table_border_light, theme.bg_frame)
    put(col.table_row_bg, (0.0, 0.0, 0.0, 0.0))
    put(col.table_row_bg_alt, (1.0, 1.0, 1.0, 0.025))
    put(col.text_selected_bg, with_alpha(theme.primary, 0.35))
    put(col.nav_cursor, theme.primary)
    put(col.drag_drop_target, theme.warning)

    style.window_rounding = 3.0
    style.child_rounding = 3.0
    style.frame_rounding = 3.0
    style.popup_rounding = 3.0
    style.grab_rounding = 3.0
    style.tab_rounding = 3.0
    style.scrollbar_rounding = 6.0
    style.window_border_size = 1.0
    style.frame_border_size = 0.0
    style.window_padding = imgui.ImVec2(8.0, 8.0)
    style.frame_padding = imgui.ImVec2(6.0, 3.0)
    style.item_spacing = imgui.ImVec2(7.0, 5.0)
    style.cell_padding = imgui.ImVec2(5.0, 2.0)
    style.indent_spacing = 16.0
    style.scrollbar_size = 12.0
    style.grab_min_size = 9.0
    style.window_title_align = imgui.ImVec2(0.0, 0.5)

    if ui_scale != 1.0:
        style.scale_all_sizes(ui_scale)
