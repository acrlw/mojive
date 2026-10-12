"""Canonical workspace icon roles and cached local glyph geometry."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from imgui_bundle import imgui

from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.severity_icons import severity_icon

from .design import BODY_SIZE
from .widget_style import PANEL

ROOT = Path(__file__).parent
GLYPHS = json.loads((ROOT / "icons.json").read_text())

ICON_GRID = 24
ICON_STROKE = 1.75
IconRole = Literal["activity", "title", "toolbar", "subtle", "inline", "status"]


@dataclass(frozen=True, slots=True)
class IconRoleSpec:
    """Canonical canvas and default hit-slot sizes in logical points."""

    canvas_size: float
    slot_size: float


ICON_ROLES: dict[IconRole, IconRoleSpec] = {
    "activity": IconRoleSpec(24, 36),
    "title": IconRoleSpec(22, 30),
    "toolbar": IconRoleSpec(18, 28),
    "subtle": IconRoleSpec(16, 28),
    "inline": IconRoleSpec(BODY_SIZE, 28),
    "status": IconRoleSpec(12, 20),
}


def icon_role_canvas(role: IconRole) -> float:
    """Keep one family scale; a small authored mark remains small beside its peers."""
    return ICON_ROLES[role].canvas_size


@lru_cache(maxsize=512)
def icon_geometry(name, size):
    """Retain local vertices; moving a dock window only translates the submitted range."""
    scale = size / ICON_GRID
    left, top, right, bottom = GLYPHS[name]["bounds"]
    dx = (ICON_GRID / 2 - (left + right) / 2) * scale
    dy = (ICON_GRID / 2 - (top + bottom) / 2) * scale
    result = []
    for contour in GLYPHS[name]["contours"]:
        points = tuple((x * scale + dx, y * scale + dy) for x, y in contour["points"])
        paths = (
            tuple(
                tuple((x * scale + dx, y * scale + dy) for x, y in path)
                for path in contour["stroke_paths"]
            )
            if "stroke_paths" in contour
            else (points,)
        )
        result.append((contour, points, paths, contour["width"] * scale))
    return tuple(result)


def draw_icon(draw, name, x, y, size, tint, scale):
    """Paint one glyph in a screen-positioned canonical canvas."""
    severity = {"info": "info", "warn": "warning", "err": "error"}.get(name)
    if severity is not None:
        severity_icon(
            ImguiDraw2D(draw),
            (x + size * scale / 2, y + size * scale / 2),
            size * scale,
            severity,
            (*[channel / 255 for channel in tint[:3]], tint[3] / 255 * imgui.get_style().alpha),
        )
        return
    first_vertex = len(draw.vtx_buffer)
    # Filled strokes retain the canonical weight below one pixel as well.
    painter = ImguiIconDraw(
        ImguiDraw2D(draw), min(1.0, ICON_STROKE * size * scale / (2 * ICON_GRID))
    )
    alpha = imgui.get_style().alpha
    for contour, pts, paths, thickness in icon_geometry(name, size * scale):
        if len(pts) < 2:
            continue
        fill = contour["fill"]
        if fill != "none":
            fill_color = PANEL if fill.startswith("var") else tint
            fill_color = (
                *(channel / 255 for channel in fill_color[:3]),
                fill_color[3] / 255 * contour["opacity"] * contour["fill_opacity"] * alpha,
            )
            painter.fringed_concave_fill(pts, fill_color)
        if contour["stroke"] != "none":
            stroke_color = (
                *(channel / 255 for channel in tint[:3]),
                tint[3] / 255 * contour["opacity"] * contour["stroke_opacity"] * alpha,
            )
            closed = contour["closed"] and "stroke_paths" not in contour
            for path in paths:
                painter.polyline(
                    path, stroke_color, thickness, closed=closed, cap="round", smoothing=0.0
                )
    # Local tessellation avoids precision loss at large screen coordinates.
    imgui.internal.shade_verts_transform_pos(
        draw, first_vertex, len(draw.vtx_buffer), (0, 0), 1.0, 0.0, (x, y)
    )
