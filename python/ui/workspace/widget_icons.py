"""Canonical workspace icon roles and cached local glyph geometry."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from imgui_bundle import imgui

from mojive.ui.imgui_draw import _cached_color
from mojive.ui.workspace.design import BODY_SIZE
from mojive.ui.workspace.icon_mesh import _freeze_icon_arrays, native_icon
from mojive.ui.workspace.widget_style import PANEL

ROOT = Path(__file__).parent
GLYPHS = _freeze_icon_arrays(json.loads((ROOT / "icons.json").read_text()))

ICON_GRID = 24
# Compiled from the canonical family's stroke in svg/sources.json.
ICON_STROKE = GLYPHS["camera"]["contours"][0]["width"]
IconRole = Literal["activity", "title", "toolbar", "subtle", "inline", "status"]


@dataclass(frozen=True, slots=True)
class IconRoleSpec:
    """Canonical canvas and default hit-slot sizes in logical points."""

    canvas_size: float
    slot_size: float


ICON_ROLES: dict[IconRole, IconRoleSpec] = {
    "activity": IconRoleSpec(28, 36),
    "title": IconRoleSpec(22, 30),
    "toolbar": IconRoleSpec(20, 28),
    "subtle": IconRoleSpec(16, 28),
    "inline": IconRoleSpec(BODY_SIZE, 28),
    "status": IconRoleSpec(12, 20),
}

# All reviewed status glyphs are now compiled; retained for inventory consumers.
PRODUCTION_GLYPHS = {}

# Context identifiers select a semantic master, never a different fitting algorithm.
COMPILED_ALIASES = {
    "tool-move": "move",
    "tool-rotate": "rotate",
    "tool-scale": "scale",
    "tool-world": "world",
    "tool-body": "body-frame",
    "tool-snap": "snap",
    **{
        prefix + name: name
        for prefix in ("playback-", "transport-")
        for name in (
            "previous",
            "next",
            "play",
            "pause",
            "reset",
            "record",
            "stop",
            "first",
            "last",
            "loop",
        )
    },
    "playback-more": "down",
    "transport-more": "down",
    "key-snapshot": "snapshot",
    "key-keyframe": "diamond",
    "key-add": "plus",
    "key-clear": "delete",
    "key-previous": "first",
    "key-next": "last",
    "key-fit": "fit",
    "key-follow": "follow",
    "key-follow-off": "follow-off",
    "key-follow-page": "follow-page",
    "key-follow-locked": "follow-locked",
    "key-view": "gear",
    "panel-search": "search",
    "panel-sort": "sort",
    "panel-clear": "close",
    "panel-visible": "eye",
    "panel-hidden": "eyeoff",
    "panel-perspective": "perspective",
    "panel-orthographic": "orthographic",
    "panel-right": "right",
    "panel-down": "down",
    "helper-camera": "camera",
    "helper-light": "light",
    "status-info": "info",
    "status-warning": "warn",
    "status-error": "err",
}


def icon_role_canvas(role: IconRole) -> float:
    """Keep one family scale; a small authored mark remains small beside its peers."""
    return ICON_ROLES[role].canvas_size


@lru_cache(maxsize=512)
def icon_geometry(name, size):
    """Retain local vertices; moving a dock window only translates the submitted range."""
    name = COMPILED_ALIASES.get(name, name)
    scale = size / ICON_GRID
    result = []
    for contour in GLYPHS[name]["contours"]:
        points = tuple((x * scale, y * scale) for x, y in contour["points"])
        paths = (
            tuple(
                tuple((x * scale, y * scale) for x, y in path) for path in contour["stroke_paths"]
            )
            if "stroke_paths" in contour
            else (points,)
        )
        result.append((contour, points, paths, contour["width"] * scale))
    return tuple(result)


def draw_icon(draw, name, x, y, size, tint, scale):
    """Paint one glyph in a screen-positioned canonical canvas."""
    name = COMPILED_ALIASES.get(name, name)
    alpha = imgui.get_style().alpha
    antialias = bool(draw.flags & imgui.ImDrawListFlags_.anti_aliased_fill.value)
    for background, opacity, mesh in native_icon(name, size * scale, antialias):
        color = PANEL if background else tint
        rgba = _cached_color(
            (color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255 * opacity * alpha)
        )
        mesh.draw(draw, (x, y), rgba)
