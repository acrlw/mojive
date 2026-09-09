"""Wrapping count filters and font-independent diagnostic glyphs."""

from __future__ import annotations

import math
from functools import lru_cache

from imgui_bundle import imgui

from ...curves2d import (
    CORNER_SMOOTHING,
    capped_polyline_points,
    offset_closed_path,
    smooth_polygon_corners,
)
from ..draw2d import ImguiDraw2D, text_line_y
from . import PanelContext, button_row_layout


def severity_group(level: str) -> str:
    """Group informational records without dropping debug, success, or critical logs."""
    if level in {"error", "critical", "fatal"}:
        return "error"
    return "warning" if level in {"warning", "warn"} else "info"


def severity_color(theme, level: str):
    return {"info": theme.info, "warning": theme.warning, "error": theme.danger}[
        severity_group(level)
    ]


def _tint(base, color, amount):
    return (*(a + (b - a) * amount for a, b in zip(base[:3], color[:3], strict=True)), 1.0)


@lru_cache(maxsize=64)
def warning_outline(size: float, smoothing: float = CORNER_SMOOTHING):
    """Cache curvature-continuous triangle corners at the actual display size."""
    points = ((0, -size * 0.5), (size * 0.433, size * 0.25), (-size * 0.433, size * 0.25))
    path = smooth_polygon_corners(points, size * 0.085, (0, 1, 2), smoothing=smoothing)
    return tuple(map(tuple, path.tolist()))


@lru_cache(maxsize=96)
def severity_meshes(size: float, kind: str, smoothing: float = CORNER_SMOOTHING):
    """Cache separate outer and internal stroke widths with clear small-size gaps.

    Native ImGui strokes center their fringe on a path; filled capsule strokes
    add it outside. Mixing them made the cross appear heavier than its ring.
    Cache local contours and indices so per-frame work is a native translation.
    """
    stroke = max(1.05, size * 0.06)
    inner_stroke = max(1.3, size * 0.075)
    radius = size * 0.42
    count = max(32, math.ceil(size * 2))
    circle = tuple(
        (radius * math.cos(i * math.tau / count), radius * math.sin(i * math.tau / count))
        for i in range(count)
    )
    path = warning_outline(size, smoothing) if kind == "warning" else circle
    outer = tuple(map(tuple, offset_closed_path(path, stroke * 0.5).tolist()))
    inner = tuple(map(tuple, offset_closed_path(path, -stroke * 0.5).tolist()))
    count = len(outer)
    indices = tuple(
        v
        for i in range(count)
        for v in (
            i,
            (i + 1) % count,
            count + i,
            (i + 1) % count,
            count + (i + 1) % count,
            count + i,
        )
    )
    meshes = [(outer + inner, indices, outer, inner)]

    def solid(points):
        points = tuple(map(tuple, points))
        indices = tuple(v for i in range(1, len(points) - 1) for v in (0, i, i + 1))
        meshes.append((points, indices, points, ()))

    def stem(a, b):
        solid(
            capped_polyline_points(
                (a, b), inner_stroke, round_start=True, round_end=True, smoothing=smoothing
            )
        )

    if kind == "error":
        d = radius * 0.36
        stem((-d, -d), (d, d))
        stem((-d, d), (d, -d))
    else:
        dot_radius = max(1.0, inner_stroke * 0.65)
        gap = max(1.4, size * 0.04)
        dot_y = size * (0.075 if kind == "warning" else -0.22)
        solid(
            tuple(
                (
                    dot_radius * math.cos(i * math.tau / 24),
                    dot_y + dot_radius * math.sin(i * math.tau / 24),
                )
                for i in range(24)
            )
        )
        if kind == "warning":
            bottom = dot_y - dot_radius - gap - inner_stroke * 0.5
            stem((0, min(bottom - size * 0.12, -size * 0.23)), (0, bottom))
        else:
            top = dot_y + dot_radius + gap + inner_stroke * 0.5
            stem((0, top), (0, max(top + inner_stroke * 0.4, size * 0.22 - inner_stroke * 0.5)))
    return tuple(meshes)


def severity_icon(draw, center, size: float, kind: str, color) -> None:
    if kind == "warning":
        center = (center[0], center[1] + size * 0.10)
    density = imgui.get_io().display_framebuffer_scale
    fringe = 1.0 / max(1.0, density.x, density.y)
    for vertices, indices, outline, hole in severity_meshes(size, kind, draw.corner_smoothing):
        draw.indexed_fill(
            vertices, indices, color, outline=outline, hole=hole, origin=center, fringe_width=fringe
        )


def filter_pills(
    ctx: PanelContext,
    scope: str,
    items: tuple[tuple[str, str, str], ...],
    selected: set[str],
    *,
    compact: bool = False,
    colors: dict | None = None,
) -> str | None:
    """Draw equal-height filters which reflow before reaching the panel edge.

    Each item is (stable ID, visible label, optional severity icon). Tooltip copy
    stays with the caller, since hierarchy counts and diagnostic levels differ.
    """
    style = imgui.get_style()
    height = float(imgui.get_frame_height())
    padding = (
        6.0 if compact else max(style.frame_padding.x / ctx.style_scale, 10.0)
    ) * ctx.style_scale
    icon_size = float(imgui.get_font_size())
    gap = (4.0 if compact else 6.0) * ctx.style_scale
    available = max(1.0, float(imgui.get_content_region_avail().x))
    widths = tuple(
        min(
            available,
            imgui.calc_text_size("999" if icon else label).x
            + 2 * padding
            + (icon_size + gap if icon else 0),
        )
        for _, label, icon in items
    )
    item_gap = 5.0 * ctx.style_scale if compact else style.item_spacing.x
    inline = button_row_layout(widths, available, item_gap)
    clicked = None
    for index, (key, label, icon) in enumerate(items):
        if inline[index]:
            imgui.same_line(0, item_gap)
        active = key in selected
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, height * 0.5)
        accent = (
            severity_color(ctx.theme, icon)
            if icon
            else colors.get(key)
            if colors and active
            else None
        )
        background = ctx.theme.bg_frame_active if active else ctx.theme.bg_frame
        if accent is not None:
            background = _tint(ctx.theme.bg_frame, accent, 0.22 if active else 0.055)
        for slot, fill in (
            (imgui.Col_.button, background),
            (
                imgui.Col_.button_hovered,
                _tint(background, accent, 0.14) if accent else ctx.theme.bg_frame_hovered,
            ),
            (
                imgui.Col_.button_active,
                _tint(background, accent, 0.24) if accent else ctx.theme.bg_frame_active,
            ),
        ):
            imgui.push_style_color(slot, imgui.ImVec4(*fill))
        pressed = imgui.button(f"##{scope}-{key}", imgui.ImVec2(widths[index], height))
        imgui.pop_style_color(3)
        imgui.pop_style_var()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        color = ctx.theme.text if active else ctx.theme.text_disabled
        if accent is not None:
            color = accent if active else _tint(ctx.theme.text_disabled, accent, 0.4)
        draw = ImguiDraw2D()
        text_width = imgui.calc_text_size(label).x
        content_width = text_width + (icon_size + gap if icon else 0.0)
        x = lo.x + max(padding, (widths[index] - content_width) * 0.5)
        y = (lo.y + hi.y) * 0.5
        imgui.push_clip_rect((lo.x + padding * 0.5, lo.y), (hi.x - padding * 0.5, hi.y), True)
        if icon:
            severity_icon(draw, (x + icon_size * 0.5, y), icon_size, icon, color)
            x += icon_size + gap
        draw.text((x, text_line_y(draw, y)), color, label)
        imgui.pop_clip_rect()
        imgui.set_item_tooltip(ctx.tr(key.capitalize()) if icon else label)
        if pressed:
            clicked = key
    return clicked
