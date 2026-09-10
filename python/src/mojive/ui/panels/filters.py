"""Wrapping count filters and font-independent diagnostic glyphs."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
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


def node_filter_color(theme, kind):
    """Keep entity badges and Hierarchy filters on the same semantic palette."""
    return {
        "all": theme.primary,
        "joint": theme.accent_purple_bright,
        "flex": theme.info,
        "skin": theme.node_color("site"),
    }.get(str(kind), theme.node_color(kind))


@lru_cache(maxsize=64)
def warning_triangle(size: float, smoothing: float = CORNER_SMOOTHING):
    """Cache an equilateral warning frame that fills the requested box.

    A warning sign reads as a triangle only while its sides stay equal, so the authoring
    points use the equilateral apex ratio and the rounded outline is then scaled until it
    spans the same box as a circle outline of this size. Rounding the corners shrinks the
    raw path, which is why the scale is solved rather than assumed.
    """

    target = size * (_FRAME_SPAN - _STROKE_RATIO)
    low, high = size * 0.2, size * 0.75
    for _ in range(18):
        half = (low + high) * 0.5
        width = _triangle_span(half, size, smoothing)
        if width < target:
            low = half
        else:
            high = half
    half = (low + high) * 0.5
    apex = half * _EQUILATERAL_APEX * 0.5
    points = ((0.0, -apex), (half, half), (-half, half))
    path = smooth_polygon_corners(
        points, min(size * 0.10, half * 0.30), (0, 1, 2), smoothing=smoothing
    )
    path = np.asarray(path, np.float64)
    # Center the box itself: an equilateral triangle is bottom-heavy around its centroid.
    path[:, 1] -= (path[:, 1].min() + path[:, 1].max()) * 0.5
    return tuple(map(tuple, path.tolist()))


def _triangle_span(half: float, size: float, smoothing: float) -> float:
    """Return the rounded triangle's horizontal span before stroke expansion."""

    apex = half * _EQUILATERAL_APEX * 0.5
    points = ((0.0, -apex), (half, half), (-half, half))
    path = np.asarray(
        smooth_polygon_corners(
            points, min(size * 0.10, half * 0.30), (0, 1, 2), smoothing=smoothing
        )
    )
    return float(path[:, 0].max() - path[:, 0].min())


# Frame strokes leave a smaller interior; all mark geometry is budgeted there.
_MIN_STROKE = 1.15
_FRAME_SPAN = 0.94
_STROKE_RATIO = 0.085
_EQUILATERAL_APEX = math.sqrt(3.0)


def _clamped(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _circle_outline(radius: float, count: int):
    return tuple(
        (
            radius * math.cos(index * math.tau / count),
            radius * math.sin(index * math.tau / count),
        )
        for index in range(count)
    )


def _place_interior_mark(contours, inner, padding, *, triangular=False):
    """Balance ink area around the frame's interior center while preserving clearance."""
    boundary = np.asarray(inner, np.float64)
    following = np.roll(boundary, -1, axis=0)
    cross = boundary[:, 0] * following[:, 1] - following[:, 0] * boundary[:, 1]
    center = (boundary.min(axis=0) + boundary.max(axis=0)) * 0.5
    edges = following - boundary
    inward = np.stack((-edges[:, 1], edges[:, 0]), axis=1) * np.sign(cross.sum())
    lengths = np.linalg.norm(inward, axis=1)
    inward /= lengths[:, None]
    moments = []
    for contour in contours:
        next_point = np.roll(contour, -1, axis=0)
        weights = contour[:, 0] * next_point[:, 1] - next_point[:, 0] * contour[:, 1]
        area = weights.sum()
        centroid = ((contour + next_point) * weights[:, None]).sum(axis=0) / (3 * area)
        moments.append((abs(area), centroid))
    offset = sum(area * centroid for area, centroid in moments) / sum(area for area, _ in moments)
    ink = np.concatenate(contours) - offset
    if triangular:
        # Center the complete mark with equal clearance to the three straight inner
        # sides. A point incenter alone ignores the stem and dot's different extents.
        sides = np.argsort(lengths)[-3:]
        center = np.linalg.solve(
            np.column_stack((inward[sides], -np.ones(3))),
            np.sum(boundary[sides] * inward[sides], axis=1) - (inward[sides] @ ink.T).min(axis=1),
        )[:2]
    available = np.sum((center - boundary) * inward, axis=1) - padding
    extent = np.maximum(-(inward @ ink.T).min(axis=1), 1e-9)
    factor = min(1.0, float(np.min(available / extent)))
    return tuple((np.asarray(points) - offset) * factor + center for points in contours)


@lru_cache(maxsize=96)
def severity_meshes(size: float, kind: str, smoothing: float = CORNER_SMOOTHING):
    """Cache filled contours for one severity glyph at its displayed pixel size.

    Every feature carries an absolute floor, because the Output list draws the glyph at
    the font size and a purely proportional stroke disappears below a physical pixel.
    The frame outline fills the requested size and the interior mark takes the room that
    remains, so the frames stay legible and the mark never crowds them.
    """

    stroke = _clamped(size * _STROKE_RATIO, _MIN_STROKE, size * 0.11)
    radius = size * 0.5 * _FRAME_SPAN - stroke * 0.5
    if kind == "warning":
        # The triangle carries its own stroke compensation, since its corners round in.
        path = warning_triangle(size, smoothing)
    else:
        path = _circle_outline(radius, max(32, math.ceil(size * 2)))
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

    inner_radius = radius - stroke * 0.5
    inner_stroke = max(0.85, stroke * (0.65 if kind == "error" else 0.85))

    def stem(a, b):
        return np.asarray(
            capped_polyline_points(
                (a, b), inner_stroke, round_start=True, round_end=True, smoothing=smoothing
            )
        )

    if kind == "error":
        reach = inner_radius * 0.44
        contours = (stem((-reach, -reach), (reach, reach)), stem((-reach, reach), (reach, -reach)))
    else:
        # Circular caps and dot share one profile. The isolated dot needs optical
        # overshoot because its antialiased edge occupies more of the visible mark.
        profile = np.asarray(_circle_outline(inner_stroke * 0.5, 32))
        diameter = inner_stroke * 1.12
        gap = max(1.0, inner_stroke * 1.6)
        bar = profile.copy()
        bar[:, 1] += np.where(bar[:, 1] >= 0, inner_stroke * 1.25, -inner_stroke * 1.25)
        bar[:, 1] += diameter * 0.5 + gap - bar[:, 1].min()
        dot = profile * 1.12
        if kind == "warning":
            # Reflect i into ! before balancing the ink against the frame interior.
            bar[:, 1] *= -1
            bar = bar[::-1]
        contours = (dot, bar)
    for contour in _place_interior_mark(
        contours, inner, max(0.6, stroke * 0.4), triangular=kind == "warning"
    ):
        solid(contour)
    return tuple(meshes)


def severity_icon(draw, center, size: float, kind: str, color) -> None:
    """Draw one severity glyph centered on the requested point.

    Every frame fills the same box, so a row of mixed severities keeps one optical
    weight and one baseline.
    """

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
