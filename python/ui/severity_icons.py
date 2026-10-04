"""Shared severity icon geometry for runtime panels, previews and asset export."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from imgui_bundle import imgui

from mojive.geometry2d.curves import CORNER_SMOOTHING, capped_polyline_points, circular_stroke_mesh

from .imgui_draw import ImguiDraw2D

# Author the family on one grid, then scale every dimension together.
_ICON_GRID = 24.0
_FRAME_DIAMETER = 22.56
_FRAME_STROKE = 1.32
_MARK_STROKE = 1.74
_DOT_DIAMETER = _MARK_STROKE * 1.18
_STEM_HEIGHT = 6.96
_MARK_GAP = 1.74
_MARK_PADDING = 0.60
SEVERITY_FRAME_PADDING = (_ICON_GRID - _FRAME_DIAMETER) * 0.5
SEVERITY_FRAME_MIN_SEGMENTS = 32
SEVERITY_FRAME_SEGMENT_DENSITY = 2.0


def severity_frame(size: float) -> tuple[float, float]:
    """Return the canonical frame centerline radius and stroke for SVG and meshes."""
    stroke = _FRAME_STROKE * size / _ICON_GRID
    return _FRAME_DIAMETER * size / _ICON_GRID * 0.5 - stroke * 0.5, stroke


def _circle_outline(radius: float, count: int):
    return tuple(
        (
            radius * math.cos(index * math.tau / count),
            radius * math.sin(index * math.tau / count),
        )
        for index in range(count)
    )


def _place_interior_mark(contours, inner, padding):
    """Center the visible mark bounds and fit them inside the frame clearance."""
    boundary = np.asarray(inner, np.float64)
    following = np.roll(boundary, -1, axis=0)
    cross = boundary[:, 0] * following[:, 1] - following[:, 0] * boundary[:, 1]
    center = (boundary.min(axis=0) + boundary.max(axis=0)) * 0.5
    edges = following - boundary
    inward = np.stack((-edges[:, 1], edges[:, 0]), axis=1) * np.sign(cross.sum())
    lengths = np.linalg.norm(inward, axis=1)
    inward /= lengths[:, None]
    ink = np.concatenate(contours)
    offset = (ink.min(axis=0) + ink.max(axis=0)) * 0.5
    ink -= offset
    available = np.sum((center - boundary) * inward, axis=1) - padding
    extent = np.maximum(-(inward @ ink.T).min(axis=1), 1e-9)
    factor = min(1.0, float(np.min(available / extent)))
    return tuple((np.asarray(points) - offset) * factor + center for points in contours)


@lru_cache(maxsize=96)
def severity_meshes(size: float, kind: str, smoothing: float = CORNER_SMOOTHING):
    """Cache filled contours for one severity glyph at its displayed pixel size.

    The glyph is authored on a 24-unit grid. Frame, stroke, mark, gap, and clearance
    all use the same scale so smaller Output glyphs retain the large glyph's proportions.
    """

    scale = size / _ICON_GRID
    radius, stroke = severity_frame(size)
    ring = circular_stroke_mesh(
        radius,
        stroke,
        max(SEVERITY_FRAME_MIN_SEGMENTS, math.ceil(size * SEVERITY_FRAME_SEGMENT_DENSITY)),
    )
    inner = ring[3]
    meshes = [ring]

    def solid(points):
        points = tuple(map(tuple, points))
        indices = tuple(v for i in range(1, len(points) - 1) for v in (0, i, i + 1))
        meshes.append((points, indices, points, ()))

    inner_radius = radius - stroke * 0.5
    # One internal stroke unit keeps i, !, their dots, and the cross visibly related.
    inner_stroke = _MARK_STROKE * scale

    def stem(a, b):
        return np.asarray(
            capped_polyline_points(
                (a, b), inner_stroke, round_start=True, round_end=True, smoothing=smoothing
            )
        )

    if kind == "error":
        # Two diagonals carry more ink than one stem, so control their visual
        # weight through length while preserving the shared stroke width.
        reach = inner_radius * 0.34
        contours = (stem((-reach, -reach), (reach, reach)), stem((-reach, reach), (reach, -reach)))
    else:
        # A small circular dot loses visible area to antialiasing on every edge,
        # so give it slight optical overshoot relative to the shared stem width.
        dot_radius = _DOT_DIAMETER * scale * 0.5
        dot = np.asarray(_circle_outline(dot_radius, 32))
        bar_height = _STEM_HEIGHT * scale
        axis_half = (bar_height - inner_stroke) * 0.5
        gap = _MARK_GAP * scale
        bar = stem((0.0, -axis_half), (0.0, axis_half))
        bar[:, 1] += dot_radius + gap - bar[:, 1].min()
        if kind == "warning":
            # Reflect i into ! before balancing the ink against the frame interior.
            bar[:, 1] *= -1
            bar = bar[::-1]
        contours = (dot, bar)
    for contour in _place_interior_mark(contours, inner, _MARK_PADDING * scale):
        solid(contour)
    return tuple(meshes)


def severity_icon(draw, center, size: float, kind: str, color) -> None:
    """Draw one severity glyph centered on the requested point.

    Every severity uses the same circular frame, optical size, and baseline.
    """

    native = isinstance(draw, ImguiDraw2D)
    smoothing = draw.corner_smoothing if native else CORNER_SMOOTHING
    fringe = 1.0
    if native:
        density = imgui.get_io().display_framebuffer_scale
        fringe /= max(1.0, density.x, density.y)
    for vertices, indices, outline, hole in severity_meshes(size, kind, smoothing):
        if not native:
            # Exporters and measurement adapters consume absolute coordinates.
            vertices = tuple((x + center[0], y + center[1]) for x, y in vertices)
            outline = tuple((x + center[0], y + center[1]) for x, y in outline)
            hole = tuple((x + center[0], y + center[1]) for x, y in hole)
        draw.indexed_fill(
            vertices,
            indices,
            color,
            outline=outline,
            hole=hole,
            origin=center if native else None,
            fringe_width=fringe,
        )
