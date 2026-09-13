"""Consistent contour antialiasing for ImGui icon primitives."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from mojive.geometry2d.compiler import Contour2D
from mojive.geometry2d.curves import capped_polyline_points, polyline_ribbon, smooth_rect_points
from mojive.geometry2d.mesh import tessellate_contours
from mojive.geometry2d.polygons import remove_interior_loops, signed_polygon_area

from .imgui_draw import ImguiDraw2D


@lru_cache(maxsize=512)
def _closed_stroke(points, width):
    # Repeat the adjacent segments so the seam gets the same join as every
    # other vertex. The two closed contours bound one hollow triangle strip.
    left, right, _ = polyline_ribbon((points[-1], *points, points[0]), width)
    left, right = left[1:-1], right[1:-1]
    outer, inner = (right, left) if signed_polygon_area(points) > 0 else (left, right)
    # Tight G3 corners can fold the inward offset. Remove those loops before
    # both triangulation and fringe construction, so neither can overdraw.
    outer, inner = remove_interior_loops(outer), remove_interior_loops(inner)
    mesh = tessellate_contours(
        (Contour2D(np.asarray(outer), True), Contour2D(np.asarray(inner), True)),
        fill_rule="evenodd",
    )
    return tuple(map(tuple, mesh.positions)), tuple(mesh.indices), outer, inner


@lru_cache(maxsize=512)
def _circle_points(center, radius, segments):
    return tuple(
        (
            center[0] + radius * math.cos(i * math.tau / segments),
            center[1] + radius * math.sin(i * math.tau / segments),
        )
        for i in range(segments)
    )


class ImguiIconDraw(ImguiDraw2D):
    """Use the same filled-stroke coverage for open lines, frames and solid marks.

    Native ImGui strokes center their AA ramp on the boundary, whereas our
    filled contours add it outside. Mixing them makes spokes heavier than
    frames. Icons use filled contours throughout, with a shared fringe that
    shrinks with their canonical stroke instead of adding a fixed pixel.
    """

    def __init__(self, draw: ImguiDraw2D, fringe_width: float):
        super().__init__(draw._dl, corner_smoothing=draw.corner_smoothing)
        self.fringe_width = fringe_width

    def _write_anti_alias_fringe(self, outline, rgba, *, inside=False, width=1.0):
        super()._write_anti_alias_fringe(
            outline, rgba, inside=inside, width=min(width, self.fringe_width)
        )

    def polyline(self, points, color, width, *, closed=False, cap="butt", smoothing=None):
        if cap not in {"butt", "round", "round_start", "round_end"}:
            raise ValueError(f"unknown polyline cap: {cap!r}")
        if not closed:
            outline = capped_polyline_points(
                points,
                width,
                round_start=cap in {"round", "round_start"},
                round_end=cap in {"round", "round_end"},
                smoothing=self.corner_smoothing if smoothing is None else smoothing,
            )
            if outline:
                self.fringed_concave_fill(outline, color)
            return
        path = (
            points if isinstance(points, tuple) else tuple((float(x), float(y)) for x, y in points)
        )
        if len(path) < 3 or width <= 0:
            return
        vertices, indices, outline, hole = _closed_stroke(path, float(width))
        self.indexed_fill(vertices, indices, color, outline=outline, hole=hole)

    def circle(self, center, radius, color, width=1.0, *, segments=0):
        points = _circle_points(tuple(center), radius, max(64, segments))
        self.polyline(points, color, width, closed=True)

    def rect(self, lo, hi, color, width=1.0, *, rounding=0.0, smoothing=None):
        if rounding > 0:
            path = smooth_rect_points(
                *lo,
                *hi,
                rounding,
                smoothing=self.corner_smoothing if smoothing is None else smoothing,
            )
        else:
            path = (lo, (hi[0], lo[1]), hi, (lo[0], hi[1]))
        self.polyline(path, color, width, closed=True)
