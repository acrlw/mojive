"""Bounded CPU preparation for retained canvas paths; no GPU resources."""

import math
from collections import OrderedDict

import numpy as np

from mojive.geometry2d import PathBuilder2D, compile_fill, compile_stroke

_CACHE_BYTES = 8 * 1024 * 1024
_CACHE_ENTRIES = 128


class PathCache:
    def __init__(self):
        self._entries = OrderedDict()
        self.bytes = 0

    def prepare(self, path, tolerance, style=None):
        key = path, float(tolerance), style
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        budgets = {
            "max_vertices": 65536,
            "max_indices": 196608,
            "max_scratch_bytes": 8 * 1024 * 1024,
        }
        mesh = (
            compile_fill(path, tolerance=tolerance, **budgets)
            if style is None
            else compile_stroke(path, style, tolerance=tolerance, **budgets)
        )
        triangles = mesh.positions[mesh.indices].reshape(-1, 3, 2)
        triangles.flags.writeable = False
        # Do not retain huge authored paths, even when their output is small.
        cost = triangles.nbytes + len(path.commands) * 256
        if cost <= _CACHE_BYTES:
            while self._entries and (
                len(self._entries) >= _CACHE_ENTRIES or self.bytes + cost > _CACHE_BYTES
            ):
                old_key, old = self._entries.popitem(last=False)
                self.bytes -= old.nbytes + len(old_key[0].commands) * 256
            self._entries[key] = triangles
            self.bytes += cost
        return triangles

    def clear(self):
        self._entries.clear()
        self.bytes = 0


def polygon_path(points):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Polygon points must be finite [N, 2]")
    path = PathBuilder2D()
    if len(points) >= 3:
        path.move_to(*points[0])
        for point in points[1:]:
            path.line_to(*point)
        path.close()
    return path.finish()


def ellipse_points(center, radii, start, sweep, segments, rotation=0.0):
    center = np.asarray(center, np.float64)
    radii = np.asarray(radii, np.float64)
    if (
        center.shape != (2,)
        or radii.shape != (2,)
        or not np.isfinite((*center, *radii, start, sweep, rotation)).all()
        or (radii < 0).any()
    ):
        raise ValueError("Ellipse requires finite center, angles, and nonnegative radii")
    if not isinstance(segments, int) or not 2 <= segments <= 4096:
        raise ValueError("Curve segments must be between 2 and 4096")
    angles = np.linspace(start, start + sweep, segments + 1)
    points = np.column_stack((np.cos(angles), np.sin(angles))) * radii
    c, s = math.cos(rotation), math.sin(rotation)
    return points @ np.array(((c, s), (-s, c))) + center


def rounded_rectangle_path(lo, hi, radius):
    x0, y0 = lo
    x1, y1 = hi
    if not np.isfinite((x0, y0, x1, y1, radius)).all() or x1 < x0 or y1 < y0 or radius < 0:
        raise ValueError("Rectangle requires ordered finite bounds and a nonnegative radius")
    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    return (
        PathBuilder2D()
        .move_to(x0 + r, y0)
        .line_to(x1 - r, y0)
        .arc(x1 - r, y0 + r, r, r, -math.pi / 2, math.pi / 2)
        .line_to(x1, y1 - r)
        .arc(x1 - r, y1 - r, r, r, 0, math.pi / 2)
        .line_to(x0 + r, y1)
        .arc(x0 + r, y1 - r, r, r, math.pi / 2, math.pi / 2)
        .line_to(x0, y0 + r)
        .arc(x0 + r, y0 + r, r, r, math.pi, math.pi / 2)
        .close()
        .finish()
    )
