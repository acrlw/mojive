"""Immutable compiled geometry and CPU contour triangulation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from mojive.native import native_module

from .buffers import snapshot_array
from .compiler import Contour2D, flatten_path
from .path import Path2D
from .style import Affine2D, StrokeStyle


def _validate_budgets(*budgets: int) -> None:
    for budget in budgets:
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("Tessellation budgets must be positive integers")


def _indices(values, width: int, vertex_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != (2 if width else 1) or (width and array.shape[1] != width):
        raise ValueError("Invalid mesh index shape")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Mesh indices must be integers")
    if array.size and (array.min() < 0 or array.max() >= vertex_count):
        raise ValueError("Mesh index outside vertex range")
    return snapshot_array(array.astype(np.uint32, copy=False))


@dataclass(frozen=True, slots=True, eq=False)
class Mesh2D:
    """Owned 2D triangles and directed exterior/hole boundary edges.

    Internal triangle edges never appear in ``boundary_edges``. Coordinates are
    triangulated in normalized float precision and returned as float64 in the
    coordinate space specified by the compiler (normally local).
    ``scratch_peak_bytes`` measures native tessellation workspace, excluding
    the input and returned arrays, which have separate vertex/index limits.
    ``coverage`` defaults to one and linearly interpolates edge coverage across
    each triangle. It multiplies both premultiplied color and alpha at rasterization.
    """

    positions: np.ndarray
    indices: np.ndarray
    boundary_edges: np.ndarray
    scratch_peak_bytes: int = 0
    coverage: np.ndarray | None = None
    bounds: tuple[float, float, float, float] | None = field(init=False, repr=False)

    def __post_init__(self):
        points = np.asarray(self.positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Mesh positions must be finite [N, 2]")
        indices = _indices(self.indices, 0, len(points))
        if indices.size % 3:
            raise ValueError("Triangle indices must be a multiple of three")
        edges = _indices(self.boundary_edges, 2, len(points))
        object.__setattr__(self, "positions", snapshot_array(points))
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "boundary_edges", edges)
        coverage = (
            np.ones(len(points), np.float32)
            if self.coverage is None
            else np.asarray(self.coverage, np.float32)
        )
        if (
            coverage.shape != (len(points),)
            or not np.isfinite(coverage).all()
            or np.any((coverage < 0) | (coverage > 1))
        ):
            raise ValueError("Mesh coverage must be finite [N] values between zero and one")
        object.__setattr__(self, "coverage", snapshot_array(coverage))
        bounds = (*points.min(axis=0), *points.max(axis=0)) if indices.size else None
        object.__setattr__(self, "bounds", bounds)


def tessellate_contours(
    contours: Iterable[Contour2D],
    *,
    fill_rule: str = "nonzero",
    max_vertices: int = 1_000_000,
    max_indices: int = 3_000_000,
    max_scratch_bytes: int = 64 * 1024 * 1024,
) -> Mesh2D:
    """Triangulate holes and self-intersections without a graphics context.

    Open contours are implicitly closed. Empty and degenerate regions produce
    empty geometry. Invalid inputs and exhausted budgets fail atomically.
    """
    if fill_rule not in ("nonzero", "evenodd"):
        raise ValueError("Fill rule must be nonzero or evenodd")
    _validate_budgets(max_vertices, max_indices, max_scratch_bytes)
    parts, offsets = [], [0]
    for contour in contours:
        if not isinstance(contour, Contour2D):
            raise TypeError("Expected Contour2D")
        count = offsets[-1] + len(contour.points)
        if count > min(max_vertices, 2**31 - 1) or len(offsets) > max_vertices:
            raise ValueError("Contour input budget exceeded")
        parts.append(contour.points)
        offsets.append(count)
    points = np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.float64)
    native = native_module()
    if not hasattr(native, "tessellate_contours"):
        raise RuntimeError("Native tessellation is unavailable; rebuild with make cpp-python")
    return Mesh2D(
        *native.tessellate_contours(
            points,
            np.asarray(offsets, dtype=np.uint32),
            fill_rule == "evenodd",
            max_vertices,
            max_indices,
            max_scratch_bytes,
        )
    )


def compile_fill(
    path: Path2D,
    *,
    projection: Affine2D = Affine2D(),
    tolerance: float = 0.25,
    max_vertices: int = 1_000_000,
    max_indices: int = 3_000_000,
    max_scratch_bytes: int = 64 * 1024 * 1024,
) -> Mesh2D:
    """Flatten and triangulate a reusable path, preserving its fill rule."""
    contours = flatten_path(
        path, projection=projection, tolerance=tolerance, max_points=max_vertices
    )
    return tessellate_contours(
        contours,
        fill_rule=path.fill_rule,
        max_vertices=max_vertices,
        max_indices=max_indices,
        max_scratch_bytes=max_scratch_bytes,
    )


def compile_stroke(
    path: Path2D,
    style: StrokeStyle = StrokeStyle(),
    *,
    projection: Affine2D = Affine2D(),
    tolerance: float = 0.25,
    max_vertices: int = 1_000_000,
    max_indices: int = 3_000_000,
    max_scratch_bytes: int = 64 * 1024 * 1024,
) -> Mesh2D:
    """Compile one stroke as a filled union, without overlapping triangles.

    Local widths return local coordinates; projection controls curve quality.
    Screen widths return physical-pixel coordinates after projection, including
    framebuffer scale. Do not apply that projection twice. Zero width is empty;
    coincident endpoints use the selected cap (butt: empty, round: disk,
    square: axis-aligned square). Miter overflow uses a bevel join.
    """
    if not isinstance(path, Path2D) or not isinstance(style, StrokeStyle):
        raise TypeError("Expected Path2D and StrokeStyle")
    if not isinstance(projection, Affine2D):
        raise TypeError("Expected Affine2D")
    _validate_budgets(max_vertices, max_indices, max_scratch_bytes)
    native = native_module()
    if not hasattr(native, "compile_stroke"):
        raise RuntimeError("Native stroke compilation is unavailable; rebuild with make cpp-python")
    return Mesh2D(
        *native.compile_stroke(
            path.packed(),
            style.width,
            ("butt", "round", "square").index(style.cap),
            ("miter", "round", "bevel").index(style.join),
            style.miter_limit,
            style.space == "screen",
            projection.values,
            float(tolerance),
            max_vertices,
            max_indices,
            max_scratch_bytes,
        )
    )
