"""CPU path preparation; native capability checks happen at compilation time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mojive.native import native_module

from .buffers import snapshot_array
from .path import Path2D
from .style import Affine2D


@dataclass(frozen=True, slots=True)
class Contour2D:
    """Owned immutable local points and authored closure state."""

    points: np.ndarray
    closed: bool

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Contour points must be finite [N, 2]")
        object.__setattr__(self, "points", snapshot_array(points))


def flatten_path(
    path: Path2D,
    *,
    projection: Affine2D = Affine2D(),
    tolerance: float = 0.25,
    max_points: int = 1_000_000,
) -> tuple[Contour2D, ...]:
    """Compile local contours using a physical-pixel projection error bound.

    Projection includes framebuffer scale. Points stay in local coordinates;
    the same prepared shape may be translated and recolored without compilation.
    Native support is required; this function never falls back to ImGui.
    """
    if not isinstance(path, Path2D) or not isinstance(projection, Affine2D):
        raise TypeError("Expected Path2D and Affine2D")
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 1:
        raise ValueError("Point budget must be a positive integer")
    native = native_module()
    if not getattr(native, "has_geometry2d", False):
        raise RuntimeError(
            "Native geometry compilation is unavailable; rebuild with make cpp-python"
        )
    return tuple(
        Contour2D(points, closed)
        for points, closed in native.flatten_path(
            path.packed(), projection.values, float(tolerance), max_points
        )
    )
