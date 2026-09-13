"""Value contracts for logical 2D coordinates, color, and stroke geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def finite_tuple(values, count: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} requires {count} finite values")
    return result


@dataclass(frozen=True, slots=True)
class Affine2D:
    """Row-major 2x3 affine transform; ``translation @ scale`` scales first.

    Axis orientation belongs to the caller (Canvas2D defaults to world XY).
    Singular transforms are valid and may produce empty rasterized geometry.
    """

    values: tuple[float, ...] = _IDENTITY

    def __post_init__(self):
        object.__setattr__(self, "values", finite_tuple(self.values, 6, "Affine2D"))

    @classmethod
    def translation(cls, x: float, y: float) -> Affine2D:
        return cls((1, 0, x, 0, 1, y))

    @classmethod
    def scale(cls, x: float, y: float | None = None) -> Affine2D:
        return cls((x, 0, 0, 0, x if y is None else y, 0))

    @classmethod
    def rotation(cls, radians: float) -> Affine2D:
        cosine, sine = math.cos(radians), math.sin(radians)
        return cls((cosine, -sine, 0, sine, cosine, 0))

    def __matmul__(self, other: Affine2D) -> Affine2D:
        if not isinstance(other, Affine2D):
            return NotImplemented
        if self.values == _IDENTITY and type(other) is Affine2D:
            return other
        if other.values == _IDENTITY and type(self) is Affine2D:
            return self
        a, b, c, d, e, f = self.values
        g, h, i, j, k, m = other.values
        return Affine2D(
            (
                a * g + b * j,
                a * h + b * k,
                a * i + b * m + c,
                d * g + e * j,
                d * h + e * k,
                d * i + e * m + f,
            )
        )

    def apply(self, points) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 2 or not np.isfinite(array).all():
            raise ValueError("Points must have shape [..., 2] and finite coordinates")
        matrix = np.asarray(self.values).reshape(2, 3)
        return array @ matrix[:, :2].T + matrix[:, 2]


@dataclass(frozen=True, slots=True)
class StrokeStyle:
    """Stroke outline with explicit cap, join, and width coordinate space.

    Local widths transform with the full affine matrix. Screen widths are in
    physical pixels. A zero width is empty, not an implicit hairline.
    """

    width: float = 1.0
    cap: str = "butt"
    join: str = "miter"
    miter_limit: float = 4.0
    space: str = "local"

    def __post_init__(self):
        width, limit = finite_tuple((self.width, self.miter_limit), 2, "StrokeStyle")
        if width < 0 or limit < 1:
            raise ValueError("Stroke width must be nonnegative and miter limit at least one")
        if self.cap not in ("butt", "round", "square"):
            raise ValueError("Stroke cap must be butt, round, or square")
        if self.join not in ("miter", "round", "bevel"):
            raise ValueError("Stroke join must be miter, round, or bevel")
        if self.space not in ("local", "screen"):
            raise ValueError("Stroke space must be local or screen")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "miter_limit", limit)
