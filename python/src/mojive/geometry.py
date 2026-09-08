"""Shared primitive-dimension conventions for editor controls and gizmos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MeshShape


@dataclass(frozen=True)
class DimensionHandle:
    """Map one body axis, or the center, to an authored dimension."""

    axis: int | None
    parameter: int
    world_to_value: float
    label: str


@dataclass(frozen=True)
class GeometryDimensions:
    """User-facing dimensions and their independent viewport handles."""

    label: str
    values: tuple[float, ...]
    handles: tuple[DimensionHandle, ...]

    def array(self) -> np.ndarray:
        return np.asarray(self.values, np.float32)

    def handle(self, axis: int | None) -> DimensionHandle | None:
        return next((item for item in self.handles if item.axis == axis), None)


def geometry_dimensions(shape: MeshShape, size) -> GeometryDimensions | None:
    """Return conventional dimensions and independent handles for a primitive."""

    value = np.asarray(size, np.float32).reshape(3)
    if shape is MeshShape.PLANE:
        return GeometryDimensions(
            "width / length",
            tuple(value[:2] * 2.0),
            (
                DimensionHandle(0, 0, 2.0, "width"),
                DimensionHandle(1, 1, 2.0, "length"),
            ),
        )
    if shape is MeshShape.BOX:
        return GeometryDimensions(
            "width / depth / height",
            tuple(value * 2.0),
            tuple(
                DimensionHandle(axis, axis, 2.0, label)
                for axis, label in enumerate(("width", "depth", "height"))
            ),
        )
    if shape is MeshShape.SPHERE:
        if np.allclose(value, value[0], rtol=1e-5, atol=1e-7):
            return GeometryDimensions(
                "radius",
                (float(value[0]),),
                (DimensionHandle(None, 0, 1.0, "radius"),),
            )
        return GeometryDimensions(
            "radii x / y / z",
            tuple(value),
            tuple(DimensionHandle(axis, axis, 1.0, f"{'XYZ'[axis]} radius") for axis in range(3)),
        )
    if shape in (MeshShape.CYLINDER, MeshShape.CONE, MeshShape.CAPSULE_SHAFT):
        shaft = shape is MeshShape.CAPSULE_SHAFT
        return GeometryDimensions(
            "diameter / shaft length" if shaft else "diameter / height",
            (float(value[0] * 2.0), float(value[2] * 2.0)),
            (
                DimensionHandle(0, 0, 2.0, "diameter"),
                DimensionHandle(2, 1, 2.0, "shaft length" if shaft else "height"),
            ),
        )
    return None


def geometry_size_from_dimensions(shape: MeshShape, size, dimensions) -> np.ndarray:
    """Convert conventional authored dimensions to the render-size vector."""

    value = np.asarray(size, np.float32).reshape(3).copy()
    dimensions = np.maximum(np.asarray(dimensions, np.float32).reshape(-1), 0.002)
    half = dimensions * 0.5
    if shape is MeshShape.PLANE:
        value[:2] = half[:2]
    elif shape is MeshShape.BOX:
        value[:] = half[:3]
    elif shape is MeshShape.SPHERE:
        if len(dimensions) == 3:
            value[:] = dimensions[:3]
        else:
            value[:] = dimensions[0]
    elif shape in (MeshShape.CYLINDER, MeshShape.CONE, MeshShape.CAPSULE_SHAFT):
        value[:2] = half[0]
        value[2] = half[1]
    return value


__all__ = [
    "DimensionHandle",
    "GeometryDimensions",
    "geometry_dimensions",
    "geometry_size_from_dimensions",
]
