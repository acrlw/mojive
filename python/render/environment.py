"""Shared environment geometry, independent of GPU resources and shader layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive.types import MeshShape

if TYPE_CHECKING:
    from mojive.render.scene import RenderScene


def reflection_plane(scene: RenderScene, index: int) -> tuple[float, float, float, float] | None:
    """Return the world plane of a planar surface, or None for a singular transform."""
    transform = np.asarray(scene.transforms[index], np.float64)
    try:
        normal = np.linalg.inv(transform[:3, :3]).T @ np.array([0.0, 0.0, 1.0])
    except np.linalg.LinAlgError:
        return None
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    normal /= length
    shape = scene.bucket_keys[int(scene.bucket[index])][0].shape
    point = transform[:3, 3]
    if shape is MeshShape.BOX:
        point = point + transform[:3, 2]
    d = -float(np.dot(normal, point))
    return (float(normal[0]), float(normal[1]), float(normal[2]), d)


def classic_skybox_vertices(slices: int) -> np.ndarray:
    """Closed unit cylinder used by MuJoCo's classic skybox display list."""
    vertices: list[tuple[float, float, float]] = []
    for index in range(slices):
        angle0 = 2.0 * np.pi * index / slices
        angle1 = 2.0 * np.pi * (index + 1) / slices
        x0, y0 = np.cos(angle0), np.sin(angle0)
        x1, y1 = np.cos(angle1), np.sin(angle1)
        lower0 = (x0, y0, -1.0)
        lower1 = (x1, y1, -1.0)
        upper0 = (x0, y0, 1.0)
        upper1 = (x1, y1, 1.0)
        vertices.extend((lower0, lower1, upper1, lower0, upper1, upper0))
        vertices.extend(((0.0, 0.0, 1.0), upper0, upper1))
        vertices.extend(((0.0, 0.0, -1.0), lower1, lower0))
    return np.asarray(vertices, np.float32)
