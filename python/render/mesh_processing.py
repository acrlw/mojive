"""Optional native geometry preparation with renderer-neutral mesh results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshData


@dataclass(frozen=True)
class MeshLod:
    """A mesh with original vertex attributes and fewer triangle indices.

    ``relative_error`` is meshoptimizer's combined geometric/attribute error,
    normalized by mesh extent. The actual ratio can exceed the requested target
    to preserve topology, seams, and the error bound.
    """

    mesh: MeshData
    relative_error: float


def simplify_mesh(mesh: MeshData, *, ratio: float, max_error: float = 0.005) -> MeshLod:
    """Prepare an explicit LOD using pinned meshoptimizer; never enabled automatically.

    The optional C++ extension is required for simplification, regardless of
    the renderer used to consume the resulting MeshData. Source arrays remain
    unchanged and computation releases the GIL. There is no GPU operation.
    """
    if not np.isfinite(ratio) or not 0 < ratio <= 1 or not np.isfinite(max_error) or max_error < 0:
        raise ValueError("ratio must be in (0, 1] and max_error must be finite and nonnegative")
    if ratio == 1:
        return MeshLod(mesh, 0.0)
    from .native.device import native_module

    indices, error = native_module().simplify_mesh_indices(
        np.ascontiguousarray(mesh.positions, dtype=np.float32),
        np.ascontiguousarray(mesh.normals, dtype=np.float32),
        np.ascontiguousarray(mesh.uvs, dtype=np.float32),
        np.ascontiguousarray(mesh.indices, dtype=np.uint32),
        ratio,
        max_error,
    )
    return MeshLod(MeshData(mesh.positions, mesh.normals, mesh.uvs, indices), float(error))
