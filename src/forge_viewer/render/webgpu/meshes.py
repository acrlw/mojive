"""GPU mesh storage for the webgpu backend.

Mirrors ``render.forge.resources.MeshStore``: builtin and source meshes keyed by
``MeshKey``, with in-place vertex updates for dynamic (deformable) meshes.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...types import MeshData, MeshKey, MeshUpdate

VERTEX_STRIDE = 8  # position(3) + normal(3) + uv(2), float32


class GpuMesh:
    __slots__ = ("_device", "_vertices", "ibo", "index_count", "triangle_count", "vbo")

    def __init__(self, device: wgpu.GPUDevice, mesh: MeshData) -> None:
        self._device = device
        n = len(mesh.positions)
        self._vertices = np.empty((n, VERTEX_STRIDE), np.float32)
        self._vertices[:, 0:3] = mesh.positions
        self._vertices[:, 3:6] = mesh.normals
        self._vertices[:, 6:8] = mesh.uvs
        self.vbo = device.create_buffer_with_data(
            data=self._vertices.tobytes(),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.ibo = device.create_buffer_with_data(
            data=np.ascontiguousarray(mesh.indices, np.uint32).tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )
        self.index_count = len(mesh.indices)
        self.triangle_count = self.index_count // 3

    def update(self, positions: np.ndarray, normals: np.ndarray) -> None:
        shape = self._vertices[:, :3].shape
        if positions.shape != shape or normals.shape != shape:
            raise ValueError(
                f"dynamic mesh vertex shape changed: expected {shape}, "
                f"got {positions.shape} / {normals.shape}"
            )
        np.copyto(self._vertices[:, :3], positions, casting="unsafe")
        np.copyto(self._vertices[:, 3:6], normals, casting="unsafe")
        self._device.queue.write_buffer(self.vbo, 0, self._vertices)

    def release(self) -> None:
        self.vbo.destroy()
        self.ibo.destroy()


class MeshStore:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._meshes: dict[MeshKey, GpuMesh] = {}
        self._sources: dict[MeshKey, MeshData] = {}

    def sync(self, meshes: dict[MeshKey, MeshData]) -> None:
        for key, data in meshes.items():
            if key not in self._meshes or self._sources.get(key) is not data:
                old = self._meshes.pop(key, None)
                if old is not None:
                    old.release()
                self._meshes[key] = GpuMesh(self._device, data)
                self._sources[key] = data
        for key in [k for k in self._meshes if k not in meshes]:
            self._meshes.pop(key).release()
            self._sources.pop(key, None)

    def update(self, meshes: dict[MeshKey, MeshUpdate] | None) -> None:
        if not meshes:
            return
        for key, data in meshes.items():
            mesh = self._meshes.get(key)
            if mesh is not None:
                mesh.update(data.positions, data.normals)

    def get(self, key: MeshKey) -> GpuMesh | None:
        return self._meshes.get(key)

    def triangle_counts(self) -> dict[MeshKey, int]:
        return {k: m.triangle_count for k, m in self._meshes.items()}

    def release(self) -> None:
        for mesh in self._meshes.values():
            mesh.release()
        self._meshes.clear()
        self._sources.clear()
