"""GPU mesh resources and dynamic vertex updates for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ...types import MeshData, MeshKey, MeshUpdate

VERTEX_STRIDE = 8  # position(3) + normal(3) + uv(2), float32
WIRE_STRIDE = VERTEX_STRIDE + 3  # + barycentric(3), float32


class GpuMesh:
    __slots__ = (
        "_device",
        "_indices",
        "_vertices",
        "_wire_count",
        "_wire_stream",
        "_wire_vbo",
        "ibo",
        "index_count",
        "triangle_count",
        "vbo",
    )

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
        self._indices = np.ascontiguousarray(mesh.indices, np.uint32)
        self.ibo = device.create_buffer_with_data(
            data=self._indices.tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )
        self.index_count = len(self._indices)
        self.triangle_count = self.index_count // 3
        self._wire_vbo: wgpu.GPUBuffer | None = None
        self._wire_stream: np.ndarray | None = None
        self._wire_count = 0

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
        if self._wire_stream is not None:
            self._refresh_wire()

    def wireframe(self) -> tuple[wgpu.GPUBuffer, int]:
        """Non-indexed vertex stream with per-vertex barycentric coordinates.

        WebGPU has no geometry stage, so opengl's wireframe.geom barycentric
        injection becomes a lazily-built expansion: every index-triangle is
        unrolled to three vertices carrying (1,0,0)/(0,1,0)/(0,0,1).  Built on
        first request and cached for the mesh's lifetime; deformable updates
        refresh the expanded positions/normals in place.
        """
        if self._wire_vbo is None:
            count = self.index_count
            stream = np.empty((count, WIRE_STRIDE), np.float32)
            bary = np.zeros((3, 3), np.float32)
            np.fill_diagonal(bary, 1.0)
            stream[:, VERTEX_STRIDE:] = np.tile(bary, (self.triangle_count, 1))
            stream[:, :VERTEX_STRIDE] = self._vertices[self._indices]
            self._wire_stream = stream
            self._wire_count = count
            self._wire_vbo = self._device.create_buffer_with_data(
                data=stream.tobytes(),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
        return self._wire_vbo, self._wire_count

    def _refresh_wire(self) -> None:
        assert self._wire_stream is not None and self._wire_vbo is not None
        self._wire_stream[:, :VERTEX_STRIDE] = self._vertices[self._indices]
        self._device.queue.write_buffer(self._wire_vbo, 0, self._wire_stream)

    def release(self) -> None:
        self.vbo.destroy()
        self.ibo.destroy()
        if self._wire_vbo is not None:
            self._wire_vbo.destroy()
            self._wire_vbo = None
        self._wire_stream = None


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
