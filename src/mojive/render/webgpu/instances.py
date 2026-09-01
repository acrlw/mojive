"""GPU instance storage and WGSL-compatible record packing."""

from __future__ import annotations

import numpy as np
import wgpu

from ..scene import RenderScene

INSTANCE_DTYPE = np.dtype(
    [
        ("model", "(4,4)f4"),  # column-major upload, like InstanceStore.pack
        ("color", "(4,)f4"),
        ("material", "(4,)f4"),
        ("texcoef", "(4,)f4"),
        ("cubecoef", "(4,)f4"),
        ("object_id", "u4"),
        # The WGSL struct ends at 132 and its array stride rounds up to 144.
        ("pad", "(3,)f4"),
    ]
)
INSTANCE_STRIDE = INSTANCE_DTYPE.itemsize
assert INSTANCE_STRIDE == 144


class InstanceStore:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self.buffer: wgpu.GPUBuffer | None = None
        self.capacity = 0
        self.count = 0
        self._staging = np.zeros(0, INSTANCE_DTYPE)

    def _ensure_capacity(self, count: int) -> None:
        if count <= self.capacity and self.buffer is not None:
            return
        new_cap = max(count, self.capacity * 2, 64)
        if self.buffer is not None:
            self.buffer.destroy()
        self.buffer = self._device.create_buffer(
            size=new_cap * INSTANCE_STRIDE,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.capacity = new_cap
        self._staging = np.zeros(new_cap, INSTANCE_DTYPE)

    def upload(self, scene: RenderScene) -> None:
        n = scene.count
        self._ensure_capacity(max(n, 1))
        self.count = n
        if n == 0:
            return
        data = self._staging[:n]
        data["model"] = scene.transforms.transpose(0, 2, 1)
        data["color"] = scene.colors
        data["material"] = scene.material
        data["texcoef"] = scene.tex_coef
        data["cubecoef"] = scene.cube_coef
        data["object_id"] = scene.object_id
        self._device.queue.write_buffer(self.buffer, 0, data)

    def release(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
        self.capacity = 0
        self.count = 0
        self._staging = np.zeros(0, INSTANCE_DTYPE)
