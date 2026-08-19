"""Per-instance storage buffer for the webgpu backend.

Same logical record as ``render.forge.instances`` (transform, color, material,
texcoef, object id), padded to a 128-byte stride for WGSL struct alignment.
Instances live in a storage buffer indexed by ``instance_index``; bucket draws
pass ``first_instance`` so no per-bucket offsets or VAOs are needed.
"""

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
        ("object_id", "u4"),
        # The WGSL struct has no pad field: it ends at 116 and its array
        # stride rounds up to 128.  Keep this pad so numpy matches.
        ("pad", "(3,)f4"),
    ]
)
INSTANCE_STRIDE = INSTANCE_DTYPE.itemsize
assert INSTANCE_STRIDE == 128


class InstanceStore:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self.buffer: wgpu.GPUBuffer | None = None
        self.capacity = 0
        self.count = 0

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

    def upload(self, scene: RenderScene) -> None:
        n = scene.count
        self._ensure_capacity(max(n, 1))
        self.count = n
        if n == 0:
            return
        data = np.zeros(n, INSTANCE_DTYPE)
        data["model"] = scene.transforms.transpose(0, 2, 1)
        data["color"] = scene.colors
        data["material"] = scene.material
        data["texcoef"] = scene.tex_coef
        data["object_id"] = scene.object_id
        self._device.queue.write_buffer(self.buffer, 0, data)

    def release(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
        self.capacity = 0
        self.count = 0
