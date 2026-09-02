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
        # WGSL aligns vec2<i32> to 8 bytes, so keep one word between it and ID.
        ("identity_pad", "u4"),
        ("segmentation", "(2,)i4"),
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
        self._last_revisions: tuple[int, int, int, int] | None = None
        self.uploaded_bytes = 0

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
        self._last_revisions = None

    def invalidate_upload(self) -> None:
        """Force the next upload after a pass-owned instance variant changes."""

        self._last_revisions = None

    def upload(self, scene: RenderScene) -> None:
        n = scene.count
        self._ensure_capacity(max(n, 1))
        self.count = n
        revisions = (
            scene.structure_revision,
            scene.pose_revision,
            scene.visual_revision,
            scene.identity_revision,
        )
        if all(revisions) and revisions == self._last_revisions:
            self.uploaded_bytes = 0
            return
        if n == 0:
            self.uploaded_bytes = 0
            self._last_revisions = revisions
            return
        data = self._staging[:n]
        data["model"] = scene.transforms.transpose(0, 2, 1)
        data["color"] = scene.colors
        data["material"] = scene.material
        data["texcoef"] = scene.tex_coef
        data["cubecoef"] = scene.cube_coef
        data["object_id"] = scene.object_id
        if scene.segmentation.shape == (n, 2):
            data["segmentation"] = scene.segmentation
        else:
            data["segmentation"] = -1
        self._device.queue.write_buffer(self.buffer, 0, data)
        self.uploaded_bytes = data.nbytes
        self._last_revisions = revisions

    def release(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
        self.capacity = 0
        self.count = 0
        self._staging = np.zeros(0, INSTANCE_DTYPE)
        self._last_revisions = None
        self.uploaded_bytes = 0
