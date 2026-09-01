"""Spatial tendon and actuator paths rendered as 3D capsules."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import wgpu

from ....types import DEFAULT_MATERIAL, MeshKey, MeshShape
from ...scene import RenderScene
from ..instances import INSTANCE_STRIDE, InstanceStore
from ..lighting import LIGHTS_BYTES
from ..targets import FRAME_BYTES

_SHAFT = MeshKey(MeshShape.CAPSULE_SHAFT)
_CAP = MeshKey(MeshShape.CAPSULE_CAP)


class TendonPass:
    name = "tendon"

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._scene = RenderScene()
        self._count = 0
        self._store = InstanceStore(device)
        self._material_ids = np.zeros(0, np.int32)
        self._transparent = np.zeros(0, bool)

    @property
    def capsule_count(self) -> int:
        return self._count

    @property
    def scene(self) -> RenderScene:
        return self._scene

    def update(
        self,
        segments,
        widths,
        colors,
        materials,
        material_ids,
        transparent,
        material_table,
    ) -> None:
        n = len(segments)
        material_ids = np.asarray(material_ids, np.int32)
        transparent = np.asarray(transparent, bool)
        if (
            n != self._count
            or material_table is not self._scene.materials
            or not np.array_equal(material_ids, self._material_ids)
            or not np.array_equal(transparent, self._transparent)
        ):
            self._resize(n, material_ids, transparent, material_table)
        if not n:
            return

        segments = np.asarray(segments, np.float32)
        widths = np.asarray(widths, np.float32)
        delta = segments[:, 1] - segments[:, 0]
        length = np.linalg.norm(delta, axis=1)
        z = delta / np.maximum(length[:, None], 1e-12)
        helper = np.zeros_like(z)
        helper[:, 2] = 1.0
        steep = np.abs(z[:, 2]) > 0.9
        helper[steep] = (0.0, 1.0, 0.0)
        x = np.cross(helper, z)
        x /= np.maximum(np.linalg.norm(x, axis=1)[:, None], 1e-12)
        y = np.cross(z, x)

        transforms = self._scene.transforms
        transforms.fill(0.0)
        transforms[:, 3, 3] = 1.0
        half = 0.5 * length

        transforms[:n, :3, 0] = x * widths[:, None]
        transforms[:n, :3, 1] = y * widths[:, None]
        transforms[:n, :3, 2] = z * half[:, None]
        transforms[:n, :3, 3] = 0.5 * (segments[:, 0] + segments[:, 1])

        transforms[n : 2 * n, :3, 0] = x * widths[:, None]
        transforms[n : 2 * n, :3, 1] = y * widths[:, None]
        transforms[n : 2 * n, :3, 2] = z * widths[:, None]
        transforms[n : 2 * n, :3, 3] = segments[:, 1]

        transforms[2 * n :, :3, 0] = x * widths[:, None]
        transforms[2 * n :, :3, 1] = -y * widths[:, None]
        transforms[2 * n :, :3, 2] = -z * widths[:, None]
        transforms[2 * n :, :3, 3] = segments[:, 0]

        rgba = np.asarray(colors, np.float32)
        for start in (0, n, 2 * n):
            dst = self._scene.colors[start : start + n]
            np.power(np.clip(rgba[:, :3], 0.0, 1.0), 2.2, out=dst[:, :3])
            dst[:, 3] = rgba[:, 3]
            self._scene.material[start : start + n] = materials
            self._scene.material[start : start + n, 3] = 0.0

    def clear(self) -> None:
        if self._count:
            self._resize(0, np.zeros(0, np.int32), np.zeros(0, bool), self._scene.materials)

    def _resize(self, n: int, material_ids, transparent, material_table) -> None:
        count = 3 * n
        scene = RenderScene(count=count)
        scene.transforms = np.zeros((count, 4, 4), np.float32)
        scene.colors = np.ones((count, 4), np.float32)
        scene.material = np.tile(np.array((0.0, 0.5, 0.5, 0.0), np.float32), (count, 1))
        scene.tex_coef = np.tile(np.array((1.0, 1.0, 0.0, 0.0), np.float32), (count, 1))
        scene.cube_coef = np.zeros((count, 4), np.float32)
        scene.object_id = np.zeros(count, np.uint32)
        scene.bucket = np.zeros(count, np.int32)
        scene.materials = material_table or (DEFAULT_MATERIAL,)
        boundaries = [0]
        for i in range(1, n):
            if material_ids[i] != material_ids[i - 1] or transparent[i] != transparent[i - 1]:
                boundaries.append(i)
        boundaries.append(n)
        runs = list(pairwise(boundaries)) if n else []
        keys = []
        ranges = []
        opaque = []
        translucent = []
        for mesh, offset in ((_SHAFT, 0), (_CAP, n), (_CAP, 2 * n)):
            for start, stop in runs:
                bucket = len(keys)
                matid = int(material_ids[start])
                keys.append((mesh, matid))
                ranges.append((offset + start, offset + stop))
                scene.bucket[offset + start : offset + stop] = bucket
                (translucent if transparent[start] else opaque).append(bucket)
                mat = scene.materials[matid]
                scene.tex_coef[offset + start : offset + stop, :2] = mat.tex_repeat
        scene.bucket_keys = tuple(keys)
        scene.bucket_ranges = tuple(ranges)
        scene.opaque_buckets = tuple(opaque)
        scene.transparent_buckets = tuple(translucent)
        self._scene = scene
        self._count = n
        self._material_ids = material_ids.copy()
        self._transparent = transparent.copy()

    def bind_group0(
        self,
        layout: wgpu.GPUBindGroupLayout,
        frame_buffer: wgpu.GPUBuffer,
        lights_buffer: wgpu.GPUBuffer,
    ) -> wgpu.GPUBindGroup | None:
        """Upload the capsule instances and bind them as scene group0.

        Returns None when there is nothing to draw.  A fresh group every
        frame, like the backend's main group0: the storage buffer is
        reallocated on growth, so no bind group survives across frames.
        """
        if not self._count:
            return None
        self._store.upload(self._scene)
        return self._device.create_bind_group(
            layout=layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": frame_buffer, "offset": 0, "size": FRAME_BYTES},
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._store.buffer,
                        "offset": 0,
                        "size": self._store.capacity * INSTANCE_STRIDE,
                    },
                },
                {
                    "binding": 2,
                    "resource": {"buffer": lights_buffer, "offset": 0, "size": LIGHTS_BYTES},
                },
            ],
        )

    def release(self) -> None:
        self._store.release()
