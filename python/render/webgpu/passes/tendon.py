"""Spatial tendon and actuator paths rendered as 3D capsules."""

from __future__ import annotations

import wgpu

from ...tendon import TendonScene
from ..instances import InstanceStore
from ..lighting import LIGHTS_BYTES
from ..targets import FRAME_BYTES


class TendonPass(TendonScene):
    name = "tendon"

    def __init__(self, device: wgpu.GPUDevice) -> None:
        super().__init__()
        self._device = device
        self._store = InstanceStore(device)
        self._group0: wgpu.GPUBindGroup | None = None
        self._group0_key: tuple | None = None

    def bind_group0(
        self,
        layout: wgpu.GPUBindGroupLayout,
        frame_buffer: wgpu.GPUBuffer,
        lights_buffer: wgpu.GPUBuffer,
    ) -> wgpu.GPUBindGroup | None:
        """Upload the capsule instances and bind them as scene group0.

        Returns None when there is nothing to draw. The binding persists until
        an instance stream is reallocated on growth.
        """
        if not self._count:
            return None
        self._store.upload(self._scene)
        pose, visual, identity = self._store.bindings()
        key = (
            frame_buffer,
            pose[0],
            pose[1],
            visual[0],
            visual[1],
            identity[0],
            identity[1],
            lights_buffer,
        )
        if self._group0 is not None and self._group0_key == key:
            return self._group0
        self._group0 = self._device.create_bind_group(
            layout=layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": frame_buffer, "offset": 0, "size": FRAME_BYTES},
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": pose[0],
                        "offset": 0,
                        "size": pose[1],
                    },
                },
                {
                    "binding": 2,
                    "resource": {"buffer": visual[0], "offset": 0, "size": visual[1]},
                },
                {
                    "binding": 3,
                    "resource": {"buffer": identity[0], "offset": 0, "size": identity[1]},
                },
                {
                    "binding": 4,
                    "resource": {"buffer": lights_buffer, "offset": 0, "size": LIGHTS_BYTES},
                },
            ],
        )
        self._group0_key = key
        return self._group0

    def release(self) -> None:
        self._group0 = None
        self._group0_key = None
        self._store.release()
