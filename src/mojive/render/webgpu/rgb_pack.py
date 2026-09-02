"""Persistent GPU-side packing for WebGPU RGB capture."""

from __future__ import annotations

from concurrent.futures import Future
from functools import partial

import numpy as np
import wgpu

from .programs import load_wgsl
from .readback import WgpuReadbackQueue, decode_packed_rgb

_WORKGROUP_SIZE = 64
_MAX_DISPATCH_X = 65_535


class WgpuRgbPacker:
    """Persistently pack RGBA8 render targets into transfer-efficient RGB bytes."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        module = device.create_shader_module(code=load_wgsl("rgb_pack.wgsl"))
        self._layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ]
        )
        self._pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            compute={"module": module, "entry_point": "cs_rgb_pack"},
        )
        self._buffer: wgpu.GPUBuffer | None = None
        self._group: wgpu.GPUBindGroup | None = None
        self._key: tuple[wgpu.GPUTextureView, int, int] | None = None
        self._size = 0
        self._invocations = 0
        limits = device.limits
        self._max_size = min(
            int(limits["max-storage-buffer-binding-size"]),
            int(limits["max-buffer-size"]),
        )

    @staticmethod
    def packed_size(width: int, height: int) -> int:
        """Return the four-pixel-aligned RGB storage size."""

        return ((int(width) * int(height) + 3) // 4) * 12

    def supports(self, width: int, height: int) -> bool:
        """Whether the device can bind the target-sized packing buffer."""

        return self.packed_size(width, height) <= self._max_size

    def _ensure(self, view: wgpu.GPUTextureView, width: int, height: int) -> None:
        key = (view, int(width), int(height))
        if self._key == key:
            return
        self.invalidate()
        pixel_count = int(width) * int(height)
        self._invocations = (pixel_count + 3) // 4
        self._size = self.packed_size(width, height)
        self._buffer = self._device.create_buffer(
            size=self._size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._group = self._device.create_bind_group(
            layout=self._layout,
            entries=[
                {"binding": 0, "resource": view},
                {
                    "binding": 1,
                    "resource": {"buffer": self._buffer, "offset": 0, "size": self._size},
                },
            ],
        )
        self._key = key

    def _encode(self, encoder: wgpu.GPUCommandEncoder) -> None:
        assert self._group is not None
        workgroups = (self._invocations + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE
        dispatch_x = min(workgroups, _MAX_DISPATCH_X)
        dispatch_y = (workgroups + dispatch_x - 1) // dispatch_x
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self._pipeline)
        compute.set_bind_group(0, self._group)
        compute.dispatch_workgroups(dispatch_x, dispatch_y)
        compute.end()

    def read(
        self,
        view: wgpu.GPUTextureView,
        *,
        width: int,
        height: int,
        flip: bool,
        out: np.ndarray | None,
    ) -> np.ndarray:
        """Pack and synchronously map one render target."""

        self._ensure(view, width, height)
        assert self._buffer is not None
        encoder = self._device.create_command_encoder()
        self._encode(encoder)
        self._device.queue.submit([encoder.finish()])
        data = self._device.queue.read_buffer(self._buffer, 0, self._size)
        return decode_packed_rgb(data, width=width, height=height, flip=flip, out=out)

    def read_async(
        self,
        queue: WgpuReadbackQueue,
        view: wgpu.GPUTextureView,
        *,
        width: int,
        height: int,
        flip: bool,
        out: np.ndarray | None,
    ) -> Future[np.ndarray]:
        """Pack and copy one target into the queue's next staging slot."""

        self._ensure(view, width, height)
        source = self._buffer
        assert source is not None

        def encode(encoder: wgpu.GPUCommandEncoder, destination: wgpu.GPUBuffer) -> None:
            self._encode(encoder)
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, self._size)

        decode = partial(
            decode_packed_rgb,
            width=width,
            height=height,
            flip=flip,
            out=out,
        )
        return queue.enqueue_copy(self._size, encode, decode)

    def invalidate(self) -> None:
        """Drop target-sized resources while retaining the compiled pipeline."""

        self._group = None
        self._key = None
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        self._size = 0
        self._invocations = 0
