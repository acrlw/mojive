"""SEGMENT and IDCOLOR presentation for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ...backend import DebugView
from ..programs import load_wgsl
from ..timing import TimestampWriter

_MODE = {DebugView.SEGMENT: 1, DebugView.IDCOLOR: 2}

_PRESENT_DTYPE = np.dtype([("params", "(4,)u4")])  # x: mode, y: selected id


class PresentPass:
    name = "present"

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        module = device.create_shader_module(code=load_wgsl("present.wgsl"))
        self._layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "uint", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self._pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            vertex={"module": module, "entry_point": "vs_present", "buffers": []},
            fragment={
                "module": module,
                "entry_point": "fs_present",
                "targets": [{"format": "rgba8unorm"}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
            multisample={"count": 1},
        )
        self._block = np.zeros((), _PRESENT_DTYPE)
        self._uniforms = device.create_buffer(
            size=_PRESENT_DTYPE.itemsize, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def execute(
        self,
        encoder: wgpu.GPUCommandEncoder,
        color: wgpu.GPUTexture,
        ids: wgpu.GPUTexture,
        debug_view: DebugView,
        selected_id: int,
        timestamp: TimestampWriter | None = None,
    ) -> int:
        """Rewrite ``color`` with the pseudocolor view; returns the draw-call count."""
        mode = _MODE.get(debug_view)
        if mode is None:
            return 0
        block = self._block
        block["params"][:] = (mode, int(selected_id), 0, 0)
        self._device.queue.write_buffer(self._uniforms, 0, block.tobytes())
        group = self._device.create_bind_group(
            layout=self._layout,
            entries=[
                {"binding": 0, "resource": ids.create_view()},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._uniforms,
                        "offset": 0,
                        "size": _PRESENT_DTYPE.itemsize,
                    },
                },
            ],
        )
        present_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": color.create_view(),
                    # Every pixel is overwritten by the fullscreen draw.
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
            timestamp_writes=timestamp("present") if timestamp is not None else None,
        )
        present_pass.set_pipeline(self._pipeline)
        present_pass.set_bind_group(0, group)
        present_pass.draw(3)
        present_pass.end()
        return 1

    def release(self) -> None:
        self._uniforms.destroy()
