"""Antialiased selection outlines for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ...backend import RenderFlag
from ...scene import RenderScene
from ..blend import ALPHA_BLEND
from ..instances import IDENTITY_STRIDE, POSE_STRIDE
from ..programs import load_wgsl
from ..timing import TimestampWriter

OUTLINE_RADIUS = 3
OUTLINE_COLOR = (1.0, 0.63, 0.20, 1.0)

_MASK_DTYPE = np.dtype(
    [
        ("view_proj", "(4,4)f4"),
        ("params", "(4,)u4"),  # x: selected id
    ]
)

_COMPOSITE_DTYPE = np.dtype(
    [
        ("color", "(4,)f4"),
        ("size", "(4,)u4"),  # xy: target size
    ]
)


class OutlinePass:
    name = "outline"

    def __init__(self, device: wgpu.GPUDevice, samples: int) -> None:
        self._device = device
        module = device.create_shader_module(code=load_wgsl("outline.wgsl"))

        self._mask_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                },
            ]
        )
        self._mask_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._mask_layout]),
            vertex={
                "module": module,
                "entry_point": "vs_outline_mask",
                # Only position is read; the standard scene vertex layout.
                "buffers": [
                    {
                        "array_stride": 32,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": module,
                "entry_point": "fs_outline_mask",
                "targets": [{"format": "r8unorm"}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "back"},
            multisample={"count": 1},
        )

        self._composite_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self._composite_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._composite_layout]),
            vertex={"module": module, "entry_point": "vs_outline", "buffers": []},
            fragment={
                "module": module,
                "entry_point": "fs_outline",
                "targets": [{"format": "rgba8unorm", "blend": ALPHA_BLEND}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
            # state_overlay(depth_test=False): blended overlay, depth untouched.
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": False,
                "depth_compare": "always",
            },
            multisample={"count": samples},
        )

        self._mask_block = np.zeros((), _MASK_DTYPE)
        self._mask_uniforms = device.create_buffer(
            size=_MASK_DTYPE.itemsize, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._composite_block = np.zeros((), _COMPOSITE_DTYPE)
        self._composite_uniforms = device.create_buffer(
            size=_COMPOSITE_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._mask: wgpu.GPUTexture | None = None
        self._mask_view: wgpu.GPUTextureView | None = None
        self._mask_size: tuple[int, int] = (0, 0)
        self._mask_group: wgpu.GPUBindGroup | None = None
        self._mask_group_key: tuple | None = None
        self._composite_group: wgpu.GPUBindGroup | None = None

        self._buckets: tuple[int, ...] = ()
        self._sel_id = -1
        self._sel_ids: object = None
        self.color: tuple[float, float, float, float] = OUTLINE_COLOR

    def prepare(
        self, scene: RenderScene, selected_id: int, flags: dict[RenderFlag, bool]
    ) -> tuple[int, ...]:
        """Buckets to draw into the mask; empty when the pass is inactive."""
        if not flags.get(RenderFlag.OUTLINE, True):
            return ()
        sel = int(selected_id)
        if sel == 0:
            return ()
        ids = scene.object_id
        if sel == self._sel_id and ids is self._sel_ids:
            return self._buckets
        self._buckets = tuple(
            b
            for b, (start, stop) in enumerate(scene.bucket_ranges)
            if stop > start and bool(np.any(ids[start:stop] == sel))
        )
        self._sel_id = sel
        self._sel_ids = ids
        return self._buckets

    def _ensure_mask(self, width: int, height: int) -> wgpu.GPUTexture:
        if self._mask is not None and self._mask_size == (width, height):
            return self._mask
        if self._mask is not None:
            self._mask.destroy()
        self._mask = self._device.create_texture(
            size=(width, height, 1),
            format="r8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._mask_view = self._mask.create_view()
        self._mask_size = (width, height)
        self._mask_group = None
        self._mask_group_key = None
        self._composite_group = None
        return self._mask

    def render_mask(
        self,
        encoder: wgpu.GPUCommandEncoder,
        scene: RenderScene,
        meshes,
        instances,
        view_proj: np.ndarray,
        width: int,
        height: int,
        timestamp: TimestampWriter | None = None,
    ) -> int:
        """Rasterize the selected silhouette; returns the draw-call count."""
        self._ensure_mask(width, height)
        block = self._mask_block
        block["view_proj"][:] = np.asarray(view_proj, np.float32).T
        block["params"][:] = (self._sel_id, 0, 0, 0)
        self._device.queue.write_buffer(self._mask_uniforms, 0, block.tobytes())

        key = (
            instances.pose_buffer,
            instances.capacity * POSE_STRIDE,
            instances.identity_buffer,
            instances.capacity * IDENTITY_STRIDE,
        )
        if self._mask_group is None or self._mask_group_key != key:
            self._mask_group = self._device.create_bind_group(
                layout=self._mask_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._mask_uniforms,
                            "offset": 0,
                            "size": _MASK_DTYPE.itemsize,
                        },
                    },
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": instances.pose_buffer,
                            "offset": 0,
                            "size": instances.capacity * POSE_STRIDE,
                        },
                    },
                    {
                        "binding": 4,
                        "resource": {
                            "buffer": instances.identity_buffer,
                            "offset": 0,
                            "size": instances.capacity * IDENTITY_STRIDE,
                        },
                    },
                ],
            )
            self._mask_group_key = key
        assert self._mask_view is not None
        mask_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self._mask_view,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
            timestamp_writes=timestamp("outline") if timestamp is not None else None,
        )
        mask_pass.set_pipeline(self._mask_pipeline)
        mask_pass.set_bind_group(0, self._mask_group)
        calls = 0
        for b in self._buckets:
            start, stop = scene.bucket_ranges[b]
            if stop <= start:
                continue
            mesh = meshes.get(scene.bucket_keys[b][0])
            if mesh is None:
                continue
            mask_pass.set_vertex_buffer(0, mesh.vbo)
            mask_pass.set_index_buffer(mesh.ibo, "uint32")
            mask_pass.draw_indexed(mesh.index_count, stop - start, 0, 0, start)
            calls += 1
        mask_pass.end()
        return calls

    def composite(self, pass_encoder: wgpu.GPURenderPassEncoder, width: int, height: int) -> int:
        """Dilate the mask into the main color pass; returns the draw-call count."""
        assert self._mask is not None
        block = self._composite_block
        block["color"][:] = self.color
        block["size"][:] = (width, height, 0, 0)
        self._device.queue.write_buffer(self._composite_uniforms, 0, block.tobytes())
        if self._composite_group is None:
            assert self._mask_view is not None
            self._composite_group = self._device.create_bind_group(
                layout=self._composite_layout,
                entries=[
                    {"binding": 2, "resource": self._mask_view},
                    {
                        "binding": 3,
                        "resource": {
                            "buffer": self._composite_uniforms,
                            "offset": 0,
                            "size": _COMPOSITE_DTYPE.itemsize,
                        },
                    },
                ],
            )
        pass_encoder.set_pipeline(self._composite_pipeline)
        pass_encoder.set_bind_group(0, self._composite_group)
        pass_encoder.draw(3)
        return 1

    def release(self) -> None:
        if self._mask is not None:
            self._mask.destroy()
            self._mask = None
        self._mask_view = None
        self._mask_size = (0, 0)
        self._mask_group = None
        self._mask_group_key = None
        self._composite_group = None
        self._mask_uniforms.destroy()
        self._composite_uniforms.destroy()
