"""GPU box filtering for power-of-two texture mip chains."""

from __future__ import annotations

import wgpu

from .programs import load_wgsl


class MipmapGenerator:
    """Reuse format pipelines; each pass reads and writes separate mip subresources."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._module = device.create_shader_module(code=load_wgsl("mipmap.wgsl"))
        self._layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        self._pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self._layout])
        self._sampler = device.create_sampler(mag_filter="linear", min_filter="linear")
        self._pipelines: dict[str, wgpu.GPURenderPipeline] = {}

    def generate(self, texture: wgpu.GPUTexture) -> None:
        """Generate all levels and cube faces, preserving linear-light RGB and alpha."""
        if texture.mip_level_count <= 1:
            return
        pipeline = self._pipelines.get(texture.format)
        if pipeline is None:
            pipeline = self._device.create_render_pipeline(
                layout=self._pipeline_layout,
                vertex={"module": self._module, "entry_point": "vs_mipmap", "buffers": []},
                fragment={
                    "module": self._module,
                    "entry_point": "fs_mipmap",
                    "targets": [{"format": texture.format}],
                },
                primitive={"topology": "triangle-list", "cull_mode": "none"},
            )
            self._pipelines[texture.format] = pipeline
        encoder = self._device.create_command_encoder(label="texture mipmaps")
        for layer in range(texture.depth_or_array_layers):
            source = texture.create_view(
                dimension="2d",
                base_array_layer=layer,
                array_layer_count=1,
                base_mip_level=0,
                mip_level_count=1,
            )
            for level in range(1, texture.mip_level_count):
                destination = texture.create_view(
                    dimension="2d",
                    base_array_layer=layer,
                    array_layer_count=1,
                    base_mip_level=level,
                    mip_level_count=1,
                )
                group = self._device.create_bind_group(
                    layout=self._layout,
                    entries=[
                        {"binding": 0, "resource": source},
                        {"binding": 1, "resource": self._sampler},
                    ],
                )
                render = encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": destination,
                            "load_op": "clear",
                            "store_op": "store",
                            "clear_value": (0, 0, 0, 0),
                        }
                    ]
                )
                render.set_pipeline(pipeline)
                render.set_bind_group(0, group)
                render.draw(3)
                render.end()
                source = destination
        self._device.queue.submit([encoder.finish()])
