"""Environment skybox and horizon haze for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ....types import CameraView, ShadingModel
from ...backend import DebugView, RenderFlag
from ...scene import RenderScene
from ..programs import load_wgsl
from ..textures import TextureStore

EXPOSURE = 1.0  # mirrors render.forge.color.EXPOSURE

SKYBOX_DTYPE = np.dtype(
    [
        ("inv_view_proj", "(4,4)f4"),
        ("view_proj", "(4,4)f4"),
        ("eye_distance", "(4,)f4"),  # xyz camera eye, w classic cylinder distance
        ("params", "(4,)f4"),  # exposure, tonemap on, MuJoCo classic
    ]
)

HAZE_DTYPE = np.dtype(
    [
        ("view_proj", "(4,4)f4"),
        ("eye", "(4,)f4"),
        ("basis_x", "(4,)f4"),
        ("basis_y", "(4,)f4"),
        ("normal", "(4,)f4"),
        ("geometry", "(4,)f4"),  # skybox distance, elevation, radius, transition height
        ("color", "(4,)f4"),  # raw sRGB
        ("params", "(4,)f4"),  # exposure, tonemap on, MuJoCo classic
    ]
)

_HAZE_VERTEX_LAYOUT = {
    "array_stride": 12,
    "step_mode": "vertex",
    "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0}],
}
_SKYBOX_VERTEX_LAYOUT = _HAZE_VERTEX_LAYOUT

# moderngl blend_func(SRC_ALPHA, ONE_MINUS_SRC_ALPHA) applied to color and alpha.
_ALPHA_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
    "alpha": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
}


def _make_haze_pipeline(
    device: wgpu.GPUDevice,
    module: wgpu.GPUShaderModule,
    layout: wgpu.GPUPipelineLayout,
    samples: int,
    *,
    depth_write: bool,
) -> wgpu.GPURenderPipeline:
    return device.create_render_pipeline(
        layout=layout,
        vertex={
            "module": module,
            "entry_point": "vs_haze",
            "buffers": [_HAZE_VERTEX_LAYOUT],
        },
        fragment={
            "module": module,
            "entry_point": "fs_haze",
            "targets": [{"format": "rgba8unorm", "blend": _ALPHA_BLEND}],
        },
        primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
        depth_stencil={
            "format": "depth24plus",
            "depth_write_enabled": depth_write,
            "depth_compare": "less-equal",
        },
        multisample={"count": samples},
    )


class SkyboxPass:
    name = "skybox"

    def __init__(self, device: wgpu.GPUDevice, samples: int) -> None:
        self._device = device
        module_skybox = device.create_shader_module(code=load_wgsl("common.wgsl", "skybox.wgsl"))
        module_haze = device.create_shader_module(code=load_wgsl("common.wgsl", "haze.wgsl"))

        self._skybox_uniform_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        self._skybox_texture_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "cube"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        skybox_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._skybox_uniform_layout, self._skybox_texture_layout]
        )
        self._skybox_pipeline = device.create_render_pipeline(
            layout=skybox_pipeline_layout,
            vertex={"module": module_skybox, "entry_point": "vs_skybox", "buffers": []},
            fragment={
                "module": module_skybox,
                "entry_point": "fs_skybox",
                "targets": [{"format": "rgba8unorm"}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": False,
                "depth_compare": "less-equal",
            },
            multisample={"count": samples},
        )
        self._classic_skybox_pipeline = device.create_render_pipeline(
            layout=skybox_pipeline_layout,
            vertex={
                "module": module_skybox,
                "entry_point": "vs_classic_skybox",
                "buffers": [_SKYBOX_VERTEX_LAYOUT],
            },
            fragment={
                "module": module_skybox,
                "entry_point": "fs_skybox",
                "targets": [{"format": "rgba8unorm"}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less-equal",
            },
            multisample={"count": samples},
        )

        self._haze_uniform_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        haze_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._haze_uniform_layout]
        )
        self._haze_pipeline = _make_haze_pipeline(
            device, module_haze, haze_pipeline_layout, samples, depth_write=False
        )
        self._classic_haze_pipeline = _make_haze_pipeline(
            device, module_haze, haze_pipeline_layout, samples, depth_write=True
        )

        self._skybox_block = np.zeros((), SKYBOX_DTYPE)
        self._skybox_uniforms = device.create_buffer(
            size=SKYBOX_DTYPE.itemsize, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._skybox_uniform_group = device.create_bind_group(
            layout=self._skybox_uniform_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._skybox_uniforms,
                        "offset": 0,
                        "size": SKYBOX_DTYPE.itemsize,
                    },
                }
            ],
        )
        self._skybox_groups: dict[int, wgpu.GPUBindGroup] = {}
        self._classic_slices = 64
        classic_vertices = self._classic_skybox_vertices(self._classic_slices)
        self._classic_buffer = device.create_buffer_with_data(
            data=classic_vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self._classic_vertex_count = len(classic_vertices)

        self._haze_block = np.zeros((), HAZE_DTYPE)
        self._haze_uniforms = device.create_buffer(
            size=HAZE_DTYPE.itemsize, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._haze_uniform_group = device.create_bind_group(
            layout=self._haze_uniform_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._haze_uniforms,
                        "offset": 0,
                        "size": HAZE_DTYPE.itemsize,
                    },
                }
            ],
        )
        self._haze_slices = 64
        vertices = self._haze_vertices(self._haze_slices)
        self._haze_buffer = device.create_buffer_with_data(
            data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self._haze_vertex_count = len(vertices)

    @staticmethod
    def _classic_skybox_vertices(slices: int) -> np.ndarray:
        """Closed unit cylinder used by MuJoCo's classic skybox display list."""
        vertices: list[tuple[float, float, float]] = []
        for index in range(slices):
            angle0 = 2.0 * np.pi * index / slices
            angle1 = 2.0 * np.pi * (index + 1) / slices
            x0, y0 = np.cos(angle0), np.sin(angle0)
            x1, y1 = np.cos(angle1), np.sin(angle1)
            lower0 = (x0, y0, -1.0)
            lower1 = (x1, y1, -1.0)
            upper0 = (x0, y0, 1.0)
            upper1 = (x1, y1, 1.0)
            vertices.extend((lower0, lower1, upper1, lower0, upper1, upper0))
            vertices.extend(((0.0, 0.0, 1.0), upper0, upper1))
            vertices.extend(((0.0, 0.0, -1.0), lower1, lower0))
        return np.asarray(vertices, np.float32)

    @staticmethod
    def _haze_vertices(slices: int = 64) -> np.ndarray:
        """Two-layer ring band, identical to forge SkyboxPass._haze_vertices."""
        vertices: list[tuple[float, float, float]] = []
        for layer0, layer1 in ((0.0, 1.0), (1.0, 2.0)):
            for index in range(slices):
                angle0 = 2.0 * np.pi * index / slices
                angle1 = 2.0 * np.pi * (index + 1) / slices
                p00 = (np.cos(angle0), np.sin(angle0), layer0)
                p01 = (np.cos(angle1), np.sin(angle1), layer0)
                p11 = (np.cos(angle1), np.sin(angle1), layer1)
                p10 = (np.cos(angle0), np.sin(angle0), layer1)
                vertices.extend((p00, p01, p11, p00, p11, p10))
        return np.asarray(vertices, np.float32)

    def reset(self) -> None:
        """Drop cached texture bind groups (textures were re-synced)."""
        self._skybox_groups.clear()

    def _skybox_group(
        self, view: wgpu.GPUTextureView, sampler: wgpu.GPUSampler
    ) -> wgpu.GPUBindGroup:
        key = id(view)
        group = self._skybox_groups.get(key)
        if group is None:
            group = self._device.create_bind_group(
                layout=self._skybox_texture_layout,
                entries=[{"binding": 0, "resource": view}, {"binding": 1, "resource": sampler}],
            )
            self._skybox_groups[key] = group
        return group

    def render(
        self,
        pass_encoder: wgpu.GPURenderPassEncoder,
        *,
        textures: TextureStore,
        flags: dict[RenderFlag, bool],
        debug_view: DebugView,
        camera: CameraView,
        view_proj: np.ndarray,
        scene: RenderScene,
    ) -> int:
        """Draw skybox and horizon haze; returns the draw-call count."""
        view = textures.skybox
        if view is None or not flags.get(RenderFlag.SKYBOX, True):
            return 0
        if debug_view is not DebugView.SHADED:
            return 0

        tonemap = 1.0 if flags.get(RenderFlag.TONEMAP, True) else 0.0
        is_classic = scene.shading_model is ShadingModel.MUJOCO_CLASSIC
        classic = 1.0 if is_classic else 0.0
        block = self._skybox_block
        block["inv_view_proj"][:] = np.linalg.inv(view_proj.astype(np.float64)).T
        block["view_proj"][:] = np.asarray(view_proj, np.float32).T
        block["eye_distance"][:] = (*camera.eye, float(camera.far) * 0.70)
        block["params"][:] = (EXPOSURE, tonemap, classic, 0.0)
        self._device.queue.write_buffer(self._skybox_uniforms, 0, block.tobytes())

        pass_encoder.set_pipeline(
            self._classic_skybox_pipeline if is_classic else self._skybox_pipeline
        )
        pass_encoder.set_bind_group(0, self._skybox_uniform_group)
        pass_encoder.set_bind_group(1, self._skybox_group(view, textures.cube_sampler))
        if is_classic:
            self._ensure_classic_buffer(max(3, int(scene.lights.horizon_haze_slices)))
            pass_encoder.set_vertex_buffer(0, self._classic_buffer)
            pass_encoder.draw(self._classic_vertex_count)
        else:
            pass_encoder.draw(3)
        calls = 1

        if self._write_haze_uniforms(flags, tonemap, camera, view_proj, scene):
            pass_encoder.set_pipeline(
                self._classic_haze_pipeline if is_classic else self._haze_pipeline
            )
            pass_encoder.set_bind_group(0, self._haze_uniform_group)
            pass_encoder.set_vertex_buffer(0, self._haze_buffer)
            pass_encoder.draw(self._haze_vertex_count)
            calls += 1
        return calls

    def _write_haze_uniforms(
        self,
        flags: dict[RenderFlag, bool],
        tonemap: float,
        camera: CameraView,
        view_proj: np.ndarray,
        scene: RenderScene,
    ) -> bool:
        """Enable conditions and uniform packing of forge _prepare_horizon_haze."""
        lights = scene.lights
        if not (
            flags.get(RenderFlag.HAZE, False)
            and lights.horizon_haze
            and bool(scene.infinite_planes)
            and lights.haze_density > 0.0
        ):
            return False

        transform = scene.transforms[scene.infinite_planes[0]]
        normal = np.asarray(transform[:3, 2], np.float64)
        normal /= np.linalg.norm(normal)
        elevation = float(np.dot(np.asarray(camera.eye, np.float64) - transform[:3, 3], normal))
        if elevation < 0.0:
            return False

        radius = float(lights.haze_density)
        alpha = np.arctan2(1.0, radius)
        beta = 0.75 * np.pi - alpha
        transition = float(np.sqrt(0.5) * radius * np.sin(alpha) / np.sin(beta))
        distance = float(camera.far) * 0.70
        self._ensure_haze_buffer(max(3, int(lights.horizon_haze_slices)))

        block = self._haze_block
        block["view_proj"][:] = np.asarray(view_proj, np.float32).T
        block["eye"][:3] = camera.eye
        block["basis_x"][:3] = transform[:3, 0] / np.linalg.norm(transform[:3, 0])
        block["basis_y"][:3] = transform[:3, 1] / np.linalg.norm(transform[:3, 1])
        block["normal"][:3] = normal
        block["geometry"][:] = (distance, elevation, radius, transition)
        block["color"][:3] = lights.haze_color
        block["params"][:] = (
            EXPOSURE,
            tonemap,
            1.0 if scene.shading_model is ShadingModel.MUJOCO_CLASSIC else 0.0,
            0.0,
        )
        self._device.queue.write_buffer(self._haze_uniforms, 0, block.tobytes())
        return True

    def _ensure_haze_buffer(self, slices: int) -> None:
        if slices == self._haze_slices:
            return
        self._haze_buffer.destroy()
        vertices = self._haze_vertices(slices)
        self._haze_buffer = self._device.create_buffer_with_data(
            data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self._haze_vertex_count = len(vertices)
        self._haze_slices = slices

    def _ensure_classic_buffer(self, slices: int) -> None:
        if slices == self._classic_slices:
            return
        self._classic_buffer.destroy()
        vertices = self._classic_skybox_vertices(slices)
        self._classic_buffer = self._device.create_buffer_with_data(
            data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self._classic_vertex_count = len(vertices)
        self._classic_slices = slices

    def release(self) -> None:
        self._skybox_uniforms.destroy()
        self._haze_uniforms.destroy()
        self._classic_buffer.destroy()
        self._haze_buffer.destroy()
        self._skybox_groups.clear()
