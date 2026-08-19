"""wgpu-py render backend for forge-viewer.

Offscreen ``RenderBackend`` implementation on WebGPU (via wgpu-py).  It consumes
the same renderer-neutral contracts as ``ForgeBackend`` — ``SceneSourceBuilder``
produces the ``RenderScene``, this package owns device management, WGSL shader
pipeline, MSAA targets, and CPU readbacks.  No GL context or window is needed,
which makes it usable headless on any platform with a Vulkan/Metal/D3D12 driver.

Scope: the offscreen ``Renderer`` contract (color/depth/segmentation), lights
(directional/point/spot + headlight), 2D textures, transparency sorting, fog
and haze, selection highlight.  Not yet implemented: shadows, planar
reflections, skybox, image lights, wireframe, tendons, debug draw, gizmo,
labels, and the interactive viewer surface path.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import wgpu

from ...adapters.base import SceneFrame, SceneSource
from ...types import CameraView
from ..backend import BackendCaps, DebugView, FrameMode, LabelMode, RenderFlag, RenderStats
from ..builder import SceneSourceBuilder
from ..mesh import all_builtin
from ..scene import RenderScene
from .instances import InstanceStore
from .lighting import LightUniforms
from .meshes import MeshStore
from .shaders import SCENE_WGSL
from .targets import FRAME_DTYPE, RenderTargetWgpu, proj_matrix_wgpu
from .textures import TextureStore

HIGHLIGHT_COLOR = (1.0, 0.82, 0.45, 0.0)
HIGHLIGHT_BLEND = 0.35
HIGHLIGHT_EMISSION = 0.35
EXPOSURE = 1.0

_SUPPORTED_FLAGS = frozenset(
    {
        RenderFlag.TRANSPARENT,
        RenderFlag.TEXTURE,
        RenderFlag.MSAA,
        RenderFlag.CULL_FACE,
        RenderFlag.TONEMAP,
        RenderFlag.FOG,
        RenderFlag.HAZE,
        RenderFlag.ADDITIVE,
    }
)

_DEBUG_ENTRIES = {
    DebugView.SHADED: "fs_scene",
    DebugView.ALBEDO: "fs_albedo",
    DebugView.NORMAL: "fs_normal",
    DebugView.DEPTH: "fs_depth",
}

_VERTEX_LAYOUT = {
    "array_stride": 32,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x3", "offset": 12, "shader_location": 1},
        {"format": "float32x2", "offset": 24, "shader_location": 2},
    ],
}

_ALPHA_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
    "alpha": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
}
_ADDITIVE_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one"},
    "alpha": {"src_factor": "src-alpha", "dst_factor": "one"},
}


class WgpuBackend:
    """Offscreen scene renderer over wgpu-py; see module docstring for scope."""

    def __init__(self, width: int = 1280, height: int = 720, samples: int = 4) -> None:
        self.device = wgpu.utils.get_default_device()
        self.meshes = MeshStore(self.device)
        self.textures = TextureStore(self.device)
        self.instances = InstanceStore(self.device)
        self.lights = LightUniforms(self.device)
        self.target = RenderTargetWgpu(self.device, width, height, samples)

        self._module = self.device.create_shader_module(code=SCENE_WGSL)
        self._group0_layout = self.device.create_bind_group_layout(
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
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "read-only-storage"},
                },
            ]
        )
        self._group1_layout = self.device.create_bind_group_layout(
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
        self._scene_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._group0_layout, self._group1_layout]
        )
        self._export_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._group0_layout]
        )
        self._pipelines: dict[tuple, wgpu.GPURenderPipeline] = {}
        self._texture_groups: dict[int, wgpu.GPUBindGroup] = {}

        self.stats = RenderStats()
        self.debug = None
        self._camera = CameraView()
        self._background = (0.13, 0.14, 0.16, 1.0)
        self._selected = 0
        self._include_transparent_ids = False
        self._debug_view = DebugView.SHADED
        self._label_mode = LabelMode.NONE
        self._frame_mode = FrameMode.NONE
        self._bvh_depth = 0
        self._flags: dict[RenderFlag, bool] = dict.fromkeys(_SUPPORTED_FLAGS, True)
        self._flags[RenderFlag.ADDITIVE] = False
        self._flags[RenderFlag.FOG] = False
        self._flags[RenderFlag.HAZE] = False

        self._source: SceneSource | None = None
        self._scene: RenderScene | None = None
        self._builder: SceneSourceBuilder | None = None
        self._frame = np.zeros((), FRAME_DTYPE)
        self.caps = self._build_caps()

    # -- capabilities ---------------------------------------------------------

    def _build_caps(self) -> BackendCaps:
        info = self.device.adapter.info
        return BackendCaps(
            name="webgpu",
            gpu_pick=True,
            render_flags=frozenset(self._flags),
            debug_views=frozenset(_DEBUG_ENTRIES),
            capture=True,
            orthographic=True,
            shadows=False,
            outline=False,
            gizmo=False,
            msaa_samples=self.target.samples,
            id_msaa=False,
            renderer=f"wgpu-py {wgpu.__version__} on {info.vendor} {info.device}",
            notes=(
                "offscreen only; no shadows/reflections/skybox/tendons/debug draw yet",
            ),
        )

    # -- scene contract -------------------------------------------------------

    def set_background(self, rgba: tuple[float, float, float, float]) -> None:
        self._background = tuple(float(c) for c in rgba)

    def set_scene(self, source: SceneSource) -> None:
        self._source = source
        self.meshes.sync({**all_builtin(), **source.meshes})
        self.textures.sync(source.textures)
        self._texture_groups.clear()
        self._builder = SceneSourceBuilder()
        self._scene = self._builder.set_source(source, self._camera)

    def update(self, frame: SceneFrame) -> None:
        if self._builder is None:
            return
        self.meshes.update(frame.mesh_updates)
        self._scene = self._builder.update(frame, self._camera)

    def set_camera(self, camera: CameraView) -> None:
        self._camera = camera

    # -- pipelines ------------------------------------------------------------

    def _scene_pipeline(self, fs_entry: str, blend: str, cull: str) -> wgpu.GPURenderPipeline:
        key = ("scene", fs_entry, blend, cull, self.target.samples)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        target: dict = {"format": "rgba8unorm"}
        if blend == "alpha":
            target["blend"] = _ALPHA_BLEND
        elif blend == "additive":
            target["blend"] = _ADDITIVE_BLEND
        pipeline = self.device.create_render_pipeline(
            layout=self._scene_layout,
            vertex={
                "module": self._module,
                "entry_point": "vs_scene",
                "buffers": [_VERTEX_LAYOUT],
            },
            fragment={
                "module": self._module,
                "entry_point": fs_entry,
                "targets": [target],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": cull},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": blend == "opaque",
                "depth_compare": "less",
            },
            multisample={"count": self.target.samples},
        )
        self._pipelines[key] = pipeline
        return pipeline

    def _export_pipeline(self, cull: str) -> wgpu.GPURenderPipeline:
        key = ("export", cull)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        pipeline = self.device.create_render_pipeline(
            layout=self._export_layout,
            vertex={"module": self._module, "entry_point": "vs_export", "buffers": [_VERTEX_LAYOUT]},
            fragment={
                "module": self._module,
                "entry_point": "fs_export",
                "targets": [{"format": "r32float"}, {"format": "r32uint"}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": cull},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
            multisample={"count": 1},
        )
        self._pipelines[key] = pipeline
        return pipeline

    def _bind_group0(self) -> wgpu.GPUBindGroup:
        return self.device.create_bind_group(
            layout=self._group0_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.target.frame_buffer, "offset": 0, "size": 336}},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.instances.buffer,
                        "offset": 0,
                        "size": self.instances.capacity * 128,
                    },
                },
                {"binding": 2, "resource": {"buffer": self.lights.buffer, "offset": 0, "size": 8000}},
            ],
        )

    def _texture_group(self, name: str | None) -> wgpu.GPUBindGroup:
        view = self.textures.get(name) if name else None
        if view is None:
            view = self.textures.white
        key = id(view)
        group = self._texture_groups.get(key)
        if group is None:
            group = self.device.create_bind_group(
                layout=self._group1_layout,
                entries=[
                    {"binding": 0, "resource": view},
                    {"binding": 1, "resource": self.textures.sampler},
                ],
            )
            self._texture_groups[key] = group
        return group

    # -- frame encoding -------------------------------------------------------

    def _write_frame_uniforms(self, light_count: int) -> None:
        cam = self._camera.with_aspect(self.target.width / max(self.target.height, 1))
        view = np.asarray(cam.view_matrix(), np.float32)
        proj = proj_matrix_wgpu(cam)
        lights = self._scene.lights if self._scene is not None else None

        f = self._frame
        f["view_proj"][:] = (proj @ view).T
        f["view"][:] = view.T
        f["camera_pos"][:3] = cam.eye
        f["camera_dir"][:3] = cam.forward()
        if lights is not None:
            f["ambient"][:3] = lights.ambient
            diffuse, specular = self.lights.headlight_terms(lights)
            f["headlight_diffuse"][:] = diffuse
            f["headlight_specular"][:] = specular
            fog_on = self.get_flag(RenderFlag.FOG) and lights.fog_end > lights.fog_start
            haze = (
                float(lights.haze_density)
                if self.get_flag(RenderFlag.HAZE) and not lights.horizon_haze
                else 0.0
            )
            f["fog"][:] = (lights.fog_start, lights.fog_end, 1.0 if fog_on else 0.0, haze)
            f["fog_color"][:3] = lights.fog_color
            f["haze_color"][:3] = lights.haze_color
        f["highlight_color"][:] = HIGHLIGHT_COLOR
        f["highlight"][:] = (HIGHLIGHT_BLEND, HIGHLIGHT_EMISSION, 0.0, 0.0)
        f["shading"][:] = (
            EXPOSURE,
            1.0 if self.get_flag(RenderFlag.TONEMAP) else 0.0,
            float(cam.near),
            float(cam.far),
        )
        f["flags"][:] = (1.0 if cam.orthographic else 0.0, 0.0, 0.0, 0.0)
        f["ids"][:] = (self._selected, light_count, 0, 0)
        self.device.queue.write_buffer(self.target.frame_buffer, 0, self._frame.tobytes())

    def _draw_buckets(self, pass_encoder, group0, buckets, blend: str, cull: str) -> tuple[int, int]:
        scene = self._scene
        assert scene is not None
        draw_calls = 0
        textured = self.get_flag(RenderFlag.TEXTURE)
        fs_entry = _DEBUG_ENTRIES.get(self._debug_view, "fs_scene")
        for b in buckets:
            start, stop = scene.bucket_ranges[b]
            if stop <= start:
                continue
            mesh_key, matid = scene.bucket_keys[b]
            mesh = self.meshes.get(mesh_key)
            if mesh is None:
                continue
            pipeline = self._scene_pipeline(fs_entry, blend, cull)
            pass_encoder.set_pipeline(pipeline)
            texture_name = None
            if textured and matid < len(scene.materials):
                texture_name = scene.materials[matid].texture
            pass_encoder.set_bind_group(0, group0)
            pass_encoder.set_bind_group(1, self._texture_group(texture_name))
            pass_encoder.set_vertex_buffer(0, mesh.vbo)
            pass_encoder.set_index_buffer(mesh.ibo, "uint32")
            pass_encoder.draw_indexed(mesh.index_count, stop - start, 0, 0, start)
            draw_calls += 1
        return draw_calls, sum(
            max(0, stop - start) for start, stop in (scene.bucket_ranges[b] for b in buckets)
        )

    def render(self, frame: SceneFrame | None = None) -> None:
        if frame is not None:
            self.update(frame)
        scene = self._scene
        if scene is None:
            return None

        t0 = time.perf_counter()
        self.instances.upload(scene)
        light_count = self.lights.upload(scene.lights)
        self._write_frame_uniforms(light_count)

        cull = "back" if self.get_flag(RenderFlag.CULL_FACE) else "none"
        group0 = self._bind_group0()
        target = self.target
        encoder = self.device.create_command_encoder()

        color_view = target.color_ms.create_view() if target.color_ms is not None else None
        color_attachment = {
            "view": color_view if color_view is not None else target.color.create_view(),
            "clear_value": self._background,
            "load_op": "clear",
            "store_op": "store" if color_view is None else "discard",
        }
        if color_view is not None:
            color_attachment["resolve_target"] = target.color.create_view()
        pass1 = encoder.begin_render_pass(
            color_attachments=[color_attachment],
            depth_stencil_attachment={
                "view": target.zbuf.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        draw_calls = 0
        instances_drawn = 0
        if scene.count:
            calls, drawn = self._draw_buckets(pass1, group0, scene.opaque_buckets, "opaque", cull)
            draw_calls += calls
            instances_drawn += drawn
            if scene.transparent_buckets and self.get_flag(RenderFlag.TRANSPARENT):
                blend = "additive" if self.get_flag(RenderFlag.ADDITIVE) else "alpha"
                order = scene.transparent_draw_order()
                calls, drawn = self._draw_buckets(pass1, group0, order, blend, cull)
                draw_calls += calls
                instances_drawn += drawn
        pass1.end()

        pass2 = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target.export_depth.create_view(),
                    # depth 1.0 == far plane, matching the GL depth clear
                    "clear_value": (1.0, 1.0, 1.0, 1.0),
                    "load_op": "clear",
                    "store_op": "store",
                },
                {
                    "view": target.export_id.create_view(),
                    "clear_value": (0, 0, 0, 0),
                    "load_op": "clear",
                    "store_op": "store",
                },
            ],
            depth_stencil_attachment={
                "view": target.export_zbuf.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        if scene.count:
            pass2.set_pipeline(self._export_pipeline(cull))
            pass2.set_bind_group(0, group0)
            buckets = list(scene.opaque_buckets)
            if self._include_transparent_ids and self.get_flag(RenderFlag.TRANSPARENT):
                buckets += list(scene.transparent_draw_order())
            for b in buckets:
                start, stop = scene.bucket_ranges[b]
                if stop <= start:
                    continue
                mesh = self.meshes.get(scene.bucket_keys[b][0])
                if mesh is None:
                    continue
                pass2.set_vertex_buffer(0, mesh.vbo)
                pass2.set_index_buffer(mesh.ibo, "uint32")
                pass2.draw_indexed(mesh.index_count, stop - start, 0, 0, start)
        pass2.end()

        self.device.queue.submit([encoder.finish()])

        self.stats.draw_calls = draw_calls
        self.stats.instances = instances_drawn
        self.stats.buckets = scene.bucket_count()
        self.stats.triangles = scene.triangle_count(self.meshes.triangle_counts())
        self.stats.frame_cpu_ms = (time.perf_counter() - t0) * 1000.0
        return None

    # -- misc protocol surface --------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        self.target.resize(width, height)

    def capture(
        self,
        path: Path,
        camera: CameraView | None = None,
        size: tuple[int, int] | None = None,
    ) -> bool:
        from PIL import Image

        saved_camera = self._camera
        saved_size = (self.target.width, self.target.height)
        try:
            if camera is not None:
                self._camera = camera
            if size is not None:
                self.target.resize(int(size[0]), int(size[1]))
            self.render()
            image = self.target.read_color(flip=True)[..., :3]
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ascontiguousarray(image)).save(path)
            return True
        finally:
            self._camera = saved_camera
            if size is not None:
                self.target.resize(*saved_size)

    def pick(self, x: int, y: int) -> int:
        # Match forge's GL viewport convention: y measured from the bottom.
        return self.target.read_id(int(x), self.target.height - 1 - int(y))

    def highlight(self, object_id: int) -> None:
        self._selected = int(object_id)

    def set_transparent_id_rendering(self, enabled: bool) -> None:
        self._include_transparent_ids = bool(enabled)

    def set_gizmo(self, gizmo) -> bool:
        return False

    def set_flag(self, flag: RenderFlag, value: bool) -> bool:
        self._flags[flag] = bool(value)
        return flag in _SUPPORTED_FLAGS

    def get_flag(self, flag: RenderFlag) -> bool:
        return self._flags.get(flag, False)

    def set_debug_view(self, view: DebugView) -> bool:
        if view not in _DEBUG_ENTRIES:
            return False
        self._debug_view = view
        return True

    def get_debug_view(self) -> DebugView:
        return self._debug_view

    def set_label_mode(self, mode: LabelMode) -> bool:
        self._label_mode = mode
        return False

    def get_label_mode(self) -> LabelMode:
        return self._label_mode

    def set_frame_mode(self, mode: FrameMode) -> bool:
        self._frame_mode = mode
        return False

    def get_frame_mode(self) -> FrameMode:
        return self._frame_mode

    def set_bvh_depth(self, depth: int) -> bool:
        self._bvh_depth = int(depth)
        return False

    def get_bvh_depth(self) -> int:
        return self._bvh_depth

    def render_options(self) -> tuple[RenderFlag, ...]:
        return tuple(flag for flag in self._flags if flag in _SUPPORTED_FLAGS)

    def release(self) -> None:
        self.meshes.release()
        self.textures.release()
        self.instances.release()
        self.lights.release()
        self.target.release()
        self._pipelines.clear()
        self._texture_groups.clear()
        self._scene = None
        self._builder = None
        self._source = None
