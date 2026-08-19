"""wgpu-py render backend for forge-viewer.

Offscreen ``RenderBackend`` implementation on WebGPU (via wgpu-py).  It consumes
the same renderer-neutral contracts as ``ForgeBackend`` — ``SceneSourceBuilder``
produces the ``RenderScene``, this package owns device management, WGSL shader
pipeline, MSAA targets, and CPU readbacks.  No GL context or window is needed,
which makes it usable headless on any platform with a Vulkan/Metal/D3D12 driver.

Scope: the offscreen ``Renderer`` contract (color/depth/segmentation), lights
(directional/point/spot + headlight), 2D and cube textures, skybox, IBL image
light, transparency sorting, fog and haze, selection highlight and outline,
debug views (albedo/normal/depth/segment/idcolor/overdraw/wireframe), shadows
(directional CSM atlas + spot/point/area distance maps), and planar
reflections.  Not yet implemented: tendons, debug draw, gizmo, labels, and
the interactive viewer surface path.
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
from .lighting import (
    IMAGE_LIGHT_REFERENCE_INTENSITY,
    LIGHTS_BYTES,
    LightUniforms,
    active_image_light,
)
from .meshes import WIRE_STRIDE, MeshStore
from .passes import (
    OutlinePass,
    PresentPass,
    ReflectPass,
    ShadowPass,
    SkyboxPass,
)
from .programs import load_wgsl
from .targets import FRAME_BYTES, FRAME_DTYPE, RenderTargetWgpu, proj_matrix_wgpu
from .textures import TextureStore

HIGHLIGHT_COLOR = (1.0, 0.82, 0.45, 0.0)
HIGHLIGHT_BLEND = 0.35
HIGHLIGHT_EMISSION = 0.35
EXPOSURE = 1.0
# forge opaque.NO_CLIP: a clip-plane equation that never discards.
NO_CLIP = (0.0, 0.0, 0.0, 1.0)
# forge opaque.OVERDRAW_CLEAR: the overdraw view accumulates on black.
OVERDRAW_CLEAR = (0.0, 0.0, 0.0, 1.0)

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
        RenderFlag.SKYBOX,
        RenderFlag.SHADOW,
        RenderFlag.REFLECTION,
        RenderFlag.OUTLINE,
        RenderFlag.WIREFRAME,
    }
)

# Fragment-swap debug views; OVERDRAW/WIREFRAME are scene pipeline variants and
# SEGMENT/IDCOLOR are rebuilt by the present pass, so they have no entry here.
_DEBUG_ENTRIES = {
    DebugView.SHADED: "fs_scene",
    DebugView.ALBEDO: "fs_albedo",
    DebugView.NORMAL: "fs_normal",
    DebugView.DEPTH: "fs_depth",
}
_SUPPORTED_VIEWS = frozenset(DebugView)

_VERTEX_LAYOUT = {
    "array_stride": 32,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x3", "offset": 12, "shader_location": 1},
        {"format": "float32x2", "offset": 24, "shader_location": 2},
    ],
}

# Non-indexed wireframe stream from MeshStore: the scene vertex attributes plus
# a barycentric triple per vertex (see shaders/scene.wgsl vs_scene_wire).
_WIREFRAME_LAYOUT = {
    "array_stride": WIRE_STRIDE * 4,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x3", "offset": 12, "shader_location": 1},
        {"format": "float32x2", "offset": 24, "shader_location": 2},
        {"format": "float32x3", "offset": 32, "shader_location": 3},
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
# forge state_overdraw: blend_func(ONE, ONE), no depth test or write.
_OVERDRAW_BLEND = {
    "color": {"src_factor": "one", "dst_factor": "one"},
    "alpha": {"src_factor": "one", "dst_factor": "one"},
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

        self._module = self.device.create_shader_module(
            code=load_wgsl("shadow_sample.wgsl", "scene.wgsl")
        )
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
        self._group2_layout = self.device.create_bind_group_layout(
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
        # The shadow pass owns its sampling layout (atlas + local distance
        # array); the scene pipeline binds it as group 3 with live maps or
        # 1x1 fallbacks, mirroring forge's always-bound shadow uniforms.
        # Group 4 is the reflection pass's four color textures + sampler.
        self._shadows = ShadowPass(self.device)
        self._reflect = ReflectPass(self.device)
        self._scene_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[
                self._group0_layout,
                self._group1_layout,
                self._group2_layout,
                self._shadows.bind_layout,
                self._reflect.sample_layout,
            ]
        )
        self._export_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._group0_layout]
        )
        self._pipelines: dict[tuple, wgpu.GPURenderPipeline] = {}
        self._texture_groups: dict[int, wgpu.GPUBindGroup] = {}
        self._image_light_groups: dict[int, wgpu.GPUBindGroup] = {}
        self._skybox = SkyboxPass(self.device, self.target.samples)
        self._outline = OutlinePass(self.device, self.target.samples)
        self._present = PresentPass(self.device)

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
        self._flags[RenderFlag.WIREFRAME] = False
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
            debug_views=_SUPPORTED_VIEWS,
            capture=True,
            orthographic=True,
            shadows=True,
            outline=True,
            gizmo=False,
            msaa_samples=self.target.samples,
            id_msaa=False,
            renderer=f"wgpu-py {wgpu.__version__} on {info.vendor} {info.device}",
            notes=("offscreen only; no tendons/debug draw yet",),
        )

    # -- scene contract -------------------------------------------------------

    def set_background(self, rgba: tuple[float, float, float, float]) -> None:
        self._background = tuple(float(c) for c in rgba)

    def set_scene(self, source: SceneSource) -> None:
        self._source = source
        self.meshes.sync({**all_builtin(), **source.meshes})
        self.textures.sync(source.textures, source.skybox)
        self._texture_groups.clear()
        self._image_light_groups.clear()
        self._skybox.reset()
        self._builder = SceneSourceBuilder()
        self._scene = self._builder.set_source(source, self._camera)

    def set_render_scene(self, scene: RenderScene) -> None:
        # Unlike forge there is no per-scene VAO to rebuild; the instance
        # storage buffer is re-uploaded from the stored scene every render().
        self._scene = scene

    def update(self, frame: SceneFrame) -> None:
        if self._builder is None:
            return
        self.meshes.update(frame.mesh_updates)
        self._scene = self._builder.update(frame, self._camera)

    def set_camera(self, camera: CameraView) -> None:
        self._camera = camera

    # -- pipelines ------------------------------------------------------------

    def _scene_pipeline(
        self, fs_entry: str, blend: str, cull: str, wireframe: bool = False
    ) -> wgpu.GPURenderPipeline:
        key = ("scene", fs_entry, blend, cull, wireframe, self.target.samples)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        overdraw = fs_entry == "fs_overdraw"
        # The wireframe variant always pairs vs_scene_wire with fs_scene_wire.
        wireframe = wireframe or fs_entry == "fs_scene_wire"
        target: dict = {"format": "rgba8unorm"}
        if overdraw:
            target["blend"] = _OVERDRAW_BLEND
        elif blend == "alpha":
            target["blend"] = _ALPHA_BLEND
        elif blend == "additive":
            target["blend"] = _ADDITIVE_BLEND
        pipeline = self.device.create_render_pipeline(
            layout=self._scene_layout,
            vertex={
                "module": self._module,
                "entry_point": "vs_scene_wire" if wireframe else "vs_scene",
                "buffers": [_WIREFRAME_LAYOUT if wireframe else _VERTEX_LAYOUT],
            },
            fragment={
                "module": self._module,
                "entry_point": fs_entry,
                "targets": [target],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": cull},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": blend == "opaque" and not overdraw,
                "depth_compare": "always" if overdraw else "less",
            },
            multisample={"count": self.target.samples},
        )
        self._pipelines[key] = pipeline
        return pipeline

    def _reflect_pipeline(self, blend: str, cull: str) -> wgpu.GPURenderPipeline:
        """Scene pipeline variant for the reflection pass.

        Single-sampled rgba16float target, mirrored winding (front_face cw),
        always the shaded fragment entry — mirroring forge's ReflectPass FBO
        and gl.front_face handling.
        """
        key = ("reflect", blend, cull)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        target: dict = {"format": "rgba16float"}
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
                "entry_point": "fs_scene",
                "targets": [target],
            },
            primitive={"topology": "triangle-list", "front_face": "cw", "cull_mode": cull},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": blend == "opaque",
                "depth_compare": "less",
            },
            multisample={"count": 1},
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
            vertex={
                "module": self._module,
                "entry_point": "vs_export",
                "buffers": [_VERTEX_LAYOUT],
            },
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
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.target.frame_buffer,
                        "offset": 0,
                        "size": FRAME_BYTES,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.instances.buffer,
                        "offset": 0,
                        "size": self.instances.capacity * 128,
                    },
                },
                {
                    "binding": 2,
                    "resource": {"buffer": self.lights.buffer, "offset": 0, "size": LIGHTS_BYTES},
                },
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

    def _image_light_binding(self, lights) -> tuple[wgpu.GPUBindGroup, float, float]:
        """Bind the image-light cube (or the black fallback) and its shader terms.

        Mirrors ``OpaquePass._light_uniforms``: the gain is normalized by
        MuJoCo's 5000 reference intensity, the max mip comes from the bound
        texture's face size, and a missing cube binds ``black_cube``.
        """
        image = active_image_light(lights)
        gain = (
            max(float(image.intensity), 0.0) / IMAGE_LIGHT_REFERENCE_INTENSITY
            if image is not None
            else 0.0
        )
        entry = self.textures.cube(image.texture) if image is not None else None
        view, size = entry if entry is not None else (self.textures.black_cube, 1)
        key = id(view)
        group = self._image_light_groups.get(key)
        if group is None:
            group = self.device.create_bind_group(
                layout=self._group2_layout,
                entries=[
                    {"binding": 0, "resource": view},
                    {"binding": 1, "resource": self.textures.cube_sampler},
                ],
            )
            self._image_light_groups[key] = group
        return group, gain, float(np.floor(np.log2(max(size, 1))))

    # -- frame encoding -------------------------------------------------------

    def _write_frame_uniforms(
        self,
        cam: CameraView,
        light_count: int,
        image_light: tuple[float, float],
        reflection_size: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        view = np.asarray(cam.view_matrix(), np.float32)
        proj = proj_matrix_wgpu(cam)
        self._pack_frame_block(
            self._frame,
            cam,
            view,
            proj @ view,
            cam.eye,
            light_count,
            image_light,
            NO_CLIP,
            reflection_size,
            0.0,
        )
        self.device.queue.write_buffer(self.target.frame_buffer, 0, self._frame.tobytes())
        return proj @ view

    def _pack_frame_block(
        self,
        f: np.ndarray,
        cam: CameraView,
        view: np.ndarray,
        view_proj: np.ndarray,
        eye,
        light_count: int,
        image_light: tuple[float, float],
        clip_plane: tuple[float, float, float, float],
        reflection_size: tuple[float, float],
        linear_out: float,
    ) -> None:
        """Fill one frame uniform block; shared by the main and mirror views."""
        lights = self._scene.lights if self._scene is not None else None

        f["view_proj"][:] = np.asarray(view_proj, np.float32).T
        f["view"][:] = np.asarray(view, np.float32).T
        f["camera_pos"][:3] = eye
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
        f["flags"][:] = (1.0 if cam.orthographic else 0.0, linear_out, 0.0, 0.0)
        f["ids"][:] = (self._selected, light_count, 0, 0)
        f["image_light"][:] = (*image_light, 0.0, 0.0)
        f["clip_plane"][:] = clip_plane
        f["reflection"][:] = (*reflection_size, 0.0, 0.0)

    def _draw_buckets(
        self,
        pass_encoder,
        group0,
        group2,
        group3,
        group4,
        buckets,
        blend: str,
        cull: str,
        reflect: bool = False,
    ) -> tuple[int, int]:
        scene = self._scene
        assert scene is not None
        draw_calls = 0
        textured = self.get_flag(RenderFlag.TEXTURE)
        fs_entry = "fs_scene" if reflect else _DEBUG_ENTRIES.get(self._debug_view, "fs_scene")
        # OVERDRAW and WIREFRAME are variants of the shaded pipeline, mirroring
        # forge's OpaquePass spec selection (opaque.DEBUG_DEFINE/WIREFRAME).
        overdraw = not reflect and self._debug_view is DebugView.OVERDRAW
        wireframe = (
            not reflect
            and not overdraw
            and fs_entry == "fs_scene"
            and (self._debug_view is DebugView.WIREFRAME or self.get_flag(RenderFlag.WIREFRAME))
        )
        if overdraw:
            fs_entry = "fs_overdraw"
        elif wireframe:
            fs_entry = "fs_scene_wire"
        for b in buckets:
            start, stop = scene.bucket_ranges[b]
            if stop <= start:
                continue
            mesh_key, matid = scene.bucket_keys[b]
            mesh = self.meshes.get(mesh_key)
            if mesh is None:
                continue
            if reflect:
                pipeline = self._reflect_pipeline(blend, cull)
            else:
                pipeline = self._scene_pipeline(fs_entry, blend, cull, wireframe)
            pass_encoder.set_pipeline(pipeline)
            texture_name = None
            if textured and matid < len(scene.materials):
                texture_name = scene.materials[matid].texture
            pass_encoder.set_bind_group(0, group0)
            pass_encoder.set_bind_group(1, self._texture_group(texture_name))
            pass_encoder.set_bind_group(2, group2)
            pass_encoder.set_bind_group(3, group3)
            pass_encoder.set_bind_group(4, group4)
            if wireframe:
                wire_vbo, wire_count = mesh.wireframe()
                pass_encoder.set_vertex_buffer(0, wire_vbo)
                pass_encoder.draw(wire_count, stop - start, 0, start)
            else:
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
        target = self.target
        cam = self._camera.with_aspect(target.width / max(target.height, 1))
        # Plane detection runs before the instance upload so the encoded
        # negative reflectance reaches the GPU in the same write (forge relies
        # on its persistent scene object for the same ordering).
        reflective = self._reflect.prepare(scene, cam, self._flags, target.width, target.height)
        self.instances.upload(scene)
        # Shadow maps render before the scene pass, matching forge's
        # PASS_ORDER; the same frame's scene pass samples them.
        shadow = self._shadows.prepare(scene, cam, self._flags)
        schedule = self.lights.upload(scene.lights, shadow)
        light_count = len(schedule.lights)
        group2, image_gain, image_mip = self._image_light_binding(scene.lights)
        reflection_size = (float(target.width), float(target.height)) if reflective else (0.0, 0.0)
        view_proj = self._write_frame_uniforms(
            cam, light_count, (image_gain, image_mip), reflection_size
        )
        if reflective:
            self._reflect.write_frames(
                cam, light_count, (image_gain, image_mip), self._pack_frame_block
            )
        group3 = self._shadows.sample_group(shadow)
        group4 = self._reflect.sample_group()

        cull = "back" if self.get_flag(RenderFlag.CULL_FACE) else "none"
        group0 = self._bind_group0()
        encoder = self.device.create_command_encoder()

        draw_calls = 0
        if shadow is not None:
            draw_calls += self._shadows.execute(encoder, scene, self.meshes, self.instances)
        # Mirrored scene passes run between the shadow pass and the main scene
        # pass, matching forge's PASS_ORDER (shadow, reflect, opaque, ...).
        if reflective:
            group4_fallback = self._reflect.fallback_group()

            def draw_reflected(pass_encoder, plane_group0, buckets, blend):
                return self._draw_buckets(
                    pass_encoder,
                    plane_group0,
                    group2,
                    group3,
                    group4_fallback,
                    buckets,
                    blend,
                    cull,
                    reflect=True,
                )

            calls, _ = self._reflect.execute(
                encoder, scene, self._group0_layout, self.instances, self.lights, draw_reflected
            )
            draw_calls += calls

        color_view = target.color_ms.create_view() if target.color_ms is not None else None
        color_attachment = {
            "view": color_view if color_view is not None else target.color.create_view(),
            "clear_value": (
                OVERDRAW_CLEAR if self._debug_view is DebugView.OVERDRAW else self._background
            ),
            "load_op": "clear",
            "store_op": "store" if color_view is None else "discard",
        }
        if color_view is not None:
            color_attachment["resolve_target"] = target.color.create_view()
        # The outline mask renders ahead of the main pass; the dilation
        # composite joins the main pass after transparent geometry, matching
        # forge's PASS_ORDER (outline between transparent and present).
        outline_buckets = self._outline.prepare(scene, self._selected, self._flags)
        if outline_buckets:
            draw_calls += self._outline.render_mask(
                encoder,
                scene,
                self.meshes,
                self.instances,
                view_proj,
                target.width,
                target.height,
            )
        pass1 = encoder.begin_render_pass(
            color_attachments=[color_attachment],
            depth_stencil_attachment={
                "view": target.zbuf.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        instances_drawn = 0
        if scene.count:
            calls, drawn = self._draw_buckets(
                pass1, group0, group2, group3, group4, scene.opaque_buckets, "opaque", cull
            )
            draw_calls += calls
            instances_drawn += drawn
        # Skybox and horizon haze sit between opaque and transparent, matching
        # forge's PASS_ORDER (the id pass is the separate export pass here).
        draw_calls += self._skybox.render(
            pass1,
            textures=self.textures,
            flags=self._flags,
            debug_view=self._debug_view,
            camera=cam,
            view_proj=view_proj,
            scene=scene,
        )
        if scene.count and scene.transparent_buckets and self.get_flag(RenderFlag.TRANSPARENT):
            blend = "additive" if self.get_flag(RenderFlag.ADDITIVE) else "alpha"
            order = scene.transparent_draw_order()
            calls, drawn = self._draw_buckets(
                pass1, group0, group2, group3, group4, order, blend, cull
            )
            draw_calls += calls
            instances_drawn += drawn
        if outline_buckets:
            draw_calls += self._outline.composite(pass1, target.width, target.height)
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

        # SEGMENT/IDCOLOR rebuild the resolved color from the export ids;
        # other views need no present work (the resolve happened in pass1).
        draw_calls += self._present.execute(
            encoder, target.color, target.export_id, self._debug_view, self._selected
        )

        self.device.queue.submit([encoder.finish()])

        self.stats.draw_calls = draw_calls
        self.stats.instances = instances_drawn
        self.stats.buckets = scene.bucket_count()
        self.stats.triangles = scene.triangle_count(self.meshes.triangle_counts())
        self.stats.frame_cpu_ms = (time.perf_counter() - t0) * 1000.0
        # Same report keys as ForgeBackend._update_light_stats.
        self.stats.notes = {
            "scene lights": (f"{len(schedule.lights)} active, {schedule.deferred_lights} deferred"),
            "shadow casters": (
                f"{schedule.selected_shadow_count} active, {schedule.deferred_shadows} deferred"
            ),
        }
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
        if view not in _SUPPORTED_VIEWS:
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
        self._skybox.release()
        self._outline.release()
        self._present.release()
        self._shadows.release()
        self._reflect.release()
        self._pipelines.clear()
        self._texture_groups.clear()
        self._image_light_groups.clear()
        self._scene = None
        self._builder = None
        self._source = None
