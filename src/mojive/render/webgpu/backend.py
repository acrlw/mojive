"""wgpu render backend for offscreen and interactive rendering.

The backend consumes the shared scene contracts and owns WebGPU resources,
WGSL pipelines, render passes, readback, and the viewport texture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import wgpu

from ...adapters.base import SceneFrame, SceneSource
from ...gizmo import GizmoFrame
from ...log import get_logger
from ...types import CameraView, ShadingModel, ViewportImage
from ..backend import (
    BackendCaps,
    DebugView,
    FrameMode,
    LabelMode,
    RenderFlag,
    RenderProduct,
    RenderRequest,
    RenderStats,
)
from ..builder import SceneSourceBuilder
from ..debugdraw import DebugDraw
from ..mesh import all_builtin
from ..overlay import OverlayPublisher, OverlayState
from ..scene import RenderScene
from .blend import ADDITIVE_BLEND, ALPHA_BLEND, OVERDRAW_BLEND
from .instances import InstanceStore
from .lighting import (
    IMAGE_LIGHT_REFERENCE_INTENSITY,
    LIGHTS_BYTES,
    LightUniforms,
    active_image_light,
)
from .meshes import WIRE_STRIDE, MeshStore
from .passes import (
    DebugPass,
    GizmoPass,
    OutlinePass,
    PresentPass,
    ReflectPass,
    ShadowPass,
    SkyboxPass,
    TendonPass,
)
from .programs import WgslWatch, load_wgsl
from .targets import FRAME_BYTES, FRAME_DTYPE, RenderTargetWgpu, proj_matrix_wgpu
from .textures import TextureStore
from .timing import WgpuTiming, default_device

log = get_logger("wgpu")

HIGHLIGHT_COLOR = (1.0, 0.82, 0.45, 0.0)
HIGHLIGHT_BLEND = 0.35
HIGHLIGHT_EMISSION = 0.35
EXPOSURE = 1.0
# opengl opaque.NO_CLIP: a clip-plane equation that never discards.
NO_CLIP = (0.0, 0.0, 0.0, 1.0)
# opengl opaque.OVERDRAW_CLEAR: the overdraw view accumulates on black.
OVERDRAW_CLEAR = (0.0, 0.0, 0.0, 1.0)

# Scene module sources in load order; the hot-reload watch tracks this set.
_SCENE_SHADERS = ("shadow_sample.wgsl", "scene.wgsl")

# Render flags implemented by the wgpu pass graph.
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
        RenderFlag.TENDON,
        RenderFlag.STATIC,
        RenderFlag.SKIN,
        RenderFlag.FLEXFACE,
        RenderFlag.FLEXSKIN,
        RenderFlag.ISLAND,
        RenderFlag.CONVEXHULL,
        RenderFlag.CONTACTPOINT,
        RenderFlag.CONTACTFORCE,
        RenderFlag.CONTACTSPLIT,
        RenderFlag.AUTOCONNECT,
        RenderFlag.ACTUATOR,
        RenderFlag.ACTIVATION,
        RenderFlag.JOINT,
        RenderFlag.COM,
        RenderFlag.INERTIA,
        RenderFlag.SCLINERTIA,
        RenderFlag.CAMERA,
        RenderFlag.LIGHT,
        RenderFlag.RANGEFINDER,
        RenderFlag.CONSTRAINT,
        RenderFlag.FLEXVERT,
        RenderFlag.FLEXEDGE,
        RenderFlag.BODYBVH,
        RenderFlag.MESHBVH,
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


@dataclass(frozen=True)
class WgpuRenderPlan:
    """Concrete WGPU workloads compiled from a product request."""

    request: RenderRequest
    color: bool
    export_depth: bool
    export_identity: bool

    @property
    def export(self) -> bool:
        return self.export_depth or self.export_identity


def compile_render_plan(
    request: RenderRequest | None,
    debug_view: DebugView = DebugView.SHADED,
) -> WgpuRenderPlan:
    """Compile requested products into WGPU scene and export workloads."""

    request = request or RenderRequest.viewport()
    color = request.needs(RenderProduct.COLOR)
    export_depth = request.needs(RenderProduct.METRIC_DEPTH)
    export_identity = request.needs(RenderProduct.OBJECT_ID) or request.needs(
        RenderProduct.SEGMENTATION
    )
    if color and debug_view in {DebugView.SEGMENT, DebugView.IDCOLOR}:
        export_identity = True
    return WgpuRenderPlan(request, color, export_depth, export_identity)


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


class WgpuBackend:
    """Offscreen scene renderer over wgpu-py; see module docstring for scope."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        samples: int = 4,
        device: wgpu.GPUDevice | None = None,
        *,
        gpu_timing: bool = True,
        scene_shader_code: str | None = None,
    ) -> None:
        # The viewer window shares this device (surface configuration and the
        # imgui texture binding must match the device that rendered the scene).
        self.device = device if device is not None else default_device()
        self.meshes = MeshStore(self.device)
        self.textures = TextureStore(self.device)
        self.instances = InstanceStore(self.device)
        self.lights = LightUniforms(self.device)
        self.target = RenderTargetWgpu(self.device, width, height, samples)
        self._configured_samples = self.target.samples
        self.timing = WgpuTiming(self.device, enabled=gpu_timing)

        self._scene_shader_code = scene_shader_code or load_wgsl(*_SCENE_SHADERS)
        self._module = self.device.create_shader_module(code=self._scene_shader_code)
        self._shader_watch = WgslWatch(*_SCENE_SHADERS)
        self._shader_reload_error = ""
        self._hot_reload = False
        self._shader_generation = 0
        self._frame_serial = 0
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
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 4,
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
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "cube"},
                },
                {
                    "binding": 3,
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
        # 1x1 fallbacks, mirroring opengl's always-bound shadow uniforms.
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
        self._texture_groups: dict[tuple[int, int], wgpu.GPUBindGroup] = {}
        self._image_light_groups: dict[int, wgpu.GPUBindGroup] = {}
        self._skybox = SkyboxPass(self.device, self.target.samples)
        self._outline = OutlinePass(self.device, self.target.samples)
        self._present = PresentPass(self.device)
        self._tendons = TendonPass(self.device)

        self.stats = RenderStats()
        self.debug = DebugDraw()
        self._debug = DebugPass(self.device, self.target.samples, self.debug)
        self._gizmo = GizmoPass(self.device, self.target.samples)

        self._gizmo_frame: GizmoFrame | None = None
        self._camera = CameraView()
        self._background = (0.13, 0.14, 0.16, 1.0)
        self._selected = 0
        self._include_transparent_ids = False
        self._debug_view = DebugView.SHADED
        self._label_mode = LabelMode.NONE
        self._frame_mode = FrameMode.NONE
        self._bvh_depth = 0
        self._flags: dict[RenderFlag, bool] = dict.fromkeys(_SUPPORTED_FLAGS, True)
        self._flags[RenderFlag.MSAA] = self.target.samples > 1
        self._flags[RenderFlag.WIREFRAME] = False
        self._flags[RenderFlag.ADDITIVE] = False
        self._flags[RenderFlag.FOG] = False
        self._flags[RenderFlag.CONTACTPOINT] = False
        self._flags[RenderFlag.CONTACTFORCE] = False
        self._flags[RenderFlag.CONTACTSPLIT] = False
        self._flags[RenderFlag.ISLAND] = False
        self._flags[RenderFlag.CONVEXHULL] = False
        self._flags[RenderFlag.AUTOCONNECT] = False
        self._flags[RenderFlag.ACTUATOR] = False
        self._flags[RenderFlag.ACTIVATION] = False
        self._flags[RenderFlag.JOINT] = False
        self._flags[RenderFlag.COM] = False
        self._flags[RenderFlag.INERTIA] = False
        self._flags[RenderFlag.SCLINERTIA] = False
        self._flags[RenderFlag.BODYBVH] = False
        self._flags[RenderFlag.MESHBVH] = False
        self._flags[RenderFlag.CAMERA] = False
        self._flags[RenderFlag.LIGHT] = False
        self._flags[RenderFlag.RANGEFINDER] = False
        self._flags[RenderFlag.CONSTRAINT] = False
        self._flags[RenderFlag.FLEXFACE] = False
        self._flags[RenderFlag.FLEXVERT] = False

        self._source: SceneSource | None = None
        self._scene: RenderScene | None = None
        self._builder: SceneSourceBuilder | None = None
        # Tendon publishing state, mirroring OpenGLBackend: per-source lookup
        # tables captured in set_scene plus reusable capsule packing buffers.
        # The actuator palette lives in the shared overlay publisher.
        self._overlay = OverlayPublisher(self.debug, self._flags)
        self._tendon_visible = np.zeros(0, bool)
        self._actuator_visible = np.zeros(0, bool)
        self._material_values = np.zeros((0, 4), np.float32)
        self._tendon_material_table: tuple = ()
        self._island_tendon_material_table: tuple = ()
        self._tendon_actuator = np.zeros(0, np.int32)
        self._capsule_segments = np.zeros((0, 2, 3), np.float32)
        self._capsule_widths = np.zeros(0, np.float32)
        self._capsule_colors = np.zeros((0, 4), np.float32)
        self._capsule_materials = np.zeros((0, 4), np.float32)
        self._capsule_material_ids = np.zeros(0, np.int32)
        self._capsule_transparent = np.zeros(0, bool)
        self._frame = np.zeros((), FRAME_DTYPE)
        self.caps = self._build_caps()

    # -- capabilities ---------------------------------------------------------

    def create_peer(self, width: int, height: int) -> WgpuBackend:
        peer = WgpuBackend(
            width,
            height,
            samples=self._configured_samples,
            device=self.device,
            scene_shader_code=self._scene_shader_code,
        )
        peer._hot_reload = self._hot_reload
        return peer

    def _build_caps(self) -> BackendCaps:
        info = self.device.adapter.info
        return BackendCaps(
            name="wgpu",
            gpu_pick=True,
            debug_draw=True,
            render_flags=frozenset(self._flags),
            debug_views=_SUPPORTED_VIEWS,
            label_modes=frozenset(LabelMode),
            frame_modes=frozenset(FrameMode),
            capture=True,
            orthographic=True,
            shadows=True,
            outline=True,
            gizmo=True,
            gpu_timing=self.timing.active,
            msaa_samples=self.target.samples,
            id_msaa=False,
            gl_version=f"WebGPU {info.get('backend_type', '')}".rstrip(),
            renderer=f"wgpu-py {wgpu.__version__} on {info.device}",
            notes=(
                *(
                    ()
                    if self.timing.active
                    else ("GPU timer queries unavailable; CPU frame timing only",)
                ),
                "MSAA changes rebuild multisampled targets and pipelines at runtime",
                # id_msaa=False: the export MRT pass re-rasterizes the scene
                # single-sampled instead of resolving the MSAA id/depth targets.
                "Object ID/depth export is single-sampled; WebGPU cannot resolve "
                "multisampled integer or depth attachments",
            ),
        )

    # -- scene contract -------------------------------------------------------

    def set_background(self, rgba: tuple[float, float, float, float]) -> None:
        self._background = tuple(float(c) for c in rgba)

    def set_scene(self, source: SceneSource) -> None:
        self._source = source
        self._tendon_visible = source.tendon_visible
        self._actuator_visible = source.actuator_visible
        self._material_values = np.asarray(
            [
                (mat.emission, mat.specular, mat.shininess, mat.reflectance)
                for mat in source.materials
            ],
            np.float32,
        )
        self._tendon_material_table = tuple(source.materials)
        self._island_tendon_material_table = tuple(
            replace(material, texture=None) for material in source.materials
        )
        self._tendon_actuator = np.full(len(source.tendon_rgba), -1, np.int32)
        for actuator, tendon in enumerate(source.actuator_tendon):
            if 0 <= tendon < len(self._tendon_actuator):
                self._tendon_actuator[tendon] = actuator
        self._overlay.set_scene(source)
        self.meshes.sync({**all_builtin(), **source.meshes})
        self.textures.sync(source.textures, source.skybox)
        self._texture_groups.clear()
        self._image_light_groups.clear()
        self._skybox.reset()
        self._builder = SceneSourceBuilder()
        self._scene = self._builder.set_source(source, self._camera)
        self._sync_instance_visibility()

    def set_render_scene(self, scene: RenderScene) -> None:
        # Unlike opengl there is no per-scene VAO to rebuild; the instance
        # storage buffer is re-uploaded from the stored scene every render().
        self._scene = scene

    def update(self, frame: SceneFrame) -> None:
        if self._builder is None:
            return
        self.meshes.update(frame.mesh_updates)
        island_rgba = frame.island_rgba if self.get_flag(RenderFlag.ISLAND) else None
        self.set_render_scene(self._builder.update(frame, self._camera, island_rgba))
        self._publish_tendons(frame)
        self._overlay.publish(frame, self._overlay_state())

    def _overlay_state(self) -> OverlayState:
        return OverlayState(
            camera=self._camera,
            viewport_height=self.target.height,
            selected=self._selected,
            label_mode=self._label_mode,
            frame_mode=self._frame_mode,
            bvh_depth=self._bvh_depth,
        )

    def _publish_tendons(self, frame: SceneFrame) -> None:
        """Pack visible tendon segments into capsule instances (opengl parity)."""
        segments, ids, widths = (
            frame.tendon_segments,
            frame.tendon_ids,
            frame.tendon_widths,
        )
        if segments is None or ids is None or widths is None or not len(segments):
            self._tendons.clear()
            return

        base_indices = (
            np.flatnonzero(self._tendon_visible[ids])
            if self.get_flag(RenderFlag.TENDON)
            else np.zeros(0, np.intp)
        )
        base_count = len(base_indices)
        actuator_indices = np.zeros(0, np.intp)
        segment_actuators = np.zeros(0, np.int32)
        if self.get_flag(RenderFlag.ACTUATOR) and frame.ctrl is not None:
            segment_actuators = self._tendon_actuator[ids]
            available = segment_actuators >= 0
            available[available] &= self._actuator_visible[segment_actuators[available]]
            actuator_indices = np.flatnonzero(available)
        total = base_count + len(actuator_indices)
        if not total:
            self._tendons.clear()
            return

        if total > len(self._capsule_widths):
            capacity = max(total, 2 * len(self._capsule_widths), 64)
            self._capsule_segments = np.zeros((capacity, 2, 3), np.float32)
            self._capsule_widths = np.zeros(capacity, np.float32)
            self._capsule_colors = np.zeros((capacity, 4), np.float32)
            self._capsule_materials = np.zeros((capacity, 4), np.float32)
            self._capsule_material_ids = np.zeros(capacity, np.int32)
            self._capsule_transparent = np.zeros(capacity, bool)

        assert self._source is not None
        if base_count:
            self._capsule_segments[:base_count] = segments[base_indices]
            self._capsule_widths[:base_count] = widths[base_indices]
            tendon_rgba = self._source.tendon_rgba
            if self.get_flag(RenderFlag.ISLAND) and frame.tendon_island_rgba is not None:
                tendon_rgba = frame.tendon_island_rgba
            np.take(
                tendon_rgba,
                ids[base_indices],
                axis=0,
                out=self._capsule_colors[:base_count],
                mode="clip",
            )
            np.take(
                self._material_values,
                self._source.tendon_material[ids[base_indices]],
                axis=0,
                out=self._capsule_materials[:base_count],
                mode="clip",
            )
            self._capsule_material_ids[:base_count] = self._source.tendon_material[
                ids[base_indices]
            ]

        if len(actuator_indices):
            palette = self._overlay.fill_actuator_palette(frame)
            start = base_count
            stop = start + len(actuator_indices)
            self._capsule_segments[start:stop] = segments[actuator_indices]
            self._capsule_widths[start:stop] = (
                widths[actuator_indices] * self._source.actuator_tendon_scale
            )
            np.take(
                palette,
                segment_actuators[actuator_indices],
                axis=0,
                out=self._capsule_colors[start:stop],
            )
            np.take(
                self._material_values,
                self._source.tendon_material[ids[actuator_indices]],
                axis=0,
                out=self._capsule_materials[start:stop],
                mode="clip",
            )
            self._capsule_material_ids[start:stop] = self._source.tendon_material[
                ids[actuator_indices]
            ]

        np.less(
            self._capsule_colors[:total, 3],
            1.0,
            out=self._capsule_transparent[:total],
        )

        material_table = (
            self._island_tendon_material_table
            if self.get_flag(RenderFlag.ISLAND)
            else self._tendon_material_table
        )
        self._tendons.update(
            self._capsule_segments[:total],
            self._capsule_widths[:total],
            self._capsule_colors[:total],
            self._capsule_materials[:total],
            self._capsule_material_ids[:total],
            self._capsule_transparent[:total],
            material_table,
        )

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
            target["blend"] = OVERDRAW_BLEND
        elif blend == "alpha":
            target["blend"] = ALPHA_BLEND
        elif blend == "additive":
            target["blend"] = ADDITIVE_BLEND
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
        always the shaded fragment entry — mirroring opengl's ReflectPass FBO
        and gl.front_face handling.
        """
        key = ("reflect", blend, cull)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        target: dict = {"format": "rgba16float"}
        if blend == "alpha":
            target["blend"] = ALPHA_BLEND
        elif blend == "additive":
            target["blend"] = ADDITIVE_BLEND
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
                "targets": [
                    {"format": "r32float"},
                    {"format": "r32uint"},
                    {"format": "rg32sint"},
                ],
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
        pose, visual, identity = self.instances.bindings()
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
                    "resource": {"buffer": self.lights.buffer, "offset": 0, "size": LIGHTS_BYTES},
                },
            ],
        )

    def _texture_group(self, name: str | None) -> wgpu.GPUBindGroup:
        view = self.textures.get(name) if name else None
        cube = self.textures.cube(name) if name else None
        view = view if view is not None else self.textures.white
        cube_view = cube[0] if cube is not None else self.textures.white_cube
        key = (id(view), id(cube_view))
        group = self._texture_groups.get(key)
        if group is None:
            group = self.device.create_bind_group(
                layout=self._group1_layout,
                entries=[
                    {"binding": 0, "resource": view},
                    {"binding": 1, "resource": self.textures.sampler},
                    {"binding": 2, "resource": cube_view},
                    {"binding": 3, "resource": self.textures.cube_sampler},
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
        classic = bool(
            self._scene is not None and self._scene.shading_model is ShadingModel.MUJOCO_CLASSIC
        )
        f["flags"][:] = (
            1.0 if cam.orthographic else 0.0,
            linear_out,
            1.0 if classic else 0.0,
            0.0,
        )
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
        scene: RenderScene | None = None,
    ) -> tuple[int, int]:
        # Defaults to the main scene; the tendon pass draws its own scene
        # through the same pipelines with its own instance buffer in group0.
        if scene is None:
            scene = self._scene
        assert scene is not None
        draw_calls = 0
        textured = self.get_flag(RenderFlag.TEXTURE)
        fs_entry = "fs_scene" if reflect else _DEBUG_ENTRIES.get(self._debug_view, "fs_scene")
        # OVERDRAW and WIREFRAME are variants of the shaded pipeline, mirroring
        # opengl's OpaquePass spec selection (opaque.DEBUG_DEFINE/WIREFRAME).
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

    def _encode_export_pass(
        self,
        encoder,
        scene: RenderScene,
        group0,
        cull: str,
        timestamp,
        metric_far: float,
    ) -> tuple[int, int]:
        """Encode the unlit geometry export shared by depth and identity products."""

        target = self.target
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target.export_depth.create_view(),
                    "clear_value": (metric_far, metric_far, metric_far, metric_far),
                    "load_op": "clear",
                    "store_op": "store",
                },
                {
                    "view": target.export_id.create_view(),
                    "clear_value": (0, 0, 0, 0),
                    "load_op": "clear",
                    "store_op": "store",
                },
                {
                    "view": target.export_segmentation.create_view(),
                    "clear_value": (-1, -1, -1, -1),
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
            timestamp_writes=timestamp("export"),
        )
        draw_calls = 0
        instances_drawn = 0
        if scene.count:
            render_pass.set_pipeline(self._export_pipeline(cull))
            render_pass.set_bind_group(0, group0)
            buckets = list(scene.opaque_buckets)
            if self._include_transparent_ids and self.get_flag(RenderFlag.TRANSPARENT):
                buckets += list(scene.transparent_draw_order())
            for bucket in buckets:
                start, stop = scene.bucket_ranges[bucket]
                if stop <= start:
                    continue
                mesh = self.meshes.get(scene.bucket_keys[bucket][0])
                if mesh is None:
                    continue
                render_pass.set_vertex_buffer(0, mesh.vbo)
                render_pass.set_index_buffer(mesh.ibo, "uint32")
                render_pass.draw_indexed(mesh.index_count, stop - start, 0, 0, start)
                draw_calls += 1
                instances_drawn += stop - start
        render_pass.end()
        return draw_calls, instances_drawn

    def _render_export_only(
        self,
        scene: RenderScene,
        plan: WgpuRenderPlan,
        started: float,
    ) -> None:
        """Render depth or identity without touching the shaded color graph."""

        target = self.target
        camera = self._camera.with_aspect(target.width / max(target.height, 1))
        self.instances.upload(scene)
        schedule = self.lights.upload(scene.lights, None)
        self._write_frame_uniforms(camera, 0, (0.0, 0.0), (0.0, 0.0))
        group0 = self._bind_group0()
        cull = "back" if self.get_flag(RenderFlag.CULL_FACE) else "none"
        encoder = self.device.create_command_encoder()
        self.timing.begin_frame()
        draw_calls, instances_drawn = self._encode_export_pass(
            encoder,
            scene,
            group0,
            cull,
            self.timing.timestamp_writes,
            float(camera.far),
        )
        pending_timing = self.timing.resolve(encoder)
        self.device.queue.submit([encoder.finish()])
        self.timing.submitted(pending_timing)

        self.stats.draw_calls = draw_calls
        self.stats.instances = instances_drawn
        self.stats.buckets = scene.bucket_count()
        self.stats.triangles = scene.triangle_count(self.meshes.triangle_counts())
        self.stats.frame_cpu_ms = (time.perf_counter() - started) * 1000.0
        self.stats.gpu_ms = dict(self.timing.gpu_ms)
        self.stats.notes = {
            "scene lights": f"{len(schedule.lights)} active, 0 used by export",
            "shadow casters": "0 active, export-only plan",
            "render products": plan.request.products.name,
            "render passes": "export",
            "instance upload": f"{self.instances.uploaded_bytes} bytes",
            "instance streams": ", ".join(
                f"{name}={count}" for name, count in self.instances.uploaded_streams.items()
            )
            or "unchanged",
        }
        return None

    def render(
        self,
        frame: SceneFrame | None = None,
        request: RenderRequest | None = None,
    ) -> ViewportImage | None:
        if frame is not None:
            self.update(frame)
        scene = self._scene
        if scene is None:
            return None
        self._frame_serial += 1
        if self._hot_reload:
            self._reload_wgsl()

        t0 = time.perf_counter()
        plan = compile_render_plan(request, self._debug_view)
        if not plan.color:
            return self._render_export_only(scene, plan, t0)
        target = self.target
        cam = self._camera.with_aspect(target.width / max(target.height, 1))
        # Plane detection produces a separate identity-stream variant; scene
        # material data remains canonical and is never mutated by a pass.
        reflective = self._reflect.prepare(
            scene,
            cam,
            self._flags,
            target.width,
            target.height,
            frame_serial=self._frame_serial,
            mesh_revision=self.meshes.revision,
            texture_revision=self.textures.revision,
            shader_generation=self._shader_generation,
            selected_id=self._selected,
            debug_view=self._debug_view.value,
        )
        self.instances.upload(scene, self._reflect.reflection_info)
        # Shadow maps render before the scene pass, matching opengl's
        # PASS_ORDER; the same frame's scene pass samples them.
        shadow = self._shadows.prepare(
            scene,
            cam,
            self._flags,
            frame_serial=self._frame_serial,
            mesh_revision=self.meshes.revision,
            shader_generation=self._shader_generation,
        )
        schedule = self.lights.upload(scene.lights, shadow)
        light_count = len(schedule.lights)
        group2, image_gain, image_mip = self._image_light_binding(scene.lights)
        reflection_size = (float(target.width), float(target.height)) if reflective else (0.0, 0.0)
        view_proj = self._write_frame_uniforms(
            cam, light_count, (image_gain, image_mip), reflection_size
        )
        if reflective and not self._reflect.cache_hit:
            self._reflect.write_frames(
                cam, light_count, (image_gain, image_mip), self._pack_frame_block
            )
        group3 = self._shadows.sample_group(shadow)
        group4 = self._reflect.sample_group()

        cull = "back" if self.get_flag(RenderFlag.CULL_FACE) else "none"
        group0 = self._bind_group0()
        tendon_group0 = self._tendons.bind_group0(
            self._group0_layout, target.frame_buffer, self.lights.buffer
        )
        encoder = self.device.create_command_encoder()
        self.timing.begin_frame()
        timestamp = self.timing.timestamp_writes

        draw_calls = 0
        if shadow is not None and not self._shadows.cache_hit:
            draw_calls += self._shadows.execute(
                encoder, scene, self.meshes, self.instances, timestamp
            )
        # Mirrored scene passes run between the shadow pass and the main scene
        # pass, matching opengl's PASS_ORDER (shadow, reflect, opaque, ...).
        if reflective and not self._reflect.cache_hit:
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
                encoder,
                scene,
                self._group0_layout,
                self.instances,
                self.lights,
                draw_reflected,
                timestamp,
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
        # opengl's PASS_ORDER (outline between transparent and present).
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
                timestamp,
            )
        # Upload the debug frame before the main pass so execute() only encodes.
        view = np.asarray(cam.view_matrix(), np.float32)
        proj = proj_matrix_wgpu(cam)
        self._debug.prepare(view, proj, view_proj, target.width, target.height, time.monotonic())
        self._gizmo.prepare(self._gizmo_frame, cam, view, proj, view_proj, target.height)
        pass1 = encoder.begin_render_pass(
            color_attachments=[color_attachment],
            depth_stencil_attachment={
                "view": target.zbuf.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
            timestamp_writes=timestamp("scene"),
        )
        instances_drawn = 0
        if scene.count:
            calls, drawn = self._draw_buckets(
                pass1, group0, group2, group3, group4, scene.opaque_buckets, "opaque", cull
            )
            draw_calls += calls
            instances_drawn += drawn
        # Skybox and horizon haze sit between opaque and transparent, matching
        # opengl's PASS_ORDER (the id pass is the separate export pass here).
        draw_calls += self._skybox.render(
            pass1,
            textures=self.textures,
            flags=self._flags,
            debug_view=self._debug_view,
            camera=cam,
            view_proj=view_proj,
            scene=scene,
        )
        # Tendon capsule chains sit between skybox and transparent geometry,
        # matching opengl's PASS_ORDER (shadow, reflect, opaque, id, skybox,
        # tendon, transparent, ...).
        if tendon_group0 is not None:
            tendon_scene = self._tendons.scene
            # stats.instances mirrors opengl's scene.count, so tendon capsules
            # count as draw calls only.
            calls, _ = self._draw_buckets(
                pass1,
                tendon_group0,
                group2,
                group3,
                group4,
                tendon_scene.opaque_buckets,
                "opaque",
                cull,
                scene=tendon_scene,
            )
            draw_calls += calls
            if tendon_scene.transparent_buckets and self.get_flag(RenderFlag.TRANSPARENT):
                blend = "additive" if self.get_flag(RenderFlag.ADDITIVE) else "alpha"
                # The tendon scene carries the default camera, so its
                # back-to-front bucket order matches opengl's TendonPass.
                order = tendon_scene.transparent_draw_order()
                calls, _ = self._draw_buckets(
                    pass1,
                    tendon_group0,
                    group2,
                    group3,
                    group4,
                    order,
                    blend,
                    cull,
                    scene=tendon_scene,
                )
                draw_calls += calls
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
        # Debug primitives and world text draw after the outline composite,
        # matching opengl's PASS_ORDER (debug between outline and gizmo).
        draw_calls += self._debug.execute(pass1)
        # The gizmo draws last inside the main pass, matching opengl's
        # PASS_ORDER (gizmo between debug and present).
        draw_calls += self._gizmo.execute(pass1)
        pass1.end()

        export_draw_calls = 0
        if plan.export:
            export_draw_calls, _ = self._encode_export_pass(
                encoder, scene, group0, cull, timestamp, float(cam.far)
            )

        # SEGMENT/IDCOLOR rebuild the resolved color from the export ids;
        # other views need no present work (the resolve happened in pass1).
        draw_calls += self._present.execute(
            encoder,
            target.color,
            target.export_id,
            self._debug_view,
            self._selected,
            timestamp,
        )

        pending_timing = self.timing.resolve(encoder)
        self.device.queue.submit([encoder.finish()])
        self.timing.submitted(pending_timing)

        self.stats.draw_calls = draw_calls
        self.stats.instances = instances_drawn
        self.stats.buckets = scene.bucket_count()
        self.stats.triangles = scene.triangle_count(self.meshes.triangle_counts())
        self.stats.frame_cpu_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.gpu_ms = dict(self.timing.gpu_ms)
        # Same report keys as OpenGLBackend._update_light_stats.
        self.stats.notes = {
            "scene lights": (f"{len(schedule.lights)} active, {schedule.deferred_lights} deferred"),
            "shadow casters": (
                f"{schedule.selected_shadow_count} active, {schedule.deferred_shadows} deferred"
            ),
            "render products": plan.request.products.name,
            "render passes": "scene, export" if plan.export else "scene",
            "instance upload": f"{self.instances.uploaded_bytes} bytes",
            "instance streams": ", ".join(
                f"{name}={count}" for name, count in self.instances.uploaded_streams.items()
            )
            or "unchanged",
            "export draw calls": str(export_draw_calls),
            "shadow cache": self._shadows.cache_status,
            "reflection cache": self._reflect.cache_status,
        }
        # The resolved color is display-domain (finish_color gamma-encodes it)
        # and top-row-first, so the viewer presents it without a y flip.
        return ViewportImage(
            texture_id=0,
            width=target.width,
            height=target.height,
            flip_y=False,
            payload=target.color_view,
        )

    # -- misc protocol surface --------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        self.target.resize(width, height)

    def _set_msaa_enabled(self, enabled: bool) -> None:
        samples = self._configured_samples if enabled else 1
        if not self.target.set_samples(samples):
            return
        self._pipelines.clear()
        self._skybox.release()
        self._skybox = SkyboxPass(self.device, self.target.samples)
        self._outline.release()
        self._outline = OutlinePass(self.device, self.target.samples)
        self._debug.set_samples(self.target.samples)
        self._gizmo.set_samples(self.target.samples)
        self.caps = replace(self.caps, msaa_samples=self.target.samples)

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
            self.render(request=RenderRequest.color())
            image = self.target.read_color(flip=True)[..., :3]
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ascontiguousarray(image)).save(path)
            return True
        finally:
            self._camera = saved_camera
            if size is not None:
                self.target.resize(*saved_size)

    def pick(self, x: int, y: int) -> int:
        # Match opengl's GL viewport convention: y measured from the bottom.
        return self.target.read_id(int(x), self.target.height - 1 - int(y))

    def highlight(self, object_id: int) -> None:
        self._selected = int(object_id)

    def set_transparent_id_rendering(self, enabled: bool) -> None:
        self._include_transparent_ids = bool(enabled)

    def set_gizmo(self, gizmo: GizmoFrame | None) -> bool:
        self._gizmo_frame = gizmo
        return True

    def configure_text(
        self,
        primary: str = "",
        primary_index: int = 0,
        fallback: str = "",
        fallback_index: int = 0,
        size_px: float = 14.0,
    ) -> None:
        self._debug.configure_text(primary, primary_index, fallback, fallback_index, size_px)

    def set_flag(self, flag: RenderFlag, value: bool) -> bool:
        if flag not in self.caps.render_flags:
            return False
        if flag is RenderFlag.MSAA and value and self._configured_samples == 1:
            return False
        self._flags[flag] = bool(value)
        if flag is RenderFlag.MSAA:
            self._set_msaa_enabled(bool(value))
        if flag in {
            RenderFlag.STATIC,
            RenderFlag.SKIN,
            RenderFlag.FLEXFACE,
            RenderFlag.FLEXSKIN,
            RenderFlag.ISLAND,
            RenderFlag.CONVEXHULL,
        }:
            self._sync_instance_visibility()
        return True

    def _sync_instance_visibility(self) -> None:
        if self._builder is None:
            return
        changed = self._builder.set_visual_options(
            static=self.get_flag(RenderFlag.STATIC),
            skin=self.get_flag(RenderFlag.SKIN),
            flex_face=self.get_flag(RenderFlag.FLEXFACE),
            flex_skin=self.get_flag(RenderFlag.FLEXSKIN),
            island=self.get_flag(RenderFlag.ISLAND),
            convex_hull=self.get_flag(RenderFlag.CONVEXHULL),
        )
        if changed:
            self.set_render_scene(self._builder.scene)

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
        if mode not in self.caps.label_modes:
            return False
        self._label_mode = mode
        return True

    def get_label_mode(self) -> LabelMode:
        return self._label_mode

    def set_frame_mode(self, mode: FrameMode) -> bool:
        if mode not in self.caps.frame_modes:
            return False
        self._frame_mode = mode
        return True

    def get_frame_mode(self) -> FrameMode:
        return self._frame_mode

    def set_bvh_depth(self, depth: int) -> bool:
        self._bvh_depth = max(int(depth), 0)
        return True

    def get_bvh_depth(self) -> int:
        return self._bvh_depth

    def render_options(self) -> tuple[RenderFlag, ...]:
        return tuple(sorted(self.caps.render_flags, key=lambda flag: flag.value))

    def enable_hot_reload(self, on: bool = True) -> None:
        self._hot_reload = bool(on)

    def _reload_wgsl(self) -> None:
        """Rebuild the scene shader module when its sources changed on disk.

        Mirrors opengl's ``ProgramCache.reload_changed``: a failed compile
        keeps the previous module and pipelines, and the new mtimes are
        recorded either way so one broken edit logs once instead of every
        frame.  wgpu-py raises ``GPUValidationError`` synchronously from
        ``create_shader_module``; pipelines bake in the module, so a
        successful reload drops the cache and the next draw rebuilds them.
        """
        watch = self._shader_watch
        if not watch.changed():
            return
        try:
            code = load_wgsl(*watch.names)
            module = self.device.create_shader_module(code=code)
        except Exception as e:
            msg = str(e)
            if msg != self._shader_reload_error:
                self._shader_reload_error = msg
                log.error("Scene shaders failed to compile; keeping the previous version:\n{}", msg)
        else:
            self._module = module
            self._scene_shader_code = code
            self._pipelines.clear()
            self._shader_generation += 1
            self._shader_reload_error = ""
            log.info("Scene shaders reloaded")
        watch.mark()

    def release(self) -> None:
        self.timing.release()
        self.meshes.release()
        self.textures.release()
        self.instances.release()
        self.lights.release()
        self.target.release()
        self._skybox.release()
        self._outline.release()
        self._present.release()
        self._tendons.release()
        self._shadows.release()
        self._reflect.release()
        self._debug.release()
        self._gizmo.release()
        self._pipelines.clear()
        self._texture_groups.clear()
        self._image_light_groups.clear()
        self._scene = None
        self._builder = None
        self._source = None

    def describe(self) -> str:
        caps = self.caps
        lines = [
            f"wgpu  {caps.gl_version}  ({caps.renderer})",
            f"  MSAA              : {caps.msaa_samples}×",
            f"  ID buffer MSAA    : {caps.id_msaa}",
            f"  GPU timing        : {caps.gpu_timing}",
        ]
        lines.extend(f"  · {note}" for note in caps.notes)
        return "\n".join(lines)
