"""Directional and local light shadow-map render pass for the webgpu backend.

Port of ``render.forge.passes.shadow.ShadowPass``.  WebGPU translations:

- the 4096² depth atlas is a ``depth32float`` texture; the three cascade
  tiles render inside one depth-only pass with per-tile viewports (no FBO or
  attachment juggling — the sampling side reads it as ``texture_depth_2d``
  with manual comparison, mirroring forge's NEAREST + manual PCF);
- the local distance map is an ``r16float`` 2D array (8 slots × 6 layers);
  per-layer 2D views are native WebGPU attachments, replacing
  ``glFramebufferTextureLayer``; MIN blending replaces the depth buffer so
  the nearest occluder distance survives, like forge's blend equation;
- 1×1 fallback textures keep the scene bind group valid when a map family is
  inactive, mirroring forge's local-array fallback;
- resource creation failure degrades to "no shadows this frame", mirroring
  forge's ``_ensure_atlas``/``_ensure_local`` error paths.

Per-draw matrices ride one uniform buffer at 256-byte dynamic offsets
(3 cascade tiles + up to 48 local layers), packed once per frame in
``prepare`` — WebGPU queue writes are submit-ordered, so per-draw
``write_buffer`` calls inside the encoder would not be seen per draw.
"""

from __future__ import annotations

import numpy as np
import wgpu

from .... import math3d as M
from ....log import get_logger
from ....types import CameraView, Light, LightKind
from ...backend import RenderFlag
from ...scene import RenderScene
from ..cascades import ATLAS_SIZE, CascadeSet, build_cascades, slot_pixels
from ..lighting import LOCAL_SHADOW_SLOTS, LightSchedule, ShadowState, schedule_lights
from ..meshes import MeshStore
from ..programs import load_wgsl
from ..targets import perspective_wgpu

log = get_logger("shadow")

LOCAL_PIXELS = 1024

# One uniform block per draw: 3 cascade tiles + 8 slots × 6 cube faces.
MAX_DRAWS = 3 + LOCAL_SHADOW_SLOTS * 6
UNIFORM_STRIDE = 256  # minUniformBufferOffsetAlignment

SHADOW_DRAW_DTYPE = np.dtype(
    {
        "names": ["view_proj", "light"],
        "formats": ["(4,4)f4", "(4,)f4"],
        "offsets": [0, 64],
        "itemsize": UNIFORM_STRIDE,
    }
)

# Same vbo as the scene pass; only the position attribute is consumed.
_POSITION_LAYOUT = {
    "array_stride": 32,
    "step_mode": "vertex",
    "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0}],
}

# moderngl blend_func(ONE, ONE) + blend_equation(MIN), matching forge _draw_locals.
_MIN_BLEND = {
    "color": {"operation": "min", "src_factor": "one", "dst_factor": "one"},
    "alpha": {"operation": "min", "src_factor": "one", "dst_factor": "one"},
}

_LOCAL_LAYERS = LOCAL_SHADOW_SLOTS * 6


def _draw_opaque(pass_encoder: wgpu.GPURenderPassEncoder, scene: RenderScene, meshes) -> int:
    """Draw all opaque buckets with the bound pipeline; returns draw calls."""
    calls = 0
    for b in scene.opaque_buckets:
        start, stop = scene.bucket_ranges[b]
        if stop <= start:
            continue
        mesh = meshes.get(scene.bucket_keys[b][0])
        if mesh is None:
            continue
        pass_encoder.set_vertex_buffer(0, mesh.vbo)
        pass_encoder.set_index_buffer(mesh.ibo, "uint32")
        pass_encoder.draw_indexed(mesh.index_count, stop - start, 0, 0, start)
        calls += 1
    return calls


class ShadowPass:
    """Renders the cascade atlas and the local distance maps for one frame."""

    name = "shadow"

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self.bind_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d-array"},
                },
            ]
        )
        self._draw_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": True,
                        "min_binding_size": 80,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": "read-only-storage"},
                },
            ]
        )
        self._draw_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._draw_layout]
        )
        self._uniforms = device.create_buffer(
            size=MAX_DRAWS * UNIFORM_STRIDE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._arena = np.zeros(MAX_DRAWS, SHADOW_DRAW_DTYPE)

        self._depth_pipeline: wgpu.GPURenderPipeline | None = None
        self._dist_pipeline: wgpu.GPURenderPipeline | None = None
        self.atlas: wgpu.GPUTexture | None = None
        self._atlas_view: wgpu.GPUTextureView | None = None
        self._local_tex: wgpu.GPUTexture | None = None
        self._local_view: wgpu.GPUTextureView | None = None
        self._layer_views: list[wgpu.GPUTextureView] = []
        self._atlas_fallback: wgpu.GPUTextureView | None = None
        self._local_fallback: wgpu.GPUTextureView | None = None

        self._cascades = CascadeSet()
        self._local_kinds = np.zeros(LOCAL_SHADOW_SLOTS, np.int32)
        self._point_matrices = np.zeros((LOCAL_SHADOW_SLOTS, 6, 4, 4), np.float32)
        self._state: ShadowState | None = None
        self._draw_groups: dict[int, wgpu.GPUBindGroup] = {}
        self._sample_groups: dict[tuple[int, int], wgpu.GPUBindGroup] = {}
        self._failed = ""

    # -- lazy resources ---------------------------------------------------------

    def _ensure_pipelines(self) -> bool:
        if self._depth_pipeline is not None and self._dist_pipeline is not None:
            return True
        try:
            module_shadow = self._device.create_shader_module(code=load_wgsl("shadow.wgsl"))
            module_dist = self._device.create_shader_module(code=load_wgsl("spot_dist.wgsl"))
            self._depth_pipeline = self._device.create_render_pipeline(
                layout=self._draw_pipeline_layout,
                vertex={
                    "module": module_shadow,
                    "entry_point": "vs_shadow",
                    "buffers": [_POSITION_LAYOUT],
                },
                primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "back"},
                depth_stencil={
                    "format": "depth32float",
                    "depth_write_enabled": True,
                    "depth_compare": "less",
                },
                multisample={"count": 1},
            )
            self._dist_pipeline = self._device.create_render_pipeline(
                layout=self._draw_pipeline_layout,
                vertex={
                    "module": module_dist,
                    "entry_point": "vs_dist",
                    "buffers": [_POSITION_LAYOUT],
                },
                fragment={
                    "module": module_dist,
                    "entry_point": "fs_dist",
                    "targets": [{"format": "r16float", "blend": _MIN_BLEND}],
                },
                primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "back"},
                multisample={"count": 1},
            )
        except Exception as e:
            if self._failed != str(e):
                self._failed = str(e)
                log.error("Shadow shader compilation failed; shadows are skipped this frame: {}", e)
            return False
        return True

    def _ensure_atlas(self) -> bool:
        if self.atlas is not None:
            return True
        try:
            tex = self._device.create_texture(
                size=(ATLAS_SIZE, ATLAS_SIZE, 1),
                format="depth32float",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
        except Exception as e:
            if self._failed != str(e):
                self._failed = str(e)
                log.error(
                    "Could not create the {}x{} depth atlas; shadows are skipped: {}",
                    ATLAS_SIZE,
                    ATLAS_SIZE,
                    e,
                )
            return False
        self.atlas = tex
        self._atlas_view = tex.create_view()
        return True

    def _ensure_local(self) -> bool:
        if self._local_tex is not None:
            return True
        try:
            tex = self._device.create_texture(
                size=(LOCAL_PIXELS, LOCAL_PIXELS, _LOCAL_LAYERS),
                format="r16float",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
        except Exception as e:
            if self._failed != str(e):
                self._failed = str(e)
                log.error("Could not create the local shadow array; local shadows are off: {}", e)
            return False
        self._local_tex = tex
        self._local_view = tex.create_view(dimension="2d-array")
        self._layer_views = [
            tex.create_view(dimension="2d", base_array_layer=layer, array_layer_count=1)
            for layer in range(_LOCAL_LAYERS)
        ]
        return True

    def _fallback_atlas_view(self) -> wgpu.GPUTextureView:
        if self._atlas_fallback is None:
            tex = self._device.create_texture(
                size=(1, 1, 1), format="depth32float", usage=wgpu.TextureUsage.TEXTURE_BINDING
            )
            self._atlas_fallback = tex.create_view()
        return self._atlas_fallback

    def _fallback_local_view(self) -> wgpu.GPUTextureView:
        if self._local_fallback is None:
            tex = self._device.create_texture(
                size=(1, 1, 1), format="r16float", usage=wgpu.TextureUsage.TEXTURE_BINDING
            )
            self._local_fallback = tex.create_view(dimension="2d-array")
        return self._local_fallback

    # -- per-frame scheduling -----------------------------------------------------

    def prepare(
        self, scene: RenderScene, camera: CameraView, flags: dict[RenderFlag, bool]
    ) -> ShadowState | None:
        """Select casters and build matrices; None means no shadow pass this frame."""
        if not flags.get(RenderFlag.SHADOW, True):
            return None
        if not scene.opaque_buckets:
            return None
        schedule = schedule_lights(scene.lights)
        sun = (
            schedule.lights[schedule.directional_shadow]
            if schedule.directional_shadow >= 0
            else None
        )
        state = ShadowState()
        local_count = self._prepare_locals(scene, schedule, state)
        if sun is None and local_count == 0:
            return None
        if not self._ensure_pipelines():
            return None
        if sun is not None and not self._ensure_atlas():
            return None
        if local_count and not self._ensure_local():
            state.local_count = local_count = 0
            if sun is None:
                return None

        if sun is not None:
            build_cascades(
                sun.direction,
                camera.target,
                scene.scene_extent,
                scene_center=scene.scene_center,
                shadow_clip=scene.shadow_clip,
                into=self._cascades,
            )
        else:
            self._cascades.count = 0

        c = self._cascades
        state.cascade_count = c.count
        state.shadow_light = schedule.directional_shadow
        state.matrices[:] = c.matrices
        state.splits[:] = c.splits
        state.texel_world[:] = c.texel_world
        state.tile_uv[:] = c.tile_uv
        state.enabled = True
        self._state = state
        self._write_draw_uniforms(state)
        return state

    def _prepare_locals(
        self, scene: RenderScene, schedule: LightSchedule, state: ShadowState
    ) -> int:
        for packed_index in schedule.local_shadows:
            light = schedule.lights[packed_index]
            slot = state.local_count
            state.local_light_indices[slot] = packed_index
            self._local_kinds[slot] = int(light.kind)
            state.local_positions[slot, :3] = light.position
            state.local_positions[slot, 3] = light.range
            state.local_radius[slot] = light.area_radius
            if light.kind is LightKind.SPOT:
                self._prepare_spot(scene, light, slot, state)
            else:
                self._prepare_point(scene, light, slot, state)
            state.local_count += 1
        return state.local_count

    def _prepare_spot(
        self, scene: RenderScene, light: Light, slot: int, state: ShadowState
    ) -> None:
        pos = np.asarray(light.position, np.float64)

        fov = float(np.deg2rad(2.0 * min(max(light.cutoff, 1.0), 89.0)))
        extent = float(scene.scene_extent)
        near = max(extent * 0.02, 1e-3)
        far = light.range if light.range > near else extent * 6.0
        target = pos + np.asarray(light.direction, np.float64) * extent
        view = M.look_at(pos, target, np.array([0.0, 0.0, 1.0]))
        proj = perspective_wgpu(fov, 1.0, near, far)
        np.copyto(state.local_matrices[slot], proj.astype(np.float64) @ view.astype(np.float64))
        state.local_texel[slot] = 2.0 * float(np.tan(fov * 0.5)) / LOCAL_PIXELS

    def _prepare_point(
        self, scene: RenderScene, light: Light, slot: int, state: ShadowState
    ) -> None:
        state.local_texel[slot] = 2.0 / LOCAL_PIXELS

        pos = np.asarray(light.position, np.float64)
        extent = float(scene.scene_extent)
        near = max(extent * 0.02, 1e-3)
        scene_far = float(np.linalg.norm(pos - scene.scene_center)) + 2.0 * extent
        far = light.range if light.range > near else max(scene_far, near * 2.0)
        proj = perspective_wgpu(np.pi * 0.5, 1.0, near, far)
        faces = (
            ((1, 0, 0), (0, -1, 0)),
            ((-1, 0, 0), (0, -1, 0)),
            ((0, 1, 0), (0, 0, 1)),
            ((0, -1, 0), (0, 0, -1)),
            ((0, 0, 1), (0, -1, 0)),
            ((0, 0, -1), (0, -1, 0)),
        )
        for i, (direction, up) in enumerate(faces):
            view = M.look_at(pos, pos + direction, up)
            np.copyto(
                self._point_matrices[slot, i],
                proj.astype(np.float64) @ view.astype(np.float64),
            )

    def _write_draw_uniforms(self, state: ShadowState) -> None:
        """Pack one uniform block per draw (tiles first, then local layers)."""
        arena = self._arena
        n = 0
        for i in range(state.cascade_count):
            arena["view_proj"][n] = self._cascades.matrices[i].T
            arena["light"][n] = (0.0, 0.0, 0.0, 0.0)
            n += 1
        for slot in range(state.local_count):
            pos_range = state.local_positions[slot]
            if self._local_kinds[slot] == int(LightKind.SPOT):
                arena["view_proj"][n] = state.local_matrices[slot].T
                arena["light"][n] = pos_range
                n += 1
            else:
                for face in range(6):
                    arena["view_proj"][n] = self._point_matrices[slot, face].T
                    arena["light"][n] = pos_range
                    n += 1
        self._device.queue.write_buffer(self._uniforms, 0, arena[:n].tobytes())

    # -- rendering ----------------------------------------------------------------

    def execute(
        self,
        encoder: wgpu.GPUCommandEncoder,
        scene: RenderScene,
        meshes: MeshStore,
        instances,
    ) -> int:
        """Encode the shadow passes; returns the draw-call count."""
        state = self._state
        assert state is not None
        calls = 0
        draw = 0
        group0 = self._draw_group(instances)

        if state.cascade_count:
            depth_pass = encoder.begin_render_pass(
                color_attachments=[],
                depth_stencil_attachment={
                    "view": self._atlas_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": "clear",
                    "depth_store_op": "store",
                },
            )
            depth_pass.set_pipeline(self._depth_pipeline)
            for i in range(state.cascade_count):
                x, y, w, h = slot_pixels(self._cascades.slots[i])
                depth_pass.set_viewport(float(x), float(y), float(w), float(h), 0.0, 1.0)
                depth_pass.set_bind_group(0, group0, [draw * UNIFORM_STRIDE])
                calls += _draw_opaque(depth_pass, scene, meshes)
                draw += 1
            depth_pass.end()

        for slot in range(state.local_count):
            if self._local_kinds[slot] == int(LightKind.SPOT):
                layers = (slot * 6,)
            else:
                layers = tuple(slot * 6 + face for face in range(6))
            for layer in layers:
                layer_pass = encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": self._layer_views[layer],
                            # f16 max, mirroring forge's 65504.0 distance clear
                            "clear_value": (65504.0, 0.0, 0.0, 1.0),
                            "load_op": "clear",
                            "store_op": "store",
                        }
                    ]
                )
                layer_pass.set_pipeline(self._dist_pipeline)
                layer_pass.set_bind_group(0, group0, [draw * UNIFORM_STRIDE])
                calls += _draw_opaque(layer_pass, scene, meshes)
                draw += 1
                layer_pass.end()
        return calls

    def _draw_group(self, instances) -> wgpu.GPUBindGroup:
        key = id(instances.buffer)
        group = self._draw_groups.get(key)
        if group is None:
            group = self._device.create_bind_group(
                layout=self._draw_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._uniforms,
                            "offset": 0,
                            "size": UNIFORM_STRIDE,
                        },
                    },
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": instances.buffer,
                            "offset": 0,
                            "size": instances.capacity * 128,
                        },
                    },
                ],
            )
            self._draw_groups[key] = group
        return group

    def sample_group(self, state: ShadowState | None) -> wgpu.GPUBindGroup:
        """Scene-pass bind group: live maps when active, 1×1 fallbacks otherwise."""
        has_cascades = state is not None and state.cascade_count > 0
        has_local = state is not None and state.local_count > 0
        atlas_view = self._atlas_view if has_cascades else self._fallback_atlas_view()
        local_view = self._local_view if has_local else self._fallback_local_view()
        key = (id(atlas_view), id(local_view))
        group = self._sample_groups.get(key)
        if group is None:
            group = self._device.create_bind_group(
                layout=self.bind_layout,
                entries=[
                    {"binding": 0, "resource": atlas_view},
                    {"binding": 1, "resource": local_view},
                ],
            )
            self._sample_groups[key] = group
        return group

    def release(self) -> None:
        for tex in (self.atlas, self._local_tex):
            if tex is not None:
                tex.destroy()
        self._uniforms.destroy()
        self._draw_groups.clear()
        self._sample_groups.clear()
        self.atlas = None
        self._atlas_view = None
        self._local_tex = None
        self._local_view = None
        self._layer_views = []
        self._atlas_fallback = None
        self._local_fallback = None
        self._depth_pipeline = None
        self._dist_pipeline = None
        self._state = None
