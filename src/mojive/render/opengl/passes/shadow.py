"""Directional and local light shadow-map render pass."""

from __future__ import annotations

import time

import moderngl
import numpy as np

from .... import math3d as M
from ....log import get_logger
from ....types import Light, LightType, ShadingModel
from ...backend import RenderFlag
from ...dependencies import camera_key, lifecycle_key, lights_key
from ...scene import RenderScene
from .. import gl_native as G
from ..cascades import (
    ATLAS_SIZE,
    CascadeSet,
    build_cascades,
    slot_pixels,
)
from ..programs import ProgramSpec
from ..registry import register_pass
from .base import (
    LOCAL_SHADOW_SLOTS,
    MAX_SCENE_LIGHTS,
    BasePass,
    LightSchedule,
    PassContext,
    ShadowResult,
    schedule_lights,
    state_opaque,
)
from .idbuffer import IdGeometry

log = get_logger("shadow")

SHADOW_TEXTURE_UNIT = 1
LOCAL_TEXTURE_UNIT = 3
LOCAL_PIXELS = 1024
SPOT_PIXELS = 2048


SHADOW_BIAS = (1.0, 2.5)


_GL_MATRICES = np.zeros((3, 4, 4), np.float32)


class DistanceGeometry(IdGeometry):
    def _make_spec(self, attachment: int) -> ProgramSpec:
        return ProgramSpec(name="spot_dist", vertex="spot_dist.vert", fragment="spot_dist.frag")

    def set_matrix(self, matrix: np.ndarray, scratch: np.ndarray) -> None:
        assert self.program is not None
        np.copyto(scratch, matrix.T)
        self.program["u_view_proj"].write(scratch)

    def set_light(self, pos, light_range: float) -> None:
        assert self.program is not None
        self.program["u_light_pos"].value = tuple(float(v) for v in pos)
        if "u_light_range" in self.program:
            self.program["u_light_range"].value = float(light_range)


class ShadowGeometry(IdGeometry):
    def _make_spec(self, attachment: int) -> ProgramSpec:
        return ProgramSpec(name="shadow", vertex="shadow.vert", fragment="shadow.frag")

    def set_matrix(self, matrix: np.ndarray, scratch: np.ndarray) -> None:
        assert self.program is not None
        np.copyto(scratch, matrix.T)
        self.program["u_view_proj"].write(scratch)


class ShadowPass(BasePass):
    name = "shadow"

    def __init__(self) -> None:
        self.atlas: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._geom = ShadowGeometry()
        self._distance_geom = DistanceGeometry()
        self._local_tex: moderngl.TextureArray | None = None
        self._local_fallback: moderngl.TextureArray | None = None
        self._local_fbo: moderngl.Framebuffer | None = None
        self._local_placeholder: moderngl.Texture | None = None
        self._local_pixels = 0
        self._local_layers = 0
        self._point_matrices = np.zeros((LOCAL_SHADOW_SLOTS, 6, 4, 4), np.float32)
        self._cascades = CascadeSet()
        self._scratch = np.zeros((4, 4), np.float32)
        self.cascade_cpu_ms = 0.0
        self.cache_status = "off"
        self._cache_scene: RenderScene | None = None
        self._cache_key: tuple | None = None
        self._pending_key: tuple | None = None
        self._state: ShadowResult | None = None

        self._failed = ""

    def _ensure_local(self, ctx: PassContext, pixels: int, layers: int) -> bool:
        if (
            self._local_tex is not None
            and self._local_pixels == pixels
            and self._local_layers == layers
        ):
            return True
        if not G.native().has_array_layer:
            log.error(
                "This GL entry point cannot attach texture-array layers; local shadows are disabled"
            )
            return False
        self._release_local()
        tex = ctx.ctx.texture_array((pixels, pixels, layers), 1, dtype="f2")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        placeholder = ctx.ctx.texture((pixels, pixels), 1, dtype="f2")
        self._local_tex = tex
        self._local_placeholder = placeholder
        self._local_fbo = ctx.ctx.framebuffer([placeholder])
        self._local_pixels = pixels
        self._local_layers = layers
        return True

    def _release_local(self) -> None:
        for obj in (self._local_fbo, self._local_placeholder, self._local_tex):
            if obj is not None:
                obj.release()
        self._local_fbo = None
        self._local_placeholder = None
        self._local_tex = None
        self._local_pixels = 0
        self._local_layers = 0

    def _ensure_local_fallback(self, ctx: PassContext) -> None:
        if self._local_fallback is None:
            self._local_fallback = ctx.ctx.texture_array((1, 1, 1), 1, dtype="f2")
            self._local_fallback.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _ensure_atlas(self, ctx: PassContext) -> bool:
        if self.atlas is not None:
            return True
        try:
            tex = ctx.ctx.depth_texture((ATLAS_SIZE, ATLAS_SIZE))
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

        tex.compare_func = ""

        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = tex.repeat_y = False
        self.atlas = tex
        self._fbo = ctx.ctx.framebuffer(depth_attachment=tex)
        return True

    def prepare(self, ctx: PassContext) -> bool:
        self.cache_status = "off"
        self._pending_key = None
        if not ctx.flag(RenderFlag.SHADOW):
            return False
        if not ctx.scene.opaque_buckets:
            return False
        key = self._dependency_key(ctx)
        if self._cache_scene is ctx.scene and self._cache_key == key and self._state is not None:
            ctx.shadow = self._state
            self.cache_status = "reused"
            return False

        # Keep only the latest complete product. Invalidating before resource
        # changes prevents a failed reallocation from reviving released maps.
        self._cache_scene = None
        self._cache_key = None
        self._state = None
        self._pending_key = key
        s = ctx.shadow
        self._reset(s)
        schedule = schedule_lights(ctx.scene.lights)
        sun = (
            schedule.lights[schedule.directional_shadow]
            if schedule.directional_shadow >= 0
            else None
        )
        local_pixels = self._local_resolution(schedule)
        local_count, local_layers = self._prepare_locals(ctx, schedule, local_pixels)
        if sun is None and local_count == 0:
            return False
        if sun is not None and not self._ensure_atlas(ctx):
            return False
        if local_count and not self._ensure_local(ctx, local_pixels, local_layers):
            s.local_count = local_count = 0
            if sun is None:
                return False
        if local_count == 0:
            self._ensure_local_fallback(ctx)
        s.local_tex = self._local_tex if local_count else self._local_fallback

        t0 = time.perf_counter()

        if sun is not None:
            build_cascades(
                sun.direction,
                ctx.camera.target,
                ctx.scene.scene_extent,
                scene_center=ctx.scene.scene_center,
                shadow_clip=ctx.scene.shadow_clip,
                into=self._cascades,
            )
        else:
            self._cascades.count = 0
        self.cascade_cpu_ms = (time.perf_counter() - t0) * 1000.0

        try:
            ready = sun is None or self._geom.ensure(ctx, 0)
        except Exception as e:
            if self._failed != str(e):
                self._failed = str(e)
                log.error("Shadow shader compilation failed; shadows are skipped this frame: {}", e)
            return False
        if not ready:
            return False
        if sun is not None:
            self._geom.upload(ctx)

        if local_count:
            ready = self._distance_geom.ensure(ctx, 0)
            if not ready:
                return False
            self._distance_geom.upload(ctx)

        c = self._cascades
        s.atlas = self.atlas
        s.matrices = c.matrices
        s.splits = c.splits
        s.texel_world = c.texel_world
        s.tile_uv = c.tile_uv
        s.cascade_count = c.count
        s.enabled = True
        return True

    @staticmethod
    def _dependency_key(ctx: PassContext) -> tuple:
        scene = ctx.scene
        return (
            lifecycle_key(scene, ctx.frame_serial),
            int(ctx.mesh_revision),
            camera_key(ctx.camera),
            lights_key(scene.lights),
            tuple(float(value) for value in np.asarray(scene.scene_center).flat),
            float(scene.scene_extent),
            float(scene.shadow_clip),
            scene.shading_model.value,
            int(ctx.programs.generation),
        )

    @staticmethod
    def _local_resolution(schedule: LightSchedule) -> int:
        return (
            SPOT_PIXELS
            if schedule.local_shadows
            and all(
                schedule.lights[index].type is LightType.SPOT for index in schedule.local_shadows
            )
            else LOCAL_PIXELS
        )

    def _prepare_locals(
        self, ctx: PassContext, schedule: LightSchedule, pixels: int
    ) -> tuple[int, int]:
        s = ctx.shadow
        layer = 0
        for packed_index in schedule.local_shadows:
            light = schedule.lights[packed_index]
            light_range = (
                0.0
                if ctx.scene.shading_model is ShadingModel.MUJOCO_CLASSIC
                else float(light.range)
            )
            slot = s.local_count
            s.local_light_indices[slot] = packed_index
            s.local_kinds[slot] = int(light.type)
            s.local_layers[slot] = layer
            s.local_positions[slot, :3] = light.position
            s.local_positions[slot, 3] = light_range
            s.local_radius[slot] = light.area_radius
            if light.type is LightType.SPOT:
                self._prepare_spot(ctx, light, light_range, slot, pixels)
                layer += 1
            else:
                self._prepare_point(ctx, light, light_range, slot, pixels)
                layer += 6
            s.local_count += 1
        return s.local_count, layer

    def _prepare_spot(
        self, ctx: PassContext, light: Light, light_range: float, slot: int, pixels: int
    ) -> None:
        s = ctx.shadow

        pos = np.asarray(light.position, np.float64)

        fov = float(np.deg2rad(2.0 * min(max(light.cutoff, 1.0), 89.0)))
        extent = float(ctx.scene.scene_extent)
        near = max(extent * 0.02, 1e-3)
        far = light_range if light_range > near else extent * 6.0
        target = pos + np.asarray(light.direction, np.float64) * extent
        view = M.look_at(pos, target, np.array([0.0, 0.0, 1.0]))
        proj = M.perspective(fov, 1.0, near, far)
        np.copyto(s.local_matrices[slot], proj.astype(np.float64) @ view.astype(np.float64))
        s.local_texel[slot] = 2.0 * float(np.tan(fov * 0.5)) / pixels

    def _prepare_point(
        self, ctx: PassContext, light: Light, light_range: float, slot: int, pixels: int
    ) -> None:
        s = ctx.shadow
        s.local_texel[slot] = 2.0 / pixels

        pos = np.asarray(light.position, np.float64)
        extent = float(ctx.scene.scene_extent)
        near = max(extent * 0.02, 1e-3)
        scene_far = float(np.linalg.norm(pos - ctx.scene.scene_center)) + 2.0 * extent
        far = light_range if light_range > near else max(scene_far, near * 2.0)
        proj = M.perspective(np.pi * 0.5, 1.0, near, far)
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

    @staticmethod
    def _reset(s: ShadowResult) -> None:
        s.atlas = None
        s.cascade_count = 0
        s.enabled = False
        s.local_count = 0
        s.local_light_indices.fill(-1)
        s.local_tex = None

    def execute(self, ctx: PassContext) -> None:
        gl = ctx.ctx
        if ctx.shadow.cascade_count > 0 and self._fbo is not None:
            self._draw_cascades(ctx, gl)
        if ctx.shadow.local_count and self._local_tex is not None:
            self._draw_locals(ctx)
        self._cache_scene = ctx.scene
        self._cache_key = self._pending_key
        self._state = ctx.shadow
        self._pending_key = None
        self.cache_status = "rendered"

    def _draw_cascades(self, ctx: PassContext, gl) -> None:
        fbo = self._fbo
        assert fbo is not None

        fbo.depth_mask = True
        fbo.use()
        fbo.clear(depth=1.0)

        state_opaque(gl)

        gl.multisample = False

        c = self._cascades
        for i in range(c.count):
            gl.viewport = slot_pixels(c.slots[i])
            self._geom.set_matrix(c.matrices[i], self._scratch)
            ctx.draw_calls += self._geom.draw(ctx, ctx.scene.opaque_buckets)

    def _draw_locals(self, ctx: PassContext) -> None:
        gl, fbo = ctx.ctx, self._local_fbo
        assert fbo is not None
        fbo.use()
        gl.viewport = (0, 0, self._local_pixels, self._local_pixels)
        state_opaque(gl)
        gl.multisample = False
        gl.enable(moderngl.BLEND)
        gl.blend_func = (moderngl.ONE, moderngl.ONE)
        gl.blend_equation = moderngl.MIN
        try:
            s = ctx.shadow
            for slot in range(s.local_count):
                pos_range = s.local_positions[slot]
                self._distance_geom.set_light(pos_range[:3], pos_range[3])
                base_layer = int(s.local_layers[slot])
                if s.local_kinds[slot] == int(LightType.SPOT):
                    self._draw_local_layer(ctx, base_layer, s.local_matrices[slot])
                else:
                    for face in range(6):
                        self._draw_local_layer(
                            ctx, base_layer + face, self._point_matrices[slot, face]
                        )
        finally:
            gl.disable(moderngl.BLEND)
            gl.blend_equation = moderngl.FUNC_ADD

    def _draw_local_layer(self, ctx: PassContext, layer: int, matrix: np.ndarray) -> None:
        assert self._local_tex is not None and self._local_fbo is not None
        G.native().attach_array_layer(self._local_tex.glo, layer)
        self._local_fbo.clear(65504.0, 0.0, 0.0, 1.0)
        self._distance_geom.set_matrix(matrix, self._scratch)
        ctx.draw_calls += self._distance_geom.draw(ctx, ctx.scene.opaque_buckets)

    def release(self) -> None:
        self._geom.release()
        self._distance_geom.release()
        self._release_local()
        for obj in (self._fbo, self.atlas, self._local_fallback):
            if obj is not None:
                obj.release()
        self._fbo = None
        self.atlas = None
        self._local_fallback = None
        self._cache_scene = None
        self._cache_key = None
        self._pending_key = None
        self._state = None
        self.cache_status = "off"


def bind_shadow_uniforms(
    program_or_cache, result: ShadowResult, unit: int = SHADOW_TEXTURE_UNIT
) -> bool:
    prog, cache = _resolve(program_or_cache)
    if not result.enabled:
        _set(prog, cache, "u_shadow_count", 0)
        _set(prog, cache, "u_local_count", 0)
        return False

    has_cascades = result.cascade_count > 0 and result.atlas is not None
    has_local = result.local_count > 0 and result.local_tex is not None
    if not has_cascades and not has_local:
        _set(prog, cache, "u_shadow_count", 0)
        _set(prog, cache, "u_local_count", 0)
        return False

    if has_cascades:
        result.atlas.use(int(unit))

        _force(prog, cache, "u_shadow_atlas", int(unit))
    _set(prog, cache, "u_shadow_count", int(result.cascade_count))
    _set(prog, cache, "u_shadow_bias", SHADOW_BIAS)

    _set(prog, cache, "u_local_count", int(result.local_count))
    _force(prog, cache, "u_local_shadow", LOCAL_TEXTURE_UNIT)
    if result.local_tex is not None:
        result.local_tex.use(LOCAL_TEXTURE_UNIT)
    if has_local:
        _LOCAL_SLOT_BY_LIGHT.fill(-1)
        for slot in range(result.local_count):
            index = int(result.local_light_indices[slot])
            if index < len(_LOCAL_SLOT_BY_LIGHT):
                _LOCAL_SLOT_BY_LIGHT[index] = slot
        _write_uniform_array(prog, "u_local_slot", _LOCAL_SLOT_BY_LIGHT)
        _write_uniform_array(prog, "u_local_pos", result.local_positions)
        _write_uniform_array(prog, "u_local_texel", result.local_texel)
        _write_uniform_array(prog, "u_local_radius", result.local_radius)
        _write_uniform_array(prog, "u_local_layer", result.local_layers)
        if "u_local_matrix" in prog:
            k = min(result.local_count, len(_GL_LOCAL_MATRICES))
            np.copyto(_GL_LOCAL_MATRICES[:k], result.local_matrices[:k].transpose(0, 2, 1))
            prog["u_local_matrix"].write(_GL_LOCAL_MATRICES)
    _set(prog, cache, "u_shadow_splits", tuple(float(v) for v in result.splits[:3]))

    _set(prog, cache, "u_shadow_texel", tuple(float(v) for v in result.texel_world[:3]))

    if "u_shadow_matrix" in prog:
        k = min(len(result.matrices), len(_GL_MATRICES))
        np.copyto(_GL_MATRICES[:k], result.matrices[:k].transpose(0, 2, 1))
        prog["u_shadow_matrix"].write(_GL_MATRICES[: max(k, 1)])
    if "u_shadow_tile" in prog and len(result.tile_uv):
        prog["u_shadow_tile"].write(np.ascontiguousarray(result.tile_uv, np.float32))
    return True


def _resolve(target) -> tuple[moderngl.Program, object | None]:
    if isinstance(target, moderngl.Program):
        return target, None
    prog = getattr(target, "_program", None)
    if not isinstance(prog, moderngl.Program):
        raise TypeError(f"bind_shadow_uniforms expects Program or UniformCache, got {type(target)}")
    return prog, target


def _set(prog: moderngl.Program, cache, name: str, value) -> None:
    if cache is not None:
        cache.set(name, value)
    elif name in prog:
        prog[name].value = value


def _force(prog: moderngl.Program, cache, name: str, value) -> None:
    if cache is not None:
        cache.force(name, value)
    elif name in prog:
        prog[name].value = value


def _write_uniform_array(prog: moderngl.Program, name: str, data: np.ndarray) -> None:
    if name not in prog:
        return
    slots = max(int(prog[name].array_length), 1)
    prog[name].write(data[:slots])


_GL_LOCAL_MATRICES = np.zeros((LOCAL_SHADOW_SLOTS, 4, 4), np.float32)
_LOCAL_SLOT_BY_LIGHT = np.full(MAX_SCENE_LIGHTS, -1, np.int32)


register_pass("shadow", ShadowPass)
