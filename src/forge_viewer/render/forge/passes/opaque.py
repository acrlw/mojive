"""Opaque geometry render pass."""

from __future__ import annotations

import moderngl
import numpy as np

from ....log import get_logger
from ....types import LightSet, LightType, ShadingModel
from ...backend import DebugView, RenderFlag
from .. import color
from ..programs import ProgramSpec, UniformCache
from ..registry import register_pass
from ..targets import IdLayout
from .base import (
    MAX_SCENE_LIGHTS,
    BasePass,
    PassContext,
    schedule_lights,
    state_opaque,
    state_overdraw,
)

log = get_logger("opaque")

LIGHT_BLOCK_BINDING = 0


DEBUG_DEFINE = {
    DebugView.SHADED: 0,
    DebugView.ALBEDO: 1,
    DebugView.NORMAL: 2,
    DebugView.DEPTH: 3,
    DebugView.OVERDRAW: 4,
    DebugView.WIREFRAME: 0,
}


OVERDRAW_CLEAR = (0.0, 0.0, 0.0, 1.0)


HIGHLIGHT_COLOR = (1.0, 0.82, 0.45)
HIGHLIGHT_MIX = 0.35
HIGHLIGHT_EMISSION = 0.35


WIRE_COLOR = (0.10, 0.10, 0.12)
WIRE_WIDTH = 1.2

TEX_ALBEDO = 0
TEX_CUBE_ALBEDO = 8
TEX_REFLECTIONS = (2, 4, 5, 6)
TEX_IMAGE_LIGHT = 7
IMAGE_LIGHT_REFERENCE_INTENSITY = 5000.0

NO_CLIP = (0.0, 0.0, 0.0, 1.0)


_MASK_ON = (True, True, True, True)
_MASK_OFF = (False, False, False, False)


def draw_buckets(ctx: PassContext, buckets) -> None:
    scene, store = ctx.scene, ctx.instances
    textured = ctx.flag(RenderFlag.TEXTURE)
    white = ctx.textures.white
    white_cube = ctx.textures.white_cube
    for b in buckets:
        key = scene.bucket_keys[b]
        mat = scene.materials[key[1]] if key[1] < len(scene.materials) else None
        tex = ctx.textures.get(mat.texture) if (textured and mat is not None) else None

        (tex if isinstance(tex, moderngl.Texture) else white).use(TEX_ALBEDO)
        (tex if isinstance(tex, moderngl.TextureCube) else white_cube).use(TEX_CUBE_ALBEDO)
        drawn = store.draw(b)
        if drawn:
            ctx.instance_count += drawn
            ctx.draw_calls += 1


class OpaquePass(BasePass):
    name = "opaque"

    def __init__(self) -> None:
        self.program: moderngl.Program | None = None
        self._spec_used: ProgramSpec | None = None

        self._uniforms: UniformCache | None = None
        self._vao_program: moderngl.Program | None = None
        self._shadow_include: bool | None = None
        self._shadow_checked = -1
        self._light_buffer: moderngl.Buffer | None = None

        self._m = np.zeros((4, 4), np.float32)
        self._m2 = np.zeros((4, 4), np.float32)
        self._light_block = np.zeros((5, MAX_SCENE_LIGHTS, 4), np.float32)

    def scene_program(self, backend) -> moderngl.Program | None:
        spec = self._spec_used or self._spec(
            backend.debug_view,
            self._wireframe_on(backend.debug_view, backend.get_flag(RenderFlag.WIREFRAME)),
            use_shadow=bool(
                backend.caps.shadows
                and backend.get_flag(RenderFlag.SHADOW)
                and self._has_shadow_include(backend.programs)
            ),
        )
        try:
            return backend.programs.get(spec)
        except Exception as e:
            log.error("Scene shader compilation failed: {}", e)
            return None

    @staticmethod
    def _wireframe_on(view: DebugView, flag: bool) -> bool:
        return view is DebugView.WIREFRAME or bool(flag)

    def _spec(self, view: DebugView, wireframe: bool, use_shadow: bool) -> ProgramSpec:
        defines: dict[str, object] = {"DEBUG_VIEW": DEBUG_DEFINE.get(view, 0)}
        if wireframe:
            defines["WIREFRAME"] = 1
        if use_shadow:
            defines["USE_SHADOW"] = 1
        return ProgramSpec(
            name="scene",
            vertex="scene.vert",
            fragment="scene_shadow.frag" if use_shadow else "scene.frag",
            geometry="wireframe.geom" if wireframe else None,
            defines=defines,
        )

    def _has_shadow_include(self, programs) -> bool:
        if self._shadow_include is None or self._shadow_checked != programs.generation:
            self._shadow_include = (programs.dir / "shadow_sample.glsl").exists()
            self._shadow_checked = programs.generation
        return self._shadow_include

    def prepare(self, ctx: PassContext) -> bool:
        wireframe = self._wireframe_on(ctx.debug_view, ctx.flag(RenderFlag.WIREFRAME, False))
        use_shadow = (
            ctx.shadow.enabled
            and (ctx.shadow.atlas is not None or ctx.shadow.local_count > 0)
            and ctx.flag(RenderFlag.SHADOW)
            and self._has_shadow_include(ctx.programs)
        )
        spec = self._spec(ctx.debug_view, wireframe, use_shadow)
        try:
            prog = ctx.programs.get(spec)
        except Exception as e:
            log.error(
                "Scene shader compilation failed; opaque geometry is skipped this frame: {}", e
            )
            return False

        self.program = prog
        self._spec_used = spec

        ctx.scene_program = prog
        if self._uniforms is None:
            self._uniforms = UniformCache(prog, ctx.programs.generation)
        else:
            self._uniforms.rebind(prog, ctx.programs.generation)

        self._ensure_vaos(ctx, prog)
        self._bind_light_block(ctx, prog)
        self._frame_uniforms(ctx, prog, wireframe, use_shadow)

        return True

    def _bind_light_block(self, ctx: PassContext, prog: moderngl.Program) -> None:
        block = prog.get("ForgeLights", None)
        if block is None:
            return
        if self._light_buffer is None:
            self._light_buffer = ctx.ctx.buffer(reserve=self._light_block.nbytes)
        block.binding = LIGHT_BLOCK_BINDING
        self._light_buffer.bind_to_uniform_block(LIGHT_BLOCK_BINDING)

    def _ensure_vaos(self, ctx: PassContext, prog: moderngl.Program) -> None:
        if self._vao_program is prog:
            return
        ctx.instances.rebuild(ctx.scene, prog, ctx.meshes, ctx.programs.generation)

        ctx.instances.upload(ctx.scene)
        self._vao_program = prog

    def _frame_uniforms(
        self, ctx: PassContext, prog: moderngl.Program, wireframe: bool, use_shadow: bool
    ) -> None:
        u = self._uniforms
        assert u is not None

        view, view_proj, eye = self.view_matrices(ctx)
        self._write_mat(prog, "u_view_proj", view_proj, self._m)
        self._write_mat(prog, "u_view", view, self._m2)

        cam = ctx.camera

        self._set_direct(prog, "u_camera_pos", tuple(float(x) for x in eye))
        self._set_direct(prog, "u_camera_dir", tuple(float(v) for v in cam.forward()))
        u.set("u_selected_id", int(ctx.selected_id))
        u.set("u_texture", TEX_ALBEDO)
        u.set("u_cube_texture", TEX_CUBE_ALBEDO)
        u.set(
            "u_classic_lighting",
            1 if ctx.scene.shading_model is ShadingModel.MUJOCO_CLASSIC else 0,
        )
        u.set("u_exposure", color.EXPOSURE)
        u.set("u_tonemap", 1 if ctx.flag(RenderFlag.TONEMAP) else 0)
        u.set("u_depth_range", (float(cam.near), float(cam.far)))
        u.set("u_highlight_color", HIGHLIGHT_COLOR)
        u.set("u_highlight", (HIGHLIGHT_MIX, HIGHLIGHT_EMISSION))
        self._set_direct(prog, "u_clip_plane", self.clip_plane(ctx))
        self._set_direct(prog, "u_linear_out", 1 if self.linear_out else 0)
        self.reflection_uniforms(ctx, prog)
        if wireframe:
            u.set("u_wire_color", WIRE_COLOR)
            u.set("u_wire_width", WIRE_WIDTH)

        self._light_uniforms(ctx, prog, u, ctx.scene.lights)
        self._atmosphere_uniforms(ctx, u, ctx.scene.lights)
        if use_shadow:
            self._shadow_uniforms(ctx, u)

    linear_out = False

    def view_matrices(self, ctx: PassContext):
        return ctx.view, ctx.view_proj, ctx.camera.eye

    def clip_plane(self, ctx: PassContext) -> tuple[float, float, float, float]:
        return NO_CLIP

    def reflection_uniforms(self, ctx: PassContext, prog: moderngl.Program) -> None:
        textures = () if self.linear_out else tuple(ctx.reflection or ())
        for index, unit in enumerate(TEX_REFLECTIONS):
            texture = textures[index] if index < len(textures) else ctx.textures.white
            texture.use(unit)
            self._set_direct(prog, f"u_reflection{index}", unit)
        size = (float(textures[0].width), float(textures[0].height)) if textures else (0.0, 0.0)
        self._set_direct(prog, "u_reflection_size", size)

    @staticmethod
    def _set_direct(prog: moderngl.Program, name: str, value) -> None:
        member = prog.get(name, None)
        if member is not None:
            member.value = value

    @staticmethod
    def _write_mat(prog: moderngl.Program, name: str, m: np.ndarray, scratch: np.ndarray) -> None:
        if name in prog:
            np.copyto(scratch, m.T)
            prog[name].write(scratch)

    def _light_uniforms(
        self, ctx: PassContext, prog: moderngl.Program, u: UniformCache, lights: LightSet
    ) -> None:
        pos, dirs, diff, spec, atten = self._light_block
        n = 0
        for light in schedule_lights(lights).lights:
            d = np.asarray(light.direction, np.float64)
            norm = float(np.linalg.norm(d))
            pos[n, :3] = light.position
            pos[n, 3] = float(int(light.type))
            dirs[n, :3] = d / norm if norm > 1e-9 else (0.0, 0.0, -1.0)
            dirs[n, 3] = np.cos(np.deg2rad(min(max(light.cutoff, 0.0), 180.0)))

            diff[n, :3] = color.srgb_to_linear(light.diffuse)
            diff[n, 3] = float(light.exponent)
            spec[n, :3] = color.srgb_to_linear(light.specular)
            atten[n, :3] = light.attenuation
            atten[n, 3] = float(light.range)
            n += 1

        u.set("u_light_count", n)
        if n and self._light_buffer is not None:
            self._light_buffer.write(self._light_block)

        u.set("u_ambient", tuple(float(v) for v in lights.ambient))
        image_light = next(
            (
                light
                for light in reversed(lights.lights)
                if light.active and light.type is LightType.IMAGE
            ),
            None,
        )
        image_texture = ctx.textures.get(image_light.texture) if image_light is not None else None
        if not isinstance(image_texture, moderngl.TextureCube):
            image_texture = ctx.textures.black_cube
        image_texture.use(TEX_IMAGE_LIGHT)
        u.set("u_image_light_texture", TEX_IMAGE_LIGHT)
        u.set(
            "u_image_light",
            (
                max(float(image_light.intensity), 0.0) / IMAGE_LIGHT_REFERENCE_INTENSITY
                if image_light is not None
                else 0.0,
                float(np.floor(np.log2(max(image_texture.size[0], 1)))),
            ),
        )

        hl = lights.headlight
        if hl is not None and hl.active:
            rgb = color.srgb_to_linear(hl.diffuse)
            u.set("u_headlight_diffuse", (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0))
            s = color.srgb_to_linear(hl.specular)
            u.set("u_headlight_specular", (float(s[0]), float(s[1]), float(s[2])))
        else:
            u.set("u_headlight_diffuse", (0.0, 0.0, 0.0, 0.0))

        u.set("u_shadow_light", self._shadow_light_index(lights) if ctx.shadow.enabled else -1)

    @staticmethod
    def _shadow_light_index(lights: LightSet) -> int:
        n = 0
        for light in lights.lights:
            if not light.active:
                continue
            if light.type is LightType.IMAGE:
                continue
            if light.cast_shadow and light.type is LightType.DIRECTIONAL:
                return n
            n += 1
        return -1

    @staticmethod
    def _atmosphere_uniforms(ctx: PassContext, u: UniformCache, lights: LightSet) -> None:
        fog_on = ctx.flag(RenderFlag.FOG, False) and lights.fog_end > lights.fog_start
        haze = (
            lights.haze_density
            if ctx.flag(RenderFlag.HAZE, False) and not lights.horizon_haze
            else 0.0
        )
        u.set(
            "u_fog",
            (
                float(lights.fog_start),
                float(lights.fog_end),
                1.0 if fog_on else 0.0,
                float(haze),
            ),
        )
        u.set("u_fog_color", tuple(float(v) for v in lights.fog_color))
        u.set("u_haze_color", tuple(float(v) for v in lights.haze_color))

    @staticmethod
    def _shadow_uniforms(ctx: PassContext, u: UniformCache) -> None:
        from .shadow import bind_shadow_uniforms

        bind_shadow_uniforms(u, ctx.shadow)

    def execute(self, ctx: PassContext) -> None:
        target, gl = ctx.target, ctx.ctx
        overdraw = ctx.debug_view is DebugView.OVERDRAW

        target.use_main()
        target.clear_main(OVERDRAW_CLEAR if overdraw else ctx.background)

        if target.id_layout is IdLayout.SHARED:
            target.fbo.color_mask = (_MASK_ON, _MASK_OFF)

        if overdraw:
            state_overdraw(gl)
        else:
            state_opaque(gl)
        if not ctx.flag(RenderFlag.CULL_FACE):
            gl.disable(moderngl.CULL_FACE)

        gl.multisample = bool(ctx.flag(RenderFlag.MSAA))

        draw_buckets(ctx, ctx.scene.opaque_buckets)

        if target.id_layout is IdLayout.SHARED:
            target.fbo.color_mask = (_MASK_ON, _MASK_ON)

    def release(self) -> None:
        if self._light_buffer is not None:
            self._light_buffer.release()
        self._light_buffer = None
        self.program = None
        self._spec_used = None
        self._uniforms = None
        self._vao_program = None


register_pass("opaque", OpaquePass)
