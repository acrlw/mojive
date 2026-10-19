"""Antialiased selection outline render pass."""

from __future__ import annotations

import moderngl
import numpy as np

from ...backend import SELECTION_XRAY_ALPHA, RenderFlag
from ...dependencies import camera_key, lifecycle_key
from .. import gl_native as G
from ..programs import ProgramSpec, UniformCache
from ..registry import register_pass
from .base import BasePass, PassContext, state_overlay
from .idbuffer import IdGeometry

OUTLINE_RADIUS = 3


OUTLINE_COLOR = (1.0, 0.63, 0.20, 1.0)


def circular_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    r2 = radius * radius
    return tuple(
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= r2
    )


class OutlinePass(BasePass):
    name = "outline"

    def __init__(self) -> None:
        self._geom = IdGeometry(only_selected=True, float_mask=True)
        self._mask_ms_tex: moderngl.Texture | None = None
        self._mask_ms_fbo: moderngl.Framebuffer | None = None
        self._mask_tex: moderngl.Texture | None = None
        self._mask_fbo: moderngl.Framebuffer | None = None
        self._alpha_tex: moderngl.Texture | None = None
        self._alpha_fbo: moderngl.Framebuffer | None = None
        self._mask_samples = -1
        self._mask_w = -1
        self._mask_h = -1
        self._size: tuple[int, int] = (0, 0)
        self._spec = ProgramSpec(
            name="outline",
            vertex="fullscreen.vert",
            fragment="outline.frag",
            defines={"OUTLINE_RADIUS": OUTLINE_RADIUS},
        )
        self._alpha_spec = ProgramSpec(
            name="outline",
            vertex="fullscreen.vert",
            fragment="outline.frag",
            defines={"OUTLINE_RADIUS": OUTLINE_RADIUS, "OUTLINE_ALPHA": 1},
        )
        self._composite_spec = ProgramSpec(
            name="outline_composite",
            vertex="fullscreen.vert",
            fragment="outline_composite.frag",
        )
        self._alpha_vao: moderngl.VertexArray | None = None
        self._alpha_uniforms: UniformCache | None = None
        self._composite_vao: moderngl.VertexArray | None = None
        self._composite_uniforms: UniformCache | None = None
        self._vao: moderngl.VertexArray | None = None
        self._uniforms: UniformCache | None = None
        self._generation = -1

        self._buckets: tuple[int, ...] = ()
        self._last_scene = None
        self._last_key: tuple | None = None
        self._cache_key: tuple | None = None
        self._pending_key: tuple | None = None
        self._build_alpha = False
        self.cache_status = "off"
        self.color: tuple[float, float, float, float] = OUTLINE_COLOR
        self.xray = False

    def prepare(self, ctx: PassContext) -> bool:
        self.cache_status = "off"
        self._pending_key = None
        self._build_alpha = False
        if not ctx.flag(RenderFlag.OUTLINE):
            return False
        sel = int(ctx.selected_id if ctx.outline_id is None else ctx.outline_id)
        if sel == 0:
            return False
        scene = ctx.scene
        cacheable = all(
            revision > 0
            for revision in (
                scene.structure_revision,
                scene.pose_revision,
                scene.visual_revision,
                scene.identity_revision,
            )
        )
        key = self._dependency_key(ctx, sel)
        stable = cacheable and self._last_scene is scene and self._last_key == key
        if stable and self._cache_key == key:
            self.cache_status = "reused"
            return True

        # Invalidate before replacing or drawing resources: a failed refresh cannot
        # make the previous selection's alpha valid again.
        self._last_scene = None
        self._last_key = None
        self._cache_key = None
        self._buckets = self._selected_buckets(ctx, sel)
        if not self._buckets:
            return False
        if not self._geom.ensure(ctx, 0):
            return False
        self._geom.upload(ctx)
        self._ensure_mask(ctx)
        # A changing selection uses the original single-pass kernel. Only a
        # repeated managed input pays for the extra cached-alpha product.
        self._build_alpha = stable
        if stable:
            self._ensure_alpha(ctx)
        self._ensure_programs(ctx)
        self._pending_key = key if cacheable else None
        return True

    def _dependency_key(self, ctx: PassContext, selected: int) -> tuple:
        return (
            lifecycle_key(ctx.scene, ctx.frame_serial, visual=True, identity=True),
            camera_key(ctx.camera),
            int(ctx.mesh_revision),
            int(ctx.programs.generation),
            (ctx.target.width, ctx.target.height, ctx.target.samples),
            bool(ctx.flag(RenderFlag.MSAA)),
            selected,
            bool(self.xray),
        )

    def execute(self, ctx: PassContext) -> None:
        gl = ctx.ctx
        reused = self.cache_status == "reused"
        if not reused:
            self._render_mask(ctx)
            if self._build_alpha:
                self._render_alpha(ctx)

        ctx.target.use_main()
        state_overlay(gl, depth_test=False)
        gl.wireframe = False
        shared_id = ctx.target.id_fbo is ctx.target.fbo
        if shared_id:
            ctx.target.fbo.color_mask = (
                (True, True, True, True),
                (False, False, False, False),
            )
        try:
            if reused or self._build_alpha:
                assert self._alpha_tex is not None
                vao, uniforms = self._composite_vao, self._composite_uniforms
                assert vao is not None and uniforms is not None
                self._alpha_tex.use(0)
                uniforms.force("u_alpha", 0)
            else:
                vao, uniforms = self._vao, self._uniforms
                assert vao is not None and uniforms is not None
                self._bind_mask(uniforms)
            uniforms.set("u_color", self.color)
            vao.render(moderngl.TRIANGLES, vertices=3)
        finally:
            if shared_id:
                ctx.target.fbo.color_mask = (
                    (True, True, True, True),
                    (True, True, True, True),
                )
        if not reused:
            if self._pending_key is not None:
                self._last_scene = ctx.scene
                self._last_key = self._pending_key
                if self._build_alpha:
                    self._cache_key = self._pending_key
            self._pending_key = None
            self.cache_status = "primed" if self._build_alpha else "rendered"

    def _render_mask(self, ctx: PassContext) -> None:
        gl = ctx.ctx
        assert self._mask_ms_fbo is not None and self._mask_fbo is not None

        self._mask_ms_fbo.use()
        gl.viewport = (0, 0, self._mask_w, self._mask_h)
        gl.enable_only(moderngl.NOTHING)
        self._mask_ms_fbo.clear(0.0, 0.0, 0.0, 0.0)
        gl.enable_only(moderngl.CULL_FACE)
        gl.front_face = "ccw"
        gl.cull_face = "back"
        gl.wireframe = False
        gl.multisample = bool(ctx.flag(RenderFlag.MSAA))
        assert self._geom.program is not None
        self._geom.set_view_proj(ctx)
        selected = ctx.selected_id if ctx.outline_id is None else ctx.outline_id
        self._geom.program["u_selected"].value = int(selected)
        ctx.draw_calls += self._geom.draw(ctx, self._buckets)

        if self._mask_ms_fbo is not self._mask_fbo:
            if not G.native().blit_color(
                self._mask_ms_fbo.glo, self._mask_fbo.glo, self._mask_w, self._mask_h
            ):
                gl.copy_framebuffer(self._mask_fbo, self._mask_ms_fbo)
            self._mask_fbo.use()

    def _render_alpha(self, ctx: PassContext) -> None:
        assert self._alpha_fbo is not None
        assert self._alpha_vao is not None and self._alpha_uniforms is not None
        self._alpha_fbo.use()
        ctx.ctx.enable_only(moderngl.NOTHING)
        self._bind_mask(self._alpha_uniforms)
        self._alpha_vao.render(moderngl.TRIANGLES, vertices=3)

    def _bind_mask(self, uniforms: UniformCache) -> None:
        assert self._mask_tex is not None
        self._mask_tex.use(0)
        uniforms.force("u_mask", 0)
        uniforms.set("u_size", self._size)
        uniforms.set("u_xray_alpha", SELECTION_XRAY_ALPHA if self.xray else 0.0)

    def _selected_buckets(self, ctx: PassContext, selected: int) -> tuple[int, ...]:
        scene = ctx.scene
        ids = scene.object_id
        return tuple(
            b
            for b, (start, stop) in enumerate(scene.bucket_ranges)
            if stop > start and bool(np.any(ids[start:stop] == selected))
        )

    def _ensure_mask(self, ctx: PassContext) -> None:
        w, h = ctx.target.width, ctx.target.height
        max_samples = int(ctx.ctx.info.get("GL_MAX_SAMPLES", 1) or 1)
        samples = min(4, max_samples) if max_samples > 1 else 0
        if (
            self._mask_fbo is not None
            and self._mask_w == w
            and self._mask_h == h
            and self._mask_samples == samples
        ):
            return
        self._release_mask()
        self._mask_tex = ctx.ctx.texture((w, h), 1, dtype="f1")
        self._mask_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._mask_fbo = ctx.ctx.framebuffer([self._mask_tex])
        if samples:
            self._mask_ms_tex = ctx.ctx.texture((w, h), 1, samples=samples, dtype="f1")
            self._mask_ms_fbo = ctx.ctx.framebuffer([self._mask_ms_tex])
        else:
            self._mask_ms_tex = self._mask_tex
            self._mask_ms_fbo = self._mask_fbo
        self._mask_samples = samples
        self._mask_w, self._mask_h = w, h
        self._size = (w, h)

    def _ensure_alpha(self, ctx: PassContext) -> None:
        # Preserve the shader's alpha precision before blending into the main target.
        if self._alpha_tex is None:
            self._alpha_tex = ctx.ctx.texture(self._size, 1, dtype="f4")
            self._alpha_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        if self._alpha_fbo is None:
            self._alpha_fbo = ctx.ctx.framebuffer([self._alpha_tex])

    def _ensure_programs(self, ctx: PassContext) -> None:
        if self._generation != ctx.programs.generation:
            self._release_programs()
            self._generation = ctx.programs.generation
        if self._vao is None:
            prog = ctx.programs.get(self._spec)
            self._vao = ctx.ctx.vertex_array(prog, [])
            self._uniforms = UniformCache(prog, self._generation)
        if self._build_alpha:
            if self._alpha_vao is None:
                prog = ctx.programs.get(self._alpha_spec)
                self._alpha_vao = ctx.ctx.vertex_array(prog, [])
                self._alpha_uniforms = UniformCache(prog, self._generation)
            if self._composite_vao is None:
                prog = ctx.programs.get(self._composite_spec)
                self._composite_vao = ctx.ctx.vertex_array(prog, [])
                self._composite_uniforms = UniformCache(prog, self._generation)

    def _release_mask(self) -> None:
        self._last_scene = None
        self._last_key = None
        self._cache_key = None
        if self._mask_ms_fbo is not None and self._mask_ms_fbo is not self._mask_fbo:
            self._mask_ms_fbo.release()
        if self._mask_ms_tex is not None and self._mask_ms_tex is not self._mask_tex:
            self._mask_ms_tex.release()
        if self._mask_fbo is not None:
            self._mask_fbo.release()
        if self._mask_tex is not None:
            self._mask_tex.release()
        if self._alpha_fbo is not None:
            self._alpha_fbo.release()
        if self._alpha_tex is not None:
            self._alpha_tex.release()
        self._mask_ms_fbo = None
        self._mask_ms_tex = None
        self._mask_fbo = None
        self._mask_tex = None
        self._alpha_fbo = None
        self._alpha_tex = None
        self._mask_samples = -1
        self._mask_w = self._mask_h = -1

    def _release_programs(self) -> None:
        for obj in (self._vao, self._alpha_vao, self._composite_vao):
            if obj is not None:
                obj.release()
        self._vao = None
        self._alpha_vao = None
        self._uniforms = None
        self._alpha_uniforms = None
        self._composite_vao = None
        self._composite_uniforms = None
        self._generation = -1

    def release(self) -> None:
        self._release_mask()
        self._release_programs()
        self._pending_key = None
        self._build_alpha = False
        self.cache_status = "off"
        self._geom.release()


register_pass("outline", OutlinePass)
