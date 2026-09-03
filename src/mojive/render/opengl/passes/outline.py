"""Antialiased selection outline render pass."""

from __future__ import annotations

import moderngl
import numpy as np

from ...backend import SELECTION_XRAY_ALPHA, RenderFlag
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
        self._prog: moderngl.Program | None = None
        self._vao: moderngl.VertexArray | None = None
        self._uniforms: UniformCache | None = None
        self._generation = -1

        self._buckets: tuple[int, ...] = ()
        self._sel_id = -1
        self._sel_ids: object = None
        self.color: tuple[float, float, float, float] = OUTLINE_COLOR
        self.xray = False

    def prepare(self, ctx: PassContext) -> bool:
        if not ctx.flag(RenderFlag.OUTLINE):
            return False
        sel = int(ctx.selected_id)
        if sel == 0:
            return False
        self._buckets = self._selected_buckets(ctx, sel)
        if not self._buckets:
            return False
        if not self._geom.ensure(ctx, 0):
            return False
        self._geom.upload(ctx)
        self._ensure_mask(ctx)
        self._ensure_composite(ctx)
        return True

    def execute(self, ctx: PassContext) -> None:
        gl = ctx.ctx
        assert self._mask_ms_fbo is not None and self._mask_fbo is not None
        assert self._mask_tex is not None
        assert self._prog is not None and self._vao is not None and self._uniforms is not None

        self._mask_ms_fbo.use()
        gl.viewport = (0, 0, self._mask_w, self._mask_h)
        gl.enable_only(moderngl.NOTHING)
        self._mask_ms_fbo.clear(0.0, 0.0, 0.0, 0.0)
        gl.enable_only(moderngl.CULL_FACE)
        gl.front_face = "ccw"
        gl.cull_face = "back"
        gl.wireframe = False
        assert self._geom.program is not None
        self._geom.set_view_proj(ctx)
        self._geom.program["u_selected"].value = int(ctx.selected_id)
        ctx.draw_calls += self._geom.draw(ctx, self._buckets)

        if self._mask_ms_fbo is not self._mask_fbo:
            if not G.native().blit_color(
                self._mask_ms_fbo.glo, self._mask_fbo.glo, self._mask_w, self._mask_h
            ):
                gl.copy_framebuffer(self._mask_fbo, self._mask_ms_fbo)
            self._mask_fbo.use()

        ctx.target.use_main()
        state_overlay(gl, depth_test=False)
        shared_id = ctx.target.id_fbo is ctx.target.fbo
        if shared_id:
            ctx.target.fbo.color_mask = (
                (True, True, True, True),
                (False, False, False, False),
            )
        self._mask_tex.use(0)
        u = self._uniforms
        u.force("u_mask", 0)
        u.set("u_size", self._size)
        u.set("u_color", self.color)
        u.set("u_xray_alpha", SELECTION_XRAY_ALPHA if self.xray else 0.0)
        self._vao.render(moderngl.TRIANGLES, vertices=3)
        if shared_id:
            ctx.target.fbo.color_mask = (
                (True, True, True, True),
                (True, True, True, True),
            )

    def _selected_buckets(self, ctx: PassContext, selected: int) -> tuple[int, ...]:
        scene = ctx.scene
        if selected == self._sel_id and scene.object_id is self._sel_ids:
            return self._buckets
        ids = scene.object_id
        found = [
            b
            for b, (start, stop) in enumerate(scene.bucket_ranges)
            if stop > start and bool(np.any(ids[start:stop] == selected))
        ]
        self._buckets = tuple(found)
        self._sel_id = selected
        self._sel_ids = ids
        return self._buckets

    def _ensure_mask(self, ctx: PassContext) -> None:
        w, h = ctx.target.width, ctx.target.height
        max_samples = int(ctx.ctx.info.get("GL_MAX_SAMPLES", 1) or 1)
        samples = min(4, max_samples) if max_samples > 1 else 0
        if (
            self._mask_tex is not None
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

    def _ensure_composite(self, ctx: PassContext) -> None:
        if self._prog is not None and self._generation == ctx.programs.generation:
            return
        prog = ctx.programs.get(self._spec)

        if self._vao is not None:
            self._vao.release()
        self._vao = ctx.ctx.vertex_array(prog, [])
        self._generation = ctx.programs.generation
        self._prog = prog
        if self._uniforms is None:
            self._uniforms = UniformCache(prog, self._generation)
        else:
            self._uniforms.rebind(prog, self._generation)

    def _release_mask(self) -> None:
        if self._mask_ms_fbo is not None and self._mask_ms_fbo is not self._mask_fbo:
            self._mask_ms_fbo.release()
        if self._mask_ms_tex is not None and self._mask_ms_tex is not self._mask_tex:
            self._mask_ms_tex.release()
        if self._mask_fbo is not None:
            self._mask_fbo.release()
        if self._mask_tex is not None:
            self._mask_tex.release()
        self._mask_ms_fbo = None
        self._mask_ms_tex = None
        self._mask_fbo = None
        self._mask_tex = None
        self._mask_samples = -1
        self._mask_w = self._mask_h = -1

    def release(self) -> None:
        self._release_mask()
        for obj in (self._vao,):
            if obj is not None:
                obj.release()
        self._vao = None
        self._uniforms = None
        self._geom.release()


register_pass("outline", OutlinePass)
