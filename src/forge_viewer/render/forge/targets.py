"""Multisample render targets and object ID attachments."""

from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass

import moderngl
import numpy as np

from . import gl_native as G


class IdLayout(enum.StrEnum):
    SHARED = "shared"

    SPLIT = "split"


def probe_id_layout(ctx: moderngl.Context, samples: int) -> IdLayout:
    if samples <= 1:
        return IdLayout.SHARED
    tex_c = tex_i = fbo = None
    try:
        tex_c = ctx.texture((8, 8), 4, samples=samples, dtype="f1")
        tex_i = ctx.texture((8, 8), 1, samples=samples, dtype="u4")
        fbo = ctx.framebuffer([tex_c, tex_i], ctx.depth_renderbuffer((8, 8), samples=samples))
        return IdLayout.SHARED
    except Exception:
        return IdLayout.SPLIT
    finally:
        for obj in (fbo, tex_i, tex_c):
            if obj is not None:
                obj.release()

        G.native().drain_errors()


_CLEAR_VS = """#version 330 core
// Full-screen triangle generated from gl_VertexID.
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
_CLEAR_FS = """#version 330 core
uniform uint u_value;
layout(location = 0) out uint o_id;
void main() { o_id = u_value; }
"""
_CLEAR_VS_FAR = _CLEAR_VS.replace("p * 2.0 - 1.0, 0.0, 1.0", "p * 2.0 - 1.0, 1.0, 1.0")


@dataclass(frozen=True)
class TargetInfo:
    width: int
    height: int
    samples: int
    id_layout: IdLayout
    id_samples: int


class RenderTarget:
    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        samples: int = 4,
        id_layout: IdLayout | None = None,
    ) -> None:
        self.ctx = ctx
        self.width = max(1, int(width))
        self.height = max(1, int(height))

        self.samples = 0 if int(samples) <= 1 else int(samples)
        self._gl = G.native()
        self.id_layout = id_layout if id_layout is not None else probe_id_layout(ctx, self.samples)

        s = self.samples

        self.color_ms = ctx.texture((self.width, self.height), 4, samples=s, dtype="f1")
        self.depth_ms = ctx.depth_texture((self.width, self.height), samples=s)

        if self.id_layout is IdLayout.SHARED:
            self.id_tex = ctx.texture((self.width, self.height), 1, samples=s, dtype="u4")
            self.fbo = ctx.framebuffer([self.color_ms, self.id_tex], self.depth_ms)
            self.id_fbo = self.fbo
            self.id_draw_buffer = 1
            self.id_samples = s
            self.id_depth = None
        else:
            self.fbo = ctx.framebuffer([self.color_ms], self.depth_ms)

            self.id_tex = ctx.texture((self.width, self.height), 1, dtype="u4")
            self.id_depth = ctx.depth_texture((self.width, self.height))
            self.id_fbo = ctx.framebuffer([self.id_tex], self.id_depth)
            self.id_draw_buffer = 0
            self.id_samples = 0

        self._blit_src = (
            ctx.framebuffer([self.color_ms], self.depth_ms)
            if self.id_layout is IdLayout.SHARED
            else self.fbo
        )

        self.resolve_tex = ctx.texture((self.width, self.height), 4, dtype="f1")
        self.resolve_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.resolve_fbo = ctx.framebuffer([self.resolve_tex])

        self._clear_prog = ctx.program(vertex_shader=_CLEAR_VS_FAR, fragment_shader=_CLEAR_FS)
        self._clear_vao = ctx.vertex_array(self._clear_prog, [])
        self._pixel = bytearray(4)

    @property
    def info(self) -> TargetInfo:
        return TargetInfo(self.width, self.height, self.samples, self.id_layout, self.id_samples)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def id_multisample(self) -> bool:
        return self.id_samples > 1

    def use_main(self) -> None:
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

    def use_id(self) -> None:
        self.id_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

    def clear_main(self, color: tuple[float, float, float, float]) -> None:
        self.fbo.depth_mask = True
        self.fbo.use()
        if self.id_layout is not IdLayout.SHARED:
            self.fbo.clear(*color)
            return

        ok = self._gl.clear_color_float(0, color) and self._gl.clear_depth_only(1.0)
        if not ok:
            self.fbo.clear(*color)
            self._gl.drain_errors()

    def clear_id(self, value: int = 0) -> None:
        self.id_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        split = self.id_layout is IdLayout.SPLIT

        if split:
            self.id_fbo.depth_mask = True

        depth_done = self._gl.clear_depth_only(1.0) if split else True
        if depth_done and self._gl.clear_color_uint(self.id_draw_buffer, int(value)):
            return

        self.ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
        if split and not depth_done:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_func = "1"  # GL_ALWAYS
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        if self.id_layout is IdLayout.SHARED:
            self.fbo.color_mask = ((False, False, False, False), (True, True, True, True))
        self._clear_prog["u_value"].value = int(value)
        self._clear_vao.render(moderngl.TRIANGLES, vertices=3)
        if self.id_layout is IdLayout.SHARED:
            self.fbo.color_mask = ((True, True, True, True), (True, True, True, True))
        self.ctx.depth_func = "<"

    def resolve(self) -> None:
        prev = self.ctx.fbo
        if self._gl.blit_color(self._blit_src.glo, self.resolve_fbo.glo, self.width, self.height):
            prev.use()
            return
        self.ctx.copy_framebuffer(self.resolve_fbo, self._blit_src)

    def read_id(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0
        if self.id_layout is IdLayout.SHARED and self.samples > 1:
            return 0
        raw = self.id_fbo.read(
            viewport=(int(x), int(y), 1, 1),
            components=1,
            dtype="u4",
            attachment=self.id_draw_buffer,
        )
        return int(np.frombuffer(raw, np.uint32)[0])

    def read_color(self, flip: bool = True) -> np.ndarray:
        raw = self.resolve_fbo.read(components=4, dtype="f1")
        img = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 4)
        return img[::-1] if flip else img

    def read_ids(self) -> np.ndarray:
        raw = self.id_fbo.read(components=1, dtype="u4", attachment=self.id_draw_buffer)
        return np.frombuffer(raw, np.uint32).reshape(self.height, self.width)

    def release(self) -> None:
        with contextlib.suppress(Exception):
            self.ctx.screen.use()
        for obj in (
            self._clear_vao,
            self._clear_prog,
            self.resolve_fbo,
            self.resolve_tex,
            self._blit_src if self._blit_src is not self.fbo else None,
            self.id_fbo if self.id_fbo is not self.fbo else None,
            self.id_depth,
            self.id_tex,
            self.fbo,
            self.depth_ms,
            self.color_ms,
        ):
            if obj is not None:
                obj.release()
