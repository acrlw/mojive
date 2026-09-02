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
    tex_c = tex_i = depth = fbo = None
    try:
        tex_c = ctx.texture((8, 8), 4, samples=samples, dtype="f1")
        tex_i = ctx.texture((8, 8), 1, samples=samples, dtype="u4")
        depth = ctx.depth_renderbuffer((8, 8), samples=samples)
        fbo = ctx.framebuffer([tex_c, tex_i], depth)
        return IdLayout.SHARED
    except Exception:
        return IdLayout.SPLIT
    finally:
        for obj in (fbo, depth, tex_i, tex_c):
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

        self.depth_resolve = ctx.depth_texture((self.width, self.height))
        self.depth_resolve_fbo = ctx.framebuffer(depth_attachment=self.depth_resolve)
        self.metric_depth = ctx.texture((self.width, self.height), 1, dtype="f4")
        self.segmentation = ctx.texture((self.width, self.height), 2, dtype="i4")
        self.export_depth = ctx.depth_texture((self.width, self.height))
        self.export_fbo = ctx.framebuffer(
            [self.metric_depth, self.segmentation],
            self.export_depth,
        )
        if self.id_layout is IdLayout.SHARED and self.samples > 1:
            self.id_resolve = ctx.texture((self.width, self.height), 1, dtype="u4")
            self.id_resolve_fbo = ctx.framebuffer([self.id_resolve])
        else:
            self.id_resolve = None
            self.id_resolve_fbo = None

        self._clear_prog = ctx.program(vertex_shader=_CLEAR_VS_FAR, fragment_shader=_CLEAR_FS)
        self._clear_vao = ctx.vertex_array(self._clear_prog, [])
        self._pixel = bytearray(4)
        self._rgb_stage = np.empty((self.height, self.width, 3), np.uint8)
        self._metric_stage = np.empty((self.height, self.width), np.float32)
        self._segmentation_stage = np.empty((self.height, self.width, 2), np.int32)

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

    def use_export(self) -> None:
        self.export_fbo.use()
        # Readback can restore the native draw binding to zero without
        # updating ModernGL's cache. Export-only frames may bind this same FBO
        # again, so force the native binding as well.
        self._gl.bind_draw_framebuffer(self.export_fbo.glo)
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

    def clear_export(self, metric_far: float) -> None:
        self.use_export()
        self.export_fbo.depth_mask = True
        cleared = (
            self._gl.clear_depth_only(1.0)
            and self._gl.clear_color_float(
                0,
                (metric_far, metric_far, metric_far, metric_far),
            )
            and self._gl.clear_color_int(1, (-1, -1, -1, -1))
        )
        if not cleared:
            raise RuntimeError("OpenGL typed export-buffer clear is unavailable")

    def resolve(self) -> None:
        prev = self.ctx.fbo
        if self._gl.blit_color(self._blit_src.glo, self.resolve_fbo.glo, self.width, self.height):
            prev.use()
            return
        self.ctx.copy_framebuffer(self.resolve_fbo, self._blit_src)

    def read_id(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0
        fbo, attachment = self._id_read_target()
        raw = fbo.read(
            viewport=(int(x), int(y), 1, 1),
            components=1,
            dtype="u4",
            attachment=attachment,
        )
        return int(np.frombuffer(raw, np.uint32)[0])

    def read_color(self, flip: bool = True) -> np.ndarray:
        raw = self.resolve_fbo.read(components=4, dtype="f1")
        img = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 4)
        return img[::-1] if flip else img

    def read_rgb(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """Read packed RGB, optionally directly into a caller-owned result."""

        shape = (self.height, self.width, 3)
        if out is None:
            out = np.empty(shape, np.uint8)
        elif out.shape != shape or out.dtype != np.uint8 or not out.flags.c_contiguous:
            raise ValueError(
                f"Expected C-contiguous uint8 destination with shape {shape}, "
                f"got {out.dtype} {out.shape}"
            )
        if flip:
            self.resolve_fbo.read_into(
                self._rgb_stage,
                components=3,
                dtype="f1",
                alignment=1,
            )
            np.copyto(out, self._rgb_stage[::-1])
        else:
            self.resolve_fbo.read_into(out, components=3, dtype="f1", alignment=1)
        return out

    def read_depth(self, flip: bool = True) -> np.ndarray:
        fbo = self._blit_src
        if self.samples > 1:
            if not self._gl.blit_depth(
                self._blit_src.glo,
                self.depth_resolve_fbo.glo,
                self.width,
                self.height,
            ):
                raise RuntimeError("Multisample depth resolve is unavailable")
            fbo = self.depth_resolve_fbo
        raw = fbo.read(components=1, dtype="f4", attachment=-1)
        depth = np.frombuffer(raw, np.float32).reshape(self.height, self.width)
        return depth[::-1] if flip else depth

    def read_metric_depth(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        shape = (self.height, self.width)
        if out is None:
            out = np.empty(shape, np.float32)
        elif out.shape != shape or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError(
                f"Expected C-contiguous float32 destination with shape {shape}, "
                f"got {out.dtype} {out.shape}"
            )
        if flip:
            self.export_fbo.read_into(
                self._metric_stage,
                components=1,
                dtype="f4",
                attachment=0,
            )
            np.copyto(out, self._metric_stage[::-1])
        else:
            self.export_fbo.read_into(out, components=1, dtype="f4", attachment=0)
        # ModernGL readback may leave its framebuffer cache pointing at the
        # export target while the native draw binding changed. Move to a known
        # different target so the next export pass performs a real bind.
        self.resolve_fbo.use()
        return out

    def read_ids(self, flip: bool = False) -> np.ndarray:
        fbo, attachment = self._id_read_target()
        raw = fbo.read(components=1, dtype="u4", attachment=attachment)
        ids = np.frombuffer(raw, np.uint32).reshape(self.height, self.width)
        return ids[::-1] if flip else ids

    def read_segmentation(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        shape = (self.height, self.width, 2)
        if out is None:
            out = np.empty(shape, np.int32)
        elif out.shape != shape or out.dtype != np.int32 or not out.flags.c_contiguous:
            raise ValueError(
                f"Expected C-contiguous int32 destination with shape {shape}, "
                f"got {out.dtype} {out.shape}"
            )
        if flip:
            self.export_fbo.read_into(
                self._segmentation_stage,
                components=2,
                dtype="i4",
                attachment=1,
            )
            np.copyto(out, self._segmentation_stage[::-1])
        else:
            self.export_fbo.read_into(out, components=2, dtype="i4", attachment=1)
        self.resolve_fbo.use()
        return out

    def _id_read_target(self):
        fbo, attachment = self.id_fbo, self.id_draw_buffer
        if self.id_resolve_fbo is not None:
            if not self._gl.blit_color(
                self.id_fbo.glo,
                self.id_resolve_fbo.glo,
                self.width,
                self.height,
                self.id_draw_buffer,
            ):
                raise RuntimeError("Multisample integer ID resolve is unavailable")
            fbo, attachment = self.id_resolve_fbo, 0
        return fbo, attachment

    def release(self) -> None:
        with contextlib.suppress(Exception):
            self.ctx.screen.use()
        for obj in (
            self._clear_vao,
            self._clear_prog,
            self.resolve_fbo,
            self.resolve_tex,
            self.depth_resolve_fbo,
            self.depth_resolve,
            self.export_fbo,
            self.export_depth,
            self.metric_depth,
            self.segmentation,
            self.id_resolve_fbo,
            self.id_resolve,
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
