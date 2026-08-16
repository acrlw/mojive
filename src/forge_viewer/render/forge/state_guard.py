from __future__ import annotations

import ctypes

import moderngl

from . import gl_native as G


class GLStateGuard:
    __slots__ = ("_bools", "_gl", "_ints4", "_saved", "_state", "available")

    def __init__(self) -> None:
        self._gl = G.native()
        self.available = self._gl.has_state
        self._saved = False

        self._ints4 = (ctypes.c_int * 4)()
        self._bools = (ctypes.c_ubyte * 4)()
        self._state: dict[str, object] = {}

    # ------------------------------------------------------------------
    def _read_int(self, pname: int) -> int:
        self._gl.get_int(pname, self._ints4)
        return int(self._ints4[0])

    def _read_int4(self, pname: int) -> tuple[int, int, int, int]:
        self._gl.get_int(pname, self._ints4)
        return (int(self._ints4[0]), int(self._ints4[1]), int(self._ints4[2]), int(self._ints4[3]))

    def _read_bool(self, pname: int) -> bool:
        self._gl.get_bool(pname, self._bools)
        return bool(self._bools[0])

    def _read_bool4(self, pname: int) -> tuple[bool, bool, bool, bool]:
        self._gl.get_bool(pname, self._bools)
        return tuple(bool(self._bools[i]) for i in range(4))  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def capture(self) -> None:
        if not self.available:
            return
        s = self._state
        for cap in G.ENABLE_BITS:
            s[f"en{cap}"] = self._gl.is_enabled(cap)
        s["src_rgb"] = self._read_int(G.GL_BLEND_SRC_RGB)
        s["dst_rgb"] = self._read_int(G.GL_BLEND_DST_RGB)
        s["src_a"] = self._read_int(G.GL_BLEND_SRC_ALPHA)
        s["dst_a"] = self._read_int(G.GL_BLEND_DST_ALPHA)
        s["eq_rgb"] = self._read_int(G.GL_BLEND_EQUATION_RGB)
        s["eq_a"] = self._read_int(G.GL_BLEND_EQUATION_ALPHA)
        s["depth_func"] = self._read_int(G.GL_DEPTH_FUNC)
        s["depth_mask"] = self._read_bool(G.GL_DEPTH_WRITEMASK)
        s["color_mask"] = self._read_bool4(G.GL_COLOR_WRITEMASK)
        s["viewport"] = self._read_int4(G.GL_VIEWPORT)
        s["scissor"] = self._read_int4(G.GL_SCISSOR_BOX)
        s["polygon_mode"] = self._read_int(G.GL_POLYGON_MODE)
        s["front_face"] = self._read_int(G.GL_FRONT_FACE)
        s["cull_face"] = self._read_int(G.GL_CULL_FACE_MODE)
        self._saved = True

    def restore(self) -> None:
        if not (self.available and self._saved):
            return
        gl, s = self._gl, self._state
        for cap in G.ENABLE_BITS:
            gl.set_enabled(cap, bool(s[f"en{cap}"]))
        gl.blend_func_separate(s["src_rgb"], s["dst_rgb"], s["src_a"], s["dst_a"])  # type: ignore[arg-type]
        gl.blend_equation_separate(s["eq_rgb"], s["eq_a"])  # type: ignore[arg-type]
        gl.depth_func(s["depth_func"])  # type: ignore[arg-type]
        gl.depth_mask(bool(s["depth_mask"]))
        gl.color_mask(*s["color_mask"])  # type: ignore[misc]
        gl.viewport(*s["viewport"])  # type: ignore[misc]
        gl.scissor(*s["scissor"])  # type: ignore[misc]
        gl.polygon_mode(s["polygon_mode"])  # type: ignore[arg-type]
        gl.front_face(s["front_face"])  # type: ignore[arg-type]
        gl.cull_face(s["cull_face"])  # type: ignore[arg-type]
        self._saved = False

    def __enter__(self) -> GLStateGuard:
        self.capture()
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:

        if not self.available:
            return {}
        out: dict = {f"enable_{cap:#x}": self._gl.is_enabled(cap) for cap in G.ENABLE_BITS}
        out["blend_func"] = (
            self._read_int(G.GL_BLEND_SRC_RGB),
            self._read_int(G.GL_BLEND_DST_RGB),
            self._read_int(G.GL_BLEND_SRC_ALPHA),
            self._read_int(G.GL_BLEND_DST_ALPHA),
        )
        out["blend_equation"] = (
            self._read_int(G.GL_BLEND_EQUATION_RGB),
            self._read_int(G.GL_BLEND_EQUATION_ALPHA),
        )
        out["depth_func"] = self._read_int(G.GL_DEPTH_FUNC)
        out["depth_mask"] = self._read_bool(G.GL_DEPTH_WRITEMASK)
        out["color_mask"] = self._read_bool4(G.GL_COLOR_WRITEMASK)
        out["viewport"] = self._read_int4(G.GL_VIEWPORT)
        out["scissor"] = self._read_int4(G.GL_SCISSOR_BOX)
        out["polygon_mode"] = self._read_int(G.GL_POLYGON_MODE)
        out["front_face"] = self._read_int(G.GL_FRONT_FACE)
        out["cull_face"] = self._read_int(G.GL_CULL_FACE_MODE)
        return out


def bind_default_framebuffer(ctx: moderngl.Context) -> None:

    viewport = ctx.viewport
    scissor = ctx.scissor
    ctx.screen.use()
    ctx.viewport = viewport
    ctx.scissor = scissor
