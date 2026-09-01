"""Low-level OpenGL state and framebuffer helpers."""

from __future__ import annotations

import ctypes
import sys
from ctypes import c_float, c_int, c_ubyte, c_uint, c_void_p

GL_ARRAY_BUFFER = 0x8892
GL_COLOR = 0x1800
GL_FLOAT = 0x1406
GL_UNSIGNED_INT = 0x1405
GL_FALSE = 0
GL_DEBUG_SOURCE_APPLICATION = 0x824A


GL_DEPTH_TEST = 0x0B71
GL_CULL_FACE = 0x0B44
GL_BLEND = 0x0BE2
GL_SCISSOR_TEST = 0x0C11
GL_MULTISAMPLE = 0x809D
ENABLE_BITS = (GL_DEPTH_TEST, GL_CULL_FACE, GL_BLEND, GL_SCISSOR_TEST, GL_MULTISAMPLE)


GL_BLEND_SRC_RGB = 0x80C9
GL_BLEND_DST_RGB = 0x80C8
GL_BLEND_SRC_ALPHA = 0x80CB
GL_BLEND_DST_ALPHA = 0x80CA
GL_BLEND_EQUATION_RGB = 0x8009
GL_BLEND_EQUATION_ALPHA = 0x883D

GL_DEPTH_FUNC = 0x0B74
GL_DEPTH_WRITEMASK = 0x0B72
GL_COLOR_WRITEMASK = 0x0C23
GL_VIEWPORT = 0x0BA2
GL_SCISSOR_BOX = 0x0C10
GL_POLYGON_MODE = 0x0B40
GL_FRONT_FACE = 0x0B46
GL_CULL_FACE_MODE = 0x0B45
GL_FRONT_AND_BACK = 0x0408
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_COLOR_BUFFER_BIT = 0x00004000
GL_FRAMEBUFFER = 0x8D40
GL_READ_FRAMEBUFFER = 0x8CA8
GL_DRAW_FRAMEBUFFER = 0x8CA9
GL_NEAREST = 0x2600
GL_COLOR_ATTACHMENT0 = 0x8CE0


GL_TIME_ELAPSED = 0x88BF
GL_QUERY_RESULT = 0x8866
GL_QUERY_RESULT_AVAILABLE = 0x8867


def _load_gl() -> ctypes.CDLL | None:
    candidates: tuple[str, ...]
    if sys.platform == "darwin":
        candidates = ("/System/Library/Frameworks/OpenGL.framework/OpenGL",)
    elif sys.platform.startswith("linux"):
        candidates = ("libGL.so.1", "libGL.so")
    else:
        candidates = ("opengl32.dll",)
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class GLNative:
    def __init__(self) -> None:
        self._lib = _load_gl()
        self.has_clear_buffer_uiv = False
        self.has_clear_buffer_fv = False
        self.has_attrib_pointer = False
        self.has_debug_group = False
        self.has_state = False
        self.has_clear_depth = False
        self.has_blit = False
        self.has_query = False
        self.has_array_layer = False
        if self._lib is None:
            return

        self.has_clear_buffer_uiv = self._bind(
            "glClearBufferuiv", [c_uint, c_int, ctypes.POINTER(c_uint)], None
        )

        self.has_clear_buffer_fv = self._bind(
            "glClearBufferfv", [c_uint, c_int, ctypes.POINTER(c_float)], None
        )
        self.has_attrib_pointer = all(
            (
                self._bind("glBindVertexArray", [c_uint], None),
                self._bind("glBindBuffer", [c_uint, c_uint], None),
                self._bind(
                    "glVertexAttribPointer",
                    [c_uint, c_int, c_uint, c_ubyte, c_int, c_void_p],
                    None,
                ),
                self._bind(
                    "glVertexAttribIPointer",
                    [c_uint, c_int, c_uint, c_int, c_void_p],
                    None,
                ),
                self._bind("glVertexAttribDivisor", [c_uint, c_uint], None),
                self._bind("glEnableVertexAttribArray", [c_uint], None),
            )
        )

        self.has_debug_group = all(
            (
                self._bind("glPushDebugGroup", [c_uint, c_uint, c_int, c_void_p], None),
                self._bind("glPopDebugGroup", [], None),
            )
        )
        self._bind("glGetFloatv", [c_uint, ctypes.POINTER(c_float)], None)
        self._bind("glGetError", [], c_uint)

        self.has_state = all(
            (
                self._bind("glGetIntegerv", [c_uint, ctypes.POINTER(c_int)], None),
                self._bind("glGetBooleanv", [c_uint, ctypes.POINTER(c_ubyte)], None),
                self._bind("glIsEnabled", [c_uint], c_ubyte),
                self._bind("glEnable", [c_uint], None),
                self._bind("glDisable", [c_uint], None),
                self._bind("glBlendFuncSeparate", [c_uint, c_uint, c_uint, c_uint], None),
                self._bind("glBlendEquationSeparate", [c_uint, c_uint], None),
                self._bind("glDepthFunc", [c_uint], None),
                self._bind("glDepthMask", [c_ubyte], None),
                self._bind("glColorMask", [c_ubyte, c_ubyte, c_ubyte, c_ubyte], None),
                self._bind("glViewport", [c_int, c_int, c_int, c_int], None),
                self._bind("glScissor", [c_int, c_int, c_int, c_int], None),
                self._bind("glPolygonMode", [c_uint, c_uint], None),
                self._bind("glFrontFace", [c_uint], None),
                self._bind("glCullFace", [c_uint], None),
            )
        )

        self.has_clear_depth = all(
            (
                self._bind("glClear", [c_uint], None),
                self._bind("glClearDepth", [ctypes.c_double], None),
            )
        )

        self.has_query = all(
            (
                self._bind("glGenQueries", [c_int, ctypes.POINTER(c_uint)], None),
                self._bind("glDeleteQueries", [c_int, ctypes.POINTER(c_uint)], None),
                self._bind("glBeginQuery", [c_uint, c_uint], None),
                self._bind("glEndQuery", [c_uint], None),
                self._bind("glGetQueryObjectuiv", [c_uint, c_uint, ctypes.POINTER(c_uint)], None),
                self._bind(
                    "glGetQueryObjectui64v",
                    [c_uint, c_uint, ctypes.POINTER(ctypes.c_uint64)],
                    None,
                ),
            )
        )

        self.has_blit = all(
            (
                self._bind("glBindFramebuffer", [c_uint, c_uint], None),
                self._bind(
                    "glBlitFramebuffer",
                    [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_uint, c_uint],
                    None,
                ),
                self._bind("glReadBuffer", [c_uint], None),
            )
        )

        self.has_array_layer = self._bind(
            "glFramebufferTextureLayer", [c_uint, c_uint, c_uint, c_int, c_int], None
        )

    def _bind(self, name: str, argtypes: list, restype) -> bool:
        try:
            fn = getattr(self._lib, name)
        except AttributeError:
            return False
        fn.argtypes = argtypes
        fn.restype = restype
        setattr(self, "_" + name, fn)
        return True

    def clear_color_uint(self, draw_buffer: int, value: int = 0) -> bool:
        if not self.has_clear_buffer_uiv:
            return False
        buf = (c_uint * 4)(value, 0, 0, 0)
        self._glClearBufferuiv(GL_COLOR, draw_buffer, buf)  # type: ignore[attr-defined]
        return True

    def rebind_instance_attributes(
        self,
        vao_glo: int,
        buffer_glo: int,
        stride: int,
        base_offset: int,
        attributes: tuple[tuple[int, int, int, int], ...],
    ) -> bool:
        if not self.has_attrib_pointer:
            return False
        self._glBindVertexArray(vao_glo)  # type: ignore[attr-defined]
        self._glBindBuffer(GL_ARRAY_BUFFER, buffer_glo)  # type: ignore[attr-defined]
        for loc, comps, off, gl_type in attributes:
            if loc < 0:
                continue
            self._glEnableVertexAttribArray(loc)  # type: ignore[attr-defined]
            ptr = c_void_p(base_offset + off)
            if gl_type == GL_FLOAT:
                self._glVertexAttribPointer(loc, comps, gl_type, GL_FALSE, stride, ptr)  # type: ignore[attr-defined]
            else:
                self._glVertexAttribIPointer(loc, comps, gl_type, stride, ptr)  # type: ignore[attr-defined]
            self._glVertexAttribDivisor(loc, 1)  # type: ignore[attr-defined]
        self._glBindVertexArray(0)  # type: ignore[attr-defined]
        return True

    def clear_color_float(self, draw_buffer: int, rgba: tuple[float, float, float, float]) -> bool:
        if not self.has_clear_buffer_fv:
            return False
        buf = (c_float * 4)(*rgba)
        self._glClearBufferfv(GL_COLOR, draw_buffer, buf)  # type: ignore[attr-defined]
        return True

    def clear_depth_only(self, value: float = 1.0) -> bool:
        if not self.has_clear_depth:
            return False
        self._glClearDepth(ctypes.c_double(value))  # type: ignore[attr-defined]
        self._glClear(GL_DEPTH_BUFFER_BIT)  # type: ignore[attr-defined]
        return True

    def blit_color(
        self,
        src_glo: int,
        dst_glo: int,
        width: int,
        height: int,
        attachment: int = 0,
    ) -> bool:
        if not self.has_blit:
            return False
        self._glBindFramebuffer(GL_READ_FRAMEBUFFER, src_glo)  # type: ignore[attr-defined]
        self._glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_glo)  # type: ignore[attr-defined]
        self._glReadBuffer(GL_COLOR_ATTACHMENT0 + int(attachment))  # type: ignore[attr-defined]
        self._glBlitFramebuffer(  # type: ignore[attr-defined]
            0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST
        )
        return True

    def blit_depth(self, src_glo: int, dst_glo: int, width: int, height: int) -> bool:
        if not self.has_blit:
            return False
        self._glBindFramebuffer(GL_READ_FRAMEBUFFER, src_glo)  # type: ignore[attr-defined]
        self._glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_glo)  # type: ignore[attr-defined]
        self._glBlitFramebuffer(  # type: ignore[attr-defined]
            0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST
        )
        return True

    def attach_array_layer(self, texture_glo: int, layer: int) -> bool:
        if not self.has_array_layer:
            return False
        self._glFramebufferTextureLayer(  # type: ignore[attr-defined]
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, int(texture_glo), 0, int(layer)
        )
        return True

    def gen_query(self) -> int:
        out = (c_uint * 1)()
        self._glGenQueries(1, out)  # type: ignore[attr-defined]
        return int(out[0])

    def delete_query(self, qid: int) -> None:
        buf = (c_uint * 1)(qid)
        self._glDeleteQueries(1, buf)  # type: ignore[attr-defined]

    def begin_time_query(self, qid: int) -> None:
        self._glBeginQuery(GL_TIME_ELAPSED, qid)  # type: ignore[attr-defined]

    def end_time_query(self) -> None:
        self._glEndQuery(GL_TIME_ELAPSED)  # type: ignore[attr-defined]

    def query_ready(self, qid: int, out) -> bool:
        self._glGetQueryObjectuiv(qid, GL_QUERY_RESULT_AVAILABLE, out)  # type: ignore[attr-defined]
        return bool(out[0])

    def query_result_ns(self, qid: int, out) -> int:
        self._glGetQueryObjectui64v(qid, GL_QUERY_RESULT, out)  # type: ignore[attr-defined]
        return int(out[0])

    def push_debug_group(self, label: str) -> None:
        if not self.has_debug_group:
            return
        data = label.encode()
        self._glPushDebugGroup(  # type: ignore[attr-defined]
            GL_DEBUG_SOURCE_APPLICATION, 0, len(data), ctypes.c_char_p(data)
        )

    def pop_debug_group(self) -> None:
        if not self.has_debug_group:
            return
        self._glPopDebugGroup()  # type: ignore[attr-defined]

    def get_error(self) -> int:
        fn = getattr(self, "_glGetError", None)
        return int(fn()) if fn else 0

    def drain_errors(self) -> int:
        first = 0
        for _ in range(16):
            e = self.get_error()
            if e == 0:
                break
            first = first or e
        return first

    def get_int(self, pname: int, out) -> None:
        self._glGetIntegerv(pname, out)  # type: ignore[attr-defined]

    def get_bool(self, pname: int, out) -> None:
        self._glGetBooleanv(pname, out)  # type: ignore[attr-defined]

    def is_enabled(self, cap: int) -> bool:
        return bool(self._glIsEnabled(cap))  # type: ignore[attr-defined]

    def set_enabled(self, cap: int, on: bool) -> None:
        (self._glEnable if on else self._glDisable)(cap)  # type: ignore[attr-defined]

    def blend_func_separate(self, sr: int, dr: int, sa: int, da: int) -> None:
        self._glBlendFuncSeparate(sr, dr, sa, da)  # type: ignore[attr-defined]

    def blend_equation_separate(self, rgb: int, alpha: int) -> None:
        self._glBlendEquationSeparate(rgb, alpha)  # type: ignore[attr-defined]

    def depth_func(self, func: int) -> None:
        self._glDepthFunc(func)  # type: ignore[attr-defined]

    def depth_mask(self, on: bool) -> None:
        self._glDepthMask(1 if on else 0)  # type: ignore[attr-defined]

    def color_mask(self, r: bool, g: bool, b: bool, a: bool) -> None:
        self._glColorMask(int(r), int(g), int(b), int(a))  # type: ignore[attr-defined]

    def viewport(self, x: int, y: int, w: int, h: int) -> None:
        self._glViewport(x, y, w, h)  # type: ignore[attr-defined]

    def scissor(self, x: int, y: int, w: int, h: int) -> None:
        self._glScissor(x, y, w, h)  # type: ignore[attr-defined]

    def polygon_mode(self, mode: int) -> None:
        self._glPolygonMode(GL_FRONT_AND_BACK, mode)  # type: ignore[attr-defined]

    def front_face(self, mode: int) -> None:
        self._glFrontFace(mode)  # type: ignore[attr-defined]

    def cull_face(self, mode: int) -> None:
        self._glCullFace(mode)  # type: ignore[attr-defined]

    def line_width_range(self) -> tuple[float, float]:
        fn = getattr(self, "_glGetFloatv", None)
        if fn is None:
            return (1.0, 1.0)
        out = (c_float * 2)()
        fn(0x846E, out)  # GL_ALIASED_LINE_WIDTH_RANGE
        return (float(out[0]), float(out[1]))


_INSTANCE: GLNative | None = None


def native() -> GLNative:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = GLNative()
    return _INSTANCE
