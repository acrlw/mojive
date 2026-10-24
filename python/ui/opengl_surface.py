"""Context-owned OpenGL entry points for the window's default framebuffer."""

from __future__ import annotations

import ctypes as C
import sys
from collections.abc import Callable

import numpy as np

_FRAMEBUFFER = 0x8D40
_BACK = 0x0405
_PACK_ALIGNMENT = 0x0D05
_RGB = 0x1907
_UNSIGNED_BYTE = 0x1401
_SCISSOR_TEST = 0x0C11
_COLOR_BUFFER_BIT = 0x00004000
_DEPTH_BUFFER_BIT = 0x00000100
_VERSION = 0x1F02
_RENDERER = 0x1F01


class OpenGLSurface:
    """Resolve only window operations through the owning GLFW context.

    GLFW chooses WGL, GLX or EGL; pointers stay with their window, including on
    Windows where entry points may differ between contexts/pixel formats.
    """

    def __init__(self, resolve: Callable[[str], int]) -> None:
        prototype = C.WINFUNCTYPE if sys.platform == "win32" else C.CFUNCTYPE

        def bind(name, result, *args):
            address = resolve(name)
            if not address:
                raise RuntimeError(f"OpenGL window entry point unavailable: {name}")
            return prototype(result, *args)(address)

        self._bind_framebuffer = bind("glBindFramebuffer", None, C.c_uint, C.c_uint)
        self._viewport = bind("glViewport", None, C.c_int, C.c_int, C.c_int, C.c_int)
        self._disable = bind("glDisable", None, C.c_uint)
        self._depth_mask = bind("glDepthMask", None, C.c_ubyte)
        self._clear_color = bind("glClearColor", None, *([C.c_float] * 4))
        self._clear = bind("glClear", None, C.c_uint)
        self._read_buffer = bind("glReadBuffer", None, C.c_uint)
        self._pixel_store = bind("glPixelStorei", None, C.c_uint, C.c_int)
        self._read_pixels = bind(
            "glReadPixels", None, *([C.c_int] * 4), C.c_uint, C.c_uint, C.c_void_p
        )
        self._get_error = bind("glGetError", C.c_uint)
        get_string = bind("glGetString", C.c_char_p, C.c_uint)
        self.version = get_string(_VERSION).decode()
        self.renderer = get_string(_RENDERER).decode()

    def clear(self, width: int, height: int, color: tuple[float, float, float, float]) -> None:
        """Clear the current window before submitting its ImGui draw lists."""
        self._bind_framebuffer(_FRAMEBUFFER, 0)
        self._viewport(0, 0, width, height)
        self._disable(_SCISSOR_TEST)
        self._depth_mask(1)
        self._clear_color(*color)
        self._clear(_COLOR_BUFFER_BIT | _DEPTH_BUFFER_BIT)
        self._check_error("clear")

    def _check_error(self, operation: str) -> None:
        # Preserve GL failure evidence at the operation boundary, rather than
        # querying the driver after each state-setting call.
        error = self._get_error()
        if error:
            raise RuntimeError(f"OpenGL window {operation} failed: 0x{error:04x}")

    def read_rgb(self, pixels: np.ndarray) -> None:
        """Read bottom-up RGB bytes into the window's reusable contiguous buffer."""
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or not pixels.flags.c_contiguous
            or not pixels.flags.writeable
        ):
            raise ValueError("Window readback requires a writable contiguous uint8 RGB array")
        height, width = pixels.shape[:2]
        self._bind_framebuffer(_FRAMEBUFFER, 0)
        self._read_buffer(_BACK)
        self._pixel_store(_PACK_ALIGNMENT, 1)
        self._read_pixels(0, 0, width, height, _RGB, _UNSIGNED_BYTE, pixels.ctypes.data)
        self._check_error("readback")
