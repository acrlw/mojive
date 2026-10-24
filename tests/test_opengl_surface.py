"""Window readback validates host storage before passing a pointer to OpenGL."""

import ctypes

import numpy as np
import pytest

from mojive.ui.opengl_surface import OpenGLSurface


@pytest.mark.parametrize(
    "pixels",
    [
        np.empty((3, 7, 3), np.float32),
        np.empty((3, 7, 4), np.uint8),
        np.empty((3, 7), np.uint8),
        np.empty((3, 14, 3), np.uint8)[:, ::2],
        np.frombuffer(bytes(63), np.uint8).reshape(3, 7, 3),
    ],
)
def test_readback_rejects_unsafe_host_storage_before_any_gl_call(pixels):
    surface = object.__new__(OpenGLSurface)
    with pytest.raises(ValueError, match="contiguous uint8 RGB"):
        surface.read_rgb(pixels)


def test_readback_writes_the_owned_buffer_with_byte_aligned_odd_width():
    surface = object.__new__(OpenGLSurface)
    calls = []
    surface._bind_framebuffer = lambda *args: calls.append(("bind", args))
    surface._read_buffer = lambda *args: calls.append(("buffer", args))
    surface._pixel_store = lambda *args: calls.append(("pack", args))
    surface._get_error = lambda: 0
    pixels = np.zeros((3, 7, 3), np.uint8)

    def read(x, y, width, height, format_, type_, pointer):
        assert (x, y, width, height, format_, type_) == (0, 0, 7, 3, 0x1907, 0x1401)
        assert calls == [("bind", (0x8D40, 0)), ("buffer", (0x0405,)), ("pack", (0x0D05, 1))]
        ctypes.memset(pointer, 127, width * height * 3)

    surface._read_pixels = read
    surface.read_rgb(pixels)
    assert np.all(pixels == 127)


def test_missing_context_entry_point_fails_explicitly():
    with pytest.raises(RuntimeError, match="glBindFramebuffer"):
        OpenGLSurface(lambda _name: 0)


def test_native_operation_reports_driver_errors_instead_of_returning_stale_pixels():
    surface = object.__new__(OpenGLSurface)
    surface._bind_framebuffer = lambda *_args: None
    surface._read_buffer = lambda *_args: None
    surface._pixel_store = lambda *_args: None
    surface._read_pixels = lambda *_args: None
    surface._get_error = lambda: 0x0502
    with pytest.raises(RuntimeError, match="readback failed: 0x0502"):
        surface.read_rgb(np.zeros((3, 7, 3), np.uint8))
