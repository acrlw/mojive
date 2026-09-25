"""Windows extension entry points come from the current WGL context."""

import ctypes
from types import SimpleNamespace

import pytest

from mojive.render.opengl import gl_native as gl


def test_extension_binding_uses_wgl(monkeypatch):
    monkeypatch.setattr(gl.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(lambda value: value + 1)
    names = []

    def resolve(name):
        names.append(name)
        return ctypes.cast(callback, ctypes.c_void_p).value

    driver = gl.GLNative.__new__(gl.GLNative)
    driver._lib = SimpleNamespace(wglGetProcAddress=resolve)
    assert driver._bind("glTestExtension", [ctypes.c_int], ctypes.c_int)
    assert driver._glTestExtension(41) == 42
    assert names == [b"glTestExtension"]


@pytest.mark.parametrize("address", [None, 0, 1, 2, 3, ctypes.c_void_p(-1).value])
def test_invalid_wgl_addresses_are_not_callable(monkeypatch, address):
    monkeypatch.setattr(gl.sys, "platform", "win32")
    driver = gl.GLNative.__new__(gl.GLNative)
    driver._lib = SimpleNamespace(wglGetProcAddress=lambda name: address)
    assert not driver._bind("glMissingExtension", [], None)


def test_wgl_cache_tracks_context_changes_and_threads(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    context = [None]

    class Driver:
        def __init__(self):
            self._context_handle = context[0]

        def _current_context(self):
            return context[0]

    monkeypatch.setattr(gl.sys, "platform", "win32")
    monkeypatch.setattr(gl, "GLNative", Driver)
    monkeypatch.setattr(gl, "_WINDOWS_LOCAL", threading.local())
    empty = gl.native()
    context[0] = 1
    first = gl.native()
    assert first is not empty and gl.native() is first
    context[0] = 2
    second = gl.native()
    assert second is not first
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(gl.native).result() is not second
    assert gl.native() is second
