"""Native clear arguments are reused without sharing mutable storage across threads."""

import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest

from mojive.render.opengl import gl_native as gl


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(gl, "_load_gl", lambda: None)
    return gl.GLNative()


@pytest.mark.parametrize(
    "method,capability,entry,first,second",
    [
        ("clear_color_uint", "has_clear_buffer_uiv", "_glClearBufferuiv", 16_777_217, 0),
        ("clear_color_int", "has_clear_buffer_iv", "_glClearBufferiv", (-1, 2, 3, 4), (5, 6, 7, 8)),
        (
            "clear_color_float",
            "has_clear_buffer_fv",
            "_glClearBufferfv",
            (1, 0, 0.5, 1),
            (0, 1, 0, 0),
        ),
    ],
)
def test_clear_reuses_buffer_and_replaces_all_values(
    driver, method, capability, entry, first, second
):
    calls = []

    def clear(target, attachment, buffer):
        calls.append((target, attachment, ctypes.addressof(buffer), tuple(buffer)))

    setattr(driver, capability, True)
    setattr(driver, entry, clear)
    assert getattr(driver, method)(2, first)
    assert getattr(driver, method)(3, second)
    assert [call[:2] for call in calls] == [(gl.GL_COLOR, 2), (gl.GL_COLOR, 3)]
    assert calls[0][2] == calls[1][2]
    for call, values in zip(calls, (first, second), strict=True):
        assert call[3] == (values if isinstance(values, tuple) else (values, 0, 0, 0))


def test_clear_buffers_are_thread_local(driver):
    driver.has_clear_buffer_uiv = True
    driver._glClearBufferuiv = lambda *_: None
    driver.clear_color_uint(0, 42)
    main = driver._clear_buffers.uint

    def other_thread():
        driver.clear_color_uint(0, 9)
        return driver._clear_buffers.uint

    with ThreadPoolExecutor(max_workers=1) as worker:
        other = worker.submit(other_thread).result()
    assert main is not other
    assert tuple(main) == (42, 0, 0, 0)
    assert tuple(other) == (9, 0, 0, 0)


def test_unavailable_clear_does_not_call_entry_point(driver):
    assert not driver.clear_color_uint(0, 1)
    assert not driver.clear_color_int(0, (1, 2, 3, 4))
    assert not driver.clear_color_float(0, (1, 0, 0, 1))
