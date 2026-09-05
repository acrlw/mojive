"""Context selection and actionable errors without requiring a Linux GPU."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from mojive.render import context as renderer


@pytest.fixture
def contexts(monkeypatch):
    monkeypatch.setattr(renderer.sys, "platform", "linux")
    for key in ("MOJIVE_GL", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    result = SimpleNamespace(close=lambda: calls.append("close"))

    def egl(backend):
        calls.append(backend)
        return result

    def glfw(width, height):
        calls.append(("glfw", width, height))
        return result

    monkeypatch.setattr(renderer, "_StandaloneContext", egl)
    monkeypatch.setattr(renderer, "_GLFWContext", glfw)
    return calls, result


def _break_egl(monkeypatch, calls):
    def fail(backend):
        calls.append(backend)
        raise RuntimeError("eglInitialize failed (0x3001)")

    monkeypatch.setattr(renderer, "_StandaloneContext", fail)


@pytest.mark.parametrize("requested", ("", "auto", "egl"))
def test_linux_prefers_egl_without_opening_a_window(contexts, monkeypatch, requested):
    calls, result = contexts
    monkeypatch.setenv("MOJIVE_GL", requested)
    assert renderer._create_context(64, 48) is result
    assert calls == ["egl"]


@pytest.mark.parametrize("requested", ("native", "glfw", " GLFW "))
def test_explicit_glfw_does_not_try_egl(contexts, monkeypatch, requested):
    calls, result = contexts
    monkeypatch.setenv("MOJIVE_GL", requested)
    assert renderer._create_context(64, 48) is result
    assert calls == [("glfw", 64, 48)]


@pytest.mark.parametrize("platform", ("darwin", "win32"))
def test_non_linux_auto_keeps_the_existing_glfw_path(contexts, monkeypatch, platform):
    calls, result = contexts
    monkeypatch.setattr(renderer.sys, "platform", platform)
    assert renderer._create_context(64, 48) is result
    assert calls == [("glfw", 64, 48)]


@pytest.mark.parametrize("display", ("DISPLAY", "WAYLAND_DISPLAY"))
def test_auto_warns_when_egl_falls_back_to_a_desktop(contexts, monkeypatch, display):
    calls, result = contexts
    _break_egl(monkeypatch, calls)
    monkeypatch.setenv(display, ":0")
    with pytest.warns(RuntimeWarning, match="using a hidden GLFW window"):
        assert renderer._create_context(64, 48) is result
    assert calls == ["egl", ("glfw", 64, 48)]


def test_explicit_egl_failure_is_not_silently_replaced(contexts, monkeypatch):
    calls, _ = contexts
    _break_egl(monkeypatch, calls)
    monkeypatch.setenv("MOJIVE_GL", "egl")
    monkeypatch.setenv("DISPLAY", ":0")
    with pytest.raises(RuntimeError, match="Explicit EGL selection was preserved") as caught:
        renderer._create_context(64, 48)
    assert calls == ["egl"]
    assert "eglInitialize failed (0x3001)" in str(caught.value.__cause__)
    assert "MOJIVE_GL=glfw" in str(caught.value)


def test_headless_failure_explains_why_glfw_is_not_a_fallback(contexts, monkeypatch):
    calls, _ = contexts
    _break_egl(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="requires an X11 or Wayland display") as caught:
        renderer._create_context(64, 48)
    assert calls == ["egl"]
    assert "container GPU access" in str(caught.value)
    assert "configuration.md" in str(caught.value)


def test_worker_failure_does_not_initialize_glfw(contexts, monkeypatch):
    calls, _ = contexts
    _break_egl(monkeypatch, calls)
    monkeypatch.setenv("DISPLAY", ":0")
    errors = []

    def worker():
        try:
            renderer._create_context(64, 48)
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert calls == ["egl"]
    assert "requires the main thread" in errors[0]


def test_both_failed_attempts_survive_in_the_final_error(contexts, monkeypatch):
    calls, _ = contexts
    _break_egl(monkeypatch, calls)
    monkeypatch.setenv("DISPLAY", ":0")

    def fail_glfw(*_):
        calls.append("glfw")
        raise RuntimeError("X11: Failed to open display :0")

    monkeypatch.setattr(renderer, "_GLFWContext", fail_glfw)
    with pytest.raises(RuntimeError) as caught:
        renderer._create_context(64, 48)
    assert calls == ["egl", "glfw"]
    assert "eglInitialize failed (0x3001)" in str(caught.value)
    assert "X11: Failed to open display :0" in str(caught.value)
    assert "X11" in str(caught.value.__cause__)


def test_warning_as_error_releases_the_fallback_context(contexts, monkeypatch):
    import warnings

    calls, _ = contexts
    _break_egl(monkeypatch, calls)
    monkeypatch.setenv("DISPLAY", ":0")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning):
            renderer._create_context(64, 48)
    assert calls[-1] == "close"


def test_unknown_context_mode_does_not_start_any_backend(contexts, monkeypatch):
    calls, _ = contexts
    monkeypatch.setenv("MOJIVE_GL", "typo")
    with pytest.raises(ValueError, match="Unsupported MOJIVE_GL"):
        renderer._create_context(64, 48)
    assert not calls
