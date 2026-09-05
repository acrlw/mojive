"""Shared offscreen graphics context ownership, independent of any scene adapter."""

from __future__ import annotations

import os
import sys
import threading
import warnings
from contextlib import contextmanager

from .selection import render_backend_name


class _GLFWContext:
    def __init__(self, width: int, height: int) -> None:
        import glfw

        self._glfw = glfw
        if not glfw.init():
            raise RuntimeError(f"GLFW initialization failed: {glfw.get_error()}")
        for hint, value in (
            (glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API),
            (glfw.CONTEXT_VERSION_MAJOR, 3),
            (glfw.CONTEXT_VERSION_MINOR, 3),
            (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
            (glfw.OPENGL_FORWARD_COMPAT, True),
            (glfw.VISIBLE, False),
        ):
            glfw.window_hint(hint, value)
        self.window = glfw.create_window(
            max(1, width), max(1, height), "Mojive Renderer", None, None
        )
        if not self.window:
            raise RuntimeError(f"Failed to create an OpenGL 3.3 core context: {glfw.get_error()}")
        self.gl_context = None

    @contextmanager
    def current(self):
        glfw = self._glfw
        previous = glfw.get_current_context()
        glfw.make_context_current(self.window)
        try:
            yield
        finally:
            glfw.make_context_current(previous)

    def close(self) -> None:
        if self.window is None:
            return
        current = self._glfw.get_current_context()
        if current == self.window:
            self._glfw.make_context_current(None)
        self._glfw.destroy_window(self.window)
        self.window = None


class _StandaloneContext:
    def __init__(self, backend: str) -> None:
        import moderngl

        self.gl_context = moderngl.create_standalone_context(require=330, backend=backend)

    @contextmanager
    def current(self):
        # Each offscreen owner selects its context before access. Keep it current
        # between calls: unbinding every frame forces driver synchronization.
        self.gl_context.__enter__()
        yield

    def close(self) -> None:
        if self.gl_context is None:
            return
        self.gl_context.release()
        self.gl_context = None


def _create_context(width: int, height: int):
    requested = os.environ.get("MOJIVE_GL", "").strip().lower()
    if requested not in {"", "auto", "egl", "glfw", "native"}:
        raise ValueError(f"Unsupported MOJIVE_GL backend: {requested}")
    auto = requested in {"", "auto"}
    use_egl = requested == "egl" or (auto and sys.platform.startswith("linux"))
    failures: list[tuple[str, Exception]] = []
    if use_egl:
        try:
            return _StandaloneContext("egl")
        except Exception as exc:
            failures.append(("EGL", exc))
            if not auto:
                raise _context_error(
                    requested, failures, "Explicit EGL selection was preserved."
                ) from exc
            if not any(os.environ.get(key) for key in ("DISPLAY", "WAYLAND_DISPLAY")):
                raise _context_error(
                    requested, failures, "GLFW fallback requires an X11 or Wayland display."
                ) from exc
            if threading.current_thread() is not threading.main_thread():
                raise _context_error(
                    requested, failures, "GLFW fallback requires the main thread."
                ) from exc
    try:
        context = _GLFWContext(width, height)
    except Exception as exc:
        failures.append(("GLFW", exc))
        raise _context_error(requested, failures) from exc
    if failures:
        try:
            warnings.warn(
                f"Mojive EGL context creation failed ({failures[0][1]}); "
                "using a hidden GLFW window instead. This requires a desktop display and may "
                "select a different GPU. Set MOJIVE_GL=egl to require EGL or MOJIVE_GL=glfw "
                "to select GLFW explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:
            context.close()
            raise
    return context


def _context_error(requested: str, failures: list[tuple[str, Exception]], note: str = ""):
    details = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in failures)
    return RuntimeError(
        f"Mojive could not create an OpenGL 3.3 context (MOJIVE_GL={requested or 'auto'}, "
        f"platform={sys.platform}). {details}. {note} "
        "On an X11/Wayland desktop, try MOJIVE_GL=glfw on the main thread. "
        "On a server without a display, check the GPU driver, EGL libraries, and container GPU "
        "access; a hidden GLFW window is not a display-free fallback. "
        "See docs/reference/configuration.md#render-backend-requirements."
    )


def _select_backend(width: int, height: int, samples: int, renderer: str | None = None):
    """Create the explicitly selected or environment-configured render backend.

    Returns ``(context, backend)``; ``context`` is ``None`` for backends that
    manage no GL state of their own (the webgpu backend needs no window, EGL,
    or GLFW at all).
    """
    requested = render_backend_name(renderer)
    if requested in {"wgpu", "webgpu"}:
        from .webgpu.backend import WgpuBackend

        return None, WgpuBackend(max(1, width), max(1, height), samples, gpu_timing=False)
    if requested not in {"", "opengl"}:
        raise ValueError(f"Unsupported MOJIVE_BACKEND: {requested}")
    from .opengl.backend import OpenGLBackend

    context = _create_context(width, height)
    try:
        with context.current():
            backend = OpenGLBackend(context.gl_context, max(1, width), max(1, height), samples)
    except Exception:
        context.close()
        raise
    return context, backend
