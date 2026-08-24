from __future__ import annotations

import os

import pytest

from forge_viewer.render.forge import gl_native as G


def _load_glfw():
    from imgui_bundle._glfw_set_search_path import _glfw_set_search_path

    _glfw_set_search_path()
    import glfw

    return glfw


glfw = _load_glfw()

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


@pytest.fixture(autouse=True, scope="session")
def _english_ui():
    previous = os.environ.get("FORGE_VIEWER_LANGUAGE")
    os.environ["FORGE_VIEWER_LANGUAGE"] = "en"
    yield
    if previous is None:
        del os.environ["FORGE_VIEWER_LANGUAGE"]
    else:
        os.environ["FORGE_VIEWER_LANGUAGE"] = previous


@pytest.fixture(scope="session")
def backend_name():

    requested = os.environ.get("FORGE_VIEWER_BACKEND", "").strip().lower()
    if requested == "webgpu":
        requested = "wgpu"
    return requested or "forge"


@pytest.fixture(scope="session")
def require_forge(backend_name):

    if backend_name != "forge":
        pytest.skip("GL-internals test, forge backend only")


@pytest.fixture(scope="session")
def _gl_session():

    if glfw is None or moderngl is None:
        pytest.skip("glfw or moderngl unavailable")
    if not glfw.init():
        pytest.skip("GLFW initialization failed")
    for hint, value in (
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        (glfw.OPENGL_FORWARD_COMPAT, True),
        (glfw.VISIBLE, False),
    ):
        glfw.window_hint(hint, value)
    window = glfw.create_window(256, 192, "forge gpu tests", None, None)
    if not window:
        glfw.terminate()
        pytest.skip("OpenGL 3.3 core context unavailable")
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    G.native().drain_errors()
    yield ctx
    glfw.terminate()


@pytest.fixture
def gl_ctx(_gl_session):

    ctx = _gl_session

    G.native().drain_errors()
    ctx.wireframe = False
    ctx.front_face = "ccw"
    ctx.cull_face = "back"
    ctx.depth_func = "<"
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    ctx.enable_only(moderngl.NOTHING)
    ctx.multisample = True
    G.native().depth_mask(True)
    return ctx
