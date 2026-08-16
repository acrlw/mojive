from __future__ import annotations

import pytest

from forge_viewer.render.forge import gl_native as G

try:
    import glfw
except ImportError:  # pragma: no cover
    glfw = None

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


@pytest.fixture(scope="session")
def _gl_session():

    if glfw is None or moderngl is None:
        pytest.skip("没装 glfw / moderngl")
    if not glfw.init():
        pytest.skip("glfw 起不来")
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
        pytest.skip("建不出 3.3 core 上下文")
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
