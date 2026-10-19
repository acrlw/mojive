"""Context-owned lifetime for ImGui Bundle's compiled OpenGL renderer."""

from collections.abc import Callable
from contextlib import suppress

from imgui_bundle import imgui


class OpenGLImGuiRenderer:
    """Leave GLFW input with Mojive and submit draw lists through the native backend."""

    def __init__(self, make_current: Callable[[], None]) -> None:
        self._context = imgui.get_current_context()
        if self._context is None:
            raise RuntimeError("ImGui rendering requires an active context")
        self._make_current = make_current
        self._initialized = False
        self._activate()
        io = imgui.get_io()
        if io.backend_renderer_user_data is not None:
            raise RuntimeError("This ImGui context already has a renderer")
        try:
            if not imgui.backends.opengl3_init("#version 330"):
                raise RuntimeError("Failed to initialize ImGui's OpenGL renderer")
            self._initialized = True
            if not io.backend_flags & imgui.BackendFlags_.renderer_has_textures:
                raise RuntimeError("ImGui's OpenGL renderer requires dynamic font texture support")
            if not imgui.backends.opengl3_create_device_objects():
                raise RuntimeError("Failed to create ImGui's OpenGL resources")
        except BaseException:
            # Initialization may have allocated backend data before failing.
            self._initialized = io.backend_renderer_user_data is not None
            with suppress(Exception):
                self.shutdown()
            raise

    def _activate(self) -> None:
        self._make_current()
        imgui.set_current_context(self._context)

    def new_frame(self) -> None:
        self._activate()
        imgui.backends.opengl3_new_frame()

    def render(self, draw_data) -> None:
        self._activate()
        # The backend consumes dynamic atlas requests before submitting draw commands.
        imgui.backends.opengl3_render_draw_data(draw_data)

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._activate()
        try:
            imgui.backends.opengl3_shutdown()
        finally:
            self._initialized = False
