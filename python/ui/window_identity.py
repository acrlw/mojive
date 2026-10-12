"""Keep desktop application identifiers independent of localized window titles."""

import sys
from contextlib import contextmanager


@contextmanager
def window_identity(glfw):
    hints = ()
    if sys.platform.startswith("linux"):
        platform = glfw.get_platform()
        if platform == glfw.PLATFORM_X11:
            hints = ((glfw.X11_CLASS_NAME, "Mojive"), (glfw.X11_INSTANCE_NAME, "mojive"))
        elif platform == glfw.PLATFORM_WAYLAND:
            hints = ((glfw.WAYLAND_APP_ID, "mojive"),)
    try:
        for hint, value in hints:
            glfw.window_hint_string(hint, value)
        yield
    finally:
        for hint, _ in hints:
            glfw.window_hint_string(hint, "")
