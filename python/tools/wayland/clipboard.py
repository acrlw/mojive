"""A separate GLFW client for clipboard ownership and transfer acceptance."""

import sys
import time

import glfw

if not glfw.init():
    raise SystemExit("GLFW initialization failed")
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
window = glfw.create_window(200, 100, "Mojive clipboard test", None, None)
try:
    glfw.show_window(window)
    for _ in range(10):
        glfw.poll_events()
        time.sleep(0.01)
    if sys.argv[1] == "copy":
        glfw.set_clipboard_string(None, sys.argv[2])
        glfw.hide_window(window)
        glfw.poll_events()
        print("ready", flush=True)
        while True:
            glfw.wait_events_timeout(0.05)
    else:
        value = glfw.get_clipboard_string(None)
        if value is None:
            raise SystemExit("No clipboard selection")
        sys.stdout.buffer.write(value)
finally:
    glfw.destroy_window(window)
    glfw.terminate()
