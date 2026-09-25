"""GLFW platform window with native GPU composition of the existing ImGui UI."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from . import native_drop
from . import theme as theme_mod
from . import window as _window_module
from .window import GlfwInputAdapter, ResizeLatch, Window, WindowConfig

glfw: Any = None
imgui: Any = None


def _load_window_deps():
    global glfw, imgui
    _window_module._load_window_deps()
    glfw, imgui = _window_module.glfw, _window_module.imgui


class NativeWindow(Window):
    """Window events and ImGui stay on the UI thread; GPU work has a native owner."""

    def __init__(
        self, config: WindowConfig | None = None, *, device: Any = None, device_factory: Any = None
    ) -> None:
        self._destroyed = False
        self._window = None
        self._imgui_context = None
        self._native_drop_token = 0
        self.device = device
        self._device_factory = device_factory
        self.api = self.runtime = None
        if device is not None:
            self.api, self.runtime = device.api, device.runtime
        self._surface = self._frame_target = None
        self._imgui_backend = None
        self._counted_window = False
        try:
            self._initialize(config)
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def _initialize(self, config):
        _load_window_deps()
        self.config = config or WindowConfig()
        configured_scale = os.environ.get("MOJIVE_UI_SCALE")
        self._scale_override = float(configured_scale) if configured_scale else self.config.ui_scale
        if self._scale_override is not None and self._scale_override <= 0.0:
            raise ValueError("UI scale must be positive")
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        # Share the GL window module's live-window count so glfw.terminate()
        # runs when the last GLFW or native window closes.
        _window_module._live_windows += 1
        self._counted_window = True
        self._wayland = (
            sys.platform.startswith("linux") and glfw.get_platform() == glfw.PLATFORM_WAYLAND
        )
        if self.device is None:
            self.device = self._device_factory(wayland=self._wayland)
            self.api, self.runtime = self.device.api, self.device.runtime

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)

        try:
            handle = glfw.create_window(
                self.config.width, self.config.height, self.config.title, None, None
            )
        finally:
            # Window hints are process-global: leave CLIENT_API at its default
            # so a later GL window (opengl backend) still gets a context.
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.TRUE)
        if not handle:
            raise RuntimeError("Failed to create a GLFW window (CLIENT_API=NO_API)")
        self._window = handle
        _window_module._restore_requested_size(handle, self.config.width, self.config.height)
        self._maximized = bool(glfw.get_window_attrib(handle, glfw.MAXIMIZED))
        self._shown = False
        self._destroyed = False
        self._frame_index = 0
        self._readback: np.ndarray | None = None
        self._file_drops: list[Path] = []
        self._file_drag_active = False
        self._native_drop_token = 0

        self._surface = None
        self._frame_target = None
        self._frame_token = None
        self._frame_size = (0, 0)
        self._imgui_backend = None

        self.set_vsync(self.config.vsync)

        self._content_scale = 1.0
        self._ui_scale = 1.0
        self._pixel_scale = 1.0
        self._style_scale = 1.0
        self._scale_generation = 0
        self._refresh_scales()

        self._imgui_context = imgui.create_context()
        imgui.set_current_context(self._imgui_context)
        io = imgui.get_io()
        if self.config.docking:
            io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        io.config_windows_move_from_title_bar_only = True

        ini = self.config.ini_path or ""
        self._ini_existed = bool(ini) and Path(ini).exists()
        io.set_ini_filename(ini)

        self._impl = GlfwInputAdapter(handle)
        self._input = self._impl
        glfw.set_drop_callback(handle, self._on_file_drop)
        self._native_drop_token = native_drop.install(glfw, handle, self)

        self._impl.process_inputs()
        theme_mod.apply(imgui, ui_scale=self._style_scale)
        self._applied_style_scale = self._style_scale
        self._load_fonts(io)

        self.dockspace_id = 0
        self._layout_done = False
        self.latch = ResizeLatch()

        from .native_backend import NativeImguiBackend

        self._imgui_backend = NativeImguiBackend(self.device)
        if not self._wayland:
            self._create_surface()

    def _create_surface(self):
        handle, display = self._native_handle()
        self._surface = self.runtime.create_surface(
            handle, *self.size_pixels, display, self._wayland
        )
        self.set_vsync(self._vsync)

    def show(self) -> None:
        super().show()
        # Wayland cannot present to an unmapped wl_surface. GLFW's show completes
        # the initial xdg-shell configuration; hidden windows only render offscreen.
        if self._surface is None:
            self._create_surface()

    def make_current(self) -> None:
        # NO_API window: there is no GL context to make current.
        pass

    def set_vsync(self, on: bool) -> None:
        self._vsync = bool(on)
        if self._surface is not None:
            self.runtime.set_vsync(self._surface, self._vsync)

    def _native_handle(self):
        if sys.platform == "darwin":
            return int(glfw.get_cocoa_window(self._window)), 0
        if sys.platform == "win32":
            return int(glfw.get_win32_window(self._window)), 0
        if self._wayland:
            return int(glfw.get_wayland_window(self._window)), int(glfw.get_wayland_display())
        return int(glfw.get_x11_window(self._window)), int(glfw.get_x11_display())

    def viewport_texture_ref(self, image):
        return self._imgui_backend.register_texture(image.payload)

    def end_frame(self, *, readback=False):
        imgui.set_current_context(self._imgui_context)
        imgui.render()
        draw = imgui.get_draw_data()
        size = self.size_pixels
        result = None
        if min(size) > 0:
            if self._frame_size != size:
                if self._surface is not None:
                    self.runtime.resize(self._surface, *size)
                if self._frame_target is None:
                    self._frame_target = self.runtime.create_target(*size)
                else:
                    self.runtime.resize(self._frame_target, *size)
                self._frame_size = size
            vertices, indices, commands = self._imgui_backend.prepare(draw, size)
            self._frame_token = self.runtime.render_ui(
                *size, vertices, indices, commands, self._frame_target
            )
            # Both submissions sample GPU textures directly. CPU readback only occurs on request.
            if self._surface is not None:
                vertices, indices, commands = self._present_packet(size)
                self.runtime.render_ui(*size, vertices, indices, commands, self._surface)
            self.runtime.advance()
            if readback:
                result = self.read_frame()
        self.device.drain_logs()
        self._frame_index += 1
        return result

    def _present_packet(self, size):
        if getattr(self, "_present_size", None) != size:
            w, h = size
            data = np.array(
                [[0, 0, 0, 0, 0], [w, 0, 1, 0, 0], [w, h, 1, 1, 0], [0, h, 0, 1, 0]], np.float32
            )
            data.view(np.uint32)[:, 4] = 0xFFFFFFFF
            command = self.api.UiCommand()
            command.index_count = 6
            command.clip = [0, 0, w, h]
            command.texture = self.runtime.target_texture(self._frame_target)
            self._presentation = (
                data.view(np.uint8).reshape(-1),
                np.array([0, 1, 2, 0, 2, 3], np.uint32),
                [command],
            )
            self._present_size = size
        return self._presentation

    def read_frame(self):
        if self._frame_token is None:
            return None
        result = self.runtime.read(self._frame_token, self.api.Product.COLOR)
        return result.image[::-1].copy() if result.state == self.api.ReadbackState.READY else None

    def close(self):
        if self._destroyed:
            return
        self._destroyed = True
        if self._imgui_context is not None:
            imgui.set_current_context(self._imgui_context)
        native_drop.uninstall(self._native_drop_token)
        try:
            if self.device is not None:
                try:
                    if self._imgui_backend is not None:
                        self._imgui_backend.close()
                    for target in (self._frame_target, self._surface):
                        if target is not None:
                            self.runtime.destroy(target)
                    # Drain deferred destruction while the native OS window is alive.
                    for _ in range(3):
                        self.runtime.advance()
                finally:
                    self.device.release()
        finally:
            if self._imgui_context is not None:
                imgui.destroy_context(self._imgui_context)
            if self._window is not None:
                glfw.destroy_window(self._window)
            if self._counted_window:
                _window_module._live_windows = max(0, _window_module._live_windows - 1)
                if _window_module._live_windows == 0:
                    glfw.terminate()
