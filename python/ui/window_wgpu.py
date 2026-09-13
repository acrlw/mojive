"""GLFW NO_API window with imgui composition and wgpu presentation.

Frames are composed into a readable texture before presentation. The
non-sRGB surface preserves the display-domain colors emitted by the renderer.
"""

from __future__ import annotations

import os
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import wgpu

from ..log import get_logger
from . import native_drop
from . import theme as theme_mod
from . import window as _window_module
from .wgpu_backend import WgpuImguiBackend
from .window import GlfwInputAdapter, ResizeLatch, Window, WindowConfig

if TYPE_CHECKING:
    from ..types import ViewportImage

log = get_logger("window_wgpu")

glfw: Any = None
imgui: Any = None
get_glfw_present_info: Any = None


def _default_device() -> wgpu.GPUDevice:
    with suppress(RuntimeError):
        wgpu.utils.preconfigure_default_device(
            "mojive GPU timing", preferred_features={wgpu.FeatureName.timestamp_query}
        )
    return wgpu.utils.get_default_device()


def _load_window_deps() -> None:
    global glfw, imgui, get_glfw_present_info
    if glfw is not None:
        return
    _window_module._load_window_deps()
    from rendercanvas.glfw import get_glfw_present_info as _get_glfw_present_info

    glfw = _window_module.glfw
    imgui = _window_module.imgui
    get_glfw_present_info = _get_glfw_present_info


class WgpuWindow(Window):
    """``Window`` contract over a wgpu surface instead of a GL context."""

    def __init__(self, config: WindowConfig | None = None, device: Any = None) -> None:
        _load_window_deps()
        self.config = config or WindowConfig()
        configured_scale = os.environ.get("MOJIVE_UI_SCALE")
        self._scale_override = float(configured_scale) if configured_scale else self.config.ui_scale
        if self._scale_override is not None and self._scale_override <= 0.0:
            raise ValueError("UI scale must be positive")
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        # Share the GL window module's live-window count so glfw.terminate()
        # runs when the last GLFW or wgpu window closes.
        _window_module._live_windows += 1

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
            glfw.terminate()
            raise RuntimeError("Failed to create a GLFW window (CLIENT_API=NO_API)")
        self._window = handle
        self._maximized = bool(glfw.get_window_attrib(handle, glfw.MAXIMIZED))
        self._shown = False
        self._destroyed = False
        self._frame_index = 0
        self._readback: np.ndarray | None = None
        self._file_drops: list[Path] = []
        self._file_drag_active = False
        self._native_drop_token = 0

        self._device = device if device is not None else _default_device()
        self._gpu_context: Any = None
        self._surface_format = ""
        self._rgb_channels = (0, 1, 2)
        self._frame_tex: Any = None
        self._frame_tex_view: Any = None
        self._frame_tex_size = (0, 0)
        self._imgui_backend: WgpuImguiBackend | None = None
        self._viewport_textures: dict[int, tuple[Any, Any]] = {}
        self._vsync_interval: float | None = None
        self._next_frame_at: float | None = None

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

        # Window pacing (or the passive display scheduler) owns the frame rate.
        # wgpu otherwise defaults to FIFO and waits on native vsync as well.
        self._gpu_context = wgpu.gpu.get_canvas_context(
            {**get_glfw_present_info(handle), "vsync": False}
        )
        self._gpu_context.set_physical_size(*self.size_pixels)
        self._configure_surface()
        self._imgui_backend = WgpuImguiBackend(self._device, self._surface_format)
        # ImguiWgpuBackend.__init__ clears the ini filename; restore it.
        io.set_ini_filename(ini)

    @property
    def device(self) -> Any:
        return self._device

    def _configure_surface(self) -> None:
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_DST
        for fmt in ("bgra8unorm", "rgba8unorm"):
            try:
                self._gpu_context.configure(device=self._device, format=fmt, usage=usage)
            except ValueError:
                continue
            self._surface_format = fmt
            break
        else:
            fmt = self._gpu_context.get_preferred_format(self._device.adapter)
            log.warning("No non-sRGB surface format; falling back to {} (colors may shift)", fmt)
            self._gpu_context.configure(device=self._device, format=fmt, usage=usage)
            self._surface_format = fmt
        self._rgb_channels = (2, 1, 0) if self._surface_format.startswith("bgra") else (0, 1, 2)

    def make_current(self) -> None:
        # NO_API window: there is no GL context to make current.
        pass

    def set_vsync(self, on: bool) -> None:
        # The surface requests immediate presentation; software pacing can be
        # toggled without recreating its swapchain (see _pace_frame).
        self._vsync = bool(on)
        self._next_frame_at = None

    def _pace_frame(self) -> None:
        if self._vsync_interval is None:
            refresh = 0
            monitor = glfw.get_window_monitor(self._window) or glfw.get_primary_monitor()
            if monitor:
                mode = glfw.get_video_mode(monitor)
                if mode is not None:
                    refresh = mode.refresh_rate
            self._vsync_interval = 1.0 / (refresh if refresh > 0 else 60)
        now = time.perf_counter()
        target = self._next_frame_at
        if target is None or now >= target + self._vsync_interval:
            self._next_frame_at = now + self._vsync_interval  # resync when behind
            return
        if now < target:
            time.sleep(target - now)
        self._next_frame_at = target + self._vsync_interval

    def viewport_texture_ref(self, image: ViewportImage) -> Any:
        """Bind a backend color view with the imgui renderer."""
        view = image.payload
        if view is None:
            return imgui.ImTextureRef(image.texture_id)
        key = id(view)
        cached = self._viewport_textures.get(key)
        if cached is None or cached[0] is not view:
            cached = (view, self._imgui_backend.register_texture(view))
            self._viewport_textures[key] = cached
        return cached[1]

    def _frame_view(self, width: int, height: int) -> Any:
        if self._frame_tex_size != (width, height):
            if self._frame_tex is not None:
                self._frame_tex.destroy()
            self._frame_tex = self._device.create_texture(
                size=(width, height, 1),
                format=self._surface_format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            )
            self._frame_tex_view = self._frame_tex.create_view()
            self._frame_tex_size = (width, height)
        return self._frame_tex_view

    def end_frame(self, *, readback: bool = False) -> np.ndarray | None:
        imgui.set_current_context(self._imgui_context)
        imgui.render()
        draw_data = imgui.get_draw_data()
        fb_w, fb_h = self.size_pixels
        frame = None
        if fb_w > 0 and fb_h > 0:
            self._gpu_context.set_physical_size(fb_w, fb_h)
            frame_view = self._frame_view(fb_w, fb_h)
            try:
                surface = self._gpu_context.get_current_texture()
            except wgpu.DrawCancelled:
                surface = None
            encoder = self._device.create_command_encoder()
            render_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": frame_view,
                        "clear_value": self.config.clear_color,
                        "load_op": "clear",
                        "store_op": "store",
                    }
                ]
            )
            self._imgui_backend.render(draw_data, render_pass, (fb_w, fb_h))
            render_pass.end()
            if surface is not None:
                encoder.copy_texture_to_texture(
                    {"texture": self._frame_tex}, {"texture": surface}, (fb_w, fb_h, 1)
                )
            self._device.queue.submit([encoder.finish()])
            if surface is not None:
                self._gpu_context.present()
            if readback:
                frame = self.read_frame()
            if self._vsync:
                self._pace_frame()
        self._frame_index += 1
        return frame

    def read_frame(self) -> np.ndarray | None:
        # Mirrors the GL Window.read_frame contract: bottom row first, RGB.
        if self._frame_tex is None:
            return None
        w, h = self._frame_tex_size
        row_bytes = (w * 4 + 255) // 256 * 256
        data = self._device.queue.read_texture(
            {"texture": self._frame_tex, "origin": (0, 0, 0)},
            {"bytes_per_row": row_bytes, "rows_per_image": h},
            (w, h, 1),
        )
        texels = np.frombuffer(data, np.uint8).reshape(h, row_bytes)[:, : w * 4].reshape(h, w, 4)
        if self._readback is None or self._readback.shape[:2] != (h, w):
            self._readback = np.empty((h, w, 3), np.uint8)
        # Advanced channel indexing creates a planar full-frame temporary.
        # Copy strided channels directly into the reusable packed RGB buffer.
        for destination, source in enumerate(self._rgb_channels):
            np.copyto(self._readback[..., destination], texels[::-1, :, source])
        return self._readback

    def close(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        imgui.set_current_context(self._imgui_context)
        native_drop.uninstall(self._native_drop_token)
        self._close_window_resources()

    def _close_window_resources(self) -> None:
        for _view, texture_ref in self._viewport_textures.values():
            self._imgui_backend.unregister_texture(texture_ref)
        self._viewport_textures.clear()
        self._imgui_backend.close()
        if self._gpu_context is not None:
            # The surface holds the native window handle, so it must be
            # released before the GLFW window is destroyed — otherwise the
            # canvas context's __del__ releases it after glfw.terminate()
            # and segfaults (pygfx#642). _release() is idempotent.
            try:
                self._gpu_context._release()
            except Exception as e:
                log.debug("Failed to release the wgpu surface: {}", e)
            self._gpu_context = None
        if self._frame_tex is not None:
            self._frame_tex.destroy()
            self._frame_tex = None
        imgui.destroy_context(self._imgui_context)
        glfw.destroy_window(self._window)
        _window_module._live_windows = max(0, _window_module._live_windows - 1)
        if _window_module._live_windows == 0:
            glfw.terminate()
