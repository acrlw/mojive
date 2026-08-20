"""wgpu viewer window: GLFW NO_API surface, wgpu canvas context, imgui over wgpu.

Counterpart to ``ui/window.py`` for the wgpu backend, exposing the same
external contract so ``ViewerApp`` runs unchanged.  The GLFW window is created
with ``GLFW_CLIENT_API=GLFW_NO_API`` (no GL context exists); the platform
present-info query comes from rendercanvas and the surface is driven through
``wgpu.gpu.get_canvas_context`` directly, so rendercanvas's canvas/loop
machinery (forced show, asyncio scheduling, callback ownership) stays out of
the way of the viewer's explicit frame loop.

Presentation composites imgui into a persistent frame texture and copies it
into the swapchain texture.  That extra copy keeps ``read_frame()`` well
defined — reading a swapchain texture after present is not — and matches the
GL ``read_frame`` contract (full window including panels, bottom row first).

The surface format is deliberately non-sRGB: the scene color target is already
gamma-encoded by ``finish_color`` and imgui colors are display-domain, so an
sRGB surface would apply a second hardware encode (double gamma).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import glfw
import numpy as np
import wgpu
from imgui_bundle import imgui
from imgui_bundle.python_backends import compute_fb_scale
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from rendercanvas.glfw import get_glfw_present_info
from wgpu.utils.imgui import ImguiWgpuBackend

from ..log import get_logger
from . import native_drop
from . import theme as theme_mod
from . import window as _window_module
from .window import ResizeLatch, Window, WindowConfig

if TYPE_CHECKING:
    from ..types import ViewportImage

log = get_logger("window_wgpu")


def _upstream_needs_imgui_192_fix() -> bool:
    """Whether the installed wgpu imgui backend hits the imgui 1.92 removal.

    wgpu 0.32's ``ImguiWgpuBackend.render`` reads ``ImDrawData.cmd_lists_count``,
    which imgui 1.92 removed (AttributeError on the first rendered frame).  The
    vendored override applies only while both sides of the mismatch are
    present; once upstream stops referencing the removed attribute the
    subclass defers to it.
    """
    return "cmd_lists_count" in ImguiWgpuBackend.render.__code__.co_names and not hasattr(
        imgui.ImDrawData, "cmd_lists_count"
    )


class _WgpuImguiBackend(ImguiWgpuBackend):
    """wgpu's imgui backend with the imgui 1.92 draw-data guard vendored in.

    The render() body below is wgpu 0.32's with the single broken line fixed
    (``draw_data.cmd_lists_count`` -> ``draw_data.cmd_lists.size()``); patching
    site-packages is not an option, so the fixed copy lives here.
    """

    def render(self, draw_data, render_pass) -> None:
        if not _upstream_needs_imgui_192_fix():
            return super().render(draw_data, render_pass)

        self._clear_pending_textures()

        if draw_data is None:
            return

        display_width, display_height = draw_data.display_size
        fb_width = int(display_width * draw_data.framebuffer_scale.x)
        fb_height = int(display_height * draw_data.framebuffer_scale.y)

        if fb_width <= 0 or fb_height <= 0 or draw_data.cmd_lists.size() == 0:
            return

        if draw_data.textures is not None:
            for tex in draw_data.textures:
                if tex.status != imgui.ImTextureStatus.ok:
                    self._update_texture(tex)

        self._update_vertex_buffer(draw_data)

        self._set_render_state(draw_data)
        render_pass.set_viewport(0, 0, fb_width, fb_height, 0, 1)

        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_vertex_buffer(0, self._vertex_buffer)
        if imgui.INDEX_SIZE == 2:
            index_fmt = wgpu.IndexFormat.uint16
        else:
            index_fmt = wgpu.IndexFormat.uint32
        render_pass.set_index_buffer(self._index_buffer, index_fmt, 0)
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.set_blend_constant((0.0, 0.0, 0.0, 0.0))

        global_vtx_offset = 0
        global_idx_offset = 0

        clip_scale = draw_data.framebuffer_scale
        clip_off = draw_data.display_pos

        for commands in draw_data.cmd_lists:
            for command in commands.cmd_buffer:
                tex_id = command.tex_ref.get_tex_id()

                tex_view = self._texture_views[tex_id]

                texture_bind_group = getattr(tex_view, "__imgui_bind_group", None)

                if texture_bind_group is None:
                    texture_bind_group = self._device.create_bind_group(
                        layout=self._texture_bind_group_layout,
                        entries=[
                            {
                                "binding": 0,
                                "resource": tex_view,
                            }
                        ],
                    )
                    # cache the bind group
                    tex_view.__imgui_bind_group = texture_bind_group

                render_pass.set_bind_group(1, texture_bind_group)

                clip_rect = command.clip_rect
                clip_min = [
                    (clip_rect.x - clip_off.x) * clip_scale.x,
                    (clip_rect.y - clip_off.y) * clip_scale.y,
                ]
                clip_max = [
                    (clip_rect.z - clip_off.x) * clip_scale.x,
                    (clip_rect.w - clip_off.y) * clip_scale.y,
                ]
                if clip_min[0] < 0:
                    clip_min[0] = 0
                if clip_min[1] < 0:
                    clip_min[1] = 0
                if clip_max[0] > fb_width:
                    clip_max[0] = fb_width
                if clip_max[1] > fb_height:
                    clip_max[1] = fb_height

                if clip_max[0] - clip_min[0] <= 0 or clip_max[1] - clip_min[1] <= 0:
                    continue

                render_pass.set_scissor_rect(
                    int(clip_min[0]),
                    int(clip_min[1]),
                    int(clip_max[0] - clip_min[0]),
                    int(clip_max[1] - clip_min[1]),
                )

                render_pass.draw_indexed(
                    command.elem_count,
                    1,
                    command.idx_offset + global_idx_offset,
                    command.vtx_offset + global_vtx_offset,
                    0,
                )

            global_vtx_offset += commands.vtx_buffer.size()
            global_idx_offset += commands.idx_buffer.size()


class _GlfwInputAdapter:
    """imgui input translation over raw GLFW callbacks for a NO_API window.

    imgui_bundle's GlfwRenderer couples this input half with an OpenGL
    renderer (its __init__ creates GL device objects), so it cannot serve a
    window without a GL context.  This adapter reuses its key table and
    mirrors its callbacks; ``process_inputs()`` plays the same role in
    ``begin_frame()`` as GlfwRenderer's does for the GL window.
    """

    def __init__(self, window: Any) -> None:
        self.window = window
        self.io = imgui.get_io()
        self.key_map: dict[Any, Any] = {}
        GlfwRenderer._map_keys(self)  # fills key_map only; no GL involved

        glfw.set_key_callback(window, self.keyboard_callback)
        glfw.set_cursor_pos_callback(window, self.mouse_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
        glfw.set_char_callback(window, self.char_callback)
        glfw.set_scroll_callback(window, self.scroll_callback)

        def get_clipboard_text(_ctx: Any) -> str:
            return glfw.get_clipboard_string(self.window).decode()

        def set_clipboard_text(_ctx: Any, text: str) -> None:
            glfw.set_clipboard_string(self.window, text)

        imgui.get_platform_io().platform_get_clipboard_text_fn = get_clipboard_text
        imgui.get_platform_io().platform_set_clipboard_text_fn = set_clipboard_text
        self._gui_time = None

    def keyboard_callback(
        self, window: Any, glfw_key: int, scancode: int, action: int, mods: int
    ) -> None:
        io = self.io
        if glfw_key not in self.key_map:
            return
        imgui_key = self.key_map[glfw_key]
        down = action != glfw.RELEASE
        io.add_key_event(imgui_key, down)

        # Handle modifiers, since ImGui has an additional mod_ctrl / shift / etc
        if imgui_key == imgui.Key.left_ctrl or imgui_key == imgui.Key.right_ctrl:
            io.add_key_event(imgui.Key.mod_ctrl, down)
        if imgui_key == imgui.Key.left_shift or imgui_key == imgui.Key.right_shift:
            io.add_key_event(imgui.Key.mod_shift, down)
        if imgui_key == imgui.Key.left_alt or imgui_key == imgui.Key.right_alt:
            io.add_key_event(imgui.Key.mod_alt, down)
        if imgui_key == imgui.Key.left_super or imgui_key == imgui.Key.right_super:
            io.add_key_event(imgui.Key.mod_super, down)

    def char_callback(self, window: Any, char: int) -> None:
        if 0 < char < 0x10000:
            self.io.add_input_character(char)

    def mouse_callback(self, *args: Any, **kwargs: Any) -> None:
        if glfw.get_window_attrib(self.window, glfw.FOCUSED):
            mouse_pos = glfw.get_cursor_pos(self.window)
            self.io.add_mouse_pos_event(mouse_pos[0], mouse_pos[1])
        else:
            self.io.add_mouse_pos_event(-1, -1)

    def mouse_button_callback(self, window: Any, button: int, action: int, mods: int) -> None:
        self.io.add_mouse_button_event(button, action == glfw.PRESS)

    def scroll_callback(self, window: Any, x_offset: float, y_offset: float) -> None:
        self.io.add_mouse_wheel_event(x_offset, y_offset)

    def process_inputs(self) -> None:
        io = self.io

        window_size = glfw.get_window_size(self.window)
        fb_size = glfw.get_framebuffer_size(self.window)

        io.display_size = imgui.ImVec2(*window_size)
        io.display_framebuffer_scale = imgui.ImVec2(*compute_fb_scale(window_size, fb_size))

        current_time = glfw.get_time()
        if self._gui_time:
            io.delta_time = current_time - self._gui_time
        else:
            io.delta_time = 1.0 / 60.0
        if io.delta_time <= 0.0:
            io.delta_time = 1.0 / 1000.0
        self._gui_time = current_time


class WgpuWindow(Window):
    """``Window`` contract over a wgpu surface instead of a GL context."""

    def __init__(self, config: WindowConfig | None = None, device: Any = None) -> None:
        _window_module._load_gl_deps()  # populate the base module's lazy globals
        self.config = config or WindowConfig()
        configured_scale = os.environ.get("FORGE_VIEWER_UI_SCALE")
        self._scale_override = float(configured_scale) if configured_scale else self.config.ui_scale
        if self._scale_override is not None and self._scale_override <= 0.0:
            raise ValueError("UI scale must be positive")
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        # Share the GL window module's live-window count so glfw.terminate()
        # runs when the last window of either kind closes.
        _window_module._live_windows += 1

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)

        handle = glfw.create_window(
            self.config.width, self.config.height, self.config.title, None, None
        )
        if not handle:
            glfw.terminate()
            raise RuntimeError("Failed to create a GLFW window (CLIENT_API=NO_API)")
        self._window = handle
        self._shown = False
        self._destroyed = False
        self._frame_index = 0
        self._readback: np.ndarray | None = None
        self._file_drops: list[Path] = []
        self._file_drag_active = False
        self._native_drop_token = 0

        self._device = device if device is not None else wgpu.utils.get_default_device()
        self._gpu_context: Any = None
        self._surface_format = ""
        self._rgb_channels = (0, 1, 2)
        self._frame_tex: Any = None
        self._frame_tex_view: Any = None
        self._frame_tex_size = (0, 0)
        self._imgui_backend: _WgpuImguiBackend | None = None
        self._scene_view: Any = None
        self._scene_ref: Any = None
        self._vsync_interval: float | None = None
        self._next_frame_at: float | None = None

        self.set_vsync(self.config.vsync)

        self._content_scale = 1.0
        self._ui_scale = 1.0
        self._pixel_scale = 1.0
        self._style_scale = 1.0
        self._scale_generation = 0
        self._refresh_scales()

        imgui.create_context()
        io = imgui.get_io()
        if self.config.docking:
            io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard

        ini = self.config.ini_path or ""
        self._ini_existed = bool(ini) and Path(ini).exists()
        io.set_ini_filename(ini)

        self._impl = _GlfwInputAdapter(handle)
        glfw.set_drop_callback(handle, self._on_file_drop)
        self._native_drop_token = native_drop.install(glfw, handle, self)

        self._impl.process_inputs()
        theme_mod.apply(imgui, ui_scale=self._style_scale)
        imgui.get_style().font_scale_dpi = self._style_scale
        self._applied_style_scale = self._style_scale
        self._load_fonts(io)

        self.dockspace_id = 0
        self._layout_done = False
        self.latch = ResizeLatch()

        self._gpu_context = wgpu.gpu.get_canvas_context(get_glfw_present_info(handle))
        self._gpu_context.set_physical_size(*self.size_pixels)
        self._configure_surface()
        self._imgui_backend = _WgpuImguiBackend(self._device, self._surface_format)
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
        # wgpu-py 0.32 picks the immediate present mode and exposes no knob,
        # so vsync is emulated by pacing the frame loop (see _pace_frame).
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
        """Bind the backend's color view with the imgui renderer (cached)."""
        view = image.payload
        if view is None:
            return imgui.ImTextureRef(image.texture_id)
        if view is not self._scene_view:
            if self._scene_ref is not None:
                self._imgui_backend.unregister_texture(self._scene_ref)
            self._scene_ref = self._imgui_backend.register_texture(view)
            self._scene_view = view
        return self._scene_ref

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
        imgui.render()
        fb_w, fb_h = self.size_pixels
        frame = None
        if fb_w > 0 and fb_h > 0:
            self._gpu_context.set_physical_size(fb_w, fb_h)
            frame_view = self._frame_view(fb_w, fb_h)
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
            self._imgui_backend.render(imgui.get_draw_data(), render_pass)
            render_pass.end()
            surface = self._gpu_context.get_current_texture()
            encoder.copy_texture_to_texture(
                {"texture": self._frame_tex}, {"texture": surface}, (fb_w, fb_h, 1)
            )
            self._device.queue.submit([encoder.finish()])
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
        self._readback[:] = texels[::-1, :, self._rgb_channels]
        return self._readback

    def close(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        native_drop.uninstall(self._native_drop_token)
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
        glfw.destroy_window(self._window)
        _window_module._live_windows = max(0, _window_module._live_windows - 1)
        if _window_module._live_windows == 0:
            glfw.terminate()
