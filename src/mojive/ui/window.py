"""GLFW window lifecycle, scaling, and input state."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..log import get_logger
from . import fonts, native_drop
from . import theme as theme_mod

if TYPE_CHECKING:
    from ..types import ViewportImage

log = get_logger("window")


_live_windows = 0


glfw: Any = None
gl: Any = None
imgui: Any = None
GlfwRenderer: Any = None


def _install_glfw_clipboard_callbacks(glfw_api: Any, imgui_api: Any) -> None:
    """Connect ImGui to GLFW's process-wide clipboard without a legacy window argument."""

    def get_clipboard_text(_ctx: Any) -> str:
        value = glfw_api.get_clipboard_string(None)
        if value is None:
            return ""
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def set_clipboard_text(_ctx: Any, text: str) -> None:
        glfw_api.set_clipboard_string(None, text)

    platform_io = imgui_api.get_platform_io()
    platform_io.platform_get_clipboard_text_fn = get_clipboard_text
    platform_io.platform_set_clipboard_text_fn = set_clipboard_text


def _is_dock_tab_nav_target(nav_id: int, tab_id: int, docked: bool) -> bool:
    """Return whether ImGui navigation currently points at a dock tab."""

    return bool(docked and int(tab_id) != 0 and int(nav_id) == int(tab_id))


def _suppress_dock_tab_nav_cursor() -> bool:
    """Hide tab-only keyboard focus chrome before the dock host renders.

    Controls retain their normal navigation cursor because their IDs differ
    from the owning window's dock-tab ID.
    """

    if imgui is None:
        return False
    context = imgui.get_current_context()
    nav_window = context.nav_window
    if nav_window is None or not _is_dock_tab_nav_target(
        context.nav_id,
        nav_window.tab_id,
        nav_window.dock_node is not None,
    ):
        return False
    context.nav_cursor_visible = False
    return True


def layout_settings_path() -> Path:
    """Return the persistent ImGui layout file outside the working tree."""

    override = os.environ.get("MOJIVE_IMGUI_INI")
    if override:
        return Path(override).expanduser()
    config_override = os.environ.get("MOJIVE_CONFIG_DIR")
    if config_override:
        root = Path(config_override).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "mojive"
    elif os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "mojive"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "mojive"
    return root / "imgui.ini"


def layout_scale(ui_scale: float, framebuffer_scale: float) -> float:
    """Convert the physical UI scale into ImGui window coordinates."""
    return float(ui_scale) / max(float(framebuffer_scale), 1e-6)


def resolve_ui_scales(
    content_scale: float,
    framebuffer_scale: float,
    layout_override: float | None = None,
) -> tuple[float, float]:
    """Return physical and logical UI scales for the current display."""
    if layout_override is not None:
        style_scale = float(layout_override)
        return style_scale * float(framebuffer_scale), style_scale
    ui_scale = float(content_scale)
    return ui_scale, layout_scale(ui_scale, framebuffer_scale)


def resolve_context_api(requested: str) -> str:
    """Resolve the GLFW context creation API."""
    value = requested.strip().lower()
    if value in {"", "auto", "native", "glfw"}:
        return "native"
    if value == "egl":
        return "egl"
    raise ValueError(f"Unsupported MOJIVE_GL backend: {requested}")


def _load_window_deps() -> None:
    global glfw, imgui, GlfwRenderer
    if glfw is not None:
        return
    from imgui_bundle import imgui as _imgui
    from imgui_bundle._glfw_set_search_path import _glfw_set_search_path

    _glfw_set_search_path()
    import glfw as _glfw
    from imgui_bundle.python_backends.glfw_backend import GlfwRenderer as _GlfwRenderer

    glfw, imgui, GlfwRenderer = _glfw, _imgui, _GlfwRenderer


def _load_gl_deps() -> None:
    global gl
    _load_window_deps()
    if gl is not None:
        return
    from OpenGL import GL as _gl

    gl = _gl


@dataclass
class ResizeLatch:
    settle_seconds: float = 0.6
    warmup_frames: int = 30

    _committed: tuple[int, int] | None = field(default=None, init=False)
    _pending: tuple[int, int] | None = field(default=None, init=False)
    _pending_since: float = field(default=0.0, init=False)
    _frames: int = field(default=0, init=False)
    _rebuilds: int = field(default=0, init=False)

    @property
    def committed(self) -> tuple[int, int] | None:
        return self._committed

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def rebuilds(self) -> int:
        return self._rebuilds

    def update(
        self,
        size: tuple[int, int],
        now: float,
        *,
        immediate: bool = False,
    ) -> tuple[int, int] | None:
        self._frames += 1
        w, h = (max(1, int(size[0])), max(1, int(size[1])))
        target = (w, h)

        if target == self._committed:
            self._pending = None
            return None

        if immediate or self._frames <= self.warmup_frames:
            return self._commit(target)

        if target != self._pending:
            self._pending = target
            self._pending_since = now
            return None

        if now - self._pending_since >= self.settle_seconds:
            return self._commit(target)
        return None

    def reset(self) -> None:
        self._committed = None
        self._pending = None

    def _commit(self, size: tuple[int, int]) -> tuple[int, int]:
        self._committed = size
        self._pending = None
        self._rebuilds += 1
        return size


@dataclass(frozen=True)
class WindowConfig:
    title: str = "Mojive"
    width: int = 1600
    height: int = 900
    vsync: bool = True

    gl_major: int = 3
    gl_minor: int = 3
    samples: int = 0

    docking: bool = True
    ini_path: str | None = "imgui.ini"
    show_on_start: bool = True

    clear_color: tuple[float, float, float, float] = (0.09, 0.10, 0.11, 1.0)
    font_size_pt: float = 14.0
    ui_scale: float | None = None
    context_api: str = "auto"


class Window:
    def __init__(self, config: WindowConfig | None = None) -> None:
        _load_gl_deps()
        self.config = config or WindowConfig()
        configured_scale = os.environ.get("MOJIVE_UI_SCALE")
        self._scale_override = float(configured_scale) if configured_scale else self.config.ui_scale
        if self._scale_override is not None and self._scale_override <= 0.0:
            raise ValueError("UI scale must be positive")
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        global _live_windows
        _live_windows += 1

        requested_api = os.environ.get("MOJIVE_GL", self.config.context_api)
        self._context_api = resolve_context_api(requested_api)
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, self.config.gl_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, self.config.gl_minor)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, self.config.samples)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        glfw.window_hint(
            glfw.CONTEXT_CREATION_API,
            glfw.EGL_CONTEXT_API if self._context_api == "egl" else glfw.NATIVE_CONTEXT_API,
        )
        handle = glfw.create_window(
            self.config.width, self.config.height, self.config.title, None, None
        )
        if not handle:
            _live_windows = max(0, _live_windows - 1)
            glfw.terminate()
            raise RuntimeError(
                f"Failed to create an OpenGL {self.config.gl_major}.{self.config.gl_minor} "
                f"core context with GLFW {self._context_api}"
            )
        self._window = handle
        self._maximized = bool(glfw.get_window_attrib(handle, glfw.MAXIMIZED))
        self._shown = False
        self._destroyed = False
        self._frame_index = 0
        self._readback: np.ndarray | None = None
        self._file_drops: list[Path] = []
        self._file_drag_active = False
        self._native_drop_token = 0

        glfw.make_context_current(self._window)
        self.set_vsync(self.config.vsync)
        log.info(
            "OpenGL {}.{} core context created with GLFW {}",
            self.config.gl_major,
            self.config.gl_minor,
            self._context_api,
        )

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

        self._impl = GlfwRenderer(self._window)
        # imgui_bundle still passes the deprecated GLFW window argument.
        # Replace only its clipboard callbacks; input and GL rendering remain upstream.
        _install_glfw_clipboard_callbacks(glfw, imgui)
        glfw.set_drop_callback(self._window, self._on_file_drop)
        self._native_drop_token = native_drop.install(glfw, self._window, self)

        self._impl.process_inputs()
        theme_mod.apply(imgui, ui_scale=self._style_scale)
        self._applied_style_scale = self._style_scale
        self._load_fonts(io)

        self.dockspace_id = 0
        self._layout_done = False
        self.latch = ResizeLatch()

    def _load_fonts(self, io) -> None:
        self._font_atlas_scale = self._style_scale
        imgui.get_style().font_scale_dpi = 1.0
        self.font_report = fonts.load(
            imgui,
            io,
            size_pt=self.config.font_size_pt * self._font_atlas_scale,
        )
        for note in self.font_report.notes:
            log.warning("Font fallback: {}", note)

    @property
    def gl_context(self) -> Any:
        return self._window

    @property
    def context_api(self) -> str:
        return self._context_api

    def make_current(self) -> None:
        glfw.make_context_current(self._window)

    @property
    def gl_version(self) -> str:
        return gl.glGetString(gl.GL_VERSION).decode()

    @property
    def renderer_name(self) -> str:
        return gl.glGetString(gl.GL_RENDERER).decode()

    @property
    def ui_scale(self) -> float:
        return self._ui_scale

    @property
    def content_scale(self) -> float:
        return self._content_scale

    @property
    def style_scale(self) -> float:
        return self._style_scale

    @property
    def pixel_scale(self) -> float:
        return self._pixel_scale

    @property
    def scale_generation(self) -> int:
        return self._scale_generation

    def points_to_pixels(self, value: Any) -> Any:
        s = self._pixel_scale
        if isinstance(value, (int, float)):
            return float(value) * s
        return tuple(float(v) * s for v in value)

    def pixels_to_points(self, value: Any) -> Any:
        s = self._pixel_scale
        if isinstance(value, (int, float)):
            return float(value) / s
        return tuple(float(v) / s for v in value)

    @property
    def size_points(self) -> tuple[int, int]:
        w, h = glfw.get_window_size(self._window)
        return int(w), int(h)

    @property
    def size_pixels(self) -> tuple[int, int]:
        w, h = glfw.get_framebuffer_size(self._window)
        return int(w), int(h)

    def _refresh_scales(self) -> None:
        try:
            sx, sy = glfw.get_window_content_scale(self._window)
            content_scale = max(float(sx), float(sy)) or 1.0
        except Exception:
            content_scale = 1.0
        win_w, win_h = self.size_points
        fb_w, fb_h = self.size_pixels
        pixel_scale = max(
            (fb_w / win_w) if win_w > 0 else 1.0,
            (fb_h / win_h) if win_h > 0 else 1.0,
        )
        ui_scale, style_scale = resolve_ui_scales(
            content_scale,
            pixel_scale,
            self._scale_override,
        )
        changed = any(
            not math.isclose(current, updated, abs_tol=1e-3)
            for current, updated in (
                (self._content_scale, content_scale),
                (self._ui_scale, ui_scale),
                (self._pixel_scale, pixel_scale),
                (self._style_scale, style_scale),
            )
        )
        self._content_scale = content_scale
        self._ui_scale = ui_scale
        self._pixel_scale = pixel_scale
        self._style_scale = style_scale
        if changed:
            self._scale_generation += 1

    def _sync_style_scale(self) -> None:
        if math.isclose(self._applied_style_scale, self._style_scale, abs_tol=1e-3):
            return
        style = imgui.get_style()
        style.scale_all_sizes(self._style_scale / self._applied_style_scale)
        style.font_scale_dpi = self._style_scale / self._font_atlas_scale
        self._applied_style_scale = self._style_scale

    def poll_render_size(
        self, viewport_points: tuple[float, float], now: float | None = None
    ) -> tuple[int, int] | None:
        px = self.points_to_pixels(viewport_points)
        maximized = bool(glfw.get_window_attrib(self._window, glfw.MAXIMIZED))
        maximize_changed = maximized != self._maximized
        self._maximized = maximized
        return self.latch.update(
            (int(px[0]), int(px[1])),
            time.perf_counter() if now is None else now,
            immediate=maximize_changed,
        )

    def show(self) -> None:
        glfw.show_window(self._window)
        self._shown = True

    @property
    def shown(self) -> bool:
        return self._shown

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self._window))

    def request_close(self) -> None:
        glfw.set_window_should_close(self._window, True)

    def cancel_close(self) -> None:
        glfw.set_window_should_close(self._window, False)

    def consume_file_drops(self) -> tuple[Path, ...]:
        paths = tuple(self._file_drops)
        self._file_drops.clear()
        return paths

    @property
    def file_drag_active(self) -> bool:
        return self._file_drag_active

    def _on_file_drop(self, _window: Any, paths: list[str]) -> None:
        self._file_drag_active = False
        self._file_drops.extend(Path(path).expanduser().resolve() for path in paths)

    def set_vsync(self, on: bool) -> None:
        glfw.swap_interval(1 if on else 0)

    def set_title(self, title: str) -> None:
        glfw.set_window_title(self._window, title)

    def begin_frame(self) -> None:
        # A process may host multiple viewers. Window creation and shutdown both
        # change GLFW's process-local current context, so each frame must restore
        # the context that owns this window before ImGui or render resources run.
        self.make_current()
        imgui.set_current_context(self._imgui_context)
        glfw.poll_events()
        self._refresh_scales()
        self._sync_style_scale()
        self._impl.process_inputs()
        imgui.new_frame()

    def begin_dockspace(self) -> None:
        if self.config.docking:
            _suppress_dock_tab_nav_cursor()
            self.dockspace_id = imgui.dock_space_over_viewport(
                0,
                imgui.get_main_viewport(),
                imgui.DockNodeFlags_.passthru_central_node,
            )
            self._build_default_layout()

    def reset_layout(self) -> None:
        """Discard the active docking arrangement and rebuild the product default."""

        if not self.config.docking:
            return
        imgui.load_ini_settings_from_memory("")
        self._ini_existed = False
        self._layout_done = False
        if self.dockspace_id:
            self._build_default_layout()
            self._save_layout_settings()

    def _save_layout_settings(self) -> None:
        """Persist an explicit layout reset without waiting for a later ImGui frame."""

        ini = self.config.ini_path or ""
        if not ini:
            return
        try:
            target = Path(ini)
            target.parent.mkdir(parents=True, exist_ok=True)
            imgui.save_ini_settings_to_disk(str(target))
            self._ini_existed = True
        except Exception as exc:
            log.error("Dock layout save failed: {}", exc)

    _LAYOUT_LEFT = ("Hierarchy", "Assets")
    _LAYOUT_RIGHT_TOP = ("Control", "Joints", "Camera", "Settings", "Sensors")
    _LAYOUT_RIGHT_BOTTOM = ("Inspector",)
    _LAYOUT_BOTTOM = ("Stats", "Output", "Keyframes", "Plot", "Help", "Info")

    def _build_default_layout(self) -> None:
        if self._layout_done:
            return
        self._layout_done = True
        if self._ini_existed:
            return

        try:
            ii = imgui.internal
            root = self.dockspace_id
            ii.dock_builder_remove_node(root)

            ii.dock_builder_add_node(root, imgui.internal.DockNodeFlagsPrivate_.dock_space)
            ii.dock_builder_set_node_size(root, imgui.get_main_viewport().size)

            _, left, rest = ii.dock_builder_split_node_py(root, imgui.Dir.left, 0.22)
            _, right, rest = ii.dock_builder_split_node_py(rest, imgui.Dir.right, 0.30)
            _, bottom, center = ii.dock_builder_split_node_py(rest, imgui.Dir.down, 0.26)
            _, right_bottom, right_top = ii.dock_builder_split_node_py(
                right,
                imgui.Dir.down,
                0.50,
            )

            ii.dock_builder_dock_window("Viewport", center)
            for name in self._LAYOUT_LEFT:
                ii.dock_builder_dock_window(name, left)
            for name in self._LAYOUT_RIGHT_TOP:
                ii.dock_builder_dock_window(name, right_top)
            for name in self._LAYOUT_RIGHT_BOTTOM:
                ii.dock_builder_dock_window(name, right_bottom)
            for name in self._LAYOUT_BOTTOM:
                ii.dock_builder_dock_window(name, bottom)
            ii.dock_builder_finish(root)
        except Exception as e:
            log.error("Default dock layout failed; panels will float: {}", e)

    def end_frame(self, *, readback: bool = False) -> np.ndarray | None:
        imgui.set_current_context(self._imgui_context)
        imgui.render()
        fb_w, fb_h = self.size_pixels
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, fb_w, fb_h)

        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClearColor(*self.config.clear_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._impl.render(imgui.get_draw_data())

        frame = self.read_frame() if readback else None
        glfw.swap_buffers(self._window)
        self._frame_index += 1
        return frame

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def read_frame(self) -> np.ndarray:
        w, h = self.size_pixels
        if self._readback is None or self._readback.shape[:2] != (h, w):
            self._readback = np.empty((h, w, 3), np.uint8)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glReadBuffer(gl.GL_BACK)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self._readback)
        return self._readback

    def viewport_texture_ref(self, image: ViewportImage) -> Any:
        """The imgui texture reference used to draw the viewport image.

        The GL path addresses the resolved color texture by its integer name;
        surface-backed backends override this to bind their payload instead.
        """
        return imgui.ImTextureRef(image.texture_id)

    def close(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        native_drop.uninstall(self._native_drop_token)
        global _live_windows

        imgui.set_current_context(self._imgui_context)
        try:
            glfw.make_context_current(self._window)
        except Exception as e:
            log.debug("Failed to activate the GL context before shutdown: {}", e)
        try:
            self._impl.shutdown()
        finally:
            imgui.destroy_context(self._imgui_context)
            glfw.destroy_window(self._window)
            _live_windows = max(0, _live_windows - 1)
            if _live_windows == 0:
                glfw.terminate()

    def __enter__(self) -> Window:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
