"""Own the design editor's native window independently of the preview viewport."""

from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.app.ui.window import create_window
from mojive.ui.scrollbar_visibility import NativeScrollbarVisibility
from mojive.ui.window import WindowConfig
from mojive.ui.workspace.widgets import UI


@dataclass
class EditorSurface:
    """The editor needs local fonts and scale; scene actions target the preview."""

    ui: UI
    s: float


class WorkbenchWindow:
    def __init__(
        self,
        workbench,
        *,
        height=900,
        scale=1,
        visible=True,
        detached=True,
        title="Mojive · UI Design Workbench",
        renderer="opengl",
    ):
        self.workbench = workbench
        self.visible = visible
        self.window = create_window(
            WindowConfig(
                title=title,
                width=round(workbench.width * scale),
                height=height,
                docking=False,
                ini_path="",
                show_on_start=visible and detached,
                load_default_fonts=False,
                # The preview owns frame pacing for the pair of windows.
                vsync=False,
            ),
            renderer,
        )
        try:
            self.surface = EditorSurface(
                UI(
                    self.window,
                    portable_fonts=workbench.runtime.spec["tokens"]["font_family"] == "portable",
                ),
                self.window.style_scale,
            )
            from mojive.design.native.presentation import DesignPresentation

            self.surface.ui.design = DesignPresentation(self.surface.ui, workbench.runtime)
            self._font_revision = -1
            self.scrollbars = NativeScrollbarVisibility()
            workbench.detached = detached
            workbench._viewport_rect = None
            if visible and detached:
                self.window.show()
        except BaseException:
            self.window.close()
            raise

    def apply_mode_request(self, preview):
        """Move the editor between surfaces only after both native frames finish."""
        bench = self.workbench
        requested = bench.requested_detached
        if requested is None:
            return
        preview.app.preferences.update({"workbench_detached": requested})
        bench.editor.end_gesture()
        bench.detached = requested
        bench.requested_detached = None
        bench._viewport_rect = None
        bench.editor_rect = None
        bench._ready_written = False
        bench.hits.clear()
        if requested and self.visible:
            self.window.show()
        else:
            self.window.hide()

    def frame(self, *, readback=False):
        self.window.begin_frame()
        surface, bench = self.surface, self.workbench
        surface.s = surface.ui.s = self.window.style_scale
        bench.poll_editor()
        if self._font_revision != bench.runtime.revision:
            bench.runtime.prepare_fonts(surface.ui)
            self._font_revision = bench.runtime.revision
        surface.ui.design.begin_frame()
        self.scrollbars.update()
        with (
            bench.editor_scope(surface),
            self.scrollbars.scope(),
            self.scrollbars.panel_scope(
                "PythonDesignWorkbench", window_name="PythonDesignWorkbench"
            ),
        ):
            viewport = imgui.get_main_viewport()
            imgui.set_next_window_pos(viewport.pos)
            imgui.set_next_window_size(viewport.size)
            flags = (
                imgui.WindowFlags_.no_decoration
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_saved_settings
                | imgui.WindowFlags_.no_bring_to_front_on_focus
            )
            imgui.begin("PythonDesignWorkbench", flags=flags)
            self.scrollbars.bind_current_window()
            try:
                bench.editor_rect = (*imgui.get_window_pos(), *imgui.get_window_size())
                bench.draw_contents(surface)
            finally:
                imgui.end()
        return self.window.end_frame(readback=readback)

    def close(self):
        self.window.close()
        self.workbench.detached = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def run_workbench(preview, editor_window, *, max_frames=None):
    """Render edits before the preview; use the preview's document-close workflow."""
    count = 0
    while max_frames is None or count < max_frames:
        editor_window.apply_mode_request(preview)
        if editor_window.workbench.detached and editor_window.window.should_close():
            preview.window.request_close()
            editor_window.window.cancel_close()
        if preview.app._should_close():
            break
        if editor_window.workbench.detached:
            editor_window.scrollbars.set_idle_seconds(preview.app.scrollbar_visibility.idle_seconds)
            editor_window.frame()
        preview.sync()
        count += 1
