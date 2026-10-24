"""Stable panel registration with the native shell owning visibility and docking."""

from contextlib import nullcontext
from dataclasses import replace

from imgui_bundle import imgui

from mojive.design.groups import component_groups, group_scope
from mojive.ui.panels import PanelManager, _panel_id
from mojive.ui.workspace.constants import LEFT_PANELS
from mojive.ui.workspace.design import DESIGN, current_design, use_design
from mojive.ui.workspace.property_style import property_controls
from mojive.ui.workspace.window_chrome import (
    floating_dock_root,
    native_chrome,
    native_window_geometry,
    panel_content_corners,
    panel_surface,
)

PANEL_NAMES = {
    "hierarchy": "Scene",
    "joints": "Pose",
    "control": "Control",
    "assets": "Assets",
    "camera": "Cameras",
    "sensors": "Sensors",
    "layers": "Layers",
    "output": "Output",
    "stats": "Statistics",
    "plot": "Plot",
    "inspector": "Inspector",
    "keyframes": "Timeline",
    "settings": "Settings",
}


class WorkspacePanels(PanelManager):
    def __init__(self, *, config=None, builtin_ids=None):
        self.initial_config = dict(config or {})
        super().__init__(config=config, builtin_ids=builtin_ids)
        self.workspace = None
        self._visibility_snapshot = {}
        self._legacy_titles = {panel.name: panel.id for panel in self.panels}
        for panel in self.panels:
            if panel.id in PANEL_NAMES:
                panel.name = PANEL_NAMES[panel.id]

    def get(self, panel_id):
        name = str(panel_id)
        stable = next((key for key, title in PANEL_NAMES.items() if title == name), name)
        return super().get(getattr(self, "_legacy_titles", {}).get(stable, stable))

    def load(self, panel_id):
        name = str(panel_id)
        stable = next((key for key, title in PANEL_NAMES.items() if title == name), name)
        panel = self.get(stable)
        if panel is None:
            panel = super().load(stable)
            if panel is not None and self.workspace is not None:
                native = self._native_panel(panel)
                self.panels[self.panels.index(panel)] = native
                panel = native
        return panel

    def bind_workspace(self, workspace):
        """Apply native presentation to both initial and subsequently loaded panels."""
        self.workspace = workspace
        self.panels[:] = [self._native_panel(panel) for panel in self.panels]

    def _native_panel(self, panel):
        workspace = self.workspace
        if panel.id == "settings":
            native = workspace.settings
        elif panel.id == "output":
            from .output_panel import WorkspaceOutput

            native = WorkspaceOutput(workspace)
        elif panel.id == "stats":
            from .statistics import WorkspaceStatistics

            native = WorkspaceStatistics(workspace)
        elif panel.id == "sensors":
            from .sensors_panel import WorkspaceSensors

            native = WorkspaceSensors(workspace)
        else:
            native = panel
            if panel.id in ("plot", "info", "help"):
                from .property_style import NativePropertyPresentation

                native.presentation = NativePropertyPresentation(workspace.ui, prefix=panel.id)
        if panel.id == "info":
            native.shortcut = ""
            native.open = False
            return native
        native.enabled, native.open = panel.enabled, panel.open
        native.name = PANEL_NAMES.get(panel.id, panel.name)
        return native

    def set_open(self, panel_id, open):
        changed = super().set_open(panel_id, open)
        panel = self.get(panel_id)
        if changed and self.workspace is not None and panel.id in PANEL_NAMES:
            if open:
                self.workspace.activate(panel.name)
            else:
                self.workspace.open[panel.name] = False
        return changed

    def register(self, panel):
        super().register(panel)
        self._legacy_titles[panel.name] = panel.id
        if panel.id in PANEL_NAMES:
            panel.name = PANEL_NAMES[panel.id]
            if self.workspace is not None:
                self.workspace.open[panel.name] = panel.enabled and panel.open

    def _begin_panel_window(self, panel, title, style_scale, dock_neighbor_title=""):
        """Custom native panels use the same outer window geometry as built-ins."""
        from mojive.ui.workspace.panel_host import _requested_docked

        flags, request = 0, None
        if panel.standalone:
            viewport = imgui.get_main_viewport()
            width, height = panel.initial_size
            if width > 0 and height > 0:
                imgui.set_next_window_size(
                    (width * style_scale, height * style_scale), imgui.Cond_.first_use_ever
                )
            imgui.set_next_window_pos(viewport.get_center(), imgui.Cond_.first_use_ever, (0.5, 0.5))
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            request = 0, imgui.Cond_.always
            flags = imgui.WindowFlags_.no_docking
        elif panel.dock_with:
            self._dock_with_neighbor(panel.dock_with, dock_neighbor_title)
            data = imgui.get_current_context().next_window_data
            if data.has_flags & imgui.internal.NextWindowDataFlags_.has_dock:
                request = data.dock_id, data.dock_cond
        window = imgui.internal.find_window_by_name(title)
        docked = _requested_docked(panel.name, window, request)
        with panel_surface(
            self.workspace.ui,
            name=panel.name,
            docked=docked,
            floating=not docked or floating_dock_root(window) is not None,
        ):
            return imgui.begin(title, True if panel.closable else None, flags)

    def draw(self, ctx):
        ctx.panels = self
        ctx.status_hints_by_panel.clear()
        visibility = getattr(self.workspace.app, "scrollbar_visibility", None)
        # Public panel methods (including Assets.focus) use Panel.open directly.
        # Route a newly opened sidebar through the same single-pane activation.
        for panel in self.panels:
            if (
                panel.name in LEFT_PANELS
                and panel.enabled
                and panel.open
                and not self._visibility_snapshot.get(panel.id, False)
                and panel.name != self.workspace.left_active
            ):
                self.workspace.activate(panel.name)
        for panel in self.panels:
            ctx.status_hints = ()
            if panel.id in PANEL_NAMES:
                panel.open = self.workspace.open[panel.name] and panel.enabled
                if panel.open and not self._visibility_snapshot.get(panel.id, False):
                    self.workspace.new_panel = panel.name
                if self.workspace.panel(panel.name, ctx):
                    ctx.status_hints_by_panel[panel.name] = tuple(ctx.status_hints)
                self._visibility_snapshot[panel.id] = panel.open
            else:
                owner = _panel_id(panel)
                showing = panel.enabled and panel.open
                if visibility is not None:
                    visibility.set_visible(owner, showing)
                if not showing:
                    continue
                title = (
                    ctx.tr(panel.name) + "###" + panel.name
                    if ctx.tr(panel.name) != panel.name
                    else panel.name
                )
                scope = (
                    visibility.panel_scope(owner, window_name=panel.name)
                    if visibility is not None
                    else nullcontext()
                )
                with (
                    scope,
                    group_scope(panel.name) if panel.name in component_groups() else nullcontext(),
                ):
                    with native_chrome(self.workspace.ui, "window_title"):
                        visible, opened = self._begin_panel_window(panel, title, ctx.style_scale)
                    if visibility is not None:
                        visibility.bind_current_window()
                    try:
                        if opened is not None:
                            panel.open = opened
                            if visibility is not None:
                                visibility.set_visible(owner, panel.enabled and panel.open)
                        if visible:
                            ui = self.workspace.ui
                            parent_window = imgui.internal.get_current_window()
                            pos, size = (
                                imgui.get_cursor_screen_pos(),
                                imgui.get_content_region_avail(),
                            )
                            box = ui.component_box(
                                "panel_body",
                                pos.x,
                                pos.y,
                                size.x,
                                size.y,
                                key="panel-body-" + panel.name,
                            )
                            imgui.set_cursor_screen_pos((box.content.x, box.content.y))
                            imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
                            imgui.begin_child(
                                "Panel content", (box.content.width, box.content.height)
                            )
                            imgui.pop_style_var()
                            style = ui.component_style("panel_body")
                            dimensions = replace(
                                current_design(), row_gap=style.get("rowGap", DESIGN.row_gap)
                            )
                            try:
                                with (
                                    panel_content_corners(ui, panel.name, parent_window),
                                    use_design(lambda _, dimensions=dimensions: dimensions),
                                    ui.component_scope("panel_body"),
                                    property_controls(ui),
                                ):
                                    panel.draw(ctx)
                            finally:
                                imgui.end_child()
                            window = imgui.get_current_context().current_window
                            if not window.dock_id:
                                bar = window.title_bar_rect()
                                ui.design.record(
                                    "window-title/" + panel.name,
                                    ("window_title",),
                                    (
                                        bar.min.x,
                                        bar.min.y,
                                        bar.max.x - bar.min.x,
                                        bar.max.y - bar.min.y,
                                    ),
                                    "header",
                                    **native_window_geometry(window),
                                )
                        panel.finish_frame(ctx)
                    finally:
                        imgui.end()
        self.workspace.menus.finish_color_edit()
        self.workspace.finish_numeric_frame(ctx)
