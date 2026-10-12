"""Stable panel registration with the native shell owning visibility and docking."""

from mojive.ui.panels import PanelManager

from .constants import LEFT_PANELS
from .property_style import property_controls

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
        return super().load(stable)

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
        if panel.id in PANEL_NAMES:
            panel.name = PANEL_NAMES[panel.id]
            if self.workspace is not None:
                self.workspace.open[panel.name] = panel.enabled and panel.open

    def draw(self, ctx):
        ctx.panels = self
        ctx.status_hints_by_panel.clear()
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
            elif panel.enabled and panel.open:
                title = (
                    ctx.tr(panel.name) + "###" + panel.name
                    if ctx.tr(panel.name) != panel.name
                    else panel.name
                )
                visible, opened = self._begin_panel_window(panel, title, ctx.style_scale)
                if opened is not None:
                    panel.open = opened
                if visible:
                    with property_controls(self.workspace.ui):
                        panel.draw(ctx)
                panel.finish_frame(ctx)
                from imgui_bundle import imgui

                imgui.end()
        self.workspace.menus.finish_color_edit()
        self.workspace.finish_numeric_frame(ctx)
