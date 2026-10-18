"""Compose native UI presenters; Session and ViewerApp retain application state."""

from dataclasses import replace

from mojive import commands as cmd
from mojive.ui.workspace import (
    chrome,
    docking,
    inspector,
    panel_host,
    scene_panels,
    transactions,
    viewport,
)
from mojive.ui.workspace.camera_panel import camera_panel
from mojive.ui.workspace.constants import BOTTOM_PANELS, ICONS, LEFT_PANELS, PANELS
from mojive.ui.workspace.localization import workspace_text
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.output_panel import WorkspaceOutput
from mojive.ui.workspace.property_style import NativePropertyPresentation
from mojive.ui.workspace.sensors_panel import WorkspaceSensors
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.statistics import WorkspaceStatistics
from mojive.ui.workspace.visibility import PanelVisibility
from mojive.ui.workspace.widgets import ACCENT, DIM, INPUT, LINE, PANEL, TEXT, UI


class WorkspaceShell:
    def __init__(self, app, *, portable_fonts=False, ui=None):
        self.app, self.window = app, app.window
        self.s = self.window.style_scale
        self.ui = UI(self.window, portable_fonts=portable_fonts) if ui is None else ui
        self.ui.translate = lambda text: workspace_text(app.localizer, text)
        self.cube = app.view_cube
        self.open = PanelVisibility(app.panels)
        for name in self.open:
            self.open[name] = False
        self.open.update(Scene=True, Inspector=True)
        for name in self.open:
            panel = app.panels.get(name)
            override = app.panels.initial_config.get(panel.id) if panel is not None else None
            if panel is None or not panel.enabled:
                self.open[name] = False
            elif override is not None and override.open is not None:
                self.open[name] = override.open
        self.needs_layout = not self.window._ini_existed
        self.timeline_layout_pending = False
        self.active = self.left_active = "Scene"
        self.bottom_active = "Timeline"
        self.bottom_focus = None
        self.header_actions = {name: set() for name in LEFT_PANELS}
        self.header_actions.update(
            Scene={"search", "frame", "add"},
            Control={"reset"},
            Pose={"search", "sort", "copy"},
            Cameras={"frame"},
            Layers={"gizmos"},
        )
        self.search_visible = False
        self.header_add_pending = False
        self._numeric_edit = None
        self.pending = None
        self.restore_next = self.float_next = False
        self.focus_next = None
        self.saved = None
        self.show_viewport_tab = False
        self.query, self.filter = "", "All"
        self.inspector_tab = "Properties"
        self.pinned = None
        self.material = 0
        self.timeline_panel = None
        self.expanded = {"TRANSFORM": True, "DISPLAY": True, "IDENTITY": False}
        self.rects, self.widths = {}, {}
        self.input_overlays = []
        self.tool_panels = []
        self.menus = Menus(self, tuple(zip(PANELS, ICONS, strict=True)))
        self.settings = WorkspaceSettings(self)
        app.panels.workspace = self
        for panel_id in ("plot", "info", "help"):
            panel = app.panels.get(panel_id)
            if panel is not None:
                panel.presentation = NativePropertyPresentation(self.ui, prefix=panel_id)
        for replacement in (
            self.settings,
            WorkspaceOutput(self),
            WorkspaceStatistics(self),
            WorkspaceSensors(self),
        ):
            previous = app.panels.get(replacement.id)
            if previous is not None:
                replacement.enabled = previous.enabled
                replacement.open = previous.open
                app.panels.panels[app.panels.panels.index(previous)] = replacement
        app.theme = replace(
            app.theme,
            **{
                key: tuple(c / 255 for c in value)
                for key, value in (
                    ("text", TEXT),
                    ("text_disabled", DIM),
                    ("bg_window", PANEL),
                    ("bg_child", PANEL),
                    ("bg_frame", INPUT),
                    ("border", LINE),
                    ("primary", ACCENT),
                )
            },
        )
        self.window.apply_theme(app.theme)

    @property
    def time(self):
        return self.app.session.frame.time

    @property
    def playing(self):
        return not self.app.session.paused or self.app.session.state_take_playing

    @playing.setter
    def playing(self, value):
        self.app.session.submit(cmd.Play() if value else cmd.Pause())

    @property
    def speed(self):
        return self.app.session.speed

    @speed.setter
    def speed(self, value):
        self.app.session.submit(cmd.SetSpeed(float(value)))

    @property
    def loop(self):
        return self.app.session.state_take_loop_enabled

    @loop.setter
    def loop(self, value):
        self.app.session.submit(cmd.SetStateTakeLoopEnabled(bool(value)))

    @property
    def snap(self):
        return self.app._snap_latched

    @snap.setter
    def snap(self, value):
        self.app._snap_latched = bool(value)

    @property
    def space(self):
        return self.app.gizmo.space.title()

    @space.setter
    def space(self, value):
        self.app.gizmo.set_space(value.lower())

    def undo(self):
        self.app.session.submit(cmd.Undo())

    @property
    def bottom_open(self):
        return any(self.open[name] for name in BOTTOM_PANELS)

    def begin_frame(self):
        self.s = self.window.style_scale
        self.ui.s = self.s
        self.menus.s = self.s
        self.menus.begin_frame()
        self.ui.hits.clear()
        self.ui.numeric_edits.clear()
        if self.app.design is not None and self.app.design.workbench is not None:
            self.app.design.workbench.begin_frame(self)
            self.app.design.workbench.draw(self)
            self.app.design.workbench.claim_production_pointer()
        self.chrome()
        # Native menu actions can replace documents or start capture workers.
        # Execute them after all menu and sidebar Begin/End scopes have closed.
        self.menus.apply_actions()

    def register_tool(self, panel):
        self.app.panels.register(panel)
        self.tool_panels.append(panel.name)
        self.needs_layout = True

    def snapshot(self):
        result = {key: list(value) for key, value in self.rects.items()}
        result["viewport_content"] = self.rects.get("viewport_content")
        result["overlays"] = self.rects.get("overlays", [])
        result["window_size"] = list(self.window.size_pixels)
        if self.ui.hits:
            result["hits"] = {
                str(key): [round(float(v), 2) for v in rect]
                for key, rect in self.ui.hits.items()
                if isinstance(rect, (list, tuple))
                and len(rect) == 4
                and rect[2] > 0
                and rect[3] > 0
            }
        design = self.app.design
        if design is not None and (design.workbench is not None or design.observe_components):
            primitives = getattr(self.ui, "design_primitives", None)
            raw_records = list(primitives.observed) if primitives else []
            raw_records.extend(design.observed)
            components = []
            seen = set()
            w, h = self.window.size_pixels
            window_clip = (0.0, 0.0, float(w), float(h))
            for record in raw_records:
                hit = record.get("hit")
                if not hit or len(hit) != 4:
                    continue
                clip = record.get("clip") or window_clip
                left = max(float(hit[0]), float(clip[0]), 0.0)
                top = max(float(hit[1]), float(clip[1]), 0.0)
                right = min(float(hit[0] + hit[2]), float(clip[2]), float(w))
                bottom = min(float(hit[1] + hit[3]), float(clip[3]), float(h))
                if right - left < 2 or bottom - top < 2:
                    continue
                visible_rect = [
                    round(left, 2),
                    round(top, 2),
                    round(right - left, 2),
                    round(bottom - top, 2),
                ]
                parts = [str(p) for p in record.get("parts", ()) if p]
                key = str(record.get("key", ""))
                sig = (key, tuple(parts), tuple(visible_rect))
                if sig in seen:
                    continue
                seen.add(sig)
                components.append(
                    {
                        "key": key,
                        "parts": parts,
                        "kind": str(record.get("kind", "")),
                        "rect": visible_rect,
                    }
                )
            result["components"] = components
        return result

    def sidebar(self, name, direction, size):
        return chrome.sidebar(self, name, direction, size)

    def end_sidebar(self):
        return chrome.end_sidebar(self)

    def chrome(self):
        return chrome.chrome(self)

    def activate(self, name):
        return docking.activate(self, name)

    def toggle_activity(self, name):
        return docking.toggle_activity(self, name)

    def build(self):
        return docking.build(self)

    def ensure_layout(self, force=False):
        if not self.window.config.docking:
            self.needs_layout = False
            return
        if force:
            self.needs_layout = True
        if self.needs_layout and not self.window.layout_reset_pending:
            self.build()

    def toggle_timeline(self):
        return docking.toggle_timeline(self)

    def collapse_bottom(self):
        return docking.collapse_bottom(self)

    def apply_timeline_layout(self):
        return docking.apply_timeline_layout(self)

    def command(self, action):
        return docking.command(self, action)

    def inspector_grouped_with(self, name):
        return docking.inspector_grouped_with(self, name)

    def panel(self, name, ctx):
        return panel_host.panel(self, name, ctx)

    def panel_header(self, name, x, y, w):
        return panel_host.panel_header(self, name, x, y, w)

    def header_action(self, name, key):
        return panel_host.header_action(self, name, key)

    def inspector(self, x, y, w, h):
        return inspector.inspector(self, x, y, w, h)

    def scene_panel(self, x, y, w, h):
        return scene_panels.scene_panel(self, x, y, w, h)

    def control_panel(self, x, y, w):
        return scene_panels.control_panel(self, x, y, w)

    def assets_panel(self, x, y, w):
        return scene_panels.assets_panel(self, x, y, w)

    def pose_panel(self, x, y, w, *, node_id=None):
        return scene_panels.pose_panel(self, x, y, w, node_id=node_id)

    def camera_panel(self, x, y, w):
        return camera_panel(self, x, y, w)

    def layers_panel(self, x, y, w):
        return scene_panels.layers_panel(self, x, y, w)

    def timeline(self, x, y, w, h):
        return scene_panels.timeline(self, x, y, w, h)

    def viewport(self, preview_name=""):
        return viewport.viewport(self, preview_name)

    def pending_model_edits(self, x, y, w, h, overlay, *, bottom_limit=None):
        return viewport.pending_model_edits(self, x, y, w, h, overlay, bottom_limit=bottom_limit)

    def transport(self, x, y, w, shell):
        return viewport.transport(self, x, y, w, shell)

    def begin_numeric_edit(self, key, *, deferred=False, label="Inspector edit"):
        return transactions.begin_numeric_edit(self, key, deferred=deferred, label=label)

    def finish_numeric_edit(self, key, state):
        return transactions.finish_numeric_edit(self, key, state)

    def finish_numeric_frame(self, ctx):
        return transactions.finish_numeric_frame(self, ctx)
