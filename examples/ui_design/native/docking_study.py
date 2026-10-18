"""Instrument reference rendered with native ImGui controls and Mojive 3D backends."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.app.ui.window import create_window
from mojive.ui.input_bindings import InputAction
from mojive.ui.layers import LAYER_LABELS, debug_layer_group
from mojive.ui.panels import activate_edit_gizmo, pointer_pressed, publish_focus_item_hint
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewcube import BACKDROP_RADIUS_PT, widget_center
from mojive.ui.window import WindowConfig
from mojive.ui.workspace.inspector_components import draw_component
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.widgets import (
    AMBER,
    DIM,
    GLYPHS,
    ICON_ROLES,
    LINE,
    MUTED,
    PANEL,
    SAGE,
    SWITCH_WIDTH,
    TEXT,
    UI,
    color,
)

from .localization import reference_text
from .reference_menus import Menus
from .scene_preview import Preview

ROOT = Path(__file__).parent
OUTPUT = Path("output/ui_design/native")
PANELS = (
    "Scene",
    "Pose",
    "Control",
    "Assets",
    "Cameras",
    "Sensors",
    "Layers",
    "Output",
    "Statistics",
    "Settings",
)
ICONS = (
    "scene",
    "pose",
    "control",
    "assets",
    "camera",
    "sensor",
    "layers",
    "terminal",
    "performance",
    "settings",
)
LEFT_PANELS = PANELS[:7]
BOTTOM_PANELS = ("Timeline", "Output", "Statistics")
BOTTOM_HEIGHT_PT = 268
VIEWPORT_TOOLS = (
    ("select", None),
    ("move", InputAction.GIZMO_TRANSLATE),
    ("rotate", InputAction.GIZMO_ROTATE),
    ("scale", InputAction.GIZMO_DIMENSIONS),
)


class Study:
    @property
    def app(self):
        """Expose the reference's application through the production workspace contract."""
        return self.preview

    def __init__(
        self,
        window,
        backend="opengl",
        restore=None,
        *,
        output_directory=OUTPUT,
        portable_fonts=False,
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.s = window.style_scale
        self.preview = Preview(window, backend)
        self.ui = UI(window, portable_fonts=portable_fonts)
        self.ui.translate = lambda value: reference_text(self.preview.localizer, value)
        self.cube = self.preview.view_cube
        self.open = dict.fromkeys(PANELS, False)
        self.open.update(Scene=True, Inspector=True, Timeline=False)
        self.needs_layout = restore is None
        self.timeline_layout_pending = False
        self.active = "Scene"
        self.left_active = "Scene"
        self.bottom_active = "Timeline"
        self.bottom_focus = None
        self.header_actions = {name: set() for name in LEFT_PANELS}
        self.header_actions["Scene"] = {"search", "frame", "add"}
        self.header_actions["Control"] = {"reset"}
        self.header_actions["Pose"] = {"search", "sort", "copy"}
        self.header_actions["Cameras"] = {"frame"}
        self.header_actions["Layers"] = {"gizmos"}
        self.search_visible = False
        self.header_add_pending = False
        self._numeric_edit = None
        self.pending = None
        self.restore_next = False
        self.float_next = False
        self.focus_next = None
        self.saved = restore
        self.show_viewport_tab = False
        self.menus = Menus(self, tuple(zip(PANELS, ICONS, strict=True)))
        self.workspace = "Edit"
        self.query = ""
        self.filter = "All"
        self.inspector_tab = "Properties"
        self.pinned = None
        self.material = 0
        self.timeline_panel = None
        self.expanded = {"TRANSFORM": True, "DISPLAY": True, "IDENTITY": False}
        self.events = []
        self.settings = WorkspaceSettings(self)
        self.rects = {}
        self.input_overlays = []
        self.widths = {}
        self.preview.bind(self)
        if restore:
            imgui.load_ini_settings_from_memory(restore)

    @property
    def time(self):
        return self.preview.session.frame.time

    @property
    def playing(self):
        return not self.preview.session.paused or self.preview.session.state_take_playing

    @playing.setter
    def playing(self, value):
        self.preview.session.submit(cmd.Play() if value else cmd.Pause())

    @property
    def speed(self):
        return self.preview.session.speed

    @speed.setter
    def speed(self, value):
        self.preview.session.submit(cmd.SetSpeed(float(value)))

    @property
    def loop(self):
        return self.preview.session.state_take_loop_enabled

    @loop.setter
    def loop(self, value):
        self.preview.session.submit(cmd.SetStateTakeLoopEnabled(bool(value)))

    @property
    def snap(self):
        return self.preview._snap_latched

    @snap.setter
    def snap(self, value):
        self.preview._snap_latched = bool(value)

    @property
    def space(self):
        return self.preview.gizmo.space.title()

    @space.setter
    def space(self, value):
        self.preview.gizmo.set_space(value.lower())

    @property
    def history(self):
        return self.preview.session.can_undo

    def undo(self):
        self.preview.session.submit(cmd.Undo())

    def begin_frame(self):
        self.menus.begin_frame()
        self.ui.hits.clear()
        self.ui.numeric_edits.clear()
        self.chrome()

    @property
    def bottom_open(self):
        return any(self.open[name] for name in BOTTOM_PANELS)

    def draw(self, *, readback=False):
        if self.restore_next:
            # Reload before NewFrame so the restored DockSpace is submitted this frame.
            imgui.set_current_context(self.window._imgui_context)
            imgui.load_ini_settings_from_memory(self.saved)
            self.restore_next = False
        self.preview.readback = readback
        self.preview.sync()
        return self.preview.pixels

    def sidebar(self, name, direction, size):
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_docking
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
        )
        if name == "Application":
            flags |= imgui.WindowFlags_.menu_bar
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        visible = imgui.internal.begin_viewport_side_bar(
            name, imgui.get_main_viewport(), direction, size * self.s, flags
        )
        return visible

    def end_sidebar(self):
        imgui.end()
        imgui.pop_style_var()

    def chrome(self):
        u = self.ui
        s = self.s
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (8 * s, 11 * s))
        if self.sidebar("Application", imgui.Dir.up, 36):
            p = imgui.get_window_pos()
            w = imgui.get_window_width()
            x = p.x
            y = p.y
            imgui.push_clip_rect((x, y), (x + w, y + 36 * s), False)
            self.menus.bar()
            center_y = self.menus.menu_center_y
            brand_width = u.measure("M", 18)
            u.text(x + (44 * s - brand_width) / 2, center_y - 9 * s, "M", SAGE, 18, bold=True)
            self.rects["application-logo"] = [x + 8 * s, center_y - 9 * s, 28 * s, 18 * s]
            if w > 950 * s:
                tx = x + w * 0.49
                u.role_icon("file", tx, y + 4 * s, 14 * s, 28 * s, role="inline", c=DIM)
                u.text(tx + 22 * s, y + 11.5 * s, "Joint study", MUTED, 13)
            if w >= 760 * s:
                workspaces = ("Edit", "Author", "Review")
                group_width = 3 * (
                    max(u.measure(u.translate(name)) for name in workspaces) / s + 16
                )
                chosen = u.segmented(
                    "workspace",
                    x + w - (group_width + 65) * s,
                    y + 4 * s,
                    group_width,
                    workspaces,
                    workspaces.index(self.workspace),
                    keys=tuple("workspace-" + name for name in workspaces),
                )
                if workspaces[chosen] != self.workspace:
                    self.preset(workspaces[chosen])
            if u.button(
                "toggle-inspector",
                x + w - 57 * s,
                y + 4 * s,
                26,
                28,
                icon="inspect",
                active=self.open["Inspector"],
                tooltip="Toggle Inspector",
            ):
                self.open["Inspector"] = not self.open["Inspector"]
            settings_clicked = u.button(
                "preferences",
                x + w - 29 * s,
                y + 4 * s,
                26,
                28,
                icon="settings",
                tooltip="Settings · F9",
                active=self.open["Settings"],
            )
            if settings_clicked:
                self.activate("Settings")
            self.menus.dialogs()
            imgui.pop_clip_rect()
        self.end_sidebar()
        imgui.pop_style_var()
        if self.sidebar("Navigation", imgui.Dir.left, 48):
            p = imgui.get_window_pos()
            h = imgui.get_window_height()
            compact = h < 470 * s
            for n, (name, icon) in enumerate(zip(PANELS, ICONS, strict=True)):
                yy = (
                    p.y + (8 + n * 42) * s - imgui.get_scroll_y()
                    if compact or n < 7
                    else p.y + h - (3 - (n - 7)) * 42 * s
                )
                if u.button(
                    "rail-" + name,
                    p.x + 6 * s,
                    yy,
                    36,
                    36,
                    icon=icon,
                    icon_role="activity",
                    active=(
                        self.left_active == name
                        if name in LEFT_PANELS
                        else self.bottom_active == name
                    )
                    and self.open[name],
                    tooltip=name,
                ):
                    self.activate(name)
            if compact:
                imgui.set_cursor_screen_pos((p.x, p.y + 440 * s - imgui.get_scroll_y()))
                imgui.dummy((1, 1))
        self.end_sidebar()

    def activate(self, name):
        if name in LEFT_PANELS:
            for other in LEFT_PANELS:
                self.open[other] = other == name
            self.left_active = self.active = name
        elif name in BOTTOM_PANELS:
            for other in BOTTOM_PANELS:
                self.open[other] = True
            self.bottom_active = name
            self.bottom_focus = name
            self.timeline_layout_pending = True
        if not self.open[name]:
            self.open[name] = True
            self.new_panel = name
        self.focus_next = name
        self.preview.panels.get(name).open = True

    def build(self):
        ii = imgui.internal
        ii.dock_builder_remove_node(self.window.dockspace_id)
        ii.dock_builder_add_node(self.window.dockspace_id, ii.DockNodeFlagsPrivate_.dock_space)
        work = imgui.get_current_context().viewports[0].get_build_work_rect()
        size = work.get_size()
        ii.dock_builder_set_node_size(self.window.dockspace_id, size)
        left_width = min(252 * self.s, size.x * 0.26)
        right_width = min(320 * self.s, size.x * 0.3)
        _, self.left, rest = ii.dock_builder_split_node_py(
            self.window.dockspace_id, imgui.Dir.left, left_width / size.x
        )
        _, self.right, self.center = ii.dock_builder_split_node_py(
            rest, imgui.Dir.right, right_width / (size.x - left_width)
        )
        if self.bottom_open:
            _, self.bottom, self.center = ii.dock_builder_split_node_py(
                self.center, imgui.Dir.down, min(0.68, BOTTOM_HEIGHT_PT * self.s / size.y)
            )
            for name in BOTTOM_PANELS:
                ii.dock_builder_dock_window(name, self.bottom)
        for name in LEFT_PANELS:
            ii.dock_builder_dock_window(name, self.left)
        ii.dock_builder_dock_window("Inspector", self.right)
        ii.dock_builder_dock_window("Viewport", self.center)
        ii.dock_builder_finish(self.window.dockspace_id)
        self.needs_layout = False
        self.timeline_layout_pending = False

    def toggle_timeline(self):
        if self.bottom_open and self.bottom_active == "Timeline":
            self.collapse_bottom()
        else:
            self.activate("Timeline")

    def collapse_bottom(self):
        for name in BOTTOM_PANELS:
            self.open[name] = False
        self.bottom_focus = None
        self.timeline_layout_pending = True

    def apply_timeline_layout(self):
        if not self.timeline_layout_pending:
            return
        self.timeline_layout_pending = False
        if self.bottom_open:
            for name in BOTTOM_PANELS:
                window = imgui.internal.find_window_by_name(name)
                if window is not None and imgui.internal.dock_builder_get_node(window.dock_id):
                    self.bottom = window.dock_id
                    break
            else:
                self.bottom = None
            window = imgui.internal.find_window_by_name("Viewport")
            node = window.dock_node if window is not None else None
            if self.bottom is None and node is not None:
                _, self.bottom, self.center = imgui.internal.dock_builder_split_node_py(
                    node.id_, imgui.Dir.down, min(0.68, BOTTOM_HEIGHT_PT * self.s / node.size.y)
                )
            if self.bottom is not None:
                for name in BOTTOM_PANELS:
                    imgui.internal.dock_builder_dock_window(name, self.bottom)
                imgui.internal.dock_builder_finish(self.window.dockspace_id)

    def preset(self, name):
        self.workspace = name
        self.open["Inspector"] = True
        if name == "Review":
            for n in PANELS:
                self.open[n] = False
            self.activate("Timeline")
        else:
            self.collapse_bottom()
            self.activate("Assets" if name == "Author" else "Scene")

    def command(self, action):
        ii = imgui.internal
        if action == "reset":
            for n in PANELS:
                self.open[n] = n == "Scene"
            self.open.update(Inspector=True, Timeline=False)
            self.left_active = self.active = "Scene"
            self.float_next = False
            self.focus_next = "Scene"
            self.bottom_focus = None
            self.restore_next = False
            self.workspace = "Edit"
            self.build()
        elif action == "float":
            self.open["Inspector"] = True
            self.float_next = True
            self.focus_next = "Inspector"
        elif action in ("dock", "group"):
            if action == "group":
                target = ii.find_window_by_name("Scene").dock_id
            else:
                root = ii.dock_builder_get_node(self.window.dockspace_id)
                current = ii.find_window_by_name("Inspector").dock_node
                if (
                    current
                    and abs(current.pos.x + current.size.x - root.pos.x - root.size.x) < 2
                    and abs(current.size.y - root.size.y) < 2
                ):
                    target = current.id_
                else:
                    _, target, _ = ii.dock_builder_split_node_py(
                        self.window.dockspace_id, imgui.Dir.right, 320 * self.s / root.size.x
                    )
            ii.dock_builder_dock_window("Inspector", target)
            ii.dock_builder_finish(self.window.dockspace_id)
            self.focus_next = "Inspector"
        elif action == "save":
            self.saved = imgui.save_ini_settings_to_memory()
        elif action == "restore" and self.saved:
            self.restore_next = True
        else:
            raise ValueError(action)

    def panel(self, name, ctx):
        panel = self.preview.panels.get(name)
        if not self.open[name]:
            if name == "Inspector":
                presentation = getattr(self, "inspector_presentation", None)
                if presentation is not None and presentation.color_edit_open:
                    if panel._edit_transaction:
                        ctx.submit(cmd.CancelEditTransaction())
                        panel._edit_transaction = False
                    presentation.color_edit_open = False
                    presentation.color_edit_owner = None
            panel.finish_frame(ctx)
            return False
        if getattr(self, "new_panel", None) == name:
            if name == "Settings":
                viewport = imgui.get_main_viewport()
                size = (
                    min(820 * self.s, viewport.size.x - 32 * self.s),
                    min(620 * self.s, viewport.size.y - 64 * self.s),
                )
                imgui.set_next_window_dock_id(0, imgui.Cond_.always)
                imgui.set_next_window_pos(
                    (
                        viewport.pos.x + (viewport.size.x - size[0]) / 2,
                        viewport.pos.y + (viewport.size.y - size[1]) / 2,
                    ),
                    imgui.Cond_.appearing,
                )
                imgui.set_next_window_size(size, imgui.Cond_.appearing)
            elif name in LEFT_PANELS:
                imgui.set_next_window_dock_id(self.left, imgui.Cond_.first_use_ever)
            self.new_panel = None
        if name == "Inspector" and self.float_next:
            viewport = imgui.get_main_viewport()
            size = (
                min(320 * self.s, viewport.work_size.x - 24 * self.s),
                min(570 * self.s, viewport.work_size.y - 24 * self.s),
            )
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            imgui.set_next_window_pos(
                (
                    viewport.work_pos.x + (viewport.work_size.x - size[0]) / 2,
                    viewport.work_pos.y + (viewport.work_size.y - size[1]) / 2,
                ),
                imgui.Cond_.always,
            )
            imgui.set_next_window_size(size, imgui.Cond_.always)
            self.float_next = False
        if self.focus_next == name or self.bottom_focus == name:
            imgui.set_next_window_focus()
            self.focus_next = None
            window = imgui.internal.find_window_by_name(name)
            if (
                window is not None
                and window.dock_node is not None
                and window.dock_node.tab_bar is not None
            ):
                window.dock_node.tab_bar.next_selected_tab_id = window.tab_id
        wc = imgui.WindowClass()
        wc.dock_node_flags_override_set = (
            imgui.internal.DockNodeFlagsPrivate_.no_window_menu_button
            | imgui.internal.DockNodeFlagsPrivate_.no_close_button
        )
        window = imgui.internal.find_window_by_name(name)
        solo_inspector = name == "Inspector" and (
            window is None or window.dock_node is None or len(window.dock_node.windows) <= 1
        )
        if (name in LEFT_PANELS and not self.inspector_grouped_with(name)) or solo_inspector:
            wc.dock_node_flags_override_set |= imgui.internal.DockNodeFlagsPrivate_.no_tab_bar
        imgui.set_next_window_class(wc)
        title = ctx.tr(name) + "###" + name if ctx.tr(name) != name else name
        flags = (
            imgui.WindowFlags_.no_title_bar
            if name in (*LEFT_PANELS, "Inspector", "Settings")
            else 0
        )
        if name == "Settings":
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            flags |= imgui.WindowFlags_.no_docking
        if name in LEFT_PANELS:
            flags |= imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
        visible, self.open[name] = imgui.begin(title, True, flags)
        if visible:
            if name in BOTTOM_PANELS:
                window = imgui.internal.get_current_window()
                if (
                    window.dock_node is not None
                    and window.dock_node.selected_tab_id == window.tab_id
                ):
                    self.bottom_active = name
                    if self.bottom_focus == name:
                        self.bottom_focus = None
            self.widths[name] = imgui.get_content_region_avail().x
            self._panel_context = ctx
            self.ui.value_context = ctx
            p = imgui.get_cursor_screen_pos()
            size = imgui.get_content_region_avail()
            x, y, w, h = p.x, p.y, size.x, size.y
            if name in LEFT_PANELS:
                self.panel_header(name, x, y, w)
                y += 38 * self.s
                h -= 38 * self.s
                imgui.set_cursor_screen_pos((x, y))
                imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
                imgui.begin_child("Panel content", (w, max(self.s, h)))
                imgui.pop_style_var()
                p = imgui.get_cursor_screen_pos()
                size = imgui.get_content_region_avail()
                x, y, w, h = p.x, p.y, size.x, size.y
            draw = {
                "Scene": self.scene_panel,
                "Inspector": self.inspector,
                "Control": self.control_panel,
                "Assets": self.assets_panel,
                "Pose": self.pose_panel,
                "Cameras": self.camera_panel,
                "Layers": self.layers_panel,
                "Timeline": self.timeline,
            }.get(name)
            if name == "Settings":
                self.settings.draw(ctx)
            elif name in ("Scene", "Inspector", "Timeline"):
                draw(x, y, w, h)
            elif draw is not None:
                draw(x, y, w)
            elif name in ("Sensors", "Output", "Statistics"):
                panel.draw(ctx)
            if name in LEFT_PANELS:
                imgui.end_child()
            self.preview.apply_document_edits()
        if name == "Inspector":
            panel.finish_frame(
                ctx,
                keep_edit_transaction=bool(
                    getattr(getattr(self, "inspector_presentation", None), "color_edit_open", False)
                ),
            )
        else:
            panel.finish_frame(ctx)
        panel.open = self.open[name]
        imgui.end()
        return visible and self.open[name]

    def panel_header(self, name, x, y, w):
        u, s = self.ui, self.s
        actions = {
            "Scene": (
                ("search", "search", "Search scene"),
                ("frame", "frame", "Frame all"),
                ("add", "plus", "Add"),
            ),
            "Control": (("reset", "reset", "Reset controls"),),
            "Pose": (
                ("search", "search", "Search joints"),
                ("sort", "sort", "Sort by name"),
                ("copy", "copy", "Copy state"),
            ),
            "Cameras": (("frame", "frame", "Frame all"),),
            "Layers": (("gizmos", "move", "Transform handles"),),
        }.get(name, ())
        if name == "Pose" and not hasattr(self, "navigation"):
            from .navigation import Navigation

            self.navigation = Navigation(self)
        enabled = [item for item in actions if item[0] in self.header_actions[name]]
        # Keep the title readable as dock panels narrow; remaining actions stay in More.
        slots = max(0, int((w / s - 74) / 28) - 1)
        visible = enabled[:slots]
        right = x + w - 28 * s
        imgui.push_clip_rect((x, y), (right - len(visible) * 28 * s - 4 * s, y + 30 * s), True)
        u.text(x, y + 7.5 * s, name, TEXT, 13, bold=True)
        imgui.pop_clip_rect()
        for index, (key, icon, label) in enumerate(visible):
            xx = right - (len(visible) - index) * 28 * s
            active = (
                self.navigation.pose_action_active(key)
                if name == "Pose"
                else (key == "search" and self.search_visible)
                or (key == "gizmos" and self.preview.gizmos)
            )
            if u.button(
                "header-" + name + "-" + key, xx, y, 26, 28, icon=icon, tooltip=label, active=active
            ):
                self.header_action(name, key)
        popup = "Panel actions###" + name
        filtered = name == "Scene" and self.filter != "All"
        hint = u.translate("Filter") + ": " + u.translate(self.filter) if filtered else "More"
        if u.button(
            "header-" + name + "-more", right, y, 26, 28, icon="more", tooltip=hint, active=filtered
        ):
            imgui.open_popup(popup)
        with self.menus.style():
            if imgui.begin_popup(popup):
                for key, _icon, label in actions:
                    clicked, _ = imgui.menu_item(u.translate(label), "", False)
                    if clicked:
                        self.header_action(name, key)
                if name == "Scene":
                    imgui.separator()
                    if imgui.begin_menu(u.translate("Filter")):
                        from .navigation import Navigation

                        if not hasattr(self, "navigation"):
                            self.navigation = Navigation(self)
                        for kind, count in self.navigation.scene_filters(self._panel_context):
                            label = f"{u.translate(kind.title())} {count}"
                            clicked, _ = imgui.menu_item(label, "", self.filter.casefold() == kind)
                            if clicked:
                                self.filter = kind.title()
                        imgui.end_menu()
                if actions:
                    imgui.separator()
                    if imgui.begin_menu(u.translate("Header actions")):
                        for key, _icon, label in actions:
                            changed, checked = imgui.checkbox(
                                u.translate(label), key in self.header_actions[name]
                            )
                            if changed:
                                if checked:
                                    self.header_actions[name].add(key)
                                else:
                                    self.header_actions[name].discard(key)
                        imgui.end_menu()
                imgui.separator()
                clicked, _ = imgui.menu_item(u.translate("Hide panel"), "", False)
                if clicked:
                    self.open[name] = False
                imgui.end_popup()
        if name == "Scene":
            if self.header_add_pending or imgui.is_popup_open("Add"):
                self.menus.popup("Add", (right, y, 26 * s, 28 * s), self.header_add_pending)
            self.header_add_pending = False
        u.line(x - 12 * s, y + 32 * s, w + 24 * s)

    def header_action(self, name, key):
        if name == "Pose":
            self.navigation.pose_action(key)
        elif key == "search":
            self.search_visible = not self.search_visible
            if not self.search_visible:
                self.query = ""
        elif key == "frame":
            self.preview._frame_scene(animate=True)
        elif key == "add":
            self.header_add_pending = True
        elif name == "Control" and key == "reset":
            session = self.preview.session
            if session.adapter.caps.write_ctrl and session.frame.ctrl is not None:
                session.submit(cmd.SetCtrlVector(np.zeros_like(session.frame.ctrl)))
        elif key == "gizmos":
            self.preview.gizmos = not self.preview.gizmos

    def begin_numeric_edit(self, key, *, deferred=False, label="Inspector edit"):
        state = self.ui.numeric_edits[key]
        if state.activated and self._numeric_edit is None and not self.preview.session.editing:
            draft = self.preview.model_edits
            if draft._checkpoint is None:
                if deferred:
                    draft.begin_transaction(label)
                    self._numeric_edit = key
                elif self._panel_context.submit(cmd.BeginEditTransaction(label)).ok:
                    self._numeric_edit = key
        return state

    def finish_numeric_edit(self, key, state):
        if self._numeric_edit != key:
            return
        if state.canceled or (state.deactivated and not state.active):
            command = cmd.CancelEditTransaction() if state.canceled else cmd.EndEditTransaction()
            self._panel_context.submit(command)
            self._numeric_edit = None
            if state.canceled:
                self.preview.refresh_document()

    def finish_numeric_frame(self, ctx):
        if self._numeric_edit is not None and not imgui.is_any_item_active():
            ctx.submit(cmd.EndEditTransaction())
            self._numeric_edit = None

    def inspector_grouped_with(self, name):
        window = imgui.internal.find_window_by_name(name)
        inspector = imgui.internal.find_window_by_name("Inspector")
        return bool(
            self.open["Inspector"]
            and window is not None
            and inspector is not None
            and window.dock_id
            and window.dock_id == inspector.dock_id
        )

    def scene_panel(self, x, y, w, h):
        from .navigation import Navigation

        if not hasattr(self, "navigation"):
            self.navigation = Navigation(self)
        self.navigation.draw_scene(self._panel_context, x, y, w, h)

    def control_panel(self, x, y, w):
        u, s = self.ui, self.s
        session = self.preview.session
        ctx = self._panel_context
        panel = self.preview.panels.get("control")
        if panel._selection_revision != session.selection_revision:
            panel._selected_address = -1
            panel._selection_revision = session.selection_revision
        joint_nodes = {node.joint_index: node for node in session.nodes if node.joint_index >= 0}
        values = session.frame.ctrl
        enabled = session.adapter.caps.write_ctrl and values is not None
        row = 0
        for actuator in session.actuators:
            lo, hi = actuator.ctrl_range if actuator.ctrl_limited else (-1.0, 1.0)
            for component in range(actuator.ctrl_count):
                address = actuator.ctrl_address + component
                yy = y + (8 + row * 64) * s
                suffix = f"[{component}]" if actuator.ctrl_count > 1 else ""
                target = session.node(actuator.target_node_id) or joint_nodes.get(actuator.joint)
                pressed = u.button(
                    "control-target-" + str(address),
                    x,
                    yy,
                    w / s,
                    22,
                    label=actuator.name + suffix,
                    text_color=SAGE if panel._selected_address == address else MUTED,
                    active=panel._selected_address == address,
                    translate=False,
                    align="left",
                )
                hovered = imgui.is_item_hovered()
                focused = hovered and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
                if hovered and target is not None:
                    publish_focus_item_hint(ctx)
                if target is not None and (pressed or focused):
                    ctx.submit(cmd.SelectNode(target.node_id))
                    if focused:
                        joint = next(
                            (
                                item
                                for item in session.joints
                                if item.joint_id == target.joint_index
                            ),
                            None,
                        )
                        activate_edit_gizmo(ctx, target, joint)
                        if joint is not None and ctx.focus_joint is not None:
                            ctx.focus_joint(joint.joint_id)
                        elif ctx.focus_node is not None:
                            ctx.focus_node(target.node_id)
                if pressed or focused:
                    panel._selection_revision = session.selection_revision
                    panel._selected_address = address
                current = 0.0 if values is None else float(values[address])
                imgui.begin_disabled(not enabled)
                changed, value = u.slider(
                    "##ctrl-" + str(address), x, yy + 22 * s, w, current, lo, hi, "%.3f"
                )
                imgui.end_disabled()
                if changed and enabled:
                    session.submit(cmd.SetCtrl(address, value))
                row += 1
        yy = y + (24 + row * 64) * s
        if not session.actuators:
            u.text(x, y + 8 * s, "No actuators in this scene", MUTED, 13)
            yy += 22 * s
        if (
            u.button(
                "reset-actuators",
                x,
                yy,
                w / s,
                28,
                icon="reset",
                label="Reset controls",
                filled=True,
                enabled=enabled and row > 0,
            )
            and enabled
        ):
            session.submit(cmd.SetCtrlVector(np.zeros_like(values)))
        if u.button(
            "open-pose",
            x,
            yy + 48 * s,
            w / s,
            28,
            icon="pose",
            label="Open pose editor",
            filled=True,
        ):
            self.activate("Pose")
        imgui.set_cursor_screen_pos((x, yy + 92 * s))
        imgui.dummy((1, 1))

    def inspector(self, x, y, w, h):
        u = self.ui
        s = self.s
        e = (
            next((e for e in self.preview.entities if e["identity"] == self.pinned), None)
            if self.pinned
            else self.preview.entity
        )
        if self.pinned and e is None:
            self.pinned = None
            e = self.preview.entity
        imgui.set_cursor_screen_pos((x, y))
        imgui.invisible_button("inspector-header", (max(s, w - 60 * s), 30 * s))
        u.hits["inspector-header"] = (x, y, max(s, w - 60 * s), 30 * s)
        if imgui.is_item_active() and imgui.is_mouse_dragging(0):
            window = imgui.internal.get_current_window()
            if window.dock_node is not None:
                imgui.internal.dock_context_queue_undock_window(imgui.get_current_context(), window)
            else:
                imgui.internal.start_mouse_moving_window(window)
        u.role_icon(
            e["icon"] if e else "inspect",
            x,
            y,
            20 * s,
            30 * s,
            role="title",
            c=e["icon_color"] if e else MUTED,
        )
        imgui.push_clip_rect((x, y), (x + w - 64 * s, y + 30 * s), True)
        u.text(
            x + 28 * s,
            y + 7.5 * s,
            e["name"] if e else "Inspector",
            TEXT,
            15,
            bold=True,
            max_width=max(0, w - 92 * s),
            translate=e is None,
        )
        imgui.pop_clip_rect()
        if e and imgui.is_item_hovered():
            u.tooltip(e["name"], translate=False)
        if u.button(
            "pin",
            x + w - 56 * s,
            y + s,
            26,
            28,
            icon="pin",
            icon_role="subtle",
            active=bool(self.pinned),
            enabled=e is not None,
            tooltip="Pin Inspector",
        ):
            self.pinned = None if self.pinned else e["identity"]
        if u.button(
            "inspector-close",
            x + w - 26 * s,
            y + s,
            26,
            28,
            icon="close",
            icon_role="subtle",
            tooltip="Close",
        ):
            self.open["Inspector"] = False
        if not e:
            u.text(x, y + 44 * s, "Select an entity to inspect.", MUTED)
            imgui.set_cursor_screen_pos((x, y + 44 * s))
            imgui.dummy((0, 28 * s))
            return
        tabs = ["Properties", "Physics", "Material", "Joint"]
        tab_icons = ("inspect", "body", "shading", "pose")
        tab_minimums = [u.measure(u.translate(tab)) + 12 * s for tab in tabs]
        compact_tabs = w < sum(tab_minimums)
        tab_x = x
        for n, t in enumerate(tabs):
            tw = w / 4 if compact_tabs else tab_minimums[n] + (w - sum(tab_minimums)) / 4
            tx = tab_x
            if u.button(
                "tab-" + t,
                tx,
                y + 38 * s,
                tw / s,
                32,
                label="" if compact_tabs else t,
                icon=tab_icons[n] if compact_tabs else None,
                tooltip=t if compact_tabs else None,
                align="center",
                text_color=SAGE if self.inspector_tab == t else MUTED,
                icon_color=SAGE if self.inspector_tab == t else MUTED,
            ):
                self.inspector_tab = t
            if self.inspector_tab == t:
                u.draw.add_line(
                    (tx + 8 * s, y + 70 * s), (tx + tw - 8 * s, y + 70 * s), color(SAGE), 2 * s
                )
            tab_x += tw
        body_y = y + 82 * s
        imgui.set_cursor_screen_pos((x, body_y))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Inspector content", (w, max(s, h - 82 * s)))
        imgui.pop_style_var()
        p = imgui.get_cursor_screen_pos()
        bx, yy = p.x, p.y
        bw = imgui.get_content_region_avail().x
        original = next(
            (
                value
                for value in self.preview.initial_entities
                if value["identity"] == e["identity"]
            ),
            None,
        )
        if draw_component(self, self._panel_context, e, self.inspector_tab):
            yy = max(yy, imgui.get_cursor_screen_pos().y)
        elif self.inspector_tab == "Properties":
            if u.section("TRANSFORM", bx, yy, bw - 32 * s, self.expanded["TRANSFORM"]):
                self.expanded["TRANSFORM"] = not self.expanded["TRANSFORM"]
            if u.button(
                "restore-transform",
                bx + bw - 28 * s,
                yy + 6 * s,
                26,
                26,
                icon="reset",
                tooltip="Restore transform",
                enabled=not self.playing and e["posable"] and original is not None,
            ):
                for key in ("position", "rotation", "scale"):
                    e[key] = original[key]
                self.preview.apply_document_edits()
            if self.expanded["TRANSFORM"]:
                layout = u.vector_layout(bw)
                row_y = yy + 42 * s
                for key, label, unit in (
                    ("position", "Position", "m"),
                    ("rotation", "Rotation", "°"),
                    ("scale", "Scale", ""),
                ):
                    imgui.begin_disabled(
                        self.playing or not e["scalable" if key == "scale" else "posable"]
                    )
                    changed, val = u.vector(
                        "##" + e["id"] + key,
                        bx,
                        row_y,
                        bw,
                        label,
                        e[key],
                        unit,
                        speed=0.5 if key == "rotation" else 0.01,
                        lo=0.01 if key == "scale" else 0,
                        hi=100 if key == "scale" else 0,
                        format="%.3g",
                        reset_values=(1.0, 1.0, 1.0) if key == "scale" else (0.0, 0.0, 0.0),
                        layout=layout,
                    )
                    imgui.end_disabled()
                    edit_key = "##" + e["id"] + key
                    state = self.begin_numeric_edit(edit_key, deferred=key == "scale")
                    if (
                        changed
                        and not state.canceled
                        and all(np.isfinite(val))
                        and (key != "scale" or (min(val) >= 0.01 and max(val) <= 100))
                    ):
                        e[key] = val
                    self.preview.apply_document_edits()
                    self.finish_numeric_edit(edit_key, state)
                    row_y += layout.height + 8 * s
                yy = row_y + 16 * s
            else:
                yy += 40 * s
            if u.section("DISPLAY", bx, yy, bw, self.expanded["DISPLAY"]):
                self.expanded["DISPLAY"] = not self.expanded["DISPLAY"]
            if self.expanded["DISPLAY"]:
                u.text(bx, yy + 44 * s, "Visible", MUTED, 13)
                e["hidden"] = not u.switch(
                    "entity-visible", bx + bw - SWITCH_WIDTH * s, yy + 36.5 * s, not e["hidden"]
                )
                u.text(bx, yy + 78 * s, "Transform handles", MUTED, 13)
                self.preview.gizmos = u.switch(
                    "gizmos", bx + bw - SWITCH_WIDTH * s, yy + 70.5 * s, self.preview.gizmos
                )
                yy += 113 * s
            else:
                yy += 40 * s
            if u.section("IDENTITY", bx, yy, bw, self.expanded["IDENTITY"]):
                self.expanded["IDENTITY"] = not self.expanded["IDENTITY"]
            yy += 42 * s
            if self.expanded["IDENTITY"]:
                u.text(bx, yy, "Object ID", MUTED, 13)
                u.text(bx + 100 * s, yy, str(e["object_id"]), TEXT, 13)
                u.text(bx, yy + 30 * s, "Type", MUTED, 13)
                u.text(bx + 100 * s, yy + 30 * s, e["type"], TEXT, 13)
                yy += 64 * s
        elif self.inspector_tab == "Material":
            u.section("SURFACE", bx, yy, bw)
            imgui.begin_disabled(not e["colorable"])
            self.menus.color_control(e, bx, yy + 48 * s, bw)
            layout = u.numeric_layout(bw, 4, column_options=(4, 2, 1))
            for n, axis in enumerate("RGBA"):
                dx, dy, field_w, _ = layout.fields[n]
                key = "##" + e["id"] + "color" + axis
                changed, value = u.input(
                    key,
                    bx + dx,
                    yy + 112 * s + dy,
                    field_w,
                    e["rgba"][n],
                    axis=axis,
                    speed=0.005,
                    lo=0,
                    hi=1,
                    format="%.3g",
                    reset_value=original["rgba"][n] if original else (1.0 if axis == "A" else 0.0),
                )
                state = self.begin_numeric_edit(key)
                if changed and not state.canceled:
                    e["rgba"][n] = value
                    e["color"] = "#" + "".join(f"{round(v * 255):02x}" for v in e["rgba"][:3])
                self.preview.apply_document_edits()
                self.finish_numeric_edit(key, state)
            imgui.end_disabled()
            yy += 136 * s + layout.height
        imgui.set_cursor_screen_pos((bx, yy))
        imgui.dummy((1, 1))
        imgui.end_child()

    def assets_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 5 * s, "MATERIAL LIBRARY", MUTED, 13, bold=True)
        palette = [
            ("Sage ceramic", "#9cbf8d"),
            ("Amber satin", "#e3bc67"),
            ("Slate matte", "#829cb7"),
            ("Coral polymer", "#d58979"),
        ]
        for n, (name, hexcolor) in enumerate(palette):
            yy = y + (38 + n * 56) * s
            u.rect(x, yy, w, 48 * s, (37, 41, 46, 255), 6)
            if u.button("material-" + name, x, yy, w / s, 48, active=self.material == n):
                self.material = n
            u.draw.add_rect(
                (x, yy),
                (x + w, yy + 48 * s),
                color((*SAGE[:3], 90) if self.material == n else LINE),
                6 * s,
                thickness=s,
            )
            u.draw.add_circle_filled(
                (x + 28 * s, yy + 24 * s),
                16 * s,
                color((*[int(hexcolor[i : i + 2], 16) for i in (1, 3, 5)], 255)),
            )
            u.text(x + 56 * s, yy + 17 * s, name, TEXT, 13)
            if self.material == n:
                u.role_icon(
                    "check",
                    x + w - 26 * s,
                    yy + 10 * s,
                    16 * s,
                    28 * s,
                    role="inline",
                    c=SAGE,
                )
        if u.button(
            "apply-material",
            x,
            y + 276 * s,
            w / s,
            28,
            icon="check",
            label="Apply material",
            align="center",
            active=True,
            solid=True,
            enabled=self.preview.entity is not None and self.preview.entity["colorable"],
        ):
            self.preview.entity["color"] = palette[self.material][1]
            self.preview.apply_document_edits()
        imgui.set_cursor_screen_pos((x, y + 320 * s))
        imgui.dummy((1, 1))

    def pose_panel(self, x, y, w, *, node_id=None):
        from .navigation import Navigation

        if not hasattr(self, "navigation"):
            self.navigation = Navigation(self)
        self.navigation.draw_pose(self._panel_context, x, y, w, node_id=node_id)

    def camera_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 8 * s, "VIEW DIRECTION", MUTED, 13, bold=True)
        for n, name in enumerate(["Front", "Back", "Left", "Right", "Top", "Bottom"]):
            if u.button(
                "camera-" + name,
                x + (n % 2) * (w / 2 + 3 * s),
                y + (42 + n // 2 * 36) * s,
                (w / 2 - 3 * s) / s,
                28,
                icon="camera",
                label=name,
                align="center",
                filled=True,
            ):
                self.preview.camera.set_preset(name.lower())
        projection = u.segmented(
            "projection",
            x,
            y + 176 * s,
            w / s,
            ("Perspective", "Orthographic"),
            int(self.preview.camera.orthographic),
            icons=("perspective", "orthographic"),
            layout="content",
        )
        if bool(projection) != self.preview.camera.orthographic:
            self.preview.camera.set_orthographic(bool(projection))
        projection_rect = u.hits["projection"]
        frame_y = projection_rect[1] + projection_rect[3] + 12 * s
        if u.button(
            "camera-frame", x, frame_y, w / s, 28, icon="frame", label="Frame all", filled=True
        ):
            self.preview._frame_scene(animate=True)
        imgui.set_cursor_screen_pos((x, frame_y + 54 * s))
        imgui.dummy((1, 1))

    def layers_panel(self, x, y, w):
        u, s, ctx = self.ui, self.s, self._panel_context
        yy = y

        def visibility_row(key, label, visible, tooltip=""):
            toggled = u.button(
                key + "-label",
                x,
                yy,
                max(1, w / s - SWITCH_WIDTH - 8),
                28,
                label=label,
                text_color=MUTED,
                tooltip=tooltip,
                align="left",
            )
            value = u.switch(key, x + w - SWITCH_WIDTH * s, yy, visible)
            return not visible if toggled else value

        labels = (
            "Viewport UI",
            "Gizmos",
            "Selection",
            "Helpers",
            "Perturbation",
            "3D drawings",
            "2D drawings",
        )
        for (name, description), label in zip(LAYER_LABELS, labels, strict=True):
            layers = self.preview.viewport_layers
            value = visibility_row(
                "layer-" + name, ctx.tr(label), getattr(layers, name), ctx.tr(description)
            )
            if value != getattr(layers, name):
                ctx.set_viewport_layers(replace(layers, **{name: value}))
            yy += 34 * s
        draw = ctx.backend.debug
        named = (
            tuple(
                layer
                for layer in draw.layers()
                if debug_layer_group(layer.name) in ("debug_3d", "debug_2d")
                and (
                    layer.primitives
                    or layer.name in self.preview.viewport_layers.hidden_debug_layers
                )
            )
            if draw is not None
            else ()
        )
        if named:
            u.section(ctx.tr("Named drawing layers"), x, yy + 8 * s, w)
            yy += 50 * s
        for layer in named:
            layers = self.preview.viewport_layers
            visible = layer.name not in layers.hidden_debug_layers
            imgui.begin_disabled(not getattr(layers, debug_layer_group(layer.name)))
            value = visibility_row(
                "drawing-layer-" + layer.name, layer.name.removeprefix("canvas2d:"), visible
            )
            imgui.end_disabled()
            if value != visible:
                hidden = tuple(name for name in layers.hidden_debug_layers if name != layer.name)
                if not value:
                    hidden += (layer.name,)
                ctx.set_viewport_layers(replace(layers, hidden_debug_layers=hidden))
            yy += 34 * s
        imgui.set_cursor_screen_pos((x, yy))
        imgui.dummy((1, 1))

    def timeline(self, x, y, w, h):
        from .timeline_panel import ReferenceTimeline

        if self.timeline_panel is None:
            self.timeline_panel = ReferenceTimeline(self)
        self.timeline_panel.draw(self._panel_context, x, y, w, h)

    def viewport(self, preview_name=""):
        u = self.ui
        s = self.s
        app = self.preview
        image = app._viewport_image
        app._viewport_rect = app.viewport_surface.draw_image(
            image, self.window.viewport_texture_ref
        )
        x, y, w, h = app._viewport_rect
        self.rects["viewport_content"] = [x, y, w, h]
        horizontal = h < 440 * s
        step = 32 if w < 280 * s else 40
        wrapped = horizontal and w < 260 * s
        tools_height = 76 if wrapped else 40
        footer_space = (
            10
            if app.model_edits.active and h < 300 * s
            else 42
            if h >= 210 * s and not self.open["Inspector"]
            else 32
        )
        tools_y = (
            y + h - (tools_height + footer_space) * s
            if horizontal
            else y + max(148, (h / s - 248) / 2) * s
        )
        compact_camera = w < 240 * s
        stacked_camera = w < 160 * s
        camera_height = 68 if stacked_camera else 36
        transport_visible = h >= (248 if w < 260 * s else 220) * s
        transport_width = 168 if w < 340 * s else 282
        rail_width = step * (3 if wrapped else 6) + 8 if horizontal else 44
        collapsed_controls = (
            w < (rail_width + 20) * s
            or (horizontal and tools_y < y + (camera_height + 18) * s)
            or (transport_visible and w < (transport_width + 20) * s)
            or (app.camera_preview.enabled and bool(preview_name) and h < 300 * s)
        )
        if app.model_edits.active and horizontal:
            pending_height = 70 if w < 412 * s and h >= 300 * s else 40
            collapsed_controls |= tools_y - (pending_height + 8) * s < y + (camera_height + 18) * s
        cube_scale = self.cube.scale_for((x, y, w, h), s)
        cube_rect = None
        if cube_scale > 0:
            center = widget_center((x, y, w, h), cube_scale)
            radius = BACKDROP_RADIUS_PT * cube_scale
            cube_rect = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            cx, cy, cw, ch = cube_rect
            if horizontal:
                collapsed_controls |= (
                    x + 10 * s < cx + cw
                    and cx < x + (10 + rail_width) * s
                    and tools_y < cy + ch
                    and cy < tools_y + tools_height * s
                )
        imgui.push_clip_rect((x, y), (x + w, y + h), True)
        app._draw_scene_overlays(self.window.painter())
        overlay = []
        if app.model_edits.active:
            self.pending_model_edits(
                x,
                y,
                w,
                h,
                overlay,
                bottom_limit=tools_y if horizontal and not collapsed_controls else None,
            )
        if not app.viewport_layers.viewport_ui:
            self.rects["overlays"] = overlay
            self.input_overlays = list(overlay)
            imgui.pop_clip_rect()
            app._draw_reference_camera_preview(preview_name)
            imgui.end()
            return

        def shell(rx, ry, rw, rh):
            u.rect(rx, ry, rw * s, rh * s, (22, 25, 28, 230), 8)
            overlay.append((rx, ry, rw * s, rh * s))

        if collapsed_controls:
            shell(x + 10 * s, y + 10 * s, 36, 36)
            popup = "Viewport controls###viewport-more"
            if u.button(
                "viewport-more",
                x + 14 * s,
                y + 14 * s,
                28,
                28,
                icon="more",
                tooltip="Viewport controls",
                active=imgui.is_popup_open(popup),
            ):
                imgui.open_popup(popup)
            if (
                self.menus.requested_path
                and self.menus.requested_path[0]
                in (
                    "Camera",
                    "Shading",
                    "Playback",
                )
                and not imgui.is_popup_open(popup)
            ):
                imgui.open_popup(popup)
            with self.menus.style():
                if imgui.begin_popup(popup):

                    def menu_action(key, label, shortcut="", checked=False, enabled=True):
                        clicked, _ = imgui.menu_item(u.translate(label), shortcut, checked, enabled)
                        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                        u.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
                        return clicked

                    for name in ("Camera", "Shading"):
                        self.menus.reveal((name,))
                        self.menus.constrain(name)
                        if imgui.begin_menu(u.translate(name) + "###" + name):
                            self.menus.content(name, (name,))
                            imgui.end_menu()
                    if (
                        cube_scale <= 0
                        and self.cube.preview_active
                        and imgui.begin_menu(u.translate("View direction"))
                    ):
                        for name in ("Front", "Back", "Left", "Right", "Top", "Bottom"):
                            if menu_action("camera-preview-view-" + name, name):
                                app._leave_model_camera()
                                app.camera.set_preset(name.lower())
                        imgui.end_menu()
                    imgui.separator()
                    for name, action in VIEWPORT_TOOLS:
                        shortcut = "V" if action is None else app.input_bindings.label(action)
                        if shortcut == "Unbound":
                            shortcut = ""
                        if menu_action("tool-" + name, name.title(), shortcut, app.tool == name):
                            app.tool = name
                    if menu_action(
                        "frame",
                        "Body frame",
                        app.input_bindings.label(InputAction.GIZMO_SPACE),
                        self.space == "Body",
                    ):
                        self.space = "World" if self.space == "Body" else "Body"
                    if menu_action(
                        "snap", "Snap", app.input_bindings.label(InputAction.SNAP), self.snap
                    ):
                        self.snap = not self.snap
                    if menu_action("viewport-layers", "Viewport layers"):
                        self.activate("Layers")
                    imgui.separator()
                    self.menus.reveal(("Playback",))
                    self.menus.constrain("Playback")
                    if imgui.begin_menu(u.translate("Playback options") + "###Playback"):
                        session = app.session
                        if menu_action(
                            "play", "Pause simulation" if self.playing else "Play simulation"
                        ):
                            app._toggle_playback()
                        if menu_action("go-start", "Go to start"):
                            app._reset_playback()
                        if menu_action(
                            "record",
                            "Record simulation",
                            checked=session.state_take_recording,
                            enabled=session.adapter.caps.simulation,
                        ):
                            app._toggle_state_take_recording()
                        imgui.separator()
                        self.menus.content("Playback", ("Playback",))
                        imgui.end_menu()
                    imgui.end_popup()
        else:
            camera_width = (
                44 if stacked_camera else 72 if compact_camera else 144 if w < 300 * s else 176
            )
            shell(x + 10 * s, y + 10 * s, camera_width, 68 if stacked_camera else 36)
            camera_clicked = u.button(
                "projection-button",
                x + 13 * s,
                y + 14 * s,
                28 if compact_camera else 98,
                28,
                icon="camera",
                label=""
                if compact_camera
                else "Ortho"
                if self.preview.camera.orthographic
                else "Persp",
                tooltip="Camera" if compact_camera else None,
                active=imgui.is_popup_open("Camera"),
                align="left",
            )
            if not compact_camera:
                u.role_icon("down", x + 92 * s, y + 14 * s, 12 * s, 28 * s, role="inline", c=DIM)
            self.menus.popup("Camera", u.hits["projection-button"], camera_clicked)
            shading_clicked = u.button(
                "shading",
                x + (13 if stacked_camera else 47 if compact_camera else 116) * s,
                y + (46 if stacked_camera else 14) * s,
                28,
                28,
                icon="shading",
                tooltip="Shading",
                active=imgui.is_popup_open("Shading"),
            )
            self.menus.popup("Shading", u.hits["shading"], shading_clicked)
            if camera_width == 176 and u.button(
                "viewport-layers",
                x + 148 * s,
                y + 14 * s,
                28,
                28,
                icon="layers",
                tooltip="Viewport layers",
            ):
                self.activate("Layers")
            if transport_visible:
                self.transport(x, y, w, shell)
            shell(
                x + 10 * s,
                tools_y,
                step * (3 if wrapped else 6) + 8 if horizontal else 44,
                tools_height if horizontal else 248,
            )

            def tool_pos(n):
                if wrapped:
                    return x + (14 + n % 3 * step) * s, tools_y + (4 + n // 3 * 36) * s
                return (
                    x + (14 + n * step if horizontal else 14) * s,
                    tools_y
                    + (4 if horizontal else (176, 211)[n - 4] if n >= 4 else 4 + n * 40) * s,
                )

            for n, (name, action) in enumerate(VIEWPORT_TOOLS):
                xx, yy = tool_pos(n)
                button_size = step - 4 if horizontal else 36
                shortcut = "V" if action is None else app.input_bindings.label(action)
                tip = "Dimensions: select a geometry" if name == "scale" else name.title()
                if shortcut != "Unbound":
                    tip += " · " + shortcut
                if u.button(
                    "tool-" + name,
                    xx,
                    yy,
                    button_size,
                    32 if horizontal else 36,
                    icon=name,
                    active=self.preview.tool == name,
                    solid=True,
                    tooltip=tip,
                ):
                    self.preview.tool = name
            if not horizontal:
                u.line(x + 18 * s, tools_y + 168 * s, 28 * s)
            if u.button(
                "frame",
                *tool_pos(4),
                step - 4 if horizontal else 36,
                32,
                icon="body" if self.space == "Body" else "world",
                tooltip="World / body frame · " + app.input_bindings.label(InputAction.GIZMO_SPACE),
            ):
                self.space = "Body" if self.space == "World" else "World"
            if u.button(
                "snap",
                *tool_pos(5),
                step - 4 if horizontal else 36,
                32,
                icon="snap",
                active=self.snap,
                tooltip="Snap · " + app.input_bindings.label(InputAction.SNAP),
            ):
                self.snap = not self.snap
        self.input_overlays = list(overlay)
        if cube_rect is not None:
            overlay.append(cube_rect)
        if (
            h >= 210 * s
            and not self.open["Inspector"]
            and not (app.model_edits.active and h < 300 * s)
        ):
            footer_w = min(w - 20 * s, 248 * s)
            u.rect(x + 10 * s, y + h - 38 * s, footer_w, 28 * s, (22, 25, 28, 230), 4.8)
            u.role_icon("link", x + 18 * s, y + h - 38 * s, 16 * s, 28 * s, role="inline", c=AMBER)
            u.text(
                x + 42 * s,
                y + h - 31 * s,
                self.preview.selected or "No selection",
                MUTED,
                13,
                translate=not bool(self.preview.selected),
            )
            overlay.append((x + 10 * s, y + h - 38 * s, footer_w, 28 * s))
        self.rects["overlays"] = overlay
        imgui.pop_clip_rect()
        if not self.bottom_open:
            yy = y + h
            u.rect(x, yy, w, 38 * s, PANEL, 0)
            u.line(x, yy, w)
            if u.button(
                "expand-timeline",
                x + 8 * s,
                yy + 5 * s,
                26,
                28,
                icon="right",
                tooltip="Expand Timeline",
            ):
                self.toggle_timeline()
            timeline_width = max(95, u.measure(u.translate("Timeline")) / s + 39)
            output_width = max(85, u.measure(u.translate("Output")) / s + 39)
            if u.button(
                "timeline-tab",
                x + 38 * s,
                yy + 5 * s,
                timeline_width,
                28,
                icon="key",
                label="Timeline",
            ):
                self.activate("Timeline")
            if w > (timeline_width + output_width + 53) * s and u.button(
                "output-tab",
                x + (45 + timeline_width) * s,
                yy + 5 * s,
                output_width,
                28,
                icon="terminal",
                label="Output",
            ):
                self.activate("Output")
        app._draw_reference_camera_preview(preview_name)
        imgui.end()

    def pending_model_edits(self, x, y, w, h, overlay, *, bottom_limit=None):
        u, s, app = self.ui, self.s, self.preview
        width = min(w / s - 20, 392)
        compact = width < 392 and h < 300 * s
        stacked = width < 392 and not compact
        height = 70 if stacked else 40
        px = x + (w - width * s) / 2
        py = (
            bottom_limit - (height + 8) * s
            if bottom_limit is not None
            else y + max(10, h / s - height - 10) * s
        )
        u.rect(px, py, width * s, height * s, (22, 25, 28, 245), 8)
        overlay.append((px, py, width * s, height * s))
        label = "Applying…" if app.model_edits.applying else "Pending model edits"
        if compact:
            u.button(
                "model-edit-status", px + 8 * s, py + 6 * s, 28, 28, icon="info", tooltip=label
            )
        else:
            u.text(px + 12 * s, py + 13 * s, label, AMBER, 13)
        if app.model_edits.error and imgui.is_mouse_hovering_rect(
            (px, py), (px + width * s, py + height * s)
        ):
            u.tooltip(app.model_edits.error)
        apply_width = min(76, (width - 52) / 2) if compact else 76
        discard_width = min(80, (width - 52) / 2) if compact else 80
        bx = px + (width - apply_width - discard_width - 14 if not stacked else 8) * s
        by = py + (36 if stacked else 6) * s
        enabled = not app.model_edits.applying and not app.gizmo.using
        if u.button(
            "apply-model-edits",
            bx,
            by,
            apply_width,
            28,
            label="" if compact else "Apply",
            icon="check" if compact else None,
            tooltip="Apply" if compact else None,
            active=True,
            solid=True,
            enabled=enabled,
        ):
            app._apply_model_edits_requested = True
        if u.button(
            "discard-model-edits",
            bx + (apply_width + 6) * s,
            by,
            discard_width,
            28,
            label="" if compact else "Discard",
            icon="close" if compact else None,
            tooltip="Discard" if compact else None,
            filled=True,
            enabled=enabled,
        ):
            app.model_edits.clear()
            app.gizmo._reset_model_placement()
            app._apply_model_edits_requested = False

    def transport(self, x, y, w, shell):
        u, s = self.ui, self.s
        session = self.preview.session
        compact = w < 340 * s
        width = 168 if compact else 282
        if w < (width + 20) * s:
            return
        tx = x + (w - width * s) / 2
        ty = y + (104 if w < 500 * s else 52 if w < 700 * s else 10) * s
        shell(tx, ty, width, 36)
        if u.button(
            "previous",
            tx + 4 * s,
            ty + 4 * s,
            26,
            28,
            icon="previous",
            icon_role="subtle",
            tooltip="Previous frame",
            enabled=session.can_step_back,
        ):
            session.submit(cmd.StepBack())
        if u.button(
            "play",
            tx + 33 * s,
            ty + 4 * s,
            32,
            28,
            icon="pause" if self.playing else "play",
            icon_role="subtle",
            active=True,
            tooltip="Play / pause",
        ):
            self.preview._toggle_playback()
        if u.button(
            "next",
            tx + 69 * s,
            ty + 4 * s,
            26,
            28,
            icon="next",
            icon_role="subtle",
            tooltip="Next frame",
            enabled=not self.playing and session.adapter.caps.simulation,
        ):
            session.submit(cmd.Step())
        if not compact:
            caption = f"{self.time:.3f} s"
            imgui.push_font(u.mono, 13 * s)
            caption_width = imgui.calc_text_size(caption).x
            imgui.pop_font()
            u.text(tx + 142 * s - caption_width / 2, ty + 11 * s, caption, TEXT, 13, mono=True)
        for offset in (100,) if compact else (102, 182):
            u.draw.add_line(
                (tx + offset * s, ty + 9 * s),
                (tx + offset * s, ty + 27 * s),
                color((255, 255, 255, 32)),
                s,
            )
        if u.button(
            "go-start",
            tx + (107 if compact else 189) * s,
            ty + 4 * s,
            27,
            28,
            icon="reset",
            icon_role="subtle",
            tooltip="Go to start",
        ):
            self.preview._reset_playback()
        if not compact and u.button(
            "record",
            tx + 220 * s,
            ty + 4 * s,
            26,
            28,
            icon="record",
            icon_role="subtle",
            active=self.preview.session.state_take_recording,
            tooltip="Record state take",
            enabled=session.adapter.caps.simulation,
        ):
            self.preview._toggle_state_take_recording()
        playback_clicked = u.button(
            "playback-options",
            tx + (138 if compact else 250) * s,
            ty + 4 * s,
            26,
            28,
            icon="down",
            icon_role="subtle",
            tooltip="Playback options",
            active=imgui.is_popup_open("Playback"),
        )
        self.menus.popup("Playback", u.hits["playback-options"], playback_clicked)

    def snapshot(self):
        result = {}
        for name in ["Viewport", *self.open]:
            win = imgui.internal.find_window_by_name(name)
            if win and (name == "Viewport" or self.open[name]):
                result[name] = {
                    "dock": win.dock_id,
                    "rect": [win.pos.x, win.pos.y, win.size.x, win.size.y],
                }
        result["viewport_content"] = self.rects["viewport_content"]
        result["overlays"] = self.rects["overlays"]
        return result

    def close(self):
        self.preview.close()


def frames(window, study, path=None, count=8):
    pixels = None
    for _ in range(count):
        pixels = study.draw(readback=path is not None)
    if path:
        Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(path)
    snapshot = study.snapshot()
    vx, vy, vw, vh = snapshot["viewport_content"]
    rectangles = snapshot["overlays"]
    manual_preview = (
        study.rects.get("camera_preview")
        if study.preview.camera_preview._position is not None
        else None
    )
    for n, (x, y, w, h) in enumerate(rectangles):
        assert x >= vx and y >= vy and x + w <= vx + vw + 0.01 and y + h <= vy + vh + 0.01, (
            "Overlay outside viewport",
            n,
            rectangles[n],
            snapshot["viewport_content"],
        )
        for other in rectangles[:n]:
            # A user-positioned preview may cross toolbar regions. Automatic
            # placement and every other overlay pair retain collision checks.
            if manual_preview is not None and (
                tuple(rectangles[n]) == tuple(manual_preview)
                or tuple(other) == tuple(manual_preview)
            ):
                continue
            a, b, c, d = other
            assert x >= a + c or a >= x + w or y >= b + d or b >= y + h, (
                "Overlapping viewport controls",
                rectangles[n],
                other,
            )
    return snapshot


def icon_gallery(window, path, *, portable_fonts=False):
    ui = UI(window, portable_fonts=portable_fonts)
    s = ui.s
    pixels = None
    for _ in range(8):
        window.begin_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.begin("Icon review", None, imgui.WindowFlags_.no_decoration)
        ui.text(24 * s, 22 * s, "Instrument · shared icon masters", TEXT, 17, bold=True)
        ui.text(
            24 * s,
            49 * s,
            "Native draw-list rendering at 14 / 16 / 18 / 24 logical pixels",
            MUTED,
            11,
        )
        for n, name in enumerate(GLYPHS):
            x, y = (24 + n % 12 * 116) * s, (88 + n // 12 * 90) * s
            ui.text(x, y, name, DIM, 11)
            for size, offset in [(14, 0), (16, 24), (18, 50), (24, 80)]:
                ui.icon(name, x + offset * s, y + (37 - size / 2) * s, size, TEXT)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(path)
    focus = (
        "performance",
        "snap",
        "move",
        "joint-hinge",
        "joint-slide",
        "joint-ball",
        "joint-free",
        "perspective",
        "orthographic",
        "pose",
        "rotate",
        "scale",
    )
    rounding = (
        "settings",
        "general",
        "camera",
        "mouse",
        "shading",
        "video",
        "follow",
        "follow-off",
        "follow-page",
        "follow-locked",
        "circle",
        "mujoco-visuals",
    )
    for filename, title, names in (
        ("icon-geometry.png", "Icon geometry · domain symbols", focus),
        ("icon-rounding.png", "Icon geometry · round boundaries and follow modes", rounding),
        (
            "icon-views.png",
            "Icon geometry · camera directions",
            tuple(
                "view-" + name
                for name in ("front", "back", "left", "right", "top", "bottom", "iso")
            ),
        ),
        (
            "icon-semantics.png",
            "Icon geometry · objects, projection and capture actions",
            (
                "camera",
                "capture-image",
                "snapshot",
                "perspective",
                "orthographic",
                "view-iso",
                "body-frame",
                "shape-box",
                "pose",
            ),
        ),
        (
            "icon-shapes.png",
            "Icon geometry · scene primitives",
            tuple(
                "shape-" + name
                for name in ("box", "sphere", "cylinder", "capsule", "cone", "ellipsoid", "plane")
            ),
        ),
        (
            "icon-junctions.png",
            "Icon geometry · continuous shells and clear internal gaps",
            (
                "shape-box",
                "shape-sphere",
                "shape-ellipsoid",
                "view-iso",
                "delete",
                "pose",
                "shape-cylinder",
                "shape-cone",
                "performance",
            ),
        ),
        *(
            (
                f"icon-detail-{index // 12 + 1:02d}.png",
                f"Icon geometry · inventory {index + 1}–{min(index + 12, len(GLYPHS))}",
                tuple(GLYPHS)[index : index + 12],
            )
            for index in range(0, len(GLYPHS), 12)
        ),
    ):
        for _ in range(6):
            window.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(imgui.get_io().display_size)
            imgui.begin("Icon geometry", None, imgui.WindowFlags_.no_decoration)
            ui.text(24 * s, 22 * s, title, TEXT, 17, bold=True)
            ui.text(
                24 * s,
                50 * s,
                "14 / 24 / 56 / 112 px · square slots, original silhouette proportions · guides stay outside the painter",
                MUTED,
                11,
            )
            for n, name in enumerate(names):
                x, y = (24 + n % 3 * 464) * s, (90 + n // 3 * 195) * s
                ui.text(x, y, name, TEXT, 12)
                for size, offset in ((14, 0), (24, 45), (56, 104), (112, 210)):
                    xx, yy = x + offset * s, y + (38 + (112 - size) / 2) * s
                    ui.draw.add_rect(
                        (xx, yy), (xx + size * s, yy + size * s), color((75, 83, 90, 160)), 0
                    )
                    ui.draw.add_circle(
                        (xx + size * s / 2, yy + size * s / 2),
                        size * s / 2,
                        color((100, 91, 66, 100)),
                    )
                    ui.icon(name, xx, yy, size, TEXT)
                    ui.text(xx, y + 159 * s, str(size), DIM, 11)
            imgui.end()
            pixels = window.end_frame(readback=True)
        Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(path.with_name(filename))
    for _ in range(6):
        window.begin_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.begin("Icon roles", None, imgui.WindowFlags_.no_decoration)
        ui.text(24 * s, 22 * s, "Native icon roles", TEXT, 18, bold=True)
        ui.text(24 * s, 54 * s, "Canonical canvas / centered control slot (pt)", MUTED)
        role_focus = (
            "camera",
            "inspect",
            "pin",
            "lock",
            "close",
            "previous",
            "play",
            "pause",
            "next",
            "reset",
            "record",
            "down",
        )
        for column, (role, spec) in enumerate(ICON_ROLES.items()):
            x = (210 + column * 190) * s
            ui.text(x, 98 * s, f"{role.title()} · {spec.canvas_size:g} / {spec.slot_size:g}")
            for row, name in enumerate(role_focus):
                y = (144 + row * 48) * s
                if column == 0:
                    ui.text(24 * s, y + 10 * s, name, MUTED)
                size = spec.slot_size * s
                ui.draw.add_rect((x, y), (x + size, y + size), color(LINE), 0)
                ui.role_icon(name, x, y, size, size, role=role, c=TEXT)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(path.with_name("icon-roles.png"))
    io = imgui.get_io()
    report = {
        "logical_viewport": list(io.display_size),
        "framebuffer_density": list(io.display_framebuffer_scale),
        "sizes": [14, 16, 18, 24, 56, 112],
        "anchor": [12, 12],
        "roles": {
            name: {"canvas_size": spec.canvas_size, "slot_size": spec.slot_size}
            for name, spec in ICON_ROLES.items()
        },
        "glyphs": {
            name: {"source_canvas": g["viewBox"], "visible_bounds": g["bounds"]}
            for name, g in GLYPHS.items()
        },
    }
    path.with_name("icons-report.json").write_text(json.dumps(report, indent=2))


def create_review_window(config, args):
    """Keep initial and restored windows on the same logical review coordinates."""
    window = create_window(config, args.backend)
    if args.review_density != 1:
        # Use the measured framebuffer density so desktop scaling is included too.
        process_inputs = window._input.process_inputs

        def review_inputs():
            process_inputs()
            io = imgui.get_io()
            io.display_size = (args.width, args.height)
            fw, fh = window.size_pixels
            io.display_framebuffer_scale = (fw / args.width, fh / args.height)

        window._input.process_inputs = review_inputs
        review_inputs()
    return window


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--backend", choices=["opengl", "bgfx"], default="opengl")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--portable-fonts",
        action="store_true",
        help="Use bundled sans-serif fonts on every platform",
    )
    parser.add_argument("--ui-scale", type=float, default=1)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--icons", action="store_true")
    parser.add_argument("--menus", action="store_true")
    parser.add_argument("--typography", action="store_true")
    parser.add_argument("--appearance", action="store_true")
    parser.add_argument("--review-density", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.interactive and args.review_density != 1:
        parser.error("--review-density is an offscreen acceptance fixture")
    if args.icons:
        args.width = max(args.width, ceil(1440 * args.ui_scale))
        args.height = max(
            args.height, ceil((88 + ceil(len(GLYPHS) / 12) * 90 + 24) * args.ui_scale)
        )
    args.output.mkdir(parents=True, exist_ok=True)
    config = WindowConfig(
        title="Mojive · Instrument reference · 08",
        width=args.width * args.review_density,
        height=args.height * args.review_density,
        font_size_pt=12,
        ui_scale=args.ui_scale,
        ini_path="",
        show_on_start=False,
        docking=True,
    )
    window = create_review_window(config, args)
    study = None
    try:
        if args.typography:
            from .typography_review import capture_typography

            capture_typography(
                window, args.output / "typography", portable_fonts=args.portable_fonts
            )
            return
        if args.icons:
            icon_gallery(window, args.output / "icons.png", portable_fonts=args.portable_fonts)
            return
        study = Study(
            window, args.backend, output_directory=args.output, portable_fonts=args.portable_fonts
        )
        if args.appearance:
            from .appearance_review import capture_appearance

            capture_appearance(window, study, args.output / "appearance", frames)
            return
        if args.menus:
            from .menu_review import capture_menus

            capture_menus(window, study, args.output / "menus", frames)
            return
        if args.interactive:
            window.show()
            while not window.should_close():
                study.draw()
            return
        edit = frames(window, study, args.output / "edit.png")
        if args.capture_only:
            (args.output / "report.json").write_text(
                json.dumps(
                    {
                        "backend": args.backend,
                        "ui_scale": study.s,
                        "states": {"edit": edit},
                        "checks": "Viewport overlays remain contained and do not overlap.",
                    },
                    indent=2,
                )
            )
            return
        study.activate("Control")
        frames(window, study, args.output / "control.png")
        study.activate("Assets")
        frames(window, study, args.output / "assets.png")
        study.activate("Scene")
        frames(window, study)
        study.toggle_timeline()
        default = frames(window, study, args.output / "docked.png")
        assert abs(default["Scene"]["rect"][3] - default["Inspector"]["rect"][3]) < 2
        assert abs(default["Timeline"]["rect"][0] - default["Viewport"]["rect"][0]) < 2
        assert abs(default["Timeline"]["rect"][2] - default["Viewport"]["rect"][2]) < 2
        saved = imgui.save_ini_settings_to_memory()
        (args.output / "layout.ini").write_text(saved)
        study.command("float")
        floating = frames(window, study, args.output / "floating.png")
        assert floating["Inspector"]["dock"] == 0
        fx, fy, fw, fh = floating["Inspector"]["rect"]
        vp = imgui.get_main_viewport()
        assert (
            fx >= vp.work_pos.x
            and fy >= vp.work_pos.y
            and fx + fw <= vp.work_pos.x + vp.work_size.x + 1
            and fy + fh <= vp.work_pos.y + vp.work_size.y + 1
        ), ("Floating Inspector exceeds the workspace", floating["Inspector"])
        study.command("dock")
        redocked = frames(window, study, args.output / "redocked.png")
        assert redocked["Inspector"]["dock"] != 0
        study.command("group")
        grouped = frames(window, study, args.output / "tabbed.png")
        assert grouped["Inspector"]["dock"] == grouped["Scene"]["dock"] != 0
        report = {
            "version": imgui.get_version(),
            "backend": args.backend,
            "states": {
                "edit": edit,
                "docked": default,
                "floating": floating,
                "redocked": redocked,
                "tabbed": grouped,
            },
        }
    finally:
        if study:
            study.close()
        window.close()
    window = create_review_window(config, args)
    study = None
    try:
        study = Study(
            window,
            args.backend,
            saved,
            output_directory=args.output,
            portable_fonts=args.portable_fonts,
        )
        study.open["Timeline"] = True
        restored = frames(window, study, args.output / "restored.png")
        report["states"]["restored"] = restored
        for name in ("Scene", "Inspector", "Viewport", "Timeline"):
            assert restored[name]["dock"] == default[name]["dock"]
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(restored[name]["rect"], default[name]["rect"], strict=True)
                )
                < 2
            )
        report["checks"] = (
            "Native scene, full-height sides, center timeline, floating, redocking, tab grouping and fresh-context restore passed."
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(report["checks"])
    finally:
        if study:
            study.close()
        window.close()


if __name__ == "__main__":
    main()
