"""Instrument reference rendered with native ImGui controls and Mojive 3D backends."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive import math3d
from mojive.app.ui.window import create_window
from mojive.ui.controls import search_input
from mojive.ui.viewcube import BACKDROP_RADIUS_PT, widget_center
from mojive.ui.window import WindowConfig

from .localization import reference_text
from .reference_menus import Menus
from .scene_preview import Preview
from .settings_panel import ReferenceSettings
from .widgets import AMBER, DIM, GLYPHS, LINE, MUTED, PANEL, SAGE, TEXT, UI, color

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
    "chart",
    "layers",
    "terminal",
    "chart",
    "settings",
)


class Study:
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
        self.open.update(Scene=True, Control=True, Assets=True, Inspector=True, Timeline=False)
        self.needs_layout = restore is None
        self.timeline_layout_pending = False
        self.active = "Scene"
        self.pending = None
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
        self.pose_time = 4.0
        self.expanded = {"TRANSFORM": True, "DISPLAY": True, "IDENTITY": False}
        self.events = []
        self.settings = ReferenceSettings(self)
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
        self.chrome()
        self.active = None

    def draw(self, *, readback=False):
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
        if self.sidebar("Application", imgui.Dir.up, 36):
            p = imgui.get_window_pos()
            w = imgui.get_window_width()
            x = p.x
            y = p.y
            imgui.push_clip_rect((x, y), (x + w, y + 36 * s), False)
            u.rect(x + 10 * s, y + 6 * s, 24 * s, 24 * s, SAGE, 6)
            u.text(x + 16 * s, y + 10 * s, "M", PANEL, 15, bold=True)
            self.menus.bar()
            if w > 950 * s:
                tx = x + w * 0.49
                u.icon("file", tx, y + 11 * s, 14, DIM)
                u.text(tx + 22 * s, y + 11 * s, "Joint study", MUTED, 13)
            for n, name in enumerate(["Edit", "Author", "Review"]):
                if u.button(
                    "workspace-" + name,
                    x + w - (228 - n * 55) * s,
                    y + 5 * s,
                    53,
                    26,
                    label=name,
                    align="center",
                    active=self.workspace == name,
                ):
                    self.preset(name)
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
        if self.sidebar("Status", imgui.Dir.down, 26):
            p = imgui.get_window_pos()
            w = imgui.get_window_width()
            y = p.y
            u.draw.add_circle_filled(
                (p.x + 13 * s, y + 13 * s), 2 * s, color(SAGE if self.playing else DIM)
            )
            u.text(p.x + 23 * s, y + 6 * s, "Playing" if self.playing else "Paused", MUTED, 13)
            if w > 680 * s:
                u.text(
                    p.x + w - 155 * s,
                    y + 6 * s,
                    f"{len(self.preview.session.nodes)} nodes · Z up",
                    DIM,
                    13,
                )
        self.end_sidebar()
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
                    active=self.active == name and self.open[name],
                    tooltip=name,
                ):
                    self.activate(name)
            if compact:
                imgui.set_cursor_screen_pos((p.x, p.y + 440 * s - imgui.get_scroll_y()))
                imgui.dummy((1, 1))
        self.end_sidebar()

    def activate(self, name):
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
        left_width = min(272 * self.s, size.x * 0.3)
        right_width = min(320 * self.s, size.x * 0.28)
        _, self.left, rest = ii.dock_builder_split_node_py(
            self.window.dockspace_id, imgui.Dir.left, left_width / size.x
        )
        _, self.right, self.center = ii.dock_builder_split_node_py(
            rest, imgui.Dir.right, right_width / (size.x - left_width)
        )
        if self.open["Timeline"]:
            _, self.bottom, self.center = ii.dock_builder_split_node_py(
                self.center, imgui.Dir.down, min(0.45, 196 * self.s / size.y)
            )
            ii.dock_builder_dock_window("Timeline", self.bottom)
        for name in PANELS:
            ii.dock_builder_dock_window(name, self.left)
        ii.dock_builder_dock_window("Inspector", self.right)
        ii.dock_builder_dock_window("Viewport", self.center)
        ii.dock_builder_finish(self.window.dockspace_id)
        self.needs_layout = False
        self.timeline_layout_pending = False

    def toggle_timeline(self):
        self.open["Timeline"] = not self.open["Timeline"]
        self.timeline_layout_pending = True

    def apply_timeline_layout(self):
        if not self.timeline_layout_pending:
            return
        self.timeline_layout_pending = False
        if self.open["Timeline"]:
            timeline = imgui.internal.find_window_by_name("Timeline")
            if timeline is not None and imgui.internal.dock_builder_get_node(timeline.dock_id):
                return
            window = imgui.internal.find_window_by_name("Viewport")
            node = window.dock_node if window is not None else None
            if node is not None:
                _, self.bottom, self.center = imgui.internal.dock_builder_split_node_py(
                    node.id_, imgui.Dir.down, min(0.45, 196 * self.s / node.size.y)
                )
                imgui.internal.dock_builder_dock_window("Timeline", self.bottom)
                imgui.internal.dock_builder_finish(self.window.dockspace_id)

    def preset(self, name):
        self.workspace = name
        self.open["Inspector"] = True
        if name == "Review":
            for n in PANELS:
                self.open[n] = False
            if not self.open["Timeline"]:
                self.toggle_timeline()
        else:
            if self.open["Timeline"]:
                self.toggle_timeline()
            self.activate("Assets" if name == "Author" else "Scene")

    def command(self, action):
        ii = imgui.internal
        if action == "reset":
            for n in PANELS:
                self.open[n] = n in ("Scene", "Control", "Assets")
            self.open.update(Inspector=True, Timeline=False)
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
            imgui.load_ini_settings_from_memory(self.saved)
        else:
            raise ValueError(action)

    def panel(self, name, ctx):
        panel = self.preview.panels.get(name)
        if not self.open[name]:
            panel.finish_frame(ctx)
            return
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
            else:
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
        if self.focus_next == name:
            imgui.set_next_window_focus()
            self.focus_next = None
        wc = imgui.WindowClass()
        wc.dock_node_flags_override_set = (
            imgui.internal.DockNodeFlagsPrivate_.no_window_menu_button
            | imgui.internal.DockNodeFlagsPrivate_.no_close_button
        )
        imgui.set_next_window_class(wc)
        title = ctx.tr(name) + "###" + name if ctx.tr(name) != name else name
        visible, self.open[name] = imgui.begin(title, True)
        if visible:
            if name in PANELS:
                self.active = name
            self.widths[name] = imgui.get_content_region_avail().x
            self._panel_context = ctx
            self.ui.value_context = ctx
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
            elif name == "Sensors":
                self.ui.text(x, y + 8 * self.s, "JOINT VALUES", MUTED, 13, bold=True)
                for n, (label, value) in enumerate(self.preview.document["pose"].items()):
                    self.ui.text(x, y + (44 + n * 34) * self.s, label, MUTED, 13)
                    self.ui.text(
                        x + w - 70 * self.s,
                        y + (44 + n * 34) * self.s,
                        f"{value:.2f}",
                        TEXT,
                        13,
                        mono=True,
                    )
            elif name == "Output":
                for entry in self.preview.output.entries():
                    imgui.text_wrapped(entry.text)
            elif name == "Statistics":
                for n, (label, value) in enumerate(
                    (
                        ("Draw calls", self.preview.backend.stats.draw_calls),
                        ("Triangles", self.preview.backend.stats.triangles),
                        ("Scene entities", len(self.preview.entities)),
                    )
                ):
                    self.ui.text(x, y + (40 + n * 34) * self.s, label, MUTED, 13)
                    self.ui.text(
                        x + w - 70 * self.s,
                        y + (40 + n * 34) * self.s,
                        str(value),
                        TEXT,
                        13,
                        mono=True,
                    )
            self.preview.apply_document_edits()
        panel.finish_frame(ctx)
        panel.open = self.open[name]
        imgui.end()

    def scene_panel(self, x, y, w, h):
        u = self.ui
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(w)
        _, self.query = search_input("##search", self.query, hint=u.translate("Search scene"))
        kinds = ("All", "Link", "Geom", "Light", "Camera")
        compact_filters = w < 240 * s
        filter_icons = {
            "All": "scene",
            "Link": "body",
            "Geom": "geom",
            "Light": "light",
            "Camera": "camera",
        }
        widths = [max(40, u.measure(kind, 14) / s + 12) for kind in kinds]
        spare = (w / s - sum(widths) - 3 * (len(kinds) - 1)) / len(kinds)
        filter_x = x
        for kind, width in zip(kinds, widths, strict=True):
            width += spare
            if u.button(
                "filter-" + kind,
                filter_x,
                y + 38 * s,
                width,
                28,
                label="" if compact_filters else kind,
                icon=filter_icons[kind] if compact_filters else None,
                tooltip=kind if compact_filters else None,
                align="center",
                active=self.filter == kind,
            ):
                self.filter = kind
            filter_x += (width + 3) * s
        u.line(x, y + 76 * s, w)
        u.icon("down", x + 2 * s, y + 90 * s, 12, DIM)
        u.icon("world", x + 22 * s, y + 88 * s, 16, MUTED)
        u.text(x + 46 * s, y + 88 * s, "world", MUTED, 13)
        count = str(len(self.preview.entities))
        u.text(x + w - u.measure(count, 13) - 6 * s, y + 88 * s, count, DIM, 13)
        imgui.set_cursor_screen_pos((x, y + 116 * s))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Scene entities", (w, max(s, h - 156 * s)))
        imgui.pop_style_var()
        origin = imgui.get_cursor_screen_pos()
        list_width = imgui.get_content_region_avail().x
        row = 0
        for e in self.preview.entities:
            if self.filter == "All" and e["type"] == "geom":
                continue
            if self.query.lower() not in e["name"].lower() or (
                self.filter != "All" and e["type"] != self.filter.lower()
            ):
                continue
            yy = origin.y + row * 30 * s
            row += 1
            node = self.preview.session.selected_node
            selected = node is not None and node.node_id == e["node_id"]
            if selected:
                u.rect(origin.x, yy, list_width, 28 * s, (232, 176, 79, 20), 4)
                u.rect(origin.x, yy + 6 * s, 2 * s, 16 * s, AMBER, 1)
            if u.button(
                "entity-" + str(e["node_id"]),
                origin.x + 10 * s,
                yy,
                list_width / s - 40,
                28,
                icon=e["type"],
                label=e["name"],
                text_color=(227, 201, 145, 255) if selected else TEXT,
                icon_color=(227, 201, 145, 255) if selected else (166, 159, 190, 255),
            ):
                self.preview.session.submit(cmd.SelectNode(e["node_id"]))
            if u.button(
                "visible-" + str(e["node_id"]),
                origin.x + list_width - 26 * s,
                yy + s,
                24,
                26,
                icon="eyeoff" if e["hidden"] else "eye",
                tooltip="Toggle visibility",
            ):
                e["hidden"] = not e["hidden"]
        imgui.set_cursor_screen_pos((origin.x, origin.y + row * 30 * s))
        imgui.dummy((1, 1))
        imgui.end_child()
        foot = y + h - 28 * s
        u.line(x, foot - 8 * s, w)
        u.text(x, foot + 7 * s, "1 selected" if self.preview.selected else "No selection", DIM, 13)
        action_width = max(82, u.measure("Frame all", 14) / s + 20)
        if u.button(
            "frame-all",
            x + w - action_width * s,
            foot,
            action_width,
            28,
            label="Frame all",
            align="center",
        ):
            self.preview._frame_scene(animate=True)
        imgui.set_cursor_screen_pos((x, foot + 20 * s))
        imgui.dummy((1, 1))

    def control_panel(self, x, y, w):
        u, s = self.ui, self.s
        session = self.preview.session
        values = session.frame.ctrl
        enabled = session.adapter.caps.write_ctrl and values is not None
        u.icon("control", x, y + 4 * s, 20, SAGE)
        u.text(x + 30 * s, y + 3 * s, "Actuators", TEXT, 17, bold=True)
        row = 0
        for actuator in session.actuators:
            lo, hi = actuator.ctrl_range if actuator.ctrl_limited else (-1.0, 1.0)
            for component in range(actuator.ctrl_count):
                address = actuator.ctrl_address + component
                yy = y + (52 + row * 64) * s
                suffix = f"[{component}]" if actuator.ctrl_count > 1 else ""
                u.text(x, yy, actuator.name + suffix, MUTED, 13)
                current = 0.0 if values is None else float(values[address])
                imgui.begin_disabled(not enabled)
                changed, value = u.slider(
                    "##ctrl-" + str(address), x, yy + 22 * s, w, current, lo, hi, "%.3f"
                )
                imgui.end_disabled()
                if changed and enabled:
                    session.submit(cmd.SetCtrl(address, value))
                row += 1
        yy = y + (68 + row * 64) * s
        if not session.actuators:
            u.text(x, y + 50 * s, "No actuators in this scene", MUTED, 13)
            yy += 22 * s
        if (
            u.button(
                "reset-actuators",
                x,
                yy,
                140,
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
            next((e for e in self.preview.entities if e["id"] == self.pinned), None)
            if self.pinned
            else self.preview.entity
        )
        if not e:
            u.text(x, y + 45 * s, "Select an entity to inspect.", MUTED)
            return
        u.icon(e["type"], x, y + 6 * s, 20, (166, 159, 190, 255))
        imgui.push_clip_rect((x, y), (x + w - 32 * s, y + 30 * s), True)
        u.text(x + 30 * s, y + 5 * s, e["name"], TEXT, 16, bold=True)
        imgui.pop_clip_rect()
        if u.button(
            "pin",
            x + w - 27 * s,
            y,
            26,
            26,
            icon="pin",
            active=bool(self.pinned),
            tooltip="Pin Inspector",
        ):
            self.pinned = None if self.pinned else e["id"]
        tabs = ["Properties", "Physics", "Material", "Joint"]
        tab_icons = ("inspect", "body", "shading", "pose")
        compact_tabs = w < 260 * s
        tw = w / 4
        for n, t in enumerate(tabs):
            tx = x + n * tw
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
                active=self.inspector_tab == t,
            ):
                self.inspector_tab = t
            if self.inspector_tab == t:
                u.draw.add_line(
                    (tx + 8 * s, y + 70 * s), (tx + tw - 8 * s, y + 70 * s), color(SAGE), 2 * s
                )
        body_y = y + 71 * s
        imgui.set_cursor_screen_pos((x, body_y))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Inspector content", (w, max(s, h - 112 * s)))
        imgui.pop_style_var()
        p = imgui.get_cursor_screen_pos()
        bx, yy = p.x, p.y
        bw = imgui.get_content_region_avail().x
        if self.inspector_tab == "Properties":
            if u.section("TRANSFORM", bx, yy, bw, self.expanded["TRANSFORM"]):
                self.expanded["TRANSFORM"] = not self.expanded["TRANSFORM"]
            if self.expanded["TRANSFORM"]:
                row_h = 60 if bw < 272 * s else 36
                for n, (key, label, unit) in enumerate(
                    [
                        ("position", "Position", "m"),
                        ("rotation", "Rotation", "°"),
                        ("scale", "Scale", ""),
                    ]
                ):
                    imgui.begin_disabled(
                        self.playing or not e["scalable" if key == "scale" else "posable"]
                    )
                    changed, val = u.vector(
                        "##" + e["id"] + key, bx, yy + (42 + n * row_h) * s, bw, label, e[key], unit
                    )
                    imgui.end_disabled()
                    if (
                        changed
                        and all(np.isfinite(val))
                        and (key != "scale" or (min(val) >= 0.01 and max(val) <= 100))
                    ):
                        e[key] = val
                self.preview.apply_document_edits()
                restore_y = yy + (42 + 2 * row_h + (52 if row_h == 60 else 28) + 12) * s
                original = next(
                    (
                        v
                        for v in self.preview.initial_entities
                        if (v["node_id"], v["name"], v["type"])
                        == (e["node_id"], e["name"], e["type"])
                    ),
                    None,
                )
                if u.button(
                    "restore-transform",
                    bx,
                    restore_y,
                    150,
                    28,
                    icon="reset",
                    label="Restore transform",
                    enabled=not self.playing and e["posable"] and original is not None,
                    tooltip="Joint-driven transform; use the Joint tab or the viewport handle."
                    if not e["posable"]
                    else "",
                ):
                    for key in ("position", "rotation", "scale"):
                        e[key] = original[key]
                    self.preview.apply_document_edits()
                yy = restore_y + 42 * s
            else:
                yy += 40 * s
            if u.section("DISPLAY", bx, yy, bw, self.expanded["DISPLAY"]):
                self.expanded["DISPLAY"] = not self.expanded["DISPLAY"]
            if self.expanded["DISPLAY"]:
                u.text(bx, yy + 44 * s, "Visible", MUTED, 13)
                e["hidden"] = not u.switch(
                    "entity-visible", bx + bw - 28 * s, yy + 39 * s, not e["hidden"]
                )
                u.text(bx, yy + 78 * s, "Transform handles", MUTED, 13)
                self.preview.gizmos = u.switch(
                    "gizmos", bx + bw - 28 * s, yy + 73 * s, self.preview.gizmos
                )
                yy += 113 * s
            else:
                yy += 40 * s
            if u.section("IDENTITY", bx, yy, bw, self.expanded["IDENTITY"]):
                self.expanded["IDENTITY"] = not self.expanded["IDENTITY"]
            yy += 42 * s
            if self.expanded["IDENTITY"]:
                u.text(bx, yy, "Object ID", MUTED, 13)
                u.text(bx + 83 * s, yy, str(e["object_id"]), TEXT, 13)
                u.text(bx, yy + 30 * s, "Type", MUTED, 13)
                u.text(bx + 83 * s, yy + 30 * s, e["type"], TEXT, 13)
                yy += 64 * s
        elif self.inspector_tab == "Physics":
            u.icon("info", bx, yy + 14 * s, 18, DIM)
            imgui.set_cursor_screen_pos((bx + 28 * s, yy + 13 * s))
            node = self.preview.session.node(e["node_id"])
            properties = (
                self.preview.session.body_properties(node.node_id) if node.body_index >= 0 else None
            )
            imgui.text_wrapped(
                f"Mass: {properties.mass:.3f} kg"
                if properties is not None
                else "No body properties for this entity."
            )
            yy += 130 * s
        elif self.inspector_tab == "Material":
            u.section("SURFACE", bx, yy, bw)
            imgui.begin_disabled(not e["colorable"])
            self.menus.color_control(e, bx, yy + 48 * s, bw)
            imgui.end_disabled()
            yy += 150 * s
        else:
            self.pose_panel(bx, yy, bw)
            yy += 315 * s
        imgui.set_cursor_screen_pos((bx, yy))
        imgui.dummy((1, 1))
        imgui.end_child()
        footer = y + h - 28 * s
        u.line(x - 12 * s, footer - 7 * s, w + 24 * s)
        if u.button(
            "undo-footer", x, footer, 86, 28, icon="undo", label="Undo", enabled=bool(self.history)
        ):
            self.undo()
        imgui.set_cursor_screen_pos((x, footer + 20 * s))
        imgui.dummy((1, 1))

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
            u.text(x + 56 * s, yy + 17 * s, name, TEXT, 14)
            if self.material == n:
                u.icon("check", x + w - 26 * s, yy + 16 * s, 16, SAGE)
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

    def pose_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 8 * s, "JOINT POSE", MUTED, 13, bold=True)
        for n, (name, label, lo, hi) in enumerate(
            [
                ("hinge", "Hinge angle", -60, 60),
                ("ball", "Ball angle", -90, 90),
                ("slide", "Slide position", -0.35, 0.35),
            ]
        ):
            yy = y + (42 + n * 76) * s
            u.text(x, yy, label, MUTED, 13)
            joint_name = {"hinge": "hinge_limited", "ball": "ball", "slide": "slide"}[name]
            available = any(j.name == joint_name for j in self.preview.session.joints)
            imgui.begin_disabled(
                self.playing or not self.preview.session.adapter.caps.write_qpos or not available
            )
            changed, value = u.slider(
                "##pose-" + name,
                x,
                yy + 24 * s,
                w,
                self.preview.document["pose"][name],
                lo,
                hi,
                "%.2f m" if name == "slide" else "%.1f°",
            )
            imgui.end_disabled()
            if changed:
                self.preview.set_pose_value(name, value)
        imgui.set_cursor_screen_pos((x, y + 298 * s))
        imgui.dummy((1, 1))

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
        if u.button(
            "projection",
            x,
            y + 176 * s,
            w / s,
            28,
            label="Orthographic" if self.preview.camera.orthographic else "Perspective",
            filled=True,
        ):
            self.preview.camera.set_orthographic(not self.preview.camera.orthographic)
        if u.button(
            "camera-frame", x, y + 216 * s, w / s, 28, icon="frame", label="Frame all", filled=True
        ):
            self.preview._frame_scene(animate=True)
        imgui.set_cursor_screen_pos((x, y + 270 * s))
        imgui.dummy((1, 1))

    def layers_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 8 * s, "VIEWPORT LAYERS", MUTED, 13, bold=True)
        for n, (attr, label) in enumerate(
            [("gizmos", "Transform handles"), ("shadows", "Cast shadows")]
        ):
            yy = y + (46 + n * 38) * s
            u.text(x, yy + 4 * s, label, MUTED, 13)
            setattr(
                self.preview,
                attr,
                u.switch("layer-" + attr, x + w - 29 * s, yy, getattr(self.preview, attr)),
            )
        u.section("SCENE OBJECTS", x, y + 138 * s, w)
        for n, e in enumerate(self.preview.entities[:6]):
            yy = y + (176 + n * 34) * s
            u.text(x, yy + 4 * s, e["name"], MUTED, 13)
            e["hidden"] = not u.switch("layer-" + e["id"], x + w - 29 * s, yy, not e["hidden"])
        imgui.set_cursor_screen_pos((x, y + 401 * s))
        imgui.dummy((1, 1))

    def timeline(self, x, y, w, h):
        u = self.ui
        s = self.s
        if u.button("collapse-timeline", x, y, 25, 25, icon="down", tooltip="Collapse Timeline"):
            self.toggle_timeline()
        u.text(x + 34 * s, y + 6 * s, "Joint poses", MUTED, 14)
        u.text(x + w - 72 * s, y + 6 * s, "0 – 10 s", MUTED, 13)
        u.line(x, y + 34 * s, w)
        for n in range(11):
            xx = x + 16 * s + n * (w - 32 * s) / 10
            u.text(xx, y + 46 * s, str(n), MUTED, 13)
            u.draw.add_line((xx, y + 63 * s), (xx, y + 103 * s), color(LINE))
        ty = y + 87 * s
        u.draw.add_line((x + 16 * s, ty), (x + w - 16 * s, ty), color((106, 118, 98, 120)), 2 * s)
        editable = (
            not self.playing
            and self.preview.session.adapter.caps.write_qpos
            and any(
                joint.name in ("hinge_limited", "slide", "ball")
                for joint in self.preview.session.joints
            )
        )
        imgui.begin_disabled(not editable)
        for key in self.preview.document["keys"]:
            xx = x + 16 * s + key["time"] / 10 * (w - 32 * s)
            if u.button(
                "key-" + key["name"],
                xx - 12 * s,
                ty - 13 * s,
                25,
                26,
                icon="key",
                active=abs(self.pose_time - key["time"]) < 0.02,
                tooltip=key["name"],
            ):
                self.sample_pose(key["time"])
        px = x + 16 * s + self.pose_time / 10 * (w - 32 * s)
        u.draw.add_line((px, y + 62 * s), (px, y + 110 * s), color(SAGE), 1.5 * s)
        imgui.set_cursor_screen_pos((x, y + 126 * s))
        imgui.set_next_item_width(w)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (5 * s, 3 * s))
        changed, value = imgui.slider_float("##time", self.pose_time, 0, 10, "%.3f s")
        imgui.pop_style_var()
        if changed:
            self.sample_pose(value)
        imgui.end_disabled()

    def sample_pose(self, pose_time):
        session = self.preview.session
        if self.playing or not session.adapter.caps.write_qpos:
            return False
        keys = self.preview.document["keys"]
        left = keys[0]
        right = keys[-1]
        for k in keys:
            if k["time"] <= pose_time:
                left = k
            if k["time"] >= pose_time:
                right = k
                break
        t = (
            0
            if left["time"] == right["time"]
            else (pose_time - left["time"]) / (right["time"] - left["time"])
        )
        pose = {k: left["pose"][k] * (1 - t) + right["pose"][k] * t for k in left["pose"]}

        indices, values = [], []
        for name, value in pose.items():
            joint_name = {"hinge": "hinge_limited", "slide": "slide", "ball": "ball"}[name]
            joint = next((joint for joint in session.joints if joint.name == joint_name), None)
            if joint is None:
                continue
            if name == "ball":
                indices.extend(range(joint.qpos_adr, joint.qpos_adr + 4))
                values.extend(
                    math3d.mat3_to_quat(math3d.euler_xyz_to_mat3((0, np.radians(value), 0)))
                )
            else:
                indices.append(joint.qpos_adr)
                values.append(value if name == "slide" else np.radians(value))
        if not indices:
            return False
        result = session.submit(cmd.SetQposBatch(np.asarray(indices), np.asarray(values)))
        if result.ok:
            self.pose_time = pose_time
        return result.ok

    def viewport(self):
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
        imgui.push_clip_rect((x, y), (x + w, y + h), True)
        app._draw_scene_overlays(self.window.painter())
        overlay = []
        if app.model_edits.active:
            self.pending_model_edits(
                x, y, w, h, overlay, bottom_limit=tools_y if horizontal else None
            )
        if not app.viewport_layers.viewport_ui:
            self.rects["overlays"] = overlay
            self.input_overlays = list(overlay)
            imgui.pop_clip_rect()
            imgui.end()
            return

        def shell(rx, ry, rw, rh):
            u.rect(rx, ry, rw * s, rh * s, (22, 25, 28, 230), 8)
            overlay.append((rx, ry, rw * s, rh * s))

        compact_camera = w < 240 * s
        stacked_camera = w < 160 * s
        camera_width = (
            44 if stacked_camera else 72 if compact_camera else 124 if w < 300 * s else 156
        )
        shell(x + 10 * s, y + 10 * s, camera_width, 68 if stacked_camera else 36)
        camera_clicked = u.button(
            "projection-button",
            x + 13 * s,
            y + 14 * s,
            28 if compact_camera else 84,
            28,
            icon="camera",
            label=""
            if compact_camera
            else "Ortho"
            if self.preview.camera.orthographic
            else "Persp",
            tooltip="Camera" if compact_camera else None,
            active=imgui.is_popup_open("Camera"),
        )
        if not compact_camera:
            u.icon("down", x + 82 * s, y + 23 * s, 11, DIM)
        self.menus.popup("Camera", u.hits["projection-button"], camera_clicked)
        shading_clicked = u.button(
            "shading",
            x + (13 if stacked_camera else 47 if compact_camera else 104) * s,
            y + (46 if stacked_camera else 14) * s,
            28,
            28,
            icon="shading",
            tooltip="Shading",
            active=imgui.is_popup_open("Shading"),
        )
        self.menus.popup("Shading", u.hits["shading"], shading_clicked)
        if camera_width == 156 and u.button(
            "viewport-layers",
            x + 134 * s,
            y + 14 * s,
            28,
            28,
            icon="layers",
            tooltip="Viewport layers",
        ):
            self.activate("Layers")
        if h >= (248 if w < 260 * s else 220) * s:
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
                tools_y + (4 if horizontal else (176, 211)[n - 4] if n >= 4 else 4 + n * 40) * s,
            )

        for n, (name, shortcut) in enumerate(
            [("select", "V"), ("move", "W"), ("rotate", "E"), ("scale", "R")]
        ):
            xx, yy = tool_pos(n)
            button_size = step - 4 if horizontal else 36
            if u.button(
                "tool-" + name,
                xx,
                yy,
                button_size,
                32 if horizontal else 36,
                icon=name,
                icon_size=18,
                active=self.preview.tool == name,
                solid=True,
                tooltip=("Dimensions: select a geometry" if name == "scale" else name.title())
                + " · "
                + shortcut,
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
            tooltip="World / body frame · B",
        ):
            self.space = "Body" if self.space == "World" else "World"
        if u.button(
            "snap",
            *tool_pos(5),
            step - 4 if horizontal else 36,
            32,
            icon="snap",
            active=self.snap,
            tooltip="Snap · S",
        ):
            self.snap = not self.snap
        self.input_overlays = list(overlay)
        cube_scale = self.cube.scale_for((x, y, w, h), s)
        center = widget_center((x, y, w, h), cube_scale)
        radius = BACKDROP_RADIUS_PT * cube_scale
        overlay.append((center[0] - radius, center[1] - radius, radius * 2, radius * 2))
        if (
            h >= 210 * s
            and not self.open["Inspector"]
            and not (app.model_edits.active and h < 300 * s)
        ):
            footer_w = min(w - 20 * s, 248 * s)
            u.rect(x + 10 * s, y + h - 38 * s, footer_w, 28 * s, (22, 25, 28, 230), 4.8)
            u.icon("link", x + 18 * s, y + h - 32 * s, 16, AMBER)
            u.text(
                x + 42 * s,
                y + h - 31 * s,
                self.preview.selected or "No selection",
                MUTED,
                14,
            )
            overlay.append((x + 10 * s, y + h - 38 * s, footer_w, 28 * s))
        self.rects["overlays"] = overlay
        imgui.pop_clip_rect()
        if not self.open["Timeline"]:
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
            if u.button(
                "timeline-tab", x + 38 * s, yy + 5 * s, 95, 28, icon="key", label="Timeline"
            ):
                self.toggle_timeline()
            u.draw.add_line(
                (x + 42 * s, yy + 36 * s), (x + 126 * s, yy + 36 * s), color(SAGE), 2 * s
            )
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
            else y + max(10, h / s - 96) * s
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
        width = 160 if compact else 282
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
            tooltip="Next frame",
            enabled=not self.playing and session.adapter.caps.simulation,
        ):
            session.submit(cmd.Step())
        if not compact:
            u.text(tx + 110 * s, ty + 11 * s, f"{self.time:.3f} s", TEXT, 13, mono=True)
        if u.button(
            "go-start",
            tx + (99 if compact else 189) * s,
            ty + 4 * s,
            27,
            28,
            icon="reset",
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
            active=self.preview.session.state_take_recording,
            tooltip="Record state take",
            enabled=session.adapter.caps.simulation,
        ):
            self.preview._toggle_state_take_recording()
        playback_clicked = u.button(
            "playback-options",
            tx + (130 if compact else 250) * s,
            ty + 4 * s,
            26,
            28,
            icon="down",
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
    for n, (x, y, w, h) in enumerate(rectangles):
        assert x >= vx and y >= vy and x + w <= vx + vw + 0.01 and y + h <= vy + vh + 0.01, (
            "Overlay outside viewport",
            n,
            rectangles[n],
            snapshot["viewport_content"],
        )
        for other in rectangles[:n]:
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
            x, y = (24 + n % 12 * 116) * s, (88 + n // 12 * 96) * s
            ui.text(x, y, name, DIM, 11)
            for size, offset in [(14, 0), (16, 24), (18, 50), (24, 80)]:
                ui.icon(name, x + offset * s, y + (37 - size / 2) * s, size, TEXT)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(path)
    focus = (
        "move",
        "rotate",
        "scale",
        "gear",
        "eye",
        "camera",
        "keyboard",
        "mouse",
        "world",
        "file",
        "play",
        "pause",
    )
    for _ in range(6):
        window.begin_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.begin("Icon geometry", None, imgui.WindowFlags_.no_decoration)
        ui.text(24 * s, 22 * s, "Instrument 08 · icon geometry", TEXT, 17, bold=True)
        ui.text(
            24 * s,
            50 * s,
            "14 / 24 / 56 / 112 px · square slots, original silhouette proportions · guides stay outside the painter",
            MUTED,
            11,
        )
        for n, name in enumerate(focus):
            x, y = (24 + n % 3 * 464) * s, (90 + n // 3 * 195) * s
            ui.text(x, y, name, TEXT, 12)
            for size, offset in ((14, 0), (24, 45), (56, 104), (112, 210)):
                xx, yy = x + offset * s, y + (38 + (112 - size) / 2) * s
                ui.draw.add_rect(
                    (xx, yy), (xx + size * s, yy + size * s), color((75, 83, 90, 160)), 0
                )
                ui.draw.add_circle(
                    (xx + size * s / 2, yy + size * s / 2), size * s / 2, color((100, 91, 66, 100))
                )
                ui.icon(name, xx, yy, size, TEXT)
                ui.text(xx, y + 159 * s, str(size), DIM, 11)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(path.with_name("icon-geometry.png"))
    io = imgui.get_io()
    report = {
        "logical_viewport": list(io.display_size),
        "framebuffer_density": list(io.display_framebuffer_scale),
        "sizes": [14, 16, 18, 24, 56, 112],
        "anchor": [12, 12],
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
                time.sleep(1 / 120)
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
