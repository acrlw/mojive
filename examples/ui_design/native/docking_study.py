"""Instrument reference rendered with native ImGui controls and Mojive 3D backends."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.ui.viewcube import BACKDROP_RADIUS_PT, ViewCube, widget_center
from mojive.ui.window import WindowConfig

from .manipulation import Manipulation
from .reference_menus import Menus
from .scene_preview import Preview, rgba
from .widgets import AMBER, DIM, GLYPHS, INPUT, LINE, MUTED, PANEL, SAGE, TEXT, UI, color

ROOT = Path(__file__).parent
OUTPUT = Path("output/ui_design/native")
ROOT_ID = 0x1A570006
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
)
ICONS = ("scene", "pose", "control", "assets", "camera", "chart", "layers", "terminal", "chart")


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
        self.ui = UI(window, portable_fonts=portable_fonts)
        imgui.get_style().frame_padding = (8 * self.s, 11 * self.s)
        self.preview = Preview(window, backend)
        self.cube = ViewCube()
        self.manipulation = Manipulation()
        self.open = dict.fromkeys(PANELS, False)
        self.open.update(Scene=True, Control=True, Assets=True, Inspector=True, Timeline=False)
        self.needs_layout = restore is None
        self.active = "Scene"
        self.pending = None
        self.float_next = False
        self.focus_next = None
        self.saved = restore
        self.show_viewport_tab = False
        self.time = 4.0
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self.menus = Menus(self, tuple(zip(PANELS, ICONS, strict=True)))
        self.snap = False
        self.space = "World"
        self.workspace = "Edit"
        self.query = ""
        self.filter = "All"
        self.inspector_tab = "Properties"
        self.position = [0.9, 0, 0.9]
        self.history = []
        self.redo = []
        self.events = ["Joint study opened. Local authoring is ready."]
        self.pinned = None
        self.rects = {}
        self.expanded = {"TRANSFORM": True, "DISPLAY": True, "IDENTITY": False}
        self.widths = {}
        self._last = time.monotonic()
        self.material = 0
        self.capture = None
        if restore:
            imgui.load_ini_settings_from_memory(restore)

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

    def change(self, before):
        if before != self.preview.entities:
            self.history.append(before)
            self.redo.clear()

    def undo(self):
        if self.history:
            self.redo.append(copy.deepcopy(self.preview.entities))
            self.preview.entities[:] = self.history.pop()

    def chrome(self):
        u = self.ui
        s = self.s
        if self.sidebar("Application", imgui.Dir.up, 36):
            p = imgui.get_window_pos()
            w = imgui.get_window_width()
            x = p.x
            y = p.y
            imgui.push_clip_rect((x, y), (x + w, y + 36 * s), False)
            u.rect(x + 12 * s, y + 8 * s, 20 * s, 20 * s, SAGE, 5)
            self.menus.bar()
            if w > 950 * s:
                tx = x + w * 0.49
                u.icon("file", tx, y + 11 * s, 14, DIM)
                u.text(tx + 22 * s, y + 12 * s, "Joint study", MUTED, 11)
                u.text(tx + 90 * s, y + 13 * s, "· Scene preview", DIM, 11)
            for n, name in enumerate(["Edit", "Author", "Review"]):
                if u.button(
                    "workspace-" + name,
                    x + w - (228 - n * 55) * s,
                    y + 5 * s,
                    53,
                    26,
                    label=name,
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
                tooltip="Display settings",
                active=imgui.is_popup_open("Display settings"),
            )
            self.menus.popup("Display settings", u.hits["preferences"], settings_clicked)
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
            u.text(p.x + 23 * s, y + 8 * s, "Playing" if self.playing else "Paused", MUTED, 11)
            u.text(
                p.x + 82 * s, y + 8 * s, "W E R   Tools       B   Frame       T   Timeline", DIM, 11
            )
            if w > 680 * s:
                u.text(
                    p.x + w - 249 * s,
                    y + 8 * s,
                    "8 entities · Z up        Instrument / 08",
                    DIM,
                    11,
                )
        self.end_sidebar()
        if self.sidebar("Navigation", imgui.Dir.left, 44):
            p = imgui.get_window_pos()
            h = imgui.get_window_height()
            compact = h < 400 * s
            for n, (name, icon) in enumerate(zip(PANELS, ICONS, strict=True)):
                yy = (
                    p.y + (8 + n * 42) * s - imgui.get_scroll_y()
                    if compact or n < 7
                    else p.y + h - (2 - (n - 7)) * 42 * s
                )
                if u.button(
                    "rail-" + name,
                    p.x + 4 * s,
                    yy,
                    36,
                    36,
                    icon=icon,
                    active=self.active == name and self.open[name],
                    tooltip=name,
                ):
                    self.activate(name)
            if compact:
                imgui.set_cursor_screen_pos((p.x, p.y + 390 * s - imgui.get_scroll_y()))
                imgui.dummy((1, 1))
        self.end_sidebar()

    def activate(self, name):
        if not self.open[name]:
            self.open[name] = True
            self.new_panel = name
        self.focus_next = name

    def build(self):
        ii = imgui.internal
        ii.dock_builder_remove_node(ROOT_ID)
        ii.dock_builder_add_node(ROOT_ID, ii.DockNodeFlagsPrivate_.dock_space)
        work = imgui.get_current_context().viewports[0].get_build_work_rect()
        size = work.get_size()
        ii.dock_builder_set_node_size(ROOT_ID, size)
        _, self.left, rest = ii.dock_builder_split_node_py(
            ROOT_ID, imgui.Dir.left, 272 * self.s / size.x
        )
        _, self.right, self.center = ii.dock_builder_split_node_py(
            rest, imgui.Dir.right, 320 * self.s / (size.x - 272 * self.s)
        )
        if self.open["Timeline"]:
            _, self.bottom, self.center = ii.dock_builder_split_node_py(
                self.center, imgui.Dir.down, 196 * self.s / size.y
            )
            ii.dock_builder_dock_window("Timeline", self.bottom)
        for name in PANELS:
            ii.dock_builder_dock_window(name, self.left)
        ii.dock_builder_dock_window("Inspector", self.right)
        ii.dock_builder_dock_window("Viewport", self.center)
        ii.dock_builder_finish(ROOT_ID)
        self.needs_layout = False

    def toggle_timeline(self):
        if self.open["Timeline"]:
            self.open["Timeline"] = False
        else:
            node = imgui.internal.find_window_by_name("Viewport").dock_node
            if node is not None:
                _, self.bottom, self.center = imgui.internal.dock_builder_split_node_py(
                    node.id_, imgui.Dir.down, min(0.45, 196 * self.s / node.size.y)
                )
                imgui.internal.dock_builder_dock_window("Timeline", self.bottom)
                imgui.internal.dock_builder_finish(ROOT_ID)
            self.open["Timeline"] = True

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
                root = ii.dock_builder_get_node(ROOT_ID)
                current = ii.find_window_by_name("Inspector").dock_node
                if (
                    current
                    and abs(current.pos.x + current.size.x - root.pos.x - root.size.x) < 2
                    and abs(current.size.y - root.size.y) < 2
                ):
                    target = current.id_
                else:
                    _, target, _ = ii.dock_builder_split_node_py(
                        ROOT_ID, imgui.Dir.right, 320 * self.s / root.size.x
                    )
            ii.dock_builder_dock_window("Inspector", target)
            ii.dock_builder_finish(ROOT_ID)
            self.focus_next = "Inspector"
        elif action == "save":
            self.saved = imgui.save_ini_settings_to_memory()
        elif action == "restore" and self.saved:
            imgui.load_ini_settings_from_memory(self.saved)
        else:
            raise ValueError(action)

    def panel(self, name):
        if not self.open[name]:
            return
        if getattr(self, "new_panel", None) == name:
            imgui.set_next_window_dock_id(self.left, imgui.Cond_.first_use_ever)
            self.new_panel = None
        if name == "Inspector" and self.float_next:
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            imgui.set_next_window_pos((600 * self.s, 180 * self.s), imgui.Cond_.always)
            imgui.set_next_window_size((320 * self.s, 570 * self.s), imgui.Cond_.always)
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
        visible, self.open[name] = imgui.begin(name, True)
        if visible:
            if name in PANELS:
                self.active = name
            p = imgui.get_cursor_screen_pos()
            size = imgui.get_content_region_avail()
            x = p.x
            y = p.y
            w = size.x
            s = self.s
            self.widths[name] = w
            if name == "Inspector":
                self.inspector(x, y, w, size.y)
            elif name == "Scene":
                self.scene_panel(x, y, w, size.y)
            elif name == "Control":
                self.control_panel(x, y, w)
            elif name == "Assets":
                self.assets_panel(x, y, w)
            elif name == "Timeline":
                self.timeline(x, y, w, size.y)
            elif name == "Pose":
                self.pose_panel(x, y, w)
            elif name == "Cameras":
                self.camera_panel(x, y, w)
            elif name == "Layers":
                self.layers_panel(x, y, w)
            elif name == "Sensors":
                self.ui.text(x, y + 8 * s, "JOINT SIGNALS", DIM, 10, bold=True)
                for n, (key, val) in enumerate(self.preview.document["pose"].items()):
                    self.ui.text(x, y + (44 + n * 34) * s, key, MUTED)
                    self.ui.text(
                        x + w - 70 * s, y + (44 + n * 34) * s, f"{val:.2f}", TEXT, 11, mono=True
                    )
                imgui.set_cursor_screen_pos((x, y + 180 * s))
                imgui.text_wrapped("Live joint values from the local pose preview.")
            elif name == "Output":
                for n, event in enumerate(self.events):
                    self.ui.text(x, y + n * 35 * s, event, MUTED, 11)
                imgui.set_cursor_screen_pos((x, y + len(self.events) * 35 * s))
                imgui.dummy((1, 1))
            else:
                self.ui.text(x, y + 8 * s, "RENDERER", DIM, 10, bold=True)
                for n, (label, val) in enumerate(
                    [
                        ("Draw calls", self.preview.backend.stats.draw_calls),
                        ("Triangles", self.preview.backend.stats.triangles),
                        ("Scene entities", len(self.preview.entities)),
                    ]
                ):
                    self.ui.text(x, y + (40 + n * 34) * s, label, MUTED, 11)
                    self.ui.text(
                        x + w - 65 * s, y + (40 + n * 34) * s, str(val), TEXT, 11, mono=True
                    )
        imgui.end()

    def scene_panel(self, x, y, w, h):
        u = self.ui
        s = self.s
        u.rect(x, y, w, 28 * s, INPUT)
        u.icon("search", x + 8 * s, y + 7 * s, 14, DIM)
        imgui.set_cursor_screen_pos((x + 29 * s, y))
        imgui.set_next_item_width(w - 29 * s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (2 * s, 8 * s))
        _, self.query = imgui.input_text_with_hint("##search", "Search scene", self.query)
        imgui.pop_style_var()
        for n, kind in enumerate(["All", "Link", "Light", "Camera"]):
            if u.button(
                "filter-" + kind,
                x + n * 51 * s,
                y + 37 * s,
                49,
                24,
                label=kind,
                active=self.filter == kind,
            ):
                self.filter = kind
        u.line(x - 12 * s, y + 70 * s, w + 24 * s)
        u.icon("down", x + 3 * s, y + 86 * s, 12, DIM)
        u.icon("world", x + 23 * s, y + 84 * s, 16, DIM)
        u.text(x + 46 * s, y + 87 * s, "world", MUTED, 11)
        u.text(x + w - 14 * s, y + 87 * s, "8", DIM, 11)
        row = 0
        for e in self.preview.entities:
            if self.query.lower() not in e["name"].lower() or (
                self.filter != "All" and e["type"] != self.filter.lower()
            ):
                continue
            yy = y + (110 + row * 28) * s
            row += 1
            selected = e["id"] == self.preview.selected
            if selected:
                u.rect(x, yy, w, 28 * s, (232, 176, 79, 18), 0)
            if u.button(
                "entity-" + e["id"],
                x + 15 * s,
                yy,
                w / s - 44,
                28,
                icon=e["type"],
                label=e["name"],
                text_color=(227, 201, 145, 255) if selected else TEXT,
                icon_color=(227, 201, 145, 255) if selected else (166, 159, 190, 255),
            ):
                self.preview.selected = e["id"]
            if u.button(
                "visible-" + e["id"],
                x + w - 26 * s,
                yy + s,
                24,
                26,
                icon="eyeoff" if e["hidden"] else "eye",
                tooltip="Toggle visibility",
            ):
                e["hidden"] = not e["hidden"]
        foot = max(y + (row * 28 + 150) * s, y + h - 28 * s)
        u.line(x - 12 * s, foot - 6 * s, w + 24 * s)
        u.text(x, foot + 8 * s, "1 selected" if self.preview.selected else "No selection", DIM, 11)
        if u.button("frame-all", x + w - 78 * s, foot, 78, 26, label="Frame all"):
            self.preview.camera.adopt(self.initial_camera())
        imgui.set_cursor_screen_pos((x, foot + 20 * s))
        imgui.dummy((1, 1))

    def control_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.icon("info", x + 3 * s, y + 7 * s, 16, DIM)
        imgui.set_cursor_screen_pos((x + 30 * s, y + 6 * s))
        imgui.push_text_wrap_pos(x + w - imgui.get_window_pos().x)
        imgui.text_colored(
            imgui.ImVec4(*(v / 255 for v in MUTED)),
            "No physics adapter connected. Actuator commands require a live simulation.",
        )
        imgui.pop_text_wrap_pos()
        yy = y + 84 * s
        u.section("ACTUATORS", x, yy, w)
        for n, label in enumerate(["Shoulder motor", "Elbow motor", "Gripper motor"]):
            fy = yy + (42 + n * 38) * s
            u.text(x, fy + 8 * s, label, (*MUTED[:3], 100), 11)
            u.rect(x + w * 0.49, fy, w * 0.51, 28 * s, (38, 42, 47, 255))
            u.text(x + w - 54 * s, fy + 8 * s, "0 N·m", (*MUTED[:3], 100), 11, mono=True)
        u.button(
            "reset-actuators",
            x,
            yy + 159 * s,
            115,
            28,
            icon="reset",
            label="Reset controls",
            filled=True,
            enabled=False,
        )
        yy += 206 * s
        u.section("CAPABILITIES", x, yy, w)
        for n, (label, value) in enumerate(
            [
                ("Scene authoring", "Available"),
                ("Joint pose preview", "Available"),
                ("Physics write-back", "Disconnected"),
            ]
        ):
            fy = yy + (43 + n * 36) * s
            u.text(x, fy, label, MUTED, 11)
            u.text(x + w - (68 if n < 2 else 86) * s, fy, value, SAGE if n < 2 else DIM, 11)
        if u.button(
            "open-pose",
            x,
            yy + 154 * s,
            w / s,
            28,
            icon="pose",
            label="Open pose editor",
            filled=True,
        ):
            self.activate("Pose")
        imgui.set_cursor_screen_pos((x, yy + 196 * s))
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
        u.text(x + 30 * s, y + 6 * s, e["name"], TEXT, 15, bold=True)
        u.text(x + 30 * s, y + 30 * s, "world  ›  " + e["name"], DIM, 11)
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
        tw = w / 4
        for n, t in enumerate(tabs):
            tx = x + n * tw
            if u.button("tab-" + t, tx, y + 53 * s, tw / s, 32, label=t):
                self.inspector_tab = t
            if self.inspector_tab == t:
                u.draw.add_line(
                    (tx + 5 * s, y + 85 * s), (tx + tw - 5 * s, y + 85 * s), color(SAGE), 2 * s
                )
        body_y = y + 86 * s
        imgui.set_cursor_screen_pos((x, body_y))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Inspector content", (w, max(s, h - 127 * s)))
        imgui.pop_style_var()
        p = imgui.get_cursor_screen_pos()
        bx, yy = p.x, p.y
        bw = imgui.get_content_region_avail().x
        if self.inspector_tab == "Properties":
            if u.section("TRANSFORM", bx, yy, bw, self.expanded["TRANSFORM"]):
                self.expanded["TRANSFORM"] = not self.expanded["TRANSFORM"]
            if self.expanded["TRANSFORM"]:
                u.text(bx, yy + 44 * s, "World coordinates", DIM, 11)
                u.text(bx + bw - 26 * s, yy + 44 * s, "Z up", DIM, 11)
                row_h = 56 if bw < 272 * s else 34
                before = copy.deepcopy(self.preview.entities)
                imgui.begin_disabled(self.playing)
                for n, (key, label, unit) in enumerate(
                    [
                        ("position", "Position", "m"),
                        ("rotation", "Rotation", "°"),
                        ("scale", "Scale", ""),
                    ]
                ):
                    changed, val = u.vector(
                        "##" + e["id"] + key, bx, yy + (61 + n * row_h) * s, bw, label, e[key], unit
                    )
                    if (
                        changed
                        and all(np.isfinite(val))
                        and (key != "scale" or (min(val) >= 0.01 and max(val) <= 100))
                    ):
                        e[key] = val
                imgui.end_disabled()
                self.change(before)
                restore_y = yy + (61 + 2 * row_h + (48 if row_h == 56 else 26) + 9) * s
                if u.button(
                    "restore-transform",
                    bx,
                    restore_y,
                    150,
                    28,
                    icon="reset",
                    label="Restore transform",
                    enabled=not self.playing,
                ):
                    original = next(
                        v
                        for v in json.loads((ROOT / "document.json").read_text())["entities"]
                        if v["id"] == e["id"]
                    )
                    before = copy.deepcopy(self.preview.entities)
                    for key in ("position", "rotation", "scale"):
                        e[key] = original[key]
                    self.change(before)
                yy = restore_y + 42 * s
            else:
                yy += 40 * s
            if u.section("DISPLAY", bx, yy, bw, self.expanded["DISPLAY"]):
                self.expanded["DISPLAY"] = not self.expanded["DISPLAY"]
            if self.expanded["DISPLAY"]:
                u.text(bx, yy + 45 * s, "Visible", MUTED, 11)
                e["hidden"] = not u.switch(
                    "entity-visible", bx + bw - 28 * s, yy + 39 * s, not e["hidden"]
                )
                u.text(bx, yy + 79 * s, "Transform handles", MUTED, 11)
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
                u.text(bx, yy, "Object ID", DIM, 11)
                u.text(bx + 83 * s, yy, e["id"], MUTED, 11)
                u.text(bx, yy + 30 * s, "Type", DIM, 11)
                u.text(bx + 83 * s, yy + 30 * s, e["type"], MUTED, 11)
                yy += 64 * s
        elif self.inspector_tab == "Physics":
            u.icon("info", bx, yy + 14 * s, 18, DIM)
            imgui.set_cursor_screen_pos((bx + 28 * s, yy + 13 * s))
            imgui.text_wrapped("Connect a physics adapter to inspect mass, inertia and velocity.")
            yy += 130 * s
        elif self.inspector_tab == "Material":
            u.section("SURFACE", bx, yy, bw)
            self.menus.color_control(e, bx, yy + 48 * s, bw)
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
        u.text(x + w - 113 * s, footer + 9 * s, "Local scene preview", DIM, 11)
        imgui.set_cursor_screen_pos((x, footer + 20 * s))
        imgui.dummy((1, 1))

    def assets_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 5 * s, "MATERIAL LIBRARY", DIM, 10, bold=True)
        palette = [
            ("Sage ceramic", "#9cbf8d"),
            ("Amber satin", "#e3bc67"),
            ("Slate matte", "#829cb7"),
            ("Coral polymer", "#d58979"),
        ]
        for n, (name, hexcolor) in enumerate(palette):
            yy = y + (32 + n * 72) * s
            if u.button("material-" + name, x, yy, w / s, 62, active=self.material == n):
                self.material = n
            u.draw.add_circle_filled(
                (x + 28 * s, yy + 31 * s),
                18 * s,
                color(tuple(int(c * 255) for c in rgba(hexcolor))),
            )
            u.text(x + 59 * s, yy + 20 * s, name, TEXT, 12)
            u.text(x + 59 * s, yy + 39 * s, "Surface material", DIM, 11)
        if u.button(
            "apply-material",
            x,
            y + 337 * s,
            w / s,
            28,
            icon="check",
            label="Apply material",
            filled=True,
            enabled=self.preview.entity is not None,
        ):
            before = copy.deepcopy(self.preview.entities)
            self.preview.entity["color"] = palette[self.material][1]
            self.change(before)
        imgui.set_cursor_screen_pos((x, y + 389 * s))
        imgui.dummy((1, 1))

    def pose_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 8 * s, "JOINT POSE", DIM, 10, bold=True)
        for n, (name, label, lo, hi) in enumerate(
            [
                ("hinge", "Hinge angle", -120, 120),
                ("ball", "Ball angle", -90, 90),
                ("slide", "Slide position", -0.6, 0.6),
            ]
        ):
            yy = y + (42 + n * 76) * s
            u.text(x, yy, label, MUTED, 11)
            imgui.set_cursor_screen_pos((x, yy + 23 * s))
            imgui.set_next_item_width(w)
            imgui.begin_disabled(self.playing)
            imgui.push_style_var(imgui.StyleVar_.frame_padding, (6 * s, 5 * s))
            _, self.preview.document["pose"][name] = imgui.slider_float(
                "##pose-" + name,
                self.preview.document["pose"][name],
                lo,
                hi,
                "%.2f m" if name == "slide" else "%.1f°",
            )
            imgui.pop_style_var()
            imgui.end_disabled()
        imgui.set_cursor_screen_pos((x, y + 298 * s))
        imgui.dummy((1, 1))

    def camera_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 8 * s, "VIEW DIRECTION", DIM, 10, bold=True)
        for n, name in enumerate(["Front", "Back", "Left", "Right", "Top", "Bottom"]):
            if u.button(
                "camera-" + name,
                x + (n % 2) * (w / 2 + 3 * s),
                y + (42 + n // 2 * 36) * s,
                (w / 2 - 3 * s) / s,
                28,
                icon="camera",
                label=name,
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
            self.preview.camera.adopt(self.initial_camera())
        imgui.set_cursor_screen_pos((x, y + 270 * s))
        imgui.dummy((1, 1))

    def layers_panel(self, x, y, w):
        u = self.ui
        s = self.s
        u.text(x, y + 8 * s, "VIEWPORT LAYERS", DIM, 10, bold=True)
        for n, (attr, label) in enumerate(
            [("gizmos", "Transform handles"), ("shadows", "Cast shadows")]
        ):
            yy = y + (46 + n * 38) * s
            u.text(x, yy + 5 * s, label, MUTED, 11)
            setattr(
                self.preview,
                attr,
                u.switch("layer-" + attr, x + w - 29 * s, yy, getattr(self.preview, attr)),
            )
        u.section("SCENE OBJECTS", x, y + 138 * s, w)
        for n, e in enumerate(self.preview.entities[:6]):
            yy = y + (176 + n * 34) * s
            u.text(x, yy + 4 * s, e["name"], MUTED, 11)
            e["hidden"] = not u.switch("layer-" + e["id"], x + w - 29 * s, yy, not e["hidden"])
        imgui.set_cursor_screen_pos((x, y + 401 * s))
        imgui.dummy((1, 1))

    def timeline(self, x, y, w, h):
        u = self.ui
        s = self.s
        if u.button("collapse-timeline", x, y, 25, 25, icon="down", tooltip="Collapse Timeline"):
            self.toggle_timeline()
        u.text(x + 34 * s, y + 7 * s, "Joint poses", MUTED, 11)
        u.text(x + w - 62 * s, y + 7 * s, "0 – 10 s", DIM, 11)
        u.line(x, y + 34 * s, w)
        for n in range(11):
            xx = x + 16 * s + n * (w - 32 * s) / 10
            u.text(xx, y + 47 * s, str(n), DIM, 11)
            u.draw.add_line((xx, y + 63 * s), (xx, y + 103 * s), color(LINE))
        ty = y + 87 * s
        u.draw.add_line((x + 16 * s, ty), (x + w - 16 * s, ty), color((106, 118, 98, 120)), 2 * s)
        for key in self.preview.document["keys"]:
            xx = x + 16 * s + key["time"] / 10 * (w - 32 * s)
            if u.button(
                "key-" + key["name"],
                xx - 12 * s,
                ty - 13 * s,
                25,
                26,
                icon="key",
                active=abs(self.time - key["time"]) < 0.02,
                tooltip=key["name"],
            ):
                self.time = key["time"]
                self.preview.document["pose"] = dict(key["pose"])
        px = x + 16 * s + self.time / 10 * (w - 32 * s)
        u.draw.add_line((px, y + 62 * s), (px, y + 110 * s), color(SAGE), 1.5 * s)
        imgui.set_cursor_screen_pos((x, y + 126 * s))
        imgui.set_next_item_width(w)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (5 * s, 3 * s))
        changed, self.time = imgui.slider_float("##time", self.time, 0, 10, "%.3f s")
        imgui.pop_style_var()
        if changed:
            self.sample_pose()

    @staticmethod
    def initial_camera():
        from mojive import CameraView

        return CameraView(
            eye=np.array([4.3, -5.9, 4.2]),
            target=np.array([0, 0, 0.55]),
            fov_y=np.radians(42),
            near=0.05,
            far=100,
        )

    def sample_pose(self):
        keys = self.preview.document["keys"]
        left = keys[0]
        right = keys[-1]
        for k in keys:
            if k["time"] <= self.time:
                left = k
            if k["time"] >= self.time:
                right = k
                break
        t = (
            0
            if left["time"] == right["time"]
            else (self.time - left["time"]) / (right["time"] - left["time"])
        )
        self.preview.document["pose"] = {
            k: left["pose"][k] * (1 - t) + right["pose"][k] * t for k in left["pose"]
        }

    def viewport(self):
        u = self.ui
        s = self.s
        wc = imgui.WindowClass()
        wc.dock_node_flags_override_set = (
            0 if self.show_viewport_tab else imgui.DockNodeFlags_.auto_hide_tab_bar
        )
        imgui.set_next_window_class(wc)
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin(
            "Viewport",
            None,
            imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
        )
        p = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        x, y = p.x, p.y
        w, h = avail.x, avail.y
        if not self.open["Timeline"]:
            h = max(20 * s, h - 38 * s)
        self.rects["viewport_content"] = [x, y, w, h]
        fb = imgui.get_io().display_framebuffer_scale.x
        self.preview.space = self.space
        image = self.preview.render(w * fb, h * fb, s * fb)
        if image:
            imgui.image(
                self.window.viewport_texture_ref(image),
                (w, h),
                (0, 1) if image.flip_y else (0, 0),
                (1, 0) if image.flip_y else (1, 1),
            )
        imgui.push_clip_rect((x, y), (x + w, y + h), True)
        overlay = []

        def shell(rx, ry, rw, rh):
            u.rect(rx, ry, rw * s, rh * s, (22, 25, 28, 230), 8)
            overlay.append((rx, ry, rw * s, rh * s))

        camera_width = 119 if w < 300 * s else 150
        shell(x + 10 * s, y + 10 * s, camera_width, 32)
        camera_clicked = u.button(
            "projection-button",
            x + 13 * s,
            y + 13 * s,
            80,
            26,
            icon="camera",
            label="Ortho" if self.preview.camera.orthographic else "Persp",
            active=imgui.is_popup_open("Camera"),
        )
        u.icon("down", x + 78 * s, y + 21 * s, 11, DIM)
        self.menus.popup("Camera", u.hits["projection-button"], camera_clicked)
        shading_clicked = u.button(
            "shading",
            x + 101 * s,
            y + 13 * s,
            26,
            26,
            icon="shading",
            tooltip="Shading",
            active=imgui.is_popup_open("Shading"),
        )
        self.menus.popup("Shading", u.hits["shading"], shading_clicked)
        if camera_width == 150 and u.button(
            "viewport-layers",
            x + 130 * s,
            y + 13 * s,
            26,
            26,
            icon="layers",
            tooltip="Viewport layers",
        ):
            self.activate("Layers")
        if h >= 220 * s:
            self.transport(x, y, w, shell)
        horizontal = h < 440 * s
        step = 32 if w < 280 * s else 40
        tools_y = (
            y
            + (h / s - (78 if h >= 210 * s else 46) if horizontal else max(148, (h / s - 248) / 2))
            * s
        )
        shell(x + 10 * s, tools_y, step * 6 + 8 if horizontal else 44, 40 if horizontal else 248)

        def tool_pos(n):
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
                tooltip=name.title() + " · " + shortcut,
            ):
                self.preview.tool = name
            if not horizontal:
                u.text(xx + 26 * s, yy + 24 * s, shortcut, DIM, 10)
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
        cursor = imgui.get_io().mouse_pos
        cube_scale = s * (0.65 if w < 340 * s or h < 220 * s else 0.94)
        self.cube.update(self.preview.camera.view(), (x, y, w, h), (cursor.x, cursor.y), cube_scale)
        self.cube.draw(self.window.painter(), cube_scale)
        center = widget_center((x, y, w, h), cube_scale)
        radius = BACKDROP_RADIUS_PT * cube_scale
        overlay.append((center[0] - radius, center[1] - radius, radius * 2, radius * 2))
        if self.cube.hovered and imgui.is_window_hovered() and imgui.is_mouse_clicked(0):
            self.cube.click(self.preview.camera, self.cube.hovered, self.preview.backend)
        if h >= 210 * s:
            footer_w = min(w - 20 * s, 210 * s)
            u.rect(x + 10 * s, y + h - 31 * s, footer_w, 22 * s, (22, 25, 28, 230), 4.8)
            u.icon("link", x + 17 * s, y + h - 27 * s, 13, AMBER)
            u.text(
                x + 38 * s,
                y + h - 25 * s,
                "world  ›  " + (self.preview.selected or "No selection"),
                MUTED,
                11,
            )
            overlay.append((x + 10 * s, y + h - 31 * s, footer_w, 22 * s))
        self.rects["overlays"] = overlay
        hover = (
            x <= cursor.x < x + w
            and y <= cursor.y < y + h
            and not any(a <= cursor.x < a + c and b <= cursor.y < b + d for a, b, c, d in overlay)
        )
        claimed = self.manipulation.update(
            self, (x, y, w, h), (cursor.x, cursor.y), hover and imgui.is_window_hovered()
        )
        if (
            hover
            and not claimed
            and imgui.is_window_hovered()
            and not imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
        ):
            io = imgui.get_io()
            if io.mouse_wheel:
                self.preview.camera.dolly(io.mouse_wheel)
            if imgui.is_mouse_dragging(0):
                self.preview.camera.orbit(io.mouse_delta.x, io.mouse_delta.y)
            if imgui.is_mouse_dragging(2):
                self.preview.camera.pan(io.mouse_delta.x, io.mouse_delta.y, h)
            if (
                imgui.is_mouse_released(0)
                and imgui.get_mouse_drag_delta(0).x ** 2 + imgui.get_mouse_drag_delta(0).y ** 2 < 9
            ):
                self.preview.pick((cursor.x - x) * fb, (h - (cursor.y - y)) * fb)
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
            if w > 440 * s:
                u.text(x + 149 * s, yy + 14 * s, "Signals      Events", DIM, 11)
                u.text(x + w - 72 * s, yy + 14 * s, "Pose keys", DIM, 11)
        imgui.end()
        imgui.pop_style_var()

    def transport(self, x, y, w, shell):
        u, s = self.ui, self.s
        compact = w < 340 * s
        width = 160 if compact else 282
        tx = x + (w - width * s) / 2
        ty = y + (104 if w < 500 * s else 52 if w < 700 * s else 10) * s
        shell(tx, ty, width, 36)
        if u.button(
            "previous", tx + 4 * s, ty + 4 * s, 26, 28, icon="previous", tooltip="Previous frame"
        ):
            self.time = max(0, self.time - 1 / 60)
            self.sample_pose()
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
            self.playing = not self.playing
        if u.button("next", tx + 69 * s, ty + 4 * s, 26, 28, icon="next", tooltip="Next frame"):
            self.time = min(10, self.time + 1 / 60)
            self.sample_pose()
        if not compact:
            u.text(tx + 114 * s, ty + 12 * s, f"{self.time:.3f} s", TEXT, 11, mono=True)
        if u.button(
            "go-start",
            tx + (99 if compact else 189) * s,
            ty + 4 * s,
            27,
            28,
            icon="reset",
            tooltip="Go to start",
        ):
            self.time = 0
            self.sample_pose()
        if not compact:
            u.button(
                "record",
                tx + 220 * s,
                ty + 4 * s,
                26,
                28,
                icon="record",
                enabled=False,
                tooltip="Simulation adapter required",
            )
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

    def draw(self):
        self.menus.begin_frame()
        now = time.monotonic()
        dt = min(now - self._last, 0.1)
        self._last = now
        if self.playing:
            self.time += dt * self.speed
            if self.time >= 10:
                if self.loop:
                    self.time %= 10
                else:
                    self.time = 10
                    self.playing = False
            self.sample_pose()
        from mojive.render.backend import RenderFlag

        self.preview.backend.set_flag(RenderFlag.SHADOW, self.preview.shadows)
        self.ui.hits.clear()
        self.chrome()
        self.active = None
        imgui.dock_space_over_viewport(
            ROOT_ID, imgui.get_main_viewport(), imgui.DockNodeFlags_.passthru_central_node
        )
        if self.needs_layout:
            self.build()
        if self.pending:
            self.command(self.pending)
            self.pending = None
        self.viewport()
        for name in self.open:
            self.panel(name)
        io = imgui.get_io()
        self.menus.finish_color_edit()
        if not io.want_text_input and not imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id):
            if imgui.is_key_pressed(imgui.Key.space):
                self.playing = not self.playing
            for key, name in [
                (imgui.Key.v, "select"),
                (imgui.Key.w, "move"),
                (imgui.Key.e, "rotate"),
                (imgui.Key.r, "scale"),
            ]:
                if imgui.is_key_pressed(key) and not (io.key_super or io.key_ctrl):
                    self.preview.tool = name
            if imgui.is_key_pressed(imgui.Key.t):
                self.toggle_timeline()
            if imgui.is_key_pressed(imgui.Key.b):
                self.space = "Body" if self.space == "World" else "World"
            if imgui.is_key_pressed(imgui.Key.s) and not (io.key_super or io.key_ctrl):
                self.snap = not self.snap
            if (io.key_super or io.key_ctrl) and imgui.is_key_pressed(imgui.Key.z):
                self.menus.dispatch("redo" if io.key_shift else "undo")
            if (io.key_super or io.key_ctrl) and imgui.is_key_pressed(imgui.Key.s):
                self.menus.dispatch("file:save")
            if (
                (io.key_super or io.key_ctrl)
                and imgui.is_key_pressed(imgui.Key.o)
                and (self.output_directory / "saved-scene.json").exists()
            ):
                self.menus.dispatch("file:load")
            if imgui.is_key_pressed(imgui.Key.f):
                self.menus.dispatch("frame")

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
        window.begin_frame()
        study.draw()
        pixels = window.end_frame(readback=path is not None)
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
        if args.menus:
            from .menu_review import capture_menus

            capture_menus(window, study, args.output / "menus", frames)
            return
        if args.interactive:
            window.show()
            while not window.should_close():
                window.begin_frame()
                study.draw()
                window.end_frame()
                time.sleep(1 / 120)
            return
        edit = frames(window, study, args.output / "edit.png")
        if args.capture_only:
            (args.output / "report.json").write_text(
                json.dumps(
                    {
                        "backend": args.backend,
                        "ui_scale": args.ui_scale,
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
