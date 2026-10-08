"""Native menu semantics with the Instrument spacing and shared icon masters."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from imgui_bundle import imgui

from .widgets import DIM, MUTED, SAGE, TEXT, color

ROOT = Path(__file__).parent
MENU_BG = (39, 45, 53, 255)
DISABLED = (126, 136, 149, 255)
TOP_MENUS = ("File", "Edit", "Add", "View", "Simulate", "Window", "Help")


@dataclass(frozen=True)
class Item:
    label: str
    action: str = ""
    icon: str = ""
    shortcut: str = ""
    enabled: bool = True
    checked: bool = False
    submenu: str = ""
    reason: str = ""


class Menus:
    def __init__(self, study, panels):
        self.study = study
        self.panels = panels
        self.ui = study.ui
        self.s = study.s
        self.requested_path = ()
        self.reset_pending = False
        self.observed = {}
        self.dialog = None
        self.dialog_pending = None
        self.color_before = None
        self.color_entity = None
        self.color_visible = False
        self.shading = "Solid"

    @contextmanager
    def style(self, controls=False):
        s = self.s
        # ImGui snaps each cursor advance. Use an integral row height at fractional UI scales.
        row_gap = (round(30 * s) - 12 * s) / s
        values = [
            (imgui.StyleVar_.window_padding, ((16, 14) if controls else (12, 6 + row_gap / 2))),
            (imgui.StyleVar_.window_min_size, (0, 0)),
            (imgui.StyleVar_.item_spacing, ((8, 8) if controls else (12, row_gap))),
            (imgui.StyleVar_.frame_padding, (8, 6)),
        ]
        for key, value in values:
            imgui.push_style_var(key, tuple(v * s for v in value))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, 8 * s)
        imgui.push_style_var(imgui.StyleVar_.menu_item_rounding, 4.8 * s)
        imgui.push_style_var(imgui.StyleVar_.scrollbar_size, 6 * s)
        colors = [
            (imgui.Col_.popup_bg, MENU_BG),
            (imgui.Col_.border, (76, 86, 99, 150)),
            (imgui.Col_.header, (156, 191, 141, 24)),
            (imgui.Col_.header_hovered, (255, 255, 255, 13)),
            (imgui.Col_.header_active, (255, 255, 255, 18)),
        ]
        for key, value in colors:
            imgui.push_style_color(key, imgui.ImVec4(*(v / 255 for v in value)))
        imgui.push_font(self.ui.body, 12 * s)
        try:
            yield
        finally:
            imgui.pop_font()
            imgui.pop_style_color(len(colors))
            imgui.pop_style_var(len(values) + 3)

    def items(self, name):
        st = self.study
        e = st.preview.entity
        ready = bool(e) and not st.playing
        if name == "File":
            return [
                Item("Save locally", "file:save", "save", "⌘S"),
                Item(
                    "Open saved scene",
                    "file:load",
                    "assets",
                    "⌘O",
                    (self.study.output_directory / "saved-scene.json").exists(),
                    reason="Save a local scene first.",
                ),
                None,
                Item("Export scene JSON", "file:export", "export"),
            ]
        if name == "Edit":
            return [
                Item("Undo", "undo", "undo", "⌘Z", bool(st.history), reason="No changes to undo."),
                Item("Redo", "redo", "redo", "⇧⌘Z", bool(st.redo), reason="No changes to redo."),
                None,
                Item("Restore transform", "transform:reset", "reset", enabled=ready),
                Item(
                    "Visible",
                    "visibility",
                    "eye",
                    enabled=bool(e),
                    checked=bool(e) and not e["hidden"],
                ),
            ]
        if name == "Add":
            return [
                Item(
                    n,
                    icon="sphere" if n == "Sphere" else "geom",
                    enabled=False,
                    reason="Primitive creation is available in the web reference.",
                )
                for n in ("Box", "Sphere", "Cylinder", "Capsule")
            ]
        if name == "View":
            return [
                Item("Frame all", "frame", "frame", "F"),
                None,
                Item("Projection", icon="camera", submenu="Projection"),
                Item("Transform tool", icon=st.preview.tool, submenu="Transform tool"),
                Item("Gizmo orientation", icon="world", submenu="Gizmo orientation"),
                None,
                Item("Transform handles", "gizmos", "move", checked=st.preview.gizmos),
            ]
        if name in ("Camera", "Projection"):
            rows = [
                Item(
                    "Perspective",
                    "projection:perspective",
                    "camera",
                    checked=not st.preview.camera.orthographic,
                ),
                Item(
                    "Orthographic",
                    "projection:orthographic",
                    "body",
                    checked=st.preview.camera.orthographic,
                ),
            ]
            if name == "Camera":
                rows += [
                    None,
                    Item("Frame all", "frame", "frame", "F"),
                    Item("Cameras and saved views", "panel:Cameras", "camera"),
                ]
            return rows
        if name == "Transform tool":
            return [
                Item(n.title(), "tool:" + n, n, key, checked=st.preview.tool == n)
                for n, key in [("select", "V"), ("move", "W"), ("rotate", "E"), ("scale", "R")]
            ]
        if name == "Gizmo orientation":
            return [
                Item(
                    n + " frame",
                    "space:" + n,
                    "world" if n == "World" else "body",
                    checked=st.space == n,
                )
                for n in ("World", "Body")
            ]
        if name == "Simulate":
            return [
                Item(
                    "Pause preview" if st.playing else "Play pose preview",
                    "play",
                    "pause" if st.playing else "play",
                    "Space",
                ),
                Item("Go to start", "start", "reset"),
                Item("Playback speed", icon="play", submenu="Playback speed"),
                None,
                Item(
                    "Record simulation",
                    icon="record",
                    enabled=False,
                    reason="Connect a simulation adapter to record.",
                ),
                Item("Physics adapter", "panel:Control", "control"),
            ]
        if name == "Playback speed":
            return [
                Item(f"{v:g}× speed", "speed:" + str(v), checked=st.speed == v)
                for v in (0.25, 0.5, 1, 2)
            ]
        if name == "Playback":
            return [
                Item("Previous frame", "step:-1", "previous", enabled=not st.playing),
                Item("Next frame", "step:1", "next", enabled=not st.playing),
                None,
                Item("Playback speed", icon="play", submenu="Playback speed"),
                Item("Loop playback", "loop", "loop", checked=st.loop),
            ]
        if name == "Window":
            return [
                Item("Workspace", icon="inspect", submenu="Workspace"),
                Item("Panels", icon="scene", submenu="Panels"),
                Item("Inspector position", icon="inspect", submenu="Inspector position"),
                None,
                Item("Save layout", "layout:save", "save"),
                Item(
                    "Restore saved layout",
                    "layout:restore",
                    "undo",
                    enabled=bool(st.saved),
                    reason="Save a layout first.",
                ),
                Item("Reset layout", "layout:reset", "reset"),
                None,
                Item("Show Viewport tab", "viewport-tab", "inspect", checked=st.show_viewport_tab),
            ]
        if name == "Workspace":
            return [
                Item(n, "workspace:" + n, checked=st.workspace == n)
                for n in ("Edit", "Author", "Review")
            ]
        if name == "Panels":
            return [Item(n, "toggle:" + n, icon, checked=st.open[n]) for n, icon in self.panels] + [
                None,
                Item("Inspector", "toggle:Inspector", "inspect", checked=st.open["Inspector"]),
                Item("Timeline", "timeline", "key", "T", checked=st.open["Timeline"]),
            ]
        if name == "Inspector position":
            return [
                Item("Float", "layout:float", "inspect"),
                Item("Dock on the right", "layout:dock", "inspect"),
                Item("Group with Scene", "layout:group", "scene"),
            ]
        if name == "Shading":
            return [
                Item(n, "shading:" + n, "shading", checked=self.shading == n)
                for n in ("Solid", "Wireframe")
            ]
        if name == "Display settings":
            return [
                Item("Transform handles", "gizmos", "move", checked=st.preview.gizmos),
                Item("Cast shadows", "shadows", "shading", checked=st.preview.shadows),
                None,
                Item("Interaction guide", "dialog:Interaction guide", "help"),
            ]
        if name == "Help":
            return [
                Item("Interaction guide", "dialog:Interaction guide", "help"),
                Item("About this reference", "dialog:About this reference", "info"),
            ]
        raise ValueError(name)

    def limits(self, name):
        rows = self.items(name)
        measured = max(
            (
                self.ui.measure(r.label)
                + (self.ui.measure(r.shortcut) + 16 * self.s if r.shortcut else 0)
                for r in rows
                if r
            ),
            default=0,
        )
        width = max(224 * self.s, measured + 88 * self.s)
        available = imgui.get_main_viewport().size
        return min(width, available.x - 16 * self.s), min(
            12 * self.s + sum(round((30 if r else 10) * self.s) for r in rows),
            available.y - 16 * self.s,
        )

    def constrain(self, name):
        width, _ = self.limits(name)
        imgui.set_next_window_size_constraints(
            (width, 0), (width, imgui.get_main_viewport().size.y - 16 * self.s)
        )

    def reveal(self, path):
        if tuple(path) == self.requested_path[: len(path)] and not imgui.is_popup_open(path[-1]):
            imgui.open_popup(path[-1])

    def preview(self, path):
        """Select a real popup stack for the deterministic capture gallery."""
        self.requested_path = tuple(path)
        self.reset_pending = True

    def begin_frame(self):
        self.observed.clear()
        self.color_visible = False
        # ImGui may dismiss a popup during NewFrame, before its body can handle Escape.
        if self.color_before is not None and imgui.is_key_pressed(imgui.Key.escape):
            self.study.preview.entities[:] = self.color_before
            self.color_before = None
            self.color_entity = None
        if self.reset_pending:
            if imgui.is_popup_open(
                "", imgui.PopupFlags_.any_popup_id | imgui.PopupFlags_.any_popup_level
            ):
                imgui.internal.close_popup_to_level(0, True)
            self.reset_pending = False

    def bar(self):
        if not imgui.begin_menu_bar():
            return
        imgui.set_cursor_pos_x(44 * self.s)
        with self.style():
            for name in TOP_MENUS:
                self.reveal((name,))
                self.constrain(name)
                if imgui.begin_menu(name):
                    self.content(name, (name,))
                    imgui.end_menu()
        imgui.end_menu_bar()

    def popup(self, name, anchor, clicked=False):
        with self.style():
            if clicked:
                imgui.open_popup(name)
            self.reveal((name,))
            width, height = self.limits(name)
            vp = imgui.get_main_viewport()
            x, y, _w, h = anchor
            px = min(max(vp.pos.x + 8 * self.s, x), vp.pos.x + vp.size.x - width - 8 * self.s)
            py = y + h + 4 * self.s
            if py + height > vp.pos.y + vp.size.y - 8 * self.s:
                py = max(vp.pos.y + 8 * self.s, y - height - 4 * self.s)
            imgui.set_next_window_pos((px, py))
            self.constrain(name)
            if imgui.begin_popup(name):
                self.content(name, (name,))
                imgui.end_popup()

    def shadow_and_record(self, name, path):
        p, size = imgui.get_window_pos(), imgui.get_window_size()
        dl = imgui.get_window_draw_list()
        dl.push_clip_rect_full_screen()
        for outset, alpha in ((7, 4), (5, 8), (3, 14), (1.5, 20)):
            d = outset * self.s
            dl.add_rect(
                (p.x - d, p.y - d + self.s),
                (p.x + size.x + d, p.y + size.y + d + self.s),
                color((0, 0, 0, alpha)),
                rounding=(8 + outset) * self.s,
                thickness=2 * self.s,
            )
        dl.pop_clip_rect()
        record = {
            "name": name,
            "path": list(path),
            "rect": [p.x, p.y, size.x, size.y],
            "scroll_max": imgui.get_scroll_max_y(),
            "rows": [],
        }
        self.observed["/".join(path)] = record
        return record

    def content(self, name, path):
        record = self.shadow_and_record(name, path)
        for row in self.items(name):
            if row is None:
                p = imgui.get_cursor_screen_pos()
                height = round(10 * self.s)
                gap = imgui.get_style().item_spacing.y
                self.ui.line(p.x, p.y - gap / 2 + height / 2, imgui.get_content_region_avail().x)
                imgui.push_style_var(imgui.StyleVar_.item_spacing, (12 * self.s, 0))
                imgui.dummy((1, height))
                imgui.pop_style_var()
                continue
            self.item(row, path, record)

    def item(self, row, path, record):
        u, s = self.ui, self.s
        p = imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        right = p.x + imgui.get_content_region_avail().x
        if row.submenu:
            self.reveal((*path, row.submenu))
            self.constrain(row.submenu)
        # MenuItemEx/BeginMenuEx retain native navigation, hover corridors and popup ownership.
        imgui.push_style_color(imgui.Col_.text, (0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text_disabled, (0, 0, 0, 0))
        if row.submenu:
            activated = imgui.internal.begin_menu_ex(row.label, "     ", row.enabled)
        else:
            activated = imgui.internal.menu_item_ex(
                row.label, "     ", row.shortcut, row.checked, row.enabled
            )
        imgui.pop_style_color(2)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        visible = imgui.is_item_visible()
        c = TEXT if row.enabled else DISABLED
        text_x = p.x + 28 * s
        tail_x = right - 12 * s
        if visible:
            if row.icon:
                u.icon(row.icon, p.x, p.y - 2 * s, 16, MUTED if row.enabled else DISABLED, draw=dl)
            u.text(text_x, p.y, row.label, c, draw=dl)
            if row.shortcut:
                width = u.measure(row.shortcut, 11)
                tail_x = right - 24 * s - width
                u.text(tail_x, p.y + s, row.shortcut, DIM if row.enabled else DISABLED, 11, draw=dl)
            if row.submenu or row.checked:
                u.icon(
                    "right" if row.submenu else "check",
                    right - 12 * s,
                    p.y,
                    12,
                    DIM if row.submenu else SAGE,
                    draw=dl,
                )
        record["rows"].append(
            {
                "label": row.label,
                "enabled": row.enabled,
                "checked": row.checked,
                "rect": [lo.x, lo.y, hi.x - lo.x, hi.y - lo.y],
                "label_rect": [text_x, p.y, u.measure(row.label), 12 * s],
                "tail_x": tail_x,
                "visible": visible,
            }
        )
        if (
            not row.enabled
            and row.reason
            and imgui.is_item_hovered(
                imgui.HoveredFlags_.allow_when_disabled | imgui.HoveredFlags_.delay_normal
            )
        ):
            u.tooltip(row.reason)
        if row.submenu and activated:
            self.content(row.submenu, (*path, row.submenu))
            imgui.end_menu()
        elif activated:
            self.dispatch(row.action)

    def dispatch(self, action):
        st = self.study
        prefix, _, value = action.partition(":")
        if prefix == "layout":
            st.pending = value
        elif prefix == "workspace":
            st.preset(value)
        elif prefix == "panel":
            st.activate(value)
        elif prefix == "toggle":
            if value in ("Inspector", "Timeline"):
                if value == "Timeline":
                    st.toggle_timeline()
                else:
                    st.open[value] = not st.open[value]
            elif st.open[value]:
                st.open[value] = False
            else:
                st.activate(value)
        elif prefix == "tool":
            st.preview.tool = value
        elif prefix == "space":
            st.space = value
        elif prefix == "projection":
            st.preview.camera.set_orthographic(value == "orthographic")
        elif prefix == "shading":
            from mojive.render.backend import DebugView

            self.shading = value
            st.preview.backend.set_debug_view(
                DebugView.WIREFRAME if value == "Wireframe" else DebugView.SHADED
            )
        elif prefix == "speed":
            st.speed = float(value)
        elif prefix == "step":
            if not st.playing:
                st.time = min(10, max(0, st.time + int(value) / 60))
                st.sample_pose()
        elif prefix == "dialog":
            self.dialog_pending = value
        elif action == "play":
            st.playing = not st.playing
        elif action == "start":
            st.time = 0
            st.sample_pose()
        elif action == "loop":
            st.loop = not st.loop
        elif action == "frame":
            st.preview.camera.adopt(st.initial_camera())
        elif action == "gizmos":
            st.preview.gizmos = not st.preview.gizmos
        elif action == "shadows":
            st.preview.shadows = not st.preview.shadows
        elif action == "viewport-tab":
            st.show_viewport_tab = not st.show_viewport_tab
        elif action == "timeline":
            st.toggle_timeline()
        elif action == "undo":
            st.undo()
        elif action == "redo" and st.redo:
            st.history.append(copy.deepcopy(st.preview.entities))
            st.preview.entities[:] = st.redo.pop()
        elif action in ("visibility", "transform:reset") and st.preview.entity:
            before = copy.deepcopy(st.preview.entities)
            e = st.preview.entity
            if action == "visibility":
                e["hidden"] = not e["hidden"]
            else:
                original = next(
                    v
                    for v in json.loads((ROOT / "document.json").read_text())["entities"]
                    if v["id"] == e["id"]
                )
                for key in ("position", "rotation", "scale"):
                    e[key] = original[key]
            st.change(before)
        elif prefix == "file":
            if value in ("save", "export"):
                destination = st.output_directory / (
                    "saved-scene.json" if value == "save" else "exported-scene.json"
                )
                destination.write_text(json.dumps(st.preview.document, indent=2))
                st.events.append("Scene saved: " + destination.name)
            elif value == "load":
                saved = json.loads((st.output_directory / "saved-scene.json").read_text())
                if {e["id"] for e in saved["entities"]} != {e["id"] for e in st.preview.entities}:
                    raise ValueError("Saved reference has incompatible entity identities")
                before = copy.deepcopy(st.preview.entities)
                st.preview.entities[:] = saved["entities"]
                st.preview.document["pose"] = saved["pose"]
                st.change(before)
        elif action != "redo":
            raise ValueError(action)

    def dialogs(self):
        if self.dialog_pending:
            self.dialog = self.dialog_pending
            self.dialog_pending = None
            imgui.open_popup("Reference dialog")
        with self.style(controls=True):
            vp = imgui.get_main_viewport()
            width, height = (
                min(500 * self.s, vp.size.x - 32 * self.s),
                min(430 * self.s, vp.size.y - 32 * self.s),
            )
            imgui.set_next_window_size((width, height))
            imgui.set_next_window_pos(
                (vp.pos.x + vp.size.x / 2, vp.pos.y + vp.size.y / 2),
                imgui.Cond_.appearing,
                (0.5, 0.5),
            )
            opened, _ = imgui.begin_popup_modal(
                "Reference dialog",
                None,
                imgui.WindowFlags_.no_title_bar
                | imgui.WindowFlags_.no_resize
                | imgui.WindowFlags_.no_saved_settings,
            )
            if not opened:
                return
            self.shadow_and_record("Dialog", ("Dialog",))
            u, s = self.ui, self.s
            p = imgui.get_cursor_screen_pos()
            w = imgui.get_content_region_avail().x
            u.text(p.x, p.y + 3 * s, self.dialog, TEXT, 17, bold=True)
            if u.button(
                "close-guide", p.x + w - 26 * s, p.y, 26, 26, icon="close", tooltip="Close dialog"
            ) or imgui.is_key_pressed(imgui.Key.escape):
                imgui.close_current_popup()
            u.text(p.x, p.y + 32 * s, "MOJIVE / INSTRUMENT", DIM, 11)
            imgui.set_cursor_screen_pos((p.x, p.y + 60 * s))
            imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (8 * s, 0))
            imgui.begin_child("Guide content", (w, max(24 * s, height - 138 * s)))
            if self.dialog == "Interaction guide":
                for title, value in [
                    ("Orbit", "Drag an empty area"),
                    ("Pan", "Middle mouse drag"),
                    ("Zoom", "Mouse wheel"),
                    ("Transform tools", "V / W / E / R"),
                    ("World / body frame", "B"),
                    ("Snap", "S"),
                    ("Frame all", "F"),
                    ("Undo / redo", "⌘Z / ⇧⌘Z"),
                    ("Preview playback", "Space"),
                    ("Timeline", "T"),
                    ("Dock panels", "Drag a panel tab"),
                ]:
                    q = imgui.get_cursor_screen_pos()
                    u.text(q.x, q.y + 6 * s, title, MUTED, 11)
                    u.text(q.x + w * 0.47, q.y + 6 * s, value, TEXT, 11)
                    imgui.dummy((w, 24 * s))
            else:
                imgui.text_wrapped(
                    "Instrument 07 is a native UI reference for Mojive. It shares the browser design's icon masters, typography and layout proportions."
                )
                imgui.spacing()
                imgui.text_wrapped(
                    "Menus retain Dear ImGui's keyboard navigation, hover switching, submenu behavior and docking integration."
                )
                imgui.spacing()
                imgui.text_disabled("Dear ImGui " + imgui.get_version())
                imgui.text_disabled("Local scene preview · no simulation adapter")
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
            bottom = imgui.get_window_pos().y + height - 44 * s
            u.line(p.x, bottom - 10 * s, w)
            if u.button("done-guide", p.x + w - 72 * s, bottom, 72, 28, label="Done", active=True):
                imgui.close_current_popup()
            imgui.end_popup()

    def color_control(self, entity, x, y, width):
        from .scene_preview import rgba

        u, s = self.ui, self.s
        u.text(x, y, "Base color", MUTED, 11)
        clicked = u.button("material-color", x, y + 22 * s, width / s, 30, filled=True)
        u.rect(
            x + 7 * s,
            y + 29 * s,
            16 * s,
            16 * s,
            tuple(int(v * 255) for v in rgba(entity["color"])),
            3,
        )
        u.text(x + 34 * s, y + 32 * s, entity["color"].upper(), TEXT, 11, mono=True)
        u.icon("down", x + width - 23 * s, y + 31 * s, 12, DIM)
        with self.style(controls=True):
            if (clicked or self.requested_path == ("Color",)) and not imgui.is_popup_open("Color"):
                self.color_before = copy.deepcopy(self.study.preview.entities)
                self.color_entity = entity
                imgui.open_popup("Color")
            if not imgui.is_popup_open("Color"):
                return
            vp = imgui.get_main_viewport()
            content_w = min(248 * s, max(136 * s, vp.work_size.y - 112 * s))
            popup_w, popup_h = content_w + 32 * s, content_w + 86 * s
            px = min(x, vp.pos.x + vp.size.x - popup_w - 8 * s)
            py = y + 56 * s
            if py + popup_h > vp.pos.y + vp.size.y - 8 * s:
                py = max(vp.pos.y + 8 * s, y - popup_h)
            imgui.set_next_window_pos((px, py))
            imgui.set_next_window_size_constraints((popup_w, 0), (popup_w, vp.size.y - 16 * s))
            if not imgui.begin_popup("Color"):
                return
            self.color_visible = True
            record = self.shadow_and_record("Color", ("Color",))
            record["controls"] = {}
            imgui.text_unformatted("Base color")
            imgui.set_next_item_width(content_w)
            flags = (
                imgui.ColorEditFlags_.no_side_preview
                | imgui.ColorEditFlags_.no_small_preview
                | imgui.ColorEditFlags_.no_inputs
                | imgui.ColorEditFlags_.no_options
                | imgui.ColorEditFlags_.no_label
                | imgui.ColorEditFlags_.picker_hue_bar
            )
            _, rgb = imgui.color_picker3("##base-color", rgba(entity["color"])[:3], flags)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            record["controls"]["picker"] = [lo.x, lo.y, hi.x - lo.x, hi.y - lo.y]
            imgui.set_next_item_width(content_w)
            _, rgb = imgui.color_edit3(
                "##hex",
                rgb,
                imgui.ColorEditFlags_.display_hex
                | imgui.ColorEditFlags_.no_picker
                | imgui.ColorEditFlags_.no_options
                | imgui.ColorEditFlags_.no_small_preview
                | imgui.ColorEditFlags_.no_label,
            )
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            record["controls"]["hex"] = [lo.x, lo.y, hi.x - lo.x, hi.y - lo.y]
            updated = "#" + "".join(f"{round(max(0, min(1, v)) * 255):02x}" for v in rgb)
            if updated != entity["color"].lower():
                entity["color"] = updated
            p = imgui.get_cursor_screen_pos()
            u.text(p.x, p.y + 8 * s, "Esc to cancel", DIM, 11)
            if u.button(
                "done-color", p.x + content_w - 64 * s, p.y, 64, 28, label="Done", active=True
            ):
                imgui.close_current_popup()
            record["controls"]["done"] = list(u.hits["done-color"])
            imgui.end_popup()

    def finish_color_edit(self):
        if self.color_before is not None and not self.color_visible:
            self.study.change(self.color_before)
            self.color_before = None
            self.color_entity = None
