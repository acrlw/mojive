"""Native menu semantics with the Instrument spacing and shared icon masters."""

from __future__ import annotations

from contextlib import contextmanager
from math import ceil, floor

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import InputAction
from mojive.ui.workspace import menu_color
from mojive.ui.workspace.color_editor import ColorEditor
from mojive.ui.workspace.menu_catalog import TOP_MENUS
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menus import Menus as WorkspaceMenus
from mojive.ui.workspace.widgets import DIM, MUTED, TEXT, color
from mojive.ui.workspace.window_chrome import surface_ink

DISABLED = (126, 136, 149, 255)
SAVED_SCENE = "saved-scene.mojive.json"
EXPORTED_SCENE = "exported-scene.mojive.json"


class Menus:
    local_rows = WorkspaceMenus.local_rows
    local_separator = WorkspaceMenus.local_separator
    _begin_local_row = WorkspaceMenus._begin_local_row

    def __init__(self, study, panels):
        self.study = self.workspace = study
        self.panels = panels
        self.ui = study.ui
        self.s = study.s
        self.requested_path = ()
        self.reset_pending = False
        self.observed = {}
        self.dialog = None
        self.dialog_pending = None
        self.color_edit = None
        self.color_visible = False
        self.color_editor = ColorEditor()
        self.shading = "Solid"
        self.menu_rects = {}
        self.menu_center_y = 0.0
        self.menu_height = 0.0
        self.visible_menus = TOP_MENUS
        self.overflow_menus = ()

    @contextmanager
    def style(self, controls=False):
        s = self.s
        # ImGui snaps each cursor advance. Use an integral row height at fractional UI scales.
        row_gap = (round(30 * s) - 13 * s) / s
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
        colors = [
            (imgui.Col_.popup_bg, surface_ink(self.ui, "menu", "popup")),
            (imgui.Col_.border, (76, 86, 99, 150)),
            (imgui.Col_.header, (156, 191, 141, 24)),
            (imgui.Col_.header_hovered, (255, 255, 255, 13)),
            (imgui.Col_.header_active, (255, 255, 255, 18)),
        ]
        for key, value in colors:
            imgui.push_style_color(key, imgui.ImVec4(*(v / 255 for v in value)))
        imgui.push_font(self.ui.body, 13 * s)
        try:
            yield
        finally:
            imgui.pop_font()
            imgui.pop_style_color(len(colors))
            imgui.pop_style_var(len(values) + 2)

    def items(self, name):
        st = self.study
        e = st.preview.entity
        if name == "File":
            return [
                Item("Save locally", "file:save", "save", "⌘S"),
                Item(
                    "Open saved scene",
                    "file:load",
                    "assets",
                    "⌘O",
                    (self.study.output_directory / SAVED_SCENE).exists(),
                    reason="Save a local scene first.",
                ),
                None,
                Item("Export scene JSON", "file:export", "export"),
            ]
        if name == "Edit":
            return [
                Item(
                    "Undo",
                    "undo",
                    "undo",
                    "⌘Z",
                    st.preview.session.can_undo,
                    reason="No changes to undo.",
                ),
                Item(
                    "Redo",
                    "redo",
                    "redo",
                    "⇧⌘Z",
                    st.preview.session.can_redo,
                    reason="No changes to redo.",
                ),
                None,
                Item(
                    "Visible",
                    "visibility",
                    "eye",
                    enabled=bool(e),
                    checkable=True,
                    checked=bool(e) and not e["hidden"],
                ),
            ]
        if name == "Add":
            return [
                Item(
                    n,
                    icon="shape-" + n.lower(),
                    action="create:" + n,
                    enabled=n != "Capsule" or st.preview.session.adapter.caps.topology_editing,
                )
                for n in ("Box", "Sphere", "Cylinder", "Capsule", "Cone", "Ellipsoid", "Plane")
            ]
        if name == "View":
            return [
                Item("Frame all", "frame", "frame", "F"),
                None,
                Item(
                    "Projection",
                    icon="orthographic" if st.preview.camera.orthographic else "perspective",
                    submenu="Projection",
                ),
                Item("Transform tool", icon=st.preview.tool, submenu="Transform tool"),
                Item("Gizmo orientation", icon="world", submenu="Gizmo orientation"),
                None,
                Item(
                    "Transform handles", "gizmos", "move", checkable=True, checked=st.preview.gizmos
                ),
            ]
        if name in ("Camera", "Projection"):
            rows = [
                Item(
                    "Perspective",
                    "projection:perspective",
                    "perspective",
                    checkable=True,
                    checked=not st.preview.camera.orthographic,
                ),
                Item(
                    "Orthographic",
                    "projection:orthographic",
                    "orthographic",
                    checkable=True,
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
                Item(n.title(), "tool:" + n, n, key, checkable=True, checked=st.preview.tool == n)
                for n, key in [("select", "V"), ("move", "W"), ("rotate", "E"), ("scale", "R")]
            ]
        if name == "Gizmo orientation":
            return [
                Item(
                    n + " frame",
                    "space:" + n,
                    "world" if n == "World" else "body-frame",
                    checkable=True,
                    checked=st.space == n,
                )
                for n in ("World", "Body")
            ]
        if name == "Simulate":
            return [
                Item(
                    "Pause simulation" if st.playing else "Play simulation",
                    "play",
                    "pause" if st.playing else "play",
                    "Space",
                ),
                Item("Go to start", "start", "reset"),
                Item("Playback speed", icon="play", submenu="Playback speed"),
                None,
                Item(
                    "Stop recording"
                    if st.preview.session.state_take_recording
                    else "Record simulation",
                    "record",
                    icon="record",
                    enabled=st.preview.session.adapter.caps.simulation,
                    checkable=True,
                    checked=st.preview.session.state_take_recording,
                ),
                Item("Physics adapter", "panel:Control", "control"),
            ]
        if name == "Playback speed":
            return [
                Item(f"{v:g}× speed", "speed:" + str(v), checkable=True, checked=st.speed == v)
                for v in (0.25, 0.5, 1, 2)
            ]
        if name == "Playback":
            return [
                Item("Previous frame", "step:-1", "previous", enabled=not st.playing),
                Item("Next frame", "step:1", "next", enabled=not st.playing),
                None,
                Item("Playback speed", icon="play", submenu="Playback speed"),
                Item(
                    "Loop playback",
                    "loop",
                    "loop",
                    checkable=True,
                    checked=st.loop,
                    enabled=bool(st.preview.session.state_take_times),
                    reason="Record a take first.",
                ),
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
                Item(
                    "Show Viewport tab",
                    "viewport-tab",
                    "inspect",
                    checkable=True,
                    checked=st.show_viewport_tab,
                ),
                Item("Settings", "panel:Settings", "settings", "F9"),
            ]
        if name == "Workspace":
            return [
                Item(n, "workspace:" + n, checkable=True, checked=st.workspace == n)
                for n in ("Edit", "Author", "Review")
            ]
        if name == "Panels":
            return [
                Item(n, "toggle:" + n, icon, checkable=True, checked=st.open[n])
                for n, icon in self.panels
                if n != "Settings"
            ] + [
                None,
                Item(
                    "Inspector",
                    "toggle:Inspector",
                    "inspect",
                    checkable=True,
                    checked=st.open["Inspector"],
                ),
                Item(
                    "Timeline", "timeline", "key", "T", checkable=True, checked=st.open["Timeline"]
                ),
            ]
        if name == "Inspector position":
            return [
                Item("Float", "layout:float", "inspect"),
                Item("Dock on the right", "layout:dock", "inspect"),
                Item("Group with Scene", "layout:group", "scene"),
            ]
        if name == "Shading":
            return [
                Item(n, "shading:" + n, "shading", checkable=True, checked=self.shading == n)
                for n in ("Solid", "Wireframe")
            ]
        if name == "Help":
            return [
                Item("Interaction guide", "dialog:Interaction guide", "help"),
                Item("About this reference", "dialog:About this reference", "info"),
            ]
        raise ValueError(name)

    def limits(self, name):
        from mojive.ui.workspace.menus import Menus as WorkspaceMenus

        return WorkspaceMenus.limits(self, name)

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
        if self.color_edit is not None and imgui.is_key_pressed(imgui.Key.escape):
            menu_color.finish_color_edit(self, cancel=True)
        if self.reset_pending:
            if imgui.is_popup_open(
                "", imgui.PopupFlags_.any_popup_id | imgui.PopupFlags_.any_popup_level
            ):
                imgui.internal.close_popup_to_level(0, True)
            self.reset_pending = False

    def bar_layout(self, width):
        """Reserve a native More menu before selecting the visible menu prefix."""
        spacing = imgui.get_style().item_spacing.x * 2
        widths = [ceil(self.ui.measure(name) + spacing) for name in TOP_MENUS]
        if sum(widths) <= width:
            return TOP_MENUS, ()
        available = width - ceil(self.ui.measure("More") + spacing)
        count = 0
        for item_width in widths:
            if item_width > available:
                break
            available -= item_width
            count += 1
        return TOP_MENUS[:count], TOP_MENUS[count:]

    def record_menu(self, name):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rect = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        self.menu_rects[name] = self.ui.hits["menu-" + name] = rect
        return rect

    def bar_menu(self, name):
        self.reveal((name,))
        self.constrain(name)
        opened = imgui.begin_menu(self.ui.translate(name) + "###" + name)
        self.record_menu(name)
        if opened:
            self.content(name, (name,))
            imgui.end_menu()

    def bar(self, right_edge=None):
        """Fit menus before the first reserved header control's screen-space x coordinate."""
        for name in (*TOP_MENUS, "More"):
            self.ui.hits.pop("menu-" + name, None)
        self.menu_rects.clear()
        if not imgui.begin_menu_bar():
            return
        imgui.set_cursor_pos_x(44 * self.s)
        with self.style():
            p = imgui.get_cursor_screen_pos()
            window = imgui.internal.get_current_window()
            ink = ImguiDraw2D().text_ink_bounds("H")
            # Native RenderText floors its pen position; match its cap center exactly.
            self.menu_center_y = floor(p.y + window.dc.curr_line_text_base_offset) + (
                (ink[1] + ink[3]) / 2 if ink else imgui.get_font_size() / 2
            )
            if right_edge is None:
                reserved = 228 if imgui.get_window_width() >= 760 * self.s else 57
                right_edge = imgui.get_window_pos().x + imgui.get_window_width() - reserved * self.s
            available = max(0, right_edge - 8 * self.s - p.x)
            self.visible_menus, self.overflow_menus = self.bar_layout(available)
            for name in self.visible_menus:
                self.bar_menu(name)
            if self.overflow_menus:
                if (
                    self.requested_path
                    and self.requested_path[0] in self.overflow_menus
                    and not imgui.is_popup_open("More")
                ):
                    imgui.open_popup("More")
                width = max(
                    160 * self.s,
                    max(self.ui.measure(name) for name in self.overflow_menus) + 60 * self.s,
                )
                imgui.set_next_window_size_constraints(
                    (width, 0), (width, imgui.get_main_viewport().size.y - 16 * self.s)
                )
                opened = imgui.begin_menu(self.ui.translate("More") + "###More")
                self.record_menu("More")
                if opened:
                    self.shadow_and_record("More", ("More",))
                    for name in self.overflow_menus:
                        self.bar_menu(name)
                    imgui.end_menu()
            first = self.menu_rects.get(self.visible_menus[0] if self.visible_menus else "More")
            self.menu_height = first[3] if first else 0.0
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
        from mojive.ui.workspace.menus import Menus as WorkspaceMenus

        return WorkspaceMenus.content(self, name, path)

    def item(self, row, path, record):
        from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row

        self._begin_local_row()
        submenu = None
        if row.submenu:
            self.reveal((*path, row.submenu))
            self.constrain(row.submenu)

            def submenu(_native_label):
                return imgui.internal.begin_menu_ex(row.label, "     ", row.enabled)

        result = paint_menu_row(
            self.ui,
            row,
            "menu-item-" + "/".join(path) + "/" + row.label,
            content_right=getattr(self, "_content_right", None),
            open_submenu=submenu,
            columns=getattr(self, "_columns", None),
        )
        record["rows"].append(result.observation)
        if row.submenu and result.activated:
            self.content(row.submenu, (*path, row.submenu))
            imgui.end_menu()
        elif result.activated:
            self.dispatch(row.action)
        advance_menu_row(self.ui, result)
        self._last_row_bottom = result.box.outer.bottom

    def dispatch(self, action):
        st = self.study
        app, session = st.preview, st.preview.session
        prefix, _, value = action.partition(":")
        if prefix == "layout":
            st.pending = value
        elif prefix == "workspace":
            st.preset(value)
        elif prefix == "panel":
            st.activate(value)
        elif prefix == "toggle":
            if value == "Timeline":
                st.toggle_timeline()
            elif st.open[value]:
                st.open[value] = False
            else:
                st.activate(value)
        elif prefix == "tool":
            app.tool = value
        elif prefix == "space":
            st.space = value
        elif prefix == "projection":
            app.camera.set_orthographic(value == "orthographic")
        elif prefix == "shading":
            from mojive.render.backend import DebugView

            self.shading = value
            app.backend.set_debug_view(
                DebugView.WIREFRAME if value == "Wireframe" else DebugView.SHADED
            )
        elif prefix == "speed":
            st.speed = float(value)
        elif prefix == "step":
            if not st.playing:
                session.submit(cmd.Step() if int(value) > 0 else cmd.StepBack())
        elif prefix == "create":
            from mojive.types import MeshShape

            if value == "Capsule":
                if session.adapter.caps.topology_editing:
                    app._add_model_primitive("capsule", "capsule")
            elif value == "Ellipsoid":
                app._add_scene_object(MeshShape.SPHERE, "ellipsoid", size=(0.65, 0.45, 0.35))
            else:
                app._add_scene_object(MeshShape(value.lower()), value.lower())
        elif prefix == "dialog":
            self.dialog_pending = value
        elif action == "play":
            app._toggle_playback()
        elif action == "record":
            app._toggle_state_take_recording()
        elif action == "start":
            app._reset_playback()
        elif action == "loop":
            st.loop = not st.loop
        elif action == "frame":
            app._frame_scene(animate=True)
        elif action == "gizmos":
            app.gizmos = not app.gizmos
        elif action == "shadows":
            app.shadows = not app.shadows
        elif action == "viewport-tab":
            st.show_viewport_tab = not st.show_viewport_tab
        elif action == "timeline":
            st.toggle_timeline()
        elif action == "undo":
            st.undo()
        elif action == "redo":
            session.submit(cmd.Redo())
        elif action == "visibility" and session.selected_node is not None:
            node = session.selected_node
            session.submit(cmd.SetVisible(node.node_id, not node.visible))
        elif prefix == "file":
            destination = st.output_directory / (
                EXPORTED_SCENE if value == "export" else SAVED_SCENE
            )
            if value == "load":
                app._request_document_action("open_scene", destination)
            else:
                app.save_scene(destination)
        else:
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
            u.text(p.x, p.y + 32 * s, "MOJIVE / INSTRUMENT", DIM, 13)
            imgui.set_cursor_screen_pos((p.x, p.y + 60 * s))
            imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (8 * s, 0))
            imgui.begin_child("Guide content", (w, max(24 * s, height - 138 * s)))
            if self.dialog == "Interaction guide":
                bindings = self.study.preview.input_bindings
                for title, value in [
                    ("Orbit", "Drag an empty area"),
                    ("Pan", "Middle mouse drag"),
                    ("Zoom", "Mouse wheel"),
                    (
                        "Fly camera",
                        " / ".join(
                            " ".join(bindings.label(action) for action in group)
                            for group in (
                                (
                                    InputAction.FLY_FORWARD,
                                    InputAction.FLY_LEFT,
                                    InputAction.FLY_BACK,
                                    InputAction.FLY_RIGHT,
                                ),
                                (InputAction.FLY_UP, InputAction.FLY_DOWN),
                            )
                        ),
                    ),
                    (
                        "Transform tools",
                        "V / "
                        + " / ".join(
                            bindings.label(action)
                            for action in (InputAction.GIZMO_TRANSLATE, InputAction.GIZMO_ROTATE)
                        ),
                    ),
                    ("World / body frame", bindings.label(InputAction.GIZMO_SPACE)),
                    ("Snap", bindings.label(InputAction.SNAP)),
                    (
                        "Frame all / focus",
                        bindings.label(InputAction.FRAME_SCENE) + " / double-click",
                    ),
                    ("Settings", "F9 / ⌘,"),
                    ("Precise gizmo input", "Double-click a handle"),
                    ("Undo / redo", "⌘Z / ⇧⌘Z"),
                    ("Simulation playback", "Space"),
                    ("Timeline", "T"),
                    ("Dock panels", "Drag a panel tab"),
                ]:
                    q = imgui.get_cursor_screen_pos()
                    column = w * 0.47
                    if u.measure(title, 13) + 12 * s > column or u.measure(value, 13) > w - column:
                        imgui.push_font(u.body, 13 * s)
                        imgui.push_text_wrap_pos(0)
                        imgui.text_colored(tuple(v / 255 for v in MUTED), u.translate(title))
                        imgui.text_wrapped(u.translate(value))
                        imgui.pop_text_wrap_pos()
                        imgui.pop_font()
                        imgui.dummy((0, 8 * s))
                    else:
                        u.text(q.x, q.y + 6 * s, title, MUTED, 13)
                        u.text(q.x + column, q.y + 6 * s, value, TEXT, 13)
                        imgui.dummy((w, 28 * s))
            else:
                imgui.text_wrapped("Instrument is Mojive's native UI design reference.")
                imgui.spacing()
                imgui.text_disabled("Dear ImGui " + imgui.get_version())
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
            bottom = imgui.get_window_pos().y + height - 44 * s
            u.line(p.x, bottom - 10 * s, w)
            if u.button("done-guide", p.x + w - 72 * s, bottom, 72, 28, label="Done", active=True):
                imgui.close_current_popup()
            imgui.end_popup()

    def color_control(self, entity, x, y, width):
        return menu_color.color_control(self, entity, x, y, width)

    def finish_color_edit(self):
        return menu_color.finish_color_edit(self)
