"""Native menu semantics with the Instrument spacing and shared icon masters."""

from __future__ import annotations

from contextlib import contextmanager
from math import ceil, floor

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.imgui_draw import ImguiDraw2D

from . import menu_actions, menu_catalog, menu_color, menu_dialogs
from .color_editor import ColorEditor
from .design import BODY_SIZE, CONTROL_RADIUS, DESIGN
from .menu_item import Item as Item
from .widgets import DIM, MUTED, SAGE, TEXT, color

MENU_BG = (39, 45, 53, 255)
DISABLED = (126, 136, 149, 255)
TOP_MENUS = ("File", "Edit", "Add", "View", "Simulate", "Window", "Help")


class Menus:
    def __init__(self, workspace, panels):
        self.workspace = workspace
        self.panels = panels
        self.ui = workspace.ui
        self.s = workspace.s
        self.requested_path = ()
        self.reset_pending = False
        self.observed = {}
        self.dialog = None
        self.dialog_pending = None
        self.color_before = None
        self.color_visible = False
        self.color_editor = ColorEditor()
        self.shading = "Solid"
        self.menu_rects = {}
        self.menu_center_y = 0.0
        self.menu_height = 0.0
        self.visible_menus = TOP_MENUS
        self.overflow_menus = ()
        self.actions = []

    def apply_actions(self):
        if self.actions:
            actions, self.actions = self.actions, []
            for action in actions:
                self.dispatch(action)

    @contextmanager
    def style(self, controls=False):
        s = self.s
        # The native selectable fills the adopted row; layout owns the gap between rows.
        row_gap = (round(DESIGN.menu_item_height * s) - BODY_SIZE * s) / s
        values = [
            (imgui.StyleVar_.window_padding, ((16, 14) if controls else (12, 6 + row_gap / 2))),
            (imgui.StyleVar_.window_min_size, (0, 0)),
            (imgui.StyleVar_.item_spacing, ((8, 8) if controls else (12, row_gap))),
            (imgui.StyleVar_.frame_padding, (8, 6)),
        ]
        for key, value in values:
            imgui.push_style_var(key, tuple(v * s for v in value))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, 8 * s)
        imgui.push_style_var(imgui.StyleVar_.menu_item_rounding, CONTROL_RADIUS * s)
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
        imgui.push_font(self.ui.body, BODY_SIZE * s)
        try:
            yield
        finally:
            imgui.pop_font()
            imgui.pop_style_color(len(colors))
            imgui.pop_style_var(len(values) + 3)

    def items(self, name):
        return menu_catalog.items(self, name)

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
            12 * self.s
            + sum(
                round((DESIGN.menu_item_height + DESIGN.menu_row_gap if r else 10) * self.s)
                for r in rows
            ),
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
        if self.color_before is not None and imgui.is_key_pressed(imgui.Key.escape):
            self.workspace.app.session.submit(cmd.CancelEditTransaction())
            self.workspace.app.refresh_document()
            self.color_before = None
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
                reserved = 57
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
        text_y = p.y + (DESIGN.menu_item_height - BODY_SIZE) * s / 2
        imgui.set_cursor_screen_pos((p.x, text_y))
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
        icon_x = p.x + DESIGN.menu_padding_x * s
        text_x = icon_x + (28 * s if row.icon else 0)
        tail_x = right - 12 * s
        if visible:
            if row.icon:
                u.role_icon(
                    row.icon,
                    icon_x,
                    p.y,
                    16 * s,
                    28 * s,
                    role="toolbar",
                    c=MUTED if row.enabled else DISABLED,
                    draw=dl,
                )
            u.text(text_x, text_y, row.label, c, draw=dl)
            if row.shortcut:
                width = u.measure(row.shortcut, BODY_SIZE)
                tail_x = right - 24 * s - width
                u.text(
                    tail_x,
                    text_y,
                    row.shortcut,
                    DIM if row.enabled else DISABLED,
                    BODY_SIZE,
                    draw=dl,
                )
            if row.submenu or row.checked:
                u.role_icon(
                    "right" if row.submenu else "check",
                    right - 12 * s,
                    text_y,
                    12 * s,
                    14 * s,
                    role="inline",
                    c=DIM if row.submenu else SAGE,
                    draw=dl,
                )
        record["rows"].append(
            {
                "label": row.label,
                "enabled": row.enabled,
                "checked": row.checked,
                "rect": [lo.x, lo.y, hi.x - lo.x, hi.y - lo.y],
                "label_rect": [text_x, text_y, u.measure(row.label), BODY_SIZE * s],
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
            self.actions.append(row.action)
        imgui.set_cursor_screen_pos(
            (p.x, p.y + (DESIGN.menu_item_height + DESIGN.menu_row_gap) * s)
        )
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((0, 0))
        imgui.pop_style_var()

    def dispatch(self, action):
        return menu_actions.dispatch(self, action)

    def dialogs(self):
        return menu_dialogs.dialogs(self)

    def color_control(self, entity, x, y, width):
        return menu_color.color_control(self, entity, x, y, width)

    def finish_color_edit(self):
        return menu_color.finish_color_edit(self)
