"""Native menu semantics with the Instrument spacing and shared icon masters."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from math import ceil

from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.ui.workspace import menu_actions, menu_catalog, menu_color, menu_dialogs
from mojive.ui.workspace.color_editor import ColorEditor
from mojive.ui.workspace.component_layout import control_frame_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.menu_catalog import TOP_MENUS
from mojive.ui.workspace.menu_geometry import native_unit, popup_padding
from mojive.ui.workspace.menu_item import Item as Item
from mojive.ui.workspace.widgets import color

_POPUP_CACHE_LIMIT = 128


class Menus:
    top_menus = TOP_MENUS

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
        self.color_edit = None
        self.color_visible = False
        self.color_editor = ColorEditor()
        self.shading = "Solid"
        self.menu_rects = {}
        self.menu_center_y = 0.0
        self.menu_height = 0.0
        self.visible_menus = TOP_MENUS
        self.overflow_menus = ()
        self.actions = []
        self._popup_widths = {}
        self._popup_rows = {}
        self._local_measurements = None

    def apply_actions(self):
        if self.actions:
            actions, self.actions = self.actions, []
            for action in actions:
                self.dispatch(action)

    def style(self, controls=False, *, choices=False):
        from mojive.ui.workspace.popup_style import menu_popup_style

        return menu_popup_style(
            self.ui, controls, part="select_popup" if choices else "dialog" if controls else "menu"
        )

    def items(self, name):
        return menu_catalog.items(self, name)

    @part_scope("menu", inherit=False)
    def limits(self, name):
        from mojive.ui.workspace.menu_rows import menu_columns, menu_row_size

        rows, ui, scale = self.items(name), self.ui, self.s
        style = ui.component_style("menu")
        padding = popup_padding(ui, "menu")
        columns = menu_columns(ui, rows)
        dimensions = [menu_row_size(ui, row, columns=columns) for row in rows if row]
        width = max((size[0] for size in dimensions), default=0) + padding[0] + padding[2]
        if style.get("widthMode", "auto") == "fixed":
            width = style.get("width", 240) * scale
        height = padding[1] + padding[3] + sum(size[1] for size in dimensions)
        height += max(0, len(dimensions) - 1) * native_unit(
            style.get("rowGap", DESIGN.menu_row_gap) * scale
        )
        height += sum(native_unit(10 * scale) for row in rows if row is None)
        if style.get("heightMode", "auto") == "fixed":
            height = style.get("height", 300) * scale
        available = imgui.get_main_viewport().size
        return min(ceil(width), available.x - 16 * scale), min(
            ceil(height), available.y - 16 * scale
        )

    def constrain(self, name):
        from mojive.ui.workspace.popup_style import prepare_menu_popup

        width, _height = self.limits(name)
        prepare_menu_popup(self.ui, auto_minimum_width=width)

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

    @part_scope("menu_bar")
    def trigger_size(self, name):
        parts = (
            self.ui.design.button_parts("menu-" + name, name)
            if self.ui.design
            else ("menu_trigger",)
        )
        style = self.ui.component_style(parts)
        padding = style.edges("padding", fallback=(8, 0, 8, 0))
        margin = style.edges("margin")
        # Round the scaled window-space width before converting back to logical
        # units; fractional scales must not elide an exactly measured title.
        width = (
            ceil(
                self.ui.measure(
                    name, style.get("fontSize", BODY_SIZE), font_resolved=True, style=style
                )
                + (padding[0] + padding[2]) * self.s
            )
            / self.s
        )
        if style.get("widthMode", "auto") == "fixed":
            width = style.get("width", width)
        height = max(
            style.get("height", 28),
            style.line_height(style.get("fontSize", BODY_SIZE) * 1.4) + padding[1] + padding[3],
        )
        return width + margin[0] + margin[2], height + margin[1] + margin[3]

    def bar_layout(self, width):
        """Reserve a More menu before selecting the measured visible prefix."""
        gap = self.ui.metric("menu_bar", "columnGap", self.ui.metric("menu_bar", "gap", 0)) * self.s
        widths = [ceil(self.trigger_size(name)[0] * self.s + gap) for name in TOP_MENUS]
        if sum(widths) <= width:
            return TOP_MENUS, ()
        available = width - ceil(self.trigger_size("More")[0] * self.s + gap)
        count = 0
        for item_width in widths:
            if item_width > available:
                break
            available -= item_width
            count += 1
        return TOP_MENUS[:count], TOP_MENUS[count:]

    @part_scope("menu_bar")
    def bar_button(self, name, position, frame):
        # Match BeginMenu's menu-set hover exception without granting the rest
        # of the application input through an open popup.
        imgui.push_item_flag(
            imgui.internal.ItemFlagsPrivate_.no_window_hoverable_check, self._bar_menu_open
        )
        try:
            clicked = self.ui.button(
                "menu-" + name,
                position.x,
                position.y,
                *frame,
                label=name,
                active=imgui.is_popup_open(name),
                align="center",
                width_limit=frame[0] * self.s,
            )
            hovered = imgui.is_item_hovered()
        finally:
            imgui.pop_item_flag()
        if clicked and imgui.is_popup_open(name):
            imgui.internal.close_popup_to_level(0, True)
            self._bar_menu_open = False
            clicked = False
        elif self._bar_menu_open and hovered and not imgui.is_popup_open(name):
            imgui.open_popup(name)
        self.menu_rects[name] = self.ui.hits["menu-" + name]
        return clicked

    def bar_menu(self, name):
        position = imgui.get_cursor_screen_pos()
        width, height = self.trigger_size(name)
        frame = control_frame_size(
            self.ui, "menu-" + name, width * self.s, height * self.s, label=name
        )
        clicked = self.bar_button(name, position, frame)
        self.popup(name, self.menu_rects[name], clicked)
        gap = self.ui.metric("menu_bar", "columnGap", self.ui.metric("menu_bar", "gap", 0)) * self.s
        imgui.set_cursor_screen_pos((position.x + width * self.s + gap, position.y))

    def bar(self, right_edge=None, *, region=None):
        """Place shared menu triggers before reserved document and title actions."""
        for name in (*TOP_MENUS, "More"):
            self.ui.hits.pop("menu-" + name, None)
        self.menu_rects.clear()
        if not imgui.begin_menu_bar():
            return
        self._bar_menu_open = any(imgui.is_popup_open(name) for name in (*TOP_MENUS, "More"))
        imgui.set_cursor_pos_x(44 * self.s)
        position = imgui.get_cursor_screen_pos()
        height = self.trigger_size("File")[1]
        title = self.ui.component_style("titlebar")
        if region is not None:
            position.x = region.x + (26 + title.get("columnGap", title.get("gap", 10))) * self.s
            position.y = title.align_rect(region, 0, height * self.s, horizontal="start").y
        else:
            padding = title.edges("padding", self.s)
            content_height = DESIGN.application_height * self.s - padding[1] - padding[3]
            position.y = (
                imgui.get_window_pos().y
                + padding[1]
                + max(0, (content_height - height * self.s) / 2)
            )
        imgui.set_cursor_screen_pos(position)
        self.menu_center_y = position.y + height * self.s / 2
        if right_edge is None:
            right_edge = imgui.get_window_pos().x + imgui.get_window_width() - 57 * self.s
        self.visible_menus, self.overflow_menus = self.bar_layout(
            max(0, right_edge - 8 * self.s - position.x)
        )
        for name in self.visible_menus:
            self.bar_menu(name)
        if self.overflow_menus:
            p = imgui.get_cursor_screen_pos()
            width, trigger_height = self.trigger_size("More")
            frame = control_frame_size(
                self.ui,
                "menu-More",
                min(width * self.s, max(0, right_edge - p.x)),
                trigger_height * self.s,
                label="More",
            )
            clicked = self.bar_button("More", p, frame)
            if clicked or (self.requested_path and self.requested_path[0] in self.overflow_menus):
                imgui.open_popup("More")
            with self.local_popup("More") as opened:
                if opened:
                    for name in self.overflow_menus:
                        self.reveal((name,))
                        self.constrain(name)
                        with self.local_submenu(name, key=name, content=False) as submenu_open:
                            if submenu_open:
                                self.content(name, (name,))
                        self.menu_rects[name] = self.ui.hits[name]
        first = self.menu_rects.get(self.visible_menus[0] if self.visible_menus else "More")
        self.menu_height = first[3] if first else 0.0
        from mojive.ui.workspace.component_layout import record_group

        record_group(self.ui, "menu_bar", "application-menu-bar", self.menu_rects.values())
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((0, 0))
        imgui.pop_style_var()
        imgui.end_menu_bar()

    def popup(self, name, anchor, clicked=False):
        if clicked:
            imgui.open_popup(name)
        self.reveal((name,))
        if not imgui.is_popup_open(name):
            return
        with self.style():
            width, height = self.limits(name)
            vp = imgui.get_main_viewport()
            x, y, _w, h = anchor
            margins = self.ui.component_style("menu").edges("margin", self.s)
            x, y = x + margins[0] - margins[2], y + margins[1] - margins[3]
            px = min(max(vp.pos.x + 8 * self.s, x), vp.pos.x + vp.size.x - width - 8 * self.s)
            py = y + h + 4 * self.s
            if py + height > vp.pos.y + vp.size.y - 8 * self.s:
                py = max(vp.pos.y + 8 * self.s, y - height - 4 * self.s)
            imgui.set_next_window_pos((px, py))
            from mojive.ui.workspace.popup_style import prepare_menu_popup

            prepare_menu_popup(self.ui, auto_minimum_width=width)
            if imgui.begin_popup(name):
                self.content(name, (name,))
                imgui.end_popup()

    def shadow_and_record(self, name, path):
        p, size = imgui.get_window_pos(), imgui.get_window_size()
        radius = imgui.internal.get_current_window().window_rounding / self.s
        dl = imgui.get_window_draw_list()
        dl.push_clip_rect_full_screen()
        for outset, alpha in ((7, 4), (5, 8), (3, 14), (1.5, 20)):
            d = outset * self.s
            dl.add_rect(
                (p.x - d, p.y - d + self.s),
                (p.x + size.x + d, p.y + size.y + d + self.s),
                color((0, 0, 0, alpha)),
                rounding=(radius + outset) * self.s,
                thickness=2 * self.s,
            )
        dl.pop_clip_rect()
        record = {
            "name": name,
            "path": list(path),
            "rect": [p.x, p.y, size.x, size.y],
            "scroll_max": imgui.get_scroll_max_y(),
            "radius": radius,
            "rows": [],
        }
        self.observed["/".join(path)] = record
        return record

    def content(self, name, path):
        from mojive.design.native.geometry import Rect
        from mojive.ui.workspace.menu_rows import menu_columns
        from mojive.ui.workspace.popup_style import popup_box

        record = self.shadow_and_record(name, path)
        ui = self.ui
        rows = self.items(name)
        position, size = imgui.get_window_pos(), imgui.get_window_size()
        box = popup_box(ui, Rect(position.x, position.y, size.x, size.y))
        ui.record_component_box("menu", box, key="menu-container-" + "/".join(path))
        previous = getattr(self, "_content_right", None)
        previous_columns = getattr(self, "_columns", None)
        previous_separator = getattr(self, "_separator_state", None)
        previous_bottom = getattr(self, "_last_row_bottom", None)
        previous_rows = getattr(self, "_local_rows", None)
        previous_measurements = getattr(self, "_local_measurements", None)
        # A named child menu owns its inventory; its commands must not expand
        # the enclosing local popup's next-frame column measurements.
        self._local_rows = self._local_measurements = None
        self._columns = menu_columns(ui, rows)
        self._last_row_bottom = None
        self._separator_state = {
            "count": 0,
            "pending": False,
            "left": box.content.x,
            "right": box.content.right,
            "record": record,
        }
        self._content_right = box.content.right
        imgui.set_cursor_screen_pos((box.content.x, box.content.y))
        imgui.push_clip_rect(
            (box.content.x, box.content.y), (box.content.right, box.content.bottom), True
        )
        try:
            with (
                ui.component_scope("menu"),
                ui.corner_scope(
                    max(0, imgui.internal.get_current_window().window_rounding - box.padding[0]),
                    "menu_item",
                ),
            ):
                # ReferenceMenus reuses this production box and painter.
                if hasattr(self, "local_rows"):
                    self.local_rows(rows, path=path, record=record)
                else:
                    for row in rows:
                        if row is not None:
                            self.item(row, path, record)
        finally:
            imgui.pop_clip_rect()
            self._content_right = previous
            p = imgui.get_cursor_screen_pos()
            bottom = self._last_row_bottom if self._last_row_bottom is not None else p.y
            imgui.set_cursor_screen_pos((position.x, ceil(bottom + box.padding[3] + box.margin[3])))
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
            imgui.dummy((1, 0))
            imgui.pop_style_var()
            self._columns = previous_columns
            self._separator_state = previous_separator
            self._last_row_bottom = previous_bottom
            self._local_rows, self._local_measurements = previous_rows, previous_measurements

    def item(self, row, path, record, *, dispatch=True, key=None, width=None, minimum_height=0):
        from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row

        self._begin_local_row()
        key = key or "menu-item-" + "/".join(path) + "/" + row.label
        submenu = None
        if row.submenu:
            self.reveal((*path, row.submenu))
            self.constrain(row.submenu)

            def submenu(_native_label):
                return imgui.internal.begin_menu_ex(row.label, "     ", row.enabled)

        content_right = getattr(self, "_content_right", None)
        if width is not None:
            content_right = imgui.get_cursor_screen_pos().x + width
        result = paint_menu_row(
            self.ui,
            row,
            key,
            content_right=content_right,
            open_submenu=submenu,
            minimum_height=minimum_height,
            part="select_option" if getattr(self, "_choice_rows", False) else "menu_item",
            columns=getattr(self, "_columns", None),
        )
        if self._local_measurements is not None:
            self._local_measurements.append(result.minimum_width)
        if getattr(self, "_local_rows", None) is not None:
            self._local_rows.append(row)
        record["rows"].append(result.observation)
        if row.submenu and result.activated:
            self.content(row.submenu, (*path, row.submenu))
            imgui.end_menu()
        elif result.activated and dispatch:
            self.actions.append(row.action)
        advance_menu_row(
            self.ui, result, part="select_popup" if getattr(self, "_choice_rows", False) else "menu"
        )
        self._last_row_bottom = result.box.outer.bottom
        return result.activated

    def local_separator(self):
        """Separate nonempty groups without submitting leading or trailing rules."""
        state = getattr(self, "_separator_state", None)
        if state is not None and state["count"]:
            state["pending"] = True

    def _begin_local_row(self):
        """Keep every row and group rule inside the popup's content budget."""
        state = getattr(self, "_separator_state", None)
        if state is None:
            return
        position = imgui.get_cursor_screen_pos()
        if state["pending"]:
            height = native_unit(10 * self.s)
            width = max(0, state["right"] - state["left"])
            y = position.y + height / 2
            self.ui.line(state["left"], y, width)
            inset = min(width / 2, self.ui.metric("separator", "margin", 0) * self.s)
            state["record"].setdefault("dividers", []).append(
                [
                    state["left"] + inset,
                    y,
                    max(0, width - 2 * inset),
                    self.ui.metric("separator", "thickness", 1) * self.s,
                ]
            )
            position.y += height
            state["pending"] = False
        imgui.set_cursor_screen_pos((state["left"], position.y))
        state["count"] += 1

    def local_rows(self, rows, *, path=None, record=None):
        """Append catalog actions to the current shared popup without a second container."""
        path = (
            path
            or getattr(self, "_local_path", None)
            or (imgui.get_current_context().current_window.name,)
        )
        if record is None:
            record = self.observed.setdefault("/".join(path), {"path": list(path), "rows": []})
        for row in rows:
            if row is None:
                self.local_separator()
            else:
                self.item(row, path, record)

    def local_item(
        self,
        label,
        shortcut="",
        checked=None,
        enabled=True,
        *,
        key,
        icon="",
        width=None,
        minimum_height=0,
        checkable=False,
    ):
        """Return native item signals without routing an application command."""
        path = getattr(self, "_local_path", None) or (
            imgui.get_current_context().current_window.name,
        )
        record = self.observed.setdefault("/".join(path), {"path": list(path), "rows": []})
        clicked = self.item(
            Item(
                label,
                icon=icon,
                shortcut=shortcut,
                checked=bool(checked),
                enabled=enabled,
                checkable=checkable or checked is not None,
            ),
            path,
            record,
            dispatch=False,
            key=key,
            width=width,
            minimum_height=minimum_height,
        )
        return clicked, bool(checked) != clicked

    @contextmanager
    def local_submenu(self, label, *, key, enabled=True, icon="", content=True):
        """Use the same row painter while retaining native nested-menu navigation."""
        from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row

        row = Item(label, icon=icon, enabled=enabled, submenu=key)
        self._begin_local_row()
        if content:
            from mojive.ui.workspace.popup_style import prepare_menu_popup

            prepare_menu_popup(self.ui, auto_minimum_width=self._popup_widths.get(key, 0))

        def begin(native_label):
            return imgui.internal.begin_menu_ex(native_label, "     ", enabled)

        result = paint_menu_row(
            self.ui,
            row,
            key,
            content_right=getattr(self, "_content_right", None),
            open_submenu=begin,
            columns=getattr(self, "_columns", None),
        )
        if self._local_measurements is not None:
            self._local_measurements.append(result.minimum_width)
        if getattr(self, "_local_rows", None) is not None:
            self._local_rows.append(row)
        path = getattr(self, "_local_path", None) or (
            imgui.get_current_context().current_window.name,
        )
        record = self.observed.setdefault("/".join(path), {"path": list(path), "rows": []})
        record["rows"].append(result.observation)
        if result.activated:
            try:
                with self.local_content(key) if content else nullcontext():
                    yield True
            finally:
                imgui.end_menu()
        else:
            yield False
        advance_menu_row(self.ui, result)
        self._last_row_bottom = result.box.outer.bottom

    @contextmanager
    def local_content(self, key, *, choices=False):
        from mojive.ui.workspace.menu_rows import menu_columns, menu_row_size
        from mojive.ui.workspace.popup_style import menu_popup_content

        previous_right = getattr(self, "_content_right", None)
        previous_path = getattr(self, "_local_path", None)
        previous_measurements = self._local_measurements
        previous_choices = getattr(self, "_choice_rows", False)
        previous_columns = getattr(self, "_columns", None)
        previous_rows = getattr(self, "_local_rows", None)
        previous_separator = getattr(self, "_separator_state", None)
        previous_bottom = getattr(self, "_last_row_bottom", None)
        self._choice_rows = choices
        self._local_measurements = []
        self._local_rows = []
        self._last_row_bottom = None
        part = "select_popup" if choices else "menu"
        with menu_popup_content(self.ui, key, part=part) as box:
            item_part = "select_option" if choices else "menu_item"
            previous_inventory = getattr(self, "_popup_rows", {}).get(key)
            self._columns = (
                menu_columns(self.ui, previous_inventory, part=item_part)
                if previous_inventory
                else None
            )
            self._content_right, self._local_path = box.content.right, (key,)
            self.observed[key] = {
                "path": [key],
                "rect": list(box.frame.tuple()),
                "rows": [],
                "radius": imgui.internal.get_current_window().window_rounding / self.s,
            }
            self._separator_state = {
                "count": 0,
                "pending": False,
                "left": box.content.x,
                "right": box.content.right,
                "record": self.observed[key],
            }
            try:
                yield box
            finally:
                padding, margin = box.padding, box.margin
                inventory = tuple(self._local_rows)
                columns = menu_columns(self.ui, inventory, part=item_part)
                measured = max(
                    (
                        menu_row_size(self.ui, row, part=item_part, columns=columns)[0]
                        for row in inventory
                    ),
                    default=0,
                )
                self._remember_popup(
                    key,
                    inventory,
                    ceil(measured + padding[0] + padding[2] + margin[0] + margin[2]),
                )
                self._local_measurements = previous_measurements
                self._choice_rows = previous_choices
                self._content_right, self._local_path = previous_right, previous_path
                self._columns, self._local_rows = previous_columns, previous_rows
                self._separator_state = previous_separator
                self._last_row_bottom = previous_bottom

    def _remember_popup(self, key, rows, width):
        """Retain recent layouts without keeping deleted entities or bookmarks alive."""
        self._popup_widths.pop(key, None)
        self._popup_rows.pop(key, None)
        self._popup_widths[key] = width
        self._popup_rows[key] = rows
        if len(self._popup_widths) > _POPUP_CACHE_LIMIT:
            oldest = next(iter(self._popup_widths))
            del self._popup_widths[oldest]
            del self._popup_rows[oldest]

    @contextmanager
    def local_popup(self, key, *, context_item=False):
        """Own a local popup's style and native lifetime; callers own its actions."""
        if context_item:
            imgui.open_popup_on_item_click(key, imgui.PopupFlags_.mouse_button_right)
        if not imgui.is_popup_open(key):
            yield False
            return
        with self.style():
            from mojive.ui.workspace.popup_style import prepare_menu_popup

            prepare_menu_popup(
                self.ui,
                auto_minimum_width=self._popup_widths.get(key, 0),
            )
            opened = imgui.begin_popup(key)
            if opened:
                try:
                    with self.local_content(key):
                        yield True
                finally:
                    imgui.end_popup()
            else:
                yield False

    def dispatch(self, action):
        return menu_actions.dispatch(self, action)

    def dialogs(self):
        return menu_dialogs.dialogs(self)

    def color_control(self, entity, x, y, width):
        return menu_color.color_control(self, entity, x, y, width)

    def finish_color_edit(self):
        return menu_color.finish_color_edit(self)
