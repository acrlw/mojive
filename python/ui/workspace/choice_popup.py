"""Choice-list presentation shared by native property panels."""

from contextlib import ExitStack

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.controls import search_input
from mojive.ui.workspace.menu_geometry import native_edges
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row
from mojive.ui.workspace.popup_style import POPUP_ROWS, menu_popup_content, menu_popup_style


class ChoicePopup:
    """Present an already opened combo while retaining each panel's edit logic."""

    def __init__(self, ui):
        self.ui = ui
        self._widths = {}
        self._scope = None

    def minimum_width(self, key):
        return self._widths.get(key, 0)

    def begin(self, key):
        self._scope = ExitStack()
        self._scope.enter_context(menu_popup_style(self.ui, part="select_popup"))
        self._box = self._scope.enter_context(menu_popup_content(self.ui, key, part="select_popup"))
        self._key, self._index, self._width = key, 0, 0

    def choice(self, label, selected=False, *, enabled=True, tooltip=""):
        enabled = enabled and not bool(
            imgui.get_current_context().current_item_flags & imgui.ItemFlags_.disabled
        )
        result = paint_menu_row(
            self.ui,
            Item(str(label).split("##", 1)[0], checked=selected, enabled=enabled),
            f"{self._key}/choice/{self._index}",
            content_right=self._box.content.right,
            part="select_option",
        )
        self._index += 1
        self._width = max(self._width, result.minimum_width)
        if tooltip and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            self.ui.tooltip(tooltip, translate=False)
        advance_menu_row(self.ui, result, part="select_popup")
        return result.activated, not selected if result.activated else selected

    def filter(self, key, value, *, hint, clear_tooltip):
        """Keep the filter and its clear action inside the popup content budget."""
        content = self._box.content
        pos = imgui.get_cursor_screen_pos()
        self.ui.line(content.x, pos.y, content.width)
        gap = self.ui.metric("select_popup", "rowGap", 6) * self.ui.s
        line_height = self.ui.metric("separator", "thickness", 1) * self.ui.s
        imgui.set_cursor_screen_pos((content.x, pos.y + line_height + gap))
        imgui.set_next_item_width(content.width)
        result = search_input(
            key, value, hint=hint, search_tooltip=hint, clear_tooltip=clear_tooltip
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        current = POPUP_ROWS.get()
        if current is not None:
            current[1].extend(
                (
                    (Rect(content.x, pos.y, content.width, line_height), gap),
                    (Rect(lo.x, lo.y, hi.x - lo.x, hi.y - lo.y), gap),
                )
            )
        imgui.set_cursor_screen_pos((content.x, hi.y + gap))
        return result

    def end(self, *, close_combo=True):
        style = self.ui.component_style("select_popup")
        padding, margin = (
            (max(native_edges(style, "padding", self.ui.s)),) * 4,
            native_edges(style, "margin", self.ui.s),
        )
        self._widths[self._key] = self._width + padding[0] + padding[2] + margin[0] + margin[2]
        try:
            self._scope.close()
        finally:
            self._scope = None
            if close_combo:
                imgui.end_combo()
