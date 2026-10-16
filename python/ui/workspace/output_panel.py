"""Native output chrome around the shared diagnostic history and selection."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.ui.files import message_path, reveal_path
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.panels import publish_status_hint
from mojive.ui.panels.output import OutputPanel, OutputRowStyle
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets import pointer_tool_hint
from mojive.ui.workspace.design import DESIGN


class WorkspaceOutput(OutputPanel):
    """Keep native controls compact while retaining production output behavior."""

    def __init__(self, workspace):
        super().__init__(multiline=True)
        self.workspace = workspace
        self.show_search = False
        self._focus_search = False
        self._toolbar_rect = None

    def _toggle_search(self):
        self.show_search = not self.show_search
        self._focus_search = self.show_search
        if not self.show_search:
            self._filter_text = ""
            self._last_sequence = 0

    def _toggle_auto_scroll(self):
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            self._last_sequence = 0

    def _copy_text(self, output):
        entries = self._filter_entries(output.entries())
        selected = tuple(entry for entry in entries if entry.sequence in self._selected_sequences)
        return output.copy_message_text(selected) if selected else output.copy_text(entries)

    @part_scope("output_toolbar")
    def _draw_toolbar(self, ctx, all_entries, newest_sequence):
        from mojive.ui.workspace.output_toolbar import draw_output_toolbar

        start = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        result = draw_output_toolbar(self, ctx, all_entries, newest_sequence)
        self._toolbar_rect = (start.x, start.y, width, imgui.get_cursor_screen_pos().y - start.y)
        return result

    def _draw_level_menu(self, levels, labels):
        for (level, _, _, _), label in zip(levels, labels, strict=True):
            changed, _ = self.workspace.menus.local_item(
                label, checked=level in self._levels, key="output-menu-level-" + level
            )
            if changed:
                self._levels.symmetric_difference_update({level})
                self._last_sequence = 0

    def row_style(self, ctx):
        ui, scale = self.workspace.ui, ctx.style_scale
        style = ui.component_style("output_row")
        return OutputRowStyle(
            style.get("height", 28) * scale,
            max(style.get("fontSize", 14), style.line_height(style.get("fontSize", 14) * 1.4))
            * scale,
            style.get("iconSize", 14) * scale,
            style.get("columnGap", style.get("gap", 6)) * scale,
            style.edges("padding", scale, (8, 4, 8, 4)),
            style.edges("margin", scale),
            style.get("textAlign", "start"),
            style.get("alignItems", "center"),
        )

    def record_row(self, sequence, outer, frame, content, **geometry):
        ui = self.workspace.ui
        if ui.design is not None:
            self._row_rectangles.append(outer)
            ui.design.record(
                "output-row-" + str(sequence),
                ("output_row",),
                frame,
                "output_row",
                outer=list(outer),
                content=list(content),
                **geometry,
            )

    def _draw_separator(self, ctx):
        ui, scale = self.workspace.ui, self.workspace.s
        pos = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        toolbar_gap = ui.metric("output_layout", "toolbarGap", 8) * scale
        list_gap = ui.metric("output_layout", "listGap", 8) * scale
        y = pos.y + toolbar_gap
        color = imgui.get_color_u32(imgui.Col_.separator)
        imgui.get_window_draw_list().add_line((pos.x, y), (pos.x + width, y), color, scale)
        self._layout_gaps = {
            "toolbarGap": [pos.x, pos.y, width, toolbar_gap],
            "listGap": [pos.x, y + scale, width, list_gap],
        }
        self._divider_rect = (pos.x, y, width, scale)
        imgui.set_cursor_screen_pos((pos.x, y + scale + list_gap))
        self._list_rect = (
            pos.x,
            y + scale + list_gap,
            width,
            max(0, imgui.get_content_region_avail().y),
        )

    def _draw_entries(self, ctx, entries, filtering):
        ui, scale = self.workspace.ui, self.workspace.s
        self._row_rectangles = []
        imgui.push_font(
            ui.output_font, ui.component_style("output_row").get("fontSize", 14) * scale
        )
        list_style = ui.component_style("output_list")
        gap = list_style.get("rowGap", list_style.get("gap", DESIGN.output_row_gap))
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, gap * ctx.style_scale)
        )
        try:
            super()._draw_entries(ctx, entries, filtering)
            from mojive.ui.workspace.component_layout import record_group

            record_group(ui, "output_list", "output-entry-list", self._row_rectangles)
            record_group(
                ui,
                "output_layout",
                "output-layout",
                [r for r in (self._toolbar_rect, self._divider_rect, self._list_rect) if r],
                fieldRegions=self._layout_gaps,
            )
        finally:
            imgui.pop_style_var()
            imgui.pop_font()

    def _context_popup(self, key):
        return self.workspace.menus.local_popup(key)

    def _context_item(self, label, shortcut="", checked=False, enabled=True, *, key):
        return self.workspace.menus.local_item(label, shortcut, checked, enabled, key=key)

    def _draw_file_actions(self, ctx, entry):
        path = message_path(entry.text, entry.copy_text)
        if path is None:
            return
        if entry.copy_text is None:
            clicked, _ = self.workspace.menus.local_item(
                "Copy path", key="output-menu-copy-path", icon="copy"
            )
            if clicked:
                imgui.set_clipboard_text(str(path))
        clicked, _ = self.workspace.menus.local_item(
            "Reveal file", key="output-menu-reveal-file", icon="folder"
        )
        if clicked:
            reveal_path(path)

    def finish_frame(self, ctx):
        bindings = ctx.input_bindings or DEFAULT_INPUT_BINDINGS
        for action, label in (
            (PointerAction.OUTPUT_SELECT, "Select entry"),
            (PointerAction.OUTPUT_RANGE, "Select range"),
            (PointerAction.OUTPUT_CONTEXT, "Output actions"),
        ):
            publish_status_hint(
                ctx,
                pointer_tool_hint(
                    action,
                    bindings,
                    ctx.tr(label),
                    hint_id=action.value,
                ),
            )
