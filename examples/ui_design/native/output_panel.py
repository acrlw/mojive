"""Native output chrome around the shared diagnostic history and selection."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui.files import message_path, reveal_path
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.panels import publish_status_hint, search_input
from mojive.ui.panels.filters import severity_color
from mojive.ui.panels.output import OutputPanel
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets import pointer_tool_hint
from mojive.ui.workspace.widgets import MUTED


class ReferenceOutput(OutputPanel):
    """Keep native controls compact while retaining production output behavior."""

    def __init__(self, study):
        super().__init__(multiline=True)
        self.study = study
        self.show_search = False
        self._focus_search = False

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

    def _draw_toolbar(self, ctx, all_entries, newest_sequence):
        u, s = self.study.ui, self.study.s
        pos = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        levels = (
            ("info", ctx.tr("Info"), "info"),
            ("warning", ctx.tr("Warning"), "warn"),
            ("error", ctx.tr("Error"), "err"),
        )
        levels = tuple(
            (*item, tuple(round(channel * 255) for channel in severity_color(ctx.theme, item[0])))
            for item in levels
        )
        labels = tuple(f"{label} {self._level_counts[level]}" for level, label, _, _ in levels)
        level_widths = tuple(max(70, u.measure(label) / s + 40) for label in labels)
        compact_actions = width < 196 * s
        actions_width = (3 * 28 + 2 * 4 if compact_actions else 5 * 28 + 4 * 4) * s
        expanded_levels = width >= sum(level_widths) * s + actions_width + 24 * s
        if expanded_levels:
            x = pos.x
            for (level, label, icon, color), shown, button_width in zip(
                levels, labels, level_widths, strict=True
            ):
                if u.button(
                    f"output-level-{level}",
                    x,
                    pos.y,
                    button_width,
                    icon=icon,
                    label=shown,
                    active=level in self._levels,
                    icon_color=color,
                    tooltip=label,
                ):
                    self._levels.symmetric_difference_update({level})
                    self._last_sequence = 0
                x += (button_width + 4) * s
        elif width >= actions_width + 36 * s:
            if u.button(
                "output-levels",
                pos.x,
                pos.y,
                icon="filter",
                active=len(self._levels) < 3,
                tooltip=ctx.tr("Severity"),
            ):
                imgui.open_popup("##output-levels")
            if imgui.begin_popup("##output-levels"):
                self._draw_level_menu(levels, labels)
                imgui.end_popup()
        x = pos.x + width - actions_width
        if u.button(
            "output-search",
            x,
            pos.y,
            icon="search",
            active=self.show_search,
            tooltip=ctx.tr("Search output"),
        ):
            self._toggle_search()
        visible = self._filter_entries(all_entries, newest_sequence)
        selected = any(entry.sequence in self._selected_sequences for entry in visible)
        copy_label = ctx.tr("Copy selection" if selected else "Copy shown")
        if compact_actions:
            if u.button(
                "output-more",
                x + 32 * s,
                pos.y,
                icon="more",
                tooltip=ctx.tr("Output actions"),
            ):
                imgui.open_popup("##output-actions")
            if imgui.begin_popup("##output-actions"):
                copy, _ = imgui.menu_item(copy_label, "", False, bool(visible))
                follow, _ = imgui.menu_item(ctx.tr("Auto-scroll"), "", self.auto_scroll)
                clear, _ = imgui.menu_item(ctx.tr("Clear output"), "", False, bool(all_entries))
                if copy:
                    imgui.set_clipboard_text(self._copy_text(ctx.output))
                if follow:
                    self._toggle_auto_scroll()
                if clear:
                    self._clear_output(ctx.output)
                    all_entries, newest_sequence = (), 0
                if width < actions_width + 36 * s:
                    imgui.separator()
                    self._draw_level_menu(levels, labels)
                imgui.end_popup()
        else:
            if u.button(
                "output-copy",
                x + 32 * s,
                pos.y,
                icon="copy",
                enabled=bool(visible),
                tooltip=copy_label,
            ):
                imgui.set_clipboard_text(self._copy_text(ctx.output))
            if u.button(
                "output-follow",
                x + 64 * s,
                pos.y,
                icon="follow",
                active=self.auto_scroll,
                tooltip=ctx.tr("Auto-scroll"),
            ):
                self._toggle_auto_scroll()
            if u.button(
                "output-clear",
                x + 96 * s,
                pos.y,
                icon="delete",
                enabled=bool(all_entries),
                tooltip=ctx.tr("Clear output"),
            ):
                self._clear_output(ctx.output)
                all_entries, newest_sequence = (), 0
        if u.button(
            "collapse-output",
            x + (64 if compact_actions else 128) * s,
            pos.y,
            icon="down",
            tooltip=ctx.tr("Collapse Output"),
        ):
            self.study.collapse_bottom()
        imgui.set_cursor_screen_pos((pos.x, pos.y + 36 * s))
        if self.show_search:
            if self._focus_search:
                imgui.set_keyboard_focus_here()
                self._focus_search = False
            imgui.set_next_item_width(-1)
            changed, self._filter_text = search_input(
                "##output-filter",
                self._filter_text,
                hint=ctx.tr("Search output"),
                search_tooltip=ctx.tr("Search output"),
                clear_tooltip=ctx.tr("Clear search"),
                icon_drawer=self._draw_search_icon,
            )
            if changed:
                self._last_sequence = 0
        return all_entries, newest_sequence

    def _draw_level_menu(self, levels, labels):
        for (level, _, _, _), label in zip(levels, labels, strict=True):
            changed, _ = imgui.menu_item(label, "", level in self._levels)
            if changed:
                self._levels.symmetric_difference_update({level})
                self._last_sequence = 0

    def _draw_search_icon(self, draw, center, size, name, color):
        u, s = self.study.ui, self.study.s
        size = min(size, 28 * s)
        u.role_icon(
            "x" if name == "clear" else name,
            center[0] - size * 0.5,
            center[1] - size * 0.5,
            size,
            size,
            role="inline",
            c=MUTED,
        )

    def _draw_entries(self, ctx, entries, filtering):
        imgui.push_font(self.study.ui.mono, 14 * self.study.s)
        try:
            super()._draw_entries(ctx, entries, filtering)
        finally:
            imgui.pop_font()

    def _draw_file_actions(self, ctx, entry):
        path = message_path(entry.text, entry.copy_text)
        if path is None:
            return
        if entry.copy_text is None:
            clicked, _ = imgui.menu_item(ctx.tr("Copy path"), "", False)
            if clicked:
                imgui.set_clipboard_text(str(path))
        clicked, _ = imgui.menu_item(ctx.tr("Reveal file"), "", False)
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
