"""Persistent in-editor runtime and command output."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ..messages import OutputMessage
from . import Panel, PanelContext, button_row_layout, button_width, search_input

_LEVEL_COLORS = {
    "debug": (0.58, 0.62, 0.68, 1.0),
    "info": (0.76, 0.80, 0.86, 1.0),
    "success": (0.40, 0.82, 0.52, 1.0),
    "warning": (0.96, 0.68, 0.25, 1.0),
    "error": (1.00, 0.38, 0.34, 1.0),
}

_LEVEL_RANKS = {
    "debug": 10,
    "info": 20,
    "success": 25,
    "warning": 30,
    "error": 40,
}

_LEVEL_FILTERS = (
    ("All levels", 0),
    ("Info and above", 20),
    ("Warnings and errors", 30),
    ("Errors only", 40),
)


def filter_output_entries(
    entries: tuple[OutputMessage, ...],
    query: str,
    minimum_rank: int,
) -> tuple[OutputMessage, ...]:
    """Return entries matching every text token and the selected severity threshold."""

    tokens = query.casefold().split()
    matched: list[OutputMessage] = []
    for entry in entries:
        if _LEVEL_RANKS.get(entry.level, _LEVEL_RANKS["info"]) < minimum_rank:
            continue
        searchable = f"{entry.timestamp} {entry.level} {entry.text}".casefold()
        if all(token in searchable for token in tokens):
            matched.append(entry)
    return tuple(matched)


class OutputPanel(Panel):
    name = "Output"
    default_open = True
    shortcut = "F12"
    closable = False
    dock_with = "Stats"

    def __init__(self) -> None:
        super().__init__()
        self._last_sequence = 0
        self._filter_text = ""
        self._level_filter = 0
        self._filter_cache_key: tuple[int, int, str, int] | None = None
        self._filtered_entries: tuple[OutputMessage, ...] = ()
        self._selected_sequences: set[int] = set()
        self._selection_anchor = 0

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        output = ctx.output
        if output is None:
            imgui.text_disabled(ctx.tr("output is unavailable"))
            return

        all_entries = output.entries()
        available = imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        level_width = min(190.0 * ctx.style_scale, max(125.0 * ctx.style_scale, available * 0.38))
        imgui.set_next_item_width(max(80.0 * ctx.style_scale, available - level_width - spacing))
        changed, self._filter_text = search_input(
            "##output-filter",
            self._filter_text,
            hint=ctx.tr("Filter text or component..."),
            search_tooltip=ctx.tr("Search output"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        if changed:
            self._filter_cache_key = None
            self._last_sequence = 0
        imgui.same_line()
        imgui.set_next_item_width(level_width)
        level_labels = tuple(ctx.tr(label) for label, _rank in _LEVEL_FILTERS)
        changed, self._level_filter = imgui.combo(
            "##output-level",
            self._level_filter,
            level_labels,
        )
        if changed:
            self._filter_cache_key = None
            self._last_sequence = 0

        newest_sequence = all_entries[-1].sequence if all_entries else 0
        minimum_rank = _LEVEL_FILTERS[self._level_filter][1]
        cache_key = (len(all_entries), newest_sequence, self._filter_text.casefold(), minimum_rank)
        if cache_key != self._filter_cache_key:
            self._filtered_entries = filter_output_entries(
                all_entries,
                self._filter_text,
                minimum_rank,
            )
            self._filter_cache_key = cache_key
        entries = self._filtered_entries
        filtering = bool(self._filter_text.strip()) or minimum_rank > 0

        copy_label = "Copy shown" if filtering else "Copy all"
        copy_text = ctx.tr(copy_label)
        clear_text = ctx.tr("Clear")
        count_text = (
            f"{len(entries)} / {len(all_entries)} {ctx.tr('messages')}"
            if filtering
            else f"{len(entries)} {ctx.tr('messages')}"
        )
        inline = button_row_layout(
            (
                button_width(copy_text),
                button_width(clear_text),
                float(imgui.calc_text_size(count_text).x),
            ),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        if imgui.small_button(copy_text):
            imgui.set_clipboard_text(output.copy_text(entries))
        if inline[1]:
            imgui.same_line()
        if imgui.small_button(clear_text):
            output.clear()
            all_entries = ()
            entries = ()
            self._filtered_entries = ()
            self._filter_cache_key = None
        if inline[2]:
            imgui.same_line()
        imgui.text_disabled(count_text)

        imgui.separator()
        imgui.begin_child("output_messages", imgui.ImVec2(0.0, 0.0), 0)
        clipper = imgui.ListClipper()
        clipper.begin(len(entries))
        while clipper.step():
            for index in range(clipper.display_start, clipper.display_end):
                entry = entries[index]
                row = (
                    f"{entry.timestamp}  [{entry.level.upper()}]  "
                    f"{entry.text.replace(chr(10), '  ↵  ')}"
                )
                start = imgui.get_cursor_screen_pos()
                row_height = imgui.get_frame_height()
                clicked = imgui.invisible_button(
                    f"##output-row-{entry.sequence}",
                    imgui.ImVec2(imgui.get_content_region_avail().x, row_height),
                )
                hovered = imgui.is_item_hovered()
                row_lo, row_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                if clicked:
                    io = imgui.get_io()
                    if io.key_shift and self._selection_anchor:
                        by_sequence = {item.sequence: offset for offset, item in enumerate(entries)}
                        anchor = by_sequence.get(self._selection_anchor, index)
                        lo, hi = sorted((anchor, index))
                        self._selected_sequences.update(
                            item.sequence for item in entries[lo : hi + 1]
                        )
                    elif io.key_ctrl or io.key_super:
                        if entry.sequence in self._selected_sequences:
                            self._selected_sequences.remove(entry.sequence)
                        else:
                            self._selected_sequences.add(entry.sequence)
                        self._selection_anchor = entry.sequence
                    else:
                        self._selected_sequences = {entry.sequence}
                        self._selection_anchor = entry.sequence
                if imgui.begin_popup_context_item(f"##output-context-{entry.sequence}"):
                    if entry.sequence not in self._selected_sequences:
                        self._selected_sequences = {entry.sequence}
                        self._selection_anchor = entry.sequence
                    copy_message, _ = imgui.menu_item(ctx.tr("Copy"), "Ctrl+C", False)
                    copy_all, _ = imgui.menu_item(ctx.tr("Copy all"), "Ctrl+Shift+C", False)
                    if copy_message:
                        selected = tuple(
                            item for item in entries if item.sequence in self._selected_sequences
                        )
                        imgui.set_clipboard_text(output.copy_text(selected))
                    if copy_all:
                        imgui.set_clipboard_text(output.copy_text(entries))
                    imgui.end_popup()
                if hovered or entry.sequence in self._selected_sequences:
                    color = (
                        ctx.theme.bg_header
                        if entry.sequence in self._selected_sequences
                        else ctx.theme.bg_frame_hovered
                    )
                    imgui.get_window_draw_list().add_rect_filled(
                        row_lo,
                        row_hi,
                        imgui.color_convert_float4_to_u32(imgui.ImVec4(*color)),
                        2.0 * ctx.style_scale,
                    )
                color = _LEVEL_COLORS.get(entry.level, _LEVEL_COLORS["info"])
                text_height = imgui.calc_text_size(row).y
                imgui.get_window_draw_list().add_text(
                    imgui.ImVec2(start.x, start.y + (row_height - text_height) * 0.5),
                    imgui.color_convert_float4_to_u32(imgui.ImVec4(*color)),
                    row,
                )
        clipper.end()
        io = imgui.get_io()
        if (io.key_ctrl or io.key_super) and imgui.is_key_pressed(imgui.Key.c, False):
            if io.key_shift:
                imgui.set_clipboard_text(output.copy_text(entries))
            elif self._selected_sequences:
                selected = tuple(
                    entry for entry in entries if entry.sequence in self._selected_sequences
                )
                imgui.set_clipboard_text(output.copy_text(selected))
        newest_visible = entries[-1].sequence if entries else 0
        if newest_visible > self._last_sequence:
            imgui.set_scroll_here_y(1.0)
        self._last_sequence = newest_visible
        imgui.end_child()
