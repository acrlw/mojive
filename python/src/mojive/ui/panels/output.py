"""Persistent in-editor runtime and command output."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ..draw2d import ImguiDraw2D, fit_text, text_line_y
from ..messages import OutputMessage
from ..pointer_bindings import PointerAction
from ..theme import ROW_PADDING_X, ROW_PADDING_Y
from . import Panel, PanelContext, button_width, pointer_pressed, search_input
from .filters import filter_pills, severity_color, severity_group, severity_icon

_LEVEL_RANKS = {
    "debug": 10,
    "info": 20,
    "success": 25,
    "warning": 30,
    "error": 40,
    "critical": 50,
    "fatal": 50,
    "warn": 30,
}


def filter_output_entries(
    entries: tuple[OutputMessage, ...],
    query: str,
    minimum_rank: int = 0,
    *,
    levels: frozenset[str] | None = None,
) -> tuple[OutputMessage, ...]:
    """Return entries matching every text token and the selected severity threshold."""

    tokens = query.casefold().split()
    matched: list[OutputMessage] = []
    for entry in entries:
        if levels is not None and severity_group(entry.level) not in levels:
            continue
        if _LEVEL_RANKS.get(entry.level, _LEVEL_RANKS["info"]) < minimum_rank:
            continue
        searchable = f"{entry.timestamp} {entry.level} {entry.text}".casefold()
        if all(token in searchable for token in tokens):
            matched.append(entry)
    return tuple(matched)


class OutputPanel(Panel):
    id = "output"
    name = "Output"
    default_open = True
    shortcut = "F12"
    closable = False
    dock_with = "Stats"

    def __init__(self) -> None:
        super().__init__()
        self.collapsed = False
        self._last_sequence = 0
        self._filter_text = ""
        self._levels = {"info", "warning", "error"}
        self._filter_cache_key: tuple[int, int, str, frozenset[str]] | None = None
        self._filtered_entries: tuple[OutputMessage, ...] = ()
        self._selected_sequences: set[int] = set()
        self._selection_anchor = 0
        self._count_key = None
        self._level_counts = {"info": 0, "warning": 0, "error": 0}

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def _counts(self, ctx):
        all_entries = ctx.output.entries()
        newest_sequence = all_entries[-1].sequence if all_entries else 0
        count_key = (len(all_entries), newest_sequence)
        if count_key != self._count_key:
            self._level_counts = dict.fromkeys(("info", "warning", "error"), 0)
            for entry in all_entries:
                self._level_counts[severity_group(entry.level)] += 1
            self._count_key = count_key
        return all_entries, newest_sequence

    def draw_collapsed(self, ctx: PanelContext) -> None:
        entries, _ = self._counts(ctx)
        width = imgui.get_content_region_avail().x
        imgui.begin_group()
        clicked = filter_pills(
            ctx,
            "output-level",
            tuple(
                (level, str(min(999, count)), level) for level, count in self._level_counts.items()
            ),
            self._levels,
            compact=True,
        )
        imgui.end_group()
        if clicked is not None:
            self._levels = {clicked}
            self.collapsed = False
        used = imgui.get_item_rect_size().x
        button = imgui.get_frame_height()
        gap = imgui.get_style().item_spacing.x
        if width > used + button + 3 * gap:
            imgui.same_line()
            space = width - used - button - 3 * gap
            latest = next(
                (
                    entry
                    for entry in reversed(entries)
                    if severity_group(entry.level) in self._levels
                ),
                None,
            )
            draw = ImguiDraw2D()
            pos = imgui.get_cursor_screen_pos()
            if latest is not None:
                draw.text(
                    (pos.x, pos.y + imgui.get_style().frame_padding.y),
                    ctx.theme.text,
                    fit_text(
                        draw,
                        f"{latest.timestamp}  {latest.text.splitlines()[0] if latest.text else ''}",
                        space,
                    ),
                )
            imgui.dummy((space, button))
            imgui.same_line()
        if imgui.button("##output-expand", (button, button)):
            self.collapsed = False
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        # Rotate the shared chevron without changing its cached geometry.
        from ..viewport_widgets import expand_glyph_path

        center = ((a.x + b.x) / 2, (a.y + b.y) / 2)
        draw = ImguiDraw2D()
        draw.fringed_concave_fill(
            tuple(
                (center[0] - x * ctx.style_scale, center[1] - y * ctx.style_scale)
                for x, y in expand_glyph_path(1.46, draw.corner_smoothing)
            ),
            ctx.theme.text,
        )
        imgui.set_item_tooltip(ctx.tr("Expand Output"))

    def draw(self, ctx: PanelContext) -> None:
        output = ctx.output
        if output is None:
            imgui.text_disabled(ctx.tr("output is unavailable"))
            return

        if self.collapsed:
            self.draw_collapsed(ctx)
            return
        all_entries, newest_sequence = self._counts(ctx)
        available = imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        clear_label = ctx.tr("Clear")
        clear_width = button_width(clear_label) + imgui.get_frame_height() + spacing
        imgui.begin_group()
        clicked = filter_pills(
            ctx,
            "output-level",
            tuple(
                (level, str(min(999, count)), level) for level, count in self._level_counts.items()
            ),
            self._levels,
            compact=True,
        )
        imgui.end_group()
        filters_width = imgui.get_item_rect_size().x
        if available >= filters_width + clear_width + 3 * spacing + 120 * ctx.style_scale:
            imgui.same_line()
        available = imgui.get_content_region_avail().x
        inline = available >= clear_width + spacing + 90.0 * ctx.style_scale
        imgui.set_next_item_width(max(1.0, available - clear_width - spacing) if inline else -1)
        changed, self._filter_text = search_input(
            "##output-filter",
            self._filter_text,
            hint=ctx.tr("Filter text or component..."),
            search_tooltip=ctx.tr("Search output"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        if changed:
            self._last_sequence = 0
        if inline:
            imgui.same_line()
        if imgui.button(clear_label):
            output.clear()
            all_entries = ()
            newest_sequence = 0
            self._selected_sequences.clear()
            self._selection_anchor = 0
        if clicked is not None:
            self._levels.symmetric_difference_update({clicked})
            self._last_sequence = 0
        if imgui.get_content_region_avail().x >= imgui.get_frame_height() + spacing:
            imgui.same_line()
        if imgui.button("##output-collapse", (imgui.get_frame_height(), imgui.get_frame_height())):
            self.collapsed = True
        from ..viewport_widgets import draw_expand_glyph

        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_expand_glyph(
            ImguiDraw2D(), ((a.x + b.x) / 2, (a.y + b.y) / 2), ctx.theme.text, ctx.style_scale
        )
        imgui.set_item_tooltip(ctx.tr("Collapse Output"))
        cache_key = (
            len(all_entries),
            newest_sequence,
            self._filter_text.casefold(),
            frozenset(self._levels),
        )
        if cache_key != self._filter_cache_key:
            self._filtered_entries = filter_output_entries(
                all_entries, self._filter_text, levels=frozenset(self._levels)
            )
            self._filter_cache_key = cache_key
            self._selected_sequences.intersection_update(entry.sequence for entry in all_entries)
        entries = self._filtered_entries
        filtering = bool(self._filter_text.strip()) or len(self._levels) < 3

        imgui.separator()
        imgui.begin_child("output_messages", imgui.ImVec2(0.0, 0.0), 0)
        row_height = imgui.get_font_size() + 2.0 * ROW_PADDING_Y * ctx.style_scale
        text_offset = text_line_y(ImguiDraw2D(), row_height * 0.5)
        padding_x = ROW_PADDING_X * ctx.style_scale
        clipper = imgui.ListClipper()
        clipper.begin(len(entries))
        while clipper.step():
            for index in range(clipper.display_start, clipper.display_end):
                entry = entries[index]
                row = f"{entry.timestamp}  {entry.text.replace(chr(10), '  ↵  ')}"
                start = imgui.get_cursor_screen_pos()
                imgui.invisible_button(
                    f"##output-row-{entry.sequence}",
                    imgui.ImVec2(imgui.get_content_region_avail().x, row_height),
                )
                hovered = imgui.is_item_hovered()
                row_lo, row_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                select = hovered and pointer_pressed(ctx, PointerAction.OUTPUT_SELECT)
                extend = hovered and pointer_pressed(ctx, PointerAction.OUTPUT_RANGE)
                add = hovered and pointer_pressed(ctx, PointerAction.OUTPUT_ADD)
                if select or extend or add:
                    if extend and self._selection_anchor:
                        by_sequence = {item.sequence: offset for offset, item in enumerate(entries)}
                        anchor = by_sequence.get(self._selection_anchor, index)
                        lo, hi = sorted((anchor, index))
                        self._selected_sequences.update(
                            item.sequence for item in entries[lo : hi + 1]
                        )
                    elif add:
                        if entry.sequence in self._selected_sequences:
                            self._selected_sequences.remove(entry.sequence)
                        else:
                            self._selected_sequences.add(entry.sequence)
                        self._selection_anchor = entry.sequence
                    else:
                        self._selected_sequences = {entry.sequence}
                        self._selection_anchor = entry.sequence
                if hovered and pointer_pressed(ctx, PointerAction.OUTPUT_CONTEXT):
                    imgui.open_popup(f"##output-context-{entry.sequence}")
                if imgui.begin_popup(f"##output-context-{entry.sequence}"):
                    if entry.sequence not in self._selected_sequences:
                        self._selected_sequences = {entry.sequence}
                        self._selection_anchor = entry.sequence
                    selected = tuple(
                        item for item in entries if item.sequence in self._selected_sequences
                    )
                    copy_message, _ = imgui.menu_item(ctx.tr("Copy message"), "Ctrl+C", False)
                    copy_full, _ = imgui.menu_item(ctx.tr("Copy full entry"), "Ctrl+Shift+C", False)
                    if entry.copy_text is not None:
                        copy_path, _ = imgui.menu_item(ctx.tr("Copy path"), "", False)
                        if copy_path:
                            imgui.set_clipboard_text(entry.copy_text)
                    imgui.separator()
                    copy_scope = "Copy shown" if filtering else "Copy all"
                    copy_scope, _ = imgui.menu_item(ctx.tr(copy_scope), "", False)
                    if copy_message:
                        imgui.set_clipboard_text(output.copy_message_text(selected))
                    if copy_full:
                        imgui.set_clipboard_text(output.copy_text(selected))
                    if copy_scope:
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
                        imgui.get_style().frame_rounding,
                    )
                color = ctx.theme.text
                icon_size = imgui.get_font_size()
                severity_icon(
                    ImguiDraw2D(),
                    (start.x + padding_x + icon_size * 0.5, start.y + row_height * 0.5),
                    icon_size,
                    severity_group(entry.level),
                    severity_color(ctx.theme, entry.level),
                )
                imgui.push_clip_rect(
                    (row_lo.x + padding_x, row_lo.y),
                    (max(row_lo.x + padding_x, row_hi.x - padding_x), row_hi.y),
                    True,
                )
                imgui.get_window_draw_list().add_text(
                    imgui.ImVec2(
                        start.x + padding_x + icon_size + 6.0 * ctx.style_scale,
                        round(start.y + text_offset),
                    ),
                    imgui.color_convert_float4_to_u32(imgui.ImVec4(*color)),
                    row,
                )
                imgui.pop_clip_rect()
        clipper.end()
        io = imgui.get_io()
        if (
            imgui.is_window_focused(imgui.FocusedFlags_.root_and_child_windows)
            and not io.want_text_input
            and (io.key_ctrl or io.key_super)
            and imgui.is_key_pressed(imgui.Key.c, False)
            and self._selected_sequences
        ):
            selected = tuple(
                entry for entry in entries if entry.sequence in self._selected_sequences
            )
            if io.key_shift:
                imgui.set_clipboard_text(output.copy_text(selected))
            else:
                imgui.set_clipboard_text(output.copy_message_text(selected))
        newest_visible = entries[-1].sequence if entries else 0
        if newest_visible > self._last_sequence:
            imgui.set_scroll_here_y(1.0)
        self._last_sequence = newest_visible
        imgui.end_child()
