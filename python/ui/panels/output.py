"""Persistent in-editor runtime and command output."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.adapters.base import FrameNeeds
from mojive.ui.files import message_path, reveal_path
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.messages import OutputMessage
from mojive.ui.panels import Panel, PanelContext, button_width, pointer_pressed, search_input
from mojive.ui.panels.filters import filter_pills, severity_color, severity_group, severity_icon
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.text_layout import fit_text, text_line_y
from mojive.ui.theme import ROW_PADDING_X, ROW_PADDING_Y


@dataclass(frozen=True)
class OutputRowStyle:
    """Physical row geometry supplied by a host without replacing log interaction."""

    height: float
    line_height: float
    icon_size: float
    gap: float
    padding: tuple[float, float, float, float]
    margin: tuple[float, float, float, float] = (0, 0, 0, 0)
    text_align: str = "start"
    align_items: str = "center"


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

    def __init__(self, *, search_icon_drawer=None, multiline=False, auto_scroll=True) -> None:
        super().__init__()
        self.search_icon_drawer = search_icon_drawer
        self.multiline = bool(multiline)
        self.auto_scroll = bool(auto_scroll)
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
        self._row_layout_key = None
        self._wrapped_rows = ()

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
            imgui.invisible_button("##output-latest", (space, button))
            if latest is not None and imgui.is_item_hovered():
                path = message_path(latest.text, latest.copy_text)
                if path is not None:
                    imgui.set_item_tooltip(ctx.tr("Ctrl+click to reveal file"))
                    if pointer_pressed(ctx, PointerAction.OUTPUT_ADD):
                        reveal_path(path)
            imgui.same_line()
        if imgui.button("##output-expand", (button, button)):
            self.collapsed = False
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        # Rotate the shared chevron without changing its cached geometry.
        from mojive.ui.viewport_widgets import expand_glyph_path

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
        all_entries, newest_sequence = self._draw_toolbar(ctx, all_entries, newest_sequence)
        entries = self._filter_entries(all_entries, newest_sequence)
        filtering = bool(self._filter_text.strip()) or len(self._levels) < 3
        self._draw_entries(ctx, entries, filtering)

    def _clear_output(self, output) -> None:
        output.clear()
        self._selected_sequences.clear()
        self._selection_anchor = 0
        self._last_sequence = 0

    def _draw_toolbar(self, ctx, all_entries, newest_sequence):
        output = ctx.output
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
            icon_drawer=self.search_icon_drawer,
        )
        if changed:
            self._last_sequence = 0
        if inline:
            imgui.same_line()
        if imgui.button(clear_label):
            self._clear_output(output)
            all_entries = ()
            newest_sequence = 0
        if clicked is not None:
            self._levels.symmetric_difference_update({clicked})
            self._last_sequence = 0
        if imgui.get_content_region_avail().x >= imgui.get_frame_height() + spacing:
            imgui.same_line()
        if imgui.button("##output-collapse", (imgui.get_frame_height(), imgui.get_frame_height())):
            self.collapsed = True
        from mojive.ui.viewport_widgets import draw_expand_glyph

        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_expand_glyph(
            ImguiDraw2D(), ((a.x + b.x) / 2, (a.y + b.y) / 2), ctx.theme.text, ctx.style_scale
        )
        imgui.set_item_tooltip(ctx.tr("Collapse Output"))
        return all_entries, newest_sequence

    def _filter_entries(self, all_entries, newest_sequence=None):
        if newest_sequence is None:
            newest_sequence = all_entries[-1].sequence if all_entries else 0
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
        return self._filtered_entries

    def row_style(self, ctx):
        scale = ctx.style_scale
        return OutputRowStyle(
            imgui.get_font_size() + 2 * ROW_PADDING_Y * scale,
            imgui.get_font_size(),
            imgui.get_font_size(),
            6 * scale,
            (
                ROW_PADDING_X * scale,
                ROW_PADDING_Y * scale,
                ROW_PADDING_X * scale,
                ROW_PADDING_Y * scale,
            ),
        )

    def record_row(self, sequence, outer, frame, content, **geometry):
        """Expose real row geometry to an optional host design audit."""

    def _draw_separator(self, ctx):
        imgui.separator()

    def _draw_entries(self, ctx, entries, filtering):
        output = ctx.output
        self._draw_separator(ctx)
        imgui.begin_child("output_messages", imgui.ImVec2(0.0, 0.0), 0)
        geometry = self.row_style(ctx)
        self._active_row_style = geometry
        ml, mt, mr, mb = geometry.margin
        pl, pt, pr, pb = geometry.padding
        row_height = (
            max(geometry.height, max(geometry.line_height, geometry.icon_size) + pt + pb) + mt + mb
        )
        text_offset = text_line_y(ImguiDraw2D(), (row_height - mt - mb) * 0.5)
        padding_x = pl + ml
        if self.multiline:
            width = imgui.get_content_region_avail().x
            wrap_width = max(1.0, width - ml - mr - pl - pr - geometry.icon_size - geometry.gap)
            key = (
                entries,
                wrap_width,
                imgui.get_font_size(),
                ctx.style_scale,
                ctx.language,
                geometry,
            )
            if key != self._row_layout_key:
                self._wrapped_rows = tuple(
                    (
                        row := f"{entry.timestamp}  {ctx.tr(entry.text)}",
                        max(
                            row_height,
                            imgui.calc_text_size(row, wrap_width=wrap_width).y + pt + pb + mt + mb,
                        ),
                    )
                    for entry in entries
                )
                self._row_layout_key = key
            for index, (row, height) in enumerate(self._wrapped_rows):
                if imgui.is_rect_visible(imgui.ImVec2(width, height)):
                    self._draw_entry(
                        ctx,
                        entries,
                        index,
                        row,
                        height,
                        row_height,
                        text_offset,
                        padding_x,
                        filtering,
                        wrap_width,
                    )
                else:
                    imgui.dummy(imgui.ImVec2(width, height))
        else:
            clipper = imgui.ListClipper()
            clipper.begin(len(entries))
            while clipper.step():
                for index in range(clipper.display_start, clipper.display_end):
                    entry = entries[index]
                    row = f"{entry.timestamp}  {ctx.tr(entry.text).replace(chr(10), '  ↵  ')}"
                    self._draw_entry(
                        ctx,
                        entries,
                        index,
                        row,
                        row_height,
                        row_height,
                        text_offset,
                        padding_x,
                        filtering,
                    )
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
        if self.auto_scroll and newest_visible > self._last_sequence:
            imgui.set_scroll_here_y(1.0)
        self._last_sequence = newest_visible
        imgui.end_child()

    @contextmanager
    def _context_popup(self, key):
        opened = imgui.begin_popup(key)
        try:
            yield opened
        finally:
            if opened:
                imgui.end_popup()

    def _context_item(self, label, shortcut="", checked=False, enabled=True, *, key):
        return imgui.menu_item(label + "###" + key, shortcut, checked, enabled)

    def _draw_file_actions(self, ctx, entry) -> None:
        """Allow a host to add file actions to the shared row context menu."""

    def _draw_entry(
        self,
        ctx,
        entries,
        index,
        row,
        row_height,
        first_line_height,
        text_offset,
        padding_x,
        filtering,
        wrap_width=0,
    ):
        output = ctx.output
        entry = entries[index]
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
        path = message_path(entry.text, entry.copy_text) if hovered else None
        if path is not None:
            imgui.set_item_tooltip(ctx.tr("Ctrl+click to reveal file") + "\n" + str(path))
        if add and path is not None:
            reveal_path(path)
        elif select or extend or add:
            if extend and self._selection_anchor:
                by_sequence = {item.sequence: offset for offset, item in enumerate(entries)}
                anchor = by_sequence.get(self._selection_anchor, index)
                lo, hi = sorted((anchor, index))
                self._selected_sequences.update(item.sequence for item in entries[lo : hi + 1])
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
        with self._context_popup(f"##output-context-{entry.sequence}") as opened:
            if opened:
                if entry.sequence not in self._selected_sequences:
                    self._selected_sequences = {entry.sequence}
                    self._selection_anchor = entry.sequence
                selected = tuple(
                    item for item in entries if item.sequence in self._selected_sequences
                )
                copy_message, _ = self._context_item(
                    ctx.tr("Copy message"), "Ctrl+C", key="output-context-copy-message"
                )
                copy_full, _ = self._context_item(
                    ctx.tr("Copy full entry"), "Ctrl+Shift+C", key="output-context-copy-full"
                )
                if entry.copy_text is not None:
                    copy_path, _ = self._context_item(
                        ctx.tr("Copy path"), key="output-context-copy-path"
                    )
                    if copy_path:
                        imgui.set_clipboard_text(entry.copy_text)
                self._draw_file_actions(ctx, entry)
                imgui.separator()
                copy_scope = "Copy shown" if filtering else "Copy all"
                copy_scope, _ = self._context_item(
                    ctx.tr(copy_scope), key="output-context-copy-scope"
                )
                if copy_message:
                    imgui.set_clipboard_text(output.copy_message_text(selected))
                if copy_full:
                    imgui.set_clipboard_text(output.copy_text(selected))
                if copy_scope:
                    imgui.set_clipboard_text(output.copy_text(entries))
        geometry = self._active_row_style
        ml, mt, mr, mb = geometry.margin
        pl, pt, pr, pb = geometry.padding
        frame_lo = imgui.ImVec2(start.x + ml, start.y + mt)
        frame_hi = imgui.ImVec2(max(frame_lo.x, row_hi.x - mr), max(frame_lo.y, row_hi.y - mb))
        content_lo = imgui.ImVec2(frame_lo.x + pl, frame_lo.y + pt)
        content_hi = imgui.ImVec2(
            max(content_lo.x, frame_hi.x - pr), max(content_lo.y, frame_hi.y - pb)
        )
        if hovered or entry.sequence in self._selected_sequences:
            color = (
                ctx.theme.bg_header
                if entry.sequence in self._selected_sequences
                else ctx.theme.bg_frame_hovered
            )
            imgui.get_window_draw_list().add_rect_filled(
                frame_lo,
                frame_hi,
                imgui.color_convert_float4_to_u32(imgui.ImVec4(*color)),
                imgui.get_style().frame_rounding,
            )
        color = ctx.theme.text
        icon_size = geometry.icon_size
        imgui.push_clip_rect(
            content_lo,
            content_hi,
            True,
        )
        content_width = max(0, content_hi.x - content_lo.x - icon_size - geometry.gap)
        text_width = min(content_width, imgui.calc_text_size(row, wrap_width=wrap_width).x)
        horizontal = (
            0.5
            if geometry.text_align == "center"
            else 1
            if geometry.text_align in ("end", "right")
            else 0
        )
        vertical = (
            0.5
            if geometry.align_items == "center"
            else 1
            if geometry.align_items in ("end", "flex-end")
            else 0
        )
        text_height = imgui.calc_text_size(row, wrap_width=wrap_width).y
        first_line = max(geometry.line_height, icon_size)
        block_height = max(text_height, geometry.line_height) + first_line - geometry.line_height
        block_y = content_lo.y + max(0, content_hi.y - content_lo.y - block_height) * vertical
        text_y = block_y + max(0, first_line - imgui.get_font_size()) / 2
        severity_icon(
            ImguiDraw2D(),
            (content_lo.x + icon_size * 0.5, block_y + first_line * 0.5),
            icon_size,
            severity_group(entry.level),
            severity_color(ctx.theme, entry.level),
        )
        text_position = imgui.ImVec2(
            content_lo.x
            + icon_size
            + geometry.gap
            + max(0, content_width - text_width) * horizontal,
            round(text_y),
        )
        text_color = imgui.color_convert_float4_to_u32(imgui.ImVec4(*color))
        if wrap_width:
            imgui.get_window_draw_list().add_text(
                imgui.get_font(),
                imgui.get_font_size(),
                text_position,
                text_color,
                row,
                wrap_width=wrap_width,
            )
        else:
            imgui.get_window_draw_list().add_text(text_position, text_color, row)
        imgui.pop_clip_rect()
        self.record_row(
            entry.sequence,
            (row_lo.x, row_lo.y, row_hi.x - row_lo.x, row_hi.y - row_lo.y),
            (frame_lo.x, frame_lo.y, frame_hi.x - frame_lo.x, frame_hi.y - frame_lo.y),
            (content_lo.x, content_lo.y, content_hi.x - content_lo.x, content_hi.y - content_lo.y),
            fields=[
                [content_lo.x, block_y, icon_size, first_line],
                [text_position.x, block_y, text_width, block_height],
            ],
        )
