"""Persistent in-editor runtime and command output."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ..messages import OutputMessage
from . import Panel, PanelContext

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

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        output = ctx.output
        if output is None:
            imgui.text_disabled("output is unavailable")
            return

        all_entries = output.entries()
        available = imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        level_width = min(190.0 * ctx.style_scale, max(125.0 * ctx.style_scale, available * 0.38))
        imgui.set_next_item_width(max(80.0 * ctx.style_scale, available - level_width - spacing))
        changed, self._filter_text = imgui.input_text_with_hint(
            "##output-filter",
            ctx.tr("Filter text or component..."),
            self._filter_text,
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
        if imgui.small_button(ctx.tr(copy_label)):
            imgui.set_clipboard_text(output.copy_text(entries))
        imgui.same_line()
        if imgui.small_button(ctx.tr("Clear")):
            output.clear()
            all_entries = ()
            entries = ()
            self._filtered_entries = ()
            self._filter_cache_key = None
        imgui.same_line()
        if filtering:
            imgui.text_disabled(f"{len(entries)} / {len(all_entries)} {ctx.tr('messages')}")
        else:
            imgui.text_disabled(f"{len(entries)} {ctx.tr('messages')}")

        imgui.separator()
        imgui.begin_child("output_messages", imgui.ImVec2(0.0, 0.0), 0)
        clipper = imgui.ListClipper()
        clipper.begin(len(entries))
        while clipper.step():
            for index in range(clipper.display_start, clipper.display_end):
                entry = entries[index]
                if imgui.small_button(f"{ctx.tr('Copy')}##output-copy-{entry.sequence}"):
                    imgui.set_clipboard_text(entry.text)
                imgui.set_item_tooltip(ctx.tr("Copy message"))
                imgui.same_line()
                imgui.text_disabled(entry.timestamp)
                imgui.same_line()
                color = _LEVEL_COLORS.get(entry.level, _LEVEL_COLORS["info"])
                imgui.text_colored(imgui.ImVec4(*color), f"[{entry.level.upper()}]")
                imgui.same_line()
                imgui.text(entry.text.replace("\n", "  ↵  "))
                imgui.set_item_tooltip(entry.text)
        clipper.end()
        newest_visible = entries[-1].sequence if entries else 0
        if newest_visible > self._last_sequence:
            imgui.set_scroll_here_y(1.0)
        self._last_sequence = newest_visible
        imgui.end_child()
