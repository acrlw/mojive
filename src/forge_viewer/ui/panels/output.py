"""Persistent in-editor runtime and command output."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import Panel, PanelContext

_LEVEL_COLORS = {
    "debug": (0.58, 0.62, 0.68, 1.0),
    "info": (0.76, 0.80, 0.86, 1.0),
    "success": (0.40, 0.82, 0.52, 1.0),
    "warning": (0.96, 0.68, 0.25, 1.0),
    "error": (1.00, 0.38, 0.34, 1.0),
}


class OutputPanel(Panel):
    name = "Output"
    default_open = True
    shortcut = "F12"
    closable = False
    dock_with = "Stats"

    def __init__(self) -> None:
        super().__init__()
        self._last_sequence = 0

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        output = ctx.output
        if output is None:
            imgui.text_disabled("output is unavailable")
            return

        entries = output.entries()
        if imgui.small_button(ctx.tr("Copy all")):
            imgui.set_clipboard_text(output.copy_text())
        imgui.same_line()
        if imgui.small_button(ctx.tr("Clear")):
            output.clear()
            entries = ()
        imgui.same_line()
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
        newest = entries[-1].sequence if entries else 0
        if newest > self._last_sequence:
            imgui.set_scroll_here_y(1.0)
        self._last_sequence = newest
        imgui.end_child()
