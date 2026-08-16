from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import Panel, PanelContext

MOUSE_GESTURES: tuple[tuple[str, str], ...] = (
    ("left drag", "orbit the camera"),
    ("right / middle drag, or Shift + left drag", "pan"),
    ("wheel", "dolly"),
    ("left click", "select the object under the cursor"),
    ("Ctrl + left drag", "push the object; physics reacts (works while running)"),
    ("Ctrl + right drag", "twist the object"),
    ("drag the view balls (top right)", "free rotate; click a ball to snap to that view"),
)

KEYS: tuple[tuple[str, str], ...] = (
    ("Space", "pause / resume"),
    ("g / r", "gizmo: translate / rotate"),
    ("t", "gizmo: body / world frame"),
    ("hold X / Y / Z", "move the mouse along that gizmo axis; no click required"),
    ("f", "frame the whole scene (eased)"),
    ("W A S D", "fly"),
    ("Q / E", "up / down along world Z"),
    ("F1 / ?", "this panel"),
)

VALUE_GESTURES: tuple[tuple[str, str], ...] = (
    ("right click", "reset to the initial value"),
    ("double click", "copy to the clipboard"),
    ("Shift + right click", "more options (drive gain on a joint slider)"),
)


class HelpPanel(Panel):
    name = "Help"
    default_open = False
    shortcut = "F1"
    aliases = ("?",)

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        _table("mouse", MOUSE_GESTURES)
        _table("keyboard", KEYS)
        _table("value controls", VALUE_GESTURES)
        self._panels(ctx)

    def _panels(self, ctx: PanelContext) -> None:
        panels: Any = ctx.panels
        if panels is None:
            imgui.text_disabled("no panel set in this context")
            return
        if not imgui.collapsing_header("panels", imgui.TreeNodeFlags_.default_open):
            return
        imgui.text_disabled("a panel closed with its X can only be reopened with its key")
        if not imgui.begin_table("help_panels", 3, _FLAGS):
            return
        imgui.table_setup_column("key")
        imgui.table_setup_column("panel")
        imgui.table_setup_column("default")
        imgui.table_headers_row()
        for key, name, default_open in panels.shortcut_table():
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(key or "—")
            imgui.table_next_column()
            imgui.text(name)
            imgui.table_next_column()
            imgui.text_disabled("open" if default_open else "closed")
        imgui.end_table()


_FLAGS = (
    imgui.TableFlags_.sizing_stretch_prop
    | imgui.TableFlags_.row_bg
    | imgui.TableFlags_.borders_inner_h
)


def _table(title: str, rows: tuple[tuple[str, str], ...]) -> None:
    if not imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open):
        return
    if not imgui.begin_table(f"help_{title}", 2, _FLAGS):
        return
    for left, right in rows:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text(left)
        imgui.table_next_column()
        imgui.text_disabled(right)
    imgui.end_table()
