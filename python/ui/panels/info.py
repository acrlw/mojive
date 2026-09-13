"""Renderer and runtime information panel."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import Panel, PanelContext

_FLAGS = (
    imgui.TableFlags_.sizing_stretch_prop
    | imgui.TableFlags_.row_bg
    | imgui.TableFlags_.borders_inner_h
)


class InfoPanel(Panel):
    id = "info"
    name = "Info"
    default_open = False
    shortcut = "F10"

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        if not ctx.info:
            imgui.text_disabled(ctx.tr("nothing pushed yet"))
            imgui.set_item_tooltip(ctx.tr("scripts fill this via the viewer's info dict"))
            return
        if not imgui.begin_table("info_kv", 2, _FLAGS):
            return
        imgui.table_setup_column(ctx.tr("Key"))
        imgui.table_setup_column(ctx.tr("value"))
        imgui.table_headers_row()
        for key, value in ctx.info.items():
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(str(key))
            imgui.table_next_column()

            imgui.text(_fmt(value))
        imgui.end_table()


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}: unprintable>"
