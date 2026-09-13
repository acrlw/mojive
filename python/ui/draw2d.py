"""Compatibility exports; new code imports the protocol or its explicit UI owner."""

from mojive.ui.paint_protocol import Draw2D

from .drag_link import draw_drag_link
from .imgui_draw import ImguiDraw2D, ink_box
from .text_layout import fit_text, text_line_y

__all__ = ("Draw2D", "ImguiDraw2D", "draw_drag_link", "fit_text", "ink_box", "text_line_y")
