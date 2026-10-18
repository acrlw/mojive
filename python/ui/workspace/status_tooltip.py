"""Readable status hints with independent padding and action rows."""

from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.ui.workspace.design import CONTROL_RADIUS
from mojive.ui.workspace.status_components import metric_tooltip_options


def status_tooltip(ui, first_line, *action_lines, scale=1.0):
    """Show an exact status value and separate, comfortably spaced action rows."""
    font, color = None, None
    size = 0.0
    if ui is None:
        padding, row_gap = (10 * scale, 8 * scale), 6 * scale
    else:
        scale = ui.s
        padding, row_gap = metric_tooltip_options(ui)
        with part_scope("status_metrics", inherit=False):
            style = ui.component_style("status_metrics")
            font, size = ui.resolve_font(style), style.get("fontSize", 14) * scale
            if style.get("textColor", "inherit") != "inherit":
                fallback = tuple(
                    round(v * 255) for v in imgui.get_style_color_vec4(imgui.Col_.text)
                )
                color = tuple(v / 255 for v in ui.style_color(style, "textColor", fallback))

    imgui.push_style_var(imgui.StyleVar_.window_padding, padding)
    imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, row_gap))
    imgui.push_style_var(imgui.StyleVar_.window_rounding, CONTROL_RADIUS * scale)
    if font is not None:
        imgui.push_font(font, size)
    if color is not None:
        imgui.push_style_color(imgui.Col_.text, color)
    try:
        if imgui.begin_tooltip():
            try:
                for line in (first_line, *action_lines):
                    imgui.text_unformatted(line)
            finally:
                imgui.end_tooltip()
    finally:
        if color is not None:
            imgui.pop_style_color()
        if font is not None:
            imgui.pop_font()
        imgui.pop_style_var(4)
