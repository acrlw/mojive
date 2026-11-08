"""Shared joint/actuator readout layout using the production numeric and rail controls."""

from math import degrees

from imgui_bundle import imgui

from mojive.ui.numeric_text import fit_numeric_text, numeric_text

from .widgets import MUTED


def scalar_value_width(ui, width, *, show_reset=True):
    """Reserve a stable header budget independent of the current numeric value."""
    if not show_reset:
        return min(width * 0.5, 106 * ui.s)
    return min(
        width * 0.65, max(124 * ui.s, ui.numeric_minimum(axis=False, format="%.3f") + 32 * ui.s)
    )


def scalar_card(
    ui,
    key,
    x,
    y,
    width,
    value,
    bounds,
    *,
    initial,
    unit="",
    angular_degrees=True,
    toggle_unit=None,
    show_reset=True,
):
    scale = ui.s
    height = ui.rail_height(unit=bool(unit))
    value_width = scalar_value_width(ui, width, show_reset=show_reset)
    reset_width = height + 8 * scale if show_reset else 0
    entry_width = max(scale, value_width - reset_width)
    # Limits have equal, fixed slots; long limits keep exact hover/copy text.
    limit_width = min(64 * scale, width * 0.22)
    rail_y = y + height + 4 * scale
    track = (x + limit_width, rail_y, max(scale, width - 2 * limit_width), height)
    edit = ui.rail(
        key,
        x,
        y,
        width,
        value,
        bounds,
        initial=initial,
        fmt="%.3f",
        show_reset=show_reset,
        flat=True,
        unit=unit,
        angular_degrees=angular_degrees,
        toggle_unit=toggle_unit,
        entry_rect=(x + width - value_width, y, entry_width, height),
        track_rect=track,
    )
    if bounds is not None:
        shown = tuple(degrees(v) for v in bounds) if unit == "rad" and angular_degrees else bounds
        style = ui.component_style("control_metadata")
        font = style.get("fontSize", 12)
        for index, number in enumerate(shown):
            inset = 4 * scale
            rect = (
                x + (width - limit_width if index else 0) + inset,
                rail_y,
                limit_width - 2 * inset,
                height,
            )
            caption = fit_numeric_text(
                number,
                "%.4g",
                max(scale, limit_width - 4 * scale),
                lambda text: (ui.measure(text, font, mono=True, translate=False, style=style), 0),
            )
            ui.component_text(
                "control_metadata",
                rect,
                caption,
                MUTED,
                mono=True,
                size=font,
                align="end" if index else "start",
                translate=False,
            )
            imgui.set_cursor_screen_pos(rect[:2])
            imgui.invisible_button(key + f"-limit-{index}", rect[2:])
            if imgui.is_item_hovered():
                if caption != numeric_text(number, "%.12g"):
                    ui.tooltip(numeric_text(number, "%.12g"), translate=False)
                if imgui.is_mouse_clicked(1):
                    imgui.set_clipboard_text(numeric_text(number, "%.17g"))
    ui.hits[key] = (
        x,
        y,
        width,
        height * (2 if bounds is not None else 1) + (4 * scale if bounds else 0),
    )
    return edit
