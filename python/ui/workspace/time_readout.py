"""Readonly timeline values share one caption, unit spacing and paint budget."""

from imgui_bundle import imgui


def draw_time_readout(ui, key, x, y, value, *, width=80):
    scale = ui.s
    imgui.set_cursor_screen_pos((x, y))
    imgui.dummy((width * scale, 28 * scale))
    rect = (x, y, width * scale, 28 * scale)
    ui.hits[key] = rect
    ui.rect(*rect, ui.palette("child"), color_resolved=True)
    style = ui.component_style("time_field")
    font_size = style.get("fontSize", 14)
    gap = style.get("columnGap", style.get("gap", 4)) * scale
    unit_width = ui.measure("s", font_size, translate=False, mono=True, style=style)
    available = max(0, (width - 16) * scale - unit_width - gap)
    caption = f"{value:.0e}"
    for digits in (3, 2, 1, 0):
        candidate = f"{value:.{digits}f}"
        if ui.measure(candidate, font_size, translate=False, mono=True, style=style) <= available:
            caption = candidate
            break
    numeric_width = available
    total = numeric_width + gap + unit_width
    left = x + (width * scale - total) / 2
    text_width = min(
        available, ui.measure(caption, font_size, translate=False, mono=True, style=style)
    )
    text_y = y + (28 - font_size) * scale / 2
    ui.text(
        left + numeric_width - text_width,
        text_y,
        caption,
        size=font_size,
        mono=True,
        max_width=text_width,
        translate=False,
        style=style,
    )
    ui.text(
        left + numeric_width + gap,
        text_y,
        "s",
        size=font_size,
        mono=True,
        max_width=unit_width,
        translate=False,
        style=style,
    )
    if ui.design is not None:
        ui.design.record(
            key,
            ("time_field",),
            rect,
            "number-unit",
            gap=gap,
            content=[left, text_y, total, font_size * scale],
        )
    return rect
