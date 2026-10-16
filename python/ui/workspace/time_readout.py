"""Readonly timeline values share one caption, unit spacing and paint budget."""

from imgui_bundle import imgui


def draw_time_readout(ui, key, x, y, value, *, width=80):
    scale = ui.s
    imgui.set_cursor_screen_pos((x, y))
    imgui.dummy((width * scale, 28 * scale))
    rect = (x, y, width * scale, 28 * scale)
    ui.hits[key] = rect
    ui.rect(*rect, (24, 27, 30, 255))
    gap = ui.metric("time_field", "columnGap", ui.metric("time_field", "gap", 4)) * scale
    unit_width = ui.measure("s", 14, translate=False, mono=True)
    available = max(0, (width - 16) * scale - unit_width - gap)
    caption = f"{value:.0e}"
    for digits in (3, 2, 1, 0):
        candidate = f"{value:.{digits}f}"
        if ui.measure(candidate, 14, translate=False, mono=True) <= available:
            caption = candidate
            break
    numeric_width = min(available, ui.measure(caption, 14, translate=False, mono=True))
    total = numeric_width + gap + unit_width
    left = x + max(0, (width * scale - total) / 2)
    ui.text(
        left,
        y + 7 * scale,
        caption,
        size=14,
        mono=True,
        max_width=numeric_width,
        translate=False,
    )
    ui.text(
        left + numeric_width + gap,
        y + 7 * scale,
        "s",
        size=14,
        mono=True,
        max_width=unit_width,
        translate=False,
    )
    if ui.design is not None:
        ui.design.record(
            key,
            ("time_field",),
            rect,
            "number-unit",
            gap=gap,
            content=[left, y + 7 * scale, total, 14 * scale],
        )
    return rect
