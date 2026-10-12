"""Compact native operation notices; Output retains the complete message history."""

from mojive.ui.panels.filters import severity_color, severity_group

from .design import BODY_SIZE, CONTROL_RADIUS


def draw_operation_notice(app, overlay, message):
    ui, scale = app.workspace.ui, app.workspace.s
    x, y, width, height = app._viewport_rect
    inset, padding, icon_size, gap, box_height = (v * scale for v in (14, 12, 18, 8, 36))
    available = width - 2 * inset
    if height < box_height + 2 * inset or available <= 2 * padding + icon_size + gap:
        return
    caption = " ".join(ui.translate(message.text).splitlines())
    box_width = min(available, ui.measure(caption, translate=False) + 2 * padding + icon_size + gap)
    left, top = x + inset, y + height - inset - box_height
    level = severity_group(message.level)
    foreground = severity_color(app.theme, level)
    background = (
        (0.20, 0.10, 0.11, 0.94)
        if level == "error"
        else (0.19, 0.15, 0.09, 0.94)
        if level == "warning"
        else (0.10, 0.13, 0.12, 0.94)
    )
    overlay.rect_filled(
        (left, top),
        (left + box_width, top + box_height),
        background,
        rounding=CONTROL_RADIUS * scale,
    )
    rgba = tuple(round(channel * 255) for channel in foreground)
    ui.icon(
        {"info": "info", "warning": "warn", "error": "err"}[level],
        left + padding,
        top + (box_height - icon_size) / 2,
        icon_size / scale,
        rgba,
        draw=overlay._dl,
    )
    ui.text(
        left + padding + icon_size + gap,
        top + (box_height - BODY_SIZE * scale) / 2,
        caption,
        rgba,
        draw=overlay._dl,
        translate=False,
        max_width=max(0, box_width - 2 * padding - icon_size - gap),
    )
    app._status_notice_bounds = (left, top, left + box_width, top + box_height)
