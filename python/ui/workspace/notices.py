"""Compact operation notices using the same native specification and text boxes."""

from mojive.design.native.geometry import Rect
from mojive.ui.design_palette import rgba as palette_rgba
from mojive.ui.panels.filters import severity_color, severity_group
from mojive.ui.workspace.design import BODY_SIZE, CONTROL_RADIUS


def draw_operation_notice(app, overlay, message):
    ui, scale = app.workspace.ui, app.workspace.s
    x, y, width, height = app._viewport_rect
    style = ui.component_style("notice")
    padding = style.edges("padding", scale, (12, 8, 12, 8))
    margin = style.edges("margin", scale)
    icon_size = style.get("iconSize", 18) * scale
    gap = style.get("columnGap", style.get("gap", 8)) * scale
    font_size = style.get("fontSize", BODY_SIZE)
    line_height = max(font_size, style.line_height(font_size * 1.4)) * scale
    box_height = max(line_height, icon_size) + padding[1] + padding[3] + margin[1] + margin[3]
    inset = 14 * scale
    available = width - 2 * inset
    if height < box_height + 2 * inset or available <= padding[0] + padding[2] + icon_size + gap:
        return
    caption = " ".join(ui.translate(message.text).splitlines())
    box_width = min(
        available,
        ui.measure(caption, font_size, translate=False)
        + padding[0]
        + padding[2]
        + margin[0]
        + margin[2]
        + icon_size
        + gap,
    )
    box = ui.component_box(
        "notice",
        x + inset,
        y + height - inset - box_height,
        box_width,
        box_height,
        key="operation-notice",
        padding=padding,
        draw=overlay._dl,
        observe=False,
    )
    level = severity_group(message.level)
    foreground = severity_color(app.theme, level)
    background = (
        (0.20, 0.10, 0.11, 0.94)
        if level == "error"
        else (0.19, 0.15, 0.09, 0.94)
        if level == "warning"
        else (*palette_rgba("child")[:3], 0.94)
    )
    overlay.rect_filled(
        (box.frame.x, box.frame.y),
        (box.frame.right, box.frame.bottom),
        background,
        rounding=style.get("radius", CONTROL_RADIUS) * scale,
    )
    rgba = tuple(round(channel * 255) for channel in foreground)
    icon_y = box.content.y + max(0, (box.content.height - icon_size) / 2)
    ui.icon(
        {"info": "info", "warning": "warn", "error": "err"}[level],
        box.content.x,
        icon_y,
        icon_size / scale,
        rgba,
        draw=overlay._dl,
    )
    ui.component_text(
        "notice",
        Rect(
            box.content.x + icon_size + gap,
            box.content.y,
            max(0, box.content.width - icon_size - gap),
            box.content.height,
        ),
        caption,
        rgba,
        draw=overlay._dl,
        translate=False,
    )
    ui.record_component_box(
        "notice",
        box,
        key="operation-notice",
        draw=overlay._dl,
        fields=[
            [box.content.x, box.content.y, icon_size, box.content.height],
            [
                box.content.x + icon_size + gap,
                box.content.y,
                max(0, box.content.width - icon_size - gap),
                box.content.height,
            ],
        ],
    )
    app._status_notice_bounds = (box.frame.x, box.frame.y, box.frame.right, box.frame.bottom)


def draw_center_notice(app, overlay, message, *, border=False):
    """Show the existing empty/drop message with catalogue spacing and alignment."""
    ui, scale = app.workspace.ui, app.workspace.s
    style = ui.component_style("empty")
    lines = message.splitlines()
    font = style.get("fontSize", BODY_SIZE)
    line_height = max(font, style.line_height(font * 1.4)) * scale
    gap = style.get("rowGap", style.get("gap", 3)) * scale
    padding = style.edges("padding", scale, (18, 12, 18, 12))
    margins = style.edges("margin", scale)
    icon = style.get("iconSize", 20) * scale
    x, y, width, height = app._viewport_rect
    measured_width = max((ui.measure(line, font, translate=False) for line in lines), default=0)
    content_width = max(icon, measured_width)
    box_width = min(
        width - 12 * scale, content_width + padding[0] + padding[2] + margins[0] + margins[2]
    )
    content_height = icon + gap + len(lines) * line_height + max(0, len(lines) - 1) * gap
    box_height = min(
        height - 12 * scale, content_height + padding[1] + padding[3] + margins[1] + margins[3]
    )
    box = ui.component_box(
        "empty",
        x + (width - box_width) / 2,
        y + (height - box_height) / 2,
        box_width,
        box_height,
        key="viewport-empty-state",
        padding=padding,
        draw=overlay._dl,
    )
    if border:
        overlay.rect(
            (x + 3 * scale, y + 3 * scale),
            (x + width - 3 * scale, y + height - 3 * scale),
            (0.95, 0.68, 0.24, 0.95),
            2 * scale,
            rounding=8 * scale,
        )
    overlay.rect_filled(
        (box.frame.x, box.frame.y),
        (box.frame.right, box.frame.bottom),
        (0.08, 0.09, 0.11, 0.88),
        rounding=7 * scale,
    )
    icon_rect = style.align_rect(
        box.content, icon, icon, horizontal=style.get("textAlign", "center"), vertical="start"
    )
    ui.icon(
        "folder", icon_rect.x, icon_rect.y, icon / scale, (165, 180, 198, 255), draw=overlay._dl
    )
    cursor = icon_rect.bottom + gap
    for line in lines:
        ui.component_text(
            "empty",
            Rect(box.content.x, cursor, box.content.width, line_height),
            line,
            (237, 240, 242, 255),
            draw=overlay._dl,
            align="center",
            translate=False,
        )
        cursor += line_height + gap


def draw_file_receipt(app, overlay, message, path, *, padding_pt=12):
    """Keep the existing file actions within one measured receipt container."""
    from imgui_bundle import imgui

    from mojive.ui.files import reveal_path
    from mojive.ui.workspace.component_layout import (
        control_frame_size,
        control_size,
        flow_layout,
    )
    from mojive.ui.workspace.widgets import PANEL, TEXT

    ui, scale = app.workspace.ui, app.workspace.s
    style = ui.component_style("file_receipt")
    text = app.localizer.text(message.text)
    for spelling in (message.copy_text, str(path)):
        if spelling and spelling in text:
            text = text.replace(spelling, path.name)
            break
    x, y, width, _height = app._viewport_rect
    padding = style.edges("padding", scale, (padding_pt,) * 4)
    margin = style.edges("margin", scale)
    gap = style.get("columnGap", style.get("gap", 8)) * scale
    font = style.get("fontSize", BODY_SIZE)
    line_height = max(font, style.line_height(font * 1.4)) * scale
    labels = (app.localizer.text("Open folder"), app.localizer.text("Copy path"))
    with ui.component_scope("file_receipt"):
        sizes = tuple(
            control_size(ui, key, max(28, ui.measure(label, font) / scale + 16), 28, label=label)
            for key, label in zip(("receipt-open-folder", "receipt-copy-path"), labels, strict=True)
        )
    button_widths = tuple(size[0] for size in sizes)
    action_width = sum(button_widths) + gap
    natural_width = (
        ui.measure(text, font, translate=False)
        + action_width
        + 2 * gap
        + padding[0]
        + padding[2]
        + margin[0]
        + margin[2]
    )
    budget = min(width - 28 * scale, 720 * scale, natural_width)
    if style.get("widthMode", "auto") == "fixed":
        budget = min(width - 28 * scale, style.get("width", budget / scale) * scale)
    content_width = max(0, budget - margin[0] - margin[2] - padding[0] - padding[2])
    inline = ui.measure(text, font, translate=False) + gap + action_width <= content_width
    sizes = [(min(content_width, w), h) for w, h in sizes]
    actions = flow_layout(
        Rect(0, 0, content_width, 0), sizes, column_gap=gap, row_gap=gap, align="end"
    )
    content_height = (
        max(line_height, actions.height) if inline else line_height + gap + actions.height
    )
    box_height = content_height + padding[1] + padding[3] + margin[1] + margin[3]
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_docking
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_focus_on_appearing
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse
    )
    imgui.set_next_window_pos((x + 14 * scale, y + _height - 14 * scale), pivot=(0, 1))
    imgui.set_next_window_size((budget, box_height))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
    imgui.push_style_color(imgui.Col_.window_bg, (0, 0, 0, 0))
    visible, _ = imgui.begin("##viewport-file-status", None, flags)
    if visible:
        pos = imgui.get_window_pos()
        box = ui.component_box(
            "file_receipt", pos.x, pos.y, budget, box_height, key="file-receipt", padding=padding
        )
        ui.rect(*box.frame.tuple(), PANEL, style.get("radius", 8))
        text_width = max(0, box.content.width - action_width - gap) if inline else box.content.width
        text_rect = Rect(box.content.x, box.content.y, text_width, line_height)
        ui.component_text("file_receipt", text_rect, text, TEXT, translate=False)
        app._status_path_bounds = (text_rect.x, text_rect.y, text_rect.right, text_rect.bottom)
        app._status_notice_bounds = (box.frame.x, box.frame.y, box.frame.right, box.frame.bottom)
        if imgui.is_mouse_hovering_rect(
            (text_rect.x, text_rect.y), (text_rect.right, text_rect.bottom)
        ):
            if ui.measure(text, font, translate=False) > text_width:
                ui.tooltip(str(path), translate=False)
            io = imgui.get_io()
            if (io.key_ctrl or io.key_super) and imgui.is_mouse_clicked(0):
                reveal_path(path)
        action_rect = Rect(
            box.content.right - action_width if inline else box.content.x,
            box.content.y if inline else box.content.y + line_height + gap,
            min(action_width, box.content.width),
            actions.height,
        )
        placements = flow_layout(action_rect, sizes, column_gap=gap, row_gap=gap, align="end")
        for index, (label, rect) in enumerate(zip(labels, placements.rectangles, strict=True)):
            key = "receipt-open-folder" if index == 0 else "receipt-copy-path"
            fw, fh = control_frame_size(ui, key, rect.width, rect.height, label=label)
            if ui.button(
                key,
                rect.x,
                rect.y,
                fw,
                fh,
                label=label,
                width_limit=fw * scale,
            ):
                if index == 0:
                    reveal_path(path)
                else:
                    imgui.set_clipboard_text(str(path))
    imgui.end()
    imgui.pop_style_color()
    imgui.pop_style_var(2)
