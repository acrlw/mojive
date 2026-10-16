"""Present model component actions without owning component edits or commands."""

from math import inf

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, budget_margins
from mojive.ui.workspace.component_layout import control_frame_size, control_size, flow_layout
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.property_style import advance
from mojive.ui.workspace.widgets import MUTED


def component_actions(ui, part, key, label, labels, enabled):
    """Allocate a caption and complete action frames inside one row budget."""
    pos, width = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail().x
    style, s = ui.component_style(part), ui.s
    padding = style.edges("padding", s, (8, 6, 8, 6))
    margin = budget_margins(style.edges("margin", s), width, inf)
    gap = style.get("columnGap", style.get("gap", DESIGN.item_spacing_x)) * s
    sizes = []
    with ui.component_scope(part):
        for index, text in enumerate(labels):
            action_key = key + "-" + str(index)
            parts = ui.design.button_parts(action_key, text) if ui.design is not None else "button"
            action = ui.component_style(parts)
            pads = action.edges("padding", s, (6, 0, 6, 0))
            text_width = ui.measure(
                text, action.get("fontSize", DESIGN.body_size), font_resolved=True
            )
            sizes.append(
                control_size(
                    ui,
                    action_key,
                    (text_width + pads[0] + pads[2]) / s + 4,
                    DESIGN.control_height,
                    label=text,
                )
            )
    font = style.get("fontSize", DESIGN.body_size)
    available = max(0, width - margin[0] - margin[2])
    frame_width = min(available, style.dimension("width", available / s) * s)
    content = Rect(
        pos.x + margin[0] + min(padding[0], frame_width),
        pos.y + margin[1] + padding[1],
        max(0, frame_width - padding[0] - padding[2]),
        0,
    )
    total = sum(w for w, _ in sizes) + gap * max(0, len(sizes) - 1)
    above = total + gap + 64 * s > content.width
    caption_height = max(font, style.line_height(font * 1.4)) * s
    action_flow = (
        flow_layout(
            Rect(content.x, content.y + caption_height + gap, content.width, 0),
            sizes,
            column_gap=gap,
            row_gap=gap,
            align_items=style.get("alignItems", "center"),
        )
        if above
        else None
    )
    required_height = (
        caption_height + gap + action_flow.height
        if above
        else max(caption_height, max((h for _, h in sizes), default=0))
    )
    height = (
        max(
            style.get("height", DESIGN.control_height) * s,
            required_height,
        )
        + padding[1]
        + padding[3]
    )
    box = ui.component_box(
        part,
        pos.x,
        pos.y,
        width,
        height + margin[1] + margin[3],
        key=key,
        minimum_height=required_height + padding[1] + padding[3],
        padding=padding,
    )
    text_rect = Rect(
        box.content.x,
        box.content.y,
        box.content.width if above else max(0, box.content.width - total - gap),
        caption_height if above else box.content.height,
    )
    ui.component_text(part, text_rect, label, MUTED, translate=False)
    if ui.measure(
        label, font, translate=False, font_resolved=True
    ) > text_rect.width and imgui.is_mouse_hovering_rect(
        (text_rect.x, text_rect.y), (text_rect.right, text_rect.bottom)
    ):
        ui.tooltip(label, translate=False)
    x = box.content.x if above else box.content.right - total
    results = []
    for index, (text, (w, h), usable) in enumerate(zip(labels, sizes, enabled, strict=True)):
        placed = (
            action_flow.rectangles[index]
            if above
            else style.align_rect(
                Rect(x, box.content.y, w, box.content.height), w, h, horizontal="start"
            )
        )
        with ui.component_scope(part):
            frame_width, frame_height = control_frame_size(
                ui, key + "-" + str(index), placed.width, placed.height, label=text
            )
            results.append(
                ui.button(
                    key + "-" + str(index),
                    placed.x,
                    placed.y,
                    frame_width,
                    frame_height,
                    width_limit=frame_width * s,
                    label=text,
                    enabled=usable,
                    translate=False,
                )
            )
        x += w + gap
    advance(ui, pos.x, box.outer.bottom + DESIGN.row_gap * s)
    return tuple(results)
