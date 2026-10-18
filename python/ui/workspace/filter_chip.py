"""Scene filter chips shared by the workspace and design specimens."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.compound_fields import draw_focus_frame
from mojive.ui.workspace.design import BODY_SIZE
from mojive.ui.workspace.menu_geometry import native_epsilon, text_budget
from mojive.ui.workspace.widgets import INPUT, MUTED


def _capsule_padding(style, height, extent, scale):
    inset = max(0, (height - extent) / 2) / scale
    return style.edges("padding", scale, (inset, 0, inset, 0))


@part_scope("type_filters", "filter")
def filter_chip_size(ui, count, *, maximum_width=None):
    """Measure the same capsule ends and content spacing used by the painter."""
    style, scale = ui.component_style("filter"), ui.s
    font = style.get("fontSize", BODY_SIZE)
    extent = style.get("iconSize", 16) * scale
    line = max(font, style.line_height(font * 1.4)) * scale
    vertical = style.edges("padding", scale)
    height = max(style.get("height", 28) * scale, max(line, extent) + vertical[1] + vertical[3])
    padding = _capsule_padding(style, height, extent, scale)
    gap = style.get("gap", 8) * scale
    caption = ui.measure(str(count), font, translate=False, font_resolved=True, style=style)
    content_width = extent + gap + caption
    box = style.box(
        0,
        0,
        content_width + padding[0] + padding[2],
        height,
        scale=scale,
        padding=padding,
        intrinsic=(content_width, max(line, extent)),
    )
    width = box.outer.width if maximum_width is None else min(maximum_width, box.outer.width)
    return width, box.outer.height


@part_scope("type_filters", "filter")
def draw_filter_chip(ui, key, rect, count, icon, tooltip, accent, active):
    u, scale = ui, ui.s
    style = u.component_style("filter")
    font = style.get("fontSize", BODY_SIZE)
    extent = style.get("iconSize", 16) * scale
    line = max(font, style.line_height(font * 1.4)) * scale
    pads = style.edges("padding", scale)
    box = u.component_box(
        "filter",
        *rect.tuple(),
        key=key,
        padding=pads,
        observe=False,
        minimum_height=max(line, extent) + pads[1] + pads[3],
    )
    frame = box.frame
    padding = _capsule_padding(style, frame.height, extent, scale)
    box = replace(box, content=frame.inset(*padding), padding=padding)
    content = box.content
    imgui.set_cursor_screen_pos((frame.x, frame.y))
    clicked = imgui.invisible_button(
        key, (max(1, frame.width), max(1, frame.height)), imgui.ButtonFlags_.enable_nav
    )
    hovered = imgui.is_item_hovered()
    u.hits[key] = frame.tuple()
    background = (*accent[:3], 40 if active else 24) if active or hovered else INPUT
    radius = style.get("radius", 14) if style.explicit("radius") else frame.height / (2 * scale)
    u.rect(*frame.tuple(), background, radius)
    gap = style.get("gap", 8) * scale
    caption_width = u.measure(str(count), font, translate=False, font_resolved=True, style=style)
    group = style.align_rect(
        content,
        min(content.width, extent + gap + caption_width),
        min(content.height, max(line, extent)),
    )
    caption_rect = Rect(
        group.x + extent + gap,
        group.y,
        text_budget(
            max(0, group.width - extent - gap),
            caption_width,
            native_epsilon(content.x, content.right),
        ),
        group.height,
    )
    u.draw.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
    icon_rect = Rect(group.x, group.y + (group.height - extent) / 2, extent, extent)
    u.icon(icon, icon_rect.x, icon_rect.y, extent / scale, accent)
    u.component_text(
        "filter",
        caption_rect,
        str(count),
        accent if active else MUTED,
        translate=False,
    )
    u.draw.pop_clip_rect()
    u.record_component_box(
        "filter",
        box,
        key=key,
        fields=[
            list(icon_rect.tuple()),
            list(caption_rect.tuple()),
        ],
        radius=radius,
        count=count,
        fontSize=font,
        fontFamily=style.get("fontFamily", "inherit"),
    )
    draw_focus_frame(
        imgui.ImVec2(frame.x, frame.y),
        imgui.ImVec2(frame.right, frame.bottom),
        rounding=radius * scale,
        item_id=imgui.get_item_id(),
    )
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
        u.tooltip(tooltip, translate=False)
    return clicked
