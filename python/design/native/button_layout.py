"""Measure button content independently of per-frame native input and feedback."""

import math

from mojive.design.native.geometry import Box, Rect
from mojive.design.native.style_cache import prepared_geometry


@prepared_geometry
def button_metrics(style, icon, label, icon_fallback, authored_icon_size, radius, align):
    """Resolve frozen control metrics; input state and parent paint remain live."""
    corner_radius = (
        style.get("radius", 6.0)
        if radius is None
        or style.explicit("radius")
        or style.origins.get("radius") == "parent_corner"
        else radius
    )
    font_size = style.get("fontSize", 14.0)
    icon_size = (
        (
            style.get("iconSize", icon_fallback)
            if style.explicit("iconSize") or authored_icon_size is None
            else authored_icon_size
        )
        if icon
        else 0.0
    )
    gap = style.get("columnGap", style.get("gap", 6.0)) if icon and label else 0.0
    line_height = style.line_height(font_size * 1.4) if label else 0.0
    return corner_radius, font_size, icon_size, gap, line_height, style.get("align", align)


@prepared_geometry
def button_layout(
    style, rect, scale, padding, text_width, line_height, icon_size, gap, horizontal, width_limit
):
    box = style.box(
        *rect.tuple(),
        scale=scale,
        padding=padding,
        intrinsic=(0, max(line_height, icon_size * scale)),
    )
    limit = max(0.0, width_limit) if width_limit is not None else math.inf
    if box.frame.width > limit:
        frame = Rect(box.frame.x, box.frame.y, limit, box.frame.height)
        outer = Rect(
            box.outer.x, box.outer.y, frame.width + box.margin[0] + box.margin[2], box.outer.height
        )
        box = Box(outer, frame, frame.inset(*box.padding), box.padding, box.margin)
    content = box.content
    icon_extent = icon_size * scale
    visible_text_width = min(text_width, max(0.0, content.width - icon_extent - gap))
    placed = style.align_rect(
        content,
        icon_extent + gap + visible_text_width,
        max(icon_extent, line_height),
        horizontal=horizontal,
    )
    icon_x = placed.x
    text_x = placed.x + icon_extent + gap if icon_size else placed.x
    if horizontal == "space-between" and icon_size and text_width:
        text_x = max(placed.x + icon_extent + gap, content.right - visible_text_width)
    icon_y = style.align_rect(content, icon_extent, icon_extent).y if icon_size else content.y
    text_y = style.align_rect(content, text_width, line_height).y
    return box, icon_x, icon_y, text_x, text_y, visible_text_width, icon_extent
