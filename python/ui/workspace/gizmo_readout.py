"""Shared numeric feedback for transform drags and joint range endpoints."""

from functools import lru_cache

from mojive.design.native.geometry import part_scope
from mojive.ui.design_palette import rgba8
from mojive.ui.text_layout import text_line_y
from mojive.ui.theme import relative_luminance
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.floating import floating_number


def readout_ink(role, ui=None):
    color = ui.ink(rgba8(role)) if ui is not None else rgba8(role)
    return tuple(channel / 255 for channel in color)


@lru_cache(maxsize=128)
def readout_marker_ink(color, surface):
    """Keep world-space marker hues legible on a themed, opaque readout."""
    background = relative_luminance(surface)

    def contrast(candidate):
        foreground = relative_luminance(candidate)
        return (max(background, foreground) + 0.05) / (min(background, foreground) + 0.05)

    opaque = (*color[:3], 1.0)
    if contrast(opaque) >= 3:
        return opaque
    target = 0.0 if background > 0.5 else 1.0
    lo, hi = 0.0, 1.0
    for _ in range(12):
        amount = (lo + hi) / 2
        candidate = (*(v + (target - v) * amount for v in color[:3]), 1.0)
        if contrast(candidate) >= 3:
            hi = amount
        else:
            lo = amount
    return (*(v + (target - v) * hi for v in color[:3]), 1.0)


@part_scope("input_number", inherit=False)
def draw_gizmo_surface(draw, rect, scale, *, ui=None):
    style = ui.component_style("input_number") if ui is not None else None
    radius = (
        style.get("radius", DESIGN.control_radius) if style else DESIGN.control_radius
    ) * scale
    draw.rect_filled(rect[:2], rect[2:], readout_ink("floating", ui), rounding=radius)


@part_scope("input_number", inherit=False)
def draw_gizmo_readout(
    draw,
    anchor,
    label,
    scale,
    *,
    ui=None,
    semantic_color=None,
    above=False,
    align_right=False,
    centered=False,
    gap=8,
    viewport=None,
):
    """Use numeric-field typography and geometry on a quiet floating surface."""
    style = ui.component_style("input_number") if ui is not None else None
    padding = style.edges("padding", scale, (8, 0, 8, 0)) if style else (8 * scale, 0, 8 * scale, 0)
    minimum_height = (
        style.get("height", DESIGN.control_height) if style else DESIGN.control_height
    ) * scale

    def ink(role):
        color = ui.ink(rgba8(role), style) if ui is not None else rgba8(role)
        if role == "text" and style:
            color = ui.style_color(style, "textColor", color)
        return tuple(channel / 255 for channel in color)

    with floating_number(ui):
        text_width, text_height = draw.text_size(label) or (len(label) * 8 * scale, 14 * scale)
        dot_radius = 3 * scale if semantic_color is not None else 0
        prefix = 12 * scale if semantic_color is not None else 0
        width = padding[0] + prefix + text_width + padding[2]
        height = max(minimum_height, text_height + max(12 * scale, padding[1] + padding[3]))
        x = (
            float(anchor[0]) - width / 2
            if centered
            else (
                float(anchor[0]) - width - gap * scale
                if align_right
                else float(anchor[0]) + gap * scale
            )
        )
        y = float(anchor[1]) - height - gap * scale if above else float(anchor[1]) + gap * scale
        if viewport is not None:
            vx, vy, vw, vh = viewport
            inset = 4 * scale
            x = max(vx + inset, min(x, vx + vw - width - inset))
            y = max(vy + inset, min(y, vy + vh - height - inset))
        draw_gizmo_surface(draw, (x, y, x + width, y + height), scale, ui=ui)
        center_y = y + height / 2
        if semantic_color is not None:
            draw.circle_filled(
                (x + padding[0] + dot_radius, center_y),
                dot_radius,
                readout_marker_ink(tuple(semantic_color), ink("floating")),
            )
        draw.text((x + padding[0] + prefix, text_line_y(draw, center_y)), ink("text"), label)
        return (x, y, x + width, y + height)
