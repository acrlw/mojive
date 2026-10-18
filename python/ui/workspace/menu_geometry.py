"""Resolve menu layout once on ImGui's native integer coordinate lattice."""

from math import floor, frexp, ldexp

from mojive.design.native.geometry import Box, Rect, independent_corner_scope
from mojive.ui.workspace.corner_geometry import nested_corners


def native_unit(value):
    """Quantize layout dimensions, independently of framebuffer density."""
    return floor(float(value) + 0.5)


def native_edges(style, prefix, scale, fallback=(0, 0, 0, 0)):
    return tuple(native_unit(value) for value in style.edges(prefix, scale, fallback))


def _bounded_edges(edges, width, height):
    left, top, right, bottom = edges
    if left + right > width:
        left = native_unit(left * width / (left + right))
        right = width - left
    if top + bottom > height:
        top = native_unit(top * height / (top + bottom))
        bottom = height - top
    return left, top, right, bottom


def native_box(box):
    """Keep outer, margin, frame and inset edges in one resolved layout space."""
    outer = Rect(*(native_unit(value) for value in box.outer.tuple()))
    margins = _bounded_edges(
        tuple(native_unit(value) for value in box.margin), outer.width, outer.height
    )
    frame = outer.inset(*margins)
    padding = _bounded_edges(
        tuple(native_unit(value) for value in box.padding), frame.width, frame.height
    )
    return Box(outer, frame, frame.inset(*padding), padding, margins)


def native_epsilon(*coordinates):
    """Bound float32 coordinate cancellation, without forgiving a visible deficit."""
    magnitude = max((abs(value) for value in coordinates), default=0)
    return 2 * ldexp(1.0, frexp(magnitude)[1] - 24) if magnitude else ldexp(1.0, -148)


def text_budget(available, measured, epsilon):
    return measured if 0 <= measured - available <= epsilon else available


def glyph_rect(slot, size):
    """Center an unquantized canonical glyph in its quantized layout slot."""
    return Rect(slot.x + (slot.width - size) / 2, slot.y + (slot.height - size) / 2, size, size)


def row_radius(style, scale, width=None, height=None):
    radius = max(0, style.get("radius", 6) * scale)
    if width is not None and height is not None:
        radius = min(radius, max(0, width / 2 - 1), max(0, height / 2 - 1))
    return radius


def popup_padding(ui, part):
    """Resolve adjacent popup edges to one actual native corner offset."""
    fallback = (8,) * 4 if part == "select_popup" else (6,) * 4
    padding = native_edges(ui.component_style(part), "padding", ui.s, fallback)
    return (max(padding),) * 4 if part in ("menu", "select_popup") else padding


def popup_radius(ui, part, padding):
    """Preserve an authored uniform concentric relation after layout quantization."""
    style = ui.component_style(part)
    if part == "select_popup":
        with independent_corner_scope():
            item = ui.component_style("select_option")
            return nested_corners(style, ui.s, padding, row_radius(item, ui.s)).radius
    outer = style.get("radius", 12 if part in ("menu", "select_popup") else 8)
    with independent_corner_scope():
        item = ui.component_style("menu_item")
    fallback = (8, 8, 8, 8) if part == "select_popup" else (6, 6, 6, 6)
    authored = style.edges("padding", fallback=fallback)
    if (
        part in ("menu", "select_popup")
        and len(set(authored)) == 1
        and len(set(padding)) == 1
        and abs(outer - item.get("radius", 6) - authored[0]) < 1e-9
    ):
        return row_radius(item, ui.s) + padding[0]
    return outer * ui.s
