"""Corner geometry shared by navigation chrome and its catalogue specimen."""

from dataclasses import replace
from math import ceil

from mojive.ui.workspace.component_layout import control_radius_limit, control_size
from mojive.ui.workspace.panel_geometry import corner_inset


def _rail_layout(ui):
    style, scale = ui.component_style("rail"), ui.s
    padding = style.edges("padding", scale, (6, 8, 6, 6))
    margin = style.edges("margin", scale)
    outer = _outer_radius(style)
    inner = max(0, outer - padding[0] - margin[0])
    requested = style.get("slotSize", 36) * scale
    slot = requested
    minimum = 2 * inner + 2
    if minimum > slot:
        slot = ceil(minimum)
    options = {"icon": "scene", "icon_size": style.get("iconSize", 24), "icon_role": "activity"}
    limit = control_radius_limit(ui, "rail-corner", slot / scale, slot / scale, **options)
    if inner > limit:
        # A generic authored width/height can keep the actual frame smaller than
        # its requested slot. Preserve that frame and reserve its real offset.
        inset = corner_inset(outer, max(padding) + margin[0], limit) - margin[0]
        padding = (inset,) * 4
        slot = requested
    child_width, _height = control_size(ui, "rail-corner", slot / scale, slot / scale, **options)
    width = max(style.get("width", 48) * scale, child_width + padding[0] + padding[2])
    return slot / scale, width / scale, padding


def rail_dimensions(ui):
    """Reserve real slots and insets without overriding authored control dimensions."""
    slot, width, _padding = _rail_layout(ui)
    return slot, width


def rail_padding(ui):
    return _rail_layout(ui)[2]


def expand_rail_box(ui, box):
    """Consume the width reserved for a physically realizable authored corner."""
    _slot, width, padding = _rail_layout(ui)
    width = max(box.frame.width, width * ui.s)
    if width == box.frame.width and padding == box.padding:
        return box
    frame = replace(box.frame, width=width)
    return replace(
        box,
        outer=replace(box.outer, width=width + box.margin[0] + box.margin[2]),
        frame=frame,
        content=frame.inset(*padding),
        padding=padding,
    )


def _outer_radius(style):
    return max(0, style.get("radius", 16))


def rail_corners(ui, box):
    """Keep the OS-facing corner fixed while the content inset follows UI scale."""
    style = ui.component_style("rail")
    # The macOS window contour is about 16 window points at Retina density 2.
    # Application UI zoom changes the inset, not that compositor-owned contour.
    outer = _outer_radius(style)
    inset = box.content.x - box.outer.x
    return outer, inset, max(0, outer - inset)
