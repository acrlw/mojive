"""Resolve the timeline's native inset and concentric child corners."""

from contextlib import contextmanager
from dataclasses import dataclass

from mojive.design.native.geometry import scoped_corner_radius
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.panel_geometry import (
    corner_inset,
    floating_panel_geometry,
    framed_control_radius_limit,
)

BODY_CORNER_PARTS = (
    "select",
    "lane_label",
    "timeline_sheet",
    "button",
    "icon_button",
    "input_number",
    "input_text",
)


@dataclass(frozen=True, slots=True)
class TimelineBodyGeometry:
    padding: tuple[float, float]
    radius: float
    inner_radius: float
    selector_radius: float
    lane_radius: float
    requested_padding: tuple[float, float]


def body_geometry(ui):
    """Derive corners from one actual parent radius and a uniform native inset."""
    style, scale = ui.component_style("timeline_body"), ui.s
    requested = tuple(
        native_unit(max(0, style.get(axis, 4)) * scale) for axis in ("paddingX", "paddingY")
    )
    radius = scoped_corner_radius(("timeline_body",))
    if radius is None:
        radius = floating_panel_geometry(ui, "Timeline").inner_radius
    inset = corner_inset(radius, max(requested), framed_control_radius_limit(ui))
    inner = max(0, radius - inset)
    return TimelineBodyGeometry((inset, inset), radius, inner, inner, inner, requested)


@contextmanager
def body_corner_scope(ui, geometry=None):
    geometry = body_geometry(ui) if geometry is None else geometry
    with ui.corner_scope(geometry.inner_radius, BODY_CORNER_PARTS):
        yield geometry
