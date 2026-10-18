"""Resolve one native corner budget for floating windows and their contents."""

from dataclasses import dataclass
from math import ceil

from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit


@dataclass(frozen=True, slots=True)
class FloatingPanelGeometry:
    padding: float
    radius: float
    inner_radius: float
    authored_radius: bool


def floating_panel_geometry(ui, name=None):
    """Use moderate defaults and derive child corners after native inset rounding."""
    timeline = name == "Timeline"
    inset = native_unit((6 if name in ("Timeline", "Settings") else 8) * ui.s)
    style = ui.component_style("panel")
    authored = style.explicit("radius")
    radius = DESIGN.control_radius * ui.s + native_unit(8 * ui.s)
    if authored:
        radius = max(0, style.get("radius", 14) * ui.s)
    elif timeline:
        body = ui.component_style("timeline_body")
        if body.explicit("radius"):
            radius = max(0, body.get("radius", 8) * ui.s) + inset
    if not timeline:
        inset = corner_inset(radius, inset, framed_control_radius_limit(ui))
    return FloatingPanelGeometry(inset, radius, max(0, radius - inset), authored)


def framed_control_radius_limit(ui):
    """Respect ImGui's half-height minus one native-unit path limit."""
    height = DESIGN.control_height
    if ui.design is not None:
        height = min(
            height,
            *(
                ui.design.primitives.field_height(part, height)
                for part in ("select", "input_number", "input_text")
            ),
        )
    return max(0, height * ui.s / 2 - 1)


def corner_inset(radius, requested, child_limit):
    """Grow an integer inset when an authored radius would force a capsule clamp."""
    required = max(0, radius - child_limit)
    integral = round(required)
    if abs(required - integral) <= native_epsilon(radius, required, child_limit):
        required = integral
    return max(requested, ceil(required))
