"""Shared quiet surfaces for editable scene-entity cards."""

from mojive.design.native.geometry import corner_radius_scope


def card_controls(ui, insets):
    """Fit every interactive surface inside the same resolved card inset."""
    radius = max(0, ui.component_style("entity_card").get("radius", 12) * ui.s - min(insets))
    return corner_radius_scope(
        radius, ("button", "icon_button", "input_number", "input_shell", "axis_input")
    )


def card_surface(ui, rect, *, selected=False):
    style = ui.component_style("entity_card")
    ui.rect(
        *rect.tuple(),
        ui.palette("selected" if selected else "child"),
        style.get("radius", 12),
        color_resolved=True,
    )
