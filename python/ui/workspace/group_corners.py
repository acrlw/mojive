"""Resolve painted action groups and their children in one native corner budget."""

from dataclasses import replace
from functools import wraps

from mojive.design.native.geometry import corner_radius_scope
from mojive.ui.workspace.component_layout import control_radius_limit
from mojive.ui.workspace.corner_geometry import nested_corners
from mojive.ui.workspace.menu_geometry import native_edges
from mojive.ui.workspace.panel_geometry import corner_inset

ACTION_PARTS = ("button", "icon_button")


def action_corners(ui, part, fallback):
    style = ui.component_style(part)
    padding = native_edges(style, "padding", ui.s, fallback)
    slot = style.get("slotSize", 36) if part == "toolbox" else 26 if part == "transport" else 28
    height = slot if part == "toolbox" else 28
    limit = control_radius_limit(
        ui,
        "action-corner-icon",
        slot,
        height,
        icon="select",
        icon_size=style.get("iconSize", 20) if part == "toolbox" else None,
    )
    if part != "toolbox":
        limit = min(limit, control_radius_limit(ui, "action-corner-label", 80, 28, label="Action"))
    child = min(limit, ui.component_style("icon_button").get("radius", 6) * ui.s)
    corners = nested_corners(style, ui.s, padding, child)
    if corners.inner_radius > limit:
        # Both native rounded rectangles and G3 controls have a finite corner
        # budget. Preserve a large authored parent by growing its actual offset.
        inset = corner_inset(corners.radius, max(padding), limit)
        corners = nested_corners(style, ui.s, (inset,) * 4, child)
    return corners


def inset_action_box(box, corners):
    return replace(box, content=box.frame.inset(*corners.padding), padding=corners.padding)


def centered_action_corners(corners, sizes):
    """Match end buttons' horizontal inset to their centered vertical inset."""
    height = max(size[1] for size in sizes)
    edge_height = min(sizes[0][1], sizes[-1][1])
    inset = corners.padding[0] + max(0, height - edge_height) / 2
    return replace(
        corners,
        padding=(inset, corners.padding[1], inset, corners.padding[3]),
        inner_radius=max(0, corners.radius - inset),
    )


def action_corner_scope(part, padding):
    """Keep local parent geometry authoritative over generic control radii."""

    def decorate(function):
        @wraps(function)
        def draw(owner, *args, **kwargs):
            ui = owner.ui if hasattr(owner, "ui") else owner.workspace.ui
            corners = action_corners(ui, part, padding)
            with corner_radius_scope(corners.inner_radius, ACTION_PARTS):
                return function(owner, *args, **kwargs)

        return draw

    return decorate
