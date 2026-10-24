"""Shared scene geometry selector for navigation and content settings."""

from imgui_bundle import imgui

from mojive.render.geometry import GeometryView, geometry_view, set_geometry_view
from mojive.ui.controls import segmented_control_width
from mojive.ui.workspace.design import BODY_SIZE, DESIGN

GEOMETRY_VIEWS = tuple(GeometryView)
GEOMETRY_ICONS = ("geometry-default", "visual", "collision", "both")


def geometry_selector(ui, ctx, x, y, width, *, key):
    """Keep scene and settings choices identical while inheriting their own owner style."""
    labels = tuple(ctx.tr(view.value.title()) for view in GEOMETRY_VIEWS)
    current = geometry_view(ctx.backend)
    with ui.component_scope("segment"):
        style = ui.component_style("button")
    font = style.get("fontSize", BODY_SIZE)
    imgui.push_font(ui.resolve_font(style), font * ui.s)
    padding = style.edges("padding", ui.s, (6, 0, 6, 0))
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding,
        ((padding[0] + padding[2]) / 2, DESIGN.control_padding_y(font) * ui.s),
    )
    # Keep captions legible before adding redundant icons. Combined content
    # needs a full text inset on both sides, not just enough width to avoid clipping.
    icon_inset = max(0, 12 * ui.s - (padding[0] + padding[2]) / 2)
    show_icons = (
        segmented_control_width(
            labels,
            icons=GEOMETRY_ICONS,
            layout="content",
            icon_width=(style.get("iconSize", 16) + style.get("gap", 5)) * ui.s,
        )
        + 2 * icon_inset * len(labels)
        <= width
    )
    imgui.pop_style_var()
    imgui.pop_font()
    chosen = ui.segmented(
        key,
        x,
        y,
        width / ui.s,
        labels,
        GEOMETRY_VIEWS.index(current) if current in GEOMETRY_VIEWS else -1,
        keys=tuple(key + "-" + view.value for view in GEOMETRY_VIEWS),
        icons=GEOMETRY_ICONS if show_icons else None,
        show_labels=True,
        layout="content",
        tooltips=(
            "Use scene visibility groups",
            "Show visual geometry",
            "Show collision geometry",
            "Show visual and collision geometry",
        ),
    )
    if chosen >= 0 and GEOMETRY_VIEWS[chosen] != current:
        set_geometry_view(ctx.backend, GEOMETRY_VIEWS[chosen])
    return ui.hits[key][3]
