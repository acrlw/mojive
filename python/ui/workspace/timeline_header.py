"""Measure the timeline model selector and add action as one control group."""

from mojive.ui.workspace.component_layout import component_flow, control_size


def model_header_layout(ui, x, y, width, *, observe=False):
    scale = ui.s
    style = ui.component_style("timeline_model_group")
    margins = style.edges("margin", scale)
    pads = style.edges("padding", scale)
    gap = style.get("columnGap", 8) * scale
    available = max(0, width - margins[0] - margins[2] - pads[0] - pads[2])
    add_width, add_height = control_size(ui, "timeline-add-key", 26, 28, icon="plus")
    add_width = min(add_width, available)
    gap = min(gap, max(0, available - add_width))
    select = ui.component_style("select")
    select_margin = select.edges("margin", scale)
    select_height = (
        (ui.design.primitives.field_height("select") if ui.design is not None else 28) * scale
        + select_margin[1]
        + select_margin[3]
    )
    return component_flow(
        ui,
        "timeline_model_group",
        x,
        y,
        width,
        ((max(0, available - add_width - gap), select_height), (add_width, add_height)),
        key="timeline-model-group",
        column_gap=8,
        observe=observe,
    )
