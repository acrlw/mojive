"""Measure timeline toolbar groups and the model selector without splitting controls."""

from math import inf

from mojive.design.native.geometry import Rect, budget_margins, part_scope
from mojive.ui.workspace.component_layout import (
    FlowLayout,
    component_flow,
    control_size,
    flow_layout,
)


def toolbar_group_layout(ui, x, y, width, groups, *, column_gap, row_gap):
    """Wrap semantic groups as units, with local wrapping only when one cannot fit."""
    style, scale = ui.component_style("timeline_toolbar"), ui.s
    pads = style.edges("padding", scale)
    margins = budget_margins(style.edges("margin", scale), width, inf)
    available = max(0, width - margins[0] - margins[2])
    frame_width = min(available, style.dimension("width", available / scale) * scale)
    content_width = max(0, frame_width - pads[0] - pads[2])
    gap = style.get("columnGap", style.get("gap", column_gap)) * scale
    vertical_gap = style.get("rowGap", style.get("gap", row_gap)) * scale
    local = tuple(
        flow_layout(
            Rect(0, 0, content_width, 0),
            sizes,
            column_gap=gap,
            row_gap=vertical_gap,
        )
        for sizes in groups
    )
    box, outer = component_flow(
        ui,
        "timeline_toolbar",
        x,
        y,
        width,
        tuple(
            (min(content_width, item.width + (gap if index < len(local) - 1 else 0)), item.height)
            for index, item in enumerate(local)
        ),
        key="timeline-toolbar",
        column_gap=column_gap,
        row_gap=row_gap,
        observe=False,
    )
    rectangles = tuple(
        Rect(group.x + item.x, group.y + item.y, item.width, item.height)
        for group, contents in zip(outer.rectangles, local, strict=True)
        for item in contents.rectangles
    )
    ui.record_component_box(
        "timeline_toolbar",
        box,
        key="timeline-toolbar",
        fields=[list(r.tuple()) for r in rectangles],
    )
    group_bounds = tuple(
        Rect(group.x, group.y, contents.width, contents.height)
        for group, contents in zip(outer.rectangles, local, strict=True)
    )
    return box, FlowLayout(rectangles, outer.width, outer.height), group_bounds


@part_scope("timeline_model_group")
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
