"""Measure and draw complete button rows within one physical content budget."""

import math
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.workspace.component_layout import control_frame_size, control_size, flow_layout
from mojive.ui.workspace.component_style import budget_margins
from mojive.ui.workspace.design import DESIGN


@dataclass(frozen=True, slots=True)
class ButtonRowGeometry:
    outer_height: float
    minimum_height: float
    rectangles: tuple[Rect, ...]


@part_scope("button_row")
def measure_button_row(ui, available, labels):
    """Return relative frames for the same styles used by the production buttons."""
    scale, row = ui.s, ui.component_style("button_row")
    padding = row.edges("padding", scale)
    margin = budget_margins(row.edges("margin", scale), available, math.inf)
    available_frame = max(0, available - margin[0] - margin[2])
    frame_width = min(available_frame, row.dimension("width", available_frame / scale) * scale)
    content = Rect(0, 0, frame_width, 0).inset(padding[0], 0, padding[2], 0)
    sizes = []
    for index, label in enumerate(labels):
        parts = ui.design.button_parts(str(index), label) if ui.design else "button"
        child = ui.component_style(parts)
        pads = child.edges("padding", scale, (6, 0, 6, 0))
        natural = ui.measure(label, child.get("fontSize", DESIGN.body_size), font_resolved=True)
        sizes.append(
            control_size(
                ui,
                str(index),
                (natural + pads[0] + pads[2]) / scale,
                DESIGN.control_height,
                label=label,
            )
        )
    flow = flow_layout(
        content,
        sizes,
        column_gap=row.get("columnGap", DESIGN.item_spacing_x) * scale,
        row_gap=row.get("rowGap", DESIGN.row_gap) * scale,
        align=row.get("align", "start"),
        align_items=row.get("alignItems", "center"),
    )
    minimum = flow.height + padding[1] + padding[3]
    height = max(minimum, row.dimension("height", minimum / scale) * scale)
    free = max(0, height - minimum)
    align = row.get("alignItems", "center")
    offset = free * (0.5 if align == "center" else 1 if align == "end" else 0)
    return ButtonRowGeometry(
        height + margin[1] + margin[3],
        minimum,
        tuple(
            Rect(
                rect.x - content.x,
                rect.y + offset,
                rect.width,
                rect.height + free
                if align == "stretch" and len(flow.rectangles) == 1
                else rect.height,
            )
            for rect in flow.rectangles
        ),
    )


@part_scope("button_row")
def draw_button_row(ui, key, labels, enabled=None, *, available=None):
    pos = imgui.get_cursor_screen_pos()
    available = imgui.calc_item_width() if available is None else available
    geometry = measure_button_row(ui, available, labels)
    box = ui.component_box(
        "button_row",
        pos.x,
        pos.y,
        available,
        geometry.outer_height,
        key=key,
        minimum_height=geometry.minimum_height,
        observe=False,
    )
    children = tuple(
        Rect(box.content.x + rect.x, box.content.y + rect.y, rect.width, rect.height)
        for rect in geometry.rectangles
    )
    ui.record_component_box(
        "button_row", box, key=key, fields=[list(rect.tuple()) for rect in children]
    )
    result = []
    for index, (label, rectangle) in enumerate(zip(labels, geometry.rectangles, strict=True)):
        width, height = control_frame_size(
            ui, key + "/" + str(index), rectangle.width, rectangle.height, label=label
        )
        result.append(
            ui.button(
                key + "/" + str(index),
                box.content.x + rectangle.x,
                box.content.y + rectangle.y,
                width,
                height,
                width_limit=width * ui.s,
                label=label,
                enabled=rectangle.width > 0 and (enabled is None or enabled[index]),
                translate=False,
            )
        )
    imgui.set_cursor_screen_pos((pos.x, pos.y + box.outer.height))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, 0))
    imgui.dummy((0, 0))
    imgui.pop_style_var()
    return tuple(result)
