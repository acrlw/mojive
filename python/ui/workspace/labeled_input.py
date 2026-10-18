"""Measure a label and numeric field as one wrapping, padded control group."""

from dataclasses import dataclass

from mojive.design.native.geometry import Rect, part_scope


@dataclass(frozen=True)
class LabeledInputLayout:
    width: float
    height: float
    label: Rect
    field: Rect
    stacked: bool


@part_scope("range_bounds")
def measure_labeled_input(ui, available, label, number_size, *, label_width=None):
    s, style = ui.s, ui.component_style("range_bounds")
    padding, margin = style.edges("padding", s), style.edges("margin", s)
    left, top = padding[0] + margin[0], padding[1] + margin[1]
    right, bottom = padding[2] + margin[2], padding[3] + margin[3]
    gap = style.get("columnGap", style.get("gap", 8)) * s
    row_gap = style.get("rowGap", 8) * s
    label_width = ui.measure(label) if label_width is None else label_width
    label_height = ui.component_style("form_label").get("fontSize", 14) * 1.4 * s
    nw, nh = number_size
    budget = max(0, available - left - right)
    stacked = label_width + gap + nw > budget
    if stacked:
        width, height = budget, label_height + row_gap + nh
        label_rect = Rect(left, top, budget, label_height)
        field = Rect(left, top + label_height + row_gap, min(nw, budget), nh)
    else:
        width, height = label_width + gap + nw, max(label_height, nh)
        label_rect = Rect(left, top + (height - label_height) / 2, label_width, label_height)
        field = Rect(left + label_width + gap, top + (height - nh) / 2, nw, nh)
    return LabeledInputLayout(
        width + left + right, height + top + bottom, label_rect, field, stacked
    )


@part_scope("range_bounds")
def draw_labeled_input(ui, key, x, y, layout, label, value, *, format="%.2f s", speed=0.01):
    from mojive.ui.workspace.widgets import TEXT

    lr, nr = layout.label, layout.field
    label_rect = Rect(x + lr.x, y + lr.y, lr.width, lr.height)
    ui.component_text("form_label", label_rect, label, TEXT, align="start")
    result = ui.input(key, x + nr.x, y + nr.y, nr.width, value, speed=speed, format=format)
    box = ui.component_box("range_bounds", x, y, layout.width, layout.height, observe=False)
    ui.record_component_box(
        "range_bounds",
        box,
        key="range-bounds/" + key,
        labelRect=label_rect.tuple(),
        fieldRect=(x + nr.x, y + nr.y, nr.width, nr.height),
        stacked=layout.stacked,
    )
    return result
