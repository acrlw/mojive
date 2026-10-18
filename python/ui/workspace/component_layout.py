"""Measured rows used by production containers and their design audit bounds."""

from dataclasses import dataclass, replace
from math import inf

from mojive.design.native.geometry import Rect, budget_margins


@dataclass(frozen=True)
class FlowLayout:
    """Keep controls within a content budget and align each completed row."""

    rectangles: tuple[Rect, ...]
    width: float
    height: float


def record_group(ui, part, key, rectangles, **geometry):
    """Observe the union of already placed children without allocating layout."""
    from mojive.design.groups import scoped_parts

    if ui.design is None or not ui.design.runtime.observes(scoped_parts((part,))):
        return
    rectangles = tuple(rectangles)
    if not rectangles:
        return
    left = min(rect[0] for rect in rectangles)
    top = min(rect[1] for rect in rectangles)
    right = max(rect[0] + rect[2] for rect in rectangles)
    bottom = max(rect[1] + rect[3] for rect in rectangles)
    ui.design.record(
        key,
        (part,),
        (left, top, right - left, bottom - top),
        "group",
        fields=[list(rect) for rect in rectangles],
        **geometry,
    )


def flow_layout(
    content, sizes, *, column_gap=0.0, row_gap=0.0, align="start", align_items="center"
):
    """Place physical control sizes without splitting a control across rows."""
    rows = []
    row, used, row_height = [], 0.0, 0.0
    for width, height in sizes:
        width = min(max(0.0, width), content.width)
        height = max(0.0, height)
        required = width + (column_gap if row else 0.0)
        if row and used + required > content.width + 0.5:
            rows.append((row, used, row_height))
            row, used, row_height = [], 0.0, 0.0
            required = width
        row.append((used + (column_gap if row else 0.0), width, height))
        used += required
        row_height = max(row_height, height)
    if row:
        rows.append((row, used, row_height))
    rectangles, y = [], content.y
    for row, used, row_height in rows:
        offset = max(0.0, content.width - used) * (
            0.5 if align == "center" else 1.0 if align in ("end", "right") else 0.0
        )
        for x, width, height in row:
            dy = max(0.0, row_height - height) * (
                0.5
                if align_items == "center"
                else 1.0
                if align_items in ("end", "flex-end")
                else 0.0
            )
            rectangles.append(Rect(content.x + offset + x, y + dy, width, height))
        y += row_height + row_gap
    height = max(0.0, y - content.y - (row_gap if rows else 0.0))
    return FlowLayout(tuple(rectangles), max((r[1] for r in rows), default=0.0), height)


def component_flow(
    ui,
    part,
    x,
    y,
    width,
    sizes,
    *,
    key=None,
    padding=(0, 0, 0, 0),
    column_gap=0,
    row_gap=0,
    observe=True,
    resolved_padding=None,
):
    """Measure a wrapping container before publishing its final native box."""
    style, scale = ui.component_style(part), ui.s
    margins = budget_margins(style.edges("margin", scale), width, inf)
    pads = (
        style.edges("padding", scale, tuple(v / scale for v in padding))
        if resolved_padding is None
        else resolved_padding
    )
    available = max(0, width - margins[0] - margins[2])
    frame_width = min(available, style.dimension("width", available / scale) * scale)
    content = Rect(
        x + margins[0] + min(pads[0], frame_width),
        y + margins[1] + pads[1],
        max(0, frame_width - pads[0] - pads[2]),
        0,
    )
    layout = flow_layout(
        content,
        sizes,
        column_gap=style.get("columnGap", style.get("gap", column_gap)) * scale,
        row_gap=style.get("rowGap", style.get("gap", row_gap)) * scale,
        align=style.get("align", "start"),
        align_items=style.get("alignItems", "center"),
    )
    minimum = layout.height + pads[1] + pads[3]
    height = max(minimum, style.dimension("height", minimum / scale) * scale)
    box = ui.component_box(
        part,
        x,
        y,
        width,
        height + margins[1] + margins[3],
        key=key,
        padding=padding,
        minimum_height=minimum,
        observe=False,
    )
    if resolved_padding is not None:
        box = replace(box, content=box.frame.inset(*pads), padding=pads)
    if observe:
        ui.record_component_box(
            part, box, key=key, fields=[list(r.tuple()) for r in layout.rectangles]
        )
    return box, layout


def _control_box(ui, key, width, height, *, label="", icon=None, icon_size=None, icon_role=None):
    from mojive.design.native.geometry import classify_button

    parts = (
        ui.design.button_parts(key, label) if ui.design is not None else classify_button(key, label)
    )
    style, scale = ui.component_style(parts), ui.s
    inset = 0 if icon and not label else 8 if icon or "menu_trigger" in parts else 6
    padding = (inset * scale, 0, inset * scale, 0)
    font_height = style.line_height(style.get("fontSize", 14) * 1.4) * scale if label else 0
    if icon:
        from mojive.ui.workspace.widget_icons import icon_role_canvas

        fallback = icon_role_canvas(icon_role or ("inline" if label else "toolbar"))
        extent = (
            style.get("iconSize", fallback)
            if style.explicit("iconSize") or icon_size is None
            else icon_size
        )
        engine = getattr(getattr(ui.design, "runtime", None), "paint_context", None)
        if not style.explicit("iconSize") and engine is not None:
            extent = engine.inherited("iconSize", extent)
        icon_height = extent * scale
    else:
        icon_height = 0
    return style.box(
        0,
        0,
        width * scale,
        height * scale,
        scale=scale,
        padding=padding,
        intrinsic=(0, max(font_height, icon_height)),
    )


def control_size(ui, key, width, height, *, label="", icon=None, icon_size=None, icon_role=None):
    """Measure the exact specialization before a parent allocates its row budget."""
    box = _control_box(
        ui, key, width, height, label=label, icon=icon, icon_size=icon_size, icon_role=icon_role
    )
    return box.outer.width, box.outer.height


def control_radius_limit(ui, key, width, height, **options):
    """Use the painted frame, including authored dimensions and intrinsic growth."""
    frame = _control_box(ui, key, width, height, **options).frame
    return max(0, min(frame.width, frame.height) / 2 - 1)


def control_frame_size(ui, key, width, height, *, label="", icon=None):
    """Return the frame budget to pass after margins have already been allocated."""
    from mojive.design.native.geometry import classify_button

    parts = (
        ui.design.button_parts(key, label) if ui.design is not None else classify_button(key, label)
    )
    margins = ui.component_style(parts).edges("margin", ui.s)
    return max(0, width - margins[0] - margins[2]) / ui.s, max(
        0, height - margins[1] - margins[3]
    ) / ui.s
