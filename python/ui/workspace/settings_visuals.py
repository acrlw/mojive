"""Measure rendering option grids before routing existing setting commands."""

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import Rect
from mojive.ui.panels.settings import render_flag_label
from mojive.ui.workspace.component_layout import component_flow
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.property_style import advance
from mojive.ui.workspace.widgets import MUTED


def _label_geometry(ui):
    style = ui.component_style("form_label")
    font = style.get("fontSize", DESIGN.body_size)
    margin = style.edges("margin", ui.s)
    line = max(font, style.line_height(font * 1.4)) * ui.s
    return font, margin, line + margin[1] + margin[3]


def _label(ui, key, rect, text):
    box = ui.component_box("form_label", *rect.tuple(), key=key)
    ui.component_text("form_label", box.content, text, MUTED, translate=False)
    font = ui.component_style("form_label").get("fontSize", DESIGN.body_size)
    if (
        ui.measure(text, font, font_resolved=True, translate=False) > box.content.width
        and imgui.is_window_hovered()
        and imgui.is_mouse_hovering_rect((rect.x, rect.y), (rect.right, rect.bottom))
    ):
        ui.tooltip(text, translate=False)


def flag_table(panel, ctx, key, flags, groups):
    ui, s = panel.workspace.ui, panel.workspace.s
    pos, width = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail().x
    item = ui.component_style("visual_group_item")
    gap = item.get("columnGap", item.get("gap", DESIGN.label_gap)) * s
    size = ui.metric("checkbox", "size", DESIGN.checkbox_size) * s
    font, label_margin, label_height = _label_geometry(ui)
    labels = tuple(render_flag_label(flag, ctx.tr, localized=True) for flag in flags)
    required = (
        max(
            (ui.measure(label, font, font_resolved=True, translate=False) for label in labels),
            default=0,
        )
        + label_margin[0]
        + label_margin[2]
        + size
        + gap
    )
    columns = max(1, min(groups, int(width / max(s, required))))
    spacing = ui.metric("visual_group", "columnGap", DESIGN.item_spacing_x) * s
    cell = max(s, (width - spacing * (columns - 1)) / columns)
    height = max(DESIGN.control_height * s, size, label_height)
    box, flow = component_flow(
        ui,
        "visual_group",
        pos.x,
        pos.y,
        width,
        [(cell, height)] * len(flags),
        key=key,
        column_gap=spacing / s,
        row_gap=DESIGN.row_gap,
    )
    for flag, label, rect in zip(flags, labels, flow.rectangles, strict=True):
        label_rect = Rect(rect.x, rect.y, max(0, rect.width - size - gap), rect.height)
        _label(ui, key + "/label/" + flag.value, label_rect, label)
        imgui.set_cursor_screen_pos((rect.right - size, rect.y + (rect.height - size) / 2))
        panel._flag_row(ctx, flag, label)
        if ui.design is not None:
            ui.design.record(
                key + "/" + flag.value, ("visual_group_item",), rect.tuple(), "visual_group_item"
            )
    advance(ui, pos.x, box.outer.bottom + DESIGN.row_gap * s)


def visual_groups(panel, ctx):
    families = ctx.session.visual_groups()
    if not families or not panel._collapsing_header(
        f"{ctx.tr('visual groups')}###visual_groups", imgui.TreeNodeFlags_.default_open
    ):
        return
    ui, s = panel.workspace.ui, panel.workspace.s
    start = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    checkbox = ui.metric("checkbox", "size", DESIGN.checkbox_size) * s
    font, label_margin, label_height = _label_geometry(ui)
    item = ui.component_style("visual_group_item")
    gap = item.get("columnGap", item.get("gap", DESIGN.label_gap)) * s
    row_gap = ui.metric("visual_group", "rowGap", DESIGN.row_gap) * s
    label_width = min(
        width * 0.35,
        max(ui.measure(ctx.tr(f.category), font, font_resolved=True) for f in families)
        + label_margin[0]
        + label_margin[2],
    )
    x, y = start.x, start.y
    for family in families:
        cell_width = max(checkbox, (width - label_width - gap * 6) / 6)
        height = max(28 * s, checkbox, label_height)
        sizes = [(label_width, height)] + [(cell_width, height)] * len(family.visible)
        box, flow = component_flow(
            ui,
            "visual_group",
            x,
            y,
            width,
            sizes,
            key="visual-group-" + family.category,
            column_gap=gap / s,
        )
        _label(
            ui, "visual-group-label-" + family.category, flow.rectangles[0], ctx.tr(family.category)
        )
        for index, (visible, rect) in enumerate(
            zip(family.visible, flow.rectangles[1:], strict=True)
        ):
            imgui.set_cursor_screen_pos(
                (rect.x + (rect.width - checkbox) / 2, rect.y + (rect.height - checkbox) / 2)
            )
            changed, value = panel._checkbox(
                f"##visual_group_{family.category}_{index}", visible, ctx.theme
            )
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                ui.tooltip(f"{ctx.tr('visual group')} {index}", translate=False)
            if changed:
                ctx.submit(cmd.SetVisualGroup(family.category, index, value))
            if ui.design is not None:
                ui.design.record(
                    f"visual-group-item-{family.category}-{index}",
                    ("visual_group_item",),
                    rect.tuple(),
                    "visual_group_item",
                )
        y = box.outer.bottom + row_gap
    advance(ui, x, y)
