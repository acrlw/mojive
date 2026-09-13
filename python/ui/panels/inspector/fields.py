"""Inspector: fields."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive.adapters.base import (
    SceneNode,
)
from mojive.ui.compound_fields import (
    borderless_numeric_input,
    draw_focus_frame,
    draw_joined_field_frame,
)
from mojive.ui.controls import begin_property_table, property_row
from mojive.ui.panels import (
    PanelContext,
    button_row_layout,
    button_width,
    pointer_hint,
    pointer_pressed,
)
from mojive.ui.pointer_bindings import PointerAction


def _compact_transform(width: float, style_scale: float) -> bool:
    return float(width) < 210.0 * float(style_scale)


def _begin_property_table(table_id: str) -> bool:
    return begin_property_table(table_id)


def _property_section(ctx: PanelContext, label: str) -> bool:
    """Draw one translated, initially open Inspector property group."""

    return imgui.collapsing_header(
        ctx.tr(label),
        imgui.TreeNodeFlags_.default_open,
    )


def _property_control_row(
    ctx: PanelContext,
    label: str,
    *,
    tooltip: str = "",
) -> None:
    """Advance a property table to one left-label/right-control row."""

    property_row(ctx.tr(label), tooltip=ctx.tr(tooltip) if tooltip else "")


def _property_button_row(
    ctx: PanelContext,
    label: str,
    buttons: tuple[str, ...],
) -> tuple[bool, ...]:
    """Draw a wrapping row of compact property actions."""

    _property_control_row(ctx, label)
    translated = tuple(ctx.tr(button) for button in buttons)
    widths = tuple(button_width(button) for button in translated)
    layout = button_row_layout(
        widths,
        imgui.get_content_region_avail().x,
        imgui.get_style().item_spacing.x,
    )
    pressed: list[bool] = []
    for index, button in enumerate(translated):
        if layout[index]:
            imgui.same_line()
        pressed.append(imgui.button(f"{button}##{label}-{index}"))
    return tuple(pressed)


def _property_color_edit3(ctx: PanelContext, item_id: str, value):
    """Keep RGB editors usable when a scaled control column becomes narrow."""

    flags = imgui.ColorEditFlags_.none
    if imgui.get_content_region_avail().x < 150.0 * ctx.style_scale:
        flags |= imgui.ColorEditFlags_.no_inputs
    return imgui.color_edit3(item_id, value, flags)


def _property_color_edit4(ctx: PanelContext, item_id: str, value):
    """Keep RGBA editors usable when a scaled control column becomes narrow."""

    flags = imgui.ColorEditFlags_.none
    if imgui.get_content_region_avail().x < 190.0 * ctx.style_scale:
        flags |= imgui.ColorEditFlags_.no_inputs
    return imgui.color_edit4(item_id, value, flags)


def _property_vector_row(
    ctx: PanelContext,
    node: SceneNode,
    label: str,
    name: str,
    values,
    *,
    editable: bool,
    speed: float,
    lo: float,
    hi: float,
    fmt: str,
    reset_values=None,
    label_tooltip: str = "",
) -> tuple[bool, np.ndarray]:
    """Draw one XYZ triplet as a property-table control with joined axis fields."""

    out = np.asarray(values, np.float64).copy()
    resets = (
        np.zeros(3, np.float64)
        if reset_values is None
        else np.asarray(reset_values, np.float64).reshape(3)
    )
    _property_control_row(ctx, label, tooltip=label_tooltip)
    compact = _compact_transform(imgui.get_content_region_avail().x, ctx.style_scale)
    flags = (
        imgui.TableFlags_.sizing_stretch_same
        | imgui.TableFlags_.no_saved_settings
        | imgui.TableFlags_.no_pad_inner_x
        | imgui.TableFlags_.no_pad_outer_x
    )
    columns = 1 if compact else 3
    if not imgui.begin_table(f"##{name}_property_axes_{node.node_id}", columns, flags):
        return False, out
    for axis in "xyz"[:columns]:
        imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)
    changed = False
    for axis, axis_label in enumerate("XYZ"):
        if compact:
            imgui.table_next_row()
        imgui.table_next_column()
        reset, edited, value = _axis_field(
            ctx,
            node,
            name,
            axis,
            axis_label,
            float(out[axis]),
            editable=editable,
            speed=speed,
            lo=lo,
            hi=hi,
            fmt=fmt,
            grouped=not compact,
        )
        if reset:
            out[axis] = resets[axis]
            changed = True
        if edited:
            out[axis] = value
            changed = True
    imgui.end_table()
    return changed, out


def vector_layout(
    width: float, label_width: float, axes_width: float, scale: float, previous: int = -1
) -> int:
    """Use inline, label-above, or stacked axes, with room to expand before switching back."""
    margin = 24.0 * scale
    inline = label_width + axes_width
    if width >= inline + (margin if previous > 0 else 0):
        return 0
    if width >= axes_width + (margin if previous == 2 else 0):
        return 1
    return 2


def _vector_fields(
    ctx: PanelContext,
    node: SceneNode,
    table_id: str,
    rows,
    *,
    editable: bool | tuple[bool, ...] = True,
    label_width: float = 0.0,
) -> tuple[tuple[bool, np.ndarray], ...]:
    label_width = max(
        label_width,
        max(imgui.calc_text_size(row[0]).x for row in rows) + 10.0 * ctx.style_scale,
    )
    width = imgui.get_content_region_avail().x
    minimum_axes_width = 3.0 * _axis_field_min_width(ctx.style_scale)
    storage = imgui.get_state_storage()
    layout_id = imgui.get_id(table_id + "##vector-layout")
    layout = vector_layout(
        width, label_width, minimum_axes_width, ctx.style_scale, storage.get_int(layout_id, -1)
    )
    storage.set_int(layout_id, layout)
    compact = layout != 0
    flags = (
        imgui.TableFlags_.sizing_stretch_same
        | imgui.TableFlags_.no_saved_settings
        | imgui.TableFlags_.no_pad_inner_x
        | imgui.TableFlags_.no_pad_outer_x
    )
    columns = 1 if compact else 4
    if not imgui.begin_table(table_id, columns, flags):
        return tuple((False, np.asarray(row[1], np.float64).copy()) for row in rows)
    if compact:
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
    else:
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_fixed, label_width)
        for axis in "xyz":
            imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)
    result = tuple(
        _vector_row(
            ctx,
            node,
            name,
            values,
            editable=editable if isinstance(editable, bool) else editable[index],
            speed=speed,
            fmt=fmt,
            compact=compact,
            stacked=layout == 2,
            reset_values=reset_values,
        )
        for index, (name, values, speed, fmt, reset_values) in enumerate(rows)
    )
    imgui.end_table()
    return result


def _vector_row(
    ctx: PanelContext,
    node: SceneNode,
    name: str,
    values,
    *,
    editable: bool,
    speed: float,
    fmt: str,
    compact: bool,
    stacked: bool = False,
    reset_values=None,
) -> tuple[bool, np.ndarray]:
    out = np.asarray(values, np.float64).copy()
    resets = (
        np.zeros(3, np.float64)
        if reset_values is None
        else np.asarray(reset_values, np.float64).reshape(3)
    )
    imgui.table_next_row()
    imgui.table_next_column()
    if not compact:
        imgui.align_text_to_frame_padding()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ctx.theme.text_disabled))
    imgui.text(name)
    imgui.pop_style_color()
    group_hovered = imgui.is_item_hovered()
    if group_hovered and pointer_pressed(ctx, PointerAction.PROPERTY_COPY):
        imgui.set_clipboard_text(_format_vector(out))
    if group_hovered:
        imgui.set_tooltip(pointer_hint(ctx, PointerAction.PROPERTY_COPY, ctx.tr("Copy XYZ")))

    if compact:
        imgui.table_next_row()
        imgui.table_next_column()
        row_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        columns = 1 if stacked else 3
        if not imgui.begin_table(f"##{name}_axes_{node.node_id}", columns, row_flags):
            return False, out
        for axis in "xyz"[:columns]:
            imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)

    changed = False
    for axis, label in enumerate("XYZ"):
        imgui.table_next_column()
        reset, edited, value = _axis_field(
            ctx,
            node,
            name,
            axis,
            label,
            float(out[axis]),
            editable=editable,
            speed=speed,
            fmt=fmt,
            grouped=not stacked,
            reset_value=float(resets[axis]),
        )
        if reset:
            out[axis] = resets[axis]
            changed = True
        if edited:
            out[axis] = value
            changed = True
    if compact:
        imgui.end_table()
    return changed, out


def _axis_field_min_width(scale: float) -> float:
    return imgui.calc_text_size("-0.000").x + 22.0 * scale + imgui.get_style().frame_padding.x


def _axis_field(
    ctx: PanelContext,
    node: SceneNode,
    name: str,
    axis: int,
    label: str,
    value: float,
    *,
    editable: bool,
    speed: float,
    fmt: str,
    lo: float = 0.0,
    hi: float = 0.0,
    grouped: bool = True,
    reset_value: float = 0.0,
) -> tuple[bool, bool, float]:
    gap = 5.0 * ctx.style_scale if grouped else 0.0
    if grouped:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + axis * gap / 3.0)
    axis_color = ctx.theme.axis_color(axis)
    color = _mix_color(ctx.theme.bg_frame, axis_color, 0.56)
    hovered_color = _mix_color(ctx.theme.bg_frame_hovered, axis_color, 0.72)
    active_color = _mix_color(ctx.theme.bg_frame_active, axis_color, 0.88)
    axis_width = min(
        22.0 * ctx.style_scale,
        max(1.0, imgui.get_content_region_avail().x * 0.4),
    )

    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(draw_list, 2)
    splitter.set_current_channel(draw_list, 1)
    transparent = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
    imgui.push_style_color(imgui.Col_.button, transparent)
    imgui.push_style_color(imgui.Col_.button_hovered, transparent)
    imgui.push_style_color(imgui.Col_.button_active, transparent)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    if not editable:
        imgui.push_style_var(imgui.StyleVar_.disabled_alpha, 1.0)
    imgui.begin_disabled(not editable)
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding, imgui.ImVec2(0.0, imgui.get_style().frame_padding.y)
    )
    imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
    reset = imgui.button(
        f"{label}##{name}_{axis}_{node.node_id}",
        imgui.ImVec2(axis_width, 0.0),
    )
    imgui.pop_style_var()
    imgui.end_disabled()
    if not editable:
        imgui.pop_style_var()
    button_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    button_active = imgui.is_item_active()
    button_lo, button_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    button_id = imgui.get_item_id()
    imgui.pop_style_color(5)
    if button_hovered:
        imgui.set_tooltip(
            f"{ctx.tr('Reset')}: {reset_value:g}" if editable else ctx.tr("Read only")
        )

    group_gap = (2 - axis) * gap / 3.0
    imgui.same_line(0.0, 0.0)
    imgui.set_next_item_width(max(1.0, imgui.get_content_region_avail().x - group_gap))
    imgui.push_style_color(imgui.Col_.frame_bg, transparent)
    imgui.push_style_color(imgui.Col_.frame_bg_hovered, transparent)
    imgui.push_style_color(imgui.Col_.frame_bg_active, transparent)
    imgui.begin_disabled(not editable)
    with borderless_numeric_input():
        edited, next_value = imgui.drag_float(
            f"##{name}_{axis}_{node.node_id}", value, speed, lo, hi, fmt
        )
    imgui.end_disabled()
    imgui.pop_style_color(3)
    field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    field_active = imgui.is_item_active()
    field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    field_id = imgui.get_item_id()
    splitter.set_current_channel(draw_list, 0)
    draw_joined_field_frame(
        draw_list,
        button_lo,
        button_hi,
        field_lo,
        field_hi,
        badge_color=(
            active_color
            if button_active and editable
            else hovered_color
            if button_hovered and editable
            else color
        ),
        field_color=(
            ctx.theme.bg_frame_active
            if field_active
            else ctx.theme.bg_frame_hovered
            if field_hovered and editable
            else ctx.theme.bg_frame
        ),
        rounding=float(imgui.get_style().frame_rounding),
        badge_opacity=float(imgui.get_style().alpha),
        field_opacity=float(imgui.get_style().alpha)
        * (1.0 if editable else float(imgui.get_style().disabled_alpha)),
    )
    splitter.set_current_channel(draw_list, 1)
    if editable:
        draw_focus_frame(
            button_lo,
            button_hi,
            rounding=imgui.get_style().frame_rounding,
            corners=imgui.ImDrawFlags_.round_corners_left,
            item_id=button_id,
        )
        draw_focus_frame(
            field_lo,
            field_hi,
            rounding=imgui.get_style().frame_rounding,
            corners=imgui.ImDrawFlags_.round_corners_right,
            item_id=field_id,
        )
    splitter.merge(draw_list)
    if field_hovered and pointer_pressed(ctx, PointerAction.PROPERTY_COPY):
        imgui.set_clipboard_text(fmt % value)
    if field_hovered:
        hint = pointer_hint(ctx, PointerAction.PROPERTY_COPY, ctx.tr("Copy value"))
        if not editable:
            hint = f"{ctx.tr('Read only')} · {hint}"
        imgui.set_tooltip(hint)

    return reset, edited, next_value


def _format_vector(values) -> str:
    return ", ".join(f"{float(value):.6g}" for value in values)


def _mix_color(background, foreground, amount: float):
    weight = min(1.0, max(0.0, float(amount)))
    return (
        *(
            background[index] + (foreground[index] - background[index]) * weight
            for index in range(3)
        ),
        foreground[3],
    )
