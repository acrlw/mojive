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
from mojive.ui.panels.presentation import property_tooltip
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.theme import mix_color as _mix_color


def _compact_transform(width: float, style_scale: float) -> bool:
    return float(width) < 210.0 * float(style_scale)


def _property_tooltip(ctx, text, *, hovered=False):
    property_tooltip(getattr(ctx, "inspector_presentation", None), text, hovered=hovered)


def _begin_property_table(
    table_id: str, ctx: PanelContext | None = None, *, labels: tuple[str, ...] = ()
) -> bool:
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.begin_table(table_id)
    return (
        begin_property_table(table_id, labels=labels) if labels else begin_property_table(table_id)
    )


def _end_property_table(ctx):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        presentation.end_table()
    else:
        imgui.end_table()


def _property_section(
    ctx: PanelContext, label: str, flags: int = imgui.TreeNodeFlags_.default_open
) -> bool:
    """Draw one translated, initially open Inspector property group."""

    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.section(label, flags)
    return imgui.collapsing_header(ctx.tr(label), flags)


def _property_control_row(
    ctx: PanelContext,
    label: str,
    *,
    tooltip: str = "",
    unit: str = "",
    toggle_unit=None,
    toggle: bool = False,
    components: int = 1,
    control_part: str = "input_number",
    control_labels=(),
) -> None:
    """Advance a property table to one left-label/right-control row."""

    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        presentation.row(
            label,
            tooltip,
            unit=unit,
            toggle_unit=toggle_unit,
            toggle=toggle,
            components=components,
            control_part=control_part,
            control_labels=control_labels,
        )
    else:
        property_row(ctx.tr(label), tooltip=ctx.tr(tooltip) if tooltip else "")


def _property_button(ctx, label, size=None):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.button(label, size)
    return imgui.button(label) if size is None else imgui.button(label, size)


def _property_combo(ctx, key, index, labels, flags=0):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.combo(key, index, labels, flags)
    return imgui.combo(key, index, labels, flags)


def _property_begin_combo(ctx, key, preview, flags=0):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.begin_combo(key, preview, flags)
    return imgui.begin_combo(key, preview, flags)


def _property_choice(ctx, label, selected=False, *, enabled=True):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.choice(label, selected, enabled=enabled)
    flags = 0 if enabled else imgui.SelectableFlags_.disabled
    return imgui.selectable(label, selected, flags)


def _property_end_combo(ctx):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        presentation.end_combo()
    else:
        imgui.end_combo()


def _property_input_text(ctx, key, value, flags=0, *, mono=False):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        if mono:
            return presentation.text(key, value, flags, mono=True)
        return presentation.text(key, value, flags)
    return imgui.input_text(key, value, flags)


def _property_segmented(ctx, key, labels, selected, *, icons=None, theme=None):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.segmented(key, labels, selected, icons=icons)
    from mojive.ui.panels import segmented_control

    return segmented_control(key, labels, selected, icons=icons, theme=theme or ctx.theme)


def _property_numeric(ctx, item_id, value, speed=1.0, lo=0.0, hi=0.0, fmt="%.3f", flags=0):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.numeric(item_id, value, speed, lo, hi, fmt, flags)
    args = (item_id, value, speed, lo, hi, fmt)
    return imgui.drag_float(*args, flags) if flags else imgui.drag_float(*args)


def _property_integer(ctx, item_id, value, speed=1.0, lo=0, hi=0, fmt="%d", flags=0):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        changed, value = presentation.numeric(
            item_id, value, speed, lo, hi, fmt, flags, integer=True
        )
        return changed, int(value)
    args = (item_id, value, speed, lo, hi, fmt)
    return imgui.drag_int(*args, flags) if flags else imgui.drag_int(*args)


def _property_numbers(ctx, item_id, values, speed=1.0, lo=0.0, hi=0.0, fmt="%.3f", flags=0):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.numbers(item_id, values, speed, lo, hi, fmt, flags)
    draw = imgui.drag_float2 if len(values) == 2 else imgui.drag_float3
    args = (item_id, values, speed, lo, hi, fmt)
    return draw(*args, flags) if flags else draw(*args)


def _property_toggle(ctx, item_id, value):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.toggle(item_id, value)
    return imgui.checkbox(item_id, value)


def _property_value_rail(ctx, item_id, value, bounds, **options):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.value_rail(item_id, value, bounds, **options)
    return value_rail(ctx, item_id, value, bounds, **options)


def _property_text(ctx, item_id, value, flags):
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.text(item_id, value, flags)
    return imgui.input_text(item_id, value, flags)


def _property_button_row(
    ctx: PanelContext,
    label: str,
    buttons: tuple[str, ...],
    *,
    enabled: tuple[bool, ...] | None = None,
) -> tuple[bool, ...]:
    """Draw a wrapping row with optional availability for each independent action."""

    translated = tuple(ctx.tr(button) for button in buttons)
    _property_control_row(ctx, label, control_part="button_row", control_labels=translated)
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.button_row("property-actions/" + label, translated, enabled)
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
        disabled = enabled is not None and not enabled[index]
        if disabled:
            imgui.begin_disabled()
        pressed.append(imgui.button(f"{button}##{label}-{index}"))
        if disabled:
            imgui.end_disabled()
    return tuple(pressed)


def _property_color_edit3(ctx: PanelContext, item_id: str, value):
    """Keep RGB editors usable when a scaled control column becomes narrow."""

    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.color(item_id, value)
    flags = imgui.ColorEditFlags_.none
    if imgui.get_content_region_avail().x < 150.0 * ctx.style_scale:
        flags |= imgui.ColorEditFlags_.no_inputs
    return imgui.color_edit3(item_id, value, flags)


def _property_color_edit4(ctx: PanelContext, item_id: str, value):
    """Keep RGBA editors usable when a scaled control column becomes narrow."""

    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.color(item_id, value)
    flags = imgui.ColorEditFlags_.none
    if imgui.get_content_region_avail().x < 190.0 * ctx.style_scale:
        flags |= imgui.ColorEditFlags_.no_inputs
    return imgui.color_edit4(item_id, value, flags)


def _edited_float_components(original, edited) -> tuple[float, ...]:
    """Retain authored precision for components only round-tripped through ImGui floats."""
    return tuple(
        float(before) if float(after) == float(np.float32(before)) else float(after)
        for before, after in zip(original, edited, strict=True)
    )


def _property_solver_rows(
    ctx: PanelContext,
    name: str,
    reference: tuple[float, float],
    impedance: tuple[float, float, float, float, float],
    *,
    label_prefix: str = "",
    reference_tooltip: str = "",
) -> tuple[bool, tuple[float, float], tuple[float, float, float, float, float]]:
    """Edit the shared solver rows, preserving untouched rows and vector components."""
    _property_control_row(ctx, f"{label_prefix}solver reference", components=2)
    reference_changed, reference_values = _property_numbers(
        ctx, f"##{name}_solver_reference", reference, 0.001, -1000000.0, 1000000.0, "%.5g"
    )
    if reference_tooltip:
        _property_tooltip(ctx, ctx.tr(reference_tooltip))
    _property_control_row(ctx, f"{label_prefix}impedance min / max / width", components=3)
    first_changed, first = _property_numbers(
        ctx, f"##{name}_impedance_first", impedance[:3], 0.001, 0.0, 1.0, "%.5g"
    )
    _property_control_row(ctx, f"{label_prefix}impedance midpoint / power", components=2)
    shape_changed, shape = _property_numbers(
        ctx, f"##{name}_impedance_shape", impedance[3:], 0.01, 0.0, 1000.0, "%.4g"
    )
    updated_reference = (
        _edited_float_components(reference, reference_values) if reference_changed else reference
    )
    updated_impedance = (
        *(_edited_float_components(impedance[:3], first) if first_changed else impedance[:3]),
        *(_edited_float_components(impedance[3:], shape) if shape_changed else impedance[3:]),
    )
    changed = updated_reference != reference or updated_impedance != impedance
    return changed, updated_reference, updated_impedance


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
    unit: str = "",
) -> tuple[bool, np.ndarray]:
    """Draw one XYZ triplet as a property-table control with joined axis fields."""

    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.vector_row(
            node,
            label,
            name,
            values,
            editable=editable,
            speed=speed,
            lo=lo,
            hi=hi,
            fmt=fmt,
            reset_values=reset_values,
            tooltip=label_tooltip,
            unit=unit,
        )
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
    units: tuple[str, ...] = (),
) -> tuple[tuple[bool, np.ndarray], ...]:
    presentation = getattr(ctx, "inspector_presentation", None)
    if presentation is not None:
        return presentation.vectors(node, table_id, rows, editable, units=units)
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
        _property_tooltip(
            ctx, pointer_hint(ctx, PointerAction.PROPERTY_COPY, ctx.tr("Copy XYZ")), hovered=True
        )

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
        _property_tooltip(
            ctx,
            f"{ctx.tr('Reset')}: {reset_value:g}" if editable else ctx.tr("Read only"),
            hovered=True,
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
        _property_tooltip(ctx, hint, hovered=True)

    return reset, edited, next_value


def _format_vector(values) -> str:
    return ", ".join(f"{float(value):.6g}" for value in values)
