"""Candidate icons, capsule controls, hints and numeric input specimens."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive.geometry2d.curves import CORNER_SMOOTHING
from mojive.ui.icons import (
    ICON_DEFAULT_PADDING,
    ICON_STROKE,
    ICON_TUNING_DEFAULTS,
    IconStyle,
    draw_concept_icon,
    draw_icon_label,
)
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.keyframe_editor import controls as keyframes_panel_module
from mojive.ui.panels import search_input, searchable_ordered_list_header
from mojive.ui.viewport_widgets import (
    OVERLAY_GEOMETRY,
    _viewport_control_colors,
    draw_mouse_hint_glyph,
    draw_tool_glyph,
    keycap_rounding,
)
from mojive.ui.viewport_widgets.chrome import draw_selection_disc

from ..ui_capsule_geometry import draw_capsule_shell
from ..ui_redesign import _capsule, _recording_menu, draw_reset_glyph
from .fixtures import CONCEPT_THEME, OVERLAY_ICON_RADIUS, OVERLAY_STATE_RADIUS
from .layout import _draw_segmented, _flags, _property_label
from .state import ProbeState


def _draw_concept_control_icon(
    draw,
    center,
    size: float,
    kind: str,
    color,
    *,
    padding: float = ICON_DEFAULT_PADDING,
    stroke_width: float = ICON_STROKE,
    state: ProbeState | None = None,
) -> None:
    """Adapt Icon Library panel candidates to the shared control callback."""

    name = f"panel-{kind}"
    if state is not None:
        padding = state.icon_padding_for_glyph(name)
        stroke_width = state.icon_stroke_for_glyph(name)
    draw_concept_icon(
        draw,
        state.icon_center(name, center, size) if state is not None else center,
        size,
        name,
        color,
        padding=padding,
        stroke_width=stroke_width,
        tuning=state.icon_tuning() if state is not None else ICON_TUNING_DEFAULTS,
        alignment=state.icon_alignment_for_glyph(name) if state is not None else None,
    )


def _draw_concept_projection_icon(
    draw,
    center,
    size: float,
    kind: str,
    color,
    *,
    padding: float = ICON_DEFAULT_PADDING,
    stroke_width: float = ICON_STROKE,
    state: ProbeState | None = None,
) -> None:
    name = "panel-perspective" if kind == "persp" else "panel-orthographic"
    if state is not None:
        padding = state.icon_padding_for_glyph(name)
        stroke_width = state.icon_stroke_for_glyph(name)
    draw_concept_icon(
        draw,
        state.icon_center(name, center, size) if state is not None else center,
        size,
        name,
        color,
        padding=padding,
        stroke_width=stroke_width,
        tuning=state.icon_tuning() if state is not None else ICON_TUNING_DEFAULTS,
        alignment=state.icon_alignment_for_glyph(name) if state is not None else None,
    )


def _preview_search_input(state: ProbeState, *args, **kwargs):
    if state.preview_icon_library:
        kwargs["icon_drawer"] = lambda *values: _draw_concept_control_icon(
            *values,
            state=state,
        )
    return search_input(*args, **kwargs)


def _preview_searchable_header(state: ProbeState, *args, **kwargs):
    if state.preview_icon_library:
        kwargs["icon_drawer"] = lambda *values: _draw_concept_control_icon(
            *values,
            state=state,
        )
    return searchable_ordered_list_header(*args, **kwargs)


def _draw_icon_library_command_icon(
    draw,
    center,
    kind: str,
    color,
    scale: float,
    *,
    smoothing: float = CORNER_SMOOTHING,
    context_scale: float = 1.0,
    padding: float = ICON_DEFAULT_PADDING,
    state: ProbeState | None = None,
) -> None:
    """Render one Keyframes command through its Icon Library candidate."""

    del smoothing
    if kind == "key" and scale < context_scale * 0.9:
        name = "key-keyframe"
    else:
        try:
            name = keyframes_panel_module._COMMAND_ICON_NAMES[kind]
        except KeyError as exc:
            raise ValueError(f"unknown keyframe command icon: {kind!r}") from exc
    if state is not None:
        padding = state.icon_padding_for_glyph(name)
        stroke_width = state.icon_stroke_for_glyph(name)
    else:
        stroke_width = ICON_STROKE
    tuning = state.icon_tuning() if state is not None else ICON_TUNING_DEFAULTS
    icon_size = 16.0 * float(scale)
    draw_concept_icon(
        draw,
        state.icon_center(name, center, icon_size) if state is not None else center,
        icon_size,
        name,
        color,
        padding=padding,
        stroke_width=stroke_width,
        tuning=tuning,
        alignment=state.icon_alignment_for_glyph(name) if state is not None else None,
    )


def _draw_icon_library_label(draw, lo, hi, color, scale, name, label, *, state: ProbeState):
    """Use the row's live geometry for both caption measurement and submission."""
    style = IconStyle(
        padding=state.icon_padding_for_glyph(name),
        stroke_width=state.icon_stroke_for_glyph(name),
        alignment=state.icon_alignment_for_glyph(name),
        tuning=state.icon_tuning(),
    )
    draw_icon_label(
        draw,
        lo,
        hi,
        color,
        scale,
        name,
        label,
        style=style,
        icon_offset=state.icon_center(name, (0.0, 0.0), 16.0 * scale),
    )


def _draw_reset_icon(
    draw: ImguiDraw2D,
    center,
    color,
    scale: float,
    _surface=None,
    *,
    stroke_width: float = OVERLAY_GEOMETRY.tool_stroke,
    head_scale: float = ICON_TUNING_DEFAULTS.reset_head_scale,
) -> None:
    draw_reset_glyph(draw, center, color, scale, stroke_width, head_scale=head_scale)


def _circular_icon_button(
    draw: ImguiDraw2D,
    item_id: str,
    position,
    icon,
    *,
    selected: bool = False,
    cell_size: float = 34.0,
    state_radius: float = OVERLAY_STATE_RADIUS,
    icon_radius: float = OVERLAY_ICON_RADIUS,
    icon_scale: float = 1.0,
    show_icon_bound: bool = False,
    show_state_circle: bool = False,
    forced_interaction: str | None = None,
    scale: float = 1.0,
) -> bool:
    diameter = cell_size * scale
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    clicked = imgui.invisible_button(item_id, imgui.ImVec2(diameter, diameter))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    if forced_interaction is not None:
        hovered = forced_interaction == "hover"
        active = forced_interaction == "press"
        selected = selected or forced_interaction == "selected"
    theme = CONCEPT_THEME
    background, foreground = _viewport_control_colors(
        theme,
        selected=selected,
        hovered=hovered,
        active=active,
        enabled=True,
    )
    center = (position[0] + diameter * 0.5, position[1] + diameter * 0.5)
    if show_state_circle:
        guide_background = background if background[3] > 0.0 else theme.viewport.hover_background
        draw_selection_disc(draw, center, state_radius * scale, guide_background)
        draw.circle(
            center,
            state_radius * scale,
            (*CONCEPT_THEME.primary_dim[:3], 0.95),
            1.0 * scale,
        )
    elif background[3] > 0.0:
        draw_selection_disc(draw, center, state_radius * scale, background)
    icon_surface = background if background[3] > 0.0 else theme.viewport.surface
    icon(draw, center, foreground, icon_scale, icon_surface)
    if show_icon_bound:
        draw.circle(
            center,
            icon_radius * scale,
            (*CONCEPT_THEME.warning[:3], 0.95),
            1.0 * scale,
        )
    return clicked


def _draw_playback(draw: ImguiDraw2D, origin, scale: float, state: ProbeState) -> None:
    state.redesign.playing = state.playing
    _capsule(state.redesign, origin, scale, state, _circular_icon_button, False)
    _recording_menu(state.redesign)
    state.playing = state.redesign.playing


def _draw_tool_icon(
    draw: ImguiDraw2D,
    center,
    color,
    scale: float,
    kind: str,
    stroke_width: float,
    rotate_ring_gap_ratio: float,
    rotate_ring_cap: str,
    _surface_color,
    frame_space: str,
) -> None:
    draw_tool_glyph(
        draw,
        center,
        color,
        scale,
        kind,
        frame_space,
        replace(
            OVERLAY_GEOMETRY,
            tool_stroke=stroke_width,
            rotate_ring_gap_ratio=rotate_ring_gap_ratio,
            rotate_ring_cap=rotate_ring_cap,
        ),
        smoothing=draw.corner_smoothing,
    )


def _draw_tool_column(draw: ImguiDraw2D, origin, scale: float, state: ProbeState) -> None:
    state.redesign.tool = state.active_tool
    state.redesign.space = state.gizmo_space
    _capsule(state.redesign, origin, scale, state, _circular_icon_button, True)
    state.active_tool = state.redesign.tool
    state.gizmo_space = state.redesign.space


def _draw_inline_text(
    draw: ImguiDraw2D,
    x: float,
    center_y: float,
    value: str,
    color,
) -> float:
    width, height = draw.text_size(value)
    draw.text((x, center_y - height * 0.5), color, value)
    return width


def _draw_mouse_input(
    draw: ImguiDraw2D,
    x: float,
    center_y: float,
    scale: float,
    *,
    width: float,
    height: float,
    button: str,
    suffix: str,
    state: ProbeState,
    muted: bool = False,
) -> float:
    return draw_mouse_hint_glyph(
        draw,
        x,
        center_y,
        button,
        suffix,
        CONCEPT_THEME,
        scale,
        size=(width, height),
        smoothing=state.mouse_smoothing,
        muted=muted,
        geometry=replace(
            OVERLAY_GEOMETRY,
            hint_mouse_width=width,
            hint_control_height=height,
            hint_mouse_stroke=state.hint_mouse_stroke,
            hint_mouse_button_width_ratio=state.hint_mouse_button_width_ratio,
            hint_mouse_button_shell_ratio=state.hint_mouse_button_shell_ratio,
            hint_mouse_button_height_ratio=state.hint_mouse_button_height_ratio,
            hint_mouse_wheel_width_ratio=state.hint_mouse_wheel_width_ratio,
            hint_mouse_wheel_height_ratio=state.hint_mouse_wheel_height_ratio,
            hint_mouse_wheel_gap_ratio=state.hint_mouse_wheel_gap_ratio,
        ),
    )


def _keycap(
    draw: ImguiDraw2D,
    x: float,
    center_y: float,
    label: str,
    scale: float,
    *,
    height: float,
    padding_x: float,
) -> float:
    text_width, text_height = draw.text_size(label)
    width = text_width + padding_x * 2.0 * scale
    height *= scale
    y = center_y - height * 0.5
    draw.rect_filled(
        (x, y),
        (x + width, y + height),
        CONCEPT_THEME.bg_frame,
        rounding=keycap_rounding(width, height),
    )
    draw.rect(
        (x, y),
        (x + width, y + height),
        CONCEPT_THEME.border,
        1.0 * scale,
        rounding=keycap_rounding(width, height),
    )
    draw.text(
        (x + (width - text_width) * 0.5, y + (height - text_height) * 0.5),
        CONCEPT_THEME.text,
        label,
    )
    return width


def _mouse_input_width(draw: ImguiDraw2D, scale: float, state: ProbeState, suffix: str) -> float:
    width = state.hint_mouse_width * scale
    if suffix:
        width += 5.0 * scale + draw.text_size(suffix)[0]
    return width


def _hint_group_widths(
    draw: ImguiDraw2D, scale: float, state: ProbeState, variant: str
) -> tuple[float, ...]:
    def key_width(label: str) -> float:
        return draw.text_size(label)[0] + state.hint_key_padding_x * 2.0 * scale

    input_gap = state.hint_input_gap * scale
    chord_gap = state.hint_chord_gap * scale
    mouse_width = _mouse_input_width(draw, scale, state, "")
    perturb_width = (
        key_width("Ctrl")
        + input_gap
        + draw.text_size("+")[0]
        + chord_gap
        + draw.text_size("Drag")[0]
        + chord_gap
        + mouse_width
        + input_gap
        + draw.text_size("Push")[0]
        + chord_gap
        + mouse_width
        + input_gap
        + draw.text_size("Twist")[0]
    )
    if variant == "camera":
        return (
            mouse_width + input_gap + draw.text_size("Orbit")[0],
            mouse_width + input_gap + draw.text_size("Pan")[0],
            mouse_width + input_gap + draw.text_size("Zoom")[0],
            key_width("F") + input_gap + draw.text_size("Frame")[0],
        )
    if variant == "dragging":
        return (key_width("Shift") + input_gap + draw.text_size("Snap")[0],)
    if variant == "perturb":
        return (perturb_width,)
    return (
        key_width("Shift") + input_gap + draw.text_size("Snap")[0],
        key_width("T") + input_gap + draw.text_size("World / Body")[0],
        _mouse_input_width(draw, scale, state, "×2") + input_gap + draw.text_size("Type value")[0],
        perturb_width,
    )


def _hint_bar_width(
    draw: ImguiDraw2D, scale: float, state: ProbeState, variant: str = "ready"
) -> float:
    groups = _hint_group_widths(draw, scale, state, variant)
    group_gap = state.hint_group_gap * scale
    return sum(groups) + group_gap * (len(groups) - 1) + state.hint_padding_x * 2.0 * scale


def _draw_hint_bar(
    draw: ImguiDraw2D,
    origin,
    scale: float,
    state: ProbeState,
    variant: str = "ready",
) -> None:
    x, y = origin
    width = _hint_bar_width(draw, scale, state, variant)
    height = (state.hint_control_height + state.hint_padding_y * 2.0) * scale
    draw_capsule_shell(draw, x, y, width, height, scale, state)
    center_y = y + height * 0.5
    cursor = x + state.hint_padding_x * scale
    input_gap = state.hint_input_gap * scale
    group_gap = state.hint_group_gap * scale

    def draw_key_group(key: str, label: str) -> None:
        nonlocal cursor
        cursor += _keycap(
            draw,
            cursor,
            center_y,
            key,
            scale,
            height=float(state.hint_control_height),
            padding_x=float(state.hint_key_padding_x),
        )
        cursor += input_gap
        cursor += _draw_inline_text(draw, cursor, center_y, label, CONCEPT_THEME.text)

    def draw_mouse_group(
        button: str,
        suffix: str,
        label: str,
        *,
        after_gap: float = 0.0,
    ) -> None:
        nonlocal cursor
        cursor += _draw_mouse_input(
            draw,
            cursor,
            center_y,
            scale,
            width=float(state.hint_mouse_width),
            height=float(state.hint_control_height),
            button=button,
            suffix=suffix,
            state=state,
        )
        cursor += input_gap
        cursor += _draw_inline_text(draw, cursor, center_y, label, CONCEPT_THEME.text)
        cursor += after_gap

    def draw_group_separator() -> None:
        nonlocal cursor
        divider_x = cursor + group_gap * 0.5
        draw.line(
            (divider_x, center_y - 5.0 * scale),
            (divider_x, center_y + 5.0 * scale),
            (*CONCEPT_THEME.border[:3], min(0.62, CONCEPT_THEME.border[3])),
            1.0 * scale,
        )
        cursor += group_gap

    def draw_perturb_chord() -> None:
        nonlocal cursor
        cursor += _keycap(
            draw,
            cursor,
            center_y,
            "Ctrl",
            scale,
            height=float(state.hint_control_height),
            padding_x=float(state.hint_key_padding_x),
        )
        cursor += input_gap
        cursor += _draw_inline_text(draw, cursor, center_y, "+", CONCEPT_THEME.text_disabled)
        cursor += state.hint_chord_gap * scale
        cursor += _draw_inline_text(draw, cursor, center_y, "Drag", CONCEPT_THEME.primary_bright)
        cursor += state.hint_chord_gap * scale
        draw_mouse_group("left", "", "Push", after_gap=state.hint_chord_gap * scale)
        draw_mouse_group("right", "", "Twist")

    if variant == "camera":
        draw_mouse_group("left", "", "Orbit")
        draw_group_separator()
        draw_mouse_group("right", "", "Pan")
        draw_group_separator()
        draw_mouse_group("wheel", "", "Zoom")
        draw_group_separator()
        draw_key_group("F", "Frame")
    elif variant == "dragging":
        draw_key_group("Shift", "Snap")
    elif variant == "perturb":
        draw_perturb_chord()
    else:
        draw_key_group("Shift", "Snap")
        draw_group_separator()
        draw_key_group("T", "World / Body")
        draw_group_separator()
        draw_mouse_group("left", "×2", "Type value")
        draw_group_separator()
        draw_perturb_chord()


def _draw_label_button(
    draw: ImguiDraw2D,
    item_id: str,
    position,
    label: str,
    value: str,
    unit: str,
    color,
    scale: float,
    *,
    forced_state: str = "",
    interactive: bool = True,
) -> None:
    value_text = f"{label} {value}"
    value_width, text_height = draw.text_size(value_text)
    unit_width, _ = draw.text_size(unit)
    width = 24.0 * scale + value_width + unit_width + 15.0 * scale
    height = 30.0 * scale
    if interactive:
        imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
        imgui.invisible_button(item_id, imgui.ImVec2(width, height))
    hovered = (interactive and imgui.is_item_hovered()) or forced_state == "hover"
    active = (interactive and imgui.is_item_active()) or forced_state == "pressed"
    background = (
        CONCEPT_THEME.bg_frame_active
        if active
        else CONCEPT_THEME.bg_frame_hovered
        if hovered
        else CONCEPT_THEME.viewport.surface
    )
    foreground = CONCEPT_THEME.primary_bright if hovered else CONCEPT_THEME.text
    x, y = position
    draw.rect_filled((x, y), (x + width, y + height), background, rounding=3.0 * scale)
    draw.rect((x, y), (x + width, y + height), CONCEPT_THEME.border, 1.0, rounding=3.0 * scale)
    draw.circle_filled((x + 10.0 * scale, y + height * 0.5), 3.5 * scale, color)
    text_y = y + (height - text_height) * 0.5
    draw.text((x + 18.0 * scale, text_y), foreground, value_text)
    draw.text((x + width - unit_width - 7.0 * scale, text_y), CONCEPT_THEME.text_disabled, unit)


def _draw_value_input(position, scale: float, state: ProbeState) -> None:
    if not state.value_open:
        return
    imgui.set_next_window_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    imgui.set_next_window_size(imgui.ImVec2(300.0 * scale, 154.0 * scale))
    flags = _flags(
        imgui.WindowFlags_.no_resize,
        imgui.WindowFlags_.no_move,
        imgui.WindowFlags_.no_collapse,
        imgui.WindowFlags_.no_saved_settings,
    )
    opened, still_open = imgui.begin("Rotate Z###ProbeValueInput", state.value_open, flags)
    if still_open is not None:
        state.value_open = still_open
    if opened:
        table_flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
        if imgui.begin_table("##probe-value-table", 2, table_flags):
            imgui.table_setup_column(
                "label", imgui.TableColumnFlags_.width_fixed.value, 66.0 * scale
            )
            imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value)
            _property_label("Mode")
            _, state.value_mode = imgui.combo(
                "##probe-value-mode", state.value_mode, ("Relative", "Absolute")
            )
            _property_label("Value")
            available = imgui.get_content_region_avail().x
            imgui.set_next_item_width(max(90.0 * scale, available - 92.0 * scale))
            _, state.value = imgui.input_double("##probe-value", state.value, 0.0, 0.0, "+%.3f")
            imgui.same_line()
            state.unit = _draw_segmented("unit", ("°", "rad"), state.unit, width=42.0 * scale)
            imgui.end_table()
    imgui.end()


def _draw_value_input_card(position, size, scale: float, state: ProbeState) -> None:
    """Draw the M10 numeric popover as an embedded interactive specimen."""

    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    if not imgui.begin_child(
        "Rotate Z###ProbeValueInputCard",
        imgui.ImVec2(float(size[0]), float(size[1])),
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return
    imgui.text("Rotate Z")
    imgui.separator()
    table_flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
    if imgui.begin_table("##probe-value-card-table", 2, table_flags):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed.value, 44.0 * scale)
        imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value)
        _property_label("Mode")
        _, state.value_mode = imgui.combo(
            "##probe-value-card-mode", state.value_mode, ("Relative", "Absolute")
        )
        _property_label("Value")
        available = imgui.get_content_region_avail().x
        imgui.set_next_item_width(max(58.0 * scale, available - 76.0 * scale))
        _, state.value = imgui.input_double(
            "##probe-value-card-value", state.value, 0.0, 0.0, "+%.3f"
        )
        imgui.same_line()
        state.unit = _draw_segmented("card-unit", ("°", "rad"), state.unit, width=34.0 * scale)
        imgui.end_table()
    imgui.end_child()


def _draw_joint_value_input(position, scale: float, state: ProbeState) -> None:
    if not state.joint_value_open:
        return
    imgui.set_next_window_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    imgui.set_next_window_size(imgui.ImVec2(220.0 * scale, 96.0 * scale))
    flags = _flags(
        imgui.WindowFlags_.no_resize,
        imgui.WindowFlags_.no_move,
        imgui.WindowFlags_.no_collapse,
        imgui.WindowFlags_.no_saved_settings,
    )
    opened, still_open = imgui.begin(
        f"{state.joint_value_title}###ProbeJointValueInput",
        state.joint_value_open,
        flags,
    )
    if still_open is not None:
        state.joint_value_open = still_open
    if opened:
        table_flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
        if imgui.begin_table("##probe-joint-value-table", 2, table_flags):
            imgui.table_setup_column(
                "label", imgui.TableColumnFlags_.width_fixed.value, 42.0 * scale
            )
            imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value)
            _property_label("Value")
            unit_width = imgui.calc_text_size(state.joint_value_unit).x + 18.0 * scale
            imgui.set_next_item_width(
                max(78.0 * scale, imgui.get_content_region_avail().x - unit_width)
            )
            _, state.joint_value = imgui.input_double(
                "##probe-joint-value", state.joint_value, 0.0, 0.0, "%+.3f"
            )
            imgui.same_line()
            imgui.text_disabled(state.joint_value_unit)
            imgui.end_table()
        if imgui.is_key_pressed(imgui.Key.enter, False) or imgui.is_key_pressed(
            imgui.Key.keypad_enter, False
        ):
            state.joint_value_open = False
    imgui.end()
