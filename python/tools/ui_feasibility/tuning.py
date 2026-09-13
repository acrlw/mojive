"""Parameter controls and reproducible exports for UI and icon review."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui import theme as theme_mod
from mojive.ui.icons import (
    ICON_ALIGNMENT_EDITABLE_ICONS,
    ICON_GLYPH_PADDING_DEFAULTS,
    ICON_GLYPH_STROKE_DEFAULTS,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_GROUP_STROKE_DEFAULTS,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_TUNING_DEFAULTS,
)
from mojive.ui.viewport_widgets import OVERLAY_GEOMETRY, overlay_divider_length

from ..ui_capsule_geometry import CAPSULE_OUTLINE_LABELS, end_padding, spacing_metrics
from .fixtures import CORNER_CONTROLS
from .layout import _deferred_icon_group_slider, _even_slider, _flags, _property_label
from .state import ProbeState


def _geometry_values_text(state: ProbeState) -> str:
    """Return the live component experiment as reviewable production fields."""

    values = (
        ("imgui_rounding", state.imgui_rounding),
        ("capsule_outline", repr(state.capsule_outline)),
        ("icon_radius", state.overlay_icon_radius),
        ("radial_step", state.overlay_radial_step),
        ("center_step", state.overlay_center_step),
        ("tool_group_gap", state.tool_group_gap),
        ("divider_width", state.divider_width),
        ("tool_stroke", state.tool_stroke_width),
        ("rotate_ring_gap_ratio", state.rotate_ring_gap_ratio),
        ("rotate_ring_cap", repr(state.rotate_ring_cap)),
        ("icon_tool_move_head_scale", state.move_head_scale),
        ("icon_tool_scale_handle_scale", state.scale_handle_scale),
        ("icon_tool_snap_endpoint_scale", state.snap_endpoint_scale),
        ("icon_key_fit_arm_length", state.key_fit_arm_length),
        ("hint_control_height", state.hint_control_height),
        ("hint_padding_x", state.hint_padding_x),
        ("hint_padding_y", state.hint_padding_y),
        ("hint_input_gap", state.hint_input_gap),
        ("hint_group_gap", state.hint_group_gap),
        ("hint_chord_gap", state.hint_chord_gap),
        ("hint_key_padding_x", state.hint_key_padding_x),
        ("hint_mouse_width", state.hint_mouse_width),
        ("hint_mouse_stroke", state.hint_mouse_stroke),
        ("hint_mouse_button_width_ratio", state.hint_mouse_button_width_ratio),
        ("hint_mouse_button_shell_ratio", state.hint_mouse_button_shell_ratio),
        ("hint_mouse_button_height_ratio", state.hint_mouse_button_height_ratio),
        ("hint_mouse_wheel_width_ratio", state.hint_mouse_wheel_width_ratio),
        ("hint_mouse_wheel_height_ratio", state.hint_mouse_wheel_height_ratio),
        ("hint_mouse_wheel_gap_ratio", state.hint_mouse_wheel_gap_ratio),
    )
    values += tuple(
        (
            "icon_padding_" + group.casefold().replace(" ", "_").replace("&", "and"),
            state.icon_padding_for(group),
        )
        for group in ICON_GROUP_LAYOUT_DEFAULTS
    )
    values += tuple(
        (
            "icon_stroke_" + group.casefold().replace(" ", "_").replace("&", "and"),
            state.icon_stroke_for(group),
        )
        for group in ICON_GROUP_STROKE_DEFAULTS
    )
    values += tuple(
        (f"icon_padding_{name.replace('-', '_')}", value)
        for name, value in sorted(state.icon_padding_by_glyph.items())
    )
    values += tuple(
        (f"icon_stroke_{name.replace('-', '_')}", value)
        for name, value in sorted(state.icon_stroke_by_glyph.items())
    )
    values += tuple((name, getattr(state, name)) for _, name in CORNER_CONTROLS)
    return "\n".join(f"{name}={value}," for name, value in values)


def _icon_values_text(state: ProbeState) -> str:
    """Return the complete editable Icon Library state in a pasteable form."""

    values = [
        ("rotate_ring_gap_ratio", state.rotate_ring_gap_ratio),
        ("rotate_ring_cap", repr(state.rotate_ring_cap)),
        ("icon_tool_move_head_scale", state.move_head_scale),
        ("icon_tool_scale_handle_scale", state.scale_handle_scale),
        ("icon_tool_snap_endpoint_scale", state.snap_endpoint_scale),
        ("icon_key_fit_arm_length", state.key_fit_arm_length),
        ("icon_status_mouse_width", state.hint_mouse_width),
    ]
    values.extend(
        (
            "icon_padding_" + group.casefold().replace(" ", "_").replace("&", "and"),
            state.icon_padding_for(group),
        )
        for group in ICON_GROUP_LAYOUT_DEFAULTS
    )
    values.extend(
        (
            "icon_stroke_" + group.casefold().replace(" ", "_").replace("&", "and"),
            state.icon_stroke_for(group),
        )
        for group in ICON_GROUP_STROKE_DEFAULTS
    )
    values.extend(
        (
            f"icon_padding_{name.replace('-', '_')}",
            state.icon_padding_for_glyph(name),
        )
        for name in sorted(ICON_GLYPH_PADDING_DEFAULTS.keys() | state.icon_padding_by_glyph.keys())
    )
    values.extend(
        (
            f"icon_stroke_{name.replace('-', '_')}",
            state.icon_stroke_for_glyph(name),
        )
        for name in sorted(ICON_GLYPH_STROKE_DEFAULTS.keys() | state.icon_stroke_by_glyph.keys())
    )
    values.extend(
        (
            f"icon_alignment_{name.replace('-', '_')}",
            repr(state.icon_alignment_for_glyph(name)),
        )
        for name in sorted(ICON_ALIGNMENT_EDITABLE_ICONS)
    )
    return "\n".join(f"{name}={value}," for name, value in values)


def _draw_corner_controls(position, size, state: ProbeState) -> None:
    imgui.set_cursor_screen_pos(imgui.ImVec2(*position))
    if imgui.begin_child(
        "Corner smoothing###CornerControls", imgui.ImVec2(*size), imgui.ChildFlags_.borders.value
    ):
        imgui.text("ImGui corner radius")
        imgui.set_next_item_width(-1.0)
        _, state.imgui_rounding = imgui.slider_float(
            "##imgui-corner-radius",
            state.imgui_rounding,
            0.0,
            16.0,
            "%.2f px",
            imgui.SliderFlags_.always_clamp.value,
        )
        imgui.text_wrapped("Standard ImGui corners. Radius does not change layout spacing.")
        imgui.separator()
        imgui.text("Custom drawing smoothing")
        imgui.separator()
        for label, name in CORNER_CONTROLS:
            imgui.text(label)
            imgui.set_next_item_width(-1.0)
            changed, value = imgui.slider_float(
                f"##corner-{name}",
                getattr(state, name),
                0.0,
                1.0,
                "%.3f",
                imgui.SliderFlags_.always_clamp.value,
            )
            if changed:
                setattr(state, name, float(value))
        imgui.separator()
        if imgui.button("Reset corners", imgui.ImVec2(-1.0, 0.0)):
            state.imgui_rounding = theme_mod.DEFAULT_CORNER_RADIUS
            for _, name in CORNER_CONTROLS:
                setattr(state, name, ProbeState.__dataclass_fields__[name].default)
        if imgui.button("Copy corner values", imgui.ImVec2(-1.0, 0.0)):
            imgui.set_clipboard_text(
                f"imgui_rounding={state.imgui_rounding:.6g}\n"
                + "\n".join(f"{name}={getattr(state, name):.6g}" for _, name in CORNER_CONTROLS)
            )
        imgui.text_wrapped(
            "0 uses the baseline profile. Positive values add smooth curvature transitions."
        )
    imgui.end_child()


def _current_icon_adjustment_group(state: ProbeState) -> str:
    if state.icon_adjustment_group not in ICON_GROUP_LAYOUT_DEFAULTS:
        state.icon_adjustment_group = "Viewport tools"
    return state.icon_adjustment_group


def _draw_icon_group_combo(item_id: str, state: ProbeState) -> str:
    groups = tuple(ICON_GROUP_LAYOUT_DEFAULTS)
    current = _current_icon_adjustment_group(state)
    changed, selected = imgui.combo(item_id, groups.index(current), groups)
    if changed:
        state.icon_adjustment_group = groups[selected]
    return state.icon_adjustment_group


def _draw_geometry_controls(position, size, state: ProbeState) -> None:
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    if not imgui.begin_child(
        "Geometry controls###ProbeGeometryControls",
        imgui.ImVec2(float(size[0]), float(size[1])),
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return

    imgui.text("Live component experiment")
    imgui.separator()
    imgui.text_wrapped(
        "Probe-only values. Review visually, then copy accepted fields into production."
    )
    imgui.text("Capsule outline")
    imgui.set_next_item_width(-1.0)
    changed, outline = imgui.combo(
        "##capsule-outline",
        CAPSULE_OUTLINE_LABELS.index(state.capsule_outline),
        CAPSULE_OUTLINE_LABELS,
    )
    if changed:
        state.capsule_outline = CAPSULE_OUTLINE_LABELS[outline]
    _, state.highlight_g3 = imgui.checkbox("Highlight G3 transitions", state.highlight_g3)
    imgui.text_wrapped(
        "Orange: curvature ramps. Neutral outline: circular arcs and straight edges."
    )
    _, state.optical_capsule_spacing = imgui.checkbox(
        "Optical end spacing", state.optical_capsule_spacing
    )
    radius = state.overlay_icon_radius + 2 * state.overlay_radial_step
    minimum, mean, maximum, side = spacing_metrics(
        radius,
        state.overlay_icon_radius + state.overlay_radial_step,
        state.capsule_smoothing,
        state.optical_capsule_spacing,
    )
    imgui.text_wrapped(f"End center {end_padding(state):.2f} · side gap {side:.2f}")
    imgui.text_wrapped(f"End gap min / mean / max: {minimum:.2f} / {mean:.2f} / {maximum:.2f}")
    imgui.spacing()
    flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
    if imgui.begin_table("##geometry-controls", 2, flags):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.46)
        imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value, 0.54)

        _property_label("Icon radius")
        state.overlay_icon_radius = _even_slider(
            "##geometry-icon-radius", state.overlay_icon_radius, 6, 16
        )
        _property_label("Radial step")
        state.overlay_radial_step = _even_slider(
            "##geometry-radial-step", state.overlay_radial_step, 2, 12
        )
        _property_label("Glyph group")
        imgui.set_next_item_width(-1.0)
        group = _draw_icon_group_combo("##geometry-glyph-group", state)
        _property_label("Glyph padding")
        _deferred_icon_group_slider(
            "##geometry-glyph-padding",
            state,
            group,
            kind="padding",
        )
        imgui.set_item_tooltip(
            "Circular clearance from the candidate to its icon slot. Rotate and the reviewed "
            "Info/Warning/Error family remain fixed. Release the slider to apply the group."
        )
        _property_label("Glyph stroke")
        _deferred_icon_group_slider(
            "##geometry-glyph-stroke",
            state,
            group,
            kind="stroke",
        )
        imgui.set_item_tooltip(
            "Default visual weight for the selected component group. Individual glyph rows can "
            "override it; reviewed Pause, First/Last, severity, and mouse geometry remain locked. "
            "Release the slider to apply the group."
        )
        _property_label("Center step")
        state.overlay_center_step = _even_slider(
            "##geometry-center-step", state.overlay_center_step, 24, 52
        )
        _property_label("Group gap")
        state.tool_group_gap = _even_slider("##geometry-group-gap", state.tool_group_gap, 4, 24)
        _property_label("Divider (Tools)")
        state.divider_width = _even_slider("##geometry-divider-width", state.divider_width, 10, 34)
        imgui.set_item_tooltip(
            f"Playback: {overlay_divider_length(state.divider_width, playback=True):.1f} pt; "
            f"Tools: {state.divider_width:g} pt. Both use the same divider-to-glyph proportion."
        )
        _property_label("Playback zoom")
        _, state.construction_playback_scale = imgui.slider_float(
            "##geometry-playback-zoom",
            state.construction_playback_scale,
            1.5,
            4.0,
            "%.1fx",
        )
        _property_label("Tool zoom")
        _, state.construction_tool_scale = imgui.slider_float(
            "##geometry-tool-zoom",
            state.construction_tool_scale,
            1.5,
            3.0,
            "%.1fx",
        )
        _property_label("Tool stroke")
        _, state.tool_stroke_width = imgui.slider_float(
            "##geometry-tool-stroke",
            state.tool_stroke_width,
            1.0,
            2.2,
            "%.2f px",
        )
        _property_label("Gap / stroke")
        _, state.rotate_ring_gap_ratio = imgui.slider_float(
            "##geometry-ring-gap-ratio",
            state.rotate_ring_gap_ratio,
            0.25,
            1.0,
            "%.2fx",
        )
        _property_label("Ring caps")
        cap_index = 1 if state.rotate_ring_cap == "round" else 0
        _, cap_index = imgui.combo(
            "##geometry-ring-cap",
            cap_index,
            ("Butt", "Round"),
        )
        state.rotate_ring_cap = ("butt", "round")[cap_index]
        imgui.end_table()

    icon_radius = state.overlay_icon_radius
    state_radius = icon_radius + state.overlay_radial_step
    shell_radius = state_radius + state.overlay_radial_step
    state_clearance = state.overlay_center_step - state_radius * 2
    imgui.text_disabled(
        f"r {icon_radius} / {state_radius} / {shell_radius}  ·  state gap {state_clearance:+d}"
    )

    imgui.spacing()
    imgui.text("Context hint")
    imgui.separator()
    if imgui.begin_table("##hint-controls", 2, flags):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.46)
        imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value, 0.54)
        _property_label("Control height")
        state.hint_control_height = _even_slider(
            "##hint-control-height", state.hint_control_height, 18, 30
        )
        _property_label("Padding X")
        state.hint_padding_x = _even_slider("##hint-padding-x", state.hint_padding_x, 8, 28)
        _property_label("Padding Y")
        state.hint_padding_y = _even_slider("##hint-padding-y", state.hint_padding_y, 4, 16)
        _property_label("Input gap")
        state.hint_input_gap = _even_slider("##hint-input-gap", state.hint_input_gap, 4, 16)
        _property_label("Group gap")
        state.hint_group_gap = _even_slider("##hint-group-gap", state.hint_group_gap, 8, 36)
        _property_label("Chord gap")
        state.hint_chord_gap = _even_slider("##hint-chord-gap", state.hint_chord_gap, 4, 20)
        _property_label("Key padding")
        state.hint_key_padding_x = _even_slider(
            "##hint-key-padding", state.hint_key_padding_x, 4, 12
        )
        _property_label("Mouse width")
        state.hint_mouse_width = _even_slider("##hint-mouse-width", state.hint_mouse_width, 12, 24)
        _property_label("Mouse stroke")
        _, state.hint_mouse_stroke = imgui.slider_float(
            "##hint-mouse-stroke", state.hint_mouse_stroke, 0.75, 2.5, "%.2f px"
        )
        _property_label("Button width")
        _, state.hint_mouse_button_width_ratio = imgui.slider_float(
            "##hint-button-width", state.hint_mouse_button_width_ratio, 0.30, 0.55, "%.2fx"
        )
        _property_label("Button shell")
        _, state.hint_mouse_button_shell_ratio = imgui.slider_float(
            "##hint-button-shell", state.hint_mouse_button_shell_ratio, 0.25, 1.25, "%.2fx"
        )
        _property_label("Button height")
        _, state.hint_mouse_button_height_ratio = imgui.slider_float(
            "##hint-button-height", state.hint_mouse_button_height_ratio, 0.30, 0.55, "%.2fx"
        )
        _property_label("Wheel width")
        _, state.hint_mouse_wheel_width_ratio = imgui.slider_float(
            "##hint-wheel-width", state.hint_mouse_wheel_width_ratio, 0.18, 0.38, "%.2fx"
        )
        _property_label("Wheel height")
        _, state.hint_mouse_wheel_height_ratio = imgui.slider_float(
            "##hint-wheel-height", state.hint_mouse_wheel_height_ratio, 0.25, 0.50, "%.2fx"
        )
        _property_label("Wheel top gap")
        _, state.hint_mouse_wheel_gap_ratio = imgui.slider_float(
            "##hint-wheel-gap", state.hint_mouse_wheel_gap_ratio, 0.10, 1.00, "%.2fx"
        )
        imgui.end_table()
    hint_height = state.hint_control_height + state.hint_padding_y * 2
    imgui.text_disabled(f"Single row  ·  shell height {hint_height}")

    imgui.spacing()
    if imgui.button("Copy current values", imgui.ImVec2(-1.0, 0.0)):
        imgui.set_clipboard_text(_geometry_values_text(state))
    if imgui.button("Reset production defaults", imgui.ImVec2(-1.0, 0.0)):
        state.imgui_rounding = theme_mod.DEFAULT_CORNER_RADIUS
        state.capsule_outline = ProbeState.__dataclass_fields__["capsule_outline"].default
        for _, name in CORNER_CONTROLS:
            setattr(state, name, ProbeState.__dataclass_fields__[name].default)
        state.overlay_icon_radius = int(OVERLAY_GEOMETRY.icon_radius)
        state.overlay_radial_step = int(OVERLAY_GEOMETRY.radial_step)
        state.icon_adjustment_group = "Viewport tools"
        state.icon_padding_by_group = dict(ICON_GROUP_LAYOUT_DEFAULTS)
        state.icon_padding_by_glyph.clear()
        state.icon_padding_draft_by_group.clear()
        state.icon_alignment_by_glyph.clear()
        state.icon_stroke_by_group = dict(ICON_GROUP_STROKE_DEFAULTS)
        state.icon_stroke_by_glyph.clear()
        state.icon_stroke_draft_by_group.clear()
        state.overlay_center_step = int(OVERLAY_GEOMETRY.center_step)
        state.tool_group_gap = int(OVERLAY_GEOMETRY.tool_group_gap)
        state.divider_width = int(OVERLAY_GEOMETRY.divider_width)
        state.construction_playback_scale = 3.0
        state.construction_tool_scale = 1.5
        state.tool_stroke_width = OVERLAY_GEOMETRY.tool_stroke
        state.rotate_ring_gap_ratio = ICON_ROTATE_RING_GAP_RATIO
        state.rotate_ring_cap = ICON_ROTATE_RING_CAP
        state.move_head_scale = ICON_TUNING_DEFAULTS.move_head_scale
        state.scale_handle_scale = ICON_TUNING_DEFAULTS.scale_handle_scale
        state.snap_endpoint_scale = ICON_TUNING_DEFAULTS.snap_endpoint_scale
        state.key_fit_arm_length = ICON_TUNING_DEFAULTS.key_fit_arm_length
        state.hint_control_height = int(OVERLAY_GEOMETRY.hint_control_height)
        state.hint_padding_x = int(OVERLAY_GEOMETRY.hint_padding_x)
        state.hint_padding_y = int(OVERLAY_GEOMETRY.hint_padding_y)
        state.hint_input_gap = int(OVERLAY_GEOMETRY.hint_input_gap)
        state.hint_group_gap = int(OVERLAY_GEOMETRY.hint_group_gap)
        state.hint_chord_gap = int(OVERLAY_GEOMETRY.hint_chord_gap)
        state.hint_key_padding_x = int(OVERLAY_GEOMETRY.hint_key_padding_x)
        state.hint_mouse_width = int(OVERLAY_GEOMETRY.hint_mouse_width)
        state.hint_mouse_stroke = OVERLAY_GEOMETRY.hint_mouse_stroke
        state.hint_mouse_button_width_ratio = OVERLAY_GEOMETRY.hint_mouse_button_width_ratio
        state.hint_mouse_button_shell_ratio = OVERLAY_GEOMETRY.hint_mouse_button_shell_ratio
        state.hint_mouse_button_height_ratio = OVERLAY_GEOMETRY.hint_mouse_button_height_ratio
        state.hint_mouse_wheel_width_ratio = OVERLAY_GEOMETRY.hint_mouse_wheel_width_ratio
        state.hint_mouse_wheel_height_ratio = OVERLAY_GEOMETRY.hint_mouse_wheel_height_ratio
        state.hint_mouse_wheel_gap_ratio = OVERLAY_GEOMETRY.hint_mouse_wheel_gap_ratio

    imgui.end_child()
