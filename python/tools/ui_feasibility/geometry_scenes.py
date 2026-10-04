"""Focused geometry specimens for the interactive design workbench."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive.tools.ui_capsule_geometry import end_padding
from mojive.tools.ui_redesign import RECORD_GLYPH_RADIUS, RESET_GLYPH_SCALE, draw_recording_glyph
from mojive.ui.icons import draw_concept_icon
from mojive.ui.viewport_widgets import (
    PLAYBACK_HALF_HEIGHT_PT,
    PLAYBACK_RESET_SCALE,
    RECORDING_OPTIONS_GLYPH_SCALE,
    RECORDING_OPTIONS_STROKE_SCALE,
    TOOL_GLYPH_SCALE,
    draw_recording_options_glyph,
    draw_status,
)

from .fixtures import (
    CONCEPT_THEME,
    OVERLAY_ICON_RADIUS,
)
from .gizmos import (
    _draw_camera_icon,
    _draw_helper_viewport,
    _draw_joint_gizmo,
    _draw_joint_rotation_feedback,
    _draw_light_icon,
    _draw_transform_gizmo,
)
from .layout import (
    _dimension_line,
)
from .panels import (
    _probe_status_hints,
)
from .widgets import (
    _circular_icon_button,
    _draw_hint_bar,
    _draw_playback,
    _draw_reset_icon,
    _draw_tool_column,
    _draw_tool_icon,
    _draw_value_input_card,
)


def draw_playback_study(draw, origin, scale, state, controls_x) -> None:
    x0, content_y = origin
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    icon_radius = float(state.overlay_icon_radius)
    state_radius = icon_radius + float(state.overlay_radial_step)
    shell_radius = state_radius + float(state.overlay_radial_step)
    center_step = float(state.overlay_center_step)
    inspection = state.construction_playback_scale
    playback_scale = scale * inspection
    left = x0 + 54.0 * scale
    construction_height = shell_radius * 2.0 * playback_scale
    draw.text(
        (left, content_y + 4.0 * scale),
        title_color,
        "Playback construction · Play and Pause",
    )
    draw.text(
        (left, content_y + 28.0 * scale),
        note_color,
        "Each section begins below the scaled bounds of the section above.",
    )

    play_y = content_y + 80.0 * scale
    play_origin = (left, play_y)
    draw.text((left, play_y - 24.0 * scale), note_color, f"Play geometry · {inspection:.1f}×")
    construction_state = replace(
        state,
        show_icon_bounds=True,
        show_state_circles=True,
        show_construction_notes=True,
    )
    imgui.push_id("geometry-playback-play")
    _draw_playback(
        draw,
        play_origin,
        playback_scale,
        replace(construction_state, redesign=replace(state.redesign), playing=False),
    )
    imgui.pop_id()

    pause_y = play_y + construction_height + 62.0 * scale
    pause_origin = (left, pause_y)
    draw.text((left, pause_y - 24.0 * scale), note_color, f"Pause geometry · {inspection:.1f}×")
    imgui.push_id("geometry-playback-pause")
    _draw_playback(
        draw,
        pause_origin,
        playback_scale,
        replace(construction_state, redesign=replace(state.redesign), playing=True),
    )
    imgui.pop_id()

    construction_bottom = pause_y + construction_height
    pb_first_x = play_origin[0] + end_padding(state) * playback_scale
    pb_last_x = pb_first_x + center_step * 3.0 * playback_scale
    _dimension_line(
        draw,
        (pb_first_x, construction_bottom + 22.0 * scale),
        (pb_last_x, construction_bottom + 22.0 * scale),
        f"3 × CENTER {state.overlay_center_step}",
        scale,
    )
    _dimension_line(
        draw,
        (play_origin[0] - 22.0 * scale, play_origin[1]),
        (
            play_origin[0] - 22.0 * scale,
            play_origin[1] + shell_radius * 2.0 * playback_scale,
        ),
        f"SHELL {int(shell_radius * 2.0)}",
        scale,
        vertical=True,
    )
    draw.text(
        (left, construction_bottom + 54.0 * scale),
        note_color,
        (
            f"Bounds icon {int(icon_radius * 2.0)} · state {int(state_radius * 2.0)} · "
            f"shell {int(shell_radius * 2.0)} · centers {state.overlay_center_step} · "
            f"radial step {state.overlay_radial_step} · playback glyph pad "
            f"{state.icon_padding_for('Viewport playback'):.2f}u"
        ),
    )

    product_state = replace(
        state,
        redesign=replace(state.redesign),
        show_icon_bounds=False,
        show_state_circles=False,
        show_construction_notes=False,
    )
    section_y = construction_bottom + 112.0 * scale
    draw.text((left, section_y), title_color, "Playback states · 2×")
    paused_y = section_y + 38.0 * scale
    imgui.push_id("product-playback-paused")
    _draw_playback(draw, (left, paused_y), scale * 2.0, replace(product_state, playing=False))
    imgui.pop_id()
    product_height = shell_radius * 4.0 * scale
    draw.text((left, paused_y + product_height + 12.0 * scale), note_color, "Paused · Play")

    playing_y = paused_y + product_height + 58.0 * scale
    imgui.push_id("product-playback-playing")
    _draw_playback(draw, (left, playing_y), scale * 2.0, replace(product_state, playing=True))
    imgui.pop_id()
    draw.text(
        (left, playing_y + product_height + 12.0 * scale),
        note_color,
        "Playing · Pause remains selected",
    )

    recording_heading_y = playing_y + product_height + 72.0 * scale
    draw.text(
        (left, recording_heading_y),
        title_color,
        "Reset / Record / Stop / Recording options · 2×",
    )
    glyph_ratio = icon_radius / OVERLAY_ICON_RADIUS
    stop_side = 2 * PLAYBACK_HALF_HEIGHT_PT * PLAYBACK_RESET_SCALE * glyph_ratio
    main_width = controls_x - left - 30.0 * scale
    specimen_step = main_width / 4.0
    specimen_y = recording_heading_y + 42.0 * scale
    for index, (label, measurement, icon) in enumerate(
        (
            (
                "Reset simulation",
                f"Envelope Ø{2 * icon_radius * RESET_GLYPH_SCALE:.2f}",
                lambda *args: _draw_reset_icon(
                    *args,
                    stroke_width=state.tool_stroke_width,
                    head_scale=state.reset_head_scale,
                ),
            ),
            (
                "Record Take / Video",
                f"Circle Ø{2 * RECORD_GLYPH_RADIUS * glyph_ratio:.2f}",
                lambda target, center, _color, icon_scale, _surface: draw_recording_glyph(
                    target, center, CONCEPT_THEME.danger, icon_scale
                ),
            ),
            (
                "Stop recording",
                f"Square side {stop_side:.2f}",
                lambda target, center, _color, icon_scale, _surface: draw_recording_glyph(
                    target, center, CONCEPT_THEME.danger, icon_scale, recording=True
                ),
            ),
            (
                "Recording options",
                (
                    "G3 · stroke "
                    f"{state.tool_stroke_width * RECORDING_OPTIONS_STROKE_SCALE * RECORDING_OPTIONS_GLYPH_SCALE * glyph_ratio:.2f}"
                ),
                lambda target, center, color, icon_scale, surface: draw_recording_options_glyph(
                    target, center, color, icon_scale, stroke=state.tool_stroke_width
                ),
            ),
        )
    ):
        position = (left + index * specimen_step, specimen_y)
        _circular_icon_button(
            draw.with_corner_smoothing(state.playback_smoothing),
            f"##geometry-recording-{index}",
            position,
            icon,
            cell_size=center_step,
            state_radius=state_radius,
            icon_radius=icon_radius,
            icon_scale=2.0 * scale * icon_radius / OVERLAY_ICON_RADIUS,
            show_icon_bound=True,
            show_state_circle=True,
            scale=2.0 * scale,
        )
        draw.text((position[0], position[1] + 96.0 * scale), note_color, label)
        draw.text((position[0], position[1] + 116.0 * scale), note_color, measurement)

    spacing_heading_y = specimen_y + 168.0 * scale
    draw.text((left, spacing_heading_y), title_color, "End-spacing comparison · 2×")
    first_spacing_y = spacing_heading_y + 38.0 * scale
    for index, (label, optical) in enumerate(
        (("Original end spacing", False), ("Optical end spacing", True))
    ):
        sample_y = first_spacing_y + index * (product_height + 62.0 * scale)
        draw.text((left, sample_y - 22.0 * scale), note_color, label)
        imgui.push_id(label)
        _draw_playback(
            draw,
            (left, sample_y),
            scale * 2.0,
            replace(
                product_state,
                redesign=replace(state.redesign),
                optical_capsule_spacing=optical,
                playing=True,
            ),
        )
        imgui.pop_id()


def draw_tool_study(draw, origin, size, scale, state) -> None:
    x0, content_y = origin
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    icon_radius = float(state.overlay_icon_radius)
    state_radius = icon_radius + float(state.overlay_radial_step)
    shell_radius = state_radius + float(state.overlay_radial_step)
    center_step = float(state.overlay_center_step)
    group_step = center_step + float(state.tool_group_gap)
    construction_state = replace(
        state,
        show_icon_bounds=True,
        show_state_circles=True,
        show_construction_notes=True,
    )
    tool_scale = scale * state.construction_tool_scale
    tool_origin = (x0 + 96.0 * scale, content_y + 56.0 * scale)
    draw.text(
        (x0 + 54.0 * scale, content_y + 4.0 * scale),
        title_color,
        f"Construction geometry · {state.construction_tool_scale:.1f}× inspection",
    )
    draw.text(
        (x0 + 54.0 * scale, content_y + 28.0 * scale),
        note_color,
        "Amber = icon bound · green = state circle · outer line = capsule",
    )
    imgui.push_id("geometry-tools")
    _draw_tool_column(draw, tool_origin, tool_scale, construction_state)
    imgui.pop_id()
    tool_notes_x = tool_origin[0] + 150.0 * scale
    for index, line in enumerate(
        (
            f"TOOL GLYPH Ø{icon_radius * TOOL_GLYPH_SCALE * 2.0:.1f}",
            f"STATE CIRCLE Ø{int(state_radius * 2.0)}",
            f"SHELL WIDTH  {int(shell_radius * 2.0)}",
            f"CENTER STEP  {state.overlay_center_step}",
            f"GROUP STEP   {int(group_step)}",
            f"RADIAL STEP  {state.overlay_radial_step:2d}",
            f"DIVIDER      {state.divider_width}",
            f"RING CAPS    {state.rotate_ring_cap.upper()}",
        )
    ):
        color = (
            CONCEPT_THEME.warning
            if index == 0
            else CONCEPT_THEME.primary_bright
            if index == 1
            else note_color
        )
        draw.text(
            (tool_notes_x, content_y + (76.0 + index * 25.0) * scale),
            color,
            line,
        )

    comparison_x = x0 + max(650.0 * scale, size.x * 0.43)
    product_tool_origin = (comparison_x, content_y + 56.0 * scale)
    product_tool_scale = scale * 1.8
    product_state = replace(
        state,
        show_icon_bounds=False,
        show_state_circles=False,
        show_construction_notes=False,
    )
    imgui.push_id("product-tools")
    _draw_tool_column(draw, product_tool_origin, product_tool_scale, product_state)
    imgui.pop_id()
    labels_x = product_tool_origin[0] + shell_radius * 2.0 * product_tool_scale + 18.0 * scale
    draw.text(
        (labels_x, content_y + 4.0 * scale),
        title_color,
        "Product scale · 1.8× inspection",
    )
    draw.text(
        (labels_x, content_y + 24.0 * scale),
        note_color,
        "Runtime and feasibility use the same vector draw path.",
    )
    product_centers = (
        shell_radius,
        shell_radius + center_step,
        shell_radius + center_step * 2.0,
        shell_radius + center_step * 2.0 + group_step,
        shell_radius + center_step * 3.0 + group_step,
    )
    for center, (label, meaning) in zip(
        product_centers,
        (
            ("Move", "Translate selected object"),
            ("Rotate", "3 half-rings + screen ring"),
            ("Scale", "Resize selected object"),
            ("World / Body", "Switch transform frame"),
            ("Snap", "Toggle snapping"),
        ),
        strict=True,
    ):
        label_y = product_tool_origin[1] + center * product_tool_scale
        draw.text((labels_x, label_y - 15.0 * scale), title_color, label)
        draw.text((labels_x, label_y + 3.0 * scale), note_color, meaning)

    frame_samples_x = labels_x + 214.0 * scale
    frame_samples_y = product_tool_origin[1] + product_centers[3] * product_tool_scale
    draw.text(
        (frame_samples_x - 12.0 * scale, frame_samples_y - 34.0 * scale),
        note_color,
        "Frame states",
    )
    for index, (space, label) in enumerate((("world", "World"), ("body", "Body"))):
        center_x = frame_samples_x + index * 52.0 * scale
        _draw_tool_icon(
            draw,
            (center_x, frame_samples_y),
            CONCEPT_THEME.text,
            product_tool_scale,
            "frame",
            state.tool_stroke_width,
            state.rotate_ring_gap_ratio,
            state.rotate_ring_cap,
            CONCEPT_THEME.bg_child,
            space,
        )
        label_width, _ = draw.text_size(label)
        draw.text(
            (center_x - label_width * 0.5, frame_samples_y + 15.0 * scale),
            note_color,
            label,
        )


def draw_hint_study(draw, origin, size, scale, state) -> None:
    x0, content_y = origin
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    hint_label_x = x0 + 54.0 * scale
    draw.text((hint_label_x, content_y + 4.0 * scale), title_color, "Context hint states")
    draw.text(
        (hint_label_x, content_y + 28.0 * scale),
        note_color,
        "Defaults are composed into Status; each whole group is dropped when space runs out.",
    )
    for index, (label, selected, variant, selection_clear) in enumerate(
        (
            ("No selection", False, "camera", False),
            ("Transform ready", True, "ready", True),
            ("Handle hovered · 0.5 s", True, "ready_minimal", True),
            ("Transform drag", True, "dragging", False),
            ("Ctrl held", True, "perturb", True),
        )
    ):
        row_y = content_y + 70.0 * scale + index * 52.0 * scale
        draw_status(
            draw,
            (hint_label_x, row_y),
            size.x * 0.75,
            28.0 * scale,
            CONCEPT_THEME,
            scale,
            selected=label if selected else "No selection",
            state="paused",
            sim_time=1.204,
            step=1204,
            metric_mode="time",
            backend=state.renderer,
            dt=0.002,
            fps=60.0,
            tool_hints=_probe_status_hints(
                variant,
                selected=selected,
                selection_clear=selection_clear,
            ),
        )

    draw.text(
        (hint_label_x, content_y + 344.0 * scale),
        note_color,
        "Reusable scene surface (custom hint providers can opt in):",
    )
    hint_x = hint_label_x
    _draw_hint_bar(draw, (hint_x, content_y + 370.0 * scale), scale, state, "camera")
    card_y = content_y + 434.0 * scale
    draw.text((hint_label_x, card_y), title_color, "Value input · M10")
    _draw_value_input_card(
        (hint_x, card_y),
        (220.0 * scale, 138.0 * scale),
        scale,
        state,
    )
    draw.text(
        (hint_x, card_y + 154.0 * scale),
        note_color,
        "Popup open: context hints are hidden; Enter commits, Esc or outside click cancels.",
    )


def draw_transform_study(draw, origin, scale, state) -> None:
    x0, content_y = origin
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    draw.text(
        (x0 + 54.0 * scale, content_y + 4.0 * scale),
        title_color,
        "Transform gizmos · production states",
    )
    draw.text(
        (x0 + 54.0 * scale, content_y + 28.0 * scale),
        note_color,
        "RGB defaults; hover/active uses Primary Bright; drag values use backed labels.",
    )
    for row, (mode, heading) in enumerate(
        (
            ("translate", "Position · RGB axes and plane handles"),
            ("rotate", "Rotation · three front half-rings and screen ring"),
        )
    ):
        row_y = content_y + (190.0 + row * 258.0) * scale
        draw.text((x0 + 54.0 * scale, row_y - 120.0 * scale), title_color, heading)
        states = (
            ("Default", "default"),
            ("Hover X", "hover"),
            ("Drag X", "pressed"),
        )
        if mode == "rotate":
            states += (("Drag + Shift", "snap"),)
        step_x = 210.0 if mode == "rotate" else 238.0
        for index, (label, forced_state) in enumerate(states):
            center = (x0 + (142.0 + index * step_x) * scale, row_y)
            _draw_transform_gizmo(
                draw,
                f"##probe-transform-{mode}-{index}",
                center,
                scale,
                forced_state=forced_state,
                mode=mode,
                smoothing=state.transform_smoothing,
            )
            label_width, _ = draw.text_size(label)
            draw.text(
                (center[0] - label_width * 0.5, row_y + 108.0 * scale),
                note_color,
                label,
            )
    draw.text(
        (x0 + 54.0 * scale, content_y + 612.0 * scale),
        note_color,
        "Hover/active + snap tick = Primary Bright; drag sector = Primary Dim at low alpha.",
    )
    draw.text(
        (x0 + 54.0 * scale, content_y + 638.0 * scale),
        note_color,
        "2D / 3D changes geometry only; label and interaction-state rules stay identical.",
    )


def draw_joint_study(draw, origin, size, scale, state) -> None:
    x0, content_y = origin
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    draw.text(
        (x0 + 54.0 * scale, content_y + 4.0 * scale),
        title_color,
        "Joint gizmos · production states",
    )
    draw.text(
        (x0 + 54.0 * scale, content_y + 28.0 * scale),
        note_color,
        "Primary handles · blue MIN tick · red MAX tick · delayed read-only labels.",
    )
    imgui.push_id("geometry-joint-gizmo")
    _draw_joint_gizmo(
        draw,
        (x0 + 54.0 * scale, content_y + 58.0 * scale),
        scale,
        state,
        item_id="geometry-joint",
    )
    imgui.pop_id()
    draw.text(
        (x0 + 54.0 * scale, content_y + 354.0 * scale),
        note_color,
        "Current ticks retain Primary; MIN/MAX ticks stay above them and expose delayed labels.",
    )
    draw.text(
        (x0 + 54.0 * scale, content_y + 380.0 * scale),
        note_color,
        "Hover a handle for the Type value hint; double-click the handle to enter it.",
    )
    draw.text(
        (x0 + 54.0 * scale, content_y + 420.0 * scale),
        title_color,
        "Hinge drag + Shift",
    )
    _draw_joint_rotation_feedback(
        draw,
        (x0 + 230.0 * scale, content_y + 536.0 * scale),
        76.0 * scale,
        scale,
    )
    draw.text(
        (x0 + 340.0 * scale, content_y + 488.0 * scale),
        CONCEPT_THEME.primary_bright,
        "Primary Bright  active arc / tick",
    )
    draw.text(
        (x0 + 340.0 * scale, content_y + 516.0 * scale),
        CONCEPT_THEME.primary_dim,
        "Primary Dim  sweep sector · 24% alpha",
    )
    draw.text(
        (x0 + 340.0 * scale, content_y + 544.0 * scale),
        note_color,
        "Text Disabled  passive snap ticks",
    )

    helper_x = x0 + max(700.0 * scale, size.x * 0.47)
    helper_width = min(500.0 * scale, x0 + size.x - helper_x - 30.0 * scale)
    draw.text((helper_x, content_y + 4.0 * scale), title_color, "Camera / light helpers")
    draw.text(
        (helper_x, content_y + 28.0 * scale),
        note_color,
        "Default · hover · selected; selected entity alone reveals its influence volume.",
    )
    _draw_helper_viewport(
        draw,
        (
            helper_x,
            content_y + 58.0 * scale,
            helper_x + helper_width,
            content_y + 376.0 * scale,
        ),
        scale,
        state,
    )
    draw.text((helper_x, content_y + 406.0 * scale), title_color, "SVG source pipeline")
    draw.text(
        (helper_x, content_y + 432.0 * scale),
        CONCEPT_THEME.primary_bright,
        "20×20 SVG  →  validated build-time paths  →  Draw2D",
    )
    draw.text(
        (helper_x, content_y + 458.0 * scale),
        note_color,
        "No SVG parser, tessellator, or raster cache in the frame loop.",
    )
    for index, icon_scale in enumerate((0.75, 1.0, 1.5)):
        center = (helper_x + (48.0 + index * 98.0) * scale, content_y + 520.0 * scale)
        if state.preview_icon_library:
            draw_concept_icon(
                draw,
                state.icon_center("helper-camera", center, 20.0 * scale * icon_scale),
                20.0 * scale * icon_scale,
                "helper-camera",
                CONCEPT_THEME.text,
                padding=state.icon_padding_for_glyph("helper-camera"),
                stroke_width=state.icon_stroke_for_glyph("helper-camera"),
                tuning=state.icon_tuning(),
                alignment=state.icon_alignment_for_glyph("helper-camera"),
            )
            draw_concept_icon(
                draw,
                state.icon_center(
                    "helper-light",
                    (center[0] + 40.0 * scale, center[1]),
                    20.0 * scale * icon_scale,
                ),
                20.0 * scale * icon_scale,
                "helper-light",
                CONCEPT_THEME.text,
                padding=state.icon_padding_for_glyph("helper-light"),
                stroke_width=state.icon_stroke_for_glyph("helper-light"),
                tuning=state.icon_tuning(),
                alignment=state.icon_alignment_for_glyph("helper-light"),
            )
        else:
            _draw_camera_icon(draw, center, CONCEPT_THEME.text, scale * icon_scale)
            _draw_light_icon(
                draw,
                (center[0] + 40.0 * scale, center[1]),
                CONCEPT_THEME.text,
                scale * icon_scale,
            )
        draw.text(
            (center[0] - 12.0 * scale, center[1] + 28.0 * scale),
            note_color,
            f"{icon_scale:.2g}×",
        )
    draw.text(
        (x0 + 54.0 * scale, content_y + 650.0 * scale),
        note_color,
        "Renderer acceptance: 3D depth, occlusion and picking stay in make gizmo / lighting tests.",
    )
