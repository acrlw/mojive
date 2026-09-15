"""Geometry experiments composed from shared controls and review specimens."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive.ui import perturb as perturb_ui
from mojive.ui import viewcube as view_ui
from mojive.ui.drag_link import draw_drag_link
from mojive.ui.icons import draw_concept_icon
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels import PanelContext
from mojive.ui.panels.filters import filter_pills, severity_color, severity_icon, severity_meshes
from mojive.ui.panels.value_cards import draw_value_rail, interval_text, value_card, value_rail
from mojive.ui.theme import rgb8
from mojive.ui.viewport_widgets import (
    PLAYBACK_HALF_HEIGHT_PT,
    PLAYBACK_RESET_SCALE,
    RECORDING_OPTIONS_GLYPH_SCALE,
    RECORDING_OPTIONS_STROKE_SCALE,
    TOOL_GLYPH_SCALE,
    capsule_points,
    draw_mouse_hint_glyph,
    draw_playback_glyph,
    draw_recording_options_glyph,
    draw_status,
    draw_tool_glyph,
)

from ..ui_capsule_geometry import capsule_outline_color, end_padding
from ..ui_redesign import RECORD_GLYPH_RADIUS, RESET_GLYPH_SCALE, draw_recording_glyph
from .fixtures import (
    CONCEPT_THEME,
    CORNER_CONTROLS,
    CORNER_VIEW_GIZMO,
    GEOMETRY_TABS,
    GIZMO_IDENTITY_F64,
    GIZMO_PROBE_CAMERA,
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
from .icon_library import _draw_icon_library_page
from .layout import (
    _dimension_line,
    _flags,
    _geometry_canvas_size,
    _icon_library_canvas_size,
    _virtual_canvas_size,
    _wrapped_tabs,
)
from .panels import (
    _draw_panel_gallery,
    _draw_shell_settings_tab,
    _draw_status_tab,
    _draw_workspaces_tab,
    _probe_status_hints,
)
from .state import ProbeState
from .tuning import _draw_corner_controls, _draw_geometry_controls
from .widgets import (
    _circular_icon_button,
    _draw_hint_bar,
    _draw_playback,
    _draw_reset_icon,
    _draw_tool_column,
    _draw_tool_icon,
    _draw_value_input_card,
    _preview_search_input,
)


def _draw_corner_page(draw: ImguiDraw2D, origin, scale: float, state: ProbeState) -> None:
    x0, y0 = origin
    _draw_corner_controls((x0 + 12 * scale, y0), (286 * scale, 790 * scale), state)
    width, height, gap = 390.0 * scale, 247.0 * scale, 12.0 * scale
    specimens = (("ImGui controls", "imgui_rounding"), *CORNER_CONTROLS)
    for index, (title, field_name) in enumerate(specimens):
        x = x0 + 320 * scale + (index % 3) * (width + gap)
        y = y0 + (index // 3) * (height + gap)
        q = 0.0 if field_name == "imgui_rounding" else getattr(state, field_name)
        item_draw = draw.with_corner_smoothing(q)
        draw.rect_filled(
            (x, y),
            (x + width, y + height),
            CONCEPT_THEME.bg_child,
            rounding=8 * scale,
            smoothing=0.0,
        )
        caption = (
            f"{title}  {state.imgui_rounding:.1f} px"
            if field_name == "imgui_rounding"
            else f"{title}  {q:.3f}"
        )
        draw.text((x + 16 * scale, y + 14 * scale), CONCEPT_THEME.text, caption)
        cx, cy = x + width * 0.5, y + height * 0.56
        imgui.push_id(field_name)
        if field_name == "imgui_rounding":
            imgui.set_cursor_screen_pos(imgui.ImVec2(x + 24 * scale, y + 64 * scale))
            imgui.button("Rounded button", imgui.ImVec2(width - 48 * scale, 40 * scale))
            imgui.set_cursor_screen_pos(imgui.ImVec2(x + 24 * scale, y + 121 * scale))
            _, state.imgui_example_enabled = imgui.checkbox("Enabled", state.imgui_example_enabled)
            imgui.set_cursor_screen_pos(imgui.ImVec2(x + 24 * scale, y + 165 * scale))
            imgui.set_next_item_width(width - 48 * scale)
            _, state.imgui_example_value = imgui.slider_float(
                "##native-corner-example", state.imgui_example_value, 0.0, 1.0, "%.2f"
            )
        elif field_name == "capsule_smoothing":
            for w, h, yy in ((280, 62, cy - 55 * scale), (88, 54, cy + 32 * scale)):
                points = capsule_points(
                    cx - w * scale * 0.5, yy - h * scale * 0.5, w * scale, h * scale, q
                )
                item_draw.convex_fill(points, CONCEPT_THEME.bg_frame)
                item_draw.polyline(points, capsule_outline_color(state), 1.5 * scale, closed=True)
        elif field_name == "playback_smoothing":
            for i, kind in enumerate(("play", "pause", "previous", "reset")):
                center = (cx + (i - 1.5) * 76 * scale, cy)
                if state.preview_icon_library:
                    concept_name = f"playback-{kind}"
                    draw_concept_icon(
                        item_draw,
                        center,
                        42.0 * scale,
                        concept_name,
                        CONCEPT_THEME.text,
                        padding=state.icon_padding_for_glyph(concept_name),
                        stroke_width=state.icon_stroke_for_glyph(concept_name),
                        tuning=state.icon_tuning(),
                        alignment=state.icon_alignment_for_glyph(concept_name),
                    )
                else:
                    draw_playback_glyph(
                        item_draw,
                        center,
                        CONCEPT_THEME.text,
                        2.1 * scale,
                        kind,
                        smoothing=q,
                    )
        elif field_name == "tool_smoothing":
            for i, kind in enumerate(("move", "rotate", "dimensions", "snap")):
                center = (cx + (i - 1.5) * 82 * scale, cy)
                if state.preview_icon_library:
                    concept_name = "tool-scale" if kind == "dimensions" else f"tool-{kind}"
                    draw_concept_icon(
                        item_draw,
                        center,
                        47.2 * scale,
                        concept_name,
                        CONCEPT_THEME.text,
                        padding=state.icon_padding_for_glyph(concept_name),
                        stroke_width=state.icon_stroke_for_glyph(concept_name),
                        rotate_ring_gap_ratio=state.rotate_ring_gap_ratio,
                        rotate_ring_cap=state.rotate_ring_cap,
                        tuning=state.icon_tuning(),
                        alignment=state.icon_alignment_for_glyph(concept_name),
                    )
                else:
                    draw_tool_glyph(
                        item_draw,
                        center,
                        CONCEPT_THEME.text,
                        2.0 * scale,
                        kind,
                        "world",
                        smoothing=q,
                    )
        elif field_name == "mouse_smoothing":
            for i, button in enumerate(("left", "right", "wheel")):
                draw_mouse_hint_glyph(
                    item_draw,
                    cx + (i - 1) * 98 * scale - 24 * scale,
                    cy,
                    button,
                    "",
                    CONCEPT_THEME,
                    3.2 * scale,
                    smoothing=q,
                )
        elif field_name == "transform_smoothing":
            for i, mode in enumerate(("translate", "dimensions")):
                _draw_transform_gizmo(
                    item_draw,
                    f"##corner-{mode}",
                    (cx + (i - 0.5) * 182 * scale, cy),
                    0.85 * scale,
                    forced_state="default",
                    mode=mode,
                    smoothing=q,
                )
            draw_drag_link(
                item_draw,
                (cx - 65 * scale, y + height - 57 * scale),
                (cx + 65 * scale, y + height - 57 * scale),
                CONCEPT_THEME.text,
                CONCEPT_THEME.text_disabled,
                2 * scale,
                5 * scale,
                0.75 * scale,
                smoothing=q,
            )
            for i in range(11):
                a = (cx + (i - 5) * 13 * scale, y + height - 24 * scale)
                item_draw.line(
                    a,
                    (a[0], a[1] + (14 if i % 5 == 0 else 8) * scale),
                    CONCEPT_THEME.text,
                    2 * scale,
                    cap="round",
                )
        elif field_name == "joint_smoothing":
            _draw_joint_gizmo(
                item_draw,
                (x + 4 * scale, y + 67 * scale),
                0.55 * scale,
                state,
                item_id="corner-joint",
                show_limit_labels=False,
            )
        elif field_name == "view_smoothing":
            zoom = 1.6 * scale
            reach = (view_ui.RADIUS_PT + view_ui.BALL_PT + view_ui.MARGIN_PT) * zoom
            rect = (cx - 100 * scale, cy - reach, 100 * scale + reach, height)
            CORNER_VIEW_GIZMO.update(GIZMO_PROBE_CAMERA, rect, (-1000, -1000), zoom, enabled=False)
            CORNER_VIEW_GIZMO.draw(item_draw, zoom, smoothing=q)
        else:
            rect = (cx - 85 * scale, cy - 85 * scale, 170 * scale, 170 * scale)
            edges = perturb_ui.silhouette_edges(
                np.zeros(3), GIZMO_IDENTITY_F64, np.ones(3) * 0.65, GIZMO_PROBE_CAMERA.eye
            )
            loop = perturb_ui.silhouette_loop(edges)
            rounded = perturb_ui.rounded_loop(
                loop, GIZMO_PROBE_CAMERA, rect, 8 * scale, smoothing=q
            )
            points = perturb_ui.project(GIZMO_PROBE_CAMERA, rounded, rect)[:, :2]
            item_draw.polyline(points, CONCEPT_THEME.text, 2 * scale, closed=True)
            perturb_ui.draw_axes(
                item_draw,
                GIZMO_PROBE_CAMERA,
                rect,
                np.zeros(3),
                GIZMO_IDENTITY_F64,
                0.7 * scale,
                smoothing=q,
            )
        if field_name == "perturb_smoothing":
            draw_drag_link(
                item_draw,
                (cx - 75 * scale, y + height - 22 * scale),
                (cx + 75 * scale, y + height - 22 * scale),
                CONCEPT_THEME.text,
                CONCEPT_THEME.text_disabled,
                2 * scale,
                5 * scale,
                0.75 * scale,
                smoothing=q,
            )
        imgui.pop_id()


def _draw_diagnostic_gallery(state, scale):
    """Inspect the production severity paths and control rows at several sizes."""
    ctx = PanelContext(None, None, theme=CONCEPT_THEME, style_scale=scale, painter=state.painter)
    draw = state.painter()
    reference = rgb8(255, 126, 48, 0.92)
    imgui.text("Output severity · production geometry")
    imgui.text_disabled("Info: circle + i   |   Warning: circle + !   |   Error: circle + cross")
    imgui.text_disabled(
        "Orange guide: nominal diameter · labels: size · frame diameter / mark height"
    )
    imgui.spacing()
    origin = imgui.get_cursor_screen_pos()
    for index, level in enumerate(("info", "warning", "error")):
        x, y = origin.x, origin.y + index * 94 * scale
        draw.text((x, y + 28 * scale), CONCEPT_THEME.text, level.capitalize())
        for column, size in enumerate((14, 20, 32, 56)):
            center = (x + (170 + 128 * column) * scale, y + 32 * scale)
            severity_icon(draw, center, size * scale, level, severity_color(ctx.theme, level))
            draw.circle(center, size * scale * 0.5, reference, 0.65 * scale, segments=64)
            meshes = severity_meshes(size * scale, level)
            frame = np.asarray(meshes[0][0], np.float64)
            extent = np.linalg.norm(frame, axis=1).max() / (size * scale * 0.5)
            mark = np.concatenate([np.asarray(mesh[0], np.float64) for mesh in meshes[1:]])
            mark_height = np.ptp(mark[:, 1]) / (size * scale)
            label = f"{size} · {extent:.0%}/{mark_height:.0%}"
            label_width, _ = draw.text_size(label)
            draw.text(
                (center[0] - label_width * 0.5, y + 67 * scale),
                CONCEPT_THEME.text_disabled,
                label,
            )
    imgui.dummy(imgui.ImVec2(720 * scale, 300 * scale))
    imgui.text("Palette: #8AB7C0 / #C9A15C / #D06744 · mark smoothing 0.618")
    imgui.text("Click a capsule to toggle it; log text keeps the same neutral color.")
    clicked = filter_pills(
        ctx,
        "diagnostic-level",
        (("info", "12", "info"), ("warning", "2", "warning"), ("error", "999", "error")),
        state.diagnostic_levels,
        compact=True,
    )
    if clicked is not None:
        state.diagnostic_levels.symmetric_difference_update({clicked})
    imgui.same_line()
    imgui.set_next_item_width(310 * scale)
    _, state.output_filter = _preview_search_input(
        state, "##diagnostic-search", state.output_filter, hint="Filter text or component..."
    )
    imgui.same_line()
    imgui.button("Clear##diagnostic-clear")
    for level, text in (
        ("info", "Loaded scene.xml"),
        ("warning", "Joint limit reached"),
        ("error", "Model could not be compiled"),
    ):
        if level not in state.diagnostic_levels:
            continue
        p = imgui.get_cursor_screen_pos()
        size = imgui.get_font_size()
        severity_icon(
            draw,
            (p.x + size * 0.5, p.y + size * 0.5),
            size,
            level,
            severity_color(ctx.theme, level),
        )
        draw.text((p.x + size + 10 * scale, p.y), ctx.theme.text, text)
        imgui.dummy(imgui.ImVec2(720 * scale, size + 8 * scale))
    imgui.spacing()
    imgui.text("Slider states: normal / hover / press / disabled")
    for index, (label, hovered, pressed, disabled) in enumerate(
        (
            ("Normal", False, False, False),
            ("Hover", True, False, False),
            ("Press", True, True, False),
            ("Disabled", False, False, True),
        )
    ):
        if index:
            imgui.same_line()
        imgui.begin_group()
        imgui.text_disabled(label)
        pos = imgui.get_cursor_screen_pos()
        draw_value_rail(
            draw,
            (pos.x + 6 * scale, pos.y + 12 * scale),
            (pos.x + 185 * scale, pos.y + 12 * scale),
            pos.x + 105 * scale,
            5 * scale,
            ctx.theme,
            scale,
            hovered=hovered,
            pressed=pressed,
            alpha=imgui.get_style().disabled_alpha if disabled else 1.0,
            disabled=disabled,
        )
        imgui.dummy((195 * scale, 30 * scale))
        imgui.end_group()
    imgui.text("Control / Joints · unit toggle, right-click reset · narrow reflow")
    for index, width in enumerate((420, 180)):
        if index:
            imgui.same_line()
        imgui.begin_child(f"##diagnostic-controls-{index}", (width * scale, 210 * scale))
        with value_card(ctx, "##diagnostic-actuator", "hip_motor", interval_text((-2, 2))):
            edit = value_rail(
                ctx,
                "##diagnostic-rail",
                state.hinge_ctrl,
                (-2, 2),
                initial=0.42,
                fmt="%+.3f",
                show_reset=False,
                unit="rad",
                angular_degrees=state.angular_degrees,
                toggle_unit=lambda: setattr(state, "angular_degrees", not state.angular_degrees),
            )
            state.hinge_ctrl = edit.value
        with value_card(ctx, "##diagnostic-unbounded", "custom_drive", "Unlimited"):
            edit = value_rail(
                ctx,
                "##diagnostic-drag",
                state.slide_ctrl,
                None,
                initial=2.5,
                fmt="%+.3f",
                show_reset=False,
            )
            state.slide_ctrl = edit.value
        imgui.end_child()


def _draw_geometry_page(available, scale: float, state: ProbeState) -> None:
    child_flags = _flags(imgui.ChildFlags_.borders)
    window_flags = imgui.WindowFlags_.none.value
    if not imgui.begin_child("Geometry###ProbeGeometry", available, child_flags, window_flags):
        imgui.end_child()
        return

    active_tab = _wrapped_tabs(
        tuple(
            (label, "geometry", label.casefold().replace(" ", "-").replace("&", "and"))
            for label in GEOMETRY_TABS
        ),
        state.geometry_tab,
        imgui.get_content_region_avail().x,
        initial=None if state.geometry_tab_initialized else state.geometry_tab,
    )
    state.geometry_tab = active_tab
    state.geometry_tab_initialized = True

    canvas_available = imgui.get_content_region_avail()
    canvas_flags = _flags(
        imgui.WindowFlags_.horizontal_scrollbar,
        imgui.WindowFlags_.always_vertical_scrollbar,
    )
    if not imgui.begin_child(
        "Geometry canvas###ProbeGeometryCanvas",
        canvas_available,
        imgui.ChildFlags_.none.value,
        canvas_flags,
    ):
        imgui.end_child()
        imgui.end_child()
        return

    canvas_cursor = imgui.get_cursor_pos()
    canvas_origin = imgui.get_cursor_screen_pos()
    canvas_width, canvas_height = _virtual_canvas_size(
        canvas_available,
        scale,
        _icon_library_canvas_size(state.icon_library_tab)
        if active_tab == "Icon library"
        else _geometry_canvas_size(active_tab, state),
    )
    size = imgui.ImVec2(canvas_width, canvas_height)
    x0, y0 = float(canvas_origin.x), float(canvas_origin.y)
    draw = state.painter()

    content_y = float(imgui.get_cursor_screen_pos().y) + 8.0 * scale
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    icon_radius = float(state.overlay_icon_radius)
    state_radius = icon_radius + float(state.overlay_radial_step)
    shell_radius = state_radius + float(state.overlay_radial_step)
    center_step = float(state.overlay_center_step)
    group_step = center_step + float(state.tool_group_gap)
    controls_width = 360.0 * scale
    controls_x = x0 + size.x - controls_width - 24.0 * scale
    controls_y = content_y
    controls_height = min(760.0 * scale, y0 + size.y - controls_y - 18.0 * scale)

    show_geometry_controls = active_tab in ("Playback", "Tools", "Hints & input")

    if active_tab == "Tools":
        draw = draw.with_corner_smoothing(state.tool_smoothing)

    if active_tab == "Corners":
        _draw_corner_page(draw, (x0, content_y), scale, state)
    elif active_tab == "Playback":
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

    elif active_tab == "Tools":
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

    elif active_tab == "Icon library":
        _draw_icon_library_page(draw, (x0, content_y), size.x, scale, state)

    elif active_tab == "Hints & input":
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

    elif active_tab == "Transform gizmos":
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

    elif active_tab == "Joint & helpers":
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
                    center,
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
                    (center[0] + 40.0 * scale, center[1]),
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

    elif active_tab == "Diagnostics":
        imgui.set_cursor_screen_pos((x0 + 18 * scale, content_y))
        _draw_diagnostic_gallery(state, scale)

    elif active_tab == "Status":
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0 * scale, content_y))
        _draw_status_tab(
            imgui.ImVec2(
                size.x - 24.0 * scale,
                y0 + size.y - content_y - 12.0 * scale,
            ),
            scale,
            state,
        )

    elif active_tab == "Shell & settings":
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0 * scale, content_y))
        _draw_shell_settings_tab(
            imgui.ImVec2(
                size.x - 24.0 * scale,
                y0 + size.y - content_y - 12.0 * scale,
            ),
            scale,
            state,
        )

    elif active_tab == "Panels":
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0 * scale, content_y))
        _draw_panel_gallery(
            imgui.ImVec2(
                size.x - 24.0 * scale,
                y0 + size.y - content_y - 12.0 * scale,
            ),
            state,
            scale,
        )

    else:
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0 * scale, content_y))
        _draw_workspaces_tab(
            imgui.ImVec2(
                size.x - 24.0 * scale,
                y0 + size.y - content_y - 12.0 * scale,
            ),
            scale,
            state,
        )

    if active_tab in (
        "Playback",
        "Tools",
        "Hints & input",
        "Transform gizmos",
        "Joint & helpers",
    ):
        extent = 690.0 * scale
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 8.0 * scale, content_y + extent))
        imgui.dummy(imgui.ImVec2(1.0, 1.0))

    if show_geometry_controls:
        _draw_geometry_controls(
            (controls_x, controls_y),
            (controls_width, controls_height),
            state,
        )

    imgui.set_cursor_pos(
        imgui.ImVec2(
            canvas_cursor.x + canvas_width - 1.0,
            canvas_cursor.y + canvas_height - 1.0,
        )
    )
    imgui.dummy(imgui.ImVec2(1.0, 1.0))
    imgui.end_child()
    imgui.end_child()
