"""Geometry experiments composed from shared controls and review specimens."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive.tools.ui_capsule_geometry import capsule_outline_color
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
    capsule_points,
    draw_mouse_hint_glyph,
    draw_playback_glyph,
    draw_tool_glyph,
)

from .fixtures import (
    CONCEPT_THEME,
    CORNER_CONTROLS,
    CORNER_VIEW_GIZMO,
    GEOMETRY_TABS,
    GIZMO_IDENTITY_F64,
    GIZMO_PROBE_CAMERA,
)
from .geometry_scenes import (
    draw_hint_study,
    draw_joint_study,
    draw_playback_study,
    draw_tool_study,
    draw_transform_study,
)
from .gizmos import (
    _draw_joint_gizmo,
    _draw_transform_gizmo,
)
from .icon_library import _draw_icon_library_page
from .layout import (
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
)
from .state import ProbeState
from .tuning import _draw_corner_controls, _draw_geometry_controls
from .widgets import (
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
                        state.icon_center(concept_name, center, 42.0 * scale),
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
                        state.icon_center(concept_name, center, 47.2 * scale),
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

    if active_tab == "SVG icons":
        from .svg_icons import draw_svg_page

        if imgui.begin_child("SVG asset canvas", imgui.get_content_region_avail()):
            draw_svg_page(state.svg, scale)
        imgui.end_child()
        imgui.end_child()
        return

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
        draw_playback_study(draw, (x0, content_y), scale, state, controls_x)

    elif active_tab == "Tools":
        draw_tool_study(draw, (x0, content_y), size, scale, state)

    elif active_tab == "Icon library":
        _draw_icon_library_page(draw, (x0, content_y), size.x, scale, state)

    elif active_tab == "Hints & input":
        draw_hint_study(draw, (x0, content_y), size, scale, state)

    elif active_tab == "Transform gizmos":
        draw_transform_study(draw, (x0, content_y), scale, state)

    elif active_tab == "Joint & helpers":
        draw_joint_study(draw, (x0, content_y), size, scale, state)

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
