"""Transform, joint and scene-helper specimens using production geometry."""

from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from mojive.geometry2d.curves import CORNER_SMOOTHING
from mojive.interaction import gizmo as gizmo_geometry
from mojive.ui import gizmo as gizmo_ui
from mojive.ui.icons import draw_concept_icon
from mojive.ui.imgui_draw import ImguiDraw2D

from .fixtures import (
    CONCEPT_THEME,
    GIZMO_IDENTITY_F32,
    GIZMO_IDENTITY_F64,
    GIZMO_PROBE_CAMERA,
    GIZMO_PROBE_SPECIMENS,
    JOINT_COLOR,
    JOINT_HINGE_ARC,
    VIEW_A,
)
from .state import ProbeState
from .widgets import _draw_label_button


def _joint_double_click(
    item_id: str,
    lo,
    hi,
    *,
    points=(),
    tolerance: float = 0.0,
) -> bool:
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(lo[0]), float(lo[1])))
    imgui.invisible_button(item_id, imgui.ImVec2(float(hi[0] - lo[0]), float(hi[1] - lo[1])))
    if not imgui.is_item_hovered() or not imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
        return False
    if not points:
        return True
    mouse = imgui.get_io().mouse_pos
    return min(math.hypot(mouse.x - point[0], mouse.y - point[1]) for point in points) <= tolerance


def _open_joint_value(state: ProbeState, title: str, value: float, unit: str) -> None:
    state.value_open = False
    state.joint_value_title = title
    state.joint_value = value
    state.joint_value_unit = unit
    state.joint_value_open = True


def _draw_joint_gizmo(
    draw: ImguiDraw2D,
    origin,
    scale: float,
    state: ProbeState | None = None,
    *,
    item_id: str = "joint",
    show_limit_labels: bool = True,
) -> None:
    """Draw the production slide/hinge silhouettes with optional delayed labels."""

    if state is not None:
        draw = draw.with_corner_smoothing(state.joint_smoothing)
    x, y = origin
    stroke = gizmo_ui.JOINT_RANGE_WIDTH_PT * scale
    tick = 12.0 * scale
    hinge_tick = gizmo_ui.JOINT_LIMIT_TICK_PT * scale

    # Slide: Primary range/handle, semantic endpoint ticks, labels offset from
    # the axis so neither the line nor the model body can pierce the text.
    slide_y = y + 148.0 * scale
    slide_min = x + 74.0 * scale
    slide_max = x + 354.0 * scale
    draw.rect_filled(
        (x + 176.0 * scale, slide_y - 30.0 * scale),
        (x + 246.0 * scale, slide_y + 30.0 * scale),
        CONCEPT_THEME.bg_frame,
        rounding=2.0 * scale,
    )
    draw.rect(
        (x + 176.0 * scale, slide_y - 30.0 * scale),
        (x + 246.0 * scale, slide_y + 30.0 * scale),
        CONCEPT_THEME.border,
        1.0 * scale,
        rounding=2.0 * scale,
    )
    draw.line((slide_min, slide_y), (slide_max, slide_y), JOINT_COLOR, stroke)
    draw.line(
        (slide_min, slide_y - tick * 0.5),
        (slide_min, slide_y + tick * 0.5),
        CONCEPT_THEME.axis_color(2),
        stroke,
        cap="round",
    )
    draw.line(
        (slide_max, slide_y - tick * 0.5),
        (slide_max, slide_y + tick * 0.5),
        CONCEPT_THEME.axis_color(0),
        stroke,
        cap="round",
    )
    current_x = x + 211.0 * scale
    current_tick = 20.0 * scale
    draw.line(
        (current_x, slide_y - current_tick * 0.5),
        (current_x, slide_y + current_tick * 0.5),
        gizmo_ui.JOINT_CURRENT_COLOR,
        stroke,
        cap="round",
    )
    # Opposing drag handles sit off the scale line; the line itself shares the
    # same pointer target so the control remains discoverable.
    arrows = gizmo_ui.joint_slide_arrow_polygons(
        np.asarray((current_x, slide_y)),
        np.asarray((1.0, 0.0)),
        scale,
        smoothing=draw.corner_smoothing,
    )
    for arrow in arrows:
        points = tuple((float(point[0]), float(point[1])) for point in arrow)
        draw.fringed_concave_fill(
            points,
            JOINT_COLOR,
        )
    hit_arrows = gizmo_ui.joint_slide_arrow_polygons(
        np.asarray((current_x, slide_y)), np.asarray((1.0, 0.0)), scale, for_hit_test=True
    )
    arrow_points = np.concatenate(
        (*hit_arrows, np.asarray(((slide_min, slide_y), (slide_max, slide_y))))
    )
    arrow_lo = np.min(arrow_points, axis=0)
    arrow_hi = np.max(arrow_points, axis=0)
    if state is not None and _joint_double_click(
        f"##{item_id}-slide-arrow",
        (arrow_lo[0] - 4.0 * scale, arrow_lo[1] - 4.0 * scale),
        (arrow_hi[0] + 4.0 * scale, arrow_hi[1] + 4.0 * scale),
    ):
        _open_joint_value(state, "slide_joint", 0.0, "m")
    if show_limit_labels:
        _draw_label_button(
            draw,
            "##joint-min-slide",
            (x + 18.0 * scale, y + 82.0 * scale),
            "MIN",
            "−0.340",
            "m",
            CONCEPT_THEME.axis_color(2),
            scale,
            interactive=False,
        )
        _draw_label_button(
            draw,
            "##joint-max-slide",
            (x + 282.0 * scale, y + 172.0 * scale),
            "MAX",
            "+0.340",
            "m",
            CONCEPT_THEME.axis_color(0),
            scale,
            interactive=False,
        )

    # Hinge: a single clean Primary arc. Endpoint ticks are radial and use the
    # same MIN/MAX semantic colors as the adjacent label dots.
    center = (x + 548.0 * scale, y + 140.0 * scale)
    radius = 72.0 * scale
    arc = tuple((center[0] + ux * radius, center[1] + uy * radius) for ux, uy in JOINT_HINGE_ARC)
    draw.polyline(arc, JOINT_COLOR, stroke, cap="round")
    for index, color in enumerate((CONCEPT_THEME.axis_color(0), CONCEPT_THEME.axis_color(2))):
        ux, uy = JOINT_HINGE_ARC[0 if index == 0 else -1]
        point = (center[0] + ux * radius, center[1] + uy * radius)
        draw.line(
            point,
            (point[0] + ux * hinge_tick, point[1] + uy * hinge_tick),
            color,
            stroke,
            cap="round",
        )
    current_ux, current_uy = JOINT_HINGE_ARC[len(JOINT_HINGE_ARC) // 2]
    current_point = (
        center[0] + current_ux * radius,
        center[1] + current_uy * radius,
    )
    draw.line(
        current_point,
        (
            current_point[0] + current_ux * current_tick,
            current_point[1] + current_uy * current_tick,
        ),
        JOINT_COLOR,
        stroke,
        cap="round",
    )
    if state is not None and _joint_double_click(
        f"##{item_id}-hinge-ring",
        (center[0] - radius - 12.0 * scale, center[1] - radius - 12.0 * scale),
        (center[0] + radius + 12.0 * scale, center[1] + radius + 12.0 * scale),
        points=arc,
        tolerance=10.0 * scale,
    ):
        _open_joint_value(state, "hinge_joint", 0.0, "°")
    if show_limit_labels:
        _draw_label_button(
            draw,
            "##joint-max-hinge",
            (x + 500.0 * scale, y + 18.0 * scale),
            "MAX",
            "+120.0",
            "°",
            CONCEPT_THEME.axis_color(0),
            scale,
            interactive=False,
        )
        _draw_label_button(
            draw,
            "##joint-min-hinge",
            (x + 392.0 * scale, y + 208.0 * scale),
            "MIN",
            "−120.0",
            "°",
            CONCEPT_THEME.axis_color(2),
            scale,
            interactive=False,
        )


def _draw_joint_rotation_feedback(draw: ImguiDraw2D, center, radius: float, scale: float) -> None:
    """Show the production hinge drag and Shift feedback colors."""

    arc = tuple((center[0] + ux * radius, center[1] + uy * radius) for ux, uy in JOINT_HINGE_ARC)
    draw.polyline(arc, JOINT_COLOR, 3.0 * scale, cap="round")

    start_index = 80
    end_index = 170
    sweep = arc[start_index : end_index + 1]
    draw.triangle_fan_fill(
        (center, *sweep),
        (*CONCEPT_THEME.primary_dim[:3], 0.24),
    )
    draw.polyline(sweep, CONCEPT_THEME.primary_bright, 3.0 * scale, cap="round")

    for index in range(0, len(JOINT_HINGE_ARC), 20):
        ux, uy = JOINT_HINGE_ARC[index]
        point = arc[index]
        length = (9.0 if index % 60 == 0 else 6.0) * scale
        draw.line(
            point,
            (point[0] + ux * length, point[1] + uy * length),
            CONCEPT_THEME.text_disabled,
            1.1 * scale,
            cap="round",
        )

    ux, uy = JOINT_HINGE_ARC[end_index]
    point = arc[end_index]
    draw.line(
        point,
        (point[0] + ux * 12.0 * scale, point[1] + uy * 12.0 * scale),
        CONCEPT_THEME.primary_bright,
        2.6 * scale,
        cap="round",
    )
    draw.circle_filled(center, 3.0 * scale, CONCEPT_THEME.text, segments=18)


def _draw_camera_icon(draw: ImguiDraw2D, center, color, scale: float) -> None:
    """Draw a compact camera helper from reusable vector primitives."""

    x, y = center
    stroke = max(1.0, 1.35 * scale)
    draw.rect(
        (x - 8.0 * scale, y - 5.0 * scale),
        (x + 8.0 * scale, y + 6.0 * scale),
        color,
        stroke,
        rounding=1.8 * scale,
    )
    draw.polyline(
        (
            (x - 4.5 * scale, y - 5.0 * scale),
            (x - 2.4 * scale, y - 8.0 * scale),
            (x + 3.2 * scale, y - 8.0 * scale),
            (x + 5.2 * scale, y - 5.0 * scale),
        ),
        color,
        stroke,
    )
    draw.circle((x + 0.5 * scale, y + 0.5 * scale), 3.2 * scale, color, stroke, segments=24)


def _draw_light_icon(draw: ImguiDraw2D, center, color, scale: float) -> None:
    """Draw a light helper as a bulb, base, and evenly spaced short rays."""

    x, y = center
    stroke = max(1.0, 1.35 * scale)
    draw.circle((x, y - 1.5 * scale), 4.8 * scale, color, stroke, segments=28)
    draw.line(
        (x - 3.1 * scale, y + 4.2 * scale),
        (x + 3.1 * scale, y + 4.2 * scale),
        color,
        stroke,
    )
    draw.line(
        (x - 2.2 * scale, y + 6.8 * scale),
        (x + 2.2 * scale, y + 6.8 * scale),
        color,
        stroke,
    )
    for ux, uy in ((0.0, -1.0), (0.707, -0.707), (1.0, 0.0), (-0.707, -0.707), (-1.0, 0.0)):
        draw.line(
            (x + ux * 7.0 * scale, y - 1.5 * scale + uy * 7.0 * scale),
            (x + ux * 9.6 * scale, y - 1.5 * scale + uy * 9.6 * scale),
            color,
            stroke,
        )


def _draw_scene_helper(
    draw: ImguiDraw2D,
    item_id: str,
    center,
    kind: str,
    index: int,
    scale: float,
    state: ProbeState,
) -> None:
    hit = 30.0 * scale
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(center[0] - hit * 0.5), float(center[1] - hit * 0.5))
    )
    clicked = imgui.invisible_button(item_id, imgui.ImVec2(hit, hit))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    if clicked:
        state.helper_selection = index
    selected = state.helper_selection == index
    color = (
        CONCEPT_THEME.primary_bright
        if selected or hovered or active
        else CONCEPT_THEME.text_disabled
    )
    if active:
        draw.circle_filled(center, 13.0 * scale, CONCEPT_THEME.bg_frame_active, segments=32)
    elif hovered:
        draw.circle_filled(center, 13.0 * scale, CONCEPT_THEME.bg_frame_hovered, segments=32)
    if state.preview_icon_library:
        name = "helper-camera" if kind == "camera" else "helper-light"
        draw_concept_icon(
            draw,
            state.icon_center(name, center, 20.0 * scale),
            20.0 * scale,
            name,
            color,
            padding=state.icon_padding_for_glyph(name),
            stroke_width=state.icon_stroke_for_glyph(name),
            tuning=state.icon_tuning(),
            alignment=state.icon_alignment_for_glyph(name),
        )
    elif kind == "camera":
        _draw_camera_icon(draw, center, color, scale)
    else:
        _draw_light_icon(draw, center, color, scale)


def _draw_transform_gizmo(
    draw: ImguiDraw2D,
    item_id: str,
    center,
    scale: float,
    *,
    forced_state: str,
    mode: str,
    smoothing: float = CORNER_SMOOTHING,
) -> None:
    """Render the same flat gizmo geometry and color states as the application."""

    draw = draw.with_corner_smoothing(smoothing)
    hit = 190.0 * scale
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(center[0] - hit * 0.5), float(center[1] - hit * 0.5))
    )
    imgui.invisible_button(item_id, imgui.ImVec2(hit, hit))
    interactive = (
        "pressed" if imgui.is_item_active() else "hover" if imgui.is_item_hovered() else ""
    )
    display_state = interactive or ("pressed" if forced_state == "snap" else forced_state)
    x, y = center
    rect = (x - 98.0 * scale, y - 98.0 * scale, 196.0 * scale, 196.0 * scale)
    camera = GIZMO_PROBE_CAMERA
    specimen = GIZMO_PROBE_SPECIMENS[mode]
    specimen._frame.corner_smoothing = smoothing
    specimen._visible = True
    specimen._interactive = True
    specimen._using = False
    specimen._snapping = False
    specimen._label = ""
    specimen._rotation_angle = 0.0
    specimen._rotation_raw_angle = 0.0
    specimen._frame.mode = gizmo_geometry.GizmoMode(mode)
    specimen._frame.position[:] = 0.0
    specimen._frame.rotation[:] = GIZMO_IDENTITY_F32
    specimen._frame.active_rotation_overlay = False
    world_size = gizmo_geometry.world_scale(
        camera, specimen._frame.position, rect[3], gizmo_geometry.SIZE_PT * scale
    )
    axis_mask, plane_mask = gizmo_geometry.visibility(
        camera, specimen._frame.position, specimen._frame.rotation, rect, world_size
    )
    specimen._frame.axis_mask = axis_mask
    specimen._frame.plane_mask = plane_mask
    hot = gizmo_geometry.GizmoHandle.X if mode != "rotate" else gizmo_geometry.GizmoHandle.ROTATE_X
    specimen._hovered = hot if display_state == "hover" else gizmo_geometry.GizmoHandle.NONE
    specimen._active = hot if display_state == "pressed" else gizmo_geometry.GizmoHandle.NONE
    specimen._frame.hovered = specimen._hovered
    specimen._frame.active = specimen._active
    if mode == "rotate" and display_state == "pressed":
        specimen._using = True
        specimen._snapping = forced_state == "snap"
        specimen._start_pos[:] = 0.0
        specimen._start_mat[:] = GIZMO_IDENTITY_F64
        specimen._start_basis[:] = GIZMO_IDENTITY_F64
        specimen._axis[:] = (1.0, 0.0, 0.0)
        specimen._rotation_start_vec[:] = (0.0, 1.0, 0.0)
        specimen._rotation_angle = math.radians(55.0)
        specimen._rotation_raw_angle = specimen._rotation_angle
        specimen._frame.active_rotation_overlay = True
        specimen._label = "X +55.0 °"
    specimen.draw_overlay(camera, rect, draw, style_scale=scale)
    if display_state == "pressed" and mode != "rotate":
        _draw_label_button(
            draw,
            f"{item_id}-value",
            (x + 28.0 * scale, y - 50.0 * scale),
            "X",
            "+0.250" if mode == "translate" else "+15.0",
            "m" if mode == "translate" else "°",
            CONCEPT_THEME.axis_color(0),
            scale,
        )


def _draw_helper_viewport(draw: ImguiDraw2D, rect, scale: float, state: ProbeState) -> None:
    x0, y0, x1, y1 = rect
    draw.rect_filled((x0, y0), (x1, y1), VIEW_A, rounding=3.0 * scale)
    draw.rect((x0, y0), (x1, y1), CONCEPT_THEME.border, 1.0 * scale, rounding=3.0 * scale)
    camera_center = (x0 + (x1 - x0) * 0.70, y0 + (y1 - y0) * 0.43)
    light_centers = (
        (x0 + (x1 - x0) * 0.24, y0 + (y1 - y0) * 0.28),
        (x0 + (x1 - x0) * 0.45, y0 + (y1 - y0) * 0.62),
        (x0 + (x1 - x0) * 0.78, y0 + (y1 - y0) * 0.75),
    )
    if state.influence_volumes:
        if state.helper_selection == 3:
            cx, cy = camera_center
            near_left = (cx - 16.0 * scale, cy + 24.0 * scale)
            near_right = (cx + 16.0 * scale, cy + 24.0 * scale)
            far_left = (cx - 76.0 * scale, cy + 104.0 * scale)
            far_right = (cx + 76.0 * scale, cy + 104.0 * scale)
            draw.polyline(
                (near_left, far_left, far_right, near_right),
                CONCEPT_THEME.primary_bright,
                1.5 * scale,
                closed=True,
            )
        elif 0 <= state.helper_selection < len(light_centers):
            center = light_centers[state.helper_selection]
            draw.circle(
                center,
                58.0 * scale,
                (*CONCEPT_THEME.primary_bright[:3], 0.72),
                1.4 * scale,
                segments=64,
            )
    for index, center in enumerate(light_centers):
        _draw_scene_helper(draw, f"##probe-light-{index}", center, "light", index, scale, state)
    _draw_scene_helper(draw, "##probe-camera-helper", camera_center, "camera", 3, scale, state)
