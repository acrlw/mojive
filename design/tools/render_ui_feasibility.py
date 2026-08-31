#!/usr/bin/env python3
"""Render the UI feasibility probe through Forge Viewer's real ImGui backend.

The image is captured from the OpenGL framebuffer. Standard controls are real
ImGui widgets; icons, capsules, viewport labels, and the timeline use the same
``ImguiDraw2D`` adapter as the application.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from forge_viewer import gizmo as gizmo_geometry
from forge_viewer.types import CameraView
from forge_viewer.ui import gizmo as gizmo_ui
from forge_viewer.ui import theme as theme_mod
from forge_viewer.ui.draw2d import ImguiDraw2D
from forge_viewer.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from forge_viewer.ui.theme import THEME, rgb8
from forge_viewer.ui.viewport_widgets import (
    CAPSULE_SURFACE_ALPHA,
    OVERLAY_GEOMETRY,
    TOOL_GLYPH_SCALE,
    capsule_points,
    default_tool_hints,
    draw_playback_glyph,
    draw_status,
    draw_tool_glyph,
    mouse_button_fill_geometry,
)
from forge_viewer.ui.window import Window, WindowConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output" / "ui-drawing-feasibility.png"

CONCEPT_THEME = replace(
    THEME,
    primary=rgb8(156, 191, 141),
    primary_bright=rgb8(184, 210, 172),
    primary_dim=rgb8(103, 135, 90),
    danger=rgb8(208, 103, 68),
    warning=rgb8(201, 161, 92),
    text=rgb8(220, 222, 227),
    text_disabled=rgb8(123, 129, 137),
    bg_window=rgb8(30, 33, 37),
    bg_child=rgb8(26, 29, 32),
    bg_popup=rgb8(24, 27, 30),
    bg_frame=rgb8(43, 47, 52),
    bg_frame_hovered=rgb8(54, 59, 65),
    bg_frame_active=rgb8(64, 70, 77),
    bg_header=rgb8(48, 53, 58),
    border=rgb8(58, 63, 69),
    axis_colors={
        "x": rgb8(220, 119, 115),
        "y": rgb8(82, 170, 92),
        "z": rgb8(111, 148, 229),
    },
)
JOINT_COLOR = rgb8(175, 132, 183)
# Gizmo strokes need to read against the viewport; Inspector axis badges need
# the inverse contrast because their 14 pt glyphs are white. Keep the hue
# identity, but use darker role-specific surfaces for the badges.
AXIS_BADGE_COLORS = (
    rgb8(168, 78, 75),
    rgb8(49, 122, 58),
    rgb8(72, 104, 173),
)
AXIS_BADGE_HOVERED = (
    rgb8(178, 85, 81),
    rgb8(55, 132, 64),
    rgb8(80, 111, 184),
)
AXIS_BADGE_ACTIVE = (
    rgb8(184, 90, 86),
    rgb8(56, 134, 65),
    rgb8(83, 113, 186),
)
AXIS_BADGE_TEXT = (1.0, 1.0, 1.0, 1.0)
VIEW_A = rgb8(31, 35, 39)
VIEW_B = rgb8(36, 40, 45)

# Overlay glyphs start from one 20×20 logical coordinate system. Tool Column
# paths receive the shared optical scale so they read at runtime size; playback
# remains on the base envelope.
OVERLAY_ICON_RADIUS = 10.0
OVERLAY_STATE_RADIUS = 16.0
OVERLAY_SHELL_RADIUS = 22.0
OVERLAY_CENTER_STEP = 34.0
# Accepted M8 hinge specimen. The arc is sampled once at import; each frame
# performs only scale + translation before handing the points to Draw2D.
JOINT_HINGE_ARC = tuple(
    (math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in range(-65, 216, 10)
)

# Screen-space unit vectors for the fixed orthographic World / Body icon.
# Each tuple stores axis (ux, uy) followed by its perpendicular (nx, ny).
FRAME_AXES = (
    (0.0, -1.0, 1.0, 0.0),
    (0.866025, 0.5, -0.5, 0.866025),
    (-0.866025, 0.5, -0.5, -0.866025),
)

# Unit semicircles for a mathematically explicit capsule perimeter. Keeping the
# arcs as constants avoids ImGui's rounded-rectangle radius clamp, which leaves
# short flat facets at the two ends when the requested radius is exactly h/2.
CAPSULE_RIGHT_ARC = (
    (0.0, -1.0),
    (0.258819, -0.965926),
    (0.5, -0.866025),
    (0.707107, -0.707107),
    (0.866025, -0.5),
    (0.965926, -0.258819),
    (1.0, 0.0),
    (0.965926, 0.258819),
    (0.866025, 0.5),
    (0.707107, 0.707107),
    (0.5, 0.866025),
    (0.258819, 0.965926),
    (0.0, 1.0),
)
CAPSULE_LEFT_ARC = tuple((-x, y) for x, y in reversed(CAPSULE_RIGHT_ARC))

# The production gizmo renderer is intentionally reused for the M8 specimens.
# Keep its fixed camera and mutable specimens alive across ImGui frames: the
# probe switches their explicit display state below, but does not rebuild the
# comparatively expensive geometry owner and NumPy buffers every refresh.
GIZMO_PROBE_CAMERA = CameraView(
    eye=np.array((4.0, -4.0, 3.0), np.float32),
    target=np.zeros(3, np.float32),
    up=np.array((0.0, 0.0, 1.0), np.float32),
    aspect=1.0,
    orthographic=True,
    ortho_height=4.0,
)
GIZMO_PROBE_SPECIMENS = {mode: gizmo_ui.ObjectGizmo(mode) for mode in ("translate", "rotate")}
GIZMO_IDENTITY_F32 = np.eye(3, dtype=np.float32)
GIZMO_IDENTITY_F64 = np.eye(3, dtype=np.float64)


@dataclass
class ProbeState:
    page: str = "Workspace"
    geometry_tab: str = "Playback"
    geometry_tab_initialized: bool = False
    show_playback: bool = True
    show_tool_column: bool = True
    show_joint_gizmos: bool = True
    show_context_hints: bool = True
    show_settings: bool = True
    show_keyframes: bool = True
    show_output: bool = True
    show_icon_bounds: bool = False
    show_state_circles: bool = False
    show_construction_notes: bool = False
    playing: bool = False
    active_tool: str = "move"
    gizmo_space: str = "world"
    gizmo_style: int = 1
    frame: int = 1
    remember_input: bool = True
    viewport_overlay_scale: float = 1.25
    position_snap: float = 0.1
    rotation_snap: float = 5.0
    tick_scale: float = 1.25
    selection_padding: float = 1.2
    corner_radius: float = 4.0
    scene_icons: bool = True
    influence_volumes: bool = True
    value_open: bool = False
    value_mode: int = 0
    value: float = 45.0
    unit: int = 0
    joint_value_open: bool = False
    joint_value_title: str = "slide_joint"
    joint_value: float = 0.0
    joint_value_unit: str = "m"
    output_filter: str = ""
    output_level: int = 0
    selected_output: int = 2
    output_cleared: bool = False
    hinge_ctrl: float = 0.25
    slide_ctrl: float = 0.0
    weld_enabled: bool = True
    connect_enabled: bool = False
    hinge_position: float = 0.35
    slide_position: float = 0.0
    hierarchy_filter: str = ""
    hierarchy_kind: int = 0
    hierarchy_selection: int = 2
    hierarchy_visibility: list[bool] = field(
        default_factory=lambda: [True, True, False, True, True, True]
    )
    asset_selection: int = 1
    helper_selection: int = 1
    camera_projection: int = 0
    settings_page: int = 1
    settings_filter: str = ""
    language: int = 1
    outline_enabled: bool = True
    tonemap_enabled: bool = True
    msaa_enabled: bool = True
    debug_view: int = 0
    debug_labels: int = 0
    debug_frames: int = 0
    sim_running: bool = False
    aux_tab: str = "Output"
    mujoco_groups: list[bool] = field(default_factory=lambda: [True] * 42)
    render_flags: list[bool] = field(default_factory=lambda: [True] * 8)
    visual_flags: list[bool] = field(default_factory=lambda: [True] * 27)
    overlay_icon_radius: int = int(OVERLAY_GEOMETRY.icon_radius)
    overlay_radial_step: int = int(OVERLAY_GEOMETRY.radial_step)
    overlay_center_step: int = int(OVERLAY_GEOMETRY.center_step)
    tool_group_gap: int = int(OVERLAY_GEOMETRY.tool_group_gap)
    divider_width: int = int(OVERLAY_GEOMETRY.divider_width)
    construction_playback_scale: float = 3.0
    construction_tool_scale: float = 1.5
    tool_stroke_width: float = OVERLAY_GEOMETRY.tool_stroke
    rotate_ring_gap_ratio: float = OVERLAY_GEOMETRY.rotate_ring_gap_ratio
    rotate_ring_cap: str = OVERLAY_GEOMETRY.rotate_ring_cap
    hint_control_height: int = int(OVERLAY_GEOMETRY.hint_control_height)
    hint_padding_x: int = int(OVERLAY_GEOMETRY.hint_padding_x)
    hint_padding_y: int = int(OVERLAY_GEOMETRY.hint_padding_y)
    hint_input_gap: int = int(OVERLAY_GEOMETRY.hint_input_gap)
    hint_group_gap: int = int(OVERLAY_GEOMETRY.hint_group_gap)
    hint_chord_gap: int = int(OVERLAY_GEOMETRY.hint_chord_gap)
    hint_key_padding_x: int = int(OVERLAY_GEOMETRY.hint_key_padding_x)
    hint_mouse_width: int = int(OVERLAY_GEOMETRY.hint_mouse_width)


def _apply_concept_theme(scale: float) -> None:
    """Apply the design palette plus its accepted interaction-state mapping."""

    theme_mod.apply(imgui, CONCEPT_THEME, ui_scale=scale)
    style = imgui.get_style()

    def put(slot, color) -> None:
        style.set_color_(int(slot), imgui.ImVec4(*color))

    put(imgui.Col_.button_active, CONCEPT_THEME.bg_frame_active)
    put(imgui.Col_.header_hovered, CONCEPT_THEME.bg_frame_hovered)
    put(imgui.Col_.header_active, CONCEPT_THEME.bg_frame_active)
    put(imgui.Col_.tab_selected_overline, (0.0, 0.0, 0.0, 0.0))


def _flags(*values) -> int:
    result = 0
    for value in values:
        result |= int(value.value if hasattr(value, "value") else value)
    return result


def _draw_play_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "play")


def _draw_pause_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "pause")


def _draw_step_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "step")


def _draw_stop_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "stop")


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
    scale: float = 1.0,
) -> bool:
    diameter = cell_size * scale
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    clicked = imgui.invisible_button(item_id, imgui.ImVec2(diameter, diameter))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    theme = CONCEPT_THEME
    background = theme.bg_frame_active if selected or active else theme.bg_frame_hovered
    foreground = theme.primary_bright if selected or hovered or active else theme.text
    center = (position[0] + diameter * 0.5, position[1] + diameter * 0.5)
    if show_state_circle:
        draw.circle_filled(center, state_radius * scale, background)
        draw.circle(
            center,
            state_radius * scale,
            (*CONCEPT_THEME.primary_dim[:3], 0.95),
            1.0 * scale,
        )
    elif selected or hovered or active:
        draw.circle_filled(center, state_radius * scale, background)
    icon_surface = (
        background if show_state_circle or selected or hovered or active else theme.bg_popup
    )
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
    x, y = origin
    icon_radius = float(state.overlay_icon_radius)
    state_radius = icon_radius + float(state.overlay_radial_step)
    capsule_radius = state_radius + float(state.overlay_radial_step)
    assert math.isclose(state_radius - icon_radius, capsule_radius - state_radius)
    hit_size = float(state.overlay_center_step)
    center_step = float(state.overlay_center_step)
    width = (capsule_radius * 2.0 + center_step * 2.0) * scale
    height = capsule_radius * 2.0 * scale
    capsule = capsule_points(x, y, width, height)
    draw.convex_fill(capsule, (*CONCEPT_THEME.bg_child[:3], CAPSULE_SURFACE_ALPHA))
    draw.polyline(capsule, CONCEPT_THEME.primary, 1.4 * scale, closed=True)
    for index, (item_id, icon, selected) in enumerate(
        (
            ("##probe-play", _draw_pause_icon if state.playing else _draw_play_icon, state.playing),
            ("##probe-step", _draw_step_icon, False),
            ("##probe-stop", _draw_stop_icon, False),
        )
    ):
        center_x = x + (capsule_radius + index * center_step) * scale
        position = (
            center_x - hit_size * 0.5 * scale,
            y + (capsule_radius - hit_size * 0.5) * scale,
        )
        clicked = _circular_icon_button(
            draw,
            item_id,
            position,
            icon,
            selected=selected,
            cell_size=hit_size,
            state_radius=state_radius,
            icon_radius=icon_radius,
            icon_scale=scale * icon_radius / OVERLAY_ICON_RADIUS,
            show_icon_bound=state.show_icon_bounds,
            show_state_circle=state.show_state_circles,
            scale=scale,
        )
        if clicked and index == 0:
            state.playing = not state.playing
        elif clicked and index == 2:
            state.playing = False


def _draw_tool_icon(
    draw: ImguiDraw2D,
    center,
    color,
    scale: float,
    kind: str,
    stroke_width: float,
    rotate_ring_gap_ratio: float,
    rotate_ring_cap: str,
    surface_color,
    frame_space: str,
) -> None:
    draw_tool_glyph(
        draw,
        center,
        color,
        scale,
        kind,
        surface_color,
        frame_space,
        replace(
            OVERLAY_GEOMETRY,
            tool_stroke=stroke_width,
            rotate_ring_gap_ratio=rotate_ring_gap_ratio,
            rotate_ring_cap=rotate_ring_cap,
        ),
    )


def _draw_tool_column(draw: ImguiDraw2D, origin, scale: float, state: ProbeState) -> None:
    x, y = origin
    icon_radius = float(state.overlay_icon_radius)
    state_radius = icon_radius + float(state.overlay_radial_step)
    capsule_radius = state_radius + float(state.overlay_radial_step)
    assert math.isclose(state_radius - icon_radius, capsule_radius - state_radius)
    hit_size = float(state.overlay_center_step)
    center_step = float(state.overlay_center_step)
    group_step = center_step + float(state.tool_group_gap)
    centers = (
        capsule_radius,
        capsule_radius + center_step,
        capsule_radius + center_step * 2.0,
        capsule_radius + center_step * 2.0 + group_step,
    )
    width = capsule_radius * 2.0 * scale
    height = (centers[-1] + capsule_radius) * scale
    capsule = capsule_points(x, y, width, height)
    draw.convex_fill(capsule, (*CONCEPT_THEME.bg_child[:3], CAPSULE_SURFACE_ALPHA))
    draw.polyline(capsule, CONCEPT_THEME.primary, 1.4 * scale, closed=True)
    separator = (*CONCEPT_THEME.border[:3], 0.72)
    separator_y = y + (centers[2] + centers[3]) * 0.5 * scale
    draw.line(
        (
            x + (capsule_radius - state.divider_width * 0.5) * scale,
            separator_y,
        ),
        (
            x + (capsule_radius + state.divider_width * 0.5) * scale,
            separator_y,
        ),
        separator,
        1.0 * scale,
    )
    for index, kind in enumerate(("move", "rotate", "frame", "snap")):
        center_y = y + centers[index] * scale
        position = (
            x + (capsule_radius - hit_size * 0.5) * scale,
            center_y - hit_size * 0.5 * scale,
        )

        def icon(target, center, color, icon_scale, surface_color, current=kind):
            _draw_tool_icon(
                target,
                center,
                color,
                icon_scale,
                current,
                state.tool_stroke_width,
                state.rotate_ring_gap_ratio,
                state.rotate_ring_cap,
                surface_color,
                state.gizmo_space,
            )

        clicked = _circular_icon_button(
            draw,
            f"##probe-tool-{kind}",
            position,
            icon,
            selected=state.active_tool == kind,
            cell_size=hit_size,
            state_radius=state_radius,
            icon_radius=icon_radius * TOOL_GLYPH_SCALE,
            icon_scale=scale * icon_radius / OVERLAY_ICON_RADIUS,
            show_icon_bound=state.show_icon_bounds,
            show_state_circle=state.show_state_circles,
            scale=scale,
        )
        if clicked and kind == "frame":
            state.gizmo_space = "body" if state.gizmo_space == "world" else "world"
        elif clicked:
            state.active_tool = kind


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
) -> float:
    width, height = width * scale, height * scale
    y = center_y - height * 0.5
    corner_radius = min(width * 0.22, height * 0.18)

    outline_width = 1.25 * scale
    button_fill = mouse_button_fill_geometry(
        x,
        y,
        width,
        height,
        button,
        outline_width=outline_width,
        safety_inset=0.85 * scale,
    )
    draw.rect(
        (x, y),
        (x + width, y + height),
        CONCEPT_THEME.text,
        outline_width,
        rounding=corner_radius,
    )
    if button_fill is not None:
        mask, fill = button_fill
        draw.convex_fill(mask, (*CONCEPT_THEME.bg_child[:3], 1.0))
        draw.convex_fill(fill, CONCEPT_THEME.primary)
    wheel_lo = (x + width * 0.36, y + 1.35 * scale)
    wheel_hi = (x + width * 0.64, y + 8.35 * scale)
    if button == "wheel":
        draw.rect_filled(
            wheel_lo,
            wheel_hi,
            CONCEPT_THEME.primary,
            rounding=1.2 * scale,
        )
    if not suffix:
        return width
    label_x = x + width + 5.0 * scale
    label_width = _draw_inline_text(draw, label_x, center_y, suffix, CONCEPT_THEME.primary_bright)
    return width + 5.0 * scale + label_width


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
    draw.rect_filled((x, y), (x + width, y + height), CONCEPT_THEME.bg_frame, rounding=3.0 * scale)
    draw.rect(
        (x, y),
        (x + width, y + height),
        CONCEPT_THEME.border,
        1.0 * scale,
        rounding=3.0 * scale,
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


def _capsule_outline(x: float, y: float, width: float, height: float):
    return capsule_points(x, y, width, height)


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
    capsule = _capsule_outline(x, y, width, height)
    draw.convex_fill(capsule, (*CONCEPT_THEME.bg_child[:3], CAPSULE_SURFACE_ALPHA))
    draw.polyline(capsule, CONCEPT_THEME.primary, 1.4 * scale, closed=True)
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
        else (*CONCEPT_THEME.bg_popup[:3], CAPSULE_SURFACE_ALPHA)
    )
    foreground = CONCEPT_THEME.primary_bright if hovered else CONCEPT_THEME.text
    x, y = position
    draw.rect_filled((x, y), (x + width, y + height), background, rounding=3.0 * scale)
    draw.rect((x, y), (x + width, y + height), CONCEPT_THEME.border, 1.0, rounding=3.0 * scale)
    draw.circle_filled((x + 10.0 * scale, y + height * 0.5), 3.5 * scale, color)
    text_y = y + (height - text_height) * 0.5
    draw.text((x + 18.0 * scale, text_y), foreground, value_text)
    draw.text((x + width - unit_width - 7.0 * scale, text_y), CONCEPT_THEME.text_disabled, unit)


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
) -> None:
    """Draw the accepted M8 slide/hinge target, not the current raw runtime labels."""

    x, y = origin
    stroke = 3.0 * scale
    tick = 8.0 * scale
    hinge_tick = 16.0 * scale

    # Slide: purple range/handle, semantic endpoint ticks, labels offset from
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
        2.0 * scale,
    )
    draw.line(
        (slide_max, slide_y - tick * 0.5),
        (slide_max, slide_y + tick * 0.5),
        CONCEPT_THEME.axis_color(0),
        2.0 * scale,
    )
    current_x = x + 211.0 * scale
    current_tick = 20.0 * scale
    draw.line(
        (current_x, slide_y - current_tick * 0.5),
        (current_x, slide_y + current_tick * 0.5),
        CONCEPT_THEME.primary_bright,
        4.0 * scale,
    )
    # Opposing drag handles sit off the scale line; the line itself shares the
    # same pointer target so the control remains discoverable.
    arrows = gizmo_ui.joint_slide_arrow_polygons(
        np.asarray((current_x, slide_y)),
        np.asarray((1.0, 0.0)),
        scale,
    )
    for arrow in arrows:
        draw.fringed_concave_fill(
            tuple((float(point[0]), float(point[1])) for point in arrow),
            JOINT_COLOR,
        )
    arrow_points = np.concatenate(
        (*arrows, np.asarray(((slide_min, slide_y), (slide_max, slide_y))))
    )
    arrow_lo = np.min(arrow_points, axis=0)
    arrow_hi = np.max(arrow_points, axis=0)
    if state is not None and _joint_double_click(
        f"##{item_id}-slide-arrow",
        (arrow_lo[0] - 4.0 * scale, arrow_lo[1] - 4.0 * scale),
        (arrow_hi[0] + 4.0 * scale, arrow_hi[1] + 4.0 * scale),
    ):
        _open_joint_value(state, "slide_joint", 0.0, "m")
    _draw_label_button(
        draw,
        "##joint-min-slide",
        (x + 18.0 * scale, y + 82.0 * scale),
        "MIN",
        "−0.340",
        "m",
        CONCEPT_THEME.axis_color(2),
        scale,
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
    )

    # Hinge: a single clean purple arc. Endpoint ticks are radial and use the
    # same MIN/MAX semantic colors as the adjacent label dots.
    center = (x + 548.0 * scale, y + 140.0 * scale)
    radius = 72.0 * scale
    arc = tuple((center[0] + ux * radius, center[1] + uy * radius) for ux, uy in JOINT_HINGE_ARC)
    draw.polyline(arc, JOINT_COLOR, stroke)
    for index, color in enumerate((CONCEPT_THEME.axis_color(0), CONCEPT_THEME.axis_color(2))):
        ux, uy = JOINT_HINGE_ARC[0 if index == 0 else -1]
        point = (center[0] + ux * radius, center[1] + uy * radius)
        draw.line(
            (point[0] - ux * hinge_tick * 0.35, point[1] - uy * hinge_tick * 0.35),
            (point[0] + ux * hinge_tick * 0.65, point[1] + uy * hinge_tick * 0.65),
            color,
            2.4 * scale,
        )
    current_ux, current_uy = JOINT_HINGE_ARC[len(JOINT_HINGE_ARC) // 2]
    current_point = (
        center[0] + current_ux * radius,
        center[1] + current_uy * radius,
    )
    draw.line(
        (
            current_point[0] - current_ux * current_tick * 0.5,
            current_point[1] - current_uy * current_tick * 0.5,
        ),
        (
            current_point[0] + current_ux * current_tick * 0.5,
            current_point[1] + current_uy * current_tick * 0.5,
        ),
        CONCEPT_THEME.primary_bright,
        4.0 * scale,
    )
    if state is not None and _joint_double_click(
        f"##{item_id}-hinge-ring",
        (center[0] - radius - 12.0 * scale, center[1] - radius - 12.0 * scale),
        (center[0] + radius + 12.0 * scale, center[1] + radius + 12.0 * scale),
        points=arc,
        tolerance=10.0 * scale,
    ):
        _open_joint_value(state, "hinge_joint", 0.0, "°")
    _draw_label_button(
        draw,
        "##joint-max-hinge",
        (x + 500.0 * scale, y + 18.0 * scale),
        "MAX",
        "+120.0",
        "°",
        CONCEPT_THEME.axis_color(0),
        scale,
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
    )


def _draw_joint_rotation_feedback(draw: ImguiDraw2D, center, radius: float, scale: float) -> None:
    """Show the accepted hinge drag + Shift colors without legacy amber/yellow."""

    arc = tuple((center[0] + ux * radius, center[1] + uy * radius) for ux, uy in JOINT_HINGE_ARC)
    draw.polyline(arc, JOINT_COLOR, 3.0 * scale)

    start_index = 8
    end_index = 17
    sweep = arc[start_index : end_index + 1]
    draw.triangle_fan_fill(
        (center, *sweep),
        (*CONCEPT_THEME.primary_dim[:3], 0.24),
    )
    draw.polyline(sweep, CONCEPT_THEME.primary_bright, 3.0 * scale)

    for index in range(0, len(JOINT_HINGE_ARC), 2):
        ux, uy = JOINT_HINGE_ARC[index]
        point = arc[index]
        length = (9.0 if index % 6 == 0 else 6.0) * scale
        draw.line(
            point,
            (point[0] + ux * length, point[1] + uy * length),
            CONCEPT_THEME.text_disabled,
            1.1 * scale,
        )

    ux, uy = JOINT_HINGE_ARC[end_index]
    point = arc[end_index]
    draw.line(
        (point[0] - ux * 2.0 * scale, point[1] - uy * 2.0 * scale),
        (point[0] + ux * 12.0 * scale, point[1] + uy * 12.0 * scale),
        CONCEPT_THEME.primary_bright,
        2.6 * scale,
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
    if kind == "camera":
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
) -> None:
    """Render the same flat gizmo geometry and color states as the application."""

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
    hot = (
        gizmo_geometry.GizmoHandle.X if mode == "translate" else gizmo_geometry.GizmoHandle.ROTATE_X
    )
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
    # Pin the specimen to the concept palette so it remains a stable reference.
    active_color = np.asarray(CONCEPT_THEME.primary_bright, np.float32)
    original_colors = (
        gizmo_ui.AXIS_COLORS,
        gizmo_ui.HOVER_COLOR,
        gizmo_ui.ACTIVE_HANDLE_COLOR,
        gizmo_ui.ACTIVE_COLOR,
        gizmo_ui.GUIDE_CORE_COLOR,
    )
    try:
        gizmo_ui.AXIS_COLORS = np.asarray(
            tuple(CONCEPT_THEME.axis_color(axis) for axis in range(3)), np.float32
        )
        gizmo_ui.HOVER_COLOR = active_color
        gizmo_ui.ACTIVE_HANDLE_COLOR = active_color
        gizmo_ui.ACTIVE_COLOR = np.asarray(CONCEPT_THEME.primary_dim, np.float32)
        gizmo_ui.GUIDE_CORE_COLOR = np.asarray((0.98, 0.98, 0.99, 1.0), np.float32)
        specimen.draw_overlay(camera, rect, draw, style_scale=scale)
    finally:
        (
            gizmo_ui.AXIS_COLORS,
            gizmo_ui.HOVER_COLOR,
            gizmo_ui.ACTIVE_HANDLE_COLOR,
            gizmo_ui.ACTIVE_COLOR,
            gizmo_ui.GUIDE_CORE_COLOR,
        ) = original_colors
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


def _table_next_control_row() -> None:
    """Start a property row with one explicit framed-control height."""

    style = imgui.get_style()
    row_height = imgui.get_frame_height() + style.cell_padding.y * 2.0
    imgui.table_next_row(imgui.TableRowFlags_.none.value, row_height)


def _table_text(value: str, *, disabled: bool = False) -> None:
    """Center text on the same visual axis as a framed control in its row."""

    cursor = imgui.get_cursor_screen_pos()
    text_height = imgui.calc_text_size(value).y
    offset_y = max(0.0, (imgui.get_frame_height() - text_height) * 0.5)
    imgui.set_cursor_screen_pos(imgui.ImVec2(cursor.x, cursor.y + offset_y))
    if disabled:
        imgui.text_disabled(value)
    else:
        imgui.text(value)


def _property_label(label: str) -> None:
    _table_next_control_row()
    imgui.table_next_column()
    width = imgui.calc_text_size(label).x
    available = imgui.get_content_region_avail().x
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, available - width))
    _table_text(label, disabled=True)
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)


def _probe_checkbox(label: str, value: bool) -> tuple[bool, bool]:
    """Draw the accepted neutral checkbox instead of ImGui's blue checked fill."""

    style = imgui.get_style()
    visible_label = label.split("##", 1)[0]
    box_size = float(imgui.get_frame_height())
    label_width = imgui.calc_text_size(visible_label).x if visible_label else 0.0
    total_width = box_size
    if visible_label:
        total_width += float(style.item_inner_spacing.x) + float(label_width)

    clicked = imgui.invisible_button(label, imgui.ImVec2(total_width, box_size))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    if clicked:
        value = not value

    lo = imgui.get_item_rect_min()
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    opacity = float(style.alpha)

    def faded(color):
        return (*color[:3], color[3] * opacity)

    background = (
        CONCEPT_THEME.bg_frame_active
        if value or active
        else CONCEPT_THEME.bg_frame_hovered
        if hovered
        else CONCEPT_THEME.bg_frame
    )
    draw.rect_filled(
        (lo.x, lo.y),
        (lo.x + box_size, lo.y + box_size),
        faded(background),
        rounding=min(float(style.frame_rounding), box_size * 0.25),
    )
    draw.rect(
        (lo.x, lo.y),
        (lo.x + box_size, lo.y + box_size),
        faded(CONCEPT_THEME.border),
        1.0,
        rounding=min(float(style.frame_rounding), box_size * 0.25),
    )
    if value:
        check = (
            (lo.x + box_size * 0.22, lo.y + box_size * 0.53),
            (lo.x + box_size * 0.43, lo.y + box_size * 0.73),
            (lo.x + box_size * 0.79, lo.y + box_size * 0.29),
        )
        draw.polyline(
            check,
            faded(CONCEPT_THEME.primary_bright),
            max(1.5, box_size * 0.12),
        )
    if visible_label:
        text_height = imgui.calc_text_size(visible_label).y
        text_color = CONCEPT_THEME.text_disabled if opacity < 0.999 else CONCEPT_THEME.text
        draw.text(
            (
                lo.x + box_size + float(style.item_inner_spacing.x),
                lo.y + (box_size - text_height) * 0.5,
            ),
            faded(text_color),
            visible_label,
        )
    return clicked, value


def _draw_segmented(
    item_id: str,
    labels: tuple[str, ...],
    selected: int,
    *,
    width: float = 82.0,
) -> int:
    result = selected
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(1.0, 0.0))
    for index, label in enumerate(labels):
        if index:
            imgui.same_line()
        if index == selected:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*CONCEPT_THEME.bg_frame_active))
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*CONCEPT_THEME.primary_bright))
        if imgui.button(f"{label}##{item_id}-{index}", imgui.ImVec2(width, 0.0)):
            result = index
        if index == selected:
            imgui.pop_style_color(2)
    imgui.pop_style_var()
    return result


def _settings_properties(table_id: str) -> bool:
    flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
    if not imgui.begin_table(table_id, 2, flags):
        return False
    imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
    imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value, 0.64)
    return True


def _draw_settings_general(state: ProbeState) -> None:
    if not _settings_properties("##probe-settings-general"):
        return
    _property_label("Language")
    _, state.language = imgui.combo("##probe-language", state.language, ("简体中文", "English"))
    _property_label("UI font")
    _table_text("JetBrains Mono", disabled=True)
    _property_label("CJK font")
    _table_text("PingFang SC", disabled=True)
    imgui.end_table()


def _draw_settings_interaction(state: ProbeState) -> None:
    imgui.text_disabled("Gizmo")
    imgui.separator()
    if _settings_properties("##probe-settings-gizmo"):
        _property_label("Style")
        state.gizmo_style = _draw_segmented("gizmo", ("2D", "3D"), state.gizmo_style)
        _property_label("Orientation")
        state.frame = _draw_segmented("frame", ("Body", "World"), state.frame)
        _property_label("Overlay size")
        _, state.viewport_overlay_scale = imgui.drag_float(
            "##probe-overlay-size", state.viewport_overlay_scale, 0.02, 0.85, 1.6, "%.2fx"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Input")
    imgui.separator()
    if _settings_properties("##probe-settings-input"):
        _property_label("Keep mode/unit")
        _, state.remember_input = _probe_checkbox("##probe-remember", state.remember_input)
        imgui.set_item_tooltip("Reuse relative/absolute mode and angle unit")
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Snap · Shift")
    imgui.separator()
    if _settings_properties("##probe-settings-snap"):
        _property_label("Position")
        _, state.position_snap = imgui.drag_float(
            "##probe-position-snap", state.position_snap, 0.01, 0.0, 10.0, "%.3f m"
        )
        _property_label("Rotation")
        _, state.rotation_snap = imgui.drag_float(
            "##probe-rotation-snap", state.rotation_snap, 0.5, 0.0, 180.0, "%.1f deg"
        )
        _property_label("Tick scale")
        _, state.tick_scale = imgui.drag_float(
            "##probe-tick-scale", state.tick_scale, 0.05, 0.25, 4.0, "%.2fx"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("View")
    imgui.separator()
    if _settings_properties("##probe-settings-view"):
        _property_label("Padding")
        _, state.selection_padding = imgui.drag_float(
            "##probe-selection-padding", state.selection_padding, 0.05, 1.0, 3.0, "%.2fx"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Perturb")
    imgui.separator()
    if _settings_properties("##probe-settings-perturb"):
        _property_label("Corner radius")
        _, state.corner_radius = imgui.drag_float(
            "##probe-corner-radius", state.corner_radius, 0.5, 0.0, 16.0, "%.1f px"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Helpers")
    imgui.separator()
    if _settings_properties("##probe-settings-helpers"):
        _property_label("Entities")
        _, state.scene_icons = _probe_checkbox("##probe-scene-icons", state.scene_icons)
        _property_label("Volumes")
        imgui.begin_disabled(not state.scene_icons)
        _, state.influence_volumes = _probe_checkbox(
            "##probe-influence-volumes", state.influence_volumes
        )
        imgui.end_disabled()
        imgui.end_table()


def _draw_settings_rendering(state: ProbeState) -> None:
    imgui.text("Forge")
    imgui.separator()
    if _settings_properties("##probe-settings-rendering"):
        for label, attribute in (
            ("outline", "outline_enabled"),
            ("tonemap", "tonemap_enabled"),
            ("msaa", "msaa_enabled"),
        ):
            _property_label(label)
            _, value = _probe_checkbox(f"##probe-{label}", bool(getattr(state, attribute)))
            setattr(state, attribute, value)
        imgui.end_table()
    imgui.spacing()
    if imgui.collapsing_header("Debug") and _settings_properties("##probe-settings-debug"):
        _property_label("Debug view")
        _, state.debug_view = imgui.combo(
            "##probe-debug-view",
            state.debug_view,
            ("shaded", "albedo", "normal", "depth", "segment", "idcolor", "overdraw", "wireframe"),
        )
        _property_label("Labels")
        _, state.debug_labels = imgui.combo(
            "##probe-debug-labels",
            state.debug_labels,
            (
                "none",
                "body",
                "joint",
                "geom",
                "site",
                "camera",
                "light",
                "tendon",
                "actuator",
                "constraint",
                "flex",
                "contact point",
                "contact force",
                "selection",
            ),
        )
        _property_label("Frames")
        _, state.debug_frames = imgui.combo(
            "##probe-debug-frames",
            state.debug_frames,
            ("none", "body", "geom", "site", "camera", "light", "contact", "world"),
        )
        imgui.end_table()
    imgui.separator()
    imgui.text_disabled("Backend  Forge / OpenGL")
    imgui.text_disabled("Device  system renderer")


def _draw_checkbox_grid(
    table_id: str,
    labels: tuple[str, ...],
    values: list[bool],
    *,
    columns: int = 3,
) -> None:
    flags = _flags(imgui.TableFlags_.sizing_stretch_same)
    if not imgui.begin_table(table_id, columns, flags):
        return
    for index, label in enumerate(labels):
        imgui.table_next_column()
        changed, value = _probe_checkbox(f"{label}##{table_id}-{index}", values[index])
        if changed:
            values[index] = value
    imgui.end_table()


def _draw_settings_mujoco(state: ProbeState) -> None:
    categories = ("geom", "site", "joint", "tendon", "actuator", "flex", "skin")
    if imgui.collapsing_header("Visual groups", imgui.TreeNodeFlags_.default_open.value):
        flags = _flags(imgui.TableFlags_.sizing_stretch_same)
        if imgui.begin_table("##probe-visual-groups", 7, flags):
            imgui.table_setup_column("category", imgui.TableColumnFlags_.width_fixed.value, 92.0)
            for group in range(6):
                imgui.table_setup_column(str(group))
            imgui.table_headers_row()
            for row, category in enumerate(categories):
                _table_next_control_row()
                imgui.table_next_column()
                _table_text(category)
                for group in range(6):
                    imgui.table_next_column()
                    index = row * 6 + group
                    changed, value = _probe_checkbox(
                        f"##probe-group-{category}-{group}", state.mujoco_groups[index]
                    )
                    if changed:
                        state.mujoco_groups[index] = value
            imgui.end_table()
    if imgui.collapsing_header("mjtRndFlag", imgui.TreeNodeFlags_.default_open.value):
        _draw_checkbox_grid(
            "##probe-rnd-flags",
            ("shadow", "wireframe", "reflection", "additive", "skybox", "fog", "haze", "cull face"),
            state.render_flags,
            columns=4,
        )
    if imgui.collapsing_header("mjtVisFlag", imgui.TreeNodeFlags_.default_open.value):
        _draw_checkbox_grid(
            "##probe-vis-flags",
            (
                "convexhull",
                "texture",
                "joint",
                "actuator",
                "activation",
                "camera",
                "light",
                "rangefinder",
                "constraint",
                "static",
                "skin",
                "flexface",
                "flexskin",
                "flexvert",
                "flexedge",
                "contactpoint",
                "contactforce",
                "contactsplit",
                "island",
                "autoconnect",
                "tendon",
                "transparent",
                "com",
                "inertia",
                "sclinertia",
                "bodybvh",
                "meshbvh",
            ),
            state.visual_flags,
        )


def _draw_settings(size, state: ProbeState) -> None:
    if not imgui.begin_child("Settings###ProbeSettings", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Settings")
    imgui.separator()
    available = imgui.get_content_region_avail()
    nav_width = min(176.0, max(132.0, available.x * 0.22))
    if imgui.begin_table("##probe-settings-layout", 2, imgui.TableFlags_.borders_inner_v.value):
        imgui.table_setup_column("categories", imgui.TableColumnFlags_.width_fixed.value, nav_width)
        imgui.table_setup_column("page", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_next_row()
        imgui.table_next_column()
        categories = ("General", "Interaction", "Rendering", "MuJoCo Visuals")
        for index, label in enumerate(categories):
            clicked, _ = imgui.selectable(
                f"{label}##probe-settings-category-{index}", state.settings_page == index
            )
            if clicked:
                state.settings_page = index

        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)
        if state.settings_filter:
            imgui.set_next_item_allow_overlap()
        _, state.settings_filter = imgui.input_text(
            "##probe-settings-filter", state.settings_filter
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        cursor_after_input = imgui.get_cursor_screen_pos()
        icon_color = imgui.color_convert_float4_to_u32(
            imgui.get_style_color_vec4(imgui.Col_.text_disabled)
        )
        radius = max(2.5, (hi.y - lo.y) * 0.16)
        icon_step = max(radius * 3.6, hi.y - lo.y)
        center = imgui.ImVec2(hi.x - radius * 2.7, (lo.y + hi.y) * 0.5 - radius * 0.2)
        draw_list = imgui.get_window_draw_list()
        draw_list.add_circle(center, radius, icon_color, 12, 1.2)
        draw_list.add_line(
            imgui.ImVec2(center.x + radius * 0.70, center.y + radius * 0.70),
            imgui.ImVec2(center.x + radius * 1.55, center.y + radius * 1.55),
            icon_color,
            1.2,
        )
        search_lo = imgui.ImVec2(center.x - icon_step * 0.5, lo.y)
        search_hi = imgui.ImVec2(center.x + icon_step * 0.5, hi.y)
        if imgui.is_mouse_hovering_rect(search_lo, search_hi):
            imgui.set_tooltip("Search settings")
        if state.settings_filter:
            clear_center = imgui.ImVec2(center.x - icon_step, (lo.y + hi.y) * 0.5)
            clear_lo = imgui.ImVec2(clear_center.x - icon_step * 0.5, lo.y)
            clear_hi = imgui.ImVec2(clear_center.x + icon_step * 0.5, hi.y)
            imgui.set_cursor_screen_pos(clear_lo)
            clear_clicked = imgui.invisible_button(
                "##probe-clear-settings-search",
                imgui.ImVec2(clear_hi.x - clear_lo.x, clear_hi.y - clear_lo.y),
            )
            clear_hovered = imgui.is_item_hovered()
            imgui.set_cursor_screen_pos(cursor_after_input)
            clear_color = (
                imgui.color_convert_float4_to_u32(imgui.get_style_color_vec4(imgui.Col_.text))
                if clear_hovered
                else icon_color
            )
            arm = radius * 0.88
            draw_list.add_line(
                imgui.ImVec2(clear_center.x - arm, clear_center.y - arm),
                imgui.ImVec2(clear_center.x + arm, clear_center.y + arm),
                clear_color,
                1.4,
            )
            draw_list.add_line(
                imgui.ImVec2(clear_center.x + arm, clear_center.y - arm),
                imgui.ImVec2(clear_center.x - arm, clear_center.y + arm),
                clear_color,
                1.4,
            )
            if clear_hovered:
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
                imgui.set_tooltip("Clear search")
            if clear_clicked:
                state.settings_filter = ""
        imgui.separator()
        if state.settings_page == 0:
            _draw_settings_general(state)
        elif state.settings_page == 1:
            _draw_settings_interaction(state)
        elif state.settings_page == 2:
            _draw_settings_rendering(state)
        else:
            _draw_settings_mujoco(state)
        imgui.end_table()
    imgui.end_child()


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


def _draw_viewport(size, scale: float, state: ProbeState) -> tuple[float, float, float, float]:
    child_flags = _flags(imgui.ChildFlags_.borders)
    window_flags = _flags(imgui.WindowFlags_.no_scrollbar, imgui.WindowFlags_.no_scroll_with_mouse)
    if not imgui.begin_child("Viewport###ProbeViewport", size, child_flags, window_flags):
        imgui.end_child()
        return (0.0, 0.0, 0.0, 0.0)

    lo = imgui.get_window_pos()
    window_size = imgui.get_window_size()
    x0, y0 = float(lo.x), float(lo.y)
    x1, y1 = x0 + float(window_size.x), y0 + float(window_size.y)
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    cell = 28.0 * scale
    row = 0
    y = y0
    while y < y1:
        column = 0
        x = x0
        while x < x1:
            draw.rect_filled(
                (x, y),
                (min(x + cell, x1), min(y + cell, y1)),
                VIEW_A if (row + column) % 2 == 0 else VIEW_B,
            )
            x += cell
            column += 1
        y += cell
        row += 1

    draw.text((x0 + 14 * scale, y0 + 11 * scale), CONCEPT_THEME.text_disabled, "joint_types.xml")
    if state.show_construction_notes:
        state_radius = state.overlay_icon_radius + state.overlay_radial_step
        shell_radius = state_radius + state.overlay_radial_step
        draw.text(
            (x0 + 14 * scale, y0 + 42 * scale),
            CONCEPT_THEME.text_disabled,
            f"Geometry  icon Ø{state.overlay_icon_radius * 2} amber · "
            f"state Ø{state_radius * 2} green · shell {shell_radius * 2} · "
            f"centers {state.overlay_center_step}",
        )
    shell_radius = state.overlay_icon_radius + state.overlay_radial_step * 2
    if state.show_playback:
        playback_width = (shell_radius * 2.0 + state.overlay_center_step * 2.0) * scale
        _draw_playback(
            draw,
            (x0 + (window_size.x - playback_width) * 0.5, y0 + 18 * scale),
            scale,
            state,
        )
    if state.show_tool_column:
        _draw_tool_column(draw, (x0 + 18 * scale, y0 + 92 * scale), scale, state)
    if state.show_joint_gizmos:
        _draw_joint_gizmo(
            draw,
            (x0 + 250 * scale, y0 + 250 * scale),
            scale,
            state,
            item_id="workspace-joint",
        )
    if state.show_context_hints:
        hint_width = _hint_bar_width(draw, scale, state)
        hint_height = (state.hint_control_height + state.hint_padding_y * 2.0) * scale
        _draw_hint_bar(
            draw,
            (
                x0 + (window_size.x - hint_width) * 0.5,
                y1 - hint_height - 16.0 * scale,
            ),
            scale,
            state,
        )

    imgui.end_child()
    return (x0, y0, x1, y1)


def _diamond(draw: ImguiDraw2D, center, size: float, color) -> None:
    x, y = center
    draw.convex_fill(((x, y - size), (x + size, y), (x, y + size), (x - size, y)), color)


def _draw_transport_button(
    draw: ImguiDraw2D, item_id: str, position, kind: str, scale: float
) -> None:
    size = 28.0 * scale
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    imgui.invisible_button(item_id, imgui.ImVec2(size, size))
    hovered = imgui.is_item_hovered()
    background = CONCEPT_THEME.bg_frame_hovered if hovered else CONCEPT_THEME.bg_frame
    foreground = CONCEPT_THEME.primary_bright if hovered else CONCEPT_THEME.text
    x, y = position
    draw.rect_filled((x, y), (x + size, y + size), background, rounding=3 * scale)
    center = (x + size * 0.5, y + size * 0.5)
    if kind == "record":
        draw.circle_filled(center, 5 * scale, CONCEPT_THEME.danger)
    elif kind == "play":
        _draw_play_icon(draw, center, foreground, scale)
    elif kind == "stop":
        _draw_stop_icon(draw, center, foreground, scale)
    elif kind in ("first", "previous"):
        x2, y2 = center
        draw.convex_fill(
            (
                (x2 - 6.0 * scale, y2),
                (x2 + 3.5 * scale, y2 - 6.0 * scale),
                (x2 + 3.5 * scale, y2 + 6.0 * scale),
            ),
            foreground,
        )
        if kind == "first":
            draw.rect_filled(
                (x2 - 7.0 * scale, y2 - 6.0 * scale),
                (x2 - 5.2 * scale, y2 + 6.0 * scale),
                foreground,
            )
    elif kind in ("next", "last"):
        x2, y2 = center
        draw.convex_fill(
            (
                (x2 + 6.0 * scale, y2),
                (x2 - 3.5 * scale, y2 - 6.0 * scale),
                (x2 - 3.5 * scale, y2 + 6.0 * scale),
            ),
            foreground,
        )
        if kind == "last":
            draw.rect_filled(
                (x2 + 5.2 * scale, y2 - 6.0 * scale),
                (x2 + 7.0 * scale, y2 + 6.0 * scale),
                foreground,
            )
    elif kind == "clear":
        x2, y2 = center
        draw.line(
            (x2 - 5.0 * scale, y2 - 5.0 * scale),
            (x2 + 5.0 * scale, y2 + 5.0 * scale),
            foreground,
            1.8 * scale,
        )
        draw.line(
            (x2 + 5.0 * scale, y2 - 5.0 * scale),
            (x2 - 5.0 * scale, y2 + 5.0 * scale),
            foreground,
            1.8 * scale,
        )
    elif kind in ("key-previous", "key-next"):
        x2, y2 = center
        _diamond(
            draw,
            (x2 + (2.5 if kind == "key-previous" else -2.5) * scale, y2),
            4.5 * scale,
            foreground,
        )
        direction = -1.0 if kind == "key-previous" else 1.0
        draw.convex_fill(
            (
                (x2 + direction * 7.0 * scale, y2),
                (x2 + direction * 3.5 * scale, y2 - 3.5 * scale),
                (x2 + direction * 3.5 * scale, y2 + 3.5 * scale),
            ),
            foreground,
        )
    elif kind == "view":
        x2, y2 = center
        draw.rect(
            (x2 - 7.0 * scale, y2 - 5.0 * scale),
            (x2 + 7.0 * scale, y2 + 5.0 * scale),
            foreground,
            1.5 * scale,
            rounding=1.5 * scale,
        )
        draw.circle_filled(center, 2.0 * scale, foreground, segments=16)
    else:
        _diamond(draw, center, 6 * scale, foreground)


def _draw_keyframes(size, scale: float) -> None:
    if not imgui.begin_child("Keyframes###ProbeKeyframes", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Keyframes")
    imgui.separator()
    imgui.text_disabled("Model")
    imgui.same_line()
    imgui.set_next_item_width(260 * scale)
    imgui.combo("##probe-keyframe-model", 0, ("joint_types",))
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    row_one = imgui.get_cursor_screen_pos()
    imgui.button("●  Record New Take", imgui.ImVec2(142.0 * scale, 28.0 * scale))
    transport_x = row_one.x + 152.0 * scale
    for index, kind in enumerate(("first", "previous", "play", "stop", "next", "last", "clear")):
        _draw_transport_button(
            draw,
            f"##probe-keyframes-take-{kind}",
            (transport_x + index * 34 * scale, row_one.y),
            kind,
            scale,
        )
    draw.text(
        (
            imgui.get_window_pos().x + imgui.get_window_size().x - 154.0 * scale,
            row_one.y + 7.0 * scale,
        ),
        CONCEPT_THEME.text_disabled,
        "frame 86 / 240",
    )

    row_two_y = row_one.y + 34.0 * scale
    imgui.set_cursor_screen_pos(imgui.ImVec2(row_one.x, row_two_y))
    imgui.button("◆  Capture Snapshot", imgui.ImVec2(142.0 * scale, 28.0 * scale))
    for index, kind in enumerate(("key-previous", "key-next", "view")):
        _draw_transport_button(
            draw,
            f"##probe-keyframes-snapshot-{kind}",
            (transport_x + index * 34 * scale, row_two_y),
            kind,
            scale,
        )
    draw.text(
        (
            imgui.get_window_pos().x + imgui.get_window_size().x - 154.0 * scale,
            row_two_y + 7.0 * scale,
        ),
        CONCEPT_THEME.text_disabled,
        "1.20 s · 3 snapshots",
    )

    timeline_y = row_two_y + 54 * scale
    left = row_one.x + 128 * scale
    right = imgui.get_window_pos().x + imgui.get_window_size().x - 18 * scale
    draw.text((row_one.x, timeline_y - 5 * scale), CONCEPT_THEME.text, "Model Keyframes")
    draw.line(
        (left, timeline_y + 4 * scale), (right, timeline_y + 4 * scale), CONCEPT_THEME.border, 1.0
    )
    for fraction, color in (
        (0.28, CONCEPT_THEME.text_disabled),
        (0.53, rgb8(225, 183, 101)),
        (0.79, CONCEPT_THEME.text_disabled),
    ):
        _diamond(draw, (left + (right - left) * fraction, timeline_y + 4 * scale), 6 * scale, color)
    playhead_x = left + (right - left) * 0.47
    draw.line(
        (playhead_x, timeline_y - 24 * scale),
        (playhead_x, timeline_y + 60 * scale),
        CONCEPT_THEME.danger,
        2 * scale,
    )
    draw.text((row_one.x, timeline_y + 39 * scale), CONCEPT_THEME.text, "Recorded Take")
    draw.line(
        (left, timeline_y + 48 * scale),
        (right, timeline_y + 48 * scale),
        CONCEPT_THEME.primary_dim,
        2 * scale,
    )
    selected_y = timeline_y + 76.0 * scale
    draw.text((row_one.x, selected_y + 7.0 * scale), CONCEPT_THEME.text_disabled, "Selected")
    draw.rect_filled(
        (left, selected_y),
        (right, selected_y + 28.0 * scale),
        CONCEPT_THEME.bg_frame,
        rounding=3.0 * scale,
    )
    draw.text(
        (left + 10.0 * scale, selected_y + 7.0 * scale),
        CONCEPT_THEME.text,
        "key1  ·  0.90 s",
    )
    imgui.end_child()


def _draw_output(size, state: ProbeState) -> None:
    if not imgui.begin_child("Output###ProbeOutput", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Output")
    imgui.separator()
    available = imgui.get_content_region_avail().x
    imgui.set_next_item_width(max(100.0, available - 155.0))
    _, state.output_filter = imgui.input_text_with_hint(
        "##probe-output-filter", "Filter text or component...", state.output_filter
    )
    imgui.same_line()
    imgui.set_next_item_width(145.0)
    _, state.output_level = imgui.combo(
        "##probe-output-level", state.output_level, ("All levels", "Warnings", "Errors")
    )
    rows = (
        ()
        if state.output_cleared
        else (
            ("09:44:04", "INFO", "pelvis"),
            ("09:44:09", "INFO", "sacrum"),
            ("09:44:13", "WARN", "femur_r limit reached"),
            ("09:44:25", "INFO", "Loaded scene.xml"),
            ("09:44:54", "INFO", "[forge/ui] Loading model /assets/joint_types.xml"),
        )
    )
    imgui.text_disabled(f"{len(rows)} messages · select a row, then right-click or press Ctrl+C")
    imgui.separator()
    if not rows:
        imgui.text_disabled("No messages")
    for index, (timestamp, level, message) in enumerate(rows):
        row_text = f"{timestamp}  [{level}]  {message}"
        clicked, _ = imgui.selectable(
            f"{row_text}##probe-output-{index}",
            state.selected_output == index,
            imgui.SelectableFlags_.none.value,
        )
        if clicked:
            state.selected_output = index
        if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
            state.selected_output = index
        if imgui.begin_popup_context_item(f"##probe-output-context-{index}"):
            copy_message, _ = imgui.menu_item("Copy message", "Ctrl+C", False)
            copy_row, _ = imgui.menu_item("Copy complete row", "", False)
            copy_all, _ = imgui.menu_item("Copy all shown", "", False)
            imgui.separator()
            clear_output, _ = imgui.menu_item("Clear output", "", False)
            if copy_message:
                imgui.set_clipboard_text(message)
            if copy_row:
                imgui.set_clipboard_text(row_text)
            if copy_all:
                imgui.set_clipboard_text(
                    "\n".join(f"{time}  [{severity}]  {text}" for time, severity, text in rows)
                )
            if clear_output:
                state.output_cleared = True
                state.selected_output = -1
            imgui.end_popup()
    io = imgui.get_io()
    copy_shortcut = bool(io.key_ctrl or io.key_super) and imgui.is_key_pressed(imgui.Key.c, False)
    if copy_shortcut and 0 <= state.selected_output < len(rows):
        timestamp, level, message = rows[state.selected_output]
        imgui.set_clipboard_text(f"{timestamp}  [{level}]  {message}")
    imgui.end_child()


def _begin_gallery_panel(title: str, item_id: str, size) -> bool:
    opened = imgui.begin_child(
        f"{title}###{item_id}",
        size,
        imgui.ChildFlags_.borders.value,
    )
    if opened:
        imgui.text(title)
        imgui.separator()
    return opened


def _begin_gallery_properties(item_id: str) -> bool:
    flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
    if not imgui.begin_table(item_id, 2, flags):
        return False
    imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
    imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value, 0.64)
    return True


def _draw_control_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Control", "ProbeControl", size)
    if opened:
        imgui.text("actuators")
        imgui.separator()
        if _begin_gallery_properties("##probe-control-actuators"):
            _property_label("hinge_pos")
            _, state.hinge_ctrl = imgui.slider_float(
                "##probe-hinge-ctrl", state.hinge_ctrl, -1.0, 1.0, "%+.3f"
            )
            _property_label("slide_pos")
            _, state.slide_ctrl = imgui.slider_float(
                "##probe-slide-ctrl", state.slide_ctrl, -1.0, 1.0, "%+.3f"
            )
            imgui.end_table()

        imgui.spacing()
        imgui.text("equality")
        imgui.separator()
        if _begin_gallery_properties("##probe-control-equality"):
            for item_id, name, enabled in (
                ("weld", "eq_weld_0", state.weld_enabled),
                ("connect", "eq_connect_0", state.connect_enabled),
            ):
                _property_label(name)
                _changed, enabled = _probe_checkbox(f"##probe-equality-{item_id}", enabled)
                if item_id == "weld":
                    state.weld_enabled = enabled
                else:
                    state.connect_enabled = enabled
            imgui.end_table()
    imgui.end_child()


def _draw_joints_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Joints", "ProbeJoints", size)
    if opened:
        if _begin_gallery_properties("##probe-direct-joints"):
            _property_label("hinge_limited")
            _, state.hinge_position = imgui.slider_float(
                "##probe-hinge-position", state.hinge_position, -1.2, 1.2, "%+.4f"
            )
            imgui.end_table()

        imgui.spacing()
        imgui.text_disabled("Other joints")
        if _begin_gallery_properties("##probe-all-joints"):
            _property_label("free")
            _table_text("free · 6 dof", disabled=True)
            _property_label("slide")
            _, state.slide_position = imgui.slider_float(
                "##probe-slide-position", state.slide_position, -0.34, 0.34, "%+.4f"
            )
            _property_label("ball")
            _table_text("ball · 3 dof", disabled=True)
            imgui.end_table()
    imgui.end_child()


def _draw_camera_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Camera", "ProbeCamera", size)
    if opened:
        imgui.set_next_item_width(-1.0)
        imgui.combo("##probe-camera", 0, ("source: free", "overview", "tracking"))
        imgui.spacing()
        imgui.text_disabled("presets")
        preset_flags = _flags(
            imgui.TableFlags_.sizing_stretch_same,
            imgui.TableFlags_.no_saved_settings,
            imgui.TableFlags_.no_pad_outer_x,
        )
        if imgui.begin_table("##probe-camera-presets", 4, preset_flags):
            for index, label in enumerate(
                ("front", "back", "left", "right", "top", "bottom", "iso", "frame all")
            ):
                imgui.table_next_column()
                imgui.button(f"{label}##probe-camera-preset-{index}", imgui.ImVec2(-1.0, 0.0))
            imgui.end_table()
        imgui.separator()
        if _begin_gallery_properties("##probe-camera-params"):
            for label, value, lo, hi, fmt in (
                ("yaw", -90.0, -180.0, 180.0, "%.1f deg"),
                ("pitch", -20.0, -89.9, 89.9, "%.1f deg"),
                ("distance", 3.0, 0.05, 200.0, "%.3f m"),
                ("fov_y_deg", 45.0, 10.0, 120.0, "%.1f deg"),
                ("far", 200.0, 1.0, 100000.0, "%.1f m"),
            ):
                _property_label(label)
                imgui.slider_float(f"##probe-camera-{label}", value, lo, hi, fmt)
            _property_label("projection")
            segment_width = max(44.0, (imgui.get_content_region_avail().x - 1.0) * 0.5)
            state.camera_projection = _draw_segmented(
                "camera-projection",
                ("persp", "ortho"),
                state.camera_projection,
                width=segment_width,
            )
            imgui.end_table()
        if imgui.collapsing_header("camera bookmarks"):
            imgui.text_disabled("camera bookmark")
            imgui.input_text("##probe-camera-bookmark", "view-1")
            imgui.button("save")
            imgui.same_line()
            imgui.button("copy")
            imgui.same_line()
            imgui.button("delete")
    imgui.end_child()


def _probe_axis_field(
    label: str,
    item_id: str,
    value: float,
    axis: int,
    fmt: str,
    scale: float,
    *,
    editable: bool = True,
) -> None:
    """Mirror the Inspector's visually continuous axis + value control."""

    color = AXIS_BADGE_COLORS[axis]
    hovered_color = AXIS_BADGE_HOVERED[axis]
    active_color = AXIS_BADGE_ACTIVE[axis]
    axis_width = 18.0 * scale
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*color))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*hovered_color))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(*active_color))
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*AXIS_BADGE_TEXT))
    if not editable:
        imgui.push_style_var(imgui.StyleVar_.disabled_alpha, 1.0)
    imgui.begin_disabled(not editable)
    imgui.button(f"{label}##{item_id}-axis", imgui.ImVec2(axis_width, 0.0))
    imgui.end_disabled()
    if not editable:
        imgui.pop_style_var()
    button_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    button_active = imgui.is_item_active()
    button_lo, button_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    imgui.pop_style_color(4)
    imgui.same_line(0.0, 0.0)
    group_gap = 3.0 * scale if axis < 2 else 0.0
    imgui.set_next_item_width(max(1.0, imgui.get_content_region_avail().x - group_gap))
    imgui.begin_disabled(not editable)
    imgui.drag_float(f"##{item_id}-value", value, 0.01 if editable else 0.0, 0.0, 0.0, fmt)
    imgui.end_disabled()
    field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    field_active = imgui.is_item_active()
    field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    _probe_fill_axis_field_seam(
        button_lo,
        button_hi,
        field_lo,
        field_hi,
        color=(
            active_color
            if button_active and editable
            else hovered_color
            if button_hovered and editable
            else color
        ),
        frame_color=(
            CONCEPT_THEME.bg_frame_active
            if field_active
            else CONCEPT_THEME.bg_frame_hovered
            if field_hovered and editable
            else CONCEPT_THEME.bg_frame
        ),
        editable=editable,
    )


def _probe_fill_axis_field_seam(
    button_lo,
    button_hi,
    field_lo,
    field_hi,
    *,
    color,
    frame_color,
    editable: bool,
) -> None:
    """Cover the two inner rounded corners so the pair reads as one field."""

    rounding = min(
        float(imgui.get_style().frame_rounding),
        max(0.0, (float(button_hi.y) - float(button_lo.y)) * 0.5),
    )
    if rounding <= 0.0:
        return
    style = imgui.get_style()

    def packed(value, opacity: float) -> int:
        background = CONCEPT_THEME.bg_popup
        mixed = tuple(
            background[index] + (value[index] - background[index]) * opacity for index in range(3)
        )
        return imgui.color_convert_float4_to_u32(imgui.ImVec4(*mixed, 1.0))

    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(
        imgui.ImVec2(button_hi.x - rounding, button_lo.y),
        button_hi,
        packed(color, float(style.alpha)),
    )
    field_opacity = float(style.alpha) * (1.0 if editable else float(style.disabled_alpha))
    for y0, y1 in (
        (field_lo.y, field_lo.y + rounding),
        (field_hi.y - rounding, field_hi.y),
    ):
        draw_list.add_rect_filled(
            imgui.ImVec2(field_lo.x, y0),
            imgui.ImVec2(field_lo.x + rounding, y1),
            packed(frame_color, field_opacity),
        )


def _probe_property_vector_row(
    name: str,
    values: tuple[float, float, float],
    fmt: str,
    scale: float,
) -> None:
    """Mirror the Inspector's responsive label/control XYZ property row."""

    _property_label(name)
    compact = imgui.get_content_region_avail().x < 210.0 * scale
    flags = _flags(
        imgui.TableFlags_.sizing_stretch_same,
        imgui.TableFlags_.no_saved_settings,
        imgui.TableFlags_.no_pad_inner_x,
        imgui.TableFlags_.no_pad_outer_x,
    )
    columns = 1 if compact else 3
    if not imgui.begin_table(f"##probe-{name}-property-axes", columns, flags):
        return
    for axis in "xyz"[:columns]:
        imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch.value, 1.0)
    for axis, label in enumerate("XYZ"):
        if compact:
            imgui.table_next_row()
        imgui.table_next_column()
        _probe_axis_field(label, f"probe-{name}-{axis}", values[axis], axis, fmt, scale)
    imgui.end_table()


def _draw_inspector_gallery(size, scale: float) -> None:
    opened = _begin_gallery_panel("Inspector", "ProbeInspector", size)
    if opened:
        style = imgui.get_style()
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(style.item_spacing.x, 0.0),
        )
        transform_open = imgui.collapsing_header(
            "transform", imgui.TreeNodeFlags_.default_open.value
        )
        imgui.pop_style_var()
        if transform_open:
            imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(*CONCEPT_THEME.bg_popup))
            imgui.push_style_var(
                imgui.StyleVar_.window_padding,
                imgui.ImVec2(6.0 * scale, 4.0 * scale),
            )
            child_flags = imgui.ChildFlags_.always_use_window_padding.value
            window_flags = _flags(
                imgui.WindowFlags_.no_scrollbar,
                imgui.WindowFlags_.no_scroll_with_mouse,
            )
            child_visible = imgui.begin_child(
                "##probe-transform-body",
                imgui.ImVec2(0.0, 156.0 * scale),
                child_flags,
                window_flags,
            )
            if child_visible:
                imgui.push_font(None, 12.0 * scale)
                if _begin_gallery_properties("##probe-transform"):
                    _probe_property_vector_row("position", (0.193, 0.047, 0.445), "%.3f", scale)
                    _probe_property_vector_row("rotation", (0.0, -0.0, 0.0), "%.1f", scale)
                    imgui.end_table()
                imgui.pop_font()
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
        imgui.spacing()
        if imgui.collapsing_header(
            "material", imgui.TreeNodeFlags_.default_open.value
        ) and _begin_gallery_properties("##probe-appearance"):
            _property_label("assigned material")
            _table_text("robot_metal")
            _property_label("base color")
            imgui.color_button("##probe-base-color", imgui.ImVec4(0.82, 0.40, 0.33, 1.0))
            _property_label("roughness")
            imgui.slider_float("##probe-roughness", 0.42, 0.0, 1.0, "%.2f")
            imgui.end_table()
    imgui.end_child()


def _draw_hierarchy_gallery(size, state: ProbeState, scale: float) -> None:
    opened = _begin_gallery_panel("Hierarchy", "ProbeHierarchy", size)
    if opened:
        imgui.set_next_item_width(-1.0)
        _changed, state.hierarchy_filter = imgui.input_text_with_hint(
            "##probe-hierarchy-filter", "Filter entities...", state.hierarchy_filter
        )
        chip_flags = _flags(
            imgui.WindowFlags_.horizontal_scrollbar, imgui.WindowFlags_.no_scroll_with_mouse
        )
        chip_height = imgui.get_frame_height() + imgui.get_style().scrollbar_size + 8.0 * scale
        if imgui.begin_child(
            "##probe-hierarchy-chips",
            imgui.ImVec2(-1.0, chip_height),
            imgui.ChildFlags_.none.value,
            chip_flags,
        ):
            for index, label in enumerate(
                ("All", "link", "geom", "joint", "site", "camera", "light", "robot", "flex")
            ):
                if index:
                    imgui.same_line()
                selected = state.hierarchy_kind == index
                if selected:
                    imgui.push_style_color(
                        imgui.Col_.button, imgui.ImVec4(*CONCEPT_THEME.bg_frame_active)
                    )
                    imgui.push_style_color(
                        imgui.Col_.text, imgui.ImVec4(*CONCEPT_THEME.primary_bright)
                    )
                if imgui.button(
                    f"{label}##probe-hierarchy-chip-{index}",
                    imgui.ImVec2(max(42.0, imgui.calc_text_size(label).x + 18.0), 0.0),
                ):
                    state.hierarchy_kind = index
                if selected:
                    imgui.pop_style_color(2)
        imgui.end_child()
        imgui.separator()
        rows = (
            (0, "▾", "world", "world", 0),
            (1, "▾", "a_sphere", "link", 1),
            (2, "", "sphere_geom", "geom", 2),
            (1, "▸", "hinge_body", "link", 3),
            (2, "", "hinge_joint", "joint", 4),
            (1, "▸", "overview", "camera", 5),
        )
        row_draw = ImguiDraw2D(imgui.get_window_draw_list())
        row_flags = _flags(
            imgui.TableFlags_.sizing_stretch_prop,
            imgui.TableFlags_.no_saved_settings,
            imgui.TableFlags_.pad_outer_x,
        )
        if imgui.begin_table("##probe-hierarchy-rows", 3, row_flags):
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch.value, 1.0)
            imgui.table_setup_column(
                "Type", imgui.TableColumnFlags_.width_fixed.value, 74.0 * scale
            )
            imgui.table_setup_column(
                "Show", imgui.TableColumnFlags_.width_fixed.value, 42.0 * scale
            )
            for depth, disclosure, name, kind, index in rows:
                row_height = 31.0 * scale
                item_height = 25.0 * scale
                imgui.table_next_row(imgui.TableRowFlags_.none.value, row_height)
                imgui.table_next_column()
                clicked, _ = imgui.selectable(
                    f"##probe-hierarchy-{index}",
                    state.hierarchy_selection == index,
                    _flags(
                        imgui.SelectableFlags_.span_all_columns,
                        imgui.SelectableFlags_.allow_overlap,
                    ),
                    imgui.ImVec2(0.0, item_height),
                )
                hovered = imgui.is_item_hovered()
                lo = imgui.get_item_rect_min()
                hi = imgui.get_item_rect_max()
                text_y = lo.y + max(0.0, (hi.y - lo.y - imgui.get_font_size()) * 0.5)
                node_x = lo.x + (6.0 + depth * 22.0) * scale
                if disclosure:
                    center_y = (lo.y + hi.y) * 0.5
                    if disclosure == "▾":
                        triangle = (
                            (node_x, center_y - 3.5 * scale),
                            (node_x + 10.0 * scale, center_y - 3.5 * scale),
                            (node_x + 5.0 * scale, center_y + 4.5 * scale),
                        )
                    else:
                        triangle = (
                            (node_x + 1.0 * scale, center_y - 5.0 * scale),
                            (node_x + 1.0 * scale, center_y + 5.0 * scale),
                            (node_x + 9.0 * scale, center_y),
                        )
                    row_draw.convex_fill(
                        triangle,
                        CONCEPT_THEME.text,
                    )
                row_draw.text(
                    (node_x + 18.0 * scale, text_y),
                    CONCEPT_THEME.text,
                    name,
                )
                if clicked:
                    state.hierarchy_selection = index

                imgui.table_next_column()
                type_x = imgui.get_cursor_screen_pos().x
                imgui.set_cursor_screen_pos(imgui.ImVec2(type_x, text_y))
                imgui.text_disabled(kind)

                imgui.table_next_column()
                cell = imgui.get_cursor_screen_pos()
                button_size = 20.0 * scale
                cell_width = imgui.get_content_region_avail().x
                imgui.set_cursor_screen_pos(
                    imgui.ImVec2(
                        cell.x + max(0.0, (cell_width - button_size) * 0.5),
                        lo.y + max(0.0, (hi.y - lo.y - button_size) * 0.5),
                    )
                )
                if imgui.invisible_button(
                    f"##probe-hierarchy-visible-{index}",
                    imgui.ImVec2(button_size, button_size),
                ):
                    state.hierarchy_visibility[index] = not state.hierarchy_visibility[index]
                visible_lo = imgui.get_item_rect_min()
                visible_hi = imgui.get_item_rect_max()
                center = (
                    (visible_lo.x + visible_hi.x) * 0.5,
                    (visible_lo.y + visible_hi.y) * 0.5,
                )
                radius_x = 6.5 * scale
                radius_y = 3.6 * scale
                color = (
                    CONCEPT_THEME.primary_bright
                    if hovered or state.hierarchy_selection == index
                    else CONCEPT_THEME.primary
                    if state.hierarchy_visibility[index]
                    else CONCEPT_THEME.text_disabled
                )
                if state.hierarchy_visibility[index]:
                    top = tuple(
                        (
                            center[0] - radius_x + radius_x * 2.0 * point / 8.0,
                            center[1] - math.sin(math.pi * point / 8.0) * radius_y,
                        )
                        for point in range(9)
                    )
                    bottom = tuple(
                        (
                            center[0] + radius_x - radius_x * 2.0 * point / 8.0,
                            center[1] + math.sin(math.pi * point / 8.0) * radius_y,
                        )
                        for point in range(9)
                    )
                    row_draw.polyline((*top, *bottom[1:-1]), color, 1.35 * scale, closed=True)
                    row_draw.circle(center, 1.8 * scale, color, 1.2 * scale, segments=16)
                else:
                    lid = tuple(
                        (
                            center[0] - radius_x + radius_x * 2.0 * point / 8.0,
                            center[1] + math.sin(math.pi * point / 8.0) * radius_y * 0.72,
                        )
                        for point in range(9)
                    )
                    row_draw.polyline(lid, color, 1.45 * scale)
                    for offset in (-0.52, 0.0, 0.52):
                        lash_x = center[0] + radius_x * offset
                        lash_y = center[1] + radius_y * 0.72 * math.sqrt(max(0.0, 1.0 - offset**2))
                        row_draw.line(
                            (lash_x, lash_y),
                            (lash_x + offset * 1.6 * scale, lash_y + 2.2 * scale),
                            color,
                            1.15 * scale,
                        )
            imgui.end_table()
    imgui.end_child()


def _draw_assets_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Assets", "ProbeAssets", size)
    if opened:
        imgui.button("Import Mesh...")
        imgui.same_line()
        imgui.button("Import Texture...")
        imgui.separator()
        for index, (name, kind) in enumerate(
            (
                ("robot_body", "mesh"),
                ("floor_albedo", "texture"),
                ("robot_metal", "material"),
                ("terrain", "height field"),
            )
        ):
            clicked, _ = imgui.selectable(
                f"{name}##probe-asset-{index}", state.asset_selection == index
            )
            if clicked:
                state.asset_selection = index
            imgui.same_line()
            width = imgui.calc_text_size(kind).x
            imgui.set_cursor_pos_x(
                max(imgui.get_cursor_pos_x(), imgui.get_window_width() - width - 18.0)
            )
            imgui.text_disabled(kind)
        imgui.separator()
        imgui.text_disabled("4 model-local assets")
    imgui.end_child()


def _draw_stats_gallery(size) -> None:
    opened = _begin_gallery_panel("Stats", "ProbeStats", size)
    if opened:
        if _begin_gallery_properties("##probe-stats"):
            for label, value in (
                ("FPS", "119.8"),
                ("Frame", "8.35 ms"),
                ("Physics", "0.42 ms"),
                ("Render", "3.61 ms"),
                ("Draw calls", "148"),
                ("Triangles", "284,612"),
            ):
                _property_label(label)
                _table_text(value)
            imgui.end_table()
        imgui.spacing()
        imgui.plot_lines(
            "##probe-frame-plot",
            np.asarray((8.4, 8.1, 8.5, 8.2, 8.3, 8.6, 8.2, 8.4, 8.3, 8.35), np.float32),
            graph_size=imgui.ImVec2(-1.0, 72.0),
        )
    imgui.end_child()


def _draw_sensors_gallery(size) -> None:
    opened = _begin_gallery_panel("Sensors", "ProbeSensors", size)
    if opened:
        imgui.input_text_with_hint("##probe-sensor-filter", "Filter sensors...", "")
        imgui.separator()
        if imgui.begin_table("##probe-sensors", 3, imgui.TableFlags_.row_bg.value):
            imgui.table_setup_column("Sensor", imgui.TableColumnFlags_.width_stretch.value, 0.5)
            imgui.table_setup_column("Value", imgui.TableColumnFlags_.width_stretch.value, 0.3)
            imgui.table_setup_column("Unit", imgui.TableColumnFlags_.width_stretch.value, 0.2)
            imgui.table_headers_row()
            for name, value, unit in (
                ("imu_accel", "0.02  0.01  9.81", "m/s²"),
                ("hinge_pos", "+0.350", "rad"),
                ("foot_force", "128.4", "N"),
                ("range_front", "2.41", "m"),
            ):
                imgui.table_next_row()
                for cell in (name, value, unit):
                    imgui.table_next_column()
                    imgui.text(cell)
            imgui.end_table()
    imgui.end_child()


def _draw_panel_gallery(available, state: ProbeState, scale: float) -> None:
    spacing = imgui.get_style().item_spacing
    row_height = (available.y - spacing.y) * 0.5
    top_width = (available.x - spacing.x * 3.0) / 4.0
    top_size = imgui.ImVec2(top_width, row_height)
    for index, draw_panel in enumerate(
        (
            lambda: _draw_control_gallery(top_size, state),
            lambda: _draw_joints_gallery(top_size, state),
            lambda: _draw_camera_gallery(top_size, state),
            lambda: _draw_inspector_gallery(top_size, scale),
        )
    ):
        draw_panel()
        if index != 3:
            imgui.same_line()

    # M14 deliberately receives the widest lower slot. Its fixed type/visibility
    # metadata must not steal width from entity names as it did in the old 320 px panel.
    lower_widths = (
        available.x * 0.28,
        available.x * 0.24,
        available.x * 0.18,
    )
    final_width = available.x - sum(lower_widths) - spacing.x * 3.0
    lower_sizes = tuple(imgui.ImVec2(width, row_height) for width in (*lower_widths, final_width))
    for index, draw_panel in enumerate(
        (
            lambda: _draw_hierarchy_gallery(lower_sizes[0], state, scale),
            lambda: _draw_assets_gallery(lower_sizes[1], state),
            lambda: _draw_stats_gallery(lower_sizes[2]),
            lambda: _draw_sensors_gallery(lower_sizes[3]),
        )
    ):
        draw_panel()
        if index != 3:
            imgui.same_line()


def _draw_status_strip(
    draw: ImguiDraw2D,
    origin,
    width: float,
    height: float,
    scale: float,
    *,
    selected: str,
    running: bool,
    fps: str,
) -> None:
    """Draw the same persistent status surface used by the application."""

    rate = float(fps)
    draw_status(
        draw,
        origin,
        width,
        height,
        CONCEPT_THEME,
        scale,
        selected=selected,
        has_selection=selected != "no selection",
        state="running" if running else "paused",
        sim_time=1.204,
        step=1204,
        metric_mode="time",
        backend="OpenGL",
        dt=0.002,
        fps=rate,
        tool_hints=default_tool_hints(
            "ready" if selected != "no selection" else "camera",
            DEFAULT_INPUT_BINDINGS,
        ),
    )


def _draw_shell_settings_tab(available, scale: float, state: ProbeState) -> None:
    spacing = imgui.get_style().item_spacing
    shell_width = max(360.0 * scale, available.x * 0.36)
    shell_size = imgui.ImVec2(shell_width, available.y)
    if imgui.begin_child("Shell###ProbeShell", shell_size, imgui.ChildFlags_.borders.value):
        imgui.text("Application shell · M1 / M4 / M7")
        imgui.separator()
        imgui.text("File   Edit   Entity   View   Window   Help")
        imgui.same_line()
        name = "showcase.xml  ●"
        name_width = imgui.calc_text_size(name).x
        imgui.set_cursor_pos_x(
            max(imgui.get_cursor_pos_x(), imgui.get_window_width() - name_width - 16.0)
        )
        imgui.text_disabled(name)

        imgui.spacing()
        imgui.text("Simulation state")
        imgui.separator()
        selected = 1 if state.sim_running else 0
        selected = _draw_segmented("simulation-state", ("Paused", "Running"), selected, width=92.0)
        state.sim_running = selected == 1

        imgui.spacing()
        imgui.text("Edit gate")
        imgui.separator()
        if _settings_properties("##probe-edit-gate"):
            _property_label("Joint position")
            imgui.begin_disabled(state.sim_running)
            imgui.slider_float("##probe-gated-joint", state.hinge_position, -1.2, 1.2, "%+.3f")
            imgui.end_disabled()
            _property_label("Actuator ctrl")
            imgui.slider_float("##probe-live-actuator", state.hinge_ctrl, -1.0, 1.0, "%+.3f")
            imgui.end_table()
        if state.sim_running:
            imgui.text_colored(imgui.ImVec4(*CONCEPT_THEME.warning), "Pause to edit (Space)")

        imgui.spacing()
        imgui.text("Dock layout")
        imgui.separator()
        if _settings_properties("##probe-dock-layout"):
            for label, value in (
                ("Panels", "Dockable / floating"),
                ("Persist", "imgui.ini"),
                ("Default", "Hierarchy 22% · Viewport 48% · Right 30%"),
            ):
                _property_label(label)
                _table_text(value, disabled=True)
            imgui.end_table()
        imgui.button("Reset Layout")

        imgui.spacing()
        imgui.text("Status bar")
        imgui.separator()
        start = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        height = 34.0 * scale
        draw = ImguiDraw2D(imgui.get_window_draw_list())
        _draw_status_strip(
            draw,
            (start.x, start.y),
            width,
            height,
            scale,
            selected="hinge_body",
            running=state.sim_running,
            fps="119.8",
        )
        imgui.dummy(imgui.ImVec2(width, height))
    imgui.end_child()
    imgui.same_line()
    _draw_settings(imgui.ImVec2(available.x - shell_width - spacing.x, available.y), state)


def _draw_status_tab(available, scale: float) -> None:
    """Show the persistent status surface and its mutually exclusive states."""

    if not imgui.begin_child(
        "Status###ProbeStatus",
        available,
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return
    imgui.text("Status · persistent application surface")
    imgui.separator()
    imgui.text_disabled(
        "Selected object names live here; viewport labels are reserved for values and limits."
    )
    imgui.spacing()
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    samples = (
        ("Paused · selected", "a_sphere", False, "119.8"),
        ("Running · selected", "a_sphere", True, "60.0"),
        ("Paused · no selection", "no selection", False, "119.8"),
    )
    width = min(1180.0 * scale, imgui.get_content_region_avail().x)
    height = 34.0 * scale
    for label, selected, running, fps in samples:
        imgui.text_disabled(label)
        lo = imgui.get_cursor_screen_pos()
        _draw_status_strip(
            draw,
            (lo.x, lo.y),
            width,
            height,
            scale,
            selected=selected,
            running=running,
            fps=fps,
        )
        imgui.dummy(imgui.ImVec2(width, height + 12.0 * scale))
    imgui.spacing()
    imgui.text_disabled(
        "Status remains visible across workspaces; Stats keeps detailed timing history."
    )
    imgui.end_child()


def _draw_plot_aux(size) -> None:
    if not imgui.begin_child("Plot###ProbePlot", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Plot")
    imgui.separator()
    imgui.text_disabled("hinge_pos · rad")
    values = np.asarray(
        (0.00, 0.08, 0.16, 0.27, 0.35, 0.41, 0.38, 0.31, 0.20, 0.11, 0.04),
        np.float32,
    )
    imgui.plot_lines("##probe-plot", values, graph_size=imgui.ImVec2(-1.0, -1.0))
    imgui.end_child()


def _draw_help_aux(size) -> None:
    if not imgui.begin_child("Help###ProbeHelp", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Help")
    imgui.separator()
    if imgui.begin_table("##probe-help", 2, imgui.TableFlags_.row_bg.value):
        imgui.table_setup_column("Input", imgui.TableColumnFlags_.width_fixed.value, 160.0)
        imgui.table_setup_column("Action", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_headers_row()
        for input_name, action in (
            ("Space", "Play / Pause"),
            ("Shift", "Snap"),
            ("T", "World / Body"),
            ("Double-click", "Type value"),
            ("Ctrl + drag", "Push / Twist"),
        ):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(input_name)
            imgui.table_next_column()
            imgui.text(action)
        imgui.end_table()
    imgui.end_child()


def _draw_info_aux(size) -> None:
    if not imgui.begin_child("Info###ProbeInfo", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Info")
    imgui.separator()
    if _settings_properties("##probe-info"):
        for label, value in (
            ("Viewer", "forge-viewer"),
            ("Backend", "Forge / OpenGL"),
            ("Scene source", "MuJoCo"),
            ("Document", "joint_types.xml"),
        ):
            _property_label(label)
            _table_text(value)
        imgui.end_table()
    imgui.end_child()


def _draw_workspaces_tab(available, scale: float, state: ProbeState) -> None:
    keyframe_height = min(320.0 * scale, max(240.0 * scale, available.y * 0.34))
    _draw_keyframes(imgui.ImVec2(available.x, keyframe_height), scale)
    imgui.text_disabled("M15 · full-width bottom dock; transport uses compact icon groups")
    active = state.aux_tab
    if imgui.begin_tab_bar("##probe-aux-tabs"):
        for label in ("Output", "Plot", "Help", "Info"):
            opened, _ = imgui.begin_tab_item(label)
            if opened:
                active = label
                imgui.end_tab_item()
        imgui.end_tab_bar()
    state.aux_tab = active
    remaining = imgui.get_content_region_avail()
    if active == "Output":
        _draw_output(remaining, state)
    elif active == "Plot":
        _draw_plot_aux(remaining)
    elif active == "Help":
        _draw_help_aux(remaining)
    else:
        _draw_info_aux(remaining)


def _dimension_line(
    draw: ImguiDraw2D,
    start,
    end,
    label: str,
    *,
    vertical: bool = False,
) -> None:
    color = (*CONCEPT_THEME.text_disabled[:3], 0.9)
    draw.line(start, end, color, 1.0)
    if vertical:
        draw.line((start[0] - 5.0, start[1]), (start[0] + 5.0, start[1]), color, 1.0)
        draw.line((end[0] - 5.0, end[1]), (end[0] + 5.0, end[1]), color, 1.0)
        draw.text((start[0] + 9.0, (start[1] + end[1]) * 0.5 - 7.0), color, label)
    else:
        draw.line((start[0], start[1] - 5.0), (start[0], start[1] + 5.0), color, 1.0)
        draw.line((end[0], end[1] - 5.0), (end[0], end[1] + 5.0), color, 1.0)
        width, _ = draw.text_size(label)
        draw.text(((start[0] + end[0] - width) * 0.5, start[1] + 8.0), color, label)


def _even_slider(item_id: str, value: int, minimum: int, maximum: int) -> int:
    changed, candidate = imgui.slider_int(item_id, value, minimum, maximum)
    if not changed:
        return value
    snapped = int(round(candidate / 2.0) * 2)
    return max(minimum, min(maximum, snapped))


def _draw_geometry_controls(position, size, state: ProbeState) -> None:
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    if not imgui.begin_child(
        "Geometry controls###ProbeGeometryControls",
        imgui.ImVec2(float(size[0]), float(size[1])),
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return

    imgui.text("Geometry controls")
    imgui.separator()
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
        _property_label("Center step")
        state.overlay_center_step = _even_slider(
            "##geometry-center-step", state.overlay_center_step, 24, 52
        )
        _property_label("Group gap")
        state.tool_group_gap = _even_slider("##geometry-group-gap", state.tool_group_gap, 4, 24)
        _property_label("Divider")
        state.divider_width = _even_slider("##geometry-divider-width", state.divider_width, 10, 34)
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
            0.9,
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
        imgui.end_table()
    hint_height = state.hint_control_height + state.hint_padding_y * 2
    imgui.text_disabled(f"Single row  ·  shell height {hint_height}")

    imgui.spacing()
    if imgui.button("Reset suggested", imgui.ImVec2(-1.0, 0.0)):
        state.overlay_icon_radius = int(OVERLAY_GEOMETRY.icon_radius)
        state.overlay_radial_step = int(OVERLAY_GEOMETRY.radial_step)
        state.overlay_center_step = int(OVERLAY_GEOMETRY.center_step)
        state.tool_group_gap = int(OVERLAY_GEOMETRY.tool_group_gap)
        state.divider_width = int(OVERLAY_GEOMETRY.divider_width)
        state.construction_playback_scale = 3.0
        state.construction_tool_scale = 1.5
        state.tool_stroke_width = OVERLAY_GEOMETRY.tool_stroke
        state.rotate_ring_gap_ratio = OVERLAY_GEOMETRY.rotate_ring_gap_ratio
        state.rotate_ring_cap = OVERLAY_GEOMETRY.rotate_ring_cap
        state.hint_control_height = int(OVERLAY_GEOMETRY.hint_control_height)
        state.hint_padding_x = int(OVERLAY_GEOMETRY.hint_padding_x)
        state.hint_padding_y = int(OVERLAY_GEOMETRY.hint_padding_y)
        state.hint_input_gap = int(OVERLAY_GEOMETRY.hint_input_gap)
        state.hint_group_gap = int(OVERLAY_GEOMETRY.hint_group_gap)
        state.hint_chord_gap = int(OVERLAY_GEOMETRY.hint_chord_gap)
        state.hint_key_padding_x = int(OVERLAY_GEOMETRY.hint_key_padding_x)
        state.hint_mouse_width = int(OVERLAY_GEOMETRY.hint_mouse_width)

    imgui.end_child()


def _draw_geometry_page(available, scale: float, state: ProbeState) -> None:
    child_flags = _flags(imgui.ChildFlags_.borders)
    window_flags = imgui.WindowFlags_.none.value
    if not imgui.begin_child("Geometry###ProbeGeometry", available, child_flags, window_flags):
        imgui.end_child()
        return

    lo = imgui.get_window_pos()
    size = imgui.get_window_size()
    x0, y0 = float(lo.x), float(lo.y)
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    draw.rect_filled((x0, y0), (x0 + size.x, y0 + size.y), CONCEPT_THEME.bg_child)

    active_tab = state.geometry_tab
    if imgui.begin_tab_bar("##geometry-spec-tabs"):
        for label in (
            "Playback",
            "Tools",
            "Hints & input",
            "Transform gizmos",
            "Joint & helpers",
            "Status",
            "Shell & settings",
            "Panels",
            "Workspaces",
        ):
            tab_flags = imgui.TabItemFlags_.none
            if not state.geometry_tab_initialized and label == state.geometry_tab:
                tab_flags = imgui.TabItemFlags_.set_selected
            opened, _ = imgui.begin_tab_item(label, None, tab_flags)
            if opened:
                active_tab = label
                imgui.end_tab_item()
        imgui.end_tab_bar()
    state.geometry_tab = active_tab
    state.geometry_tab_initialized = True

    content_y = float(imgui.get_cursor_screen_pos().y) + 8.0 * scale
    title_color = CONCEPT_THEME.text
    note_color = CONCEPT_THEME.text_disabled
    icon_radius = float(state.overlay_icon_radius)
    state_radius = icon_radius + float(state.overlay_radial_step)
    shell_radius = state_radius + float(state.overlay_radial_step)
    center_step = float(state.overlay_center_step)
    group_step = center_step + float(state.tool_group_gap)
    controls_width = min(360.0, size.x * 0.25)
    controls_x = x0 + size.x - controls_width - 24.0
    controls_y = content_y
    controls_height = min(760.0, y0 + size.y - controls_y - 18.0)

    show_geometry_controls = active_tab in ("Playback", "Tools", "Hints & input")

    if active_tab == "Playback":
        playback_scale = scale * state.construction_playback_scale
        play_origin = (x0 + 54.0, content_y + 74.0)
        pause_origin = (
            play_origin[0],
            play_origin[1] + (shell_radius * 2.0 + 26.0) * playback_scale,
        )
        draw.text(
            (x0 + 54.0, content_y + 4.0),
            title_color,
            "Playback construction · Play and Pause",
        )
        draw.text(
            (x0 + 54.0, content_y + 28.0),
            note_color,
            "Amber = icon bound · green = state circle · outer line = capsule",
        )
        draw.text((play_origin[0], play_origin[1] - 22.0), note_color, "Play geometry · 3×")
        imgui.push_id("geometry-playback-play")
        _draw_playback(
            draw,
            play_origin,
            playback_scale,
            replace(
                state,
                show_icon_bounds=True,
                show_state_circles=True,
                show_construction_notes=True,
                playing=False,
            ),
        )
        imgui.pop_id()
        draw.text((pause_origin[0], pause_origin[1] - 22.0), note_color, "Pause geometry · 3×")
        imgui.push_id("geometry-playback-pause")
        _draw_playback(
            draw,
            pause_origin,
            playback_scale,
            replace(
                state,
                show_icon_bounds=True,
                show_state_circles=True,
                show_construction_notes=True,
                playing=True,
            ),
        )
        imgui.pop_id()
        pb_center_y = play_origin[1] + shell_radius * playback_scale
        pb_first_x = play_origin[0] + shell_radius * playback_scale
        pb_last_x = pb_first_x + center_step * 2.0 * playback_scale
        _dimension_line(
            draw,
            (pb_first_x, pb_center_y + (shell_radius + 13.0) * playback_scale),
            (pb_last_x, pb_center_y + (shell_radius + 13.0) * playback_scale),
            f"2 × CENTER {state.overlay_center_step}",
        )
        _dimension_line(
            draw,
            (play_origin[0] - 22.0, play_origin[1]),
            (
                play_origin[0] - 22.0,
                play_origin[1] + shell_radius * 2.0 * playback_scale,
            ),
            f"SHELL {int(shell_radius * 2.0)}",
            vertical=True,
        )
        construction_width = (shell_radius * 2.0 + center_step * 2.0) * playback_scale
        if scale <= 1.25:
            playback_notes_x = play_origin[0] + construction_width + 30.0 * scale
            for index, line in enumerate(
                (
                    f"ICON BOUND  Ø{int(icon_radius * 2.0)}",
                    f"STATE CIRCLE Ø{int(state_radius * 2.0)}",
                    f"SHELL HEIGHT {int(shell_radius * 2.0)}",
                    f"CENTER STEP  {state.overlay_center_step}",
                    f"RADIAL STEP  {state.overlay_radial_step:2d}",
                    f"ICON CLEARANCE {int(center_step - icon_radius * 2.0)}",
                    f"STATE CLEARANCE {int(center_step - state_radius * 2.0)}",
                )
            ):
                color = (
                    CONCEPT_THEME.warning
                    if index == 0
                    else CONCEPT_THEME.primary_bright
                    if index == 1
                    else note_color
                )
                draw.text((playback_notes_x, content_y + 82.0 + index * 25.0), color, line)

        comparison_x = x0 + max(650.0, size.x * 0.43)
        draw.text((comparison_x, content_y + 4.0), title_color, "Playback states · 2×")
        product_state = replace(
            state,
            show_icon_bounds=False,
            show_state_circles=False,
            show_construction_notes=False,
        )
        paused_state = replace(product_state, playing=False)
        playing_state = replace(product_state, playing=True)
        imgui.push_id("product-playback-paused")
        _draw_playback(draw, (comparison_x, content_y + 58.0), scale * 2.0, paused_state)
        imgui.pop_id()
        playing_x = comparison_x
        playing_y = content_y + 198.0
        imgui.push_id("product-playback-playing")
        _draw_playback(draw, (playing_x, playing_y), scale * 2.0, playing_state)
        imgui.pop_id()
        draw.text((comparison_x, content_y + 158.0), note_color, "Paused · Play")
        draw.text((playing_x, playing_y + 100.0), note_color, "Playing · Pause")
        draw.text(
            (comparison_x, playing_y + 128.0),
            note_color,
            "Play is neutral · Pause stays selected while playing.",
        )

    elif active_tab == "Tools":
        construction_state = replace(
            state,
            show_icon_bounds=True,
            show_state_circles=True,
            show_construction_notes=True,
        )
        tool_scale = scale * state.construction_tool_scale
        tool_origin = (x0 + 96.0, content_y + 56.0)
        draw.text(
            (x0 + 54.0, content_y + 4.0),
            title_color,
            f"Construction geometry · {state.construction_tool_scale:.1f}× inspection",
        )
        draw.text(
            (x0 + 54.0, content_y + 28.0),
            note_color,
            "Amber = icon bound · green = state circle · outer line = capsule",
        )
        imgui.push_id("geometry-tools")
        _draw_tool_column(draw, tool_origin, tool_scale, construction_state)
        imgui.pop_id()
        tool_notes_x = tool_origin[0] + 150.0
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
            draw.text((tool_notes_x, content_y + 76.0 + index * 25.0), color, line)

        comparison_x = x0 + max(650.0, size.x * 0.43)
        product_tool_origin = (comparison_x, content_y + 56.0)
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
            (labels_x, content_y + 4.0),
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
        )
        for center, (label, meaning) in zip(
            product_centers,
            (
                ("Move", "Translate selected object"),
                ("Rotate", "3 half-rings + screen ring"),
                ("World / Body", "Switch transform frame"),
                ("Snap", "Toggle snapping"),
            ),
            strict=True,
        ):
            label_y = product_tool_origin[1] + center * product_tool_scale
            draw.text((labels_x, label_y - 15.0 * scale), title_color, label)
            draw.text((labels_x, label_y + 3.0 * scale), note_color, meaning)

        frame_samples_x = labels_x + 214.0 * scale
        frame_samples_y = product_tool_origin[1] + product_centers[2] * product_tool_scale
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

    elif active_tab == "Hints & input":
        hint_label_x = x0 + 54.0
        draw.text((hint_label_x, content_y + 4.0), title_color, "Context hint states")
        draw.text(
            (hint_label_x, content_y + 28.0),
            note_color,
            "Defaults are composed into Status; each whole group is dropped when space runs out.",
        )
        for index, (label, variant) in enumerate(
            (
                ("No selection", "camera"),
                ("Transform ready", "ready"),
                ("Transform drag", "dragging"),
                ("Ctrl held", "perturb"),
            )
        ):
            row_y = content_y + 70.0 + index * 58.0 * scale
            draw_status(
                draw,
                (hint_label_x, row_y),
                size.x * 0.75,
                28.0 * scale,
                CONCEPT_THEME,
                scale,
                selected=label,
                has_selection=variant != "camera",
                state="paused",
                sim_time=1.204,
                step=1204,
                metric_mode="time",
                backend="OpenGL",
                dt=0.002,
                fps=60.0,
                tool_hints=default_tool_hints(variant, DEFAULT_INPUT_BINDINGS),
            )

        draw.text(
            (hint_label_x, content_y + 306.0 * scale),
            note_color,
            "Reusable scene surface (custom hint providers can opt in):",
        )
        hint_x = hint_label_x
        _draw_hint_bar(draw, (hint_x, content_y + 332.0 * scale), scale, state, "camera")
        card_y = content_y + 396.0 * scale
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
        draw.text((x0 + 54.0, content_y + 4.0), title_color, "Transform gizmos · M8 target")
        draw.text(
            (x0 + 54.0, content_y + 28.0),
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
            draw.text((x0 + 54.0, row_y - 120.0 * scale), title_color, heading)
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
                )
                label_width, _ = draw.text_size(label)
                draw.text(
                    (center[0] - label_width * 0.5, row_y + 108.0 * scale),
                    note_color,
                    label,
                )
        draw.text(
            (x0 + 54.0, content_y + 612.0 * scale),
            note_color,
            "Hover/active + snap tick = Primary Bright; drag sector = Primary Dim at low alpha.",
        )
        draw.text(
            (x0 + 54.0, content_y + 638.0 * scale),
            note_color,
            "2D / 3D changes geometry only; label and interaction-state rules stay identical.",
        )

    elif active_tab == "Joint & helpers":
        draw.text((x0 + 54.0, content_y + 4.0), title_color, "Joint gizmos · M8 target")
        draw.text(
            (x0 + 54.0, content_y + 28.0),
            note_color,
            "Purple handles · blue MIN tick · red MAX tick · backed labels with white text.",
        )
        imgui.push_id("geometry-joint-gizmo")
        _draw_joint_gizmo(
            draw,
            (x0 + 54.0, content_y + 58.0 * scale),
            scale,
            state,
            item_id="geometry-joint",
        )
        imgui.pop_id()
        draw.text(
            (x0 + 54.0, content_y + 354.0 * scale),
            note_color,
            "Current = Primary Bright tick · double-click the active handle = Type value.",
        )
        draw.text(
            (x0 + 54.0, content_y + 404.0 * scale),
            title_color,
            "Hinge drag + Shift",
        )
        _draw_joint_rotation_feedback(
            draw,
            (x0 + 230.0 * scale, content_y + 520.0 * scale),
            76.0 * scale,
            scale,
        )
        draw.text(
            (x0 + 340.0 * scale, content_y + 472.0 * scale),
            CONCEPT_THEME.primary_bright,
            "Primary Bright  active arc / tick",
        )
        draw.text(
            (x0 + 340.0 * scale, content_y + 500.0 * scale),
            CONCEPT_THEME.primary_dim,
            "Primary Dim  sweep sector · 24% alpha",
        )
        draw.text(
            (x0 + 340.0 * scale, content_y + 528.0 * scale),
            note_color,
            "Text Disabled  passive snap ticks",
        )

        helper_x = x0 + max(700.0 * scale, size.x * 0.47)
        helper_width = min(500.0 * scale, x0 + size.x - helper_x - 30.0 * scale)
        draw.text((helper_x, content_y + 4.0), title_color, "Camera / light helpers")
        draw.text(
            (helper_x, content_y + 28.0),
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
            (x0 + 54.0, content_y + 650.0 * scale),
            note_color,
            "Renderer acceptance: 3D depth, occlusion and picking stay in make gizmo / lighting tests.",
        )

    elif active_tab == "Status":
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0, content_y))
        _draw_status_tab(imgui.ImVec2(size.x - 24.0, y0 + size.y - content_y - 12.0), scale)

    elif active_tab == "Shell & settings":
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0, content_y))
        _draw_shell_settings_tab(
            imgui.ImVec2(size.x - 24.0, y0 + size.y - content_y - 12.0), scale, state
        )

    elif active_tab == "Panels":
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0, content_y))
        _draw_panel_gallery(
            imgui.ImVec2(size.x - 24.0, y0 + size.y - content_y - 12.0), state, scale
        )

    else:
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 12.0, content_y))
        _draw_workspaces_tab(
            imgui.ImVec2(size.x - 24.0, y0 + size.y - content_y - 12.0), scale, state
        )

    if active_tab in (
        "Playback",
        "Tools",
        "Hints & input",
        "Transform gizmos",
        "Joint & helpers",
    ):
        extent = 690.0 * scale
        imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 8.0, content_y + extent))
        imgui.dummy(imgui.ImVec2(1.0, 1.0))

    if show_geometry_controls:
        _draw_geometry_controls(
            (controls_x, controls_y),
            (controls_width, controls_height),
            state,
        )

    imgui.end_child()


def _draw_workspace(window: Window, state: ProbeState) -> None:
    scale = window.style_scale
    display = imgui.get_io().display_size
    imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
    imgui.set_next_window_size(display)
    flags = _flags(
        imgui.WindowFlags_.no_title_bar,
        imgui.WindowFlags_.no_resize,
        imgui.WindowFlags_.no_move,
        imgui.WindowFlags_.no_collapse,
        imgui.WindowFlags_.no_saved_settings,
        imgui.WindowFlags_.menu_bar,
    )
    opened, _ = imgui.begin("Forge Viewer UI Probe", True, flags)
    if not opened:
        imgui.end()
        return

    if imgui.begin_menu_bar():
        for label in ("File", "Edit", "Entity", "View", "Window", "Help"):
            if imgui.begin_menu(label):
                imgui.menu_item("Design probe", "", False, False)
                imgui.end_menu()
        if imgui.begin_menu("Probe"):
            for page in ("Workspace", "Panels", "Geometry"):
                clicked, _ = imgui.menu_item(page, "", state.page == page)
                if clicked:
                    state.page = page
            imgui.separator()
            imgui.text_disabled("Viewport overlays")
            for label, attribute in (
                ("Playback", "show_playback"),
                ("Tool column", "show_tool_column"),
                ("Joint gizmos", "show_joint_gizmos"),
                ("Context hints", "show_context_hints"),
                ("Value input", "value_open"),
                ("Joint value input", "joint_value_open"),
            ):
                value = bool(getattr(state, attribute))
                clicked, _ = imgui.menu_item(label, "", value)
                if clicked:
                    setattr(state, attribute, not value)
            imgui.separator()
            imgui.text_disabled("Panels")
            for label, attribute in (
                ("Settings", "show_settings"),
                ("Keyframes", "show_keyframes"),
                ("Output", "show_output"),
            ):
                value = bool(getattr(state, attribute))
                clicked, _ = imgui.menu_item(label, "", value)
                if clicked:
                    setattr(state, attribute, not value)
            imgui.separator()
            imgui.text_disabled("Construction")
            all_construction = bool(
                state.show_icon_bounds
                and state.show_state_circles
                and state.show_construction_notes
            )
            clicked, _ = imgui.menu_item("All construction", "", all_construction)
            if clicked:
                target = not all_construction
                state.show_icon_bounds = target
                state.show_state_circles = target
                state.show_construction_notes = target
            for label, attribute in (
                ("Icon bounds", "show_icon_bounds"),
                ("State circles", "show_state_circles"),
                ("Geometry notes", "show_construction_notes"),
            ):
                value = bool(getattr(state, attribute))
                clicked, _ = imgui.menu_item(label, "", value)
                if clicked:
                    setattr(state, attribute, not value)
            imgui.end_menu()
        imgui.end_menu_bar()

    available = imgui.get_content_region_avail()
    if state.page == "Panels":
        _draw_panel_gallery(available, state, scale)
        imgui.end()
        return
    if state.page == "Geometry":
        _draw_geometry_page(available, scale, state)
        imgui.end()
        viewport = imgui.get_main_viewport()
        _draw_joint_value_input(
            (
                viewport.work_pos.x + 410.0 * scale,
                viewport.work_pos.y + 250.0 * scale,
            ),
            scale,
            state,
        )
        return

    spacing = imgui.get_style().item_spacing
    workspace_origin = imgui.get_cursor_screen_pos()
    status_height = 34.0 * scale
    workspace_height = max(1.0, available.y - status_height - spacing.y)
    show_bottom = state.show_keyframes or state.show_output
    bottom_height = min(245.0 * scale, workspace_height * 0.30) if show_bottom else 0.0
    top_height = workspace_height - (bottom_height + spacing.y if show_bottom else 0.0)
    right_width = max(470.0 * scale, available.x * 0.31) if state.show_settings else 0.0
    viewport_width = available.x - (right_width + spacing.x if state.show_settings else 0.0)

    viewport_rect = _draw_viewport(imgui.ImVec2(viewport_width, top_height), scale, state)
    if state.show_settings:
        imgui.same_line()
        _draw_settings(imgui.ImVec2(right_width, top_height), state)

    if state.show_keyframes:
        keyframes_width = available.x * 0.63 if state.show_output else available.x
        _draw_keyframes(imgui.ImVec2(keyframes_width, bottom_height), scale)
    if state.show_output:
        if state.show_keyframes:
            imgui.same_line()
        _draw_output(imgui.ImVec2(0.0, bottom_height), state)

    status_origin = (
        workspace_origin.x,
        workspace_origin.y + available.y - status_height,
    )
    _draw_status_strip(
        ImguiDraw2D(imgui.get_window_draw_list()),
        status_origin,
        available.x,
        status_height,
        scale,
        selected="a_sphere",
        running=state.playing,
        fps="60.0" if state.playing else "119.8",
    )

    imgui.end()

    _draw_value_input(
        (viewport_rect[2] - 326.0 * scale, viewport_rect[1] + 74.0 * scale),
        scale,
        state,
    )
    _draw_joint_value_input(
        (viewport_rect[0] + 420.0 * scale, viewport_rect[1] + 250.0 * scale),
        scale,
        state,
    )


def render(
    output: Path,
    width: int,
    height: int,
    *,
    interactive: bool,
    initial_page: str,
    initial_geometry_tab: str,
    initial_rotate_cap: str,
    ui_scale: float,
    interactive_fps: float,
) -> None:
    window = Window(
        WindowConfig(
            title="Forge Viewer UI feasibility",
            width=width,
            height=height,
            # The probe is UI, not a GPU benchmark. Interactive mode is paced,
            # and DrawList antialiasing already covers its vector edges.
            vsync=interactive,
            docking=False,
            ini_path="",
            show_on_start=False,
            samples=0,
            ui_scale=ui_scale,
        )
    )
    try:
        _apply_concept_theme(window.style_scale)
        state = ProbeState(
            page=initial_page,
            geometry_tab=initial_geometry_tab,
            rotate_ring_cap=initial_rotate_cap,
        )
        if interactive:
            window.show()
            frame_period = 1.0 / interactive_fps
            try:
                while not window.should_close():
                    frame_started = time.perf_counter()
                    window.begin_frame()
                    _draw_workspace(window, state)
                    if imgui.is_key_pressed(imgui.Key.escape, False):
                        if state.joint_value_open:
                            state.joint_value_open = False
                        elif state.value_open:
                            state.value_open = False
                        elif not imgui.is_any_item_active():
                            window.request_close()
                    window.end_frame()
                    remaining = frame_period - (time.perf_counter() - frame_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
            except KeyboardInterrupt:
                pass
            return

        pixels = None
        for _ in range(4):
            window.begin_frame()
            _draw_workspace(window, state)
            pixels = window.end_frame(readback=True)
        if pixels is None:
            raise RuntimeError("ImGui framebuffer readback returned no pixels")
        image = Image.fromarray(np.asarray(pixels)[::-1], "RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    finally:
        window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.0,
        help="Logical UI scale used for paths, strokes, controls, and text",
    )
    parser.add_argument(
        "--page",
        choices=("workspace", "panels", "geometry"),
        default="workspace",
        help="Initial probe page; interactive mode can switch from the Probe menu",
    )
    parser.add_argument(
        "--geometry-tab",
        choices=(
            "playback",
            "tools",
            "hints",
            "gizmos",
            "helpers",
            "status",
            "shell",
            "panels",
            "workspaces",
        ),
        default="playback",
        help="Initial non-closeable tab on the geometry page",
    )
    parser.add_argument(
        "--rotate-cap",
        choices=("butt", "round"),
        default=OVERLAY_GEOMETRY.rotate_ring_cap,
        help="Initial Rotate inner-ring cap style",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a real ImGui window and run until it is closed",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Maximum interactive refresh rate (default: 30)",
    )
    args = parser.parse_args()
    if not 0.75 <= args.ui_scale <= 4.0:
        parser.error("--ui-scale must be between 0.75 and 4.0")
    if not 15.0 <= args.fps <= 240.0:
        parser.error("--fps must be between 15 and 240")
    output = args.output.resolve()
    render(
        output,
        args.width,
        args.height,
        interactive=args.interactive,
        initial_page=args.page.title(),
        initial_geometry_tab={
            "playback": "Playback",
            "tools": "Tools",
            "hints": "Hints & input",
            "gizmos": "Transform gizmos",
            "helpers": "Joint & helpers",
            "status": "Status",
            "shell": "Shell & settings",
            "panels": "Panels",
            "workspaces": "Workspaces",
        }[args.geometry_tab],
        initial_rotate_cap=args.rotate_cap,
        ui_scale=args.ui_scale,
        interactive_fps=args.fps,
    )
    print("interactive probe closed" if args.interactive else output)


if __name__ == "__main__":
    main()
