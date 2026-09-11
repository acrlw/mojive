#!/usr/bin/env python3
"""Render the UI feasibility probe through Mojive's real ImGui backend.

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

if __package__:
    from .ui_capsule_geometry import (
        CAPSULE_OUTLINE_LABELS,
        CAPSULE_SMOOTHING,
        capsule_layout,
        capsule_outline_color,
        draw_capsule_shell,
        end_padding,
        spacing_metrics,
    )
    from .ui_icon_concepts import (
        ICON_BOUND_DIAMETER,
        ICON_DEFAULT_PADDING,
        ICON_FAMILIES,
        ICON_GRID,
        ICON_GROUP_BY_SLUG,
        ICON_GROUP_LAYOUT_DEFAULTS,
        ICON_LIBRARY_TABS,
        ICON_MAX_PADDING,
        ICON_MAX_STROKE,
        ICON_MIN_CLEARANCE,
        ICON_MIN_STROKE,
        ICON_STROKE,
        ROTATE_FRAME_PADDING,
        STATUS_MOUSE_DEFAULT_WIDTH,
        draw_concept_icon,
        icon_alignment_anchor,
        icon_alignment_center,
        icon_component_group,
        icon_family,
        icon_metrics,
        minimum_enclosing_circle,
    )
    from .ui_redesign import (
        RECORD_GLYPH_RADIUS,
        RESET_GLYPH_SCALE,
        RedesignState,
        _capsule,
        _recording_menu,
        draw_recording_glyph,
        draw_redesign,
        draw_reset_glyph,
    )
else:
    from ui_capsule_geometry import (
        CAPSULE_OUTLINE_LABELS,
        CAPSULE_SMOOTHING,
        capsule_layout,
        capsule_outline_color,
        draw_capsule_shell,
        end_padding,
        spacing_metrics,
    )
    from ui_icon_concepts import (
        ICON_BOUND_DIAMETER,
        ICON_DEFAULT_PADDING,
        ICON_FAMILIES,
        ICON_GRID,
        ICON_GROUP_BY_SLUG,
        ICON_GROUP_LAYOUT_DEFAULTS,
        ICON_LIBRARY_TABS,
        ICON_MAX_PADDING,
        ICON_MAX_STROKE,
        ICON_MIN_CLEARANCE,
        ICON_MIN_STROKE,
        ICON_STROKE,
        ROTATE_FRAME_PADDING,
        STATUS_MOUSE_DEFAULT_WIDTH,
        draw_concept_icon,
        icon_alignment_anchor,
        icon_alignment_center,
        icon_component_group,
        icon_family,
        icon_metrics,
        minimum_enclosing_circle,
    )
    from ui_redesign import (
        RECORD_GLYPH_RADIUS,
        RESET_GLYPH_SCALE,
        RedesignState,
        _capsule,
        _recording_menu,
        draw_recording_glyph,
        draw_redesign,
        draw_reset_glyph,
    )
from imgui_bundle import imgui
from PIL import Image

from mojive import gizmo as gizmo_geometry
from mojive.curves2d import CORNER_SMOOTHING
from mojive.types import CameraView
from mojive.ui import gizmo as gizmo_ui
from mojive.ui import perturb as perturb_ui
from mojive.ui import theme as theme_mod
from mojive.ui import viewcube as view_ui
from mojive.ui.compound_fields import draw_joined_field_frame
from mojive.ui.draw2d import ImguiDraw2D, draw_drag_link, text_line_y
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.messages import OutputBuffer
from mojive.ui.panels import (
    PanelContext,
    button_row_layout,
    button_width,
    search_input,
    searchable_ordered_list_header,
)
from mojive.ui.panels import keyframes as keyframes_panel_module
from mojive.ui.panels import output as output_panel_module
from mojive.ui.panels.filters import filter_pills, severity_color, severity_icon, severity_meshes
from mojive.ui.panels.hierarchy import disclosure_triangle
from mojive.ui.panels.keyframes import KeyframesPanel, _command_button, _draw_command_icon
from mojive.ui.panels.output import OutputPanel
from mojive.ui.panels.settings import settings_uses_stacked_layout
from mojive.ui.panels.value_cards import draw_value_rail, interval_text, value_card, value_rail
from mojive.ui.perturb import OUTLINE_CORNER_RADIUS_PT
from mojive.ui.theme import THEME, rgb8
from mojive.ui.viewcube import DEFAULT_SELECTION_PADDING
from mojive.ui.viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    OVERLAY_GEOMETRY,
    PLAYBACK_HALF_HEIGHT_PT,
    PLAYBACK_RESET_SCALE,
    RECORDING_OPTIONS_GLYPH_SCALE,
    RECORDING_OPTIONS_STROKE_SCALE,
    TOOL_GLYPH_SCALE,
    ToolHint,
    _viewport_control_colors,
    capsule_points,
    default_tool_hints,
    draw_mouse_hint_glyph,
    draw_playback_glyph,
    draw_projection_label,
    draw_recording_options_glyph,
    draw_status,
    draw_tool_glyph,
    keycap_rounding,
    overlay_divider_length,
)
from mojive.ui.window import Window, WindowConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output" / "ui-drawing-feasibility.png"
PROBE_BASE_SIZE = (1600, 1000)
GEOMETRY_CANVAS_SIZE = (1560.0, 900.0)
GEOMETRY_TABS = (
    "Corners",
    "Playback",
    "Tools",
    "Icon library",
    "Hints & input",
    "Transform gizmos",
    "Joint & helpers",
    "Status",
    "Diagnostics",
    "Shell & settings",
    "Panels",
    "Workspaces",
)
PANEL_CANVAS_SIZE = (1560.0, 900.0)
WORKSPACE_CANVAS_SIZE = (1600.0, 960.0)


def _probe_window_size(width: int, height: int, ui_scale: float) -> tuple[int, int]:
    """Keep the whole concept canvas visible while inspecting extreme UI scales.

    The probe is the design reference, so a capture that clips its right-hand panels
    misrepresents the design. Growing the window with the UI scale keeps every specimen
    laid out at its real proportions instead of hiding it behind a scrollbar. Both factors
    stay at 1.0 for the default scale, which already fits the base window.
    """

    scale = float(ui_scale)
    # The geometry canvas and its right-hand experiment controls together need about 1.25x
    # the base width, and the tallest canvas about 0.95x the base height.
    growth = scale * 1.25 if scale > 1.0 else 1.0
    vertical = max(1.0, scale * 0.95)
    return round(width * growth), round(height * vertical)


def _virtual_canvas_size(
    available,
    scale: float,
    logical_size: tuple[float, float],
) -> tuple[float, float]:
    """Preserve component proportions and let extreme scales scroll."""

    return (
        max(float(available.x), logical_size[0] * scale),
        max(float(available.y), logical_size[1] * scale),
    )


def _wrapped_tabs(
    tabs: tuple[tuple[str, str, str], ...],
    active: str,
    available: float,
    *,
    initial: str | None = None,
    gap: float | None = None,
) -> str:
    """Draw one tab row that wraps instead of clipping at large UI scales.

    Native tab bars extend past the panel edge once the labels grow, which hides the
    rightmost entries and overlaps the neighbouring content. The probe keeps the tab
    grammar but reflows it, matching how the production panels wrap their own rows.
    """

    style = imgui.get_style()
    spacing = style.item_spacing.x if gap is None else float(gap)
    width = max(1.0, float(available))
    default_focus = bool(initial) and active != initial
    if default_focus:
        active = str(initial)
    padding = 10.0
    # Every label needs its own ImGui ID, since the probe shows overlapping tab sets.
    entries = tuple(
        (label, f"{scope}-{index}-{slug}") for index, (label, scope, slug) in enumerate(tabs)
    )
    widths = tuple(
        min(width, imgui.calc_text_size(label).x + 2.0 * style.frame_padding.x + padding)
        for label, _ in entries
    )
    inline = button_row_layout(widths, width, spacing)
    height = style.frame_padding.y * 2.0 + imgui.get_text_line_height()
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    imgui.begin_group()
    for index, (label, identifier) in enumerate(entries):
        if index and inline[index]:
            imgui.same_line(0.0, spacing)
        selected = label == active
        pressed = imgui.invisible_button(
            f"##{identifier}", imgui.ImVec2(widths[index], height), imgui.ButtonFlags_.none
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        hovered = imgui.is_item_hovered()
        # A passive label keeps one click meaning "select", never "toggle off".
        color = CONCEPT_THEME.text if selected or hovered else CONCEPT_THEME.text_disabled
        text = imgui.calc_text_size(label)
        draw.text(
            (
                (lo.x + hi.x - text.x) * 0.5,
                (lo.y + hi.y) * 0.5 - text.y * 0.5 - style.frame_padding.y * 0.5,
            ),
            color,
            label,
        )
        if selected:
            draw.line(
                (lo.x, hi.y - 1.0),
                (hi.x, hi.y - 1.0),
                CONCEPT_THEME.primary,
                max(1.0, style.frame_padding.y),
            )
        if default_focus and selected:
            imgui.set_item_default_focus()
        if pressed:
            active = label
    imgui.end_group()
    return active


# Start every concept session from the production palette. Geometry and state
# controls below remain experimental, while accepted theme changes propagate
# here automatically instead of leaving a second stale color table.
CONCEPT_THEME = replace(THEME)
JOINT_COLOR = CONCEPT_THEME.primary
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
PROBE_FRAME_SAMPLES = np.asarray(
    (8.4, 8.1, 8.5, 8.2, 8.3, 8.6, 8.2, 8.4, 8.3, 8.35),
    np.float32,
)
PROBE_PLOT_SAMPLES = np.asarray(
    (0.00, 0.08, 0.16, 0.27, 0.35, 0.41, 0.38, 0.31, 0.20, 0.11, 0.04),
    np.float32,
)

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
    (math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in range(-65, 216)
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
GIZMO_PROBE_SPECIMENS = {
    mode: gizmo_ui.ObjectGizmo(mode) for mode in ("translate", "rotate", "dimensions")
}
CORNER_VIEW_GIZMO = view_ui.ViewCube()
CORNER_CONTROLS = (
    ("Capsules", "capsule_smoothing"),
    ("Playback icons", "playback_smoothing"),
    ("Tool icons", "tool_smoothing"),
    ("Mouse hints", "mouse_smoothing"),
    ("Transform gizmo", "transform_smoothing"),
    ("Joint gizmo", "joint_smoothing"),
    ("View gizmo", "view_smoothing"),
    ("Perturbation gizmo", "perturb_smoothing"),
)
GIZMO_IDENTITY_F32 = np.eye(3, dtype=np.float32)
GIZMO_IDENTITY_F64 = np.eye(3, dtype=np.float64)


@dataclass
class ProbeState:
    page: str = "Workspace"
    redesign: RedesignState = field(default_factory=RedesignState)
    imgui_rounding: float = theme_mod.DEFAULT_CORNER_RADIUS
    imgui_example_value: float = 0.0
    imgui_example_enabled: bool = True
    capsule_smoothing: float = CAPSULE_SMOOTHING
    capsule_outline: str = "Soft white"
    playback_smoothing: float = CAPSULE_SMOOTHING
    tool_smoothing: float = CAPSULE_SMOOTHING
    mouse_smoothing: float = CORNER_SMOOTHING
    transform_smoothing: float = CORNER_SMOOTHING
    joint_smoothing: float = CORNER_SMOOTHING
    view_smoothing: float = CORNER_SMOOTHING
    perturb_smoothing: float = CORNER_SMOOTHING

    geometry_tab: str = "Playback"
    geometry_tab_initialized: bool = False
    icon_library_tab: str = "Overview"
    preview_icon_library: bool = False
    show_playback: bool = True
    show_tool_column: bool = True
    show_joint_gizmos: bool = True
    show_context_hints: bool = False
    show_icon_bounds: bool = False
    show_state_circles: bool = False
    show_construction_notes: bool = False
    highlight_g3: bool = False
    optical_capsule_spacing: bool = True
    playing: bool = False
    active_tool: str = "move"
    gizmo_space: str = "world"
    gizmo_style: int = 1
    frame: int = 1
    remember_input: bool = True
    viewport_overlay_scale: float = DEFAULT_VIEWPORT_OVERLAY_SCALE
    position_snap: float = gizmo_ui.DEFAULT_TRANSLATION_SNAP_M
    rotation_snap: float = gizmo_ui.DEFAULT_ROTATION_SNAP_DEG
    tick_scale: float = gizmo_ui.DEFAULT_ROTATION_TICK_SCALE
    selection_padding: float = DEFAULT_SELECTION_PADDING
    corner_radius: float = OUTLINE_CORNER_RADIUS_PT
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
    output_panel: OutputPanel = field(default_factory=OutputPanel)
    timeline_panel: KeyframesPanel = field(default_factory=KeyframesPanel)
    timeline_session: object = None
    angular_degrees: bool = False
    output_buffer: OutputBuffer | None = None
    diagnostic_levels: set[str] = field(default_factory=lambda: {"info", "warning", "error"})
    hinge_ctrl: float = 0.25
    slide_ctrl: float = 0.0
    weld_enabled: bool = True
    connect_enabled: bool = False
    hinge_position: float = 0.35
    slide_position: float = 0.0
    hierarchy_filter: str = ""
    hierarchy_kind: int = 0
    hierarchy_selection: int = 2
    control_filter: str = ""
    control_sort_by_name: bool = False
    joint_filter: str = ""
    joint_sort_by_name: bool = False
    workspace_right_tab: str = "Control"
    hierarchy_visibility: list[bool] = field(
        default_factory=lambda: [True, True, False, True, True, True]
    )
    asset_selection: int = 1
    asset_filter: str = ""
    asset_type: int = 0
    sensor_index: int = 0
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
    icon_adjustment_group: str = "Viewport tools"
    icon_padding_by_group: dict[str, float] = field(
        default_factory=lambda: dict(ICON_GROUP_LAYOUT_DEFAULTS)
    )
    icon_stroke_width: float = ICON_STROKE
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
    hint_mouse_stroke: float = OVERLAY_GEOMETRY.hint_mouse_stroke
    hint_mouse_button_width_ratio: float = OVERLAY_GEOMETRY.hint_mouse_button_width_ratio
    hint_mouse_button_shell_ratio: float = OVERLAY_GEOMETRY.hint_mouse_button_shell_ratio
    hint_mouse_button_height_ratio: float = OVERLAY_GEOMETRY.hint_mouse_button_height_ratio
    hint_mouse_wheel_width_ratio: float = OVERLAY_GEOMETRY.hint_mouse_wheel_width_ratio
    hint_mouse_wheel_height_ratio: float = OVERLAY_GEOMETRY.hint_mouse_wheel_height_ratio
    hint_mouse_wheel_gap_ratio: float = OVERLAY_GEOMETRY.hint_mouse_wheel_gap_ratio

    def icon_padding_for(self, group: str) -> float:
        return self.icon_padding_by_group.get(group, ICON_GROUP_LAYOUT_DEFAULTS[group])

    def set_icon_padding_for(self, group: str, padding: float) -> None:
        self.icon_padding_by_group[group] = float(padding)

    def concept_mouse_width(self) -> float:
        """Return the master width that preserves the chosen status aspect ratio."""

        return (
            STATUS_MOUSE_DEFAULT_WIDTH
            * self.hint_mouse_width
            / OVERLAY_GEOMETRY.hint_mouse_width
            * OVERLAY_GEOMETRY.hint_control_height
            / self.hint_control_height
        )


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


def _draw_concept_control_icon(
    draw,
    center,
    size: float,
    kind: str,
    color,
    *,
    padding: float = ICON_DEFAULT_PADDING,
    stroke_width: float = ICON_STROKE,
) -> None:
    """Adapt Icon Library panel candidates to the shared control callback."""

    draw_concept_icon(
        draw,
        center,
        size,
        f"panel-{kind}",
        color,
        padding=padding,
        stroke_width=stroke_width,
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
) -> None:
    name = "panel-perspective" if kind == "persp" else "panel-orthographic"
    draw_concept_icon(
        draw,
        center,
        size,
        name,
        color,
        padding=padding,
        stroke_width=stroke_width,
    )


def _preview_search_input(state: ProbeState, *args, **kwargs):
    if state.preview_icon_library:
        padding = state.icon_padding_for("Panels")
        kwargs["icon_drawer"] = lambda *values: _draw_concept_control_icon(
            *values,
            padding=padding,
            stroke_width=state.icon_stroke_width,
        )
    return search_input(*args, **kwargs)


def _preview_searchable_header(state: ProbeState, *args, **kwargs):
    if state.preview_icon_library:
        padding = state.icon_padding_for("Panels")
        kwargs["icon_drawer"] = lambda *values: _draw_concept_control_icon(
            *values,
            padding=padding,
            stroke_width=state.icon_stroke_width,
        )
    return searchable_ordered_list_header(*args, **kwargs)


_KEYFRAME_CONCEPT_ICONS = {
    "first": "transport-first",
    "last": "transport-last",
    "previous": "transport-previous",
    "next": "transport-next",
    "play": "transport-play",
    "pause": "transport-pause",
    "stop": "transport-stop",
    "loop": "transport-reset",
    "reset": "transport-reset",
    "options": "transport-more",
    "record": "transport-record",
    "add": "key-add",
    "clear": "key-clear",
    "key-previous": "key-previous",
    "key-next": "key-next",
    "fit": "key-fit",
    "follow": "key-follow",
    "view": "key-view",
}


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
    name = (
        "key-keyframe"
        if kind == "key" and scale < context_scale * 0.9
        else "key-snapshot"
        if kind == "key"
        else _KEYFRAME_CONCEPT_ICONS[kind]
    )
    if state is not None:
        padding = state.icon_padding_for(icon_component_group(name))
        stroke_width = state.icon_stroke_width
    else:
        stroke_width = ICON_STROKE
    draw_concept_icon(
        draw,
        center,
        16.0 * float(scale),
        name,
        color,
        padding=padding,
        stroke_width=stroke_width,
    )


def _draw_play_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "play", smoothing=draw.corner_smoothing)


def _draw_pause_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "pause", smoothing=draw.corner_smoothing)


def _draw_step_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "step", smoothing=draw.corner_smoothing)


def _draw_previous_icon(draw: ImguiDraw2D, center, color, scale: float, _surface=None) -> None:
    draw_playback_glyph(draw, center, color, scale, "previous", smoothing=draw.corner_smoothing)


def _draw_reset_icon(
    draw: ImguiDraw2D,
    center,
    color,
    scale: float,
    _surface=None,
    *,
    stroke_width: float = OVERLAY_GEOMETRY.tool_stroke,
) -> None:
    draw_reset_glyph(draw, center, color, scale, stroke_width)


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
        draw.circle_filled(center, state_radius * scale, guide_background)
        draw.circle(
            center,
            state_radius * scale,
            (*CONCEPT_THEME.primary_dim[:3], 0.95),
            1.0 * scale,
        )
    elif background[3] > 0.0:
        draw.circle_filled(center, state_radius * scale, background)
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
        padding = state.icon_padding_for("Scene helpers")
        draw_concept_icon(
            draw,
            center,
            20.0 * scale,
            "helper-camera" if kind == "camera" else "helper-light",
            color,
            padding=padding,
            stroke_width=state.icon_stroke_width,
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
    icons: tuple[str, ...] | None = None,
    icon_drawer=None,
) -> int:
    result = selected
    draw = ImguiDraw2D()
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(1.0, 0.0))
    for index, label in enumerate(labels):
        if index:
            imgui.same_line()
        if index == selected:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*CONCEPT_THEME.bg_frame_active))
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*CONCEPT_THEME.primary_bright))
        icon = icons[index] if icons is not None and index < len(icons) else ""
        button_label = f"##{item_id}-{index}" if icon else f"{label}##{item_id}-{index}"
        if imgui.button(button_label, imgui.ImVec2(width, 0.0)):
            result = index
        item_min = imgui.get_item_rect_min()
        item_max = imgui.get_item_rect_max()
        if index == selected:
            imgui.pop_style_color(2)
        if icon:
            color = CONCEPT_THEME.primary_bright if index == selected else CONCEPT_THEME.text
            if icon_drawer is not None:
                height = item_max.y - item_min.y
                icon_size = height * 0.64
                center = (item_min.x + height * 0.5, (item_min.y + item_max.y) * 0.5)
                icon_drawer(draw, center, icon_size, icon, color)
                draw.text(
                    (item_min.x + height, text_line_y(draw, center[1])),
                    color,
                    label,
                )
            else:
                glyph_scale = max(0.65, imgui.get_frame_height() / 24.0)
                draw_projection_label(
                    draw,
                    (item_min.x, item_min.y),
                    (item_max.x, item_max.y),
                    color,
                    glyph_scale,
                    icon,
                    label,
                )
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
    imgui.text("OpenGL")
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
    imgui.text_disabled("Backend  OpenGL / OpenGL")
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


def _draw_settings(size, state: ProbeState, scale: float) -> None:
    if not imgui.begin_child("Settings###ProbeSettings", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Settings")
    imgui.separator()
    available = imgui.get_content_region_avail()
    categories = ("General", "Interaction", "Rendering", "MuJoCo Visuals")
    stacked = settings_uses_stacked_layout(available.x, scale)
    table_open = False
    if stacked:
        imgui.set_next_item_width(-1.0)
        if imgui.begin_combo("##probe-settings-category", categories[state.settings_page]):
            for index, label in enumerate(categories):
                clicked, _ = imgui.selectable(
                    f"{label}##probe-settings-category-{index}",
                    state.settings_page == index,
                )
                if clicked:
                    state.settings_page = index
            imgui.end_combo()
        imgui.spacing()
    else:
        nav_width = 132.0 * scale
        table_open = imgui.begin_table(
            "##probe-settings-layout",
            2,
            imgui.TableFlags_.borders_inner_v.value,
        )
        if not table_open:
            imgui.end_child()
            return
        imgui.table_setup_column(
            "categories",
            imgui.TableColumnFlags_.width_fixed.value,
            nav_width,
        )
        imgui.table_setup_column("page", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.push_style_var(
            imgui.StyleVar_.selectable_text_align,
            imgui.ImVec2(0.5, 0.5),
        )
        for index, label in enumerate(categories):
            clicked, _ = imgui.selectable(
                f"{label}##probe-settings-category-{index}", state.settings_page == index
            )
            if clicked:
                state.settings_page = index
        imgui.pop_style_var()
        imgui.table_next_column()

    page_open = imgui.begin_child(
        "##probe-settings-page",
        imgui.ImVec2(0.0, 0.0),
        imgui.ChildFlags_.always_use_window_padding.value,
    )
    if not page_open:
        imgui.end_child()
        if table_open:
            imgui.end_table()
        imgui.end_child()
        return
    imgui.set_next_item_width(-1.0)
    _, state.settings_filter = _preview_search_input(
        state,
        "##probe-settings-filter",
        state.settings_filter,
        search_tooltip="Search settings",
        clear_tooltip="Clear search",
    )
    imgui.separator()
    if state.settings_page == 0:
        _draw_settings_general(state)
    elif state.settings_page == 1:
        _draw_settings_interaction(state)
    elif state.settings_page == 2:
        _draw_settings_rendering(state)
    else:
        _draw_settings_mujoco(state)
    imgui.end_child()
    if table_open:
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
        playback_width = (end_padding(state) * 2.0 + state.overlay_center_step * 3.0) * scale
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
            show_limit_labels=False,
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
    _draw_command_icon(draw, center, "snapshot", color, size / 6.0, smoothing=draw.corner_smoothing)


def _draw_transport_button(
    draw: ImguiDraw2D, item_id: str, position, kind: str, scale: float
) -> None:
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(position[0]), float(position[1])))
    _command_button(item_id, kind, kind, CONCEPT_THEME, scale, smoothing=draw.corner_smoothing)


def _draw_keyframes(size, scale: float, state: ProbeState) -> None:
    """Exercise the production timeline against a real tiny simulation."""
    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.mujoco_adapter import MuJoCoAdapter
    from mojive.assets import resolve
    from mojive.session import Session

    if not imgui.begin_child("Keyframes###ProbeKeyframes", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    if state.timeline_session is None:
        state.timeline_session = Session(MuJoCoAdapter(resolve("joint_types")))
        session = state.timeline_session
        session.submit(cmd.Pause())
        for count in (0, 100, 160):
            if count:
                session.submit(cmd.Step(count))
                session.tick(FrameNeeds.none(), wall_dt=0)
            session.submit(cmd.CaptureSceneSnapshot())
    session = state.timeline_session
    session.tick(FrameNeeds.none(), wall_dt=min(0.05, imgui.get_io().delta_time))
    ctx = PanelContext(session, None, theme=CONCEPT_THEME, style_scale=scale)
    original_icon = keyframes_panel_module._draw_command_icon
    if state.preview_icon_library:
        keyframes_panel_module._draw_command_icon = lambda *args, **kwargs: (
            _draw_icon_library_command_icon(
                *args,
                context_scale=scale,
                state=state,
                **kwargs,
            )
        )
    try:
        state.timeline_panel.draw(ctx)
    finally:
        keyframes_panel_module._draw_command_icon = original_icon
    imgui.end_child()


def _draw_output(size, state: ProbeState, scale: float) -> None:
    if not imgui.begin_child("Output###ProbeOutput", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Output")
    imgui.separator()
    if state.output_buffer is None:
        state.output_buffer = OutputBuffer()
        for level, message in (
            ("info", "Loaded scene.xml"),
            ("warning", "Joint limit reached"),
            ("error", "Model could not be compiled"),
        ):
            state.output_buffer.write(message, level=level, timestamp="09:44:25")
    ctx = PanelContext(
        None, None, theme=CONCEPT_THEME, style_scale=scale, output=state.output_buffer
    )
    original_search = output_panel_module.search_input
    if state.preview_icon_library:
        padding = state.icon_padding_for("Panels")
        output_panel_module.search_input = lambda *args, **kwargs: search_input(
            *args,
            **kwargs,
            icon_drawer=lambda *values: _draw_concept_control_icon(
                *values,
                padding=padding,
                stroke_width=state.icon_stroke_width,
            ),
        )
    try:
        state.output_panel.draw(ctx)
    finally:
        output_panel_module.search_input = original_search
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


def _draw_search_header(
    state: ProbeState,
    item_id: str,
    hint: str,
    value: str,
    sort_by_name: bool,
    *,
    state_order: str,
) -> tuple[str, bool]:
    """Use the production searchable-list header with specimen state."""

    _changed, value, _sort_changed, sort_by_name = _preview_searchable_header(
        state,
        f"##{item_id}",
        value,
        sort_by_name,
        hint=hint,
        search_tooltip=hint,
        clear_tooltip="Clear search",
        state_order=state_order,
        translate=lambda text: text,
    )
    return value, sort_by_name


def _draw_copy_buttons(item_id: str, labels: tuple[str, str]) -> None:
    flags = _flags(
        imgui.TableFlags_.sizing_stretch_same,
        imgui.TableFlags_.no_pad_outer_x,
    )
    if not imgui.begin_table(f"##{item_id}-copy", 2, flags):
        return
    for label in labels:
        imgui.table_next_column()
        imgui.button(f"{label}##{item_id}", imgui.ImVec2(-1.0, 0.0))
    imgui.end_table()


def _draw_control_content(state: ProbeState) -> None:
    if imgui.collapsing_header("actuators", imgui.TreeNodeFlags_.default_open.value):
        state.control_filter, state.control_sort_by_name = _draw_search_header(
            state,
            "probe-actuator",
            "Search actuators",
            state.control_filter,
            state.control_sort_by_name,
            state_order="ctrl / act",
        )
        _draw_copy_buttons("probe-actuator-state", ("Copy ctrl", "Copy act"))
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

    if imgui.collapsing_header(
        "equality", imgui.TreeNodeFlags_.default_open.value
    ) and _begin_gallery_properties("##probe-control-equality"):
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


def _draw_control_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Control", "ProbeControl", size)
    if opened:
        _draw_control_content(state)
    imgui.end_child()


def _draw_joints_content(state: ProbeState) -> None:
    state.joint_filter, state.joint_sort_by_name = _draw_search_header(
        state,
        "probe-joint",
        "Search joints",
        state.joint_filter,
        state.joint_sort_by_name,
        state_order="qpos / qvel",
    )
    _draw_copy_buttons("probe-joint-state", ("Copy qpos", "Copy qvel"))
    if _begin_gallery_properties("##probe-all-joints"):
        _property_label("floating_base")
        _table_text("free · 6 dof", disabled=True)
        _property_label("shoulder_ball")
        _table_text("ball · 3 dof", disabled=True)
        _property_label("slide")
        _, state.slide_position = imgui.slider_float(
            "##probe-slide-position", state.slide_position, -0.34, 0.34, "%+.4f"
        )
        _property_label("hinge_limited")
        _, state.hinge_position = imgui.slider_float(
            "##probe-hinge-position", state.hinge_position, -1.2, 1.2, "%+.4f"
        )
        imgui.end_table()


def _draw_joints_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Joints", "ProbeJoints", size)
    if opened:
        _draw_joints_content(state)
    imgui.end_child()


def _draw_camera_content(state: ProbeState) -> None:
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
            icons=("persp", "ortho"),
            icon_drawer=(
                (
                    lambda *values: _draw_concept_projection_icon(
                        *values,
                        padding=state.icon_padding_for("Panels"),
                        stroke_width=state.icon_stroke_width,
                    )
                )
                if state.preview_icon_library
                else None
            ),
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


def _draw_camera_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Camera", "ProbeCamera", size)
    if opened:
        _draw_camera_content(state)
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
    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(draw_list, 2)
    splitter.set_current_channel(draw_list, 1)
    transparent = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
    imgui.push_style_color(imgui.Col_.button, transparent)
    imgui.push_style_color(imgui.Col_.button_hovered, transparent)
    imgui.push_style_color(imgui.Col_.button_active, transparent)
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
    imgui.push_style_color(imgui.Col_.frame_bg, transparent)
    imgui.push_style_color(imgui.Col_.frame_bg_hovered, transparent)
    imgui.push_style_color(imgui.Col_.frame_bg_active, transparent)
    imgui.begin_disabled(not editable)
    imgui.drag_float(f"##{item_id}-value", value, 0.01 if editable else 0.0, 0.0, 0.0, fmt)
    imgui.end_disabled()
    imgui.pop_style_color(3)
    field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    field_active = imgui.is_item_active()
    field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
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
            CONCEPT_THEME.bg_frame_active
            if field_active
            else CONCEPT_THEME.bg_frame_hovered
            if field_hovered and editable
            else CONCEPT_THEME.bg_frame
        ),
        rounding=float(imgui.get_style().frame_rounding),
        badge_opacity=float(imgui.get_style().alpha),
        field_opacity=float(imgui.get_style().alpha)
        * (1.0 if editable else float(imgui.get_style().disabled_alpha)),
    )
    splitter.merge(draw_list)


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
        _changed, state.hierarchy_filter = _preview_search_input(
            state,
            "##probe-hierarchy-filter",
            state.hierarchy_filter,
            hint="Search hierarchy",
            search_tooltip="Search hierarchy",
            clear_tooltip="Clear search",
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
                    ink = row_draw.text_ink_bounds("H")
                    center_y = (
                        round(text_y) + (ink[1] + ink[3]) * 0.5
                        if ink is not None
                        else (lo.y + hi.y) * 0.5
                    )
                    if state.preview_icon_library:
                        draw_concept_icon(
                            row_draw,
                            (node_x + 5 * scale, center_y),
                            10.0 * scale,
                            "panel-down" if disclosure == "▾" else "panel-right",
                            CONCEPT_THEME.text,
                            padding=state.icon_padding_for("Panels"),
                            stroke_width=state.icon_stroke_width,
                        )
                    else:
                        row_draw.fringed_concave_fill(
                            disclosure_triangle(
                                (node_x + 5 * scale, center_y),
                                4 * scale,
                                opened=disclosure == "▾",
                                smoothing=state.tool_smoothing,
                            ),
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
                if state.preview_icon_library:
                    draw_concept_icon(
                        row_draw,
                        center,
                        16.0 * scale,
                        "panel-visible" if state.hierarchy_visibility[index] else "panel-hidden",
                        color,
                        padding=state.icon_padding_for("Panels"),
                        stroke_width=state.icon_stroke_width,
                    )
                elif state.hierarchy_visibility[index]:
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
        if _begin_gallery_properties("##probe-asset-model"):
            _property_label("model")
            imgui.set_next_item_width(-1.0)
            imgui.combo("##probe-asset-model-name", 0, ("gizmo",))
            imgui.end_table()

        import_labels = ("Import Mesh...", "Import Height Field...", "Import Texture...")
        inline = button_row_layout(
            tuple(button_width(label) for label in import_labels),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        for index, label in enumerate(import_labels):
            if inline[index]:
                imgui.same_line()
            imgui.button(f"{label}##probe-asset-import-{index}")

        if _begin_gallery_properties("##probe-texture-type"):
            _property_label("texture type")
            imgui.set_next_item_width(-1.0)
            imgui.combo("##probe-texture-type-value", 0, ("2D", "Cube", "Skybox"))
            imgui.end_table()
        imgui.collapsing_header("Height-field import size")
        imgui.collapsing_header("New material")
        imgui.separator()
        imgui.set_next_item_width(-1.0)
        _changed, state.asset_filter = _preview_search_input(
            state,
            "##probe-asset-filter",
            state.asset_filter,
            hint="Filter assets...",
            search_tooltip="Search assets",
            clear_tooltip="Clear search",
        )
        imgui.set_next_item_width(-1.0)
        _changed, state.asset_type = imgui.combo(
            "##probe-asset-type", state.asset_type, ("All", "mesh", "texture", "material", "hfield")
        )
        list_height = min(150.0, max(86.0, imgui.get_content_region_avail().y * 0.48))
        if imgui.begin_child(
            "##probe-asset-list", imgui.ImVec2(0.0, list_height), imgui.ChildFlags_.borders.value
        ):
            flags = _flags(
                imgui.TableFlags_.sizing_stretch_prop,
                imgui.TableFlags_.row_bg,
                imgui.TableFlags_.no_saved_settings,
            )
            if imgui.begin_table("##probe-asset-table", 3, flags):
                imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch.value, 1.0)
                imgui.table_setup_column("Type", imgui.TableColumnFlags_.width_fixed.value)
                imgui.table_setup_column("Used", imgui.TableColumnFlags_.width_fixed.value)
                imgui.table_headers_row()
                for index, (name, kind, used) in enumerate(
                    (
                        ("robot_body", "mesh", "2"),
                        ("floor_albedo", "texture", "1"),
                        ("robot_metal", "material", "12"),
                        ("terrain", "hfield", "1"),
                    )
                ):
                    imgui.table_next_row()
                    imgui.table_next_column()
                    clicked, _ = imgui.selectable(
                        f"{name}##probe-asset-{index}",
                        state.asset_selection == index,
                        imgui.SelectableFlags_.span_all_columns.value,
                    )
                    if clicked:
                        state.asset_selection = index
                    imgui.table_next_column()
                    imgui.text_disabled(kind)
                    imgui.table_next_column()
                    imgui.text_disabled(used)
                imgui.end_table()
        imgui.end_child()
    imgui.end_child()


def _draw_stats_gallery(size) -> None:
    opened = _begin_gallery_panel("Stats", "ProbeStats", size)
    if opened:
        imgui.plot_lines(
            "##probe-frame-plot",
            PROBE_FRAME_SAMPLES,
            overlay_text="8.35 ms   119.8 fps",
            scale_min=0.0,
            scale_max=16.7,
            graph_size=imgui.ImVec2(-1.0, 60.0),
        )
        imgui.text_disabled("scale 0 .. 16.7 ms   (60 fps = 16.7 ms)")
        flags = _flags(imgui.TableFlags_.sizing_stretch_same, imgui.TableFlags_.row_bg)
        if imgui.begin_table("##probe-stats-counts", 2, flags):
            imgui.table_setup_column("metric", imgui.TableColumnFlags_.width_stretch.value, 1.0)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value, 1.0)
            for label, value in (
                ("draw calls", "148"),
                ("instances", "216"),
                ("triangles", "284,612"),
                ("buckets", "9"),
                ("frame cpu", "3.610 ms"),
            ):
                _property_label(label)
                _table_text(value)
            imgui.end_table()
        imgui.separator()
        if imgui.begin_table(
            "##probe-stats-passes",
            3,
            _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.row_bg),
        ):
            for heading in ("pass", "cpu ms", "gpu ms"):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for name, cpu, gpu in (("opaque", "1.842", "1.109"), ("overlay", "0.381", "0.214")):
                imgui.table_next_row()
                for cell in (name, cpu, gpu):
                    imgui.table_next_column()
                    imgui.text(cell)
            imgui.end_table()
    imgui.end_child()


def _draw_sensors_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Sensors", "ProbeSensors", size)
    if opened:
        imgui.set_next_item_width(-1.0)
        _changed, state.sensor_index = imgui.combo(
            "##probe-sensor", state.sensor_index, ("camera_grid", "imu_accel", "hinge_pos")
        )
        if _begin_gallery_properties("##probe-sensor-properties"):
            if state.sensor_index == 0:
                _property_label("type")
                _table_text("rangefinder")
                _property_label("dimension")
                _table_text("42")
            elif state.sensor_index == 1:
                _property_label("type")
                _table_text("accelerometer")
                _property_label("dimension")
                _table_text("3")
                _property_label("value")
                _table_text("[0.02, 0.01, 9.81]")
            else:
                _property_label("type")
                _table_text("jointpos")
                _property_label("dimension")
                _table_text("1")
                _property_label("value")
                _table_text("[0.350]")
            imgui.end_table()
        if state.sensor_index == 0:
            imgui.separator()
            imgui.text_disabled("value")
            imgui.button("Copy")
            imgui.same_line()
            imgui.button("Open in Plot")
            if imgui.begin_child(
                "##probe-sensor-values", imgui.ImVec2(0.0, 112.0), imgui.ChildFlags_.borders.value
            ):
                imgui.text_wrapped(
                    "[ 2.410969, -0.774737, 0.387369, 0.640125, 1.129701,  ... ,\n"
                    "  1.884203, 2.004288, 2.101055 ]"
                )
            imgui.end_child()
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
            lambda: _draw_sensors_gallery(lower_sizes[3], state),
        )
    ):
        draw_panel()
        if index != 3:
            imgui.same_line()


def _draw_workspace_right_dock(size, state: ProbeState) -> None:
    if not imgui.begin_child(
        "Right dock###ProbeWorkspaceRightDock",
        size,
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return
    active = _wrapped_tabs(
        (
            ("Control", "right-dock", "control"),
            ("Joints", "right-dock", "joints"),
            ("Camera", "right-dock", "camera"),
        ),
        state.workspace_right_tab,
        imgui.get_content_region_avail().x,
    )
    state.workspace_right_tab = active
    if active == "Control":
        _draw_control_content(state)
    elif active == "Joints":
        _draw_joints_content(state)
    else:
        _draw_camera_content(state)
    imgui.end_child()


def _draw_panel_page(available, state: ProbeState, scale: float) -> None:
    flags = _flags(
        imgui.WindowFlags_.horizontal_scrollbar,
        imgui.WindowFlags_.always_vertical_scrollbar,
    )
    if not imgui.begin_child(
        "Panel gallery canvas###ProbePanelGalleryCanvas",
        available,
        imgui.ChildFlags_.none.value,
        flags,
    ):
        imgui.end_child()
        return
    cursor = imgui.get_cursor_pos()
    width, height = _virtual_canvas_size(available, scale, PANEL_CANVAS_SIZE)
    _draw_panel_gallery(imgui.ImVec2(width, height), state, scale)
    imgui.set_cursor_pos(imgui.ImVec2(cursor.x + width - 1.0, cursor.y + height - 1.0))
    imgui.dummy(imgui.ImVec2(1.0, 1.0))
    imgui.end_child()


def _probe_status_hints(
    variant: str,
    *,
    selected: bool,
    selection_clear: bool = True,
) -> tuple[ToolHint, ...]:
    """Compose the production status grammar for deterministic specimens."""

    hints = default_tool_hints(variant, DEFAULT_INPUT_BINDINGS)
    if variant != "ready_minimal":
        hints = tuple(hint for hint in hints if hint.hint_id != "gizmo.type_value")
    if selected and selection_clear:
        hints = (
            ToolHint("key", "Esc", "Clear selection", hint_id="selection.clear"),
            *hints,
        )
    return (
        ToolHint("key", "Backspace", "Rewind", hint_id="playback.previous"),
        *hints,
    )


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
    selected_item = selected.strip() or "No selection"
    has_selection = selected_item != "No selection"
    variant = "ready" if has_selection else "camera"
    hints = _probe_status_hints(variant, selected=has_selection)
    draw_status(
        draw,
        origin,
        width,
        height,
        CONCEPT_THEME,
        scale,
        selected=selected_item,
        state="running" if running else "paused",
        sim_time=1.204,
        step=1204,
        metric_mode="time",
        backend="OpenGL",
        dt=0.002,
        fps=rate,
        tool_hints=hints,
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
    _draw_settings(
        imgui.ImVec2(available.x - shell_width - spacing.x, available.y),
        state,
        scale,
    )


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
        ("Paused · no selection", "No selection", False, "119.8"),
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
    imgui.plot_lines("##probe-plot", PROBE_PLOT_SAMPLES, graph_size=imgui.ImVec2(-1.0, -1.0))
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
            ("Backspace", "Previous frame · hold to rewind"),
            ("Esc", "Clear selection"),
            ("Shift", "Snap"),
            ("T", "World / Body"),
            ("Double-click item", "Focus item"),
            ("Double-click gizmo", "Type value after hover hint"),
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
            ("Viewer", "mojive"),
            ("Backend", "OpenGL / OpenGL"),
            ("Scene source", "MuJoCo"),
            ("Document", "joint_types.xml"),
        ):
            _property_label(label)
            _table_text(value)
        imgui.end_table()
    imgui.end_child()


def _draw_workspaces_tab(available, scale: float, state: ProbeState) -> None:
    keyframe_height = min(320.0 * scale, max(240.0 * scale, available.y * 0.34))
    _draw_keyframes(imgui.ImVec2(available.x, keyframe_height), scale, state)
    imgui.text_disabled("M15 · full-width bottom dock; transport uses compact icon groups")
    active = _wrapped_tabs(
        (
            ("Output", "aux", "output"),
            ("Plot", "aux", "plot"),
            ("Help", "aux", "help"),
            ("Info", "aux", "info"),
        ),
        state.aux_tab,
        imgui.get_content_region_avail().x,
    )
    state.aux_tab = active
    remaining = imgui.get_content_region_avail()
    if active == "Output":
        _draw_output(remaining, state, scale)
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
    scale: float,
    *,
    vertical: bool = False,
) -> None:
    color = (*CONCEPT_THEME.text_disabled[:3], 0.9)
    draw.line(start, end, color, 1.0 * scale)
    if vertical:
        draw.line(
            (start[0] - 5.0 * scale, start[1]),
            (start[0] + 5.0 * scale, start[1]),
            color,
            1.0 * scale,
        )
        draw.line(
            (end[0] - 5.0 * scale, end[1]),
            (end[0] + 5.0 * scale, end[1]),
            color,
            1.0 * scale,
        )
        draw.text(
            (start[0] + 9.0 * scale, (start[1] + end[1]) * 0.5 - 7.0 * scale),
            color,
            label,
        )
    else:
        draw.line(
            (start[0], start[1] - 5.0 * scale),
            (start[0], start[1] + 5.0 * scale),
            color,
            1.0 * scale,
        )
        draw.line(
            (end[0], end[1] - 5.0 * scale),
            (end[0], end[1] + 5.0 * scale),
            color,
            1.0 * scale,
        )
        width, _ = draw.text_size(label)
        draw.text(
            ((start[0] + end[0] - width) * 0.5, start[1] + 8.0 * scale),
            color,
            label,
        )


def _even_slider(item_id: str, value: int, minimum: int, maximum: int) -> int:
    changed, candidate = imgui.slider_int(item_id, value, minimum, maximum)
    if not changed:
        return value
    snapped = int(round(candidate / 2.0) * 2)
    return max(minimum, min(maximum, snapped))


def _geometry_values_text(state: ProbeState) -> str:
    """Return the live component experiment as reviewable production fields."""

    values = (
        ("imgui_rounding", state.imgui_rounding),
        ("capsule_outline", repr(state.capsule_outline)),
        ("icon_stroke", state.icon_stroke_width),
        ("icon_radius", state.overlay_icon_radius),
        ("radial_step", state.overlay_radial_step),
        ("center_step", state.overlay_center_step),
        ("tool_group_gap", state.tool_group_gap),
        ("divider_width", state.divider_width),
        ("tool_stroke", state.tool_stroke_width),
        ("rotate_ring_gap_ratio", state.rotate_ring_gap_ratio),
        ("rotate_ring_cap", repr(state.rotate_ring_cap)),
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
    values += tuple((name, getattr(state, name)) for _, name in CORNER_CONTROLS)
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
                    draw_concept_icon(
                        item_draw,
                        center,
                        42.0 * scale,
                        f"playback-{kind}",
                        CONCEPT_THEME.text,
                        padding=state.icon_padding_for("Viewport playback"),
                        stroke_width=state.icon_stroke_width,
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
                    draw_concept_icon(
                        item_draw,
                        center,
                        47.2 * scale,
                        "tool-scale" if kind == "dimensions" else f"tool-{kind}",
                        CONCEPT_THEME.text,
                        padding=state.icon_padding_for("Viewport tools"),
                        stroke_width=state.icon_stroke_width,
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
        padding = state.icon_padding_for(group)
        _property_label("Glyph padding")
        changed, padding = imgui.slider_float(
            "##geometry-glyph-padding",
            padding,
            ICON_MIN_CLEARANCE,
            ICON_MAX_PADDING,
            "%.2f u",
            imgui.SliderFlags_.always_clamp.value,
        )
        if changed:
            state.set_icon_padding_for(group, padding)
        imgui.set_item_tooltip(
            "Circular clearance from the candidate to its icon slot. Rotate and the reviewed "
            "Info/Warning/Error family remain fixed."
        )
        _property_label("Glyph stroke")
        _, state.icon_stroke_width = imgui.slider_float(
            "##geometry-glyph-stroke",
            state.icon_stroke_width,
            ICON_MIN_STROKE,
            ICON_MAX_STROKE,
            "%.2f u",
            imgui.SliderFlags_.always_clamp.value,
        )
        imgui.set_item_tooltip(
            "Shared visual weight for ordinary strokes, chevrons, Reset arcs, and tool shafts. "
            "Reviewed Pause, First/Last, severity, and mouse geometry remain locked."
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
        state.icon_stroke_width = ICON_STROKE
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
        state.hint_mouse_stroke = OVERLAY_GEOMETRY.hint_mouse_stroke
        state.hint_mouse_button_width_ratio = OVERLAY_GEOMETRY.hint_mouse_button_width_ratio
        state.hint_mouse_button_shell_ratio = OVERLAY_GEOMETRY.hint_mouse_button_shell_ratio
        state.hint_mouse_button_height_ratio = OVERLAY_GEOMETRY.hint_mouse_button_height_ratio
        state.hint_mouse_wheel_width_ratio = OVERLAY_GEOMETRY.hint_mouse_wheel_width_ratio
        state.hint_mouse_wheel_height_ratio = OVERLAY_GEOMETRY.hint_mouse_wheel_height_ratio
        state.hint_mouse_wheel_gap_ratio = OVERLAY_GEOMETRY.hint_mouse_wheel_gap_ratio

    imgui.end_child()


def _draw_diagnostic_gallery(state, scale):
    """Inspect the production severity paths and control rows at several sizes."""
    ctx = PanelContext(None, None, theme=CONCEPT_THEME, style_scale=scale)
    draw = ImguiDraw2D()
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
    imgui.text("Slider states: normal / hover / press")
    for index, (label, hovered, pressed) in enumerate(
        (("Normal", False, False), ("Hover", True, False), ("Press", True, True))
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


_ICON_REVIEW_SIZES = (14.0, 24.0, 56.0, 112.0)


def _icon_library_canvas_size(family: str) -> tuple[float, float]:
    if family in {"Overview", "UI context"}:
        return GEOMETRY_CANVAS_SIZE
    if family == "Capsules":
        return GEOMETRY_CANVAS_SIZE[0], 1260.0
    rows = len(icon_family(family))
    required_height = 260.0 + (max(_ICON_REVIEW_SIZES) + 20.0) * rows
    return GEOMETRY_CANVAS_SIZE[0], max(GEOMETRY_CANVAS_SIZE[1], required_height)


def _geometry_canvas_size(active_tab: str, state: ProbeState) -> tuple[float, float]:
    """Reserve the logical extent required by zoomed geometry specimens."""

    width, height = GEOMETRY_CANVAS_SIZE
    if active_tab == "Playback":
        shell_radius = state.overlay_icon_radius + 2.0 * state.overlay_radial_step
        inspection = state.construction_playback_scale
        _centers, length = capsule_layout(6, (3, 4), state)
        width = max(width, 54.0 + length * inspection + 54.0 + 360.0 + 24.0)
        construction_bottom = 80.0 + 4.0 * shell_radius * inspection + 62.0
        height = max(height, construction_bottom + 1010.0)
    elif active_tab == "Tools":
        _centers, length = capsule_layout(5, (3,), state)
        height = max(height, 120.0 + length * state.construction_tool_scale)
    return width, height


def _concept_icon_color(name: str):
    if name == "status-info":
        return CONCEPT_THEME.info
    if name == "status-warning":
        return CONCEPT_THEME.warning
    if name == "status-error":
        return CONCEPT_THEME.danger
    return CONCEPT_THEME.text


def _draw_concept_icon_specimen(
    draw,
    center,
    size: float,
    name: str,
    scale: float,
    padding: float = ICON_DEFAULT_PADDING,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
) -> None:
    """Draw one candidate inside the shared circular placement boundary."""

    guide_radius = size * ICON_BOUND_DIAMETER / ICON_GRID * 0.5
    if name.startswith("status-") and name.removeprefix("status-") in {
        "info",
        "warning",
        "error",
    }:
        # Output already has a reviewed production family. Show that exact
        # painter here so the concept library cannot silently drift from it.
        severity_icon(
            draw,
            center,
            size,
            name.removeprefix("status-"),
            _concept_icon_color(name),
        )
    else:
        draw_concept_icon(
            draw,
            center,
            size,
            name,
            _concept_icon_color(name),
            padding=padding,
            accent_color=(CONCEPT_THEME.primary if name.startswith("status-mouse-") else None),
            mouse_width=mouse_width,
            stroke_width=stroke_width,
        )
    # Draw both placement guides last so a circular collision and a displaced
    # axis-aligned bounding box remain visible instead of hiding below the
    # glyph. The square has exactly the orange circle's diameter.
    draw.rect(
        (center[0] - guide_radius, center[1] - guide_radius),
        (center[0] + guide_radius, center[1] + guide_radius),
        (*CONCEPT_THEME.text_disabled[:3], 0.34),
        max(0.6, 0.72 * scale),
    )
    draw.circle(
        center,
        guide_radius,
        (*CONCEPT_THEME.warning[:3], 0.72),
        max(0.75, 0.9 * scale),
        segments=max(32, round(guide_radius * 4.0)),
    )


def _icon_review_metrics(
    name: str,
    padding: float = ICON_DEFAULT_PADDING,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
) -> tuple[float, float, float, float, float]:
    """Return circle clearance plus enclosing-circle and box centers."""

    if name.startswith("status-") and name.removeprefix("status-") in {
        "info",
        "warning",
        "error",
    }:
        meshes = severity_meshes(ICON_GRID, name.removeprefix("status-"))
        points = np.concatenate([np.asarray(mesh[0], np.float64) for mesh in meshes])
        bounds = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )
        enclosing_center, enclosing_radius = minimum_enclosing_circle(points)
        return (
            ICON_BOUND_DIAMETER * 0.5 - enclosing_radius,
            enclosing_center[0],
            enclosing_center[1],
            (bounds[0] + bounds[2]) * 0.5,
            (bounds[1] + bounds[3]) * 0.5,
        )
    metrics = icon_metrics(
        name,
        padding=padding,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
    )
    box_x, box_y = metrics.center_offset
    return (
        metrics.circular_clearance,
        metrics.enclosing_center[0],
        metrics.enclosing_center[1],
        box_x,
        box_y,
    )


def _draw_icon_library_overview(
    draw,
    origin,
    scale: float,
    state: ProbeState,
) -> None:
    card_width = 468.0 * scale
    card_height = 244.0 * scale
    gap_x = 20.0 * scale
    gap_y = 18.0 * scale
    clip_min = imgui.get_window_draw_list().get_clip_rect_min()
    clip_max = imgui.get_window_draw_list().get_clip_rect_max()
    for family_index, (family, icons) in enumerate(ICON_FAMILIES):
        padding = state.icon_padding_for(family)
        column = family_index % 3
        row = family_index // 3
        x0 = origin[0] + column * (card_width + gap_x)
        y0 = origin[1] + row * (card_height + gap_y)
        x1, y1 = x0 + card_width, y0 + card_height
        if y1 < clip_min.y or y0 > clip_max.y:
            continue
        draw.rect_filled((x0, y0), (x1, y1), CONCEPT_THEME.bg_child, rounding=8.0 * scale)
        draw.rect(
            (x0, y0),
            (x1, y1),
            CONCEPT_THEME.border,
            max(0.75, scale),
            rounding=8.0 * scale,
        )
        draw.text((x0 + 16.0 * scale, y0 + 12.0 * scale), CONCEPT_THEME.text, family)
        draw.text(
            (x0 + 16.0 * scale, y0 + 34.0 * scale),
            CONCEPT_THEME.text_disabled,
            f"{len(icons)} candidates · 20 pt",
        )

        columns = min(5, len(icons))
        cell_width = (card_width - 28.0 * scale) / columns
        for icon_index, (label, name) in enumerate(icons):
            icon_column = icon_index % columns
            icon_row = icon_index // columns
            center = (
                x0 + 14.0 * scale + cell_width * (icon_column + 0.5),
                y0 + (86.0 + icon_row * 76.0) * scale,
            )
            _draw_concept_icon_specimen(
                draw,
                center,
                20.0 * scale,
                name,
                scale,
                padding,
                state.concept_mouse_width(),
                state.icon_stroke_width,
            )
            draw.centered_label(
                label,
                (center[0], center[1] + 29.0 * scale),
                CONCEPT_THEME.text_disabled,
                cell_width - 8.0 * scale,
            )


def _draw_context_row(
    draw,
    origin,
    width: float,
    row_height: float,
    icon_size: float,
    icon_name: str,
    label: str,
    detail: str,
    scale: float,
    padding: float,
    stroke_width: float = ICON_STROKE,
    *,
    production_kind: str | None = None,
) -> None:
    """Place one candidate in current Mojive row metrics with text guides."""

    x0, y0 = float(origin[0]), float(origin[1])
    width *= scale
    row_height *= scale
    icon_size *= scale
    center_y = y0 + row_height * 0.5
    draw.rect_filled(
        (x0, y0),
        (x0 + width, y0 + row_height),
        CONCEPT_THEME.bg_frame,
        rounding=5.0 * scale,
    )
    icon_center = (x0 + 22.0 * scale, center_y)
    text_x = x0 + 40.0 * scale
    text_y = text_line_y(draw, center_y)
    reference = draw.text_ink_bounds("H")
    cap_y = text_y + (reference[1] if reference else 0.0)
    baseline_y = text_y + float(imgui.get_font_baked().ascent)
    guide_right = x0 + width - 250.0 * scale
    draw.line(
        (x0 + 8.0 * scale, center_y),
        (guide_right, center_y),
        (*CONCEPT_THEME.warning[:3], 0.18),
        max(0.5, 0.7 * scale),
    )
    draw.line(
        (text_x, cap_y),
        (guide_right, cap_y),
        (*CONCEPT_THEME.primary[:3], 0.48),
        max(0.5, 0.7 * scale),
    )
    draw.line(
        (text_x, baseline_y),
        (guide_right, baseline_y),
        (*CONCEPT_THEME.info[:3], 0.52),
        max(0.5, 0.7 * scale),
    )
    if production_kind is None:
        draw_concept_icon(
            draw,
            icon_center,
            icon_size,
            icon_name,
            CONCEPT_THEME.text,
            padding=padding,
            stroke_width=stroke_width,
        )
    else:
        _draw_command_icon(draw, icon_center, production_kind, CONCEPT_THEME.text, scale)
    draw.circle(
        icon_center,
        icon_size * 0.5,
        (*CONCEPT_THEME.warning[:3], 0.72),
        max(0.75, 0.9 * scale),
        segments=max(32, round(icon_size * 2.0)),
    )
    draw.text((text_x, text_y), CONCEPT_THEME.text, label, pixel_snap=False)
    draw.text(
        (x0 + width - 230.0 * scale, text_y),
        CONCEPT_THEME.text_disabled,
        detail,
        pixel_snap=False,
    )


def _draw_icon_context_page(
    draw,
    origin,
    scale: float,
    state: ProbeState,
) -> None:
    """Review candidate alignment in Mojive's actual row and font metrics."""

    x0, y0 = float(origin[0]), float(origin[1])
    padding = state.icon_padding_for("Keyframes")
    draw.text((x0, y0 + 10.0 * scale), CONCEPT_THEME.text, "Alignment in UI")
    draw.text(
        (x0, y0 + 34.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Candidates use current Mojive row heights and font metrics; orange guides are review-only.",
    )
    draw.text(
        (x0, y0 + 58.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Green = H cap line · blue = text baseline · faint amber = row and icon-slot center",
    )

    card_x = x0 - 10.0 * scale
    card_width = 1370.0 * scale
    first_y = y0 + 96.0 * scale
    draw.rect_filled(
        (card_x, first_y),
        (card_x + card_width, first_y + 250.0 * scale),
        (*CONCEPT_THEME.bg_frame[:3], 0.55),
        rounding=6.0 * scale,
    )
    draw.text(
        (x0 + 10.0 * scale, first_y + 14.0 * scale),
        CONCEPT_THEME.text,
        "Camera master · one minimum enclosing-circle center",
    )
    draw.text(
        (x0 + 10.0 * scale, first_y + 38.0 * scale),
        CONCEPT_THEME.text_disabled,
        "key-snapshot and helper-camera call the same contour and use the same circle center.",
    )
    samples = (
        ("Keyframes", state.icon_padding_for("Keyframes")),
        ("Scene helpers", state.icon_padding_for("Scene helpers")),
    )
    for index, (label, sample_padding) in enumerate(samples):
        center = (x0 + (390.0 + index * 520.0) * scale, first_y + 132.0 * scale)
        draw_concept_icon(
            draw,
            center,
            112.0 * scale,
            "key-snapshot",
            CONCEPT_THEME.text,
            padding=sample_padding,
            stroke_width=state.icon_stroke_width,
        )
        draw.circle(
            center,
            56.0 * scale,
            (*CONCEPT_THEME.warning[:3], 0.72),
            max(0.75, 0.9 * scale),
            segments=max(64, round(112.0 * scale * 2.0)),
        )
        draw.centered_label(
            label,
            (center[0], first_y + 215.0 * scale),
            CONCEPT_THEME.text_disabled,
            360.0 * scale,
        )

    second_y = first_y + 274.0 * scale
    draw.text((x0, second_y), CONCEPT_THEME.text, "Current UI metrics")
    draw.text(
        (x0, second_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "The current Keyframes command still uses a diamond; camera rows below show candidate placement before production wiring.",
    )
    row_x = x0 + 10.0 * scale
    row_width = 1320.0
    _draw_context_row(
        draw,
        (row_x, second_y + 62.0 * scale),
        row_width,
        28.0,
        16.0,
        "key-keyframe",
        "Capture snapshot",
        "production · 28 pt row / 16 pt slot",
        scale,
        padding,
        state.icon_stroke_width,
        production_kind="key",
    )
    _draw_context_row(
        draw,
        (row_x, second_y + 112.0 * scale),
        row_width,
        28.0,
        16.0,
        "key-snapshot",
        "Capture snapshot",
        "candidate · circle anchor",
        scale,
        padding,
        state.icon_stroke_width,
    )
    helper_padding = state.icon_padding_for("Scene helpers")
    _draw_context_row(
        draw,
        (row_x, second_y + 162.0 * scale),
        row_width,
        26.0,
        14.0,
        "helper-camera",
        "camera0",
        "candidate · 26 pt hierarchy row",
        scale,
        helper_padding,
        state.icon_stroke_width,
    )
    _draw_context_row(
        draw,
        (row_x, second_y + 212.0 * scale),
        row_width,
        26.0,
        14.0,
        "helper-light",
        "light0",
        "candidate · 26 pt hierarchy row",
        scale,
        helper_padding,
        state.icon_stroke_width,
    )

    note_y = second_y + 270.0 * scale
    draw.text((x0, note_y), CONCEPT_THEME.text, "Review rule")
    draw.text(
        (x0, note_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "First align the slot center to the H cap-height center and text baseline, then apply the glyph's declared geometric anchor.",
    )
    draw.text(
        (x0, note_y + 48.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Directional edge marks use their box; Reset uses its ring; remaining candidates use their minimum circle.",
    )


def _draw_icon_family_detail(
    draw,
    origin,
    family: str,
    scale: float,
    padding: float,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
) -> None:
    icons = icon_family(family)
    centers = tuple(origin[0] + offset * scale for offset in (330.0, 535.0, 765.0, 1085.0))
    header_y = origin[1] + 10.0 * scale
    draw.text((origin[0], header_y), CONCEPT_THEME.text, family)
    draw.text(
        (origin[0], header_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Orange circle and gray square share one center · each glyph reports its declared anchor",
    )
    for center_x, size in zip(centers, _ICON_REVIEW_SIZES, strict=True):
        draw.centered_label(
            f"{int(size)} pt",
            (center_x, header_y + 62.0 * scale),
            CONCEPT_THEME.text_disabled,
            90.0 * scale,
        )

    rows_y = header_y + 98.0 * scale
    row_height = (max(_ICON_REVIEW_SIZES) + 20.0) * scale
    right = origin[0] + 1370.0 * scale
    clip_min = imgui.get_window_draw_list().get_clip_rect_min()
    clip_max = imgui.get_window_draw_list().get_clip_rect_max()
    for row, (label, name) in enumerate(icons):
        center_y = rows_y + row_height * row + row_height * 0.5
        row_top = center_y - row_height * 0.5
        row_bottom = center_y + row_height * 0.5
        if row_bottom < clip_min.y or row_top > clip_max.y:
            continue
        if row % 2 == 0:
            draw.rect_filled(
                (origin[0] - 10.0 * scale, row_top),
                (right, row_bottom),
                (*CONCEPT_THEME.bg_frame[:3], 0.55),
                rounding=5.0 * scale,
            )
        draw.text(
            (origin[0] + 10.0 * scale, center_y - imgui.get_text_line_height() * 0.5),
            CONCEPT_THEME.text,
            label,
        )
        for center_x, size in zip(centers, _ICON_REVIEW_SIZES, strict=True):
            _draw_concept_icon_specimen(
                draw,
                (center_x, center_y),
                size * scale,
                name,
                scale,
                padding,
                mouse_width,
                stroke_width,
            )

        meta_x = origin[0] + 1160.0 * scale
        clearance, circle_x, circle_y, box_x, box_y = _icon_review_metrics(
            name, padding, mouse_width, stroke_width
        )
        anchor = icon_alignment_anchor(name)
        anchor_x, anchor_y = icon_alignment_center(
            name, padding, mouse_width, stroke_width
        )
        diagnostic_label, diagnostic_x, diagnostic_y = (
            ("box", box_x, box_y) if anchor == "circle" else ("circle", circle_x, circle_y)
        )
        placement_label, placement_value = (
            ("frame", ROTATE_FRAME_PADDING) if name == "tool-rotate" else ("pad", clearance)
        )
        for line, line_y in (
            (f"{name} · {placement_label} {placement_value:.2f}u", -20.0),
            (f"{anchor}  {anchor_x:+.2f},{anchor_y:+.2f}u", 0.0),
            (f"{diagnostic_label}  {diagnostic_x:+.2f},{diagnostic_y:+.2f}u", 20.0),
        ):
            draw.text(
                (meta_x, center_y + line_y * scale),
                CONCEPT_THEME.text_disabled,
                line,
            )


def _draw_capsule_context_page(draw, origin, scale: float, state: ProbeState) -> None:
    """Show Icon Library candidates in their actual capsule cells and state circles."""

    x0, y0 = origin
    draw.text((x0, y0), CONCEPT_THEME.text, "Actual capsule placement")
    draw.text(
        (x0, y0 + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Amber = icon slot · green = state circle · each component group keeps its own Padding",
    )

    def preview_state(**changes):
        return replace(
            state,
            redesign=replace(state.redesign),
            preview_icon_library=True,
            show_icon_bounds=True,
            show_state_circles=True,
            show_construction_notes=False,
            **changes,
        )

    playback_x = x0 + 24.0 * scale
    playback_y = y0 + 82.0 * scale
    draw.text((playback_x, playback_y - 28.0 * scale), CONCEPT_THEME.text, "Playback · 1×")
    imgui.push_id("icon-library-capsule-playback-1x")
    _draw_playback(draw, (playback_x, playback_y), scale, preview_state(playing=False))
    imgui.pop_id()

    playback_2x_y = playback_y + 116.0 * scale
    draw.text(
        (playback_x, playback_2x_y - 28.0 * scale),
        CONCEPT_THEME.text,
        "Playback · 2× · playing",
    )
    imgui.push_id("icon-library-capsule-playback-2x")
    _draw_playback(draw, (playback_x, playback_2x_y), scale * 2.0, preview_state(playing=True))
    imgui.pop_id()

    recording_y = playback_2x_y + 140.0 * scale
    draw.text(
        (playback_x, recording_y - 28.0 * scale),
        CONCEPT_THEME.text,
        "Playback · 1× · recording",
    )
    recording_state = preview_state(playing=True)
    recording_state.redesign.recording = "video"
    imgui.push_id("icon-library-capsule-playback-recording")
    _draw_playback(draw, (playback_x, recording_y), scale, recording_state)
    imgui.pop_id()

    metrics_x = x0 + 820.0 * scale
    draw.text((metrics_x, playback_y - 28.0 * scale), CONCEPT_THEME.text, "Placement diagnostics")
    draw.text(
        (metrics_x, playback_y - 4.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Declared anchor center after placement; every value must be 0.00, 0.00",
    )
    playback_padding = state.icon_padding_for("Viewport playback")
    for index, name in enumerate(
        (
            "playback-previous",
            "playback-play",
            "playback-pause",
            "playback-next",
            "playback-reset",
            "playback-record",
            "playback-stop",
            "playback-more",
        )
    ):
        anchor = icon_alignment_anchor(name)
        anchor_center = icon_alignment_center(
            name,
            playback_padding,
            stroke_width=state.icon_stroke_width,
        )
        draw.text(
            (metrics_x, playback_y + (30.0 + index * 25.0) * scale),
            CONCEPT_THEME.text_disabled,
            (f"{name:<20} {anchor:<6} {anchor_center[0]:+0.2f}, {anchor_center[1]:+0.2f}u"),
        )

    semantic_y = playback_y + 252.0 * scale
    draw.text((metrics_x, semantic_y), CONCEPT_THEME.text, "Applied semantic states")
    semantic_specs = (
        ("Default", "playback-play", "off", False),
        ("Hover", "playback-play", "hover", False),
        ("Press", "playback-next", "press", False),
        ("Selected", "playback-pause", "selected", False),
        ("Record", "playback-record", "off", True),
        ("Stop", "playback-stop", "selected", True),
    )
    for index, (label, name, interaction, danger) in enumerate(semantic_specs):
        position = (metrics_x + index * 92.0 * scale, semantic_y + 28.0 * scale)

        def semantic_icon(
            target,
            center,
            color,
            icon_scale,
            _surface,
            *,
            name=name,
            danger=danger,
        ):
            draw_concept_icon(
                target,
                center,
                20.0 * icon_scale,
                name,
                THEME.viewport.record if danger else color,
                padding=playback_padding,
                stroke_width=state.icon_stroke_width,
            )

        _circular_icon_button(
            draw,
            f"##capsule-semantic-{index}",
            position,
            semantic_icon,
            selected=interaction == "selected",
            cell_size=42.0,
            state_radius=18.0,
            icon_radius=10.0,
            icon_scale=scale,
            forced_interaction=interaction,
            scale=scale,
        )
        draw.centered_label(
            label,
            (position[0] + 21.0 * scale, semantic_y + 80.0 * scale),
            CONCEPT_THEME.text_disabled,
            82.0 * scale,
        )

    tools_y = y0 + 590.0 * scale
    draw.text((playback_x, tools_y), CONCEPT_THEME.text, "Viewport tools · actual vertical capsule")
    draw.text(
        (playback_x, tools_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "World and Body use the same cell center; their internal strokes stop at the shell's inner edge.",
    )
    for index, (space, label) in enumerate((("world", "World"), ("body", "Body"))):
        tool_x = playback_x + index * 250.0 * scale
        tool_y = tools_y + 72.0 * scale
        draw.text((tool_x, tool_y - 26.0 * scale), CONCEPT_THEME.text_disabled, label)
        imgui.push_id(f"icon-library-capsule-tools-{space}")
        _draw_tool_column(
            draw,
            (tool_x, tool_y),
            scale * 1.5,
            preview_state(gizmo_space=space),
        )
        imgui.pop_id()

    draw.text(
        (metrics_x, tools_y),
        CONCEPT_THEME.text,
        "Application rules",
    )
    draw.text(
        (metrics_x, tools_y + 28.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Default uses Text; hover, press, and selected use Primary Bright.",
    )
    draw.text(
        (metrics_x, tools_y + 54.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Record and Stop keep the Danger red foreground in every interaction state.",
    )
    draw.text(
        (metrics_x, tools_y + 80.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Capsule cell, state circle, and each glyph's declared anchor coincide.",
    )


def _draw_icon_library_page(
    draw, origin, available_width: float, scale: float, state: ProbeState
) -> None:
    """Render concept-only icon families without changing production painters."""

    x0, y0 = float(origin[0]), float(origin[1])
    draw.text((x0 + 42.0 * scale, y0 + 2.0 * scale), CONCEPT_THEME.text, "Mojive icon library")
    draw.text(
        (x0 + 42.0 * scale, y0 + 25.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Candidate vectors · 24-unit circular placement bound",
    )
    controls_x = x0 + max(450.0 * scale, available_width - 940.0 * scale)
    draw.text(
        (controls_x, y0 + 17.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Group",
    )
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(controls_x + 58.0 * scale), float(y0 + 8.0 * scale))
    )
    imgui.set_next_item_width(190.0 * scale)
    group = _draw_icon_group_combo("##icon-library-group", state)
    padding = state.icon_padding_for(group)
    padding_x = controls_x + 268.0 * scale
    draw.text(
        (padding_x, y0 + 17.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Glyph pad",
    )
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(padding_x + 72.0 * scale), float(y0 + 8.0 * scale))
    )
    imgui.set_next_item_width(130.0 * scale)
    changed, padding = imgui.slider_float(
        "##icon-library-padding",
        padding,
        ICON_MIN_CLEARANCE,
        ICON_MAX_PADDING,
        "%.2f u",
        imgui.SliderFlags_.always_clamp.value,
    )
    if changed:
        state.set_icon_padding_for(group, padding)
    imgui.set_item_tooltip(
        "Circular clearance between candidate geometry and the orange placement circle. "
        "Rotate and the reviewed Info/Warning/Error family remain locked."
    )
    stroke_x = controls_x + 488.0 * scale
    draw.text(
        (stroke_x, y0 + 17.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Glyph stroke",
    )
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(stroke_x + 90.0 * scale), float(y0 + 8.0 * scale))
    )
    imgui.set_next_item_width(122.0 * scale)
    _, state.icon_stroke_width = imgui.slider_float(
        "##icon-library-stroke",
        state.icon_stroke_width,
        ICON_MIN_STROKE,
        ICON_MAX_STROKE,
        "%.2f u",
        imgui.SliderFlags_.always_clamp.value,
    )
    imgui.set_item_tooltip(
        "One final visual weight for ordinary strokes, chevrons, Reset arcs, and tool shafts. "
        "Reviewed Pause, First/Last, severity, and mouse geometry stay unchanged."
    )
    if group == "Status & input":
        mouse_x = controls_x + 718.0 * scale
        draw.text(
            (mouse_x, y0 + 17.0 * scale),
            CONCEPT_THEME.text_disabled,
            "Mouse W",
        )
        imgui.set_cursor_screen_pos(
            imgui.ImVec2(float(mouse_x + 80.0 * scale), float(y0 + 8.0 * scale))
        )
        imgui.set_next_item_width(135.0 * scale)
        state.hint_mouse_width = _even_slider(
            "##icon-library-mouse-width", state.hint_mouse_width, 12, 24
        )
        imgui.set_item_tooltip(
            "Visible mouse width used by Status & input specimens and whole-UI previews."
        )
    imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 42.0 * scale, y0 + 54.0 * scale))
    state.icon_library_tab = _wrapped_tabs(
        tuple(
            (label, "icon-library", label.casefold().replace(" ", "-").replace("&", "and"))
            for label in ICON_LIBRARY_TABS
        ),
        state.icon_library_tab,
        max(1.0, available_width - 84.0 * scale),
        gap=4.0 * scale,
    )
    if state.icon_library_tab in ICON_GROUP_LAYOUT_DEFAULTS:
        state.icon_adjustment_group = state.icon_library_tab
    content_y = float(imgui.get_cursor_screen_pos().y) + 14.0 * scale
    content_origin = (x0 + 42.0 * scale, content_y)
    if state.icon_library_tab == "Overview":
        _draw_icon_library_overview(draw, content_origin, scale, state)
    elif state.icon_library_tab == "UI context":
        _draw_icon_context_page(draw, content_origin, scale, state)
    elif state.icon_library_tab == "Capsules":
        _draw_capsule_context_page(draw, content_origin, scale, state)
    else:
        padding = state.icon_padding_for(state.icon_library_tab)
        _draw_icon_family_detail(
            draw,
            content_origin,
            state.icon_library_tab,
            scale,
            padding,
            state.concept_mouse_width(),
            state.icon_stroke_width,
        )


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
    draw = ImguiDraw2D(imgui.get_window_draw_list())

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
                    lambda *args: _draw_reset_icon(*args, stroke_width=state.tool_stroke_width),
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
                backend="OpenGL",
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
                padding = state.icon_padding_for("Scene helpers")
                draw_concept_icon(
                    draw,
                    center,
                    20.0 * scale * icon_scale,
                    "helper-camera",
                    CONCEPT_THEME.text,
                    padding=padding,
                    stroke_width=state.icon_stroke_width,
                )
                draw_concept_icon(
                    draw,
                    (center[0] + 40.0 * scale, center[1]),
                    20.0 * scale * icon_scale,
                    "helper-light",
                    CONCEPT_THEME.text,
                    padding=padding,
                    stroke_width=state.icon_stroke_width,
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


def _draw_workspace_canvas(available, scale: float, state: ProbeState):
    flags = _flags(
        imgui.WindowFlags_.horizontal_scrollbar,
        imgui.WindowFlags_.always_vertical_scrollbar,
    )
    if not imgui.begin_child(
        "Editor preview canvas###ProbeWorkspaceCanvas",
        available,
        imgui.ChildFlags_.none.value,
        flags,
    ):
        imgui.end_child()
        return None

    cursor = imgui.get_cursor_pos()
    origin = imgui.get_cursor_screen_pos()
    width, height = _virtual_canvas_size(available, scale, WORKSPACE_CANVAS_SIZE)
    spacing = imgui.get_style().item_spacing
    status_height = 34.0 * scale
    main_height = height - status_height - spacing.y
    left_width = 330.0 * scale
    right_width = 390.0 * scale
    center_width = width - left_width - right_width - spacing.x * 2.0
    bottom_height = 250.0 * scale
    viewport_height = main_height - bottom_height - spacing.y
    right_top_height = main_height * 0.50
    right_bottom_height = main_height - right_top_height - spacing.y

    left_x = origin.x
    center_x = left_x + left_width + spacing.x
    right_x = center_x + center_width + spacing.x
    top_y = origin.y

    imgui.set_cursor_screen_pos(imgui.ImVec2(left_x, top_y))
    _draw_hierarchy_gallery(imgui.ImVec2(left_width, main_height), state, scale)

    imgui.set_cursor_screen_pos(imgui.ImVec2(center_x, top_y))
    viewport_rect = _draw_viewport(imgui.ImVec2(center_width, viewport_height), scale, state)
    imgui.set_cursor_screen_pos(imgui.ImVec2(center_x, top_y + viewport_height + spacing.y))
    _draw_output(imgui.ImVec2(center_width, bottom_height), state, scale)

    imgui.set_cursor_screen_pos(imgui.ImVec2(right_x, top_y))
    _draw_workspace_right_dock(imgui.ImVec2(right_width, right_top_height), state)
    imgui.set_cursor_screen_pos(imgui.ImVec2(right_x, top_y + right_top_height + spacing.y))
    _draw_inspector_gallery(imgui.ImVec2(right_width, right_bottom_height), scale)

    status_origin = (origin.x, origin.y + main_height + spacing.y)
    _draw_status_strip(
        ImguiDraw2D(imgui.get_window_draw_list()),
        status_origin,
        width,
        status_height,
        scale,
        selected="a_sphere",
        running=state.playing,
        fps="60.0" if state.playing else "119.8",
    )

    imgui.set_cursor_pos(imgui.ImVec2(cursor.x + width - 1.0, cursor.y + height - 1.0))
    imgui.dummy(imgui.ImVec2(1.0, 1.0))
    imgui.end_child()
    return viewport_rect


def _draw_workspace(window: Window, state: ProbeState) -> None:
    scale = window.style_scale
    theme_mod.apply_corner_radius(imgui, state.imgui_rounding, scale)
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
    opened, _ = imgui.begin("Mojive UI Probe", True, flags)
    if not opened:
        imgui.end()
        return

    if imgui.begin_menu_bar():
        for label in ("File", "Edit", "Entity", "View", "Window", "Help"):
            if imgui.begin_menu(label):
                imgui.menu_item("Design probe", "", False, False)
                imgui.end_menu()
        if imgui.begin_menu("Probe"):
            for page in ("Workspace", "Panels", "Geometry", "Redesign"):
                clicked, _ = imgui.menu_item(page, "", state.page == page)
                if clicked:
                    state.page = page
            imgui.separator()
            clicked, _ = imgui.menu_item(
                "Preview Icon Library",
                "",
                state.preview_icon_library,
            )
            if clicked:
                state.preview_icon_library = not state.preview_icon_library
            imgui.set_item_tooltip("Use Icon Library candidates in the actual feasibility layouts")
            imgui.separator()
            _, state.imgui_rounding = imgui.slider_float(
                "ImGui corner radius",
                state.imgui_rounding,
                0.0,
                16.0,
                "%.2f px",
                imgui.SliderFlags_.always_clamp.value,
            )
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
                ("G3 transitions", "highlight_g3"),
            ):
                value = bool(getattr(state, attribute))
                clicked, _ = imgui.menu_item(label, "", value)
                if clicked:
                    setattr(state, attribute, not value)
            imgui.end_menu()
        clicked, _ = imgui.menu_item(
            "Icon Library preview",
            "",
            state.preview_icon_library,
        )
        if clicked:
            state.preview_icon_library = not state.preview_icon_library
        imgui.set_item_tooltip("Substitute candidates throughout the current page")
        imgui.end_menu_bar()

    available = imgui.get_content_region_avail()
    if state.page == "Redesign":
        draw_redesign(
            state.redesign,
            scale,
            state,
            circular_button=_circular_icon_button,
        )
        imgui.end()
        return
    if state.page == "Panels":
        _draw_panel_page(available, state, scale)
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

    viewport_rect = _draw_workspace_canvas(available, scale, state)
    imgui.end()

    if viewport_rect is None:
        return

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
    initial_icon_group: str,
    initial_rotate_cap: str,
    ui_scale: float,
    interactive_fps: float,
    initial_smoothing: float | None = None,
    initial_imgui_radius: float | None = None,
    initial_tool_stroke: float | None = None,
    initial_rotate_gap_ratio: float | None = None,
    initial_playback_zoom: float | None = None,
    initial_icon_padding: float | None = None,
    initial_icon_stroke: float | None = None,
    redesign_language: str = "en",
    redesign_section: str = "Overview",
    capsule_outline: str = "Soft white",
    preview_icon_library: bool = False,
) -> None:
    window_width, window_height = _probe_window_size(width, height, ui_scale)
    window = Window(
        WindowConfig(
            title="Mojive UI feasibility",
            width=window_width,
            height=window_height,
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
            icon_library_tab=initial_icon_group,
            rotate_ring_cap=initial_rotate_cap,
            capsule_outline=capsule_outline,
            preview_icon_library=preview_icon_library,
        )
        state.redesign.language = redesign_language
        state.redesign.section = redesign_section
        if initial_smoothing is not None:
            for _, name in CORNER_CONTROLS:
                setattr(state, name, initial_smoothing)
        if initial_imgui_radius is not None:
            state.imgui_rounding = initial_imgui_radius
        if initial_tool_stroke is not None:
            state.tool_stroke_width = initial_tool_stroke
        if initial_rotate_gap_ratio is not None:
            state.rotate_ring_gap_ratio = initial_rotate_gap_ratio
        if initial_playback_zoom is not None:
            state.construction_playback_scale = initial_playback_zoom
        initial_adjustment_group = (
            initial_icon_group
            if initial_icon_group in ICON_GROUP_LAYOUT_DEFAULTS
            else {
                "Capsules": "Viewport playback",
                "UI context": "Keyframes",
            }.get(initial_icon_group, state.icon_adjustment_group)
        )
        state.icon_adjustment_group = initial_adjustment_group
        if initial_icon_padding is not None:
            state.set_icon_padding_for(initial_adjustment_group, initial_icon_padding)
        if initial_icon_stroke is not None:
            state.icon_stroke_width = initial_icon_stroke
        if interactive:
            window.show()
            frame_period = 1.0 / interactive_fps
            try:
                while not window.should_close():
                    frame_started = time.perf_counter()
                    window.begin_frame()
                    _draw_workspace(window, state)
                    if state.page != "Redesign" and imgui.is_key_pressed(imgui.Key.escape, False):
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
        if "state" in locals() and state.timeline_session is not None:
            state.timeline_session.release()
        window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=PROBE_BASE_SIZE[0])
    parser.add_argument("--height", type=int, default=PROBE_BASE_SIZE[1])
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.0,
        help=(
            "Logical UI scale used for paths, strokes, controls, and text; "
            "the capture expands above 2x and oversized concept canvases scroll"
        ),
    )
    parser.add_argument(
        "--page",
        choices=("workspace", "panels", "geometry", "redesign"),
        default="workspace",
        help="Initial probe page; interactive mode can switch from the Probe menu",
    )
    parser.add_argument("--redesign-language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--capsule-outline", choices=("neutral-gray", "soft-white"), default="soft-white"
    )
    parser.add_argument(
        "--redesign-section",
        choices=("overview", "inspector", "control", "keys"),
        default="overview",
    )
    parser.add_argument(
        "--geometry-tab",
        choices=(
            "corners",
            "playback",
            "tools",
            "icons",
            "hints",
            "gizmos",
            "helpers",
            "status",
            "diagnostics",
            "shell",
            "panels",
            "workspaces",
        ),
        default="playback",
        help="Initial non-closeable tab on the geometry page",
    )
    parser.add_argument(
        "--icon-group",
        choices=tuple(ICON_GROUP_BY_SLUG),
        default="overview",
        help="Initial family on the concept-only Icon library geometry tab",
    )
    parser.add_argument(
        "--rotate-cap",
        choices=("butt", "round"),
        default=OVERLAY_GEOMETRY.rotate_ring_cap,
        help="Initial Rotate inner-ring cap style",
    )
    parser.add_argument(
        "--tool-stroke",
        type=float,
        default=None,
        help="Initial Tool glyph stroke, from 1.0 to 2.2 logical pixels",
    )
    parser.add_argument(
        "--rotate-gap-ratio",
        type=float,
        default=None,
        help=("Initial Rotate crossing gap as a stroke ratio, from 0.25 to 1.00"),
    )
    parser.add_argument(
        "--playback-zoom",
        type=float,
        default=None,
        help="Initial Playback construction zoom, from 1.5 to 4.0",
    )
    parser.add_argument(
        "--icon-padding",
        type=float,
        default=None,
        help=(
            f"Initial circular glyph padding for --icon-group, from {ICON_MIN_CLEARANCE:g} "
            f"to {ICON_MAX_PADDING:g} grid units"
        ),
    )
    parser.add_argument(
        "--icon-stroke",
        type=float,
        default=None,
        help=(
            f"Initial shared Icon Library stroke, from {ICON_MIN_STROKE:g} "
            f"to {ICON_MAX_STROKE:g} grid units"
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a real ImGui window and run until it is closed",
    )
    parser.add_argument(
        "--preview-icon-library",
        action="store_true",
        help="Use Icon Library candidates throughout the selected feasibility page",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Maximum interactive refresh rate (default: 30)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="Initial corner smoothing for all element groups, from 0 to 1",
    )
    parser.add_argument(
        "--imgui-radius",
        type=float,
        default=None,
        help="Initial ImGui radius in logical pixels, from 0 to 16",
    )
    args = parser.parse_args()
    if args.imgui_radius is not None and not 0.0 <= args.imgui_radius <= 16.0:
        parser.error("--imgui-radius must be between 0 and 16")
    if args.smoothing is not None and not 0.0 <= args.smoothing <= 1.0:
        parser.error("--smoothing must be between 0 and 1")
    if args.tool_stroke is not None and not 1.0 <= args.tool_stroke <= 2.2:
        parser.error("--tool-stroke must be between 1.0 and 2.2")
    if args.rotate_gap_ratio is not None and not 0.25 <= args.rotate_gap_ratio <= 1.0:
        parser.error("--rotate-gap-ratio must be between 0.25 and 1.00")
    if args.playback_zoom is not None and not 1.5 <= args.playback_zoom <= 4.0:
        parser.error("--playback-zoom must be between 1.5 and 4.0")
    if (
        args.icon_padding is not None
        and not ICON_MIN_CLEARANCE <= args.icon_padding <= ICON_MAX_PADDING
    ):
        parser.error(
            f"--icon-padding must be between {ICON_MIN_CLEARANCE:g} and {ICON_MAX_PADDING:g}"
        )
    if (
        args.icon_stroke is not None
        and not ICON_MIN_STROKE <= args.icon_stroke <= ICON_MAX_STROKE
    ):
        parser.error(
            f"--icon-stroke must be between {ICON_MIN_STROKE:g} and {ICON_MAX_STROKE:g}"
        )
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
            "corners": "Corners",
            "playback": "Playback",
            "tools": "Tools",
            "icons": "Icon library",
            "hints": "Hints & input",
            "gizmos": "Transform gizmos",
            "helpers": "Joint & helpers",
            "status": "Status",
            "diagnostics": "Diagnostics",
            "shell": "Shell & settings",
            "panels": "Panels",
            "workspaces": "Workspaces",
        }[args.geometry_tab],
        initial_icon_group=ICON_GROUP_BY_SLUG[args.icon_group],
        initial_rotate_cap=args.rotate_cap,
        ui_scale=args.ui_scale,
        interactive_fps=args.fps,
        initial_smoothing=args.smoothing,
        initial_imgui_radius=args.imgui_radius,
        initial_tool_stroke=args.tool_stroke,
        initial_rotate_gap_ratio=args.rotate_gap_ratio,
        initial_playback_zoom=args.playback_zoom,
        initial_icon_padding=args.icon_padding,
        initial_icon_stroke=args.icon_stroke,
        redesign_language=args.redesign_language,
        redesign_section=args.redesign_section.title(),
        capsule_outline=args.capsule_outline.replace("-", " ").capitalize(),
        preview_icon_library=args.preview_icon_library,
    )
    print("interactive probe closed" if args.interactive else output)


if __name__ == "__main__":
    main()
