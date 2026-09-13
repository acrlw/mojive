"""Shared palettes, dimensions and fixed geometry specimens for UI review."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from mojive.types import CameraView
from mojive.ui import gizmo as gizmo_ui
from mojive.ui import theme as theme_mod
from mojive.ui import viewcube as view_ui
from mojive.ui.theme import THEME, rgb8

ROOT = Path(__file__).resolve().parents[3]

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


_ICON_REVIEW_SIZES = (14.0, 24.0, 56.0, 112.0)
