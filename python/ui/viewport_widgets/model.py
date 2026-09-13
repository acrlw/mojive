"""Viewport widgets: model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

CAPSULE_SMOOTHING = 0.382


@dataclass(frozen=True)
class OverlayGeometry:
    """Shared logical-pixel geometry for viewport chrome and its design probe."""

    icon_radius: float = 10.0
    radial_step: float = 8.0
    center_step: float = 42.0
    tool_center_step: float = 42.0
    tool_group_gap: float = 10.0
    divider_width: float = 20.0
    tool_stroke: float = 1.46
    rotate_ring_gap_ratio: float = 0.5
    rotate_ring_cap: str = "round"
    hint_control_height: float = 18.0
    hint_padding_x: float = 16.0
    hint_padding_y: float = 8.0
    hint_input_gap: float = 8.0
    hint_group_gap: float = 24.0
    hint_chord_gap: float = 10.0
    hint_key_padding_x: float = 8.0
    hint_mouse_width: float = 14.0
    hint_mouse_stroke: float = 1.0
    hint_mouse_button_width_ratio: float = 0.40
    hint_mouse_button_shell_ratio: float = 1.25
    hint_mouse_button_height_ratio: float = 0.40
    hint_mouse_wheel_width_ratio: float = 0.32
    hint_mouse_wheel_height_ratio: float = 0.40
    hint_mouse_wheel_gap_ratio: float = 1.0
    frame_center_radius: float = 1.45
    frame_center_gap_ratio: float = 1.4
    tooltip_padding_x: float = 7.0
    tooltip_padding_y: float = 4.0

    @property
    def state_radius(self) -> float:
        return self.icon_radius + self.radial_step

    @property
    def shell_radius(self) -> float:
        return self.state_radius + self.radial_step

    @property
    def rotate_ring_gap(self) -> float:
        return self.tool_stroke * self.rotate_ring_gap_ratio


@dataclass(frozen=True)
class ViewportLabels:
    """Localized viewport chrome copy, resolved only when language changes."""

    play: str = "Play"
    pause: str = "Pause"
    previous: str = "Previous frame"
    rewind: str = "Rewind"
    step: str = "Step"
    pause_to_step: str = "Pause to step"
    reset: str = "Reset"
    record: str = "Record Take"
    stop_recording: str = "Stop recording"
    recording_options: str = "Recording Settings..."
    move: str = "Move"
    rotate: str = "Rotate"
    dimensions: str = "Dimensions"
    world_body: str = "World / Body"
    snap: str = "Snap"
    orbit: str = "Orbit"
    pan: str = "Pan"
    zoom: str = "Zoom"
    frame: str = "Frame"
    type_value: str = "Type value"
    drag: str = "Drag"
    push: str = "Push"
    twist: str = "Twist"
    running: str = "Running"
    paused: str = "Paused"
    static: str = "Static"
    time: str = "Time"
    steps: str = "Steps"
    physics: str = "Physics"
    render: str = "Render"
    no_selection: str = "No selection"
    clear_selection: str = "Clear selection"
    show_steps: str = "Click to show steps"
    show_time: str = "Click to show time"
    copy_exact: str = "Right-click to copy exact value"


DEFAULT_VIEWPORT_LABELS = ViewportLabels()


@dataclass(frozen=True)
class ToolHint:
    """One input/meaning pair that can be rendered on any chrome surface."""

    kind: str
    control: str = ""
    label: str = ""
    suffix: str = ""
    hint_id: str = ""
    modifier: str = ""


@dataclass(frozen=True)
class StatusLayout:
    """Interactive geometry emitted while drawing the application status bar."""

    metric_rect: tuple[float, float, float, float] | None = None
    metric_exact: str = ""
    recording_pause_rect: tuple[float, float, float, float] | None = None
    recording_stop_rect: tuple[float, float, float, float] | None = None
    message_rect: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class _StatusPerformanceLayout:
    """Compact, progressively collapsible right-edge telemetry columns."""

    backend_text: str
    metric_text: str
    delta_text: str
    fps_text: str
    backend_x: float
    metric_x: float
    metric_width: float
    delta_x: float
    fps_x: float
    dividers: tuple[float, ...]
    left: float


OVERLAY_GEOMETRY = OverlayGeometry()
DEFAULT_VIEWPORT_OVERLAY_SCALE = 1.25
MIN_VIEWPORT_OVERLAY_SCALE = 0.85
MAX_VIEWPORT_OVERLAY_SCALE = 2.0
MIN_VIEWPORT_CAPSULE_SCALE = 0.6
MAX_VIEWPORT_CAPSULE_SCALE = 1.6
PLAYBACK_CHROME_SCALE = 0.82
PLAYBACK_HALF_HEIGHT_PT = 6.8
PLAYBACK_STEP_SCALE = 0.88
PLAYBACK_RESET_SCALE = 0.92
TOOL_CHROME_SCALE = PLAYBACK_CHROME_SCALE
HINT_CHROME_SCALE = PLAYBACK_CHROME_SCALE
OVERLAY_CLIP_PADDING = 5.0
TOOL_GLYPH_SCALE = 1.18
RECORDING_OPTIONS_ENVELOPE_SCALE = 1.25
RECORDING_OPTIONS_GLYPH_SCALE = 1.00
RECORDING_OPTIONS_STROKE_SCALE = 0.80
_MOVE_ARROW_BASE = 6.0
_MOVE_ARROW_TIP = 9.0
_MOVE_ARROW_WING = (_MOVE_ARROW_TIP - _MOVE_ARROW_BASE) / math.sqrt(3.0)
_MOVE_SHAFT_VISUAL_RATIO = 0.5
_FRAME_ARROW_CORNER_RADIUS_PT = 0.45
FRAME_LABEL_MAX_WIDTH = 4.4
ICON_RADIUS = OVERLAY_GEOMETRY.icon_radius
STATE_RADIUS = OVERLAY_GEOMETRY.state_radius
SHELL_RADIUS = OVERLAY_GEOMETRY.shell_radius
CENTER_STEP = OVERLAY_GEOMETRY.center_step
TOOL_GROUP_GAP = OVERLAY_GEOMETRY.tool_group_gap
DIVIDER_WIDTH = OVERLAY_GEOMETRY.divider_width
_ROTATE_HALF_RINGS = (
    (
        (3.177, 4.765),
        (3.488, 4.386),
        (3.740, 3.931),
        (3.928, 3.410),
        (4.048, 2.830),
        (4.099, 2.201),
        (4.080, 1.535),
        (3.992, 0.843),
        (3.835, 0.136),
        (3.612, -0.573),
        (3.328, -1.272),
        (2.986, -1.950),
        (2.594, -2.594),
        (2.157, -3.194),
        (1.683, -3.739),
        (1.181, -4.220),
        (0.658, -4.629),
        (0.124, -4.959),
        (-0.412, -5.204),
        (-0.941, -5.360),
        (-1.454, -5.424),
        (-1.942, -5.395),
        (-2.397, -5.274),
        (-2.811, -5.063),
        (-3.177, -4.765),
    ),
    (
        (-3.177, 4.765),
        (-3.488, 4.386),
        (-3.740, 3.931),
        (-3.928, 3.410),
        (-4.048, 2.830),
        (-4.099, 2.201),
        (-4.080, 1.535),
        (-3.992, 0.843),
        (-3.835, 0.136),
        (-3.612, -0.573),
        (-3.328, -1.272),
        (-2.986, -1.950),
        (-2.594, -2.594),
        (-2.157, -3.194),
        (-1.683, -3.739),
        (-1.181, -4.220),
        (-0.658, -4.629),
        (-0.124, -4.959),
        (0.412, -5.204),
        (0.941, -5.360),
        (1.454, -5.424),
        (1.942, -5.395),
        (2.397, -5.274),
        (2.811, -5.063),
        (3.177, -4.765),
    ),
    (
        (5.800, 0.000),
        (5.750, 0.379),
        (5.602, 0.751),
        (5.359, 1.110),
        (5.023, 1.450),
        (4.601, 1.765),
        (4.101, 2.051),
        (3.531, 2.301),
        (2.900, 2.511),
        (2.220, 2.679),
        (1.501, 2.801),
        (0.757, 2.875),
        (0.000, 2.900),
        (-0.757, 2.875),
        (-1.501, 2.801),
        (-2.220, 2.679),
        (-2.900, 2.511),
        (-3.531, 2.301),
        (-4.101, 2.051),
        (-4.601, 1.765),
        (-5.023, 1.450),
        (-5.359, 1.110),
        (-5.602, 0.751),
        (-5.750, 0.379),
        (-5.800, 0.000),
    ),
)
_FRAME_AXES = ((0.0, -1.0), (0.866025, 0.5), (-0.866025, 0.5))


@dataclass(frozen=True)
class ViewportControl:
    """One declarative playback/tool action and its optional custom glyph."""

    name: str
    icon: Callable | None = None
    tooltip: str = ""


PLAYBACK_CONTROLS = tuple(
    ViewportControl(name)
    for name in ("previous", "toggle", "step", "reset", "record", "recording-options")
)
TOOL_GROUPS = (
    tuple(ViewportControl(name) for name in ("move", "rotate", "dimensions")),
    tuple(ViewportControl(name) for name in ("frame", "snap")),
)


def tool_control_centers(
    groups: Sequence[Sequence[ViewportControl]] = TOOL_GROUPS,
) -> tuple[float, ...]:
    centers: list[float] = []
    cursor = SHELL_RADIUS
    for group_index, group in enumerate(groups):
        if group_index:
            cursor += TOOL_GROUP_GAP
        for _control in group:
            centers.append(cursor)
            cursor += OVERLAY_GEOMETRY.tool_center_step
    return tuple(centers)


TOOL_CONTROLS = tuple(control for group in TOOL_GROUPS for control in group)


@lru_cache(maxsize=256)
def _transform_path(
    points: tuple[tuple[float, float], ...],
    x: float,
    y: float,
    scale: float,
) -> tuple[tuple[float, float], ...]:
    return tuple((x + px * scale, y + py * scale) for px, py in points)


RESET_GLYPH_SCALE = 0.88
_PROJECTION_HALF_WIDTH_PT = 5.2
_PROJECTION_STROKE_PT = 1.25


@dataclass(frozen=True)
class MouseButtonGeometry:
    """A highlighted button and the mouse-shell path visible around it."""

    visible_shell: tuple[tuple[float, float], ...]
    fill: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MouseWheelGeometry:
    """A highlighted wheel separated from the shell by a physical-pixel gap."""

    lo: tuple[float, float]
    hi: tuple[float, float]
    rounding: float
    gap: float


_HintGroup = ToolHint
