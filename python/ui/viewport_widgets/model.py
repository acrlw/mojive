"""Viewport widgets: model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from ..overlay_geometry import _ROTATE_HALF_RINGS as _ROTATE_HALF_RINGS
from ..overlay_geometry import CAPSULE_SMOOTHING as CAPSULE_SMOOTHING
from ..overlay_geometry import DEFAULT_RESET_HEAD_SCALE as DEFAULT_RESET_HEAD_SCALE
from ..overlay_geometry import OVERLAY_GEOMETRY as OVERLAY_GEOMETRY
from ..overlay_geometry import TOOL_GLYPH_SCALE as TOOL_GLYPH_SCALE
from ..overlay_geometry import MouseButtonGeometry as MouseButtonGeometry
from ..overlay_geometry import MouseWheelGeometry as MouseWheelGeometry
from ..overlay_geometry import OverlayGeometry as OverlayGeometry

# Source glyphs keep their authored coordinates before fitting into a control slot.
GLYPH_REFERENCE_RADIUS = 10.0


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
    replaying: str = "Replaying Take"
    paused: str = "Paused"
    static: str = "Static"
    time: str = "Time"
    steps: str = "Steps"
    physics: str = "Physics"
    render: str = "Render"
    passive: str = "Passive"
    recording_finalizing: str = "Finalizing recording"
    recording_scene: str = "SCENE"
    recording_viewport: str = "VIEW"
    recording_window: str = "WINDOW"
    recording: str = "REC"
    no_selection: str = "No selection"
    clear_selection: str = "Clear selection"
    show_steps: str = "Click to show steps"
    show_time: str = "Click to show time"
    copy_exact: str = "Right-click to copy exact value"


DEFAULT_VIEWPORT_LABELS = ViewportLabels()


@dataclass(frozen=True)
class ToolHint:
    """An input hint for either chrome surface.

    ``kind="keys"`` puts ``label`` before alternative ``(key, meaning)`` pairs,
    for example ``Speed [Up] + / [Down] -``. The whole group stays together.
    """

    kind: Literal["text", "key", "keys", "mouse", "perturb"]
    control: str = ""
    label: str = ""
    suffix: str = ""
    hint_id: str = ""
    modifier: str = ""
    keys: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.kind not in ("text", "key", "keys", "mouse", "perturb"):
            raise ValueError(f"Unsupported tool hint kind: {self.kind!r}")
        if any(
            not isinstance(value, str)
            for value in (self.control, self.label, self.suffix, self.hint_id, self.modifier)
        ):
            raise ValueError("Tool hint text fields must be strings")
        if self.kind == "keys" and (
            not isinstance(self.keys, (tuple, list))
            or not self.keys
            or any(
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(isinstance(value, str) for value in pair)
                or not pair[0]
                for pair in self.keys
            )
        ):
            raise ValueError("Grouped key hints need nonempty (key, meaning) pairs")
        if self.kind == "keys":
            object.__setattr__(self, "keys", tuple(tuple(pair) for pair in self.keys))


@dataclass(frozen=True)
class StatusLayout:
    """Interactive geometry emitted while drawing the application status bar."""

    metric_rect: tuple[float, float, float, float] | None = None
    metric_exact: str = ""
    recording_pause_rect: tuple[float, float, float, float] | None = None
    recording_stop_rect: tuple[float, float, float, float] | None = None
    message_rect: tuple[float, float, float, float] | None = None
    passive_rect: tuple[float, float, float, float] | None = None


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
    cursor = OVERLAY_GEOMETRY.end_padding
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


_HintGroup = ToolHint
