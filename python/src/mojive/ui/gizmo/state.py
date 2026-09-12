"""Gizmo: state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mojive.adapters.base import JointInfo, NodeType
from mojive.drawing.curves import CORNER_SMOOTHING
from mojive.interaction.gizmo import (
    ACTIVE_HANDLE_COLOR,
    AXIS_HEAD_LENGTH_PT,
    RING_RADIUS,
    RING_WIDTH_PT,
    GizmoHandle,
    GizmoMode,
    axis_arrow_polygon,
    axis_hover_color,
)
from mojive.scene.geometry import GeometryDimensions
from mojive.types import CameraView, MeshShape
from mojive.ui.panels.inspector import gizmo_refusal_reason
from mojive.ui.theme import THEME

if TYPE_CHECKING:
    from mojive.adapters.base import SceneNode


REASON_NO_SELECTION = "nothing selected"
DRAG_LAYER = "ui.gizmo.drag"
_WORLD_BASIS = np.eye(3, dtype=np.float64)
DEFAULT_TRANSLATION_SNAP_M = 0.1
DEFAULT_ROTATION_SNAP_DEG = 5.0
DEFAULT_ROTATION_TICK_SCALE = 1.25
SNAP_TICK_FULL_STEPS = 5.0
SNAP_TICK_FADE_STEPS = 10.0
ROTATION_TICK_MIN_ALPHA = 0.5
JOINT_RANGE_RADIUS = RING_RADIUS
JOINT_RANGE_WIDTH_PT = RING_WIDTH_PT
JOINT_COMPLEMENT_ALPHA = 0.12
JOINT_COMPLEMENT_HOVER_ALPHA = 0.26
JOINT_RANGE_OFFSET_PT = 0.0
JOINT_RANGE_COLOR = THEME.primary
JOINT_ACTIVE_DARK_COLOR = THEME.primary_dim
JOINT_LOWER_LIMIT_COLOR = THEME.axis_color("z")
JOINT_UPPER_LIMIT_COLOR = THEME.axis_color("x")
JOINT_CURRENT_COLOR = axis_hover_color(JOINT_RANGE_COLOR)
JOINT_CURRENT_TICK_PT = 20.0
JOINT_LIMIT_TICK_PT = 14.0
JOINT_LIMIT_HIT_PT = 18.0
JOINT_LIMIT_HIT_PADDING_PT = 4.0
JOINT_LIMIT_STROKE_HIT_PADDING_PT = 3.0
JOINT_LIMIT_OUTER_HIT_FRACTION = 0.30
JOINT_PRECISION_TRIGGER_PT = 30.0
JOINT_PRECISION_MAX_HINGE_SPAN_DEG = 45.0
JOINT_PRECISION_TRACK_PT = 144.0
JOINT_PRECISION_PANEL_HEIGHT_PT = 30.0
JOINT_PRECISION_OFFSET_PT = 44.0
JOINT_PRECISION_MARGIN_PT = 10.0
JOINT_PRECISION_REVEAL_DELAY_SECONDS = 0.50
JOINT_PRECISION_DWELL_GRACE_SECONDS = 0.12
JOINT_PRECISION_REVEAL_GRACE_SECONDS = 0.30
JOINT_SLIDE_ARROW_OFFSET_PT = 17.0
JOINT_SLIDE_ARROW_INSET_PT = 9.0
JOINT_SLIDE_ARROW_EXTENT_PT = 48.0
JOINT_SLIDE_ARROW_VISUAL_SCALE = 0.8
JOINT_SLIDE_ARROW_TARGET_PT = (
    2.0 * JOINT_SLIDE_ARROW_INSET_PT
    + 4.0 * (JOINT_SLIDE_ARROW_EXTENT_PT - AXIS_HEAD_LENGTH_PT)
    + JOINT_SLIDE_ARROW_EXTENT_PT
) / 7.0
JOINT_SLIDE_AXIS_HIT_PT = 7.0
TRANSLATION_GUIDE_RADIUS_PT = 5.0
TRACKBALL_RAD_PER_PT = 0.01
ROTATION_LINEAR_ESCAPE_PT = 8.0
ROTATION_LINEAR_LOCK_PT = 2.0
ROTATION_EDGE_LINEAR_ALPHA = 0.18
SLIDE_CARDINAL_ESCAPE_PT = 8.0
JOINT_DRAG_START_TICK_HALF_PT = 6.0
JOINT_ROTATION_AXIS_DASH_PT = 5.0
JOINT_ROTATION_AXIS_GAP_PT = 4.0
_FULL_TURN = 2.0 * np.pi
_JOINT_RANGE_EPSILON = 1e-9


def _with_alpha(color, alpha: float) -> tuple[float, float, float, float]:
    return (
        float(color[0]),
        float(color[1]),
        float(color[2]),
        float(color[3]) * float(alpha),
    )


def _joint_drag_label_color(state: _JointRangeState | None, *, active: bool = False):
    """Return the live joint label dot color, including endpoint clamping."""

    if active:
        return ACTIVE_HANDLE_COLOR
    return _joint_endpoint_color(state) or JOINT_CURRENT_COLOR


def _joint_endpoint_color(state: _JointRangeState | None):
    """Return the semantic limit color only while a scalar joint is clamped."""

    if state is None:
        return None
    if _joint_value_at_limit(state, state.current, lower=True):
        return JOINT_LOWER_LIMIT_COLOR
    if _joint_value_at_limit(state, state.current, lower=False):
        return JOINT_UPPER_LIMIT_COLOR
    return None


def _joint_value_at_limit(state: _JointRangeState, value: float, *, lower: bool) -> bool:
    """Return whether a scalar value is clamped to one authored limit."""

    tolerance = max(
        _JOINT_RANGE_EPSILON,
        abs(float(state.upper) - float(state.lower)) * 1e-6,
    )
    if lower:
        return float(value) <= float(state.lower) + tolerance
    return float(value) >= float(state.upper) - tolerance


def _joint_current_tick_color(range_color):
    """Keep the current-value tick distinct from the colored limit ticks."""

    return tuple(float(channel) for channel in range_color)


def joint_slide_arrow_polygons(
    current,
    tangent,
    style_scale: float,
    *,
    for_hit_test: bool = False,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smaller round-tail handles, or their unchanged picking silhouettes."""

    current = np.asarray(current, np.float64).reshape(2)
    tangent = np.asarray(tangent, np.float64).reshape(2)
    length = float(np.linalg.norm(tangent))
    if length < 1e-6:
        empty = np.empty((0, 2), np.float64)
        return empty, empty
    tangent = tangent / length
    normal = np.array((-tangent[1], tangent[0]))
    offset = normal * JOINT_SLIDE_ARROW_OFFSET_PT * style_scale
    # Keep both affordances clear of the value axis. Their opposing directions
    # communicate bidirectional motion without making the axis itself look cut.
    inset = tangent * JOINT_SLIDE_ARROW_INSET_PT * style_scale
    extent = tangent * JOINT_SLIDE_ARROW_EXTENT_PT * style_scale

    def polygon(direction: float) -> np.ndarray:
        start = current + offset + direction * inset
        end = current + offset + direction * extent
        points = axis_arrow_polygon(
            start,
            end,
            style_scale,
            round_tail=not for_hit_test,
            smoothing=CORNER_SMOOTHING if for_hit_test else smoothing,
        )
        if for_hit_test:
            return points
        center = (start + end) * 0.5
        return center + (points - center) * JOINT_SLIDE_ARROW_VISUAL_SCALE

    return polygon(1.0), polygon(-1.0)


def joint_slide_arrow_targets(
    current, tangent, style_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable interaction targets independent of arrow tessellation."""

    current = np.asarray(current, np.float64).reshape(2)
    tangent = np.asarray(tangent, np.float64).reshape(2)
    length = float(np.linalg.norm(tangent))
    if length < 1e-6:
        return current.copy(), current.copy()
    tangent = tangent / length
    normal = np.array((-tangent[1], tangent[0]))
    center = current + normal * JOINT_SLIDE_ARROW_OFFSET_PT * style_scale
    along = tangent * JOINT_SLIDE_ARROW_TARGET_PT * style_scale
    return center + along, center - along


def _screen_segment_distance(point, start, end) -> float:
    point = np.asarray(point, np.float64).reshape(2)
    start = np.asarray(start, np.float64).reshape(2)
    end = np.asarray(end, np.float64).reshape(2)
    edge = end - start
    denominator = float(np.dot(edge, edge))
    amount = (
        float(np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0))
        if denominator > 1e-12
        else 0.0
    )
    return float(np.linalg.norm(point - (start + edge * amount)))


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class _JointTarget:
    joint: JointInfo
    mode: GizmoMode
    handles: int


@dataclass(frozen=True)
class _JointRangeState:
    joint_type: str
    current: float
    lower: float
    upper: float
    joint_id: int = -1
    qpos_adr: int = -1

    @property
    def angular_span(self) -> float:
        """Reachable hinge span, clamped to one complete dial turn."""

        return float(np.clip(self.upper - self.lower, 0.0, _FULL_TURN))

    @property
    def raw_angular_span(self) -> float:
        return max(0.0, float(self.upper - self.lower))

    @property
    def covers_full_turn(self) -> bool:
        return self.angular_span >= _FULL_TURN - _JOINT_RANGE_EPSILON

    @property
    def has_ambiguous_dial_limits(self) -> bool:
        """Whether a circular dial cannot uniquely place both scalar limits."""

        return self.raw_angular_span >= _FULL_TURN - _JOINT_RANGE_EPSILON

    def contains_angle(self, angle: float) -> bool:
        """Return whether a dial angle belongs to the reachable hinge arc."""

        if self.covers_full_turn:
            return True
        relative = float((angle - self.lower) % _FULL_TURN)
        return relative <= self.angular_span + _JOINT_RANGE_EPSILON


@dataclass(frozen=True)
class _SlideRangeProjection:
    lower: np.ndarray
    current: np.ndarray
    upper: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    alpha: float


@dataclass(frozen=True)
class _HingeRangeProjection:
    alpha: float
    allowed: np.ndarray | None
    full_range: bool
    current_tick: tuple[np.ndarray, np.ndarray] | None
    lower_tick: tuple[np.ndarray, np.ndarray] | None
    upper_tick: tuple[np.ndarray, np.ndarray] | None
    complement: np.ndarray | None = None


@dataclass(frozen=True)
class _HingeAxisProjection:
    alpha: float
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    hit_polygon: np.ndarray
    occluded: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class _JointPrecisionProjection:
    """Expanded viewport rail for a scalar range that is too small on screen."""

    joint_id: int
    qpos_adr: int
    joint_type: str
    lower: float
    upper: float
    start: np.ndarray
    current: np.ndarray
    end: np.ndarray
    panel_rect: tuple[float, float, float, float]
    hit_rect: tuple[float, float, float, float]
    source: np.ndarray


@dataclass(frozen=True)
class PreciseGizmoInput:
    """Stable scalar-handle target captured when precise input opens."""

    handle: GizmoHandle
    object_id: int
    node_id: int
    joint_id: int
    action: str
    label: str
    unit: str
    space: str
    absolute_value: float | None
    absolute_label: str
    dimension_index: int = -1


@dataclass(frozen=True)
class _DimensionTarget:
    shape: MeshShape
    size: np.ndarray
    dimensions: GeometryDimensions
    pose_index: int
    node_id: int


@dataclass(frozen=True)
class JointLimitHit:
    """One screen-space joint endpoint tick with its stable write target."""

    joint_id: int
    qpos_adr: int
    value: float
    label: str
    rect: tuple[float, float, float, float]
    semantic_color: tuple[float, float, float, float]
    tick_start: tuple[float, float]
    tick_end: tuple[float, float]
    tick_width: float
    tick_cap: str
    label_anchor: tuple[float, float]
    label_above: bool
    label_align_right: bool


def _gizmo_geometry_key(
    cam: CameraView,
    rect: tuple[float, float, float, float],
    style_scale: float,
    state: _JointRangeState | None,
    position,
    rotation,
) -> tuple:
    """Describe the screen projection inputs shared by hover and drawing."""

    return (
        float(style_scale),
        tuple(float(value) for value in rect),
        tuple(float(value) for value in np.asarray(position).reshape(-1)),
        tuple(float(value) for value in np.asarray(rotation).reshape(-1)),
        None
        if state is None
        else (
            state.joint_type,
            state.current,
            state.lower,
            state.upper,
            state.joint_id,
            state.qpos_adr,
        ),
        tuple(float(value) for value in np.asarray(cam.eye).reshape(-1)),
        tuple(float(value) for value in np.asarray(cam.target).reshape(-1)),
        tuple(float(value) for value in np.asarray(cam.up).reshape(-1)),
        float(cam.fov_y),
        float(cam.near),
        float(cam.far),
        float(cam.aspect),
        bool(cam.orthographic),
        float(cam.ortho_height),
        tuple(float(value) for value in np.asarray(cam.focal_length).reshape(-1)),
        tuple(float(value) for value in np.asarray(cam.sensor_size).reshape(-1)),
        tuple(float(value) for value in np.asarray(cam.principal_offset).reshape(-1)),
        cam.orthographic_blend,
    )


def verdict(paused: bool, node: SceneNode | None) -> Verdict:
    if node is None:
        return Verdict(False, REASON_NO_SELECTION)
    if node.type in (NodeType.LIGHT, NodeType.CAMERA):
        return Verdict(True)
    reason = gizmo_refusal_reason(paused, node.posable)
    return Verdict(reason is None, reason or "")


def _axis_of(handle: GizmoHandle) -> int:
    return {
        GizmoHandle.X: 0,
        GizmoHandle.Y: 1,
        GizmoHandle.Z: 2,
        GizmoHandle.YZ: 0,
        GizmoHandle.ZX: 1,
        GizmoHandle.XY: 2,
        GizmoHandle.ROTATE_X: 0,
        GizmoHandle.ROTATE_Y: 1,
        GizmoHandle.ROTATE_Z: 2,
    }.get(handle, -1)


def _snap_value(value: float, step: float) -> float:
    return float(np.round(float(value) / float(step)) * float(step))


def _snap_translation(delta: np.ndarray, handle: GizmoHandle, step: float) -> np.ndarray:
    snapped = np.asarray(delta, np.float64).copy()
    axes = {
        GizmoHandle.X: (0,),
        GizmoHandle.Y: (1,),
        GizmoHandle.Z: (2,),
        GizmoHandle.YZ: (1, 2),
        GizmoHandle.ZX: (2, 0),
        GizmoHandle.XY: (0, 1),
        GizmoHandle.SCREEN: (0, 1, 2),
    }[handle]
    snapped[list(axes)] = np.round(snapped[list(axes)] / step) * step
    return snapped


def _format_step(value: float) -> str:
    return f"{float(value):g}"
