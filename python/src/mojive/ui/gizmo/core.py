"""Gizmo: core."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from mojive.adapters.base import FrameNeeds, JointInfo, NodeType
from mojive.interaction.gizmo import (
    ALL_HANDLE_MASK,
    AXIS_HANDLES,
    RING_SEGMENTS,
    ROTATE_AXIS_HANDLES,
    ROTATE_HANDLES,
    SIZE_PT,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    GizmoStyle,
    hit_test,
    screen_polygon_distance,
    visibility,
    world_scale,
)
from mojive.types import CameraView

if TYPE_CHECKING:
    from mojive.adapters.base import SceneNode
    from mojive.session import Session


from .dragging import _Dragging
from .drawing import _Drawing
from .guides import _Guides
from .joint_labels import (
    _closest_joint_limit_hit,
    _hinge_current_tick_hit,
    _hinge_range_path_hit,
    _joint_range_handle,
    _point_in_rect,
    _same_joint_limit,
    _slide_current_tick_hit,
    _slide_range_path_hit,
)
from .joint_ranges import _JointRanges
from .precise_input import _PreciseInput
from .state import (
    DEFAULT_ROTATION_SNAP_DEG,
    DEFAULT_ROTATION_TICK_SCALE,
    DEFAULT_TRANSLATION_SNAP_M,
    JOINT_PRECISION_DWELL_GRACE_SECONDS,
    JOINT_PRECISION_REVEAL_DELAY_SECONDS,
    JOINT_PRECISION_REVEAL_GRACE_SECONDS,
    REASON_NO_SELECTION,
    JointLimitHit,
    Verdict,
    _DimensionTarget,
    _gizmo_geometry_key,
    _HingeAxisProjection,
    _HingeRangeProjection,
    _JointPrecisionProjection,
    _JointRangeState,
    _JointTarget,
    _SlideRangeProjection,
)
from .targets import _Targets


class ObjectGizmo(_PreciseInput, _Drawing, _JointRanges, _Guides, _Dragging, _Targets):
    def __init__(self, mode: str = "translate", *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._mode = GizmoMode(mode)
        self._style = GizmoStyle.FLAT
        self._space = GizmoSpace.BODY
        self._hovered = GizmoHandle.NONE
        self._active = GizmoHandle.NONE
        self._verdict = Verdict(False, REASON_NO_SELECTION)
        self._using = False
        self._keyboard = False
        self._guide_gpu = False
        self._interactive = True
        self._display_only = False
        self._visible = False
        self._drawn = False
        self._axis_mask = 0b111
        self._plane_mask = 0b111
        self._handle_mask = ALL_HANDLE_MASK
        self._frame = GizmoFrame()
        self._style_scale = 1.0
        self._hover_cache_signature: tuple | None = None
        self._hover_geometry_signature: tuple | None = None
        self._joint_projection_signature: tuple | None = None
        self._joint_projection: _HingeRangeProjection | _SlideRangeProjection | None = None

        self._start_pos = np.zeros(3, np.float64)
        self._drag_origin_pos = np.zeros(3, np.float64)
        self._start_mat = np.eye(3, dtype=np.float64)
        self._start_basis = np.eye(3, dtype=np.float64)
        self._current_mat = np.eye(3, dtype=np.float64)
        self._start_cursor = np.zeros(2, np.float64)
        self._axis = np.zeros(3, np.float64)
        self._axis_screen = np.zeros(2, np.float64)
        self._world_per_pt = 0.0
        self._slide_cardinal_axis = -1
        self._slide_cardinal_sign = 1.0
        self._slide_last_cursor = np.zeros(2, np.float64)
        self._slide_cardinal_origin_cursor = np.zeros(2, np.float64)
        self._slide_cardinal_origin_travel = 0.0
        self._plane_normal = np.zeros(3, np.float64)
        self._plane_start = np.zeros(3, np.float64)
        self._rotation_start_vec = np.zeros(3, np.float64)
        self._last_rot_vec = np.zeros(3, np.float64)
        self._rotation_screen_ring = np.zeros((RING_SEGMENTS, 2), np.float64)
        self._rotation_screen_ring_valid = False
        self._rotation_screen_tangent = np.zeros(2, np.float64)
        self._rotation_linear = False
        self._rotation_linear_axis = -1
        self._rotation_linear_sign = 1.0
        self._rotation_last_cursor = np.zeros(2, np.float64)
        self._rotation_linear_origin_cursor = np.zeros(2, np.float64)
        self._rotation_linear_origin_angle = 0.0
        self._rotation_raw_angle = 0.0
        self._rotation_angle = 0.0
        self._trackball_angles = np.zeros(2, np.float64)
        self._snapping = False
        self._label = ""
        self._edit_started = False
        self._edit_session: Session | None = None
        self._model_preview: tuple[int, np.ndarray, np.ndarray] | None = None
        self._model_preview_session: Session | None = None
        self._model_placement_model = -1
        self._model_placement_generation = -1
        self._model_placement_session: Session | None = None
        self._model_placement_original: tuple[np.ndarray, np.ndarray] | None = None
        self._joint_selection: dict[int, int] = {}
        self._joint_structure_generation = -1
        self._active_joint: JointInfo | None = None
        self._hinge_axis = False
        self._hinge_axis_signature = None
        self._hinge_axis_geometry: _HingeAxisProjection | None = None
        self._joint_range: _JointRangeState | None = None
        self._joint_limit_hits: tuple[JointLimitHit, ...] = ()
        self._joint_limit_hovered: JointLimitHit | None = None
        self._joint_limit_active: JointLimitHit | None = None
        self._joint_precision: _JointPrecisionProjection | None = None
        self._joint_precision_dwell_key: tuple[int, int] | None = None
        self._joint_precision_dwell_started = 0.0
        self._joint_precision_dwell_last_seen = 0.0
        self._joint_precision_visible_until = 0.0
        self._joint_precision_hovered = False
        self._joint_precision_active = False
        self._start_joint_qpos = np.zeros(0, np.float64)
        self._joint_drag_origin_qpos = np.zeros(0, np.float64)
        self._dimension_start: _DimensionTarget | None = None
        self._dimension_values = np.zeros(3, np.float64)
        self._dimension_cache_session: Session | None = None
        self._dimension_cache_generation = -1
        self._dimension_cache_node = -1
        self._dimension_cache: tuple[_DimensionTarget | None, str] = (None, "")
        self.translation_snap_m = DEFAULT_TRANSLATION_SNAP_M
        self.rotation_snap_deg = DEFAULT_ROTATION_SNAP_DEG
        self.rotation_tick_scale = DEFAULT_ROTATION_TICK_SCALE
        self.remember_precise_input_choices = True

    @property
    def mode(self) -> str:
        return self._mode.value

    @property
    def style(self) -> str:
        return self._style.value

    @property
    def space(self) -> str:
        return self._space.value

    @property
    def last_drawn(self) -> bool:
        return self._drawn

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def interactive(self) -> bool:
        return self._interactive

    @property
    def using(self) -> bool:
        return self._using

    @property
    def keyboard_using(self) -> bool:
        return self._keyboard

    @property
    def last_verdict(self) -> Verdict:
        return self._verdict

    @property
    def hovered(self) -> bool:
        return self._verdict.ok and self._hovered is not GizmoHandle.NONE

    @property
    def hovered_handle(self) -> GizmoHandle:
        return self._hovered

    @property
    def precise_input_hovered(self) -> bool:
        """Return whether the pointer is over a scalar handle that accepts typed input."""

        return bool(
            self.hovered
            and not self._joint_precision_hovered
            and self._joint_limit_hovered is None
            and (
                self._hovered in (*AXIS_HANDLES, *ROTATE_HANDLES)
                or (self._mode is GizmoMode.DIMENSIONS and self._hovered is GizmoHandle.SCREEN)
            )
            and self._hovered is not GizmoHandle.ROTATE_TRACKBALL
        )

    @property
    def display_only(self) -> bool:
        """Return whether the current gizmo is a non-interactive coordinate frame."""

        return self._display_only

    @property
    def joint_limit_hits(self) -> tuple[JointLimitHit, ...]:
        return self._joint_limit_hits

    @property
    def hovered_joint_limit(self) -> JointLimitHit | None:
        return self._joint_limit_hovered

    @property
    def active_joint_limit(self) -> JointLimitHit | None:
        return self._joint_limit_active

    @property
    def joint_precision_visible(self) -> bool:
        return self._joint_precision is not None and (
            self._joint_precision_active
            or self._joint_precision_hovered
            or time.monotonic() <= self._joint_precision_visible_until
        )

    @property
    def joint_precision_hit_rect(self) -> tuple[float, float, float, float] | None:
        projection = self._joint_precision
        return None if projection is None else projection.hit_rect

    @property
    def active_handle(self) -> GizmoHandle:
        return self._active

    @property
    def value_label(self) -> str:
        return self._label

    @property
    def snapping(self) -> bool:
        return self._snapping

    def frame_needs(self, session: Session) -> FrameNeeds:
        node = session.selected_node
        target, _reason = self._joint_target(session, node)
        if target is None:
            return FrameNeeds.none()
        return FrameNeeds(poses=False, qpos=True, joint_frames=True)

    def selected_joint_id(self, body_index: int) -> int:
        return self._joint_selection.get(int(body_index), -1)

    def joint_choices(self, session: Session) -> tuple[JointInfo, ...]:
        """Return direct joints that need an explicit viewport gizmo choice."""

        node = session.selected_node
        if node is None or node.posable or node.type not in (NodeType.LINK, NodeType.ROBOT):
            return ()
        joints = tuple(session.joints_for_body(node.body_index))
        return joints if len(joints) > 1 else ()

    def select_joint(self, body_index: int, joint_id: int) -> None:
        body = int(body_index)
        joint = int(joint_id)
        if self._joint_selection.get(body) == joint:
            return
        self._end()
        self._joint_selection[body] = joint

    def reveal_joint_precision(
        self,
        hit: JointLimitHit,
        *,
        now: float | None = None,
    ) -> bool:
        """Keep the compact-range rail visible after an endpoint dwell."""

        projection = self._joint_precision
        if projection is None or (
            projection.joint_id != hit.joint_id or projection.qpos_adr != hit.qpos_adr
        ):
            return False
        timestamp = time.monotonic() if now is None else float(now)
        self._joint_precision_visible_until = max(
            self._joint_precision_visible_until,
            timestamp + JOINT_PRECISION_REVEAL_GRACE_SECONDS,
        )
        return True

    def set_mode(self, mode: str) -> None:
        """Explicit programmatic tool selection also enables that tool."""
        if mode in tuple(item.value for item in GizmoMode) and not self._using:
            self._mode = GizmoMode(mode)
            self.enabled = True

    def toggle_mode(self, mode: str) -> None:
        """Toggle an editor tool without changing scene selection."""
        if self._using:
            return
        if self.enabled and self._mode.value == mode:
            self.enabled = False
            self.cancel()
        else:
            self.set_mode(mode)

    def set_style(self, style: str) -> None:
        if style in (GizmoStyle.FLAT.value, GizmoStyle.SOLID.value) and not self._using:
            self._style = GizmoStyle(style)

    def set_space(self, space: str) -> None:
        if space in (GizmoSpace.BODY.value, GizmoSpace.WORLD.value) and not self._using:
            self._space = GizmoSpace(space)

    def toggle_space(self) -> None:
        self.set_space("world" if self._space is GizmoSpace.BODY else "body")

    def cancel(self) -> None:
        self._end()

    def update_hover(
        self,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        cursor: tuple[float, float],
        *,
        enabled: bool = True,
        style_scale: float = 1.0,
    ) -> GizmoHandle:
        self._style_scale = float(style_scale)
        node = session.selected_node
        self._verdict = self.evaluate(session, node)
        if not self._verdict.ok:
            self._hovered = GizmoHandle.NONE
            self._joint_limit_hovered = None
            self._joint_precision_hovered = False
            self._axis_mask = self._plane_mask = 0
            self._hover_cache_signature = None
            self._hover_geometry_signature = None
            return self._hovered
        if self._active is not GizmoHandle.NONE:
            self._hovered = self._active
            return self._hovered
        if not enabled:
            self._hovered = GizmoHandle.NONE
            self._joint_limit_hovered = None
            self._joint_precision_hovered = False
            self._hover_cache_signature = None
            self._hover_geometry_signature = None
            return self._hovered
        target, _reason = self._joint_target(session, node)
        dimension_target = None
        if self._mode is GizmoMode.DIMENSIONS:
            dimension_target, _reason = self._dimension_target(session, node)
        pose = (
            self._dimension_pose(session, node, dimension_target)
            if dimension_target is not None
            else self._target_pose(session, node, target)
        )
        if pose is None:
            self._hovered = GizmoHandle.NONE
            self._joint_limit_hovered = None
            self._joint_precision_hovered = False
            self._hover_cache_signature = None
            self._hover_geometry_signature = None
            return self._hovered
        pos, mat = pose
        mode = target.mode if target is not None else self._mode
        self._handle_mask = (
            self._dimension_handle_mask(dimension_target.dimensions)
            if dimension_target is not None
            else target.handles
            if target is not None
            else ALL_HANDLE_MASK
        )
        range_state = self._joint_range_state(session, target)
        basis = self._target_basis(mat, target)
        signature = self._hover_signature(
            session,
            node,
            target,
            range_state,
            cam,
            rect,
            cursor,
            pos,
            mat,
            mode,
            style_scale,
        )
        cached = signature == self._hover_cache_signature
        geometry_signature = signature[0]
        geometry_cached = geometry_signature == self._hover_geometry_signature
        range_projection = self._joint_projection
        if not geometry_cached:
            range_projection = self._joint_range_projection(
                cam,
                rect,
                self._style_scale,
                range_state,
                pos,
                basis,
            )
            self._joint_precision = self._joint_precision_projection(
                rect,
                self._style_scale,
                range_state,
                range_projection,
            )
            self._hover_geometry_signature = geometry_signature
            if target is not None and range_state is not None:
                scale = world_scale(cam, pos, rect[3], SIZE_PT * self._style_scale)
                self._axis_mask, self._plane_mask = visibility(cam, pos, basis, rect, scale)
        now = time.monotonic()
        precision_visible = bool(
            self._joint_precision is not None
            and (
                self._joint_precision_active
                or self._joint_precision_hovered
                or now <= self._joint_precision_visible_until
            )
        )
        self._joint_precision_hovered = bool(
            precision_visible
            and self._joint_precision is not None
            and _point_in_rect(cursor, self._joint_precision.hit_rect)
        )
        if self._joint_precision_hovered:
            self._joint_precision_visible_until = now + JOINT_PRECISION_REVEAL_GRACE_SECONDS
            self._update_joint_precision_dwell(True, now)
            self._joint_limit_hovered = None
            self._hovered = _joint_range_handle(range_state)
            self._hover_cache_signature = signature
            return self._hovered
        if cached:
            self._update_joint_precision_dwell(self._hovered is not GizmoHandle.NONE, now)
            return self._hovered
        axis_hit = False
        if target is not None and target.joint.type == "hinge":
            geometry = self._hinge_axis_projection(cam, rect, style_scale, pos, basis[:, 2])
            axis_hit = (
                geometry is not None
                and screen_polygon_distance(cursor, geometry.hit_polygon) <= 4.0 * style_scale
            )
        if target is not None and range_state is not None:
            self._hovered = GizmoHandle.NONE
            range_hit = False
            current_hit = False
            arrow_hit = False
            if target.joint.type == "hinge":
                hinge = (
                    range_projection
                    if isinstance(range_projection, _HingeRangeProjection)
                    else None
                )
                range_hit = _hinge_range_path_hit(cursor, hinge, self._style_scale)
                current_hit = _hinge_current_tick_hit(cursor, hinge, self._style_scale)
            else:
                slide = (
                    range_projection
                    if isinstance(range_projection, _SlideRangeProjection)
                    else None
                )
                range_hit = _slide_range_path_hit(cursor, slide, self._style_scale)
                current_hit = _slide_current_tick_hit(cursor, slide, self._style_scale)
            limit_hit = _closest_joint_limit_hit(
                cursor,
                self._joint_limit_hits,
                range_state,
                self._style_scale,
            )
            # A hinge range owns the inner crossing of its outward endpoint
            # ticks. A slide endpoint spans both sides of the axis, so its two
            # outer sections own the pointer after the center section has been
            # excluded by _closest_joint_limit_hit().
            limit_owns = limit_hit is not None and (
                target.joint.type == "slide" or not (range_hit or axis_hit)
            )
            if range_hit and not limit_owns:
                self._hovered = _joint_range_handle(range_state)
                self._joint_limit_hovered = None
            elif limit_owns:
                self._hovered = _joint_range_handle(range_state)
                self._joint_limit_hovered = limit_hit
            elif current_hit or axis_hit:
                self._hovered = _joint_range_handle(range_state)
                self._joint_limit_hovered = None
            else:
                self._joint_limit_hovered = None
            if target.joint.type == "slide" and slide is not None and not range_hit:
                polygons = self._slide_arrow_polygons(slide, self._style_scale, for_hit_test=True)
                arrow_hit = any(
                    screen_polygon_distance(cursor, polygon) <= 4.0 * self._style_scale
                    for polygon in polygons
                )
                if arrow_hit:
                    self._hovered = GizmoHandle.Z
            self._update_joint_precision_dwell(
                range_hit or current_hit or limit_hit is not None or arrow_hit or axis_hit,
                now,
            )
            self._hover_cache_signature = signature
            return self._hovered
        self._joint_limit_hovered = None
        if axis_hit:
            self._hovered = GizmoHandle.ROTATE_Z
            scale = world_scale(cam, pos, rect[3], SIZE_PT * style_scale)
            self._axis_mask, self._plane_mask = visibility(cam, pos, basis, rect, scale)
            self._hover_cache_signature = signature
            return self._hovered
        self._hovered, self._axis_mask, self._plane_mask = hit_test(
            cam,
            pos,
            basis,
            rect,
            cursor,
            mode,
            self._style_scale,
            self._handle_mask,
            self._hovered,
        )
        self._update_joint_precision_dwell(False, now)
        self._hover_cache_signature = signature
        return self._hovered

    def _update_joint_precision_dwell(
        self,
        hovered: bool,
        now: float,
    ) -> None:
        """Reveal a compact rail after dwelling anywhere on the joint gizmo."""

        projection = self._joint_precision
        if projection is None:
            self._joint_precision_dwell_key = None
            return
        key = (projection.joint_id, projection.qpos_adr)
        if hovered:
            if key != self._joint_precision_dwell_key:
                self._joint_precision_dwell_key = key
                self._joint_precision_dwell_started = now
            self._joint_precision_dwell_last_seen = now
        elif key != self._joint_precision_dwell_key or (
            now - self._joint_precision_dwell_last_seen > JOINT_PRECISION_DWELL_GRACE_SECONDS
        ):
            self._joint_precision_dwell_key = None
            return
        if now - self._joint_precision_dwell_started >= JOINT_PRECISION_REVEAL_DELAY_SECONDS:
            self._joint_precision_visible_until = now + JOINT_PRECISION_REVEAL_GRACE_SECONDS

    def _hover_signature(
        self,
        session: Session,
        node: SceneNode,
        target: _JointTarget | None,
        range_state: _JointRangeState | None,
        cam: CameraView,
        rect,
        cursor,
        position,
        rotation,
        mode: GizmoMode,
        style_scale: float,
    ) -> tuple:
        """Describe every input that can change idle gizmo hit testing."""

        joint = target.joint if target is not None else None
        return (
            (
                id(session),
                session.structure_generation,
                session.paused,
                int(node.node_id),
                int(joint.joint_id) if joint is not None else -1,
                joint.type if joint is not None else "",
                int(self._handle_mask),
                mode.value,
                self._space.value,
                _gizmo_geometry_key(
                    cam,
                    rect,
                    style_scale,
                    range_state,
                    position,
                    rotation,
                ),
            ),
            tuple(float(value) for value in cursor),
        )

    def interact(
        self,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        cursor: tuple[float, float],
        *,
        claimed: bool,
        left_down: bool,
        released: bool,
        snap: bool = False,
        style_scale: float = 1.0,
    ) -> bool:
        self._style_scale = float(style_scale)
        if self._joint_limit_active is not None:
            if not claimed:
                self._joint_limit_active = None
                return False
            if released or not left_down:
                active = self._joint_limit_active
                hovered = self._joint_limit_hovered
                self._joint_limit_active = None
                if hovered is not None and _same_joint_limit(active, hovered):
                    result = self.apply_joint_limit(session, active)
                    if not result.ok:
                        self._verdict = Verdict(False, result.message)
                        session.report_message(result.message, level="warning")
                    return result.ok
                return False
            return True
        if not claimed:
            if self._using and not left_down:
                self._end(commit=True)
            return False
        if released or not left_down:
            self._end(commit=True)
            return False
        if self._active is GizmoHandle.NONE:
            if self._joint_limit_hovered is not None:
                self._joint_limit_active = self._joint_limit_hovered
                return True
            if self._joint_precision_hovered:
                if not self._begin_joint_precision(session, cam, rect, cursor):
                    return False
            elif not self._begin(session, cam, rect, cursor):
                return False
        self._using = True
        return self._drag(session, cam, rect, cursor, snap=snap)

    def keyboard_interact(
        self,
        session,
        cam,
        rect,
        cursor,
        axis: int,
        *,
        snap: bool = False,
        style_scale: float = 1.0,
    ) -> bool:
        self._style_scale = float(style_scale)
        if axis not in (0, 1, 2):
            if self._keyboard:
                self._end(commit=True)
            return False
        node = session.selected_node
        target, _reason = self._joint_target(session, node)
        mode = target.mode if target is not None else self._mode
        handle = ROTATE_AXIS_HANDLES[axis] if mode is GizmoMode.ROTATE else AXIS_HANDLES[axis]
        dimension_target = None
        if mode is GizmoMode.DIMENSIONS:
            dimension_target, _reason = self._dimension_target(session, node)
        allowed = (
            self._dimension_handle_mask(dimension_target.dimensions)
            if dimension_target is not None
            else target.handles
            if target is not None
            else ALL_HANDLE_MASK
        )
        if not allowed & (1 << int(handle)):
            return False
        if self._keyboard and self._active is not handle:
            self._end(commit=True)
        if not self._keyboard:
            self._verdict = self.evaluate(session, node)
            if not self._verdict.ok or not self._begin_handle(session, cam, rect, cursor, handle):
                return False
            self._keyboard = self._using = True
            return True
        return self._drag(session, cam, rect, cursor, snap=snap)
