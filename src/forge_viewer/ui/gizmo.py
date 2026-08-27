"""Interactive position and rotation gizmo behavior."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import math3d
from ..adapters.base import FrameNeeds, JointInfo, NodeType
from ..commands import (
    BeginEditTransaction,
    CommandResult,
    EndEditTransaction,
    SetLight,
    SetPose,
    SetQpos,
    SetQposBatch,
    SetSceneCamera,
)
from ..gizmo import (
    ALL_HANDLE_MASK,
    AXIS_COLORS,
    AXIS_END,
    AXIS_HANDLES,
    AXIS_START,
    CENTER_COLOR,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    GUIDE_CORE_COLOR,
    HOVER_COLOR,
    JOINT_HANDLE_COLOR,
    PLANE_ACTIVE_ALPHA,
    PLANE_ALPHA,
    PLANE_HANDLES,
    RING_RADIUS,
    RING_SEGMENTS,
    RING_WIDTH_PT,
    ROTATE_AXIS_HANDLES,
    ROTATE_HANDLES,
    SCREEN_RING_RADIUS,
    SCREEN_RING_WIDTH_PT,
    SIZE_PT,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    GizmoStyle,
    axis_arrow_polygon,
    axis_handle_alpha,
    display_handles,
    handle_mask,
    hit_test,
    masked_axis_start,
    paint_order,
    plane_corners,
    plane_direction,
    plane_handle_alpha,
    project,
    rotation_dial,
    rotation_ring,
    rotation_ring_alpha,
    rotation_ring_is_full,
    visibility,
    world_scale,
)
from ..render.debugdraw import Occlusion
from ..types import CameraView, LightType
from .camera import ndc_from_viewport, unproject
from .draw2d import Draw2D
from .panels.inspector import gizmo_refusal_reason
from .scene_entities import camera_rotation, direction_basis

if TYPE_CHECKING:
    from ..adapters.base import SceneNode
    from ..session import Session

REASON_NO_SELECTION = "nothing selected"
DRAG_LAYER = "ui.gizmo.drag"
_WORLD_BASIS = np.eye(3, dtype=np.float64)
DEFAULT_TRANSLATION_SNAP_M = 0.1
DEFAULT_ROTATION_SNAP_DEG = 5.0
DEFAULT_ROTATION_TICK_SCALE = 1.25
SNAP_TICK_FULL_STEPS = 5.0
SNAP_TICK_FADE_STEPS = 10.0
ROTATION_TICK_MIN_ALPHA = 0.5
JOINT_RANGE_RADIUS = 0.90
JOINT_RANGE_WIDTH_PT = 3.0
JOINT_RANGE_OFFSET_PT = 13.0
JOINT_RANGE_COLOR = (0.30, 0.78, 0.46, 0.96)
JOINT_RANGE_UNAVAILABLE_COLOR = (0.34, 0.36, 0.40, 0.68)
JOINT_LOWER_LIMIT_COLOR = (0.30, 0.58, 1.00, 1.0)
JOINT_UPPER_LIMIT_COLOR = (1.00, 0.34, 0.28, 1.0)


class _RotationDialProjector:
    """Project every active rotation-dial layer through one shared mapping."""

    def __init__(
        self,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        center,
        axis,
        start_direction,
        size_px: float,
    ) -> None:
        self._cam = cam
        self._rect = rect
        self._center = np.asarray(center, np.float64)
        self._axis = np.asarray(axis, np.float64)
        self._start_direction = np.asarray(start_direction, np.float64)
        self._world_scale = world_scale(cam, center, rect[3], size_px)

    def points(self, radius, angles) -> np.ndarray:
        angles = np.atleast_1d(np.asarray(angles, np.float64))
        return project(
            self._cam,
            rotation_dial(
                self._center,
                self._axis,
                self._start_direction,
                self._world_scale,
                radius,
                angles,
            ),
            self._rect,
        )

    def tick(
        self,
        radius: float,
        angle: float,
        length_px: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        points = self.points((radius, radius + 1.0 / SIZE_PT), (angle, angle))
        direction = points[1, :2] - points[0, :2]
        projected_length = float(np.linalg.norm(direction))
        if np.any(points[:, 2] <= 0.0) or projected_length < 1e-6:
            return None
        radial = direction / projected_length
        start = points[0, :2]
        return start, start + radial * float(length_px)


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


def verdict(paused: bool, node: SceneNode | None) -> Verdict:
    if node is None:
        return Verdict(False, REASON_NO_SELECTION)
    if node.type in (NodeType.LIGHT, NodeType.CAMERA):
        return Verdict(True)
    reason = gizmo_refusal_reason(paused, node.posable)
    return Verdict(reason is None, reason or "")


class ObjectGizmo:
    def __init__(self, mode: str = "translate") -> None:
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
        self._visible = False
        self._drawn = False
        self._axis_mask = 0b111
        self._plane_mask = 0b111
        self._handle_mask = ALL_HANDLE_MASK
        self._frame = GizmoFrame()
        self._style_scale = 1.0

        self._start_pos = np.zeros(3, np.float64)
        self._start_mat = np.eye(3, dtype=np.float64)
        self._start_basis = np.eye(3, dtype=np.float64)
        self._current_mat = np.eye(3, dtype=np.float64)
        self._start_cursor = np.zeros(2, np.float64)
        self._axis = np.zeros(3, np.float64)
        self._axis_screen = np.zeros(2, np.float64)
        self._world_per_pt = 0.0
        self._plane_normal = np.zeros(3, np.float64)
        self._plane_start = np.zeros(3, np.float64)
        self._rotation_start_vec = np.zeros(3, np.float64)
        self._last_rot_vec = np.zeros(3, np.float64)
        self._rotation_raw_angle = 0.0
        self._rotation_angle = 0.0
        self._snapping = False
        self._label = ""
        self._edit_started = False
        self._edit_session: Session | None = None
        self._joint_selection: dict[int, int] = {}
        self._joint_structure_generation = -1
        self._active_joint: JointInfo | None = None
        self._joint_range: _JointRangeState | None = None
        self._start_joint_qpos = np.zeros(0, np.float64)
        self.translation_snap_m = DEFAULT_TRANSLATION_SNAP_M
        self.rotation_snap_deg = DEFAULT_ROTATION_SNAP_DEG
        self.rotation_tick_scale = DEFAULT_ROTATION_TICK_SCALE

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
        return FrameNeeds(poses=False, qpos=True, diagnostics=True)

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

    def set_mode(self, mode: str) -> None:
        if mode in (GizmoMode.TRANSLATE.value, GizmoMode.ROTATE.value) and not self._using:
            self._mode = GizmoMode(mode)

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

    def precise_input(self, session: Session) -> PreciseGizmoInput | None:
        """Describe the hovered scalar handle for a relative numeric edit."""

        node = session.selected_node
        handle = self._hovered
        if node is None or handle not in (*AXIS_HANDLES, *ROTATE_HANDLES):
            return None
        if not self.evaluate(session, node).ok:
            return None
        target, _reason = self._joint_target(session, node)
        mode = target.mode if target is not None else self._mode
        if mode is GizmoMode.TRANSLATE and handle not in AXIS_HANDLES:
            return None
        if mode is GizmoMode.ROTATE and handle not in ROTATE_HANDLES:
            return None
        allowed = target.handles if target is not None else ALL_HANDLE_MASK
        if not allowed & (1 << int(handle)):
            return None

        axis = _axis_of(handle)
        handle_name = "Screen" if handle is GizmoHandle.ROTATE_SCREEN else "XYZ"[axis]
        joint_id = -1
        label = handle_name
        if target is not None:
            joint = target.joint
            joint_id = int(joint.joint_id)
            joint_name = joint.name or joint.type
            label = (
                joint_name if joint.type in ("hinge", "slide") else f"{joint_name} · {handle_name}"
            )
        rotating = handle in ROTATE_HANDLES
        return PreciseGizmoInput(
            handle=handle,
            object_id=int(session.selected),
            node_id=int(node.node_id),
            joint_id=joint_id,
            action="Rotate" if rotating else "Move",
            label=label,
            unit="°" if rotating else "m",
        )

    def apply_precise_delta(
        self,
        session: Session,
        cam: CameraView,
        edit: PreciseGizmoInput,
        value: float,
    ) -> CommandResult:
        """Apply one exact scalar delta through the regular transform commands."""

        amount = float(value)
        if not np.isfinite(amount):
            return CommandResult.bad("Enter a finite numeric delta")
        node = session.selected_node
        if (
            node is None
            or int(session.selected) != edit.object_id
            or int(node.node_id) != edit.node_id
        ):
            return CommandResult.bad("The gizmo target changed; reopen precise input")
        available = self.evaluate(session, node)
        if not available.ok:
            return CommandResult.bad(available.reason)
        target, reason = self._joint_target(session, node)
        joint_id = -1 if target is None else int(target.joint.joint_id)
        if joint_id != edit.joint_id:
            return CommandResult.bad("The selected joint changed; reopen precise input")
        allowed = target.handles if target is not None else ALL_HANDLE_MASK
        if not allowed & (1 << int(edit.handle)):
            return CommandResult.bad(reason or "This gizmo handle is no longer available")

        pose = self._target_pose(session, node, target)
        if pose is None:
            return CommandResult.bad("Gizmo frame data is unavailable")
        position, rotation = pose
        position = np.asarray(position, np.float64).copy()
        rotation = np.asarray(rotation, np.float64).reshape(3, 3).copy()
        basis = self._target_basis(rotation, target)
        axis_index = _axis_of(edit.handle)
        if edit.handle in AXIS_HANDLES:
            if axis_index < 0:
                return CommandResult.bad("Precise input requires a single translation axis")
            axis = basis[:, axis_index]
            position += axis * amount
        elif edit.handle in ROTATE_HANDLES:
            axis = (
                -np.asarray(cam.forward(), np.float64)
                if edit.handle is GizmoHandle.ROTATE_SCREEN
                else basis[:, axis_index]
            )
            rotation = math3d.rotvec_to_mat3(axis * np.radians(amount)) @ rotation
        else:
            return CommandResult.bad("Precise input requires a scalar gizmo handle")

        if abs(amount) < 1e-12:
            return CommandResult.good("No change")
        if target is not None:
            joint = target.joint
            qpos = session.frame.qpos
            count = 4 if joint.type == "ball" else 1
            start = int(joint.qpos_adr)
            if qpos is None or start < 0 or start + count > len(qpos):
                return CommandResult.bad("Joint position data is unavailable")
            self._start_joint_qpos = np.asarray(qpos[start : start + count], np.float64).copy()
        self._active = edit.handle
        self._active_joint = target.joint if target is not None else None
        np.copyto(self._start_pos, pose[0])
        np.copyto(self._start_mat, pose[1])
        np.copyto(self._start_basis, basis)
        self._axis[:] = axis
        self._rotation_angle = np.radians(amount) if edit.handle in ROTATE_HANDLES else 0.0
        result, _position = self._submit_transform(session, node, position, rotation)
        self._end()
        if not result.ok:
            self._verdict = Verdict(False, result.message)
        return result

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
            self._axis_mask = self._plane_mask = 0
            return self._hovered
        if self._active is not GizmoHandle.NONE:
            self._hovered = self._active
            return self._hovered
        if not enabled:
            self._hovered = GizmoHandle.NONE
            return self._hovered
        target, _reason = self._joint_target(session, node)
        pose = self._target_pose(session, node, target)
        if pose is None:
            self._hovered = GizmoHandle.NONE
            return self._hovered
        pos, mat = pose
        mode = target.mode if target is not None else self._mode
        self._handle_mask = target.handles if target is not None else ALL_HANDLE_MASK
        self._hovered, self._axis_mask, self._plane_mask = hit_test(
            cam,
            pos,
            self._target_basis(mat, target),
            rect,
            cursor,
            mode,
            self._style_scale,
            self._handle_mask,
        )
        return self._hovered

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
        if not claimed:
            if self._using and not left_down:
                self._end()
            return False
        if released or not left_down:
            self._end()
            return False
        if self._active is GizmoHandle.NONE and not self._begin(session, cam, rect, cursor):
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
                self._end()
            return False
        node = session.selected_node
        target, _reason = self._joint_target(session, node)
        mode = target.mode if target is not None else self._mode
        handle = AXIS_HANDLES[axis] if mode is GizmoMode.TRANSLATE else ROTATE_AXIS_HANDLES[axis]
        allowed = target.handles if target is not None else ALL_HANDLE_MASK
        if not allowed & (1 << int(handle)):
            return False
        if self._keyboard and self._active is not handle:
            self._end()
        if not self._keyboard:
            self._verdict = self.evaluate(session, node)
            if not self._verdict.ok or not self._begin_handle(session, cam, rect, cursor, handle):
                return False
            self._keyboard = self._using = True
            return True
        return self._drag(session, cam, rect, cursor, snap=snap)

    def publish(
        self,
        backend: Any,
        session: Session,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        *,
        ui_scale: float,
        style_scale: float,
        yielding: bool,
        interactive: bool,
    ) -> bool:
        self._interactive = bool(interactive)
        self._joint_range = None
        node = session.selected_node
        self._verdict = self.evaluate(session, node)
        self._visible = not yielding and self._verdict.ok
        if not self._visible:
            self._clear_translation_guide(backend)
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return False
        target, _reason = self._joint_target(session, node)
        pose = self._target_pose(session, node, target)
        if pose is None:
            self._drawn = False
            return False
        pos, mat = pose
        self._joint_range = self._joint_range_state(session, target)
        mode = target.mode if target is not None else self._mode
        self._handle_mask = target.handles if target is not None else ALL_HANDLE_MASK
        active = self._active is not GizmoHandle.NONE
        if active:
            basis = self._start_basis
            if self._active in ROTATE_HANDLES:
                pos = self._start_pos
        else:
            basis = self._target_basis(mat, target)
        scale = world_scale(cam, pos, rect[3], SIZE_PT * float(style_scale))
        self._axis_mask, self._plane_mask = visibility(cam, pos, basis, rect, scale)
        frame = self._frame
        frame.mode = mode
        frame.style = self._style
        frame.space = self._space
        np.copyto(frame.position, pos, casting="unsafe")
        np.copyto(frame.rotation, basis, casting="unsafe")
        frame.size_px = SIZE_PT * float(ui_scale)
        frame.hovered = self._hovered if interactive else GizmoHandle.NONE
        frame.active = self._active
        frame.active_rotation_overlay = self._using and self._active in ROTATE_HANDLES
        frame.axis_mask = self._axis_mask
        frame.plane_mask = self._plane_mask
        frame.handle_mask = self._handle_mask
        frame.handle_color = (
            JOINT_HANDLE_COLOR
            if target is not None and target.joint.type in ("hinge", "slide")
            else None
        )
        self._publish_translation_guide(backend, ui_scale)
        if self._style is GizmoStyle.FLAT:
            if backend.caps.gizmo:
                backend.set_gizmo(None)
            self._drawn = False
            return True
        if not backend.caps.gizmo:
            self._drawn = False
            return False
        self._drawn = bool(backend.set_gizmo(frame))
        return self._drawn

    def draw_overlay(self, cam, rect, overlay: Draw2D, *, style_scale: float = 1.0) -> None:
        if not self._visible:
            return
        if self._keyboard and not self._snapping:
            self._draw_axis_constraint(overlay, cam, rect, style_scale)
        if self._style is GizmoStyle.FLAT:
            self._draw_flat(overlay, cam, rect, style_scale)
            self._drawn = True
        if self._joint_range is not None:
            self._draw_joint_range(overlay, cam, rect, style_scale)
        if self._using and self._snapping and self._active in AXIS_HANDLES:
            self._draw_translation_snap_ruler(overlay, cam, rect, style_scale)
        if self._using and self._active not in ROTATE_HANDLES and not self._guide_gpu:
            self._draw_translation_guide(overlay, cam, rect, style_scale)
        rotation_dial_projector = None
        if self._using and self._active in ROTATE_HANDLES:
            rotation_dial_projector = _RotationDialProjector(
                cam,
                rect,
                self._start_pos,
                self._axis,
                self._rotation_start_vec,
                SIZE_PT * style_scale,
            )
            if self._snapping:
                self._draw_rotation_snap_ticks(
                    overlay, cam, rect, style_scale, rotation_dial_projector
                )
            self._draw_rotation_guide(overlay, cam, rect, style_scale, rotation_dial_projector)
        if self._using and self._label:
            self._draw_value_label(overlay, cam, rect, style_scale, rotation_dial_projector)

    def _draw_flat(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        frame = self._frame
        origin = np.asarray(frame.position, np.float64)
        rotation = np.asarray(frame.rotation, np.float64)
        scale = world_scale(cam, origin, rect[3], SIZE_PT * style_scale)
        visible = display_handles(frame)

        # The draw list paints in submission order with no depth buffer, so
        # each handle group draws far-to-near (painter's order) to put the
        # nearer handle on top where handles overlap.
        planes = [axis for axis, handle in enumerate(PLANE_HANDLES) if handle in visible]
        for k in paint_order(cam, origin, [plane_direction(rotation, axis) for axis in planes]):
            axis = planes[k]
            handle = PLANE_HANDLES[axis]
            alpha = (
                1.0
                if frame.active is handle
                else plane_handle_alpha(cam, origin, rotation[:, axis])
            )
            if alpha <= 0.0:
                continue
            screen = project(cam, plane_corners(origin, rotation, scale, axis), rect)
            if np.any(screen[:, 2] <= 0.0):
                continue
            opacity = PLANE_ACTIVE_ALPHA if frame.active is handle else PLANE_ALPHA * alpha
            overlay.convex_fill(screen[:, :2], self._flat_color(handle, axis, opacity))

        axes = [axis for axis, handle in enumerate(AXIS_HANDLES) if handle in visible]
        for k in paint_order(cam, origin, [rotation[:, axis] for axis in axes]):
            axis = axes[k]
            handle = AXIS_HANDLES[axis]
            alpha = (
                1.0 if frame.active is handle else axis_handle_alpha(cam, origin, rotation[:, axis])
            )
            if alpha <= 0.0:
                continue
            screen = project(
                cam,
                (
                    origin + rotation[:, axis] * scale * AXIS_START,
                    origin + rotation[:, axis] * scale * AXIS_END,
                ),
                rect,
            )
            if np.any(screen[:, 2] <= 0.0):
                continue
            start = masked_axis_start(
                screen[0, :2],
                screen[1, :2],
                CENTER_SHELL_RADIUS * SIZE_PT * style_scale,
            )
            points = axis_arrow_polygon(start, screen[1, :2], style_scale)
            if len(points):
                overlay.concave_fill(points, self._flat_color(handle, axis, alpha))

        if GizmoHandle.SCREEN in visible:
            center = project(cam, (origin,), rect)[0]
            if center[2] > 0.0:
                color = HOVER_COLOR if self._hot(GizmoHandle.SCREEN) else CENTER_COLOR
                radius = CENTER_RADIUS * SIZE_PT * style_scale
                overlay.circle_filled(
                    center[:2],
                    radius + CONTRAST_EDGE_PT * style_scale,
                    CONTRAST_EDGE_COLOR,
                    segments=24,
                )
                overlay.circle_filled(center[:2], radius, color, segments=24)

        for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
            if handle not in visible:
                continue
            if frame.active_rotation_overlay and frame.active is handle:
                continue
            full = rotation_ring_is_full(frame, handle)
            alpha = (
                1.0
                if frame.active is handle
                else rotation_ring_alpha(cam, origin, rotation[:, axis])
            )
            if alpha <= 0.0:
                continue
            ring = rotation_ring(cam, origin, rotation, scale, axis, full=full)
            screen = project(cam, ring, rect)
            if np.any(screen[:, 2] <= 0.0):
                continue
            overlay.polyline(
                screen[:, :2],
                self._flat_color(handle, axis, alpha),
                RING_WIDTH_PT * style_scale,
                closed=full,
            )

        if GizmoHandle.ROTATE_SCREEN in visible and not (
            frame.active_rotation_overlay and frame.active is GizmoHandle.ROTATE_SCREEN
        ):
            center = project(cam, (origin,), rect)[0]
            if center[2] > 0.0:
                color = HOVER_COLOR if self._hot(GizmoHandle.ROTATE_SCREEN) else CENTER_COLOR
                radius = SCREEN_RING_RADIUS * SIZE_PT * style_scale
                overlay.circle(
                    center[:2],
                    radius,
                    CONTRAST_EDGE_COLOR,
                    (SCREEN_RING_WIDTH_PT + 2.0 * CONTRAST_EDGE_PT) * style_scale,
                    segments=RING_SEGMENTS,
                )
                overlay.circle(
                    center[:2],
                    radius,
                    color,
                    SCREEN_RING_WIDTH_PT * style_scale,
                    segments=RING_SEGMENTS,
                )

    def _flat_color(self, handle: GizmoHandle, axis: int, alpha: float = 1.0):
        color = HOVER_COLOR if self._hot(handle) else self._handle_color(axis)
        return float(color[0]), float(color[1]), float(color[2]), float(alpha)

    def _handle_color(self, axis: int) -> np.ndarray:
        color = self._frame.handle_color
        return AXIS_COLORS[axis] if color is None else np.asarray(color, np.float32)

    def _hot(self, handle: GizmoHandle) -> bool:
        return self._active is handle or (self._interactive and self._hovered is handle)

    def _draw_joint_range(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        state = self._joint_range
        if state is None:
            return
        if state.joint_type == "hinge":
            self._draw_hinge_range(overlay, cam, rect, style_scale, state)
        elif state.joint_type == "slide":
            self._draw_slide_range(overlay, cam, rect, style_scale, state)

    def _draw_hinge_range(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState,
    ) -> None:
        frame = self._frame
        origin = np.asarray(frame.position, np.float64)
        rotation = np.asarray(frame.rotation, np.float64)
        dial = _RotationDialProjector(
            cam,
            rect,
            origin,
            rotation[:, 2],
            rotation[:, 0],
            SIZE_PT * style_scale,
        )
        segments = _rotation_dial_segments(cam, origin, rotation[:, 2])
        full_angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        full_ring = dial.points(JOINT_RANGE_RADIUS, full_angles)
        if np.all(full_ring[:, 2] > 0.0):
            overlay.polyline(
                full_ring[:, :2],
                JOINT_RANGE_UNAVAILABLE_COLOR,
                JOINT_RANGE_WIDTH_PT * style_scale,
                closed=True,
            )

        span = min(state.upper - state.lower, 2.0 * np.pi)
        start_angle = state.lower
        point_count = max(2, int(np.ceil(segments * span / (2.0 * np.pi))) + 1)
        allowed_angles = np.linspace(start_angle, start_angle + span, point_count)
        allowed = dial.points(JOINT_RANGE_RADIUS, allowed_angles)
        if np.all(allowed[:, 2] > 0.0):
            overlay.polyline(
                allowed[:, :2],
                JOINT_RANGE_COLOR,
                JOINT_RANGE_WIDTH_PT * style_scale,
                closed=span >= 2.0 * np.pi - 1e-6,
            )

        self._draw_hinge_limit(
            overlay,
            dial,
            start_angle,
            JOINT_LOWER_LIMIT_COLOR,
            _joint_limit_label("MIN", state.lower, "hinge"),
            style_scale,
            label_above=True,
        )
        self._draw_hinge_limit(
            overlay,
            dial,
            start_angle + span,
            JOINT_UPPER_LIMIT_COLOR,
            _joint_limit_label("MAX", state.upper, "hinge"),
            style_scale,
            label_above=False,
        )
        current_tick = dial.tick(JOINT_RANGE_RADIUS, state.current, 5.0 * style_scale)
        if current_tick is not None:
            overlay.line(
                current_tick[0],
                current_tick[1],
                HOVER_COLOR,
                2.4 * style_scale,
            )

    @staticmethod
    def _draw_hinge_limit(
        overlay: Draw2D,
        dial: _RotationDialProjector,
        angle: float,
        limit_color,
        label: str,
        style_scale: float,
        *,
        label_above: bool,
    ) -> None:
        tick = dial.tick(JOINT_RANGE_RADIUS, angle, 9.0 * style_scale)
        if tick is None:
            return
        overlay.line(tick[0], tick[1], limit_color, 3.0 * style_scale)
        label_offset = np.array((4.0, -14.0 if label_above else 2.0)) * style_scale
        overlay.text(tick[1] + label_offset, limit_color, label)

    def _draw_slide_range(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        state: _JointRangeState,
    ) -> None:
        origin = np.asarray(self._frame.position, np.float64)
        axis = np.asarray(self._frame.rotation, np.float64)[:, 2]
        values = np.array((state.lower, state.current, state.upper), np.float64)
        positions = origin + (values - state.current)[:, None] * axis
        projected = project(cam, positions, rect)
        if np.any(projected[:, 2] <= 0.0):
            return
        lower, current, upper = projected[:, :2]
        direction = upper - lower
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return
        tangent = direction / length
        normal = np.array((-tangent[1], tangent[0]))
        offset = normal * JOINT_RANGE_OFFSET_PT * style_scale
        lower, current, upper = lower + offset, current + offset, upper + offset
        overlay.line(lower, upper, JOINT_RANGE_COLOR, JOINT_RANGE_WIDTH_PT * style_scale)

        half_tick = 6.0 * style_scale
        for point, limit_color in (
            (lower, JOINT_LOWER_LIMIT_COLOR),
            (upper, JOINT_UPPER_LIMIT_COLOR),
        ):
            overlay.line(
                point - normal * half_tick,
                point + normal * half_tick,
                limit_color,
                3.0 * style_scale,
            )
        overlay.line(
            current - normal * 4.0 * style_scale,
            current + normal * 4.0 * style_scale,
            HOVER_COLOR,
            2.4 * style_scale,
        )
        overlay.text(
            lower + np.array((4.0, -16.0)) * style_scale,
            JOINT_LOWER_LIMIT_COLOR,
            _joint_limit_label("MIN", state.lower, "slide"),
        )
        overlay.text(
            upper + np.array((4.0, 3.0)) * style_scale,
            JOINT_UPPER_LIMIT_COLOR,
            _joint_limit_label("MAX", state.upper, "slide"),
        )

    def _draw_axis_constraint(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        axis = _axis_of(self._active)
        if axis < 0:
            return
        origin = np.asarray(self._frame.position, np.float64)
        screen = project(cam, (origin, origin + self._start_basis[:, axis]), rect)
        if np.any(screen[:, 2] <= 0.0):
            return
        segment = _clip_line_to_rect(screen[0, :2], screen[1, :2] - screen[0, :2], rect)
        if segment is None:
            return
        color = self._handle_color(axis)
        overlay.line(
            segment[0],
            segment[1],
            (float(color[0]), float(color[1]), float(color[2]), 0.62),
            1.5 * style_scale,
        )

    def _draw_translation_snap_ruler(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        axis_index = _axis_of(self._active)
        if axis_index < 0:
            return
        axis = self._start_basis[:, axis_index]
        projected = project(cam, (self._start_pos, self._start_pos + axis), rect)
        if np.any(projected[:, 2] <= 0.0):
            return
        origin = projected[0, :2]
        direction = projected[1, :2] - origin
        pixels_per_meter = float(np.linalg.norm(direction))
        if pixels_per_meter < 1e-6:
            return
        direction /= pixels_per_meter
        segment = _clip_line_to_rect(origin, direction, rect)
        if segment is None:
            return

        bounds = _projected_line_parameters(
            cam,
            self._start_pos,
            axis,
            segment,
            rect,
        )
        if bounds is None:
            return

        axis_color = self._handle_color(axis_index)
        step = float(self.translation_snap_m)
        current_distance = float(np.dot(self._frame.position - self._start_pos, axis))
        current_step = current_distance / step
        lo = max(
            int(np.ceil(min(bounds) / step)),
            int(np.ceil(current_step - SNAP_TICK_FADE_STEPS)),
        )
        hi = min(
            int(np.floor(max(bounds) / step)),
            int(np.floor(current_step + SNAP_TICK_FADE_STEPS)),
        )
        if lo > hi:
            return

        normal = np.array((-direction[1], direction[0]))
        ticks_visible = pixels_per_meter * step >= 2.0 * style_scale
        ticks: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
        for index in range(lo, hi + 1):
            distance = index * step
            world = self._start_pos + axis * distance
            point = project(cam, (world,), rect)[0]
            if point[2] <= 0.0:
                continue
            alpha = _snap_tick_alpha(index - current_step) if ticks_visible else 0.0
            if alpha <= 0.01:
                continue
            major = abs(distance - round(distance)) < 1e-6
            half_length = (7.0 if major else 3.5) * style_scale
            a = point[:2] - normal * half_length
            b = point[:2] + normal * half_length
            ticks.append((a, b, alpha, False))

        current = project(cam, (self._frame.position,), rect)[0]
        mask_radius = CENTER_SHELL_RADIUS * SIZE_PT * style_scale
        if current[2] > 0.0:
            half_length = 14.0 * style_scale
            ticks.append(
                (
                    current[:2] - normal * half_length,
                    current[:2] + normal * half_length,
                    1.0,
                    True,
                )
            )
        axis_segments = _split_segment_around_point(
            segment[0], segment[1], current[:2], mask_radius
        )

        def color(value, alpha: float):
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]) * alpha)

        for start, end in axis_segments:
            overlay.line(start, end, color((*axis_color, 1.0), 0.92), 1.2 * style_scale)
        for a, b, alpha, is_active in ticks:
            tick_color = color(HOVER_COLOR if is_active else (*axis_color, 1.0), alpha)
            for start, end in _split_segment_around_point(a, b, current[:2], mask_radius):
                overlay.line(start, end, tick_color, (2.2 if is_active else 1.2) * style_scale)

    def _draw_rotation_snap_ticks(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        dial: _RotationDialProjector,
    ) -> None:
        ring_radius = (
            SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        )
        step = float(self.rotation_snap_deg)
        core = GUIDE_CORE_COLOR

        tick_radius = ring_radius

        def dial_points(angles, radius=tick_radius) -> np.ndarray:
            return dial.points(radius, angles)

        def tick_segment(angle: float, length_pt: float):
            return dial.tick(
                tick_radius,
                angle,
                length_pt * self.rotation_tick_scale * style_scale,
            )

        ticks_visible = (
            self._active is GizmoHandle.ROTATE_SCREEN
            or rotation_ring_alpha(cam, self._start_pos, self._axis) >= ROTATION_TICK_MIN_ALPHA
        )
        ticks: list[tuple[np.ndarray, np.ndarray]] = []
        for degrees in np.arange(0.0, 360.0, step):
            angle = np.radians(degrees)
            angle_step = np.radians(step)
            points = dial_points((angle, angle + angle_step))
            if np.any(points[:, 2] <= 0.0):
                continue
            spacing = float(np.linalg.norm(points[1, :2] - points[0, :2]))
            if not ticks_visible or spacing < 2.0 * style_scale:
                continue
            segment = tick_segment(angle, _rotation_tick_length_pt(degrees))
            if segment is not None:
                ticks.append(segment)

        active_tick = None
        if ticks_visible:
            angle = self._rotation_angle
            points = dial_points((angle, angle + np.radians(step)))
            spacing = float(np.linalg.norm(points[1, :2] - points[0, :2]))
            if np.all(points[:, 2] > 0.0) and spacing >= 2.0 * style_scale:
                # Highlight the existing tick without extending beyond the
                # tick field when the projected dial becomes narrow.
                active_tick = tick_segment(
                    angle,
                    _rotation_tick_length_pt(np.degrees(angle)),
                )

        for inner, outer in ticks:
            overlay.line(inner, outer, core, 1.1 * style_scale)
        if active_tick is not None:
            overlay.line(active_tick[0], active_tick[1], HOVER_COLOR, 2.2 * style_scale)

    def _draw_translation_guide(self, overlay: Draw2D, cam, rect, style_scale: float) -> None:
        screen = project(cam, (self._start_pos, self._frame.position), rect)
        if np.any(screen[:, 2] <= 0.0):
            return
        start, end = screen[:, :2]
        delta = end - start
        distance = float(np.linalg.norm(delta))
        edge = CONTRAST_EDGE_COLOR
        core = GUIDE_CORE_COLOR
        radius = 6.0 * style_scale
        core_width = 2.0 * style_scale
        edge_width = core_width + 2.0 * CONTRAST_EDGE_PT * style_scale
        if distance > 2.0 * radius:
            direction = delta / distance
            a = start + direction * radius
            b = end - direction * radius
            overlay.line(a, b, edge, edge_width)
            overlay.line(a, b, core, core_width)
        for point in (start, end):
            overlay.circle(point, radius, edge, edge_width, segments=24)
            overlay.circle(point, radius, core, core_width, segments=24)

    def _publish_translation_guide(self, backend: Any, ui_scale: float) -> None:
        dd = getattr(backend, "debug", None)
        active = self._using and self._active not in ROTATE_HANDLES
        if not active or not backend.caps.debug_draw or dd is None:
            self._clear_translation_guide(backend)
            return
        dd.layer(DRAG_LAYER, Occlusion.ALWAYS).drag_link(
            "gizmo.drag",
            self._start_pos,
            self._frame.position,
            GUIDE_CORE_COLOR,
            CONTRAST_EDGE_COLOR,
            width_px=2.0 * ui_scale,
            radius_px=6.0 * ui_scale,
            edge_px=CONTRAST_EDGE_PT * ui_scale,
        )
        self._guide_gpu = True

    def _clear_translation_guide(self, backend: Any) -> None:
        dd = getattr(backend, "debug", None)
        if self._guide_gpu and backend.caps.debug_draw and dd is not None:
            dd.layer(DRAG_LAYER, Occlusion.ALWAYS).clear()
        self._guide_gpu = False

    def _draw_rotation_guide(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        dial: _RotationDialProjector,
    ) -> None:
        sweep = _rotation_sweep(self._rotation_angle)
        ring_radius = (
            SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        )

        dial_segments = _rotation_dial_segments(cam, self._start_pos, self._axis)
        reference = dial.points(
            ring_radius,
            np.linspace(0.0, 2.0 * np.pi, dial_segments, endpoint=False),
        )
        point_count = max(2, int(np.ceil(dial_segments * abs(sweep) / (2.0 * np.pi))) + 1)
        angles = np.linspace(0.0, sweep, point_count)
        arc = dial.points(ring_radius, angles)
        center = project(cam, (self._start_pos,), rect)[0]
        if center[2] <= 0.0 or np.any(arc[:, 2] <= 0.0):
            return
        center = center[:2]
        arc = arc[:, :2]
        sector = [center, *arc]
        border = (1.0, 0.5, 0.06, 1.0)

        fill_alpha = _rotation_fill_alpha(sweep)
        if fill_alpha > 0.0:
            fill = (1.0, 0.5, 0.06, fill_alpha)
            overlay.triangle_fan_fill(sector, fill)
        if np.all(reference[:, 2] > 0.0):
            overlay.polyline(
                reference[:, :2],
                HOVER_COLOR,
                RING_WIDTH_PT * style_scale,
                closed=True,
            )
        if abs(sweep) > 1e-6:
            width = RING_WIDTH_PT * style_scale
            start_tick = dial.tick(ring_radius, float(angles[0]), 1.0)
            end_tick = dial.tick(ring_radius, float(angles[-1]), 1.0)
            stroke = _rotation_arc_stroke(
                arc,
                None if start_tick is None else start_tick[1] - start_tick[0],
                None if end_tick is None else end_tick[1] - end_tick[0],
                width,
            )
            if len(stroke):
                overlay.fringed_concave_fill(stroke, border)
            else:
                overlay.polyline(arc, border, width)

    def _draw_value_label(
        self,
        overlay: Draw2D,
        cam,
        rect,
        style_scale: float,
        dial: _RotationDialProjector | None,
    ) -> None:
        pad = 6.0 * style_scale
        gap = 14.0 * style_scale
        anchor_world = self._frame.position
        anchor = None
        if self._active in ROTATE_HANDLES:
            ring_radius = (
                SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
            )
            if dial is not None:
                anchor = dial.points(ring_radius, (self._rotation_angle,))[0]
        if anchor is None:
            anchor = project(cam, (anchor_world,), rect)[0]
        if anchor[2] <= 0.0:
            return
        width_f, height_f = overlay.text_size(self._label)
        width, height = width_f + 2.0 * pad, height_f + 2.0 * pad
        x = float(np.clip(anchor[0] + gap, rect[0] + 4.0, rect[0] + rect[2] - width - 4.0))
        y = float(np.clip(anchor[1] + gap, rect[1] + 4.0, rect[1] + rect[3] - height - 4.0))
        overlay.rect_filled(
            (x, y),
            (x + width, y + height),
            (0.08, 0.09, 0.11, 0.92),
            rounding=4.0 * style_scale,
        )
        overlay.text((x + pad, y + pad), (0.96, 0.96, 0.97, 1.0), self._label)

    def _begin(self, session, cam, rect, cursor) -> bool:
        return self._begin_handle(session, cam, rect, cursor, self._hovered)

    def _begin_handle(self, session, cam, rect, cursor, handle: GizmoHandle) -> bool:
        node = session.selected_node
        if node is None or handle is GizmoHandle.NONE:
            return False
        target, _reason = self._joint_target(session, node)
        allowed = target.handles if target is not None else ALL_HANDLE_MASK
        if not allowed & (1 << int(handle)):
            return False
        pose = self._target_pose(session, node, target)
        if pose is None:
            return False
        pos, mat = pose
        self._active_joint = target.joint if target is not None else None
        if self._active_joint is not None:
            count = 4 if self._active_joint.type == "ball" else 1
            qpos = session.frame.qpos
            if qpos is None:
                self._active_joint = None
                return False
            start = self._active_joint.qpos_adr
            self._start_joint_qpos = np.asarray(qpos[start : start + count], np.float64).copy()
            if len(self._start_joint_qpos) != count:
                self._active_joint = None
                return False
        self._active = handle
        np.copyto(self._start_pos, pos)
        np.copyto(self._start_mat, mat)
        np.copyto(self._start_basis, self._target_basis(mat, target))
        np.copyto(self._current_mat, mat)
        self._start_cursor[:] = cursor
        self._rotation_raw_angle = 0.0
        self._rotation_angle = 0.0
        self._snapping = False
        self._edit_started = False
        self._label = self._format_value(self._start_pos)

        axis = _axis_of(self._active)
        if axis >= 0:
            self._axis[:] = self._start_basis[:, axis]
        elif self._active is GizmoHandle.ROTATE_SCREEN:
            self._axis[:] = -cam.forward()

        if self._active in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
            scale = world_scale(cam, pos, rect[3], SIZE_PT * self._style_scale)
            screen = project(cam, [pos, pos + self._axis * scale * AXIS_END], rect)[:, :2]
            delta = screen[1] - screen[0]
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                self._end()
                return False
            self._axis_screen[:] = delta / length
            self._world_per_pt = scale / length
            self._start_edit(session)
            return True

        if self._active in (GizmoHandle.SCREEN, GizmoHandle.ROTATE_SCREEN):
            self._plane_normal[:] = cam.forward()
        else:
            self._plane_normal[:] = self._axis

        hit = _cursor_plane(cam, rect, cursor, pos, self._plane_normal)
        if hit is None:
            self._end()
            return False
        if self._active in ROTATE_HANDLES:
            v = hit - pos
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                self._end()
                return False
            self._rotation_start_vec[:] = self._last_rot_vec[:] = v / n
        else:
            self._plane_start[:] = hit
        self._start_edit(session)
        return True

    def _drag(self, session, cam, rect, cursor, *, snap: bool) -> bool:
        handle = self._active
        self._snapping = bool(snap)
        pos = self._start_pos.copy()
        mat = self._start_mat
        if handle in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
            travel = float(np.dot(np.asarray(cursor) - self._start_cursor, self._axis_screen))
            pos += self._axis * (travel * self._world_per_pt)
        elif handle in (GizmoHandle.SCREEN, GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY):
            hit = _cursor_plane(cam, rect, cursor, self._start_pos, self._plane_normal)
            if hit is None:
                return False
            pos += hit - self._plane_start
        else:
            hit = _cursor_plane(cam, rect, cursor, self._start_pos, self._plane_normal)
            if hit is None:
                return False
            v = hit - self._start_pos
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                return False
            v /= n
            angle = float(
                np.arctan2(
                    np.dot(self._axis, np.cross(self._last_rot_vec, v)),
                    np.dot(self._last_rot_vec, v),
                )
            )
            if abs(angle) >= 1e-9:
                self._last_rot_vec[:] = v
                self._rotation_raw_angle += angle
            self._rotation_angle = (
                _snap_value(self._rotation_raw_angle, np.radians(self.rotation_snap_deg))
                if snap
                else self._rotation_raw_angle
            )
            delta = math3d.rotvec_to_mat3(self._axis * self._rotation_angle)
            self._current_mat[:] = delta @ self._start_mat
            mat = self._current_mat

        if snap and handle not in ROTATE_HANDLES:
            delta = self._start_basis.T @ (pos - self._start_pos)
            delta = _snap_translation(delta, handle, self.translation_snap_m)
            pos = self._start_pos + self._start_basis @ delta

        unchanged = (
            abs(self._rotation_angle) < 1e-12
            if handle in ROTATE_HANDLES
            else float(np.linalg.norm(pos - self._start_pos)) < 1e-12
        )
        if unchanged and not self._edit_started:
            self._label = self._format_value(pos)
            return True

        node = session.selected_node
        if node is None:
            self._end()
            return False
        result, pos = self._submit_transform(session, node, pos, mat)
        if not result.ok:
            self._verdict = Verdict(False, result.message)
            self._end()
            return False
        self._edit_started = True
        self._label = self._format_value(pos)
        return True

    def _submit_transform(self, session, node, pos, mat) -> tuple[CommandResult, np.ndarray]:
        """Route drag and precise edits through the same target command path."""

        pos = np.asarray(pos, np.float64)
        if self._active_joint is not None:
            joint = self._active_joint
            if joint.type == "ball":
                qstart = math3d.quat_to_mat3(self._start_joint_qpos)
                relative = qstart @ self._start_mat.T @ mat
                command = SetQposBatch(
                    indices=np.arange(joint.qpos_adr, joint.qpos_adr + 4, dtype=np.intp),
                    values=np.asarray(math3d.mat3_to_quat(relative), np.float64),
                )
            else:
                delta = (
                    self._rotation_angle
                    if joint.type == "hinge"
                    else float(np.dot(pos - self._start_pos, self._axis))
                )
                value = float(self._start_joint_qpos[0]) + delta
                if joint.limited and joint.range[1] > joint.range[0]:
                    value = float(np.clip(value, joint.range[0], joint.range[1]))
                applied = value - float(self._start_joint_qpos[0])
                if joint.type == "hinge":
                    self._rotation_angle = applied
                else:
                    pos = self._start_pos + self._axis * applied
                command = SetQpos(joint.qpos_adr, value)
        elif node.type is NodeType.LIGHT:
            command = _set_light_from_world(session, node, pos, mat)
        elif node.type is NodeType.CAMERA:
            command = _set_camera_from_world(session, node, pos, mat)
        elif node.posable:
            command = SetPose(
                node_id=node.node_id,
                position=np.asarray(pos, np.float32),
                rotation=np.asarray(mat, np.float32),
            )
        else:
            reason = gizmo_refusal_reason(session.paused, False) or "Entity is not posable"
            return CommandResult.bad(reason), pos
        if command is None:
            return CommandResult.bad("Entity transform is unavailable"), pos
        return session.submit(command), pos

    def _format_value(self, position) -> str:
        axis = _axis_of(self._active)
        name = (
            "Screen"
            if self._active is GizmoHandle.ROTATE_SCREEN
            else ("XYZ"[axis] if axis >= 0 else "")
        )
        if self._active_joint is not None and self._active_joint.type in ("hinge", "slide"):
            name = self._active_joint.name or self._active_joint.type
        if self._active in ROTATE_HANDLES:
            degrees = round(float(np.degrees(self._rotation_angle)), 1)
            turns = int(abs(degrees) // 360.0)
            suffix = f" · {turns}×360°" if turns else ""
            snap = f" · SNAP {_format_step(self.rotation_snap_deg)}°" if self._snapping else ""
            return f"{name} {degrees:+.1f}°{suffix}{snap}"
        delta = np.asarray(position, np.float64) - self._start_pos
        local = self._start_basis.T @ delta
        if self._active in AXIS_HANDLES:
            value = f"{name} {local[axis]:+.3f} m"
            return self._with_translation_snap(value)
        plane_axes = {
            GizmoHandle.YZ: (1, 2),
            GizmoHandle.ZX: (2, 0),
            GizmoHandle.XY: (0, 1),
        }.get(self._active)
        if plane_axes is not None:
            a, b = plane_axes
            value = f"{'XYZ'[a]} {local[a]:+.3f}  {'XYZ'[b]} {local[b]:+.3f} m"
            return self._with_translation_snap(value)
        value = f"X {local[0]:+.3f}  Y {local[1]:+.3f}  Z {local[2]:+.3f} m"
        return self._with_translation_snap(value)

    def _with_translation_snap(self, value: str) -> str:
        if not self._snapping:
            return value
        return f"{value} · SNAP {_format_step(self.translation_snap_m)} m"

    def _joint_target(
        self, session: Session, node: SceneNode | None
    ) -> tuple[_JointTarget | None, str]:
        if self._joint_structure_generation < 0:
            self._joint_structure_generation = session.structure_generation
        elif self._joint_structure_generation != session.structure_generation:
            self._joint_structure_generation = session.structure_generation
            self._joint_selection.clear()
        if node is None or node.posable or node.type not in (NodeType.LINK, NodeType.ROBOT):
            return None, ""
        joints = session.joints_for_body(node.body_index)
        if not joints:
            return None, "this link has no editable direct joint"
        selected = self._joint_selection.get(int(node.body_index), -1)
        joint = next((item for item in joints if item.joint_id == selected), None)
        if joint is None:
            if len(joints) != 1:
                return None, ("choose one direct joint in the viewport picker or the Joints panel")
            joint = joints[0]
        if joint.type == "hinge":
            return _JointTarget(joint, GizmoMode.ROTATE, handle_mask(GizmoHandle.ROTATE_Z)), ""
        if joint.type == "slide":
            return _JointTarget(joint, GizmoMode.TRANSLATE, handle_mask(GizmoHandle.Z)), ""
        if joint.type == "ball":
            return _JointTarget(joint, GizmoMode.ROTATE, handle_mask(*ROTATE_HANDLES)), ""
        return None, f"{joint.type} joint uses the free-body transform gizmo"

    @staticmethod
    def _joint_range_state(
        session: Session, target: _JointTarget | None
    ) -> _JointRangeState | None:
        if target is None or target.joint.type not in ("hinge", "slide"):
            return None
        joint = target.joint
        lower, upper = (float(value) for value in joint.range)
        qpos = session.frame.qpos
        if (
            not joint.limited
            or upper <= lower
            or not np.isfinite((lower, upper)).all()
            or qpos is None
            or not 0 <= joint.qpos_adr < len(qpos)
        ):
            return None
        current = float(qpos[joint.qpos_adr])
        if not np.isfinite(current):
            return None
        return _JointRangeState(joint.type, current, lower, upper)

    def _target_pose(
        self, session: Session, node: SceneNode, target: _JointTarget | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if target is None:
            return _node_pose(session, node)
        frame = session.frame
        diagnostics = frame.diagnostics
        joint_id = target.joint.joint_id
        if (
            frame.qpos is None
            or diagnostics is None
            or not 0 <= joint_id < len(diagnostics.joint_xpos)
            or not 0 <= joint_id < len(diagnostics.joint_xaxis)
        ):
            return None
        position = np.asarray(diagnostics.joint_xpos[joint_id], np.float64).reshape(3)
        if target.joint.type == "slide":
            # MuJoCo's xanchor excludes this slide coordinate; the driven body pose does not.
            position, _ = _node_pose(session, node)
        if target.joint.type == "ball":
            _body_position, body_rotation = _node_pose(session, node)
            return position, body_rotation
        axis = np.asarray(diagnostics.joint_xaxis[joint_id], np.float64).reshape(3)
        return position, _basis_from_z(axis)

    def _target_basis(self, rotation, target: _JointTarget | None) -> np.ndarray:
        if target is not None and target.joint.type in ("hinge", "slide"):
            return np.asarray(rotation, np.float64).reshape(3, 3)
        return self._basis(rotation)

    def _basis(self, rotation) -> np.ndarray:
        if self._space is GizmoSpace.BODY:
            return np.asarray(rotation, np.float64).reshape(3, 3)
        return _WORLD_BASIS

    def evaluate(self, session: Session, node: SceneNode | None) -> Verdict:
        """Return the actual viewport-gizmo availability for one scene node."""
        target, reason = self._joint_target(session, node)
        if node is not None and not node.posable and node.type in (NodeType.LINK, NodeType.ROBOT):
            if not session.paused:
                return Verdict(False, gizmo_refusal_reason(False, False) or "")
            if not session.adapter.caps.write_qpos:
                return Verdict(False, f"{session.adapter.caps.name} cannot write joint positions")
            if target is None:
                return Verdict(False, reason or "joint gizmo is unavailable")
            if self._target_pose(session, node, target) is None:
                return Verdict(False, "joint frame data is unavailable")
            result = Verdict(True)
        else:
            result = verdict(session.paused, node)
        if not result.ok or node is None:
            return result
        if (
            node.posable
            and node.type not in (NodeType.LIGHT, NodeType.CAMERA)
            and not session.adapter.caps.write_pose
        ):
            return Verdict(False, f"{session.adapter.caps.name} cannot edit this transform")
        if session.entity_gizmo_locked(node):
            return Verdict(False, "gizmo is locked while simulation is running")
        if node.type is NodeType.LIGHT:
            light = _source_light(session, node)
            if light is None:
                return Verdict(False, "light transform is unavailable")
            if light.type is LightType.IMAGE:
                return Verdict(False, "image light has no spatial transform")
        return result

    def _start_edit(self, session: Session) -> None:
        if self._active_joint is not None or not session.adapter.caps.edit_history:
            return
        result = session.submit(BeginEditTransaction(f"{self._mode.value.title()} transform"))
        if result.ok:
            self._edit_session = session

    def _end(self) -> None:
        if self._edit_session is not None:
            self._edit_session.submit(EndEditTransaction())
            self._edit_session = None
        self._using = False
        self._keyboard = False
        self._snapping = False
        self._active = GizmoHandle.NONE
        self._active_joint = None
        self._start_joint_qpos = np.zeros(0, np.float64)
        self._label = ""
        self._edit_started = False


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


def _joint_limit_label(prefix: str, value: float, joint_type: str) -> str:
    if joint_type == "hinge":
        return f"{prefix} {np.degrees(value):+.1f}°"
    return f"{prefix} {value:+.3f} m"


def _rotation_sweep(angle: float) -> float:
    shown = np.radians(round(float(np.degrees(angle)), 1))
    return float(np.copysign(np.fmod(abs(shown), 2.0 * np.pi), shown))


def _rotation_fill_alpha(sweep: float) -> float:
    return 0.28 if abs(float(sweep)) > 1e-6 else 0.0


def _rotation_tick_length_pt(degrees: float) -> float:
    degrees = float(degrees) % 360.0
    rounded = round(degrees)
    if abs(degrees - rounded) < 1e-6 and rounded % 90 == 0:
        return 8.0
    if abs(degrees / 45.0 - round(degrees / 45.0)) < 1e-6:
        return 7.0
    if abs(degrees / 15.0 - round(degrees / 15.0)) < 1e-6:
        return 5.5
    return 4.0


def _rotation_arc_stroke(points, start_radial, end_radial, width: float) -> np.ndarray:
    """Build a constant-width arc whose flat caps follow projected dial radii."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    width = float(width)
    if len(points) < 2 or width <= 0.0:
        return np.empty((0, 2), np.float64)

    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    for index in np.flatnonzero(lengths < 1e-6):
        candidates = []
        if index > 0:
            candidates.append(points[index] - points[index - 1])
        if index + 1 < len(points):
            candidates.append(points[index + 1] - points[index])
        if candidates:
            tangent = max(candidates, key=np.linalg.norm)
            tangents[index] = tangent
            lengths[index] = np.linalg.norm(tangent)
    if np.any(lengths < 1e-6):
        return np.empty((0, 2), np.float64)

    tangents /= lengths[:, None]
    offsets = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    for index, radial in ((0, start_radial), (-1, end_radial)):
        if radial is None:
            continue
        radial = np.asarray(radial, np.float64).reshape(2)
        length = float(np.linalg.norm(radial))
        if length < 1e-6:
            continue
        radial /= length
        if np.dot(radial, offsets[index]) < 0.0:
            radial *= -1.0
        offsets[index] = radial

    half_width = 0.5 * width
    side_a = points - offsets * half_width
    side_b = points + offsets * half_width
    return np.vstack((side_a, side_b[::-1]))


def _clip_line_to_rect(origin, direction, rect) -> tuple[np.ndarray, np.ndarray] | None:
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    if float(np.linalg.norm(direction)) < 1e-6:
        return None
    x, y, w, h = rect
    limits = ((x, x + w), (y, y + h))
    lo, hi = -np.inf, np.inf
    for axis in range(2):
        if abs(direction[axis]) < 1e-9:
            if not limits[axis][0] <= origin[axis] <= limits[axis][1]:
                return None
            continue
        t0 = (limits[axis][0] - origin[axis]) / direction[axis]
        t1 = (limits[axis][1] - origin[axis]) / direction[axis]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if lo > hi:
        return None
    return origin + lo * direction, origin + hi * direction


def _projected_line_parameters(cam, origin, axis, segment, rect) -> tuple[float, float] | None:
    mvp = np.asarray(cam.proj_matrix(), np.float64) @ np.asarray(cam.view_matrix(), np.float64)
    clip_origin = mvp @ np.append(np.asarray(origin, np.float64), 1.0)
    clip_axis = mvp @ np.append(np.asarray(axis, np.float64), 0.0)
    x, y, width, height = rect
    values = []
    for point in segment:
        ndc = np.array(
            (
                2.0 * (float(point[0]) - x) / width - 1.0,
                1.0 - 2.0 * (float(point[1]) - y) / height,
            )
        )
        denominator = clip_axis[:2] - ndc * clip_axis[3]
        component = int(np.argmax(np.abs(denominator)))
        if abs(denominator[component]) < 1e-10:
            return None
        numerator = ndc[component] * clip_origin[3] - clip_origin[component]
        values.append(float(numerator / denominator[component]))
    return values[0], values[1]


def _snap_tick_alpha(offset_steps: float) -> float:
    distance = abs(float(offset_steps))
    if distance <= SNAP_TICK_FULL_STEPS:
        return 1.0
    if distance >= SNAP_TICK_FADE_STEPS:
        return 0.0
    t = (distance - SNAP_TICK_FULL_STEPS) / (SNAP_TICK_FADE_STEPS - SNAP_TICK_FULL_STEPS)
    return float(1.0 - t * t * (3.0 - 2.0 * t))


def _split_segment_around_point(
    start, end, center, radius: float
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    center = np.asarray(center, np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-9 or radius <= 0.0:
        return ((start, end),)
    direction = delta / length
    along = float(np.dot(center - start, direction))
    perpendicular = float(np.linalg.norm(center - (start + along * direction)))
    if perpendicular >= radius:
        return ((start, end),)
    half_gap = float(np.sqrt(radius * radius - perpendicular * perpendicular))
    gap_start = max(0.0, along - half_gap)
    gap_end = min(length, along + half_gap)
    if gap_start >= gap_end:
        return ((start, end),)
    segments = []
    if gap_start > 1e-6:
        segments.append((start, start + direction * gap_start))
    if gap_end < length - 1e-6:
        segments.append((start + direction * gap_end, end))
    return tuple(segments)


def _project_rotation_dial(
    cam: CameraView,
    rect: tuple[float, float, float, float],
    center,
    axis,
    start_direction,
    size_px: float,
    radius: float,
    angles,
) -> np.ndarray:
    """Project one dial through the same camera mapping as the idle gizmo."""
    return _RotationDialProjector(
        cam,
        rect,
        center,
        axis,
        start_direction,
        size_px,
    ).points(radius, angles)


def _project_rotation_tick(
    cam: CameraView,
    rect: tuple[float, float, float, float],
    center,
    axis,
    start_direction,
    size_px: float,
    radius: float,
    angle: float,
    length_px: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project one radial tick and keep its displayed length in pixels."""
    return _RotationDialProjector(
        cam,
        rect,
        center,
        axis,
        start_direction,
        size_px,
    ).tick(radius, angle, length_px)


def _rotation_dial_segments(
    cam: CameraView,
    center,
    axis,
) -> int:
    view = (
        np.asarray(cam.forward(), np.float64)
        if cam.orthographic
        else np.asarray(center, np.float64) - np.asarray(cam.eye, np.float64)
    )
    view /= np.linalg.norm(view)
    facing = abs(float(np.dot(np.asarray(axis, np.float64), view)))
    multiplier = min(4.0, 1.0 / max(np.sqrt(facing), 0.25))
    return int(np.ceil(RING_SEGMENTS * multiplier))


def _cursor_plane(cam, rect, cursor, point, normal) -> np.ndarray | None:
    ndc = ndc_from_viewport(cursor[0], cursor[1], rect)
    origin, direction = unproject(cam, *ndc)
    den = float(np.dot(direction, normal))
    if abs(den) < 1e-8:
        return None
    t = float(np.dot(np.asarray(point) - origin, normal) / den)
    return np.asarray(origin, np.float64) + np.asarray(direction, np.float64) * t


def _basis_from_z(axis) -> np.ndarray:
    z = np.asarray(axis, np.float64).reshape(3)
    length = float(np.linalg.norm(z))
    if length < 1e-9:
        return np.eye(3, dtype=np.float64)
    z /= length
    reference = np.array((0.0, 0.0, 1.0), np.float64)
    if abs(float(np.dot(reference, z))) > 0.9:
        reference[:] = (0.0, 1.0, 0.0)
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def _node_pose(session: Session, node: SceneNode) -> tuple[np.ndarray, np.ndarray]:
    frame = session.frame
    if node.type is NodeType.MODEL:
        info = next((item for item in session.scene_models if item.model_id == node.model_id), None)
        if info is not None:
            return (
                np.asarray(info.position, np.float64).reshape(3),
                np.asarray(info.rotation, np.float64).reshape(3, 3),
            )
    if node.type is NodeType.LIGHT:
        lights = frame.lights or (session.source.lights if session.source is not None else None)
        if lights is not None and 0 <= node.light_index < len(lights.lights):
            light = lights.lights[node.light_index]
            return (
                np.asarray(light.position, np.float64).reshape(3),
                np.asarray(direction_basis(light.direction), np.float64),
            )
    if node.type is NodeType.CAMERA:
        view = _camera_for_node(session, node)
        if view is not None:
            return (
                np.asarray(view.eye, np.float64).reshape(3),
                np.asarray(camera_rotation(view), np.float64),
            )
    if node.type is NodeType.SITE:
        i = int(node.site_index)
        pos = np.zeros(3, np.float64)
        mat = np.eye(3, dtype=np.float64)
        if frame.site_xpos is not None and 0 <= i < len(frame.site_xpos):
            pos = np.asarray(frame.site_xpos[i], np.float64).reshape(3)
        if frame.site_xmat is not None and 0 <= i < len(frame.site_xmat):
            mat = np.asarray(frame.site_xmat[i], np.float64).reshape(3, 3)
        return pos, mat
    if node.type is NodeType.GEOM:
        i = int(node.geom_index)
        pos = np.zeros(3, np.float64)
        mat = np.eye(3, dtype=np.float64)
        if frame.geom_xpos is not None and 0 <= i < len(frame.geom_xpos):
            pos = np.asarray(frame.geom_xpos[i], np.float64).reshape(3)
        if frame.geom_xmat is not None and 0 <= i < len(frame.geom_xmat):
            mat = np.asarray(frame.geom_xmat[i], np.float64).reshape(3, 3)
        return pos, mat
    i = int(node.body_index)
    pos = np.zeros(3, np.float64)
    mat = np.eye(3, dtype=np.float64)
    if frame.body_xpos is not None and 0 <= i < len(frame.body_xpos):
        pos = np.asarray(frame.body_xpos[i], np.float64).reshape(3)
    if frame.body_xmat is not None and 0 <= i < len(frame.body_xmat):
        mat = np.asarray(frame.body_xmat[i], np.float64).reshape(3, 3)
    return pos, mat


def _source_light(session: Session, node: SceneNode):
    source = session.source
    if source is None or not 0 <= node.light_index < len(source.lights.lights):
        return None
    return source.lights.lights[node.light_index]


def _camera_for_node(session: Session, node: SceneNode) -> CameraView | None:
    if not 0 <= node.camera_index < len(session.cameras):
        return None
    camera_id = session.cameras[node.camera_index].camera_id
    view = session.camera_view(camera_id)
    if view is not None:
        return view
    frame = session.frame
    if frame.cameras is not None and node.camera_index < len(frame.cameras):
        return frame.cameras[node.camera_index]
    return None


def _set_light_from_world(session: Session, node: SceneNode, position, rotation):
    light = _source_light(session, node)
    if light is None:
        return None
    position = np.asarray(position, np.float64).reshape(3)
    direction = -np.asarray(rotation, np.float64).reshape(3, 3)[:, 2]
    frame = session.frame
    body = int(node.body_index)
    if (
        frame.body_xpos is not None
        and frame.body_xmat is not None
        and 0 <= body < len(frame.body_xpos)
        and body < len(frame.body_xmat)
    ):
        body_position = np.asarray(frame.body_xpos[body], np.float64).reshape(3)
        body_rotation = np.asarray(frame.body_xmat[body], np.float64).reshape(3, 3)
        position = body_rotation.T @ (position - body_position)
        direction = body_rotation.T @ direction
    return SetLight(
        node.light_index,
        replace(
            light,
            position=np.asarray(position, np.float32),
            direction=math3d.normalize(direction),
        ),
    )


def _set_camera_from_world(session: Session, node: SceneNode, position, rotation):
    view = _camera_for_node(session, node)
    if view is None or not 0 <= node.camera_index < len(session.cameras):
        return None
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    eye = np.asarray(position, np.float32).reshape(3)
    distance = max(view.distance(), 1e-4)
    forward = -rotation[:, 2]
    target = eye + np.asarray(forward * distance, np.float32)
    up = math3d.normalize(rotation[:, 1])
    camera_id = session.cameras[node.camera_index].camera_id
    return SetSceneCamera(camera_id, replace(view, eye=eye, target=target, up=up))
