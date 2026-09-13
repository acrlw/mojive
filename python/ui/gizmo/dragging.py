"""Gizmo: dragging."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from mojive import math3d
from mojive.adapters.base import NodeType
from mojive.commands import (
    BeginEditTransaction,
    ClearSceneModelTransformPreview,
    CommandResult,
    EndEditTransaction,
    SetGeometrySize,
    SetPose,
    SetQpos,
    SetQposBatch,
    SetSceneModelTransform,
)
from mojive.interaction.gizmo import (
    ALL_HANDLE_MASK,
    AXIS_END,
    AXIS_HANDLES,
    PLANE_HANDLES,
    RING_RADIUS,
    RING_SEGMENTS,
    ROTATE_HANDLES,
    SCREEN_RING_RADIUS,
    SIZE_PT,
    GizmoHandle,
    GizmoMode,
    project,
    rotation_ring_alpha,
    world_scale,
)
from mojive.scene.geometry import geometry_size_from_dimensions
from mojive.types import CameraView
from mojive.ui.panels.inspector import gizmo_refusal_reason

if TYPE_CHECKING:
    from mojive.session import Session


from .joint_labels import _joint_range_handle
from .projection import (
    _cursor_plane,
    _RotationDialProjector,
    _ScreenRotationDialProjector,
    _set_camera_from_world,
    _set_light_from_world,
)
from .state import (
    JOINT_PRECISION_REVEAL_GRACE_SECONDS,
    ROTATION_EDGE_LINEAR_ALPHA,
    ROTATION_LINEAR_ESCAPE_PT,
    ROTATION_LINEAR_LOCK_PT,
    SLIDE_CARDINAL_ESCAPE_PT,
    TRACKBALL_RAD_PER_PT,
    Verdict,
    _axis_of,
    _format_step,
    _snap_translation,
    _snap_value,
)


class _Dragging:
    """Private dragging methods of ObjectGizmo; state belongs to its owner."""

    def _begin(self, session, cam, rect, cursor) -> bool:
        return self._begin_handle(session, cam, rect, cursor, self._hovered)

    def _begin_joint_precision(self, session, cam, rect, cursor) -> bool:
        """Start a direct scalar edit on the expanded viewport rail."""

        projection = self._joint_precision
        node = session.selected_node
        if projection is None or node is None:
            return False
        target, _reason = self._joint_target(session, node)
        if target is None or target.joint.type not in ("hinge", "slide"):
            return False
        joint = target.joint
        if int(joint.joint_id) != projection.joint_id or int(joint.qpos_adr) != projection.qpos_adr:
            return False
        qpos = session.frame.qpos
        address = int(joint.qpos_adr)
        if qpos is None or not 0 <= address < len(qpos):
            return False
        pose = self._target_pose(session, node, target)
        if pose is None:
            return False
        pos, mat = pose
        self._active_joint = joint
        self._active = _joint_range_handle(self._joint_range_state(session, target))
        self._start_joint_qpos = np.asarray((qpos[address],), np.float64)
        self._joint_drag_origin_qpos = self._start_joint_qpos.copy()
        np.copyto(self._start_pos, pos)
        np.copyto(self._drag_origin_pos, pos)
        np.copyto(self._start_mat, mat)
        np.copyto(self._start_basis, self._target_basis(mat, target))
        np.copyto(self._current_mat, mat)
        self._axis[:] = self._start_basis[:, 2]
        self._start_cursor[:] = cursor
        self._rotation_angle = 0.0
        self._snapping = False
        self._edit_started = False
        self._precision_value_label(float(qpos[address]), snap=False)
        self._joint_precision_active = True
        self._joint_precision_visible_until = time.monotonic() + (
            JOINT_PRECISION_REVEAL_GRACE_SECONDS
        )
        return True

    def _begin_handle(self, session, cam, rect, cursor, handle: GizmoHandle) -> bool:
        node = session.selected_node
        if node is None or handle is GizmoHandle.NONE:
            return False
        target, _reason = self._joint_target(session, node)
        dimension_target = None
        if self._mode is GizmoMode.DIMENSIONS:
            dimension_target, _reason = self._dimension_target(session, node)
            if dimension_target is None:
                return False
        allowed = (
            self._dimension_handle_mask(dimension_target.dimensions)
            if dimension_target is not None
            else target.handles
            if target is not None
            else ALL_HANDLE_MASK
        )
        if not allowed & (1 << int(handle)):
            return False
        pose = (
            self._dimension_pose(session, node, dimension_target)
            if dimension_target is not None
            else self._target_pose(session, node, target)
        )
        if pose is None:
            return False
        pos, mat = pose
        self._active_joint = target.joint if target is not None else None
        if dimension_target is not None:
            self._dimension_start = dimension_target
            self._dimension_values[: len(dimension_target.dimensions.values)] = (
                dimension_target.dimensions.values
            )
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
            self._joint_drag_origin_qpos = self._start_joint_qpos.copy()
        self._active = handle
        np.copyto(self._start_pos, pos)
        np.copyto(self._drag_origin_pos, pos)
        np.copyto(self._start_mat, mat)
        np.copyto(self._start_basis, self._target_basis(mat, target))
        np.copyto(self._current_mat, mat)
        self._start_cursor[:] = cursor
        self._rotation_screen_ring_valid = False
        self._rotation_linear = False
        self._rotation_linear_axis = -1
        self._rotation_linear_sign = 1.0
        self._rotation_last_cursor[:] = cursor
        self._rotation_linear_origin_cursor[:] = cursor
        self._rotation_linear_origin_angle = 0.0
        self._rotation_raw_angle = 0.0
        self._rotation_angle = 0.0
        self._slide_cardinal_axis = -1
        self._slide_cardinal_sign = 1.0
        self._slide_last_cursor[:] = cursor
        self._slide_cardinal_origin_cursor[:] = cursor
        self._slide_cardinal_origin_travel = 0.0
        self._trackball_angles[:] = 0.0
        self._snapping = False
        self._edit_started = False
        self._label = (
            self._format_dimension_value()
            if dimension_target is not None
            else self._format_value(self._start_pos)
        )

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

        if self._mode is GizmoMode.DIMENSIONS and self._active is GizmoHandle.SCREEN:
            self._axis_screen[:] = (np.sqrt(0.5), -np.sqrt(0.5))
            scale = world_scale(cam, pos, rect[3], SIZE_PT * self._style_scale)
            self._world_per_pt = scale / max(SIZE_PT * self._style_scale, 1e-6)
            self._start_edit(session)
            return True

        if self._active is GizmoHandle.ROTATE_TRACKBALL:
            self._start_edit(session)
            return True

        if self._active in (GizmoHandle.SCREEN, GizmoHandle.ROTATE_SCREEN):
            self._plane_normal[:] = cam.forward()
        else:
            self._plane_normal[:] = self._axis

        hit = _cursor_plane(cam, rect, cursor, pos, self._plane_normal)
        if hit is None:
            if target is None or target.joint.type != "hinge" or self._active not in ROTATE_HANDLES:
                self._end()
                return False
            hit = pos + self._start_basis[:, 0]
            self._rotation_linear = True
        if self._active in ROTATE_HANDLES:
            v = hit - pos
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                if target is None or target.joint.type != "hinge":
                    self._end()
                    return False
                v, n = self._start_basis[:, 0], 1.0
                self._rotation_linear = True
            self._rotation_start_vec[:] = self._last_rot_vec[:] = v / n
            self._prepare_rotation_drag(cam, rect)
        else:
            self._plane_start[:] = hit
        self._start_edit(session)
        return True

    def _prepare_rotation_drag(self, cam: CameraView, rect) -> None:
        """Cache the rendered dial used to distinguish arc and linear drags."""

        self._rotation_screen_tangent[:] = (1.0, 0.0)
        projector_type = (
            _ScreenRotationDialProjector
            if self._active is GizmoHandle.ROTATE_SCREEN
            else _RotationDialProjector
        )
        projector = projector_type(
            cam,
            rect,
            self._start_pos,
            self._axis,
            self._rotation_start_vec,
            SIZE_PT * self._style_scale,
        )
        radius = SCREEN_RING_RADIUS if self._active is GizmoHandle.ROTATE_SCREEN else RING_RADIUS
        angles = np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS, endpoint=False)
        projected = projector.points(radius, angles)
        valid = bool(
            projected.shape == (RING_SEGMENTS, 3)
            and np.isfinite(projected).all()
            and np.all(projected[:, 2] > 0.0)
        )
        self._rotation_screen_ring_valid = valid
        if valid:
            np.copyto(self._rotation_screen_ring, projected[:, :2])
            tangent = self._rotation_screen_ring[1] - self._rotation_screen_ring[-1]
            length = float(np.linalg.norm(tangent))
            if length > 1e-6:
                self._rotation_screen_tangent[:] = tangent / length
            else:
                self._rotation_screen_tangent[:] = (1.0, 0.0)
                self._rotation_screen_ring_valid = False
        if self._active is not GizmoHandle.ROTATE_SCREEN and (
            rotation_ring_alpha(cam, self._start_pos, self._axis) < ROTATION_EDGE_LINEAR_ALPHA
        ):
            self._rotation_linear = True

    def _rotation_ring_distance(self, cursor) -> float:
        """Return the exact screen distance to the cached projected ring."""

        if not self._rotation_screen_ring_valid:
            return float("inf")
        px, py = map(float, np.asarray(cursor, np.float64).reshape(2))
        best = float("inf")
        previous = self._rotation_screen_ring[-1]
        for current in self._rotation_screen_ring:
            ax, ay = float(previous[0]), float(previous[1])
            dx, dy = float(current[0]) - ax, float(current[1]) - ay
            denominator = dx * dx + dy * dy
            amount = (
                float(np.clip(((px - ax) * dx + (py - ay) * dy) / denominator, 0.0, 1.0))
                if denominator > 1e-12
                else 0.0
            )
            ex = px - (ax + dx * amount)
            ey = py - (ay + dy * amount)
            best = min(best, ex * ex + ey * ey)
            previous = current
        return float(np.sqrt(best))

    def _begin_linear_rotation_drag(self, cursor) -> None:
        """Lock one cardinal direction and preserve the latest pointer motion."""

        current = np.asarray(cursor, np.float64).reshape(2)
        total = current - self._start_cursor
        if float(np.linalg.norm(total)) < ROTATION_LINEAR_LOCK_PT * self._style_scale:
            return
        if abs(float(total[0])) > abs(float(total[1])):
            axis = 0
        elif abs(float(total[1])) > abs(float(total[0])):
            axis = 1
        else:
            axis = int(np.argmax(np.abs(self._rotation_screen_tangent)))
        tangent_component = float(self._rotation_screen_tangent[axis])
        if abs(tangent_component) >= 0.15:
            sign = 1.0 if tangent_component > 0.0 else -1.0
        else:
            # At the exact top/side of a dial the chosen cardinal drag can be
            # radial. Keep those gestures useful with the conventional mapping:
            # right and up increase, left and down decrease.
            sign = 1.0 if axis == 0 else -1.0
        self._rotation_linear = True
        self._rotation_linear_axis = axis
        self._rotation_linear_sign = sign
        self._rotation_linear_origin_cursor[:] = current
        latest_travel = float(current[axis] - self._rotation_last_cursor[axis])
        latest_travel /= max(self._style_scale, 1e-6)
        self._rotation_raw_angle += latest_travel * sign * TRACKBALL_RAD_PER_PT
        self._rotation_linear_origin_angle = float(self._rotation_raw_angle)

    def _update_linear_rotation_drag(self, cursor) -> None:
        current = np.asarray(cursor, np.float64).reshape(2)
        if self._rotation_linear_axis < 0:
            self._begin_linear_rotation_drag(current)
        if self._rotation_linear_axis < 0:
            return
        travel = float(
            current[self._rotation_linear_axis]
            - self._rotation_linear_origin_cursor[self._rotation_linear_axis]
        )
        travel /= max(self._style_scale, 1e-6)
        self._rotation_raw_angle = (
            self._rotation_linear_origin_angle
            + travel * self._rotation_linear_sign * TRACKBALL_RAD_PER_PT
        )

    def _slide_drag_travel(self, cursor) -> float:
        """Keep an axial drag exact, but accept a deliberate cardinal escape."""

        current = np.asarray(cursor, np.float64).reshape(2)
        delta = current - self._start_cursor
        projected = float(np.dot(delta, self._axis_screen))
        if self._slide_cardinal_axis < 0:
            perpendicular = delta - self._axis_screen * projected
            if float(np.linalg.norm(perpendicular)) <= (
                SLIDE_CARDINAL_ESCAPE_PT * self._style_scale
            ):
                self._slide_last_cursor[:] = current
                return projected
            axis = 0 if abs(float(delta[0])) > abs(float(delta[1])) else 1
            component = float(self._axis_screen[axis])
            if abs(component) >= 0.15:
                sign = 1.0 if component > 0.0 else -1.0
            else:
                # Preserve familiar screen controls when the rendered axis is
                # perpendicular to the chosen drag: right and up increase.
                sign = 1.0 if axis == 0 else -1.0
            previous_delta = self._slide_last_cursor - self._start_cursor
            previous_travel = float(np.dot(previous_delta, self._axis_screen))
            latest_travel = float(current[axis] - self._slide_last_cursor[axis]) * sign
            self._slide_cardinal_axis = axis
            self._slide_cardinal_sign = sign
            self._slide_cardinal_origin_cursor[:] = current
            self._slide_cardinal_origin_travel = previous_travel + latest_travel

        axis = self._slide_cardinal_axis
        travel = float(current[axis] - self._slide_cardinal_origin_cursor[axis])
        travel = self._slide_cardinal_origin_travel + travel * self._slide_cardinal_sign
        self._slide_last_cursor[:] = current
        return travel

    def _drag(self, session, cam, rect, cursor, *, snap: bool) -> bool:
        if self._joint_precision_active:
            return self._drag_joint_precision(session, cursor, snap=snap)
        handle = self._active
        self._snapping = bool(snap)
        if self._mode is GizmoMode.DIMENSIONS:
            return self._drag_dimensions(session, cam, rect, cursor, snap=snap)
        pos = self._start_pos.copy()
        mat = self._start_mat
        if handle in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
            travel = (
                self._slide_drag_travel(cursor)
                if self._active_joint is not None and self._active_joint.type == "slide"
                else float(np.dot(np.asarray(cursor) - self._start_cursor, self._axis_screen))
            )
            pos += self._axis * (travel * self._world_per_pt)
        elif handle in (GizmoHandle.SCREEN, GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY):
            hit = _cursor_plane(cam, rect, cursor, self._start_pos, self._plane_normal)
            if hit is None:
                return False
            pos += hit - self._plane_start
        elif handle is GizmoHandle.ROTATE_TRACKBALL:
            screen_delta = (np.asarray(cursor, np.float64) - self._start_cursor) / max(
                self._style_scale, 1e-6
            )
            angles = np.array((screen_delta[1], screen_delta[0]), np.float64)
            angles *= TRACKBALL_RAD_PER_PT
            if snap:
                step = np.radians(self.rotation_snap_deg)
                angles = np.array([_snap_value(value, step) for value in angles])
            self._trackball_angles[:] = angles
            view_basis = np.asarray(cam.view_matrix(), np.float64)[:3, :3].T
            rotvec = view_basis[:, 0] * angles[0] + view_basis[:, 1] * angles[1]
            self._rotation_angle = float(np.linalg.norm(rotvec))
            if self._rotation_angle > 1e-12:
                self._axis[:] = rotvec / self._rotation_angle
                self._current_mat[:] = math3d.rotvec_to_mat3(rotvec) @ self._start_mat
            else:
                self._axis[:] = 0.0
                self._current_mat[:] = self._start_mat
            mat = self._current_mat
        else:
            if not self._rotation_linear and (
                self._rotation_ring_distance(cursor) > ROTATION_LINEAR_ESCAPE_PT * self._style_scale
            ):
                self._begin_linear_rotation_drag(cursor)
            if self._rotation_linear:
                self._update_linear_rotation_drag(cursor)
                self._rotation_angle = (
                    _snap_value(self._rotation_raw_angle, np.radians(self.rotation_snap_deg))
                    if snap
                    else self._rotation_raw_angle
                )
                delta = math3d.rotvec_to_mat3(self._axis * self._rotation_angle)
                self._current_mat[:] = delta @ self._start_mat
                mat = self._current_mat
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
                self._rotation_last_cursor[:] = cursor
                self._rotation_angle = (
                    _snap_value(self._rotation_raw_angle, np.radians(self.rotation_snap_deg))
                    if snap
                    else self._rotation_raw_angle
                )
                delta = math3d.rotvec_to_mat3(self._axis * self._rotation_angle)
                self._current_mat[:] = delta @ self._start_mat
                mat = self._current_mat

        raw_joint_value = None
        if self._active_joint is not None and self._active_joint.type in ("hinge", "slide"):
            raw_delta = (
                self._rotation_raw_angle
                if self._active_joint.type == "hinge"
                else float(np.dot(pos - self._start_pos, self._axis))
            )
            raw_joint_value = float(self._start_joint_qpos[0]) + raw_delta

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
        requested_joint_value = None
        if self._active_joint is not None and self._active_joint.type in ("hinge", "slide"):
            requested_delta = (
                self._rotation_angle
                if self._active_joint.type == "hinge"
                else float(np.dot(pos - self._start_pos, self._axis))
            )
            requested_joint_value = float(self._start_joint_qpos[0]) + requested_delta
        result, pos = self._submit_transform(session, node, pos, mat)
        if not result.ok:
            self._verdict = Verdict(False, result.message)
            self._end()
            return False
        joint = self._active_joint
        if (
            requested_joint_value is not None
            and joint is not None
            and joint.limited
            and joint.range[1] > joint.range[0]
        ):
            lower, upper = joint.range
            outside = requested_joint_value < lower - 1e-12 or requested_joint_value > upper + 1e-12
            # Snapping can hide sub-step overtravel at a limit. Discard it too,
            # so releasing Snap and reversing direction has no residual dead zone.
            held_outside = (
                requested_joint_value <= lower + 1e-12 and raw_joint_value < lower - 1e-12
            ) or (requested_joint_value >= upper - 1e-12 and raw_joint_value > upper + 1e-12)
            if outside or held_outside:
                self._rebase_clamped_joint_drag(cam, rect, cursor, pos)
        self._edit_started = True
        self._label = self._format_value(pos)
        return True

    def _drag_dimensions(self, session: Session, cam, rect, cursor, *, snap: bool) -> bool:
        """Resize in the body frame, keeping coupled radii and uniform proportions."""

        target = self._dimension_start
        if session.selected_node is None or target is None:
            self._end()
            return False
        dimensions = target.dimensions
        values = np.array(dimensions.values, np.float64)
        if self._active is GizmoHandle.SCREEN:
            travel = float(np.dot(np.asarray(cursor) - self._start_cursor, self._axis_screen))
            reference = max(
                dimensions.values[item.parameter] / item.world_to_value
                for item in dimensions.handles
            )
            factor = 1.0 + travel * self._world_per_pt / max(reference, 1e-6)
            if snap:
                longest = float(values.max())
                factor = _snap_value(longest * factor, self.translation_snap_m) / longest
            positive = values[values > 0.0]
            factor = max(factor, 0.002 / float(positive.min()))
            values *= factor
        elif self._active in PLANE_HANDLES:
            hit = _cursor_plane(cam, rect, cursor, self._start_pos, self._plane_normal)
            if hit is None:
                return False
            local = self._start_basis.T @ (hit - self._plane_start)
            normal = _axis_of(self._active)
            # A radial plane can address the same diameter on both axes. Average
            # their travel so a diagonal drag does not apply that dimension twice.
            changes = np.zeros(len(values), np.float64)
            counts = np.zeros(len(values), np.int32)
            for item in dimensions.handles:
                if item.axis is not None and item.axis != normal:
                    changes[item.parameter] += local[item.axis] * item.world_to_value
                    counts[item.parameter] += 1
            for index in np.flatnonzero(counts):
                values[index] += changes[index] / counts[index]
                if snap:
                    values[index] = _snap_value(values[index], self.translation_snap_m)
            np.maximum(values, 0.002, out=values)
        else:
            mapping = dimensions.handle(_axis_of(self._active))
            if mapping is None:
                self._end()
                return False
            travel = float(np.dot(np.asarray(cursor) - self._start_cursor, self._axis_screen))
            values[mapping.parameter] += travel * self._world_per_pt * mapping.world_to_value
            if snap:
                values[mapping.parameter] = _snap_value(
                    values[mapping.parameter], self.translation_snap_m
                )
            values[mapping.parameter] = max(values[mapping.parameter], 0.002)
        self._snapping = bool(snap)
        self._label = self._format_dimension_value(values)
        previous = self._dimension_values[: len(values)]
        if np.allclose(values, previous, atol=1e-12, rtol=0.0):
            return True
        result = session.submit(
            SetGeometrySize(
                target.node_id,
                geometry_size_from_dimensions(target.shape, target.size, values),
            )
        )
        if not result.ok:
            self._verdict = Verdict(False, result.message)
            self._end()
            return False
        previous[:] = values
        self._edit_started = True
        return True

    def _drag_joint_precision(self, session, cursor, *, snap: bool) -> bool:
        """Map horizontal rail travel over the complete authored scalar range."""

        projection = self._joint_precision
        joint = self._active_joint
        if projection is None or joint is None:
            self._end()
            return False
        width = float(projection.end[0] - projection.start[0])
        if width <= 1e-6:
            self._end()
            return False
        amount = float(np.clip((float(cursor[0]) - projection.start[0]) / width, 0.0, 1.0))
        value = projection.lower + amount * (projection.upper - projection.lower)
        self._snapping = bool(snap)
        if snap:
            step = (
                np.radians(self.rotation_snap_deg)
                if joint.type == "hinge"
                else self.translation_snap_m
            )
            value = float(np.clip(_snap_value(value, step), projection.lower, projection.upper))
        current = session.frame.qpos
        address = int(projection.qpos_adr)
        if current is None or not 0 <= address < len(current):
            self._end()
            return False
        if np.isclose(float(current[address]), value, atol=1e-12, rtol=0.0):
            self._precision_value_label(value, snap=snap)
            return True
        result = session.submit(SetQpos(address, value))
        if not result.ok:
            self._verdict = Verdict(False, result.message)
            self._end()
            return False
        self._edit_started = True
        self._precision_value_label(value, snap=snap)
        return True

    def _precision_value_label(self, value: float, *, snap: bool) -> None:
        projection = self._joint_precision
        joint = self._active_joint
        if projection is None or joint is None:
            self._label = ""
            return
        name = joint.name or joint.type
        span = projection.upper - projection.lower
        if joint.type == "hinge":
            shown = float(np.degrees(value))
            shown_span = float(np.degrees(span))
            decimals = 3 if shown_span < 1.0 else 2 if shown_span < 10.0 else 1
            label = f"{name} {shown:+.{decimals}f}°"
            snap_label = f" · SNAP {_format_step(self.rotation_snap_deg)}°"
        else:
            decimals = 5 if span < 0.01 else 4 if span < 0.1 else 3
            label = f"{name} {value:+.{decimals}f} m"
            snap_label = f" · SNAP {_format_step(self.translation_snap_m)} m"
        self._label = label + (snap_label if snap else "")

    def _rebase_clamped_joint_drag(self, cam, rect, cursor, position) -> None:
        """Discard pointer over-travel when a scalar joint reaches a limit."""

        joint = self._active_joint
        if joint is None or not len(self._start_joint_qpos):
            return
        if joint.type == "hinge":
            applied = float(self._rotation_angle)
            self._start_joint_qpos[0] += applied
            applied_rotation = math3d.rotvec_to_mat3(self._axis * applied)
            self._start_mat[:] = applied_rotation @ self._start_mat
            self._current_mat[:] = self._start_mat
            self._rotation_start_vec[:] = (
                applied_rotation @ self._rotation_start_vec
                if self._rotation_linear
                else self._last_rot_vec
            )
            self._last_rot_vec[:] = self._rotation_start_vec
            self._start_cursor[:] = cursor
            self._rotation_last_cursor[:] = cursor
            self._rotation_raw_angle = 0.0
            self._rotation_angle = 0.0
            self._rotation_linear_origin_cursor[:] = cursor
            self._rotation_linear_origin_angle = 0.0
            if not self._rotation_linear:
                self._prepare_rotation_drag(cam, rect)
            return
        if joint.type == "slide":
            applied = float(np.dot(np.asarray(position) - self._start_pos, self._axis))
            self._start_joint_qpos[0] += applied
            self._start_pos[:] = position
            self._start_cursor[:] = cursor
            self._slide_last_cursor[:] = cursor
            self._slide_cardinal_origin_cursor[:] = cursor
            self._slide_cardinal_origin_travel = 0.0

    def _submit_transform(
        self, session, node, pos, mat, *, preview_model: bool = True
    ) -> tuple[CommandResult, np.ndarray]:
        """Route drag and precise edits through the same target command path."""

        pos = np.asarray(pos, np.float64)
        transform_node = self._transform_node(session, node)
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
        elif transform_node.type is NodeType.MODEL and preview_model:
            result = self.preview_model_placement(session, transform_node.model_id, pos, mat)
            return result, pos
        elif transform_node.type is NodeType.MODEL:
            command = SetSceneModelTransform(
                model_id=transform_node.model_id,
                position=np.asarray(pos, np.float32),
                rotation=np.asarray(mat, np.float32),
            )
        elif transform_node.type is NodeType.LIGHT:
            command = _set_light_from_world(session, transform_node, pos, mat)
        elif transform_node.type is NodeType.CAMERA:
            command = _set_camera_from_world(session, transform_node, pos, mat)
        elif transform_node.posable:
            command = SetPose(
                node_id=transform_node.node_id,
                position=np.asarray(pos, np.float32),
                rotation=np.asarray(mat, np.float32),
            )
        else:
            reason = gizmo_refusal_reason(session.paused, False) or "Entity is not posable"
            return CommandResult.bad(reason), pos
        if command is None:
            return CommandResult.bad("Entity transform is unavailable"), pos
        result = session.submit(command)
        return result, pos

    def _format_value(self, position) -> str:
        axis = _axis_of(self._active)
        if self._active is GizmoHandle.ROTATE_TRACKBALL:
            degrees = np.degrees(self._trackball_angles)
            snap = f" · SNAP {_format_step(self.rotation_snap_deg)}°" if self._snapping else ""
            return f"Trackball {degrees[0]:+.1f}° {degrees[1]:+.1f}°{snap}"
        name = (
            "Screen"
            if self._active is GizmoHandle.ROTATE_SCREEN
            else ("XYZ"[axis] if axis >= 0 else "")
        )
        if self._active_joint is not None and self._active_joint.type in ("hinge", "slide"):
            name = self._active_joint.name or self._active_joint.type
        if self._active in ROTATE_HANDLES:
            angle = self._rotation_angle
            if self._active_joint is not None and len(self._start_joint_qpos):
                angle += float(self._start_joint_qpos[0])
            degrees = round(float(np.degrees(angle)), 1)
            turns = int(abs(degrees) // 360.0)
            suffix = f" · {turns}×360°" if turns else ""
            snap = f" · SNAP {_format_step(self.rotation_snap_deg)}°" if self._snapping else ""
            return f"{name} {degrees:+.1f}°{suffix}{snap}"
        delta = np.asarray(position, np.float64) - self._start_pos
        local = self._start_basis.T @ delta
        if self._active in AXIS_HANDLES:
            amount = float(local[axis])
            if self._active_joint is not None and len(self._start_joint_qpos):
                amount += float(self._start_joint_qpos[0])
            value = f"{name} {amount:+.3f} m"
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

    def _format_dimension_value(self, values=None) -> str:
        target = self._dimension_start
        if target is None:
            return ""
        dimensions = target.dimensions
        shown = dimensions.values if values is None else values
        if self._active is GizmoHandle.SCREEN and len(shown) > 1:
            longest = int(np.argmax(dimensions.values))
            factor = shown[longest] / dimensions.values[longest]
            return self._with_translation_snap(f"Scale {factor:.3f}×")
        if self._active in PLANE_HANDLES:
            labels = {
                item.parameter: item.label
                for item in dimensions.handles
                if item.axis != _axis_of(self._active)
            }
            label = "  ".join(
                f"{name.title()} {shown[index]:.3f} m" for index, name in labels.items()
            )
        else:
            axis = None if self._active is GizmoHandle.SCREEN else _axis_of(self._active)
            mapping = dimensions.handle(axis)
            if mapping is None:
                return ""
            label = f"{mapping.label.title()} {shown[mapping.parameter]:.3f} m"
        return self._with_translation_snap(label)

    def _with_translation_snap(self, value: str) -> str:
        if not self._snapping:
            return value
        return f"{value} · SNAP {_format_step(self.translation_snap_m)} m"

    def _start_edit(self, session: Session) -> None:
        if self._active_joint is not None or not session.adapter.caps.edit_history:
            return
        node = session.selected_node
        if node is not None and node.type is NodeType.MODEL:
            return
        label = (
            "Resize geometry"
            if self._mode is GizmoMode.DIMENSIONS
            else f"{self._mode.value.title()} transform"
        )
        result = session.submit(BeginEditTransaction(label))
        if result.ok:
            self._edit_session = session

    def _end(self, *, commit: bool = False) -> None:
        if self._model_preview is not None and self._model_preview_session is not None:
            model_id, position, rotation = self._model_preview
            session = self._model_preview_session
            placement_active = self.model_placement_active(session, model_id)
            if placement_active:
                pass
            elif commit and self._edit_started:
                result = session.submit(SetSceneModelTransform(model_id, position, rotation))
                if not result.ok:
                    session.submit(ClearSceneModelTransformPreview(model_id))
                    self._verdict = Verdict(False, result.message)
            else:
                session.submit(ClearSceneModelTransformPreview(model_id))
            if not placement_active:
                self._model_preview = None
                self._model_preview_session = None
        if self._edit_session is not None:
            self._edit_session.submit(EndEditTransaction())
            self._edit_session = None
        self._using = False
        self._keyboard = False
        self._snapping = False
        self._active = GizmoHandle.NONE
        self._active_joint = None
        self._joint_limit_active = None
        self._joint_precision_active = False
        self._slide_cardinal_axis = -1
        self._start_joint_qpos = np.zeros(0, np.float64)
        self._joint_drag_origin_qpos = np.zeros(0, np.float64)
        self._dimension_start = None
        self._trackball_angles[:] = 0.0
        self._label = ""
        self._edit_started = False
