"""App: navigation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.types import CameraView
from mojive.ui import gestures as gs
from mojive.ui.camera import (
    ISO_PITCH,
    CameraViewTransition,
    camera_basis,
    closest_perpendicular_view_direction,
    elevated_focus_view_direction,
    oblique_axis_view_directions,
)
from mojive.ui.camera_tracking import can_track_node, tracking_position
from mojive.ui.perturb import (
    cursor_grab_point,
)

if TYPE_CHECKING:
    from mojive.adapters.base import SceneNode


from .support import (
    CLICK_SLOP_PT,
    JOINT_FOCUS_MARGIN,
    JOINT_FOCUS_OBLIQUE_DEGREES,
    JOINT_FOCUS_OCCLUSION_NEIGHBORHOOD_DEGREES,
    Keys,
)


class _Navigation:
    """Private navigation methods of ViewerApp; state belongs to its owner."""

    @property
    def tracking_node_id(self) -> int | None:
        """Return the stable hierarchy node currently followed by the editor camera."""
        tracker = getattr(self, "camera_tracker", None)
        return tracker.node_id if tracker is not None else None

    def track_node(self, node_id: int | None) -> None:
        """Follow a hierarchy node; None stops following at the current camera position."""
        if node_id is None:
            tracker = getattr(self, "camera_tracker", None)
            if tracker is not None:
                tracker.stop()
            self._tracking_adapter = None
            return
        node = self.session.node(node_id)
        if not can_track_node(node):
            raise ValueError(f"Node {node_id} has no trackable world position")
        if node_id == self.tracking_node_id:
            return
        if self._model_camera_id >= 0:
            self.camera.adopt(self._camera_view(), exact=True)
        self._leave_model_camera()
        self.camera_tracker.start(node_id, self.camera)
        self._tracking_adapter = self.session.adapter
        self._tracking_node = node

    def _sync_camera_tracking(self, dt: float) -> None:
        node_id = self.tracking_node_id
        if node_id is None:
            return
        node = self.session.node(node_id)
        previous = self._tracking_node
        if (
            not can_track_node(node)
            or self.session.adapter is not self._tracking_adapter
            or node.type != previous.type
            or node.object_id != previous.object_id
            or node.name != previous.name
        ):
            self.track_node(None)
            return
        position = tracking_position(self.session, node)
        if position is not None and self.camera_tracker.advance(self.camera, position, dt):
            self.camera.publish(self.camera_out)

    def _poll_camera(self, state: gs.InputState, keys: Keys, dt: float) -> None:
        fwd, right, up = keys.fly
        if fwd or right or up:
            self._leave_model_camera()
            self.camera.fly(dt, forward=fwd, right=right, up=up)
        if keys.frame_scene:
            self._leave_model_camera()
            self._frame_scene(animate=True)

        if self.interactions.camera.view_cube and self.router.wants_view_cube():
            ball = self.view_cube.hovered

            if self.router.travel >= CLICK_SLOP_PT and state.delta != (0.0, 0.0):
                self._leave_model_camera()
                self.view_cube.drag(self.camera, *state.delta)
            elif ball is not None and self.router.released and self.router.travel < CLICK_SLOP_PT:
                self._leave_model_camera()
                self.view_cube.click(
                    self.camera,
                    ball,
                    self.camera_out,
                    focus=self._selected_view_focus(),
                )
            return

        if not self.router.wants_camera():
            return
        gesture = self.router.camera_gesture(state)

        settled = self.router.travel >= CLICK_SLOP_PT
        if gesture is gs.CameraGesture.ORBIT and settled and self.interactions.camera.orbit:
            self._leave_model_camera()
            self.camera.orbit(*state.delta)
        elif gesture is gs.CameraGesture.PAN and settled and self.interactions.camera.pan:
            self._leave_model_camera()
            self.camera.pan(state.delta[0], state.delta[1], self._viewport_rect[3])
        elif gesture is gs.CameraGesture.DOLLY_DRAG and settled and self.interactions.camera.dolly:
            self._leave_model_camera()
            self.camera.dolly(-state.delta[1] * 0.05)
        elif gesture is gs.CameraGesture.DOLLY and self.interactions.camera.dolly:
            self._leave_model_camera()
            self.camera.dolly(state.wheel)

    def _advance_camera(self, dt: float) -> None:
        if self._model_camera_id >= 0:
            return
        if self._camera_transition is not None:
            view = self._camera_transition.advance(self.camera.view(), dt)
            self.camera_out.set_camera(view)
            if not self._camera_transition.active:
                self._camera_transition = None
            return
        self.camera.advance(dt, self.camera_out)

    def _camera_view(self):
        return self.session.camera

    def viewport_camera_state(self) -> dict:
        """Return the effective viewport camera recorded in the Session."""
        from mojive.scene.state import camera_bookmark

        return camera_bookmark(self.camera, self.session.camera, self._model_camera_id)

    def set_viewport_camera(self, view: CameraView, *, camera_id: int = -1) -> None:
        """Set viewport presentation and synchronize its public Session state."""
        self.track_node(None)
        self._leave_model_camera()
        self.camera.adopt(view, exact=True)
        self.camera.publish(self.camera_out)
        self.select_model_camera(camera_id)
        view = view.with_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.backend.set_camera(view)
        self.session.submit(cmd.SetCamera(view))

    def _select_model_camera_animated(self, camera_id: int) -> None:
        self.select_model_camera(camera_id, animate=True)

    def select_model_camera(self, camera_id: int, *, animate: bool = False) -> None:
        i = int(camera_id)
        if i >= 0 and not any(c.camera_id == i for c in self.session.cameras):
            return
        self.track_node(None)
        start = self._camera_view()
        if i < 0:
            if animate:
                self._model_camera_id = -1
                self._model_camera_view = None
                self._model_camera_projection_target = None
                self._camera_transition = CameraViewTransition(start)
            else:
                self._leave_model_camera(publish=True)
            return
        if i != self._model_camera_id:
            self._model_camera_projection_target = None
        self._model_camera_id = i
        self._camera_transition = CameraViewTransition(start) if animate else None

    def _viewing_selected_camera(self) -> bool:
        node = self.session.selected_node
        if node is None or node.type is not NodeType.CAMERA:
            return False
        index = int(node.camera_index)
        if not 0 <= index < len(self.session.cameras):
            return False
        return int(self.session.cameras[index].camera_id) == self._model_camera_id

    def _sync_model_camera(self) -> None:
        if self._model_camera_id < 0:
            return
        view = self.session.camera_view(self._model_camera_id)
        if view is None:
            self._leave_model_camera(publish=True)
            return
        aspect = max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0)
        view = view.with_aspect(aspect)
        target = bool(view.orthographic)
        if self._model_camera_projection_target is None:
            self._model_camera_projection.snap(target)
        elif target != self._model_camera_projection_target:
            self._model_camera_projection.set(target, animate=True)
        self._model_camera_projection_target = target
        self._model_camera_projection.advance(self._dt)
        if self._model_camera_projection.active:
            view = replace(view, orthographic_blend=self._model_camera_projection.value)
        self._model_camera_view = view
        if self._camera_transition is not None:
            view = self._camera_transition.advance(view, self._dt)
            if not self._camera_transition.active:
                self._camera_transition = None
        self.backend.set_camera(view)
        self.session.submit(cmd.SetCamera(view))

    def _leave_model_camera(self, *, publish: bool = False) -> None:
        if self._camera_transition is not None:
            if not publish:
                self.camera.adopt(self._camera_view(), exact=True)
            self._camera_transition = None
        if self._model_camera_id < 0:
            return
        # Model cameras remain scene entities; the editor orbit camera keeps its own view.
        self._model_camera_id = -1
        self._model_camera_view = None
        self._model_camera_projection_target = None
        if publish:
            self.camera.publish(self.camera_out)

    def _frame_scene(self, *, animate: bool = True) -> None:
        self.track_node(None)
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.frame_scene(
            self.session.bounds(),
            self.camera_out,
            animate=animate,
            clip=self.session.camera_hint(),
            minimum_pitch=ISO_PITCH,
        )

    def _reset_source_camera(self) -> None:
        """Restore the scene source's authored/default free camera."""
        self.track_node(None)
        hint = self.session.camera_hint()
        if hint is None:
            self._frame_scene(animate=False)
            return
        aspect = max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0)
        self.camera.adopt(hint.with_aspect(aspect))
        self.camera.publish(self.camera_out)

    def _poll_perturb(self, state: gs.InputState) -> None:
        st = self.session.perturb
        if (
            not (self.interactions.perturb and self.session.adapter.caps.perturb)
            or not self.router.wants_perturb()
        ):
            if st.active:
                self.perturb.end(self.session)
            return

        node = self.session.selected_node
        if node is None:
            return
        cam = self._camera_view()
        ray = self._cursor_ray(state.cursor) if self.router.mode == "translate" else None
        if not st.active:
            pos, _ = self._node_pose(node)
            grab_point = pos
            if ray is not None:
                grab_point = cursor_grab_point(cam, pos, ray[0], ray[1])
            self.perturb.begin(
                self.session,
                cam,
                node,
                grab_point,
                self.router.mode,
                local_bounds=self.session.node_local_bounds(node.node_id),
            )
        if st.mode == "translate":
            origin, direction = ray if ray is not None else self._cursor_ray(state.cursor)
            self.perturb.drag_translate(self.session, cam, origin, direction)
        else:
            self.perturb.drag_rotate(self.session, cam, state.delta[0], state.delta[1])
        self.perturb.apply(self.session)

    def _selected_view_focus(self) -> tuple[np.ndarray, float] | None:
        node = self.session.selected_node
        if node is None:
            return None
        bounds = self.session.node_world_bounds(node.node_id)
        if bounds is None:
            return None
        center, world_half = bounds
        center = np.asarray(center, np.float64)
        radius = float(np.linalg.norm(world_half))
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius < 1e-6:
            return None
        return center, radius

    def request_joint_focus(self, joint_id: int) -> bool:
        """Request one diagnostics-backed camera focus on a stable joint ID."""

        requested = int(joint_id)
        if not any(joint.joint_id == requested for joint in self.session.joints):
            return False
        self._pending_node_focus_id = None
        self._pending_joint_focus_id = requested
        return True

    def request_node_focus(self, node_id: int) -> bool:
        """Request a camera focus on one stable hierarchy node."""

        node = self.session.node(int(node_id))
        if node is None:
            return False
        if node.type is NodeType.JOINT:
            return self.request_joint_focus(node.joint_index)
        self._pending_joint_focus_id = None
        self._pending_node_focus_id = node.node_id
        return True

    def _request_node_joint_focus(self, node: SceneNode) -> bool:
        if node.type not in (NodeType.JOINT, NodeType.LINK, NodeType.ROBOT, NodeType.GEOM):
            return False
        joint_id = int(node.joint_index) if node.type is NodeType.JOINT else -1
        if joint_id < 0 and node.body_index >= 0:
            candidates = tuple(
                joint
                for joint in self.session.joints_for_body(node.body_index)
                if joint.type in ("hinge", "slide", "ball")
            )
            selected = self.gizmo.selected_joint_id(node.body_index)
            if any(joint.joint_id == selected for joint in candidates):
                joint_id = selected
            elif len(candidates) == 1:
                joint_id = candidates[0].joint_id
        return joint_id >= 0 and self.request_joint_focus(joint_id)

    def _apply_pending_joint_focus(self) -> None:
        joint_id = self._pending_joint_focus_id
        if joint_id is None:
            return
        self._pending_joint_focus_id = None
        self.track_node(None)
        joint = next(
            (candidate for candidate in self.session.joints if candidate.joint_id == joint_id),
            None,
        )
        node = next(
            (
                candidate
                for candidate in self.session.nodes
                if candidate.type is NodeType.JOINT and candidate.joint_index == joint_id
            ),
            None,
        )
        if joint is None or node is None:
            return

        body_position, body_rotation = self._node_pose(node)
        center = np.asarray(body_position, np.float64).reshape(3)
        axis = np.asarray(body_rotation, np.float64).reshape(3, 3) @ np.asarray(
            joint.axis, np.float64
        ).reshape(3)
        diagnostics = self.session.frame.diagnostics
        if diagnostics is not None:
            if 0 <= joint_id < len(diagnostics.joint_xaxis):
                candidate_axis = np.asarray(diagnostics.joint_xaxis[joint_id], np.float64).reshape(
                    3
                )
                if np.isfinite(candidate_axis).all() and np.linalg.norm(candidate_axis) > 1e-9:
                    axis = candidate_axis
            if joint.type != "slide" and 0 <= joint_id < len(diagnostics.joint_xpos):
                candidate_center = np.asarray(diagnostics.joint_xpos[joint_id], np.float64).reshape(
                    3
                )
                if np.isfinite(candidate_center).all():
                    center = candidate_center

        axis_length = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or not np.isfinite(axis_length) or axis_length <= 1e-9:
            axis = np.array((0.0, 0.0, 1.0), np.float64)
        else:
            axis = axis / axis_length

        slide_half_span = 0.0
        frame = self.session.frame
        if joint.type == "slide" and joint.limited:
            lower, upper = map(float, joint.range)
            if (
                np.isfinite((lower, upper)).all()
                and upper > lower
                and frame.qpos is not None
                and 0 <= joint.qpos_adr < len(frame.qpos)
            ):
                current = float(frame.qpos[joint.qpos_adr])
                if np.isfinite(current):
                    midpoint = (lower + upper) * 0.5
                    center = center + axis * (midpoint - current)
                    slide_half_span = (upper - lower) * 0.5

        current_view = self._camera_view()
        eye_offset = np.asarray(current_view.eye, np.float64).reshape(3) - center
        camera_right = camera_basis(current_view)[0]
        elevated_direction = elevated_focus_view_direction(eye_offset, camera_right)
        if joint.type == "hinge":
            candidates = oblique_axis_view_directions(
                axis,
                eye_offset,
                camera_right,
                JOINT_FOCUS_OBLIQUE_DEGREES,
            )
        elif joint.type == "slide":
            nearest = closest_perpendicular_view_direction(
                axis,
                eye_offset,
                camera_right,
            )
            tangent = np.cross(axis, nearest)
            candidates = (nearest, -nearest, tangent, -tangent, elevated_direction)
        else:
            candidates = (elevated_direction,)

        radius = self._joint_focus_radius(node)
        if slide_half_span > 0.0:
            radius += slide_half_span
        eye_direction = self._least_occluded_joint_direction(
            center,
            radius,
            candidates,
            node,
            eye_offset,
            preferred_up=(0.0, 0.0, 1.0),
            minimum_elevation_degrees=ISO_PITCH,
        )
        if self._model_camera_id >= 0:
            self.camera.adopt(current_view)
        self._leave_model_camera()
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.focus_target(
            center,
            radius,
            eye_direction,
            self.camera_out,
            margin=JOINT_FOCUS_MARGIN,
            animate=True,
        )

    def _least_occluded_joint_direction(
        self,
        center: np.ndarray,
        radius: float,
        candidates: tuple[np.ndarray, ...],
        node: SceneNode,
        eye_offset: np.ndarray,
        *,
        preferred_up: tuple[float, float, float] | None = None,
        minimum_elevation_degrees: float = 0.0,
    ) -> np.ndarray:
        """Prefer a requested elevation, then a clear focus ray and a short turn."""

        directions: list[np.ndarray] = []
        for value in candidates:
            direction = np.asarray(value, np.float64).reshape(3)
            length = float(np.linalg.norm(direction))
            if not np.isfinite(direction).all() or not np.isfinite(length) or length <= 1e-9:
                continue
            direction = direction / length
            if not any(float(np.dot(direction, known)) > 1.0 - 1e-7 for known in directions):
                directions.append(direction)
        if not directions:
            return np.asarray(self.camera.direction(), np.float64)

        current_direction = np.asarray(eye_offset, np.float64)
        current_length = float(np.linalg.norm(current_direction))
        if current_length > 1e-9:
            current_direction /= current_length
        else:
            current_direction = directions[0]

        up = None
        if preferred_up is not None:
            candidate_up = np.asarray(preferred_up, np.float64).reshape(3)
            up_length = float(np.linalg.norm(candidate_up))
            if np.isfinite(candidate_up).all() and np.isfinite(up_length) and up_length > 1e-9:
                up = candidate_up / up_length

        if up is not None:
            minimum_elevation = np.deg2rad(float(np.clip(minimum_elevation_degrees, 0.0, 90.0)))
            preferred = [
                direction
                for direction in directions
                if float(np.dot(direction, up)) >= float(np.sin(minimum_elevation)) - 1e-6
            ]
            if preferred:
                directions = preferred
            else:
                above = [
                    direction for direction in directions if float(np.dot(direction, up)) >= -1e-6
                ]
                if above:
                    directions = above

        def turn_from_current(direction: np.ndarray) -> float:
            return 1.0 - float(np.clip(np.dot(direction, current_direction), -1.0, 1.0))

        # Occlusion can choose a slightly different nearby side, but it must
        # not pull focus across the model to a distant canonical quadrant.
        turn_angles = [
            float(np.arccos(np.clip(np.dot(direction, current_direction), -1.0, 1.0)))
            for direction in directions
        ]
        nearest_turn = min(turn_angles)
        turn_limit = nearest_turn + np.deg2rad(JOINT_FOCUS_OCCLUSION_NEIGHBORHOOD_DEGREES)
        directions = [
            direction
            for direction, turn_angle in zip(directions, turn_angles, strict=True)
            if turn_angle <= turn_limit + 1e-9
        ]

        caps = getattr(getattr(self.session, "adapter", None), "caps", None)
        if not bool(getattr(caps, "raycast", False)):
            return min(directions, key=turn_from_current)

        lo, hi = self.session.bounds()
        scene_span = float(np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)))
        current_distance = float(np.linalg.norm(np.asarray(eye_offset, np.float64)))
        probe_distance = max(current_distance, scene_span * 0.75, float(radius) * 8.0, 1.0)

        best = directions[0]
        best_score = (2.0, 0.0, 0.0)
        for direction in directions:
            origin = np.asarray(center, np.float64) + direction * probe_distance
            try:
                hit_id, hit_distance = self.session.query(
                    cmd.Pick(origin=origin, direction=-direction)
                )
            except (AttributeError, TypeError, ValueError):
                return min(directions, key=turn_from_current)
            hit_id = int(hit_id)
            hit_distance = float(hit_distance)
            hit_node = self.session.node_by_object_id(hit_id) if hit_id > 0 else None
            hits_target = bool(
                hit_id == node.body_index
                or (hit_node is not None and hit_node.body_index == node.body_index)
            )
            blocked = bool(
                hit_id > 0
                and not hits_target
                and np.isfinite(hit_distance)
                and hit_distance < probe_distance
            )
            clearance = min(max(hit_distance, 0.0), probe_distance) if blocked else 0.0
            score = (1.0 if blocked else 0.0, -clearance, turn_from_current(direction))
            if score < best_score:
                best = direction
                best_score = score
        return best

    def _apply_pending_node_focus(self) -> None:
        node_id = self._pending_node_focus_id
        if node_id is None:
            return
        self._pending_node_focus_id = None
        self.track_node(None)
        node = self.session.node(node_id)
        if node is None:
            return

        bounds = self.session.node_world_bounds(node.node_id)
        if bounds is None and node.type in (NodeType.WORLD, NodeType.ENVIRONMENT):
            bounds = self.session.bounds()
        if bounds is not None:
            lo_or_center = np.asarray(bounds[0], np.float64).reshape(3)
            hi_or_half = np.asarray(bounds[1], np.float64).reshape(3)
            if node.type in (NodeType.WORLD, NodeType.ENVIRONMENT):
                center = (lo_or_center + hi_or_half) * 0.5
                radius = float(np.linalg.norm(hi_or_half - lo_or_center) * 0.5)
            else:
                center = lo_or_center
                radius = float(np.linalg.norm(hi_or_half))
        else:
            center, _rotation = self._node_pose(node)
            lo, hi = self.session.bounds()
            radius = float(
                np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)) * 0.025
            )
        center = np.asarray(center, np.float64).reshape(3)
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 1e-6:
            return

        current_view = self._camera_view()
        eye_offset = np.asarray(current_view.eye, np.float64).reshape(3) - center
        eye_direction = elevated_focus_view_direction(
            eye_offset,
            camera_basis(current_view)[0],
        )
        if self._model_camera_id >= 0:
            self.camera.adopt(current_view)
        self._leave_model_camera()
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.focus_target(
            center,
            radius,
            eye_direction,
            self.camera_out,
            animate=True,
        )

    def _joint_focus_radius(self, node: SceneNode) -> float:
        for bounds in (
            self.session.node_local_bounds(node.node_id),
            self.session.node_world_bounds(node.node_id),
        ):
            if bounds is None:
                continue
            half = np.sort(np.abs(np.asarray(bounds[1], np.float64).reshape(3)))
            if np.isfinite(half).all() and half[-1] > 1e-6:
                return max(float(half[-1]), 1e-4)
        lo, hi = self.session.bounds()
        diagonal = float(np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)))
        if np.isfinite(diagonal) and diagonal > 1e-6:
            return max(diagonal * 0.025, 1e-4)
        return max(self.camera.distance * 0.04, 1e-4)
