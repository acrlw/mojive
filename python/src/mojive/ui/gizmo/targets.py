"""Gizmo: targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive.adapters.base import NodeType
from mojive.interaction.gizmo import (
    AXIS_HANDLES,
    PLANE_HANDLES,
    ROTATE_HANDLES,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
    handle_mask,
)
from mojive.scene.geometry import GeometryDimensions, geometry_dimensions
from mojive.scene.queries import node_world_pose
from mojive.types import LightType
from mojive.ui.panels.inspector import gizmo_refusal_reason

if TYPE_CHECKING:
    from mojive.adapters.base import SceneNode
    from mojive.session import Session


from .projection import _basis_from_z, _joint_frame_available, _source_light
from .state import _WORLD_BASIS, Verdict, _DimensionTarget, _JointRangeState, _JointTarget, verdict


class _Targets:
    """Private targets methods of ObjectGizmo; state belongs to its owner."""

    def _joint_target(
        self, session: Session, node: SceneNode | None
    ) -> tuple[_JointTarget | None, str]:
        if self._joint_structure_generation < 0:
            self._joint_structure_generation = session.structure_generation
        elif self._joint_structure_generation != session.structure_generation:
            self._joint_structure_generation = session.structure_generation
            self._joint_selection.clear()
        if (
            node is None
            or node.posable
            or node.type
            not in (
                NodeType.LINK,
                NodeType.ROBOT,
                NodeType.JOINT,
            )
        ):
            return None, ""
        joints = session.joints_for_body(node.body_index)
        if not joints:
            return None, "this link has no editable direct joint"
        if node.type is NodeType.JOINT:
            joint = next((item for item in joints if item.joint_id == node.joint_index), None)
            if joint is None:
                return None, "this joint is unavailable"
            self.select_joint(node.body_index, joint.joint_id)
        else:
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
    def _transform_node(session: Session, node: SceneNode) -> SceneNode:
        """Resolve the pose-write target while preserving a selected free-joint node."""

        if node.type is not NodeType.JOINT:
            return node
        joint = next(
            (
                item
                for item in session.joints_for_body(node.body_index)
                if item.joint_id == node.joint_index
            ),
            None,
        )
        if joint is None or joint.type != "free":
            return node
        parent = session.node(node.parent)
        if parent is not None and parent.posable and int(parent.body_index) == int(node.body_index):
            return parent
        return next(
            (
                candidate
                for candidate in session.nodes
                if candidate.posable
                and int(candidate.body_index) == int(node.body_index)
                and candidate.type in (NodeType.LINK, NodeType.ROBOT)
            ),
            node,
        )

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
        return _JointRangeState(
            joint.type,
            current,
            lower,
            upper,
            int(joint.joint_id),
            int(joint.qpos_adr),
        )

    def _dimension_target(
        self, session: Session, node: SceneNode | None
    ) -> tuple[_DimensionTarget | None, str]:
        """Resolve one editable primitive without inventing transform scale."""

        generation = int(session.structure_generation)
        node_id = -1 if node is None else int(node.node_id)
        if (
            session is self._dimension_cache_session
            and generation == self._dimension_cache_generation
            and node_id == self._dimension_cache_node
        ):
            return self._dimension_cache
        if node is not None and node.type is NodeType.LINK and node.model_id < 0:
            children = [session.node(child) for child in node.children]
            if len(children) == 1 and children[0] is not None and children[0].type is NodeType.GEOM:
                node = children[0]
        if node is None or node.type not in (NodeType.GEOM, NodeType.SITE):
            result = (None, "Select an editable geometry or site")
        else:
            result = self._resolve_dimension_target(session, node)
        self._dimension_cache_session = session
        self._dimension_cache_generation = generation
        self._dimension_cache_node = node_id
        self._dimension_cache = result
        return result

    @staticmethod
    def _resolve_dimension_target(
        session: Session, node: SceneNode
    ) -> tuple[_DimensionTarget | None, str]:
        source = session.source
        if source is None:
            return None, "Geometry dimensions are unavailable"
        instances = np.flatnonzero(np.asarray(source.geom_node) == int(node.node_id))
        if not len(instances):
            return None, "Geometry dimensions are unavailable"
        first = int(instances[0])
        if first < len(source.geom_infinite_plane) and bool(source.geom_infinite_plane[first]):
            return None, "Infinite planes have no finite dimensions"
        shape = source.geom_mesh[first].shape
        size = np.asarray(source.geom_size[first], np.float32).reshape(3).copy()
        dimensions = geometry_dimensions(shape, size)
        if dimensions is None:
            return None, f"{shape.value} dimensions are not editable with a gizmo"
        caps = session.adapter.caps
        editable = bool(
            (node.model_id < 0 and caps.scene_authoring)
            or (node.source_editable and caps.topology_editing)
        )
        if not editable:
            return None, "This geometry has no editable source dimensions"
        pose_index = (
            int(source.geom_source[first])
            if len(source.geom_source) == source.instance_count
            else first
        )
        return _DimensionTarget(shape, size, dimensions, pose_index, node.node_id), ""

    @staticmethod
    def _dimension_handle_mask(dimensions: GeometryDimensions) -> int:
        axes = {item.axis for item in dimensions.handles if item.axis is not None}
        return handle_mask(
            GizmoHandle.SCREEN,
            *(AXIS_HANDLES[axis] for axis in axes),
            *(
                handle
                for normal, handle in enumerate(PLANE_HANDLES)
                if all(axis in axes for axis in range(3) if axis != normal)
            ),
        )

    @staticmethod
    def _dimension_pose(
        session: Session,
        node: SceneNode,
        target: _DimensionTarget,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if node.type is NodeType.SITE:
            return node_world_pose(session, node)
        frame = session.frame
        index = target.pose_index
        if (
            frame.geom_xpos is None
            or frame.geom_xmat is None
            or not 0 <= index < len(frame.geom_xpos)
            or index >= len(frame.geom_xmat)
        ):
            return None
        return (
            np.asarray(frame.geom_xpos[index], np.float64).reshape(3),
            np.asarray(frame.geom_xmat[index], np.float64).reshape(3, 3),
        )

    def _target_pose(
        self, session: Session, node: SceneNode, target: _JointTarget | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if target is None:
            return node_world_pose(session, self._transform_node(session, node))
        frame = session.frame
        diagnostics = frame.diagnostics
        joint_id = target.joint.joint_id
        if not _joint_frame_available(frame, joint_id):
            return None
        position = np.asarray(diagnostics.joint_xpos[joint_id], np.float64).reshape(3)
        if target.joint.type == "slide":
            # MuJoCo's xanchor excludes this slide coordinate; the driven body pose does not.
            position, _ = node_world_pose(session, node)
        if target.joint.type == "ball":
            _body_position, body_rotation = node_world_pose(session, node)
            return position, body_rotation
        axis = np.asarray(diagnostics.joint_xaxis[joint_id], np.float64).reshape(3)
        return position, _basis_from_z(axis)

    def _target_basis(
        self,
        rotation,
        target: _JointTarget | None,
        *,
        space: str | GizmoSpace | None = None,
    ) -> np.ndarray:
        if target is not None and target.joint.type in ("hinge", "slide"):
            return np.asarray(rotation, np.float64).reshape(3, 3)
        if self._mode is GizmoMode.DIMENSIONS:
            return np.asarray(rotation, np.float64).reshape(3, 3)
        selected = self._space if space is None else GizmoSpace(space)
        if selected is GizmoSpace.BODY:
            return np.asarray(rotation, np.float64).reshape(3, 3)
        return _WORLD_BASIS

    def _basis(self, rotation) -> np.ndarray:
        if self._space is GizmoSpace.BODY:
            return np.asarray(rotation, np.float64).reshape(3, 3)
        return _WORLD_BASIS

    @staticmethod
    def read_only_frame_available(session: Session, node: SceneNode | None) -> bool:
        """Return whether a non-editable spatial node can expose its coordinate frame."""

        if node is None or node.posable:
            return False
        frame = session.frame
        if node.type is NodeType.GEOM:
            return bool(
                frame.geom_xpos is not None
                and frame.geom_xmat is not None
                and 0 <= node.geom_index < len(frame.geom_xpos)
                and node.geom_index < len(frame.geom_xmat)
            )
        if node.type is NodeType.SITE:
            return bool(
                frame.site_xpos is not None
                and frame.site_xmat is not None
                and 0 <= node.site_index < len(frame.site_xpos)
                and node.site_index < len(frame.site_xmat)
            )
        if node.type in (NodeType.LINK, NodeType.ROBOT):
            return bool(
                frame.body_xpos is not None
                and frame.body_xmat is not None
                and 0 <= node.body_index < len(frame.body_xpos)
                and node.body_index < len(frame.body_xmat)
            )
        if node.type is NodeType.MODEL:
            return any(item.model_id == node.model_id for item in session.scene_models)
        return False

    def evaluate_mode(
        self, session: Session, node: SceneNode | None, mode: GizmoMode | str
    ) -> Verdict:
        """Return availability for one tool without changing the active mode."""

        selected = GizmoMode(mode)
        if selected is GizmoMode.DIMENSIONS:
            target, reason = self._dimension_target(session, node)
            return Verdict(target is not None, reason)
        return self._evaluate_transform(session, node)

    def evaluate(self, session: Session, node: SceneNode | None) -> Verdict:
        """Return availability for the active viewport gizmo mode."""

        availability = self.evaluate_mode(session, node, self._mode)
        if not availability.ok:
            return availability
        if not self.enabled and self.model_placement_model_id < 0:
            return Verdict(False, "Enable a transform tool to edit")
        return availability

    def _evaluate_transform(self, session: Session, node: SceneNode | None) -> Verdict:
        """Return position/rotation availability for one scene node."""

        if (
            node is not None
            and node.type is NodeType.MODEL
            and not self.model_placement_active(session, node.model_id)
        ):
            return Verdict(False, "Model placement is locked; use Edit Placement in the Inspector")
        target, reason = self._joint_target(session, node)
        transform_node = None if node is None else self._transform_node(session, node)
        if transform_node is not None and transform_node is not node:
            result = verdict(session.paused, transform_node)
        elif (
            node is not None
            and not node.posable
            and node.type
            in (
                NodeType.LINK,
                NodeType.ROBOT,
                NodeType.JOINT,
            )
        ):
            if not session.paused:
                return Verdict(False, gizmo_refusal_reason(False, False) or "")
            if not session.adapter.caps.write_qpos:
                return Verdict(False, f"{session.adapter.caps.name} cannot write joint positions")
            if target is None:
                return Verdict(False, reason or "joint gizmo is unavailable")
            if not _joint_frame_available(session.frame, target.joint.joint_id):
                return Verdict(False, "joint frame data is unavailable")
            result = Verdict(True)
        else:
            result = verdict(session.paused, node)
        if (
            node is not None
            and not node.posable
            and node.type
            not in (
                NodeType.LINK,
                NodeType.ROBOT,
                NodeType.JOINT,
                NodeType.LIGHT,
                NodeType.CAMERA,
            )
        ):
            result = Verdict(False, "This entity has no editable transform")
        if not result.ok or node is None:
            return result
        if (
            transform_node is not None
            and transform_node.posable
            and transform_node.type not in (NodeType.LIGHT, NodeType.CAMERA)
            and not session.adapter.caps.write_pose
        ):
            return Verdict(False, f"{session.adapter.caps.name} cannot edit this transform")
        if session.entity_gizmo_locked(transform_node):
            return Verdict(False, "gizmo is locked while simulation is running")
        if node.type is NodeType.LIGHT:
            light = _source_light(session, node)
            if light is None:
                return Verdict(False, "light transform is unavailable")
            if light.type is LightType.IMAGE:
                return Verdict(False, "image light has no spatial transform")
        return result
