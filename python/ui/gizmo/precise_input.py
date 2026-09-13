"""Gizmo: precise input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive import math3d
from mojive.adapters.base import NodeType
from mojive.commands import (
    ClearSceneModelTransformPreview,
    CommandResult,
    PreviewSceneModelTransform,
    SetGeometrySize,
    SetSceneModelTransform,
)
from mojive.interaction.gizmo import (
    ALL_HANDLE_MASK,
    AXIS_HANDLES,
    ROTATE_AXIS_HANDLES,
    ROTATE_HANDLES,
    GizmoHandle,
    GizmoMode,
    GizmoSpace,
)
from mojive.scene.geometry import geometry_size_from_dimensions
from mojive.types import CameraView

if TYPE_CHECKING:
    from mojive.adapters.base import SceneNode
    from mojive.session import Session


from .state import PreciseGizmoInput, Verdict, _axis_of


class _PreciseInput:
    """Private precise input methods of ObjectGizmo; state belongs to its owner."""

    @property
    def model_placement_model_id(self) -> int:
        return self._model_placement_model

    def model_placement_active(self, session: Session, model_id: int | None = None) -> bool:
        active = (
            self._model_placement_model >= 0
            and self._model_placement_session is session
            and self._model_placement_generation == session.adapter.structure_revision
        )
        return active and (model_id is None or self._model_placement_model == int(model_id))

    def begin_model_placement(self, session: Session, model_id: int) -> CommandResult:
        """Unlock one model root for preview-only placement edits."""

        model_id = int(model_id)
        if self.model_placement_active(session, model_id):
            return CommandResult.good("Model placement is already unlocked")
        if self._model_placement_model >= 0:
            result = self.cancel_model_placement(session)
            if not result.ok:
                return result
        if not session.paused:
            return CommandResult.bad("Pause the simulation before editing model placement")
        info = next((item for item in session.scene_models if item.model_id == model_id), None)
        if info is None or not info.removable:
            return CommandResult.bad(f"Model {model_id} placement cannot be edited")
        position = np.asarray(info.position, np.float64).reshape(3).copy()
        rotation = np.asarray(info.rotation, np.float64).reshape(3, 3).copy()
        self._model_placement_model = model_id
        self._model_placement_generation = session.adapter.structure_revision
        self._model_placement_session = session
        self._model_placement_original = (position, rotation)
        return CommandResult.good("Model placement unlocked; Apply rebuilds the composed model")

    def model_placement_transform(
        self, session: Session, model_id: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.model_placement_active(session, model_id):
            return None
        if self._model_preview is not None and self._model_preview[0] == int(model_id):
            return self._model_preview[1].copy(), self._model_preview[2].copy()
        original = self._model_placement_original
        if original is None:
            return None
        return original[0].copy(), original[1].copy()

    def preview_model_placement(
        self, session: Session, model_id: int, position, rotation
    ) -> CommandResult:
        """Update render-frame placement without compiling the model."""

        model_id = int(model_id)
        if not self.model_placement_active(session, model_id):
            return CommandResult.bad("Use Edit Placement in the Inspector before moving a model")
        position = np.asarray(position, np.float64).reshape(3).copy()
        rotation = np.asarray(rotation, np.float64).reshape(3, 3).copy()
        result = session.submit(PreviewSceneModelTransform(model_id, position, rotation))
        if result.ok:
            self._model_preview = (model_id, position, rotation)
            self._model_preview_session = session
        return result

    def apply_model_placement(self, session: Session) -> CommandResult:
        """Commit one staged model placement, compiling at most once."""

        model_id = self._model_placement_model
        if not self.model_placement_active(session, model_id):
            self._reset_model_placement()
            return CommandResult.bad("Model placement preview is no longer valid")
        preview = self._model_preview
        original = self._model_placement_original
        if preview is None or original is None:
            self._reset_model_placement()
            return CommandResult.good("Model placement unchanged")
        changed = not (
            np.array_equal(preview[1], original[0]) and np.array_equal(preview[2], original[1])
        )
        if not changed:
            result = session.submit(ClearSceneModelTransformPreview(model_id))
            if result.ok:
                self._reset_model_placement()
                return CommandResult.good("Model placement unchanged")
            return result
        result = session.submit(SetSceneModelTransform(model_id, preview[1], preview[2]))
        if result.ok:
            self._reset_model_placement()
        return result

    def cancel_model_placement(self, session: Session) -> CommandResult:
        """Discard a staged placement without compiling the model."""

        model_id = self._model_placement_model
        if model_id < 0:
            return CommandResult.good()
        preview_is_current = (
            self._model_preview is not None
            and self._model_preview_session is session
            and self._model_placement_session is session
            and self._model_placement_generation == session.adapter.structure_revision
        )
        if preview_is_current:
            result = session.submit(ClearSceneModelTransformPreview(model_id))
            if not result.ok:
                return result
        self._reset_model_placement()
        return CommandResult.good("Cancelled model placement")

    def _reset_model_placement(self) -> None:
        self._model_preview = None
        self._model_preview_session = None
        self._model_placement_model = -1
        self._model_placement_generation = -1
        self._model_placement_session = None
        self._model_placement_original = None

    def precise_input(self, session: Session) -> PreciseGizmoInput | None:
        """Describe the hovered scalar handle for a relative numeric edit."""

        node = session.selected_node
        handle = self._hovered
        dimension_handle = self._mode is GizmoMode.DIMENSIONS and handle is GizmoHandle.SCREEN
        if node is None or (
            handle not in (*AXIS_HANDLES, *ROTATE_HANDLES) and not dimension_handle
        ):
            return None
        if handle is GizmoHandle.ROTATE_TRACKBALL:
            return None
        if not self.evaluate(session, node).ok:
            return None
        if self._mode is GizmoMode.DIMENSIONS:
            target, _reason = self._dimension_target(session, node)
            axis = None if handle is GizmoHandle.SCREEN else _axis_of(handle)
            mapping = None if target is None else target.dimensions.handle(axis)
            if target is None or mapping is None:
                return None
            return PreciseGizmoInput(
                handle=handle,
                object_id=int(session.selected),
                node_id=int(node.node_id),
                joint_id=-1,
                action="Resize",
                label=mapping.label,
                unit="m",
                space=GizmoSpace.BODY.value,
                absolute_value=float(target.dimensions.values[mapping.parameter]),
                absolute_label=f"target {mapping.label}",
                dimension_index=mapping.parameter,
            )
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
        absolute_value = None
        absolute_label = ""
        if target is not None and target.joint.type in ("hinge", "slide"):
            qpos = session.frame.qpos
            address = int(target.joint.qpos_adr)
            if qpos is not None and 0 <= address < len(qpos):
                absolute_value = float(qpos[address])
                if target.joint.type == "hinge":
                    absolute_value = float(np.degrees(absolute_value))
                absolute_label = f"target {label} joint position"
        # Absolute body-frame and screen rotations are not scalar coordinates. World-frame
        # axis input deliberately matches the Inspector's extrinsic XYZ convention.
        elif self._space is GizmoSpace.WORLD and handle is not GizmoHandle.ROTATE_SCREEN:
            pose = self._target_pose(session, node, target)
            if pose is not None:
                position, rotation = pose
                if handle in AXIS_HANDLES:
                    absolute_value = float(position[axis])
                    absolute_label = f"target world {handle_name} position"
                elif handle in ROTATE_AXIS_HANDLES:
                    euler = np.degrees(math3d.mat3_to_euler_xyz(rotation))
                    absolute_value = float(euler[axis])
                    absolute_label = f"target world {handle_name} rotation"
        return PreciseGizmoInput(
            handle=handle,
            object_id=int(session.selected),
            node_id=int(node.node_id),
            joint_id=joint_id,
            action="Rotate" if rotating else "Move",
            label=label,
            unit="°" if rotating else "m",
            space=self._space.value,
            absolute_value=absolute_value,
            absolute_label=absolute_label,
        )

    def apply_precise_value(
        self,
        session: Session,
        cam: CameraView,
        edit: PreciseGizmoInput,
        value: float,
        *,
        absolute: bool = False,
    ) -> CommandResult:
        """Apply one exact relative delta or unambiguous absolute scalar value."""

        amount = float(value)
        if not np.isfinite(amount):
            return CommandResult.bad("Enter a finite numeric value")
        if absolute and edit.absolute_value is None:
            return CommandResult.bad("Absolute input is unavailable for this gizmo handle")
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
        if edit.dimension_index >= 0:
            return self._apply_precise_dimensions(session, node, edit, amount, absolute=absolute)
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
        basis = self._target_basis(rotation, target, space=edit.space)
        axis_index = _axis_of(edit.handle)
        if edit.handle not in (*AXIS_HANDLES, *ROTATE_HANDLES):
            return CommandResult.bad("Precise input requires a scalar gizmo handle")
        if axis_index < 0 and edit.handle is not GizmoHandle.ROTATE_SCREEN:
            return CommandResult.bad("Precise input requires a single gizmo axis")

        axis = (
            -np.asarray(cam.forward(), np.float64)
            if edit.handle is GizmoHandle.ROTATE_SCREEN
            else basis[:, axis_index]
        )
        applied_amount = (
            float(np.radians(amount))
            if target is not None and edit.handle in ROTATE_HANDLES
            else amount
        )
        if target is not None:
            joint = target.joint
            qpos = session.frame.qpos
            count = 4 if joint.type == "ball" else 1
            start = int(joint.qpos_adr)
            if qpos is None or start < 0 or start + count > len(qpos):
                return CommandResult.bad("Joint position data is unavailable")
            joint_qpos = np.asarray(qpos[start : start + count], np.float64).copy()
            self._start_joint_qpos = joint_qpos.copy()
            if absolute and joint.type in ("hinge", "slide"):
                target_value = np.radians(amount) if joint.type == "hinge" else amount
                applied_amount = float(target_value - joint_qpos[0])

        if edit.handle in AXIS_HANDLES:
            if absolute and target is None:
                position[axis_index] = amount
                applied_amount = amount - float(pose[0][axis_index])
            else:
                position += axis * applied_amount
        elif absolute and target is None:
            euler = np.asarray(math3d.mat3_to_euler_xyz(rotation), np.float64)
            applied_amount = float(np.radians(amount) - euler[axis_index])
            euler[axis_index] = np.radians(amount)
            rotation = math3d.euler_xyz_to_mat3(euler)
        else:
            angle = applied_amount if target is not None else np.radians(applied_amount)
            rotation = math3d.rotvec_to_mat3(axis * angle) @ rotation

        if abs(applied_amount) < 1e-12:
            return CommandResult.good("No change")
        self._active = edit.handle
        self._active_joint = target.joint if target is not None else None
        np.copyto(self._start_pos, pose[0])
        np.copyto(self._start_mat, pose[1])
        np.copyto(self._start_basis, basis)
        self._axis[:] = axis
        self._rotation_angle = 0.0
        if edit.handle in ROTATE_HANDLES:
            self._rotation_angle = (
                applied_amount
                if target is not None or absolute
                else float(np.radians(applied_amount))
            )
        result, _position = self._submit_transform(
            session,
            node,
            position,
            rotation,
            preview_model=node.type is NodeType.MODEL,
        )
        self._end()
        if not result.ok:
            self._verdict = Verdict(False, result.message)
        return result

    def _apply_precise_dimensions(
        self,
        session: Session,
        node: SceneNode,
        edit: PreciseGizmoInput,
        amount: float,
        *,
        absolute: bool,
    ) -> CommandResult:
        target, reason = self._dimension_target(session, node)
        axis = None if edit.handle is GizmoHandle.SCREEN else _axis_of(edit.handle)
        mapping = None if target is None else target.dimensions.handle(axis)
        if target is None or mapping is None or mapping.parameter != edit.dimension_index:
            return CommandResult.bad(
                reason or "The geometry dimension changed; reopen precise input"
            )
        values = target.dimensions.array().astype(np.float64)
        current = float(values[mapping.parameter])
        requested = amount if absolute else current + amount
        if requested < 0.002:
            return CommandResult.bad("Geometry dimensions must be at least 0.002 m")
        if abs(requested - current) < 1e-12:
            return CommandResult.good("No change")
        values[mapping.parameter] = requested
        self._active = edit.handle
        self._start_edit(session)
        result = session.submit(
            SetGeometrySize(
                target.node_id,
                geometry_size_from_dimensions(target.shape, target.size, values),
            )
        )
        self._end(commit=result.ok)
        if not result.ok:
            self._verdict = Verdict(False, result.message)
        return result
