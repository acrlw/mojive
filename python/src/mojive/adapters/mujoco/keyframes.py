"""Model-local keyframe identity and authoring."""

from __future__ import annotations

import numpy as np

from ... import math3d
from ..base import (
    KeyframeInfo,
    KeyframeProperties,
)
from .engine import mujoco


class _Keyframes:
    """Model-local keyframe identity and authoring.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def keyframes(self) -> list[KeyframeInfo]:
        m = self._m
        result = []
        for index in range(m.nkey):
            compiled_name = (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, index) or f"Key {index:03d}"
            )
            model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_KEY)
            result.append(
                KeyframeInfo(
                    keyframe_id=index,
                    name=name or compiled_name,
                    time=float(m.key_time[index]),
                    model_id=model_id,
                )
            )
        return result

    def _keyframe_identity(self, keyframe_id: int) -> tuple[int, str] | None:
        index = int(keyframe_id)
        if not 0 <= index < self._m.nkey:
            return None
        compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_KEY, index) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_KEY)
        return (model_id, name) if name else None

    def keyframe_properties(self, keyframe_id: int) -> KeyframeProperties | None:
        identity = self._keyframe_identity(keyframe_id)
        if identity is None:
            return None
        model_id, name = identity
        spec = self._spec_for_model(model_id)
        if spec is None or spec.key(name) is None:
            return None
        local_model = spec.copy().compile()
        local_key = mujoco.mj_name2id(local_model, mujoco.mjtObj.mjOBJ_KEY, name)
        if local_key < 0:
            return None
        return KeyframeProperties(
            keyframe_id=int(keyframe_id),
            model_id=model_id,
            name=name,
            time=float(local_model.key_time[local_key]),
            qpos=tuple(float(value) for value in local_model.key_qpos[local_key]),
            qvel=tuple(float(value) for value in local_model.key_qvel[local_key]),
            act=tuple(float(value) for value in local_model.key_act[local_key]),
            ctrl=tuple(float(value) for value in local_model.key_ctrl[local_key]),
            mocap_position=tuple(float(value) for value in local_model.key_mpos[local_key]),
            mocap_quaternion=tuple(float(value) for value in local_model.key_mquat[local_key]),
        )

    def _model_prefix(self, model_id: int) -> str:
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        return item.prefix if item is not None else ""

    def _model_transform(self, model_id: int) -> tuple[np.ndarray, np.ndarray]:
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        if item is None:
            return np.zeros(3, np.float64), np.eye(3, dtype=np.float64)
        return item.position, item.rotation

    def _model_object_offset(self, model_id: int, local, object_type, count_name: str) -> int:
        """Map local object indices to the composed model without requiring names."""
        prefix = self._model_prefix(model_id)
        count = int(getattr(local, count_name))
        start = 1 if object_type == mujoco.mjtObj.mjOBJ_BODY else 0
        for local_index in range(start, count):
            name = mujoco.mj_id2name(local, object_type, local_index) or ""
            if not name:
                continue
            compiled_index = mujoco.mj_name2id(self._m, object_type, f"{prefix}{name}")
            if compiled_index >= 0:
                return compiled_index - local_index
        if int(model_id) == 0:
            return 0
        if self._root_spec is None:
            return -1
        offset = int(getattr(self._root_spec.copy().compile(), count_name))
        for item in self._attached_models:
            if item.model_id == int(model_id):
                return offset
            offset += int(getattr(item.spec.copy().compile(), count_name))
        return -1

    def _capture_model_keyframe_values(self, model_id: int):
        spec = self._spec_for_model(model_id)
        if spec is None:
            return None
        local = spec.copy().compile()
        model_position, model_rotation = self._model_transform(model_id)
        joint_offset = self._model_object_offset(model_id, local, mujoco.mjtObj.mjOBJ_JOINT, "njnt")
        actuator_offset = self._model_object_offset(
            model_id, local, mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"
        )
        body_offset = self._model_object_offset(model_id, local, mujoco.mjtObj.mjOBJ_BODY, "nbody")
        qpos = np.asarray(local.qpos0, np.float64).copy()
        qvel = np.zeros(local.nv, np.float64)
        act = np.zeros(local.na, np.float64)
        ctrl = np.zeros(local.nu, np.float64)
        mocap_position = np.zeros((local.nmocap, 3), np.float64)
        mocap_quaternion = np.zeros((local.nmocap, 4), np.float64)
        if local.nmocap:
            mocap_quaternion[:, 0] = 1.0

        for local_joint in range(local.njnt):
            compiled_joint = local_joint + joint_offset
            if not 0 <= compiled_joint < self._m.njnt:
                continue
            local_qpos = self._span(local.jnt_qposadr, local_joint, local.nq)
            compiled_qpos = self._span(self._m.jnt_qposadr, compiled_joint, self._m.nq)
            local_qvel = self._span(local.jnt_dofadr, local_joint, local.nv)
            compiled_qvel = self._span(self._m.jnt_dofadr, compiled_joint, self._m.nv)
            if local_qpos.stop - local_qpos.start == compiled_qpos.stop - compiled_qpos.start:
                qpos[local_qpos] = self._d.qpos[compiled_qpos]
                if int(local.jnt_type[local_joint]) == mujoco.mjtJoint.mjJNT_FREE:
                    values = qpos[local_qpos]
                    values[:3] = (values[:3] - model_position) @ model_rotation
                    values[3:7] = math3d.mat3_to_quat(
                        model_rotation.T @ math3d.quat_to_mat3(values[3:7])
                    )
            if local_qvel.stop - local_qvel.start == compiled_qvel.stop - compiled_qvel.start:
                qvel[local_qvel] = self._d.qvel[compiled_qvel]

        for local_actuator in range(local.nu):
            compiled_actuator = local_actuator + actuator_offset
            if not 0 <= compiled_actuator < self._m.nu:
                continue
            local_ctrl = self._span(local.actuator_ctrladr, local_actuator, local.nu)
            compiled_ctrl = self._span(self._m.actuator_ctrladr, compiled_actuator, self._m.nu)
            if local_ctrl.stop - local_ctrl.start == compiled_ctrl.stop - compiled_ctrl.start:
                ctrl[local_ctrl] = self._d.ctrl[compiled_ctrl]
            local_activation = self._span(local.actuator_actadr, local_actuator, local.na)
            compiled_activation = self._span(self._m.actuator_actadr, compiled_actuator, self._m.na)
            if (
                local_activation.stop - local_activation.start
                == compiled_activation.stop - compiled_activation.start
            ):
                act[local_activation] = self._d.act[compiled_activation]

        for local_body in range(1, local.nbody):
            local_mocap = int(local.body_mocapid[local_body])
            if local_mocap < 0:
                continue
            compiled_body = local_body + body_offset
            compiled_mocap = int(self._m.body_mocapid[compiled_body]) if compiled_body >= 0 else -1
            if compiled_mocap >= 0:
                mocap_position[local_mocap] = (
                    self._d.mocap_pos[compiled_mocap] - model_position
                ) @ model_rotation
                mocap_quaternion[local_mocap] = math3d.mat3_to_quat(
                    model_rotation.T @ math3d.quat_to_mat3(self._d.mocap_quat[compiled_mocap])
                )
        return (
            qpos,
            qvel,
            act,
            ctrl,
            mocap_position.reshape(-1),
            mocap_quaternion.reshape(-1),
        )

    def add_model_keyframe(self, model_id: int, name: str) -> int:
        value = str(name).strip()
        source_spec = self._spec_for_model(model_id)
        values = self._capture_model_keyframe_values(model_id)
        if (
            source_spec is None
            or values is None
            or not value
            or any(key.name == value for key in source_spec.keys)
        ):
            return -1
        working = source_spec.copy()
        working.add_key(
            name=value,
            time=float(self._d.time),
            qpos=values[0],
            qvel=values[1],
            act=values[2],
            ctrl=values[3],
            mpos=values[4],
            mquat=values[5],
        )
        if not self._replace_model_spec(model_id, working):
            return -1
        if self._model_edit_batch_depth:
            # The batch installs its compiled model on context exit. Resolve the
            # stable declaration order here instead of querying the previous model.
            target = int(model_id)
            keyframe_id = 0
            specs = ((0, self._root_spec),) if self._root_spec is not None else ()
            specs += tuple((item.model_id, item.spec) for item in self._attached_models)
            for owner, spec in specs:
                for key in spec.keys:
                    if int(owner) == target and key.name == value:
                        return keyframe_id
                    keyframe_id += 1
            return -1
        return mujoco.mj_name2id(
            self._m,
            mujoco.mjtObj.mjOBJ_KEY,
            f"{self._model_prefix(model_id)}{value}",
        )

    def set_keyframe_properties(self, properties: KeyframeProperties) -> bool:
        identity = self._keyframe_identity(properties.keyframe_id)
        if identity is None or identity[0] != int(properties.model_id):
            return False
        model_id, current_name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        local = source_spec.copy().compile()
        expected = (
            local.nq,
            local.nv,
            local.na,
            local.nu,
            local.nmocap * 3,
            local.nmocap * 4,
        )
        arrays = tuple(
            np.asarray(values, np.float64).reshape(-1)
            for values in (
                properties.qpos,
                properties.qvel,
                properties.act,
                properties.ctrl,
                properties.mocap_position,
                properties.mocap_quaternion,
            )
        )
        if tuple(len(values) for values in arrays) != expected:
            return False
        working = source_spec.copy()
        element = working.key(current_name)
        value = str(properties.name).strip()
        duplicate = working.key(value)
        if element is None or not value or (value != current_name and duplicate is not None):
            return False
        element.name = value
        element.time = float(properties.time)
        element.qpos = arrays[0]
        element.qvel = arrays[1]
        element.act = arrays[2]
        element.ctrl = arrays[3]
        element.mpos = arrays[4]
        element.mquat = arrays[5]
        return self._replace_model_spec(model_id, working)

    def remove_model_keyframe(self, keyframe_id: int) -> bool:
        identity = self._keyframe_identity(keyframe_id)
        if identity is None:
            return False
        model_id, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        working = source_spec.copy()
        element = working.key(name)
        if element is None:
            return False
        working.delete(element)
        return self._replace_model_spec(model_id, working)

    def load_keyframe(self, keyframe_id: int) -> bool:
        i = int(keyframe_id)
        if not 0 <= i < self._m.nkey:
            return False
        mujoco.mj_resetDataKeyframe(self._m, self._d, i)
        mujoco.mj_forward(self._m, self._d)
        self._perturb_body = -1
        return True
