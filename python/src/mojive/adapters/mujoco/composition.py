"""Model identity, attachment, recompilation and transactional state restoration."""

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from ... import math3d
from ..base import (
    GEOMETRY_OBJECT_BASE,
    NodeType,
    SceneModelInfo,
    SceneNode,
)
from .engine import mujoco
from .schema import _FLEX_COPY_FIELDS
from .spec import _load_editable_spec
from .state import _AttachedModel, _CompositionEditState, _ModelTransformPreview, _NamedModelState


class _ModelComposition:
    """Model identity, attachment, recompilation and transactional state restoration.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def scene_models(self) -> tuple[SceneModelInfo, ...]:
        roots = (
            (SceneModelInfo(0, self._root_path.stem, self._root_path, False),)
            if self._root_path is not None
            else ()
        )
        preview = self._model_transform_preview
        attached = []
        for item in self._attached_models:
            position = item.position
            rotation = item.rotation
            if preview is not None and preview.model_id == item.model_id:
                position = preview.position
                rotation = preview.rotation
            attached.append(
                SceneModelInfo(
                    item.model_id,
                    item.name,
                    item.path,
                    True,
                    tuple(float(value) for value in position),
                    tuple(tuple(float(value) for value in row) for row in rotation),
                )
            )
        return (*roots, *attached)

    def capture_edit_state(self) -> object | None:
        if self._root_spec is None:
            return None
        models = tuple(
            _AttachedModel(
                item.model_id,
                item.name,
                item.path,
                item.prefix,
                item.position.copy(),
                item.rotation.copy(),
                item.spec.copy(),
                item.edited,
            )
            for item in self._attached_models
        )
        return _CompositionEditState(
            models,
            self.capture_state(),
            self._root_spec.copy(),
            self._root_edited,
            dict(self._geometry_object_ids),
            self._next_geometry_object_id,
            dict(self._component_entries),
        )

    def restore_edit_state(self, state: object) -> bool:
        if not isinstance(state, _CompositionEditState) or self._root_spec is None:
            return False
        self._attached_models = [
            _AttachedModel(
                item.model_id,
                item.name,
                item.path,
                item.prefix,
                item.position.copy(),
                item.rotation.copy(),
                item.spec.copy(),
                item.edited,
            )
            for item in state.models
        ]
        self._root_spec = state.root_spec.copy()
        self._root_edited = state.root_edited
        self._geometry_object_ids = dict(state.geometry_object_ids)
        self._next_geometry_object_id = state.next_geometry_object_id
        self._component_entries = dict(state.component_entries)
        for key, entries in self._component_entries.items():
            self._next_component_id[key] = max(
                self._next_component_id.get(key, 0),
                max((entry.component_id + 1 for entry in entries), default=0),
            )
        self._reset_next_model_id()
        self._install(self._compile_composed_model())
        return self.restore_state(state.physics)

    def add_scene_model(self, path: Path, position, rotation) -> int:
        if self._root_spec is None:
            return -1
        path = Path(path).expanduser().resolve()
        model_id = self._allocate_model_id()
        spec = _load_editable_spec(path)
        # MjSpec.copy() can omit unresolved declarations originating in an include.
        # Compiling keyed models once resolves their full actuator/state layout before
        # the stored editable spec is copied for composition.
        self._resolve_attached_keyframes(spec)
        item = _AttachedModel(
            model_id=model_id,
            name=path.stem,
            path=path,
            prefix=f"opengl_{model_id}_",
            position=np.asarray(position, np.float64).reshape(3).copy(),
            rotation=np.asarray(rotation, np.float64).reshape(3, 3).copy(),
            spec=spec,
        )
        state = self._capture_named_model_state()
        self._attached_models.append(item)
        try:
            model = self._compile_composed_model()
        except Exception as exc:
            self._attached_models.pop()
            raise RuntimeError(f"Failed to add {path}: {exc}") from exc
        self._install(model)
        self._restore_named_model_state(state)
        return model_id

    def _reset_next_model_id(self) -> None:
        """Advance the composition namespace past IDs already present in the document."""
        used = {item.model_id for item in self._attached_models}
        if self._root_spec is not None:
            used.update(
                int(match.group(1))
                for frame in self._root_spec.frames
                if (match := re.fullmatch(r"mojive_model_(\d+)", frame.name or ""))
            )
        self._next_model_id = max(used, default=0) + 1

    def _allocate_model_id(self) -> int:
        """Reserve the next collision-free namespace for an attached model."""
        model_id = self._next_model_id
        self._next_model_id += 1
        return model_id

    def remove_scene_model(self, model_id: int) -> bool:
        index = next(
            (
                index
                for index, item in enumerate(self._attached_models)
                if item.model_id == int(model_id)
            ),
            -1,
        )
        if index < 0:
            return False
        state = self._capture_named_model_state()
        item = self._attached_models.pop(index)
        try:
            model = self._compile_composed_model()
        except Exception as exc:
            self._attached_models.insert(index, item)
            raise RuntimeError(f"Failed to remove {item.name}: {exc}") from exc
        self._install(model)
        self._restore_named_model_state(state)
        return True

    def set_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        if item is None:
            return False
        next_position = np.asarray(position, np.float64).reshape(3)
        next_rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        if np.array_equal(item.position, next_position) and np.array_equal(
            item.rotation, next_rotation
        ):
            self._model_transform_preview = None
            return True
        previous_position = item.position.copy()
        previous_rotation = item.rotation.copy()
        state = self._transform_named_model_state(
            self._capture_named_model_state(),
            item.prefix,
            previous_position,
            previous_rotation,
            next_position,
            next_rotation,
        )
        item.position[:] = next_position
        item.rotation[:] = next_rotation
        try:
            model = self._compile_composed_model()
        except Exception:
            item.position[:] = previous_position
            item.rotation[:] = previous_rotation
            raise
        self._install(model)
        self._restore_named_model_state(state)
        return True

    def preview_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        """Preview placement in frame buffers; physics remains committed until release."""
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        if item is None:
            return False
        next_position = np.asarray(position, np.float64).reshape(3)
        next_rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        if not np.all(np.isfinite(next_position)) or not np.all(np.isfinite(next_rotation)):
            raise ValueError("Model transform preview must contain finite values")
        preview = self._model_transform_preview
        if preview is None or preview.model_id != item.model_id:
            preview = self._make_model_transform_preview(item)
            self._model_transform_preview = preview
        preview.position[:] = next_position
        preview.rotation[:] = next_rotation
        preview.delta_rotation[:] = next_rotation @ item.rotation.T
        return True

    def clear_scene_model_transform_preview(self, model_id: int) -> bool:
        preview = self._model_transform_preview
        if preview is None or preview.model_id != int(model_id):
            return False
        self._model_transform_preview = None
        return True

    def _make_model_transform_preview(self, item: _AttachedModel) -> _ModelTransformPreview:
        model = self._m
        body_owner = np.zeros(model.nbody, np.int32)
        for body in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            owner, _ = self._model_element_name(name, mujoco.mjtObj.mjOBJ_BODY)
            body_owner[body] = owner if owner else body_owner[int(model.body_parentid[body])]

        def owned_indices(object_type, count: int, body_ids=None) -> np.ndarray:
            owned = []
            for index in range(count):
                name = mujoco.mj_id2name(model, object_type, index) or ""
                owner, _ = self._model_element_name(name, object_type)
                if not owner and body_ids is not None:
                    owner = int(body_owner[int(body_ids[index])])
                if owner == item.model_id:
                    owned.append(index)
            return np.asarray(owned, np.intp)

        body_indices = np.flatnonzero(body_owner == item.model_id).astype(np.intp)
        geom_indices = owned_indices(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom, model.geom_bodyid)
        site_indices = owned_indices(mujoco.mjtObj.mjOBJ_SITE, model.nsite, model.site_bodyid)
        joint_indices = owned_indices(mujoco.mjtObj.mjOBJ_JOINT, model.njnt, model.jnt_bodyid)
        light_indices = owned_indices(mujoco.mjtObj.mjOBJ_LIGHT, model.nlight, model.light_bodyid)
        camera_indices = owned_indices(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam, model.cam_bodyid)
        light_mask = np.zeros(model.nlight, bool)
        camera_mask = np.zeros(model.ncam, bool)
        light_mask[light_indices] = True
        camera_mask[camera_indices] = True
        capacity = max(
            1,
            len(body_indices),
            len(geom_indices),
            len(site_indices),
            len(joint_indices),
        )
        return _ModelTransformPreview(
            model_id=item.model_id,
            previous_position=item.position.copy(),
            previous_rotation=item.rotation.copy(),
            position=item.position.copy(),
            rotation=item.rotation.copy(),
            delta_rotation=np.eye(3, dtype=np.float64),
            body_indices=body_indices,
            geom_indices=geom_indices,
            site_indices=site_indices,
            joint_indices=joint_indices,
            light_mask=light_mask,
            camera_mask=camera_mask,
            point_input=np.zeros((capacity, 3), np.float32),
            point_output=np.zeros((capacity, 3), np.float32),
            matrix_input=np.zeros((capacity, 3, 3), np.float32),
            matrix_output=np.zeros((capacity, 3, 3), np.float32),
        )

    def _spec_from_component_xml(self, model_id: int, xml: str):
        spec = mujoco.MjSpec.from_string(xml)
        path = next(
            (item.path for item in self._attached_models if item.model_id == int(model_id)),
            self._root_path,
        )
        if path is not None:
            spec.modelfiledir = str(path.parent)
        return spec

    def _replace_model_spec(self, model_id: int, spec) -> bool:
        # MjSpec.attach can defer a broken local reference until later serialization.
        # Validate the edited standalone model before it enters the composed document.
        spec.to_xml()
        state = self._capture_named_model_state()
        if int(model_id) == 0:
            if self._root_spec is None:
                return False
            previous_spec = self._root_spec
            previous_edited = self._root_edited
            self._root_spec = spec
            self._root_edited = True
            try:
                model = self._compile_composed_model()
            except Exception:
                self._root_spec = previous_spec
                self._root_edited = previous_edited
                raise
        else:
            index = next(
                (
                    index
                    for index, item in enumerate(self._attached_models)
                    if item.model_id == int(model_id)
                ),
                -1,
            )
            if index < 0:
                return False
            previous = self._attached_models[index]
            self._attached_models[index] = replace(previous, spec=spec, edited=True)
            try:
                model = self._compile_composed_model()
            except Exception:
                self._attached_models[index] = previous
                raise
        self._install(model)
        self._restore_named_model_state(state)
        return True

    def _recompile_topology(self) -> None:
        state = self._capture_named_model_state()
        self._install(self._compile_composed_model())
        self._restore_named_model_state(state)

    def _spec_for_model(self, model_id: int):
        if int(model_id) == 0:
            return self._root_spec
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        return item.spec if item is not None else None

    def _mark_model_edited(self, model_id: int) -> None:
        if int(model_id) == 0:
            self._root_edited = True
            return
        index = next(
            (
                index
                for index, item in enumerate(self._attached_models)
                if item.model_id == int(model_id)
            ),
            -1,
        )
        if index >= 0:
            self._attached_models[index] = replace(self._attached_models[index], edited=True)

    def _store_model_spec(self, model_id: int, spec) -> None:
        """Replace one stored editable spec without recompiling the composed model."""
        path = next(
            (item.path for item in self._attached_models if item.model_id == int(model_id)),
            self._root_path,
        )
        if path is not None:
            # MjSpec.from_string() drops modelfiledir.  Material texture edits
            # reparse the spec below, so restore the model's resource root before
            # a later edit, export, or topology rebuild resolves relative meshes.
            spec.modelfiledir = str(path.parent)
        if int(model_id) == 0:
            self._root_spec = spec
            self._root_edited = True
            return
        index = next(
            (
                index
                for index, item in enumerate(self._attached_models)
                if item.model_id == int(model_id)
            ),
            -1,
        )
        if index >= 0:
            self._attached_models[index] = replace(
                self._attached_models[index], spec=spec, edited=True
            )

    def _element(self, model_id: int, element_type: str, name: str):
        spec = self._spec_for_model(model_id)
        if spec is None:
            return None
        lookup = "body" if element_type in ("link", "robot") else element_type
        finder = getattr(spec, lookup, None)
        element = finder(name) if finder is not None else None
        if element is not None and element.name == name:
            return element
        # MjSpec name indices can lag behind insertion or rename, even returning
        # another element at the old index. Verify the identity before write-back.
        collection = getattr(
            spec,
            {
                "body": "bodies",
                "geom": "geoms",
                "joint": "joints",
                "site": "sites",
                "camera": "cameras",
                "light": "lights",
            }.get(lookup, ""),
            (),
        )
        return next((item for item in collection if item.name == name), None)

    def _reset_geometry_object_ids(self) -> None:
        self._geometry_object_ids.clear()
        self._next_geometry_object_id = GEOMETRY_OBJECT_BASE

    def _geometry_object_id(self, model_id: int, name: str) -> int:
        identity = (int(model_id), str(name))
        object_id = self._geometry_object_ids.get(identity)
        if object_id is None:
            object_id = self._next_geometry_object_id
            self._next_geometry_object_id += 1
            self._geometry_object_ids[identity] = object_id
        return object_id

    def _node_for_id(self, node_id: int) -> SceneNode | None:
        nodes = self.nodes()
        index = int(node_id)
        if 0 <= index < len(nodes) and nodes[index].node_id == index:
            return nodes[index]
        return None

    def _model_parent(self, node_id: int):
        node = self._node_for_id(node_id)
        while node is not None:
            identity = self._node_element.get(node.node_id)
            if identity is not None:
                model_id, node_type, name = identity
                spec = self._spec_for_model(model_id)
                if spec is None:
                    return None
                if node_type in (NodeType.MODEL, NodeType.WORLD):
                    return model_id, spec.worldbody
                if node_type in (NodeType.LINK, NodeType.ROBOT):
                    return model_id, spec.body(name)
            node = self._node_for_id(node.parent)
        return None

    def _composed_spec(self):
        if self._root_spec is None:
            raise RuntimeError("Model composition is unavailable")
        spec = self._root_spec.copy()
        if self._root_path is not None:
            self._resolve_asset_paths(spec, self._root_path.parent)
        for index, item in enumerate(self._attached_models):
            child = item.spec.copy()
            self._resolve_asset_paths(child, item.path.parent)
            # Resolve keyframes inherited from nested model assets before the child is
            # attached again. MuJoCo can then namespace their compiled state correctly.
            child_model = self._resolve_attached_keyframes(child)
            if child_model is not None:
                self._transform_attached_keyframes(
                    child,
                    child_model,
                    item.position,
                    item.rotation,
                )
            if index == 0 and self._root_path is None:
                spec.option = child.option
                spec.visual = child.visual
                spec.stat = child.stat
                spec.compiler = child.compiler
                spec.memory = child.memory
            child.option = spec.option
            frame = spec.worldbody.add_frame(name=f"mojive_model_{item.model_id}")
            frame.pos = item.position
            frame.quat = math3d.mat3_to_quat(item.rotation)
            self._copy_world_attached_flexes(
                spec,
                child,
                item.prefix,
                item.position,
                item.rotation,
            )
            skin_start = len(spec.skins)
            skin_names = {skin.name for skin in child.skins}
            material_names = {material.name for material in child.materials}
            spec.attach(child, prefix=item.prefix, frame=frame)
            # MjSpec namespaces skin bone bodies but leaves skin names and material
            # references unchanged. Limit repair to the newly attached assets.
            for skin in list(spec.skins)[skin_start:]:
                if skin.name and skin.name in skin_names:
                    skin.name = f"{item.prefix}{skin.name}"
                if skin.material and skin.material in material_names:
                    skin.material = f"{item.prefix}{skin.material}"
            self._restore_attached_world_targets(spec, item.prefix)
        return spec

    @staticmethod
    def _resolve_attached_keyframes(spec):
        """Compile an attached spec early only when unresolved keyframes require it."""
        if len(spec.keys):
            return spec.compile()
        return None

    @staticmethod
    def _transform_attached_keyframes(spec, model, position, rotation) -> None:
        """Transform model-local free and mocap poses before MjSpec attachment.

        MuJoCo namespaces attached keyframes but leaves their pose arrays unchanged,
        even though those arrays become world-space in the composed model.
        """
        offset = np.asarray(position, np.float64).reshape(3)
        matrix = np.asarray(rotation, np.float64).reshape(3, 3)
        for key_index, key in enumerate(spec.keys):
            qpos = np.asarray(model.key_qpos[key_index], np.float64).copy()
            for joint in range(model.njnt):
                if int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                address = int(model.jnt_qposadr[joint])
                qpos[address : address + 3] = qpos[address : address + 3] @ matrix.T + offset
                qpos[address + 3 : address + 7] = math3d.mat3_to_quat(
                    matrix @ math3d.quat_to_mat3(qpos[address + 3 : address + 7])
                )
            key.qpos = qpos
            if model.nmocap:
                mocap_position = np.asarray(model.key_mpos[key_index], np.float64).reshape(-1, 3)
                mocap_quaternion = np.asarray(model.key_mquat[key_index], np.float64).reshape(-1, 4)
                key.mpos = (mocap_position @ matrix.T + offset).reshape(-1)
                key.mquat = np.asarray(
                    [
                        math3d.mat3_to_quat(matrix @ math3d.quat_to_mat3(quaternion))
                        for quaternion in mocap_quaternion
                    ]
                ).reshape(-1)

    @contextmanager
    def model_edit_batch(self):
        """Keep declaration edits on MjSpec and install one final compiled model."""
        if self._model_edit_batch_depth:
            raise RuntimeError("Nested model rebuild batches are unsupported")
        self._model_edit_batch_depth = 1
        self._model_edit_batch_rebuild = False
        try:
            yield
            state = self._capture_named_model_state() if self._model_edit_batch_rebuild else None
        except BaseException:
            self._model_edit_batch_rebuild = False
            raise
        finally:
            self._model_edit_batch_depth = 0
        try:
            if self._model_edit_batch_rebuild:
                self._install(self._compile_composed_model())
                self._restore_named_model_state(state)
        finally:
            self._model_edit_batch_rebuild = False

    def _compile_composed_model(self):
        if self._model_edit_batch_depth:
            self._model_edit_batch_rebuild = True
            return self._m
        spec = self._composed_spec()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attach conflict.*")
            return spec.compile()

    @staticmethod
    def _resolve_asset_paths(spec, directory: Path) -> None:
        def asset_base(asset, compiler_field: str) -> Path:
            compiler = getattr(asset, "compiler", spec.compiler)
            relative = str(getattr(compiler, compiler_field, ""))
            return directory / relative if relative else directory

        for assets, compiler_field in (
            (spec.meshes, "meshdir"),
            (spec.textures, "texturedir"),
            (spec.hfields, ""),
            (spec.skins, "meshdir"),
        ):
            for asset in assets:
                file = str(asset.file)
                if file and not Path(file).is_absolute():
                    base = asset_base(asset, compiler_field) if compiler_field else directory
                    asset.file = str((base / file).resolve())
        spec.compiler.meshdir = ""
        spec.compiler.texturedir = ""

    @staticmethod
    def _restore_attached_world_targets(spec, prefix: str) -> None:
        """Restore camera and light targets that refer to MuJoCo's world body.

        MjSpec attachment namespaces every explicit target body. The world body is
        shared by the composed model and retains the reserved name ``world``.
        """

        namespaced_world = f"{prefix}world"
        for camera in spec.cameras:
            if camera.targetbody == namespaced_world:
                camera.targetbody = "world"
        for light in spec.lights:
            if light.targetbody == namespaced_world:
                light.targetbody = "world"

    @staticmethod
    def _copy_world_attached_flexes(spec, child, prefix: str, position, rotation) -> None:
        """Copy world-referencing flexes that MuJoCo omits during attachment.

        Flex vertices owned by the world body require the model-root transform.
        Body-owned vertices remain in body-local coordinates and move with the
        attached frame.
        """

        def namespaced_body(name: str) -> str:
            if not name or name == "world":
                return name
            return f"{prefix}{name}"

        def transformed_points(values, owners) -> list[float]:
            points = np.asarray(values, np.float64).reshape(-1, 3).copy()
            if not len(points):
                return []
            if owners:
                world_owned = np.asarray(
                    [not name or name == "world" for name in owners],
                    dtype=bool,
                )
            else:
                world_owned = np.ones(len(points), dtype=bool)
            points[world_owned] = points[world_owned] @ np.asarray(rotation).T + position
            return points.reshape(-1).tolist()

        for flex in child.flexes:
            references_world = (
                any(name == "world" for name in flex.nodebody)
                or any(name == "world" for name in flex.vertbody)
                or (len(flex.node) > 0 and not flex.nodebody)
                or (len(flex.vert) > 0 and not flex.vertbody)
            )
            if not references_world:
                continue
            fields = {
                name: getattr(flex, name).tolist()
                if hasattr(getattr(flex, name), "tolist")
                else getattr(flex, name)
                for name in _FLEX_COPY_FIELDS
            }
            fields.update(
                name=f"{prefix}{flex.name}" if flex.name else None,
                material=f"{prefix}{flex.material}" if flex.material else None,
                nodebody=[namespaced_body(name) for name in flex.nodebody],
                vertbody=[namespaced_body(name) for name in flex.vertbody],
                node=transformed_points(flex.node, flex.nodebody),
                vert=transformed_points(flex.vert, flex.vertbody),
            )
            spec.add_flex(**fields)

    @staticmethod
    def _joint_state_key(model, joint: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name:
            return name
        body = int(model.jnt_bodyid[joint])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body)
        ordinal = joint - int(model.body_jntadr[body])
        return f"{body_name}:joint:{ordinal}"

    @staticmethod
    def _actuator_state_key(model, actuator: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        return name or f"actuator:{actuator}"

    @staticmethod
    def _span(addresses, index: int, total: int) -> slice:
        start = int(addresses[index])
        stop = int(addresses[index + 1]) if index + 1 < len(addresses) else int(total)
        return slice(start, stop)

    def _capture_named_model_state(self) -> _NamedModelState:
        model, data = self._m, self._d
        joints = {}
        for joint in range(model.njnt):
            qpos = self._span(model.jnt_qposadr, joint, model.nq)
            qvel = self._span(model.jnt_dofadr, joint, model.nv)
            joints[self._joint_state_key(model, joint)] = (
                np.asarray(data.qpos[qpos]).copy(),
                np.asarray(data.qvel[qvel]).copy(),
            )
        actuators = {}
        for actuator in range(model.nactuator):
            ctrl_start = int(model.actuator_ctrladr[actuator])
            ctrl_stop = ctrl_start + int(model.actuator_ctrlnum[actuator])
            act_start = int(model.actuator_actadr[actuator])
            act_stop = act_start + int(model.actuator_actnum[actuator])
            activation = (
                np.asarray(data.act[act_start:act_stop]).copy()
                if act_start >= 0
                else np.zeros(0, np.float64)
            )
            actuators[self._actuator_state_key(model, actuator)] = (
                np.asarray(data.ctrl[ctrl_start:ctrl_stop]).copy(),
                activation,
            )
        mocap = {}
        for body in range(1, model.nbody):
            mocap_id = int(model.body_mocapid[body])
            if mocap_id < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body:{body}"
            mocap[name] = (data.mocap_pos[mocap_id].copy(), data.mocap_quat[mocap_id].copy())
        equality = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
            or f"equality:{index}": bool(data.eq_active[index])
            for index in range(model.neq)
        }
        return _NamedModelState(joints, actuators, mocap, equality, float(data.time))

    def _transform_named_model_state(
        self,
        state: _NamedModelState,
        prefix: str,
        previous_position: np.ndarray,
        previous_rotation: np.ndarray,
        next_position: np.ndarray,
        next_rotation: np.ndarray,
    ) -> _NamedModelState:
        """Move world-space free-joint and mocap state with an attached model root."""
        delta_rotation = next_rotation @ previous_rotation.T
        joints = dict(state.joints)
        for joint in range(self._m.njnt):
            if int(self._m.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            body = int(self._m.jnt_bodyid[joint])
            body_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if not body_name.startswith(prefix):
                continue
            key = self._joint_state_key(self._m, joint)
            values = joints.get(key)
            if values is None or values[0].shape != (7,) or values[1].shape != (6,):
                continue
            qpos, qvel = values[0].copy(), values[1].copy()
            qpos[:3] = next_position + delta_rotation @ (qpos[:3] - previous_position)
            qpos[3:7] = math3d.mat3_to_quat(delta_rotation @ math3d.quat_to_mat3(qpos[3:7]))
            # Free-joint linear velocity is world-space; angular velocity is in
            # the local body frame and therefore remains unchanged.
            qvel[:3] = delta_rotation @ qvel[:3]
            joints[key] = (qpos, qvel)

        mocap = dict(state.mocap)
        for name, values in state.mocap.items():
            if not name.startswith(prefix):
                continue
            position, quaternion = values[0].copy(), values[1].copy()
            position[:] = next_position + delta_rotation @ (position - previous_position)
            quaternion[:] = math3d.mat3_to_quat(delta_rotation @ math3d.quat_to_mat3(quaternion))
            mocap[name] = (position, quaternion)
        return replace(state, joints=joints, mocap=mocap)

    def _restore_named_model_state(self, state: _NamedModelState) -> None:
        model, data = self._m, self._d
        for joint in range(model.njnt):
            values = state.joints.get(self._joint_state_key(model, joint))
            if values is None:
                continue
            qpos = self._span(model.jnt_qposadr, joint, model.nq)
            qvel = self._span(model.jnt_dofadr, joint, model.nv)
            if data.qpos[qpos].shape == values[0].shape:
                data.qpos[qpos] = values[0]
            if data.qvel[qvel].shape == values[1].shape:
                data.qvel[qvel] = values[1]
        for actuator in range(model.nactuator):
            values = state.actuators.get(self._actuator_state_key(model, actuator))
            if values is None:
                continue
            ctrl_start = int(model.actuator_ctrladr[actuator])
            ctrl_stop = ctrl_start + int(model.actuator_ctrlnum[actuator])
            act_start = int(model.actuator_actadr[actuator])
            act_stop = act_start + int(model.actuator_actnum[actuator])
            if data.ctrl[ctrl_start:ctrl_stop].shape == values[0].shape:
                data.ctrl[ctrl_start:ctrl_stop] = values[0]
            if act_start >= 0 and data.act[act_start:act_stop].shape == values[1].shape:
                data.act[act_start:act_stop] = values[1]
        for body in range(1, model.nbody):
            mocap_id = int(model.body_mocapid[body])
            if mocap_id < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body:{body}"
            values = state.mocap.get(name)
            if values is not None:
                data.mocap_pos[mocap_id], data.mocap_quat[mocap_id] = values
        for index in range(model.neq):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index) or f"equality:{index}"
            )
            if name in state.equality:
                data.eq_active[index] = state.equality[name]
        data.time = state.time
        mujoco.mj_forward(model, data)
