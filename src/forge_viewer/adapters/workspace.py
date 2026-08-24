"""Composition adapter for physics models and Forge-authored scene entities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..scene import Scene
from ..types import InstancePoseSource, LightSet, Material, MeshKey, MeshShape
from .base import (
    CAMERA_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    CameraInfo,
    FrameNeeds,
    SceneAdapterBase,
    SceneFrame,
    SceneModelInfo,
    SceneNode,
    SceneSource,
)

_AUTHORED_OBJECT_BASE = 0x60000000
_AUTHORED_LIGHT_BASE = 0x74000000
_AUTHORED_CAMERA_BASE = 0x75000000


class WorkspaceAdapter(SceneAdapterBase):
    """Combines a simulation adapter with backend-neutral authored entities."""

    def __init__(self, primary: SceneAdapterBase, scene: Scene | None = None) -> None:
        self.primary = primary
        if scene is None:
            environment = primary.scene_source().lights.environment()
            scene = Scene(lights=LightSet().with_environment(environment))
        self.scene = scene
        self._path: Path | None = None
        self._resource_roots: tuple[Path, ...] = ()
        self._source: SceneSource | None = None
        self._primary_revision = -1
        self._scene_revision = -1
        self._node_to_scene: dict[int, int] = {}
        self._object_to_scene: dict[int, int] = {}
        self._light_to_scene: dict[int, int] = {}
        self._camera_to_scene: dict[int, int] = {}
        self.caps = replace(
            primary.caps,
            name=f"workspace:{primary.caps.name}",
            scene_authoring=True,
            scene_files=True,
            edit_history=True,
            model_composition=primary.caps.model_composition,
        )

    @property
    def structure_revision(self) -> int:
        return self.primary.structure_revision * 1_000_003 + self.scene.structure_revision

    def new_scene(self) -> None:
        self.primary.new_scene()
        environment = self.primary.scene_source().lights.environment()
        self.scene = Scene(lights=LightSet().with_environment(environment))
        self._path = None
        self._resource_roots = ()
        self._invalidate()

    @property
    def resource_roots(self) -> tuple[Path, ...]:
        return self._resource_roots

    def add_resource_root(self, path: Path) -> bool:
        root = Path(path).expanduser().resolve()
        if root in self._resource_roots:
            return True
        self._resource_roots = (*self._resource_roots, root)
        return True

    def remove_resource_root(self, path: Path) -> bool:
        root = Path(path).expanduser().resolve()
        if root not in self._resource_roots:
            return False
        self._resource_roots = tuple(item for item in self._resource_roots if item != root)
        return True

    def set_resource_roots(self, paths: tuple[Path, ...]) -> None:
        self._resource_roots = tuple(dict.fromkeys(Path(path).resolve() for path in paths))

    def open_scene(self, path: Path) -> None:
        from ..workspace_io import load_workspace

        load_workspace(self, path)
        self._path = Path(path).expanduser().resolve()
        self._invalidate()

    def save_scene(self, path: Path) -> None:
        from ..workspace_io import save_workspace

        self._path = save_workspace(self, path)

    def capture_edit_state(self) -> object:
        return self.scene.clone(), self.primary.capture_edit_state(), self._resource_roots

    def restore_edit_state(self, state: object) -> bool:
        if not isinstance(state, tuple) or len(state) != 3 or not isinstance(state[0], Scene):
            return False
        restored_primary = state[1] is None or self.primary.restore_edit_state(state[1])
        if not restored_primary:
            return False
        self.scene = state[0].clone()
        self.set_resource_roots(state[2])
        self._invalidate()
        return True

    def reload(self) -> None:
        if self._path is not None:
            self.open_scene(self._path)
        else:
            self.primary.reload()

    def load(self, path: Path) -> None:
        self.primary.new_scene()
        model_id = self.primary.add_scene_model(
            Path(path), np.zeros(3, np.float32), np.eye(3, dtype=np.float32)
        )
        if model_id < 0:
            raise RuntimeError(f"Failed to load {path}")
        self._invalidate()

    def scene_models(self) -> tuple[SceneModelInfo, ...]:
        return self.primary.scene_models()

    def add_scene_model(self, path: Path, position, rotation) -> int:
        model_id = self.primary.add_scene_model(path, position, rotation)
        if model_id >= 0:
            self._invalidate()
        return model_id

    def remove_scene_model(self, model_id: int) -> bool:
        changed = self.primary.remove_scene_model(model_id)
        if changed:
            self._invalidate()
        return changed

    def set_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        changed = self.primary.set_scene_model_transform(model_id, position, rotation)
        if changed:
            self._invalidate()
        return changed

    def add_model_element(self, parent_node_id: int, kind: str, name: str) -> int:
        node_id = self.primary.add_model_element(parent_node_id, kind, name)
        if node_id >= 0:
            self._invalidate()
        return node_id

    def remove_model_element(self, node_id: int) -> bool:
        changed = self.primary.remove_model_element(node_id)
        if changed:
            self._invalidate()
        return changed

    def rename_model_element(self, node_id: int, name: str) -> bool:
        changed = self.primary.rename_model_element(node_id, name)
        if changed:
            self._invalidate()
        return changed

    def scene_model_xml(self, model_id: int) -> str | None:
        return self.primary.scene_model_xml(model_id)

    def scene_model_source(self, model_id: int) -> str | None:
        return self.primary.scene_model_source(model_id)

    def set_scene_model_xml(self, model_id: int, xml: str) -> bool:
        changed = self.primary.set_scene_model_xml(model_id, xml)
        if changed:
            self._invalidate()
        return changed

    def scene_source(self) -> SceneSource:
        if (
            self._source is None
            or self._primary_revision != self.primary.structure_revision
            or self._scene_revision != self.scene.structure_revision
        ):
            self._source = self._merge_source(self.primary.scene_source(), self.scene.source)
            self._primary_revision = self.primary.structure_revision
            self._scene_revision = self.scene.structure_revision
        return self._source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        source = self.scene_source()
        primary = self.primary.frame(needs)
        authored = self.scene.frame
        frame = replace(primary)
        if needs.poses:
            frame.geom_xpos = _rows(primary.geom_xpos, authored.geom_xpos)
            frame.geom_xmat = _rows(primary.geom_xmat, authored.geom_xmat)
            frame.body_xpos = _body_rows(primary.body_xpos, authored.body_xpos)
            frame.body_xmat = _body_rows(primary.body_xmat, authored.body_xmat)
        if primary.lights is not None:
            frame.lights = replace(
                source.lights,
                lights=(*primary.lights.lights, *self.scene.lights.lights),
            )
        frame.cameras = (
            *(primary.cameras or self.primary.scene_source().cameras),
            *self.scene.source.cameras,
        )
        return frame

    def nodes(self) -> list[SceneNode]:
        return self.scene_source().nodes

    def cameras(self) -> list[CameraInfo]:
        primary = self.primary.cameras()
        return [
            *primary,
            *(
                CameraInfo(
                    len(primary) + index,
                    item.name,
                    _AUTHORED_CAMERA_BASE + item.camera_id,
                )
                for index, item in enumerate(self.scene._cameras)
            ),
        ]

    def camera_view(self, camera_id: int):
        primary_count = len(self.primary.cameras())
        if int(camera_id) < primary_count:
            return self.primary.camera_view(camera_id)
        index = int(camera_id) - primary_count
        if not 0 <= index < len(self.scene._cameras):
            return None
        return self.scene.camera_view(self.scene._cameras[index].camera_id)

    def set_camera_view(self, camera_id: int, camera) -> bool:
        primary_count = len(self.primary.cameras())
        if int(camera_id) < primary_count:
            return self.primary.set_camera_view(camera_id, camera)
        index = int(camera_id) - primary_count
        if not 0 <= index < len(self.scene._cameras):
            return False
        return self.scene.set_camera(self.scene._cameras[index].camera_id, camera)

    def set_pose(self, node_id: int, position, rotation) -> bool:
        scene_node = self._node_to_scene.get(int(node_id))
        if scene_node is not None:
            return self.scene.set_pose_by_node(scene_node, position, rotation)
        return self.primary.set_pose(node_id, position, rotation)

    def set_light(self, light_id: int, light) -> bool:
        primary_count = len(self.primary.scene_source().lights.lights)
        if int(light_id) < primary_count:
            return self.primary.set_light(light_id, light)
        return self.scene.set_light(int(light_id) - primary_count, light)

    def set_environment(self, environment) -> bool:
        return self.scene.set_environment(environment)

    def set_material(self, material_id: int, material: Material) -> bool:
        offset = len(self.primary.scene_source().materials)
        if int(material_id) < offset:
            return self.primary.set_material(material_id, material)
        return self.scene.set_material(int(material_id) - offset, material)

    def set_geometry_color(self, node_id: int, rgba) -> bool:
        scene_node = self._node_to_scene.get(int(node_id))
        if scene_node is not None:
            return self.scene.set_geometry_color(scene_node, rgba)
        return self.primary.set_geometry_color(node_id, rgba)

    def add_scene_object(self, shape, name, size, position, rotation, color, material) -> int:
        raw = self.scene.add(
            shape,
            name=name,
            size=size,
            position=position,
            rotation=rotation,
            color=color,
            material=material,
        ).object_id
        return _AUTHORED_OBJECT_BASE + raw

    def remove_scene_object(self, object_id: int) -> bool:
        raw = self._object_to_scene.get(int(object_id), int(object_id))
        try:
            self.scene.remove(raw)
        except KeyError:
            return False
        return True

    def add_scene_light(self, name: str, light) -> int:
        return (
            len(self.primary.scene_source().lights.lights)
            + self.scene.add_light(name, light).light_id
        )

    def remove_scene_light(self, light_id: int) -> bool:
        primary_count = len(self.primary.scene_source().lights.lights)
        index = int(light_id) - primary_count
        self.scene._sync_light_items()
        if not 0 <= index < len(self.scene._lights):
            return False
        try:
            self.scene.remove_light(self.scene._lights[index].light_id)
        except KeyError:
            return False
        return True

    def add_scene_camera(self, name: str, camera) -> int:
        return len(self.primary.cameras()) + self.scene.add_camera(name, camera)

    def remove_scene_camera(self, camera_id: int) -> bool:
        index = int(camera_id) - len(self.primary.cameras())
        if not 0 <= index < len(self.scene._cameras):
            return False
        try:
            self.scene.remove_camera(self.scene._cameras[index].camera_id)
        except KeyError:
            return False
        return True

    def duplicate_scene_entity(self, object_id: int) -> int:
        raw = self._object_to_scene.get(int(object_id))
        if raw is None:
            return 0
        try:
            duplicate = self.scene.duplicate_entity(raw)
            if duplicate >= CAMERA_OBJECT_BASE:
                return _AUTHORED_CAMERA_BASE + duplicate - CAMERA_OBJECT_BASE
            if duplicate >= LIGHT_OBJECT_BASE:
                return _AUTHORED_LIGHT_BASE + duplicate - LIGHT_OBJECT_BASE
            return _AUTHORED_OBJECT_BASE + duplicate
        except KeyError:
            return 0

    def remove_scene_entity(self, object_id: int) -> bool:
        raw = self._object_to_scene.get(int(object_id))
        if raw is None:
            return False
        try:
            self.scene.remove_entity(raw)
        except KeyError:
            return False
        return True

    def rename_scene_entity(self, object_id: int, name: str) -> bool:
        raw = self._object_to_scene.get(int(object_id))
        if raw is None:
            return False
        try:
            self.scene.rename_entity(raw, name)
        except (KeyError, ValueError):
            return False
        return True

    def reset(self) -> None:
        self.primary.reset()
        self.scene.reset_poses()

    def step(self, count: int = 1) -> None:
        self.primary.step(count)

    def set_paused(self, paused: bool) -> bool:
        return self.primary.set_paused(paused)

    def timestep(self) -> float:
        return self.primary.timestep()

    def joints(self):
        return self.primary.joints()

    def actuators(self):
        return self.primary.actuators()

    def keyframes(self):
        return self.primary.keyframes()

    def sensors(self):
        return self.primary.sensors()

    def equality_constraints(self):
        return self.primary.equality_constraints()

    def load_keyframe(self, keyframe_id: int) -> bool:
        return self.primary.load_keyframe(keyframe_id)

    def visual_groups(self):
        return self.primary.visual_groups()

    def set_visual_group(self, category: str, group: int, visible: bool) -> bool:
        changed = self.primary.set_visual_group(category, group, visible)
        if changed:
            self._invalidate()
        return changed

    def set_qpos(self, index: int, value: float) -> bool:
        return self.primary.set_qpos(index, value)

    def set_equality_enabled(self, constraint_id: int, enabled: bool) -> bool:
        return self.primary.set_equality_enabled(constraint_id, enabled)

    def set_ctrl(self, index: int, value: float) -> bool:
        return self.primary.set_ctrl(index, value)

    def solve_ik(self, node_id: int, target_position, target_rotation, options):
        return self.primary.solve_ik(node_id, target_position, target_rotation, options)

    def capture_state(self):
        return self.primary.capture_state()

    def restore_state(self, state) -> bool:
        return self.primary.restore_state(state)

    def apply_perturb(self, node_id: int, target_position, target_rotation, mode: str) -> bool:
        return self.primary.apply_perturb(node_id, target_position, target_rotation, mode)

    def clear_perturb(self) -> None:
        self.primary.clear_perturb()

    def raycast(self, origin, direction):
        return self.primary.raycast(origin, direction)

    def camera_hint(self):
        return self.scene.camera or self.primary.camera_hint()

    def release(self) -> None:
        self.primary.release()

    def _invalidate(self) -> None:
        self._source = None

    def _merge_source(self, primary: SceneSource, authored: SceneSource) -> SceneSource:
        body_offset = len(primary.body_names) or _body_count(primary.nodes)
        node_offset = len(primary.nodes) - 1
        geom_instances = primary.geom_pose_source == int(InstancePoseSource.GEOM)
        geom_offset = (
            int(np.max(primary.geom_source[geom_instances])) + 1 if np.any(geom_instances) else 0
        )
        material_offset = len(primary.materials)

        mesh_map, meshes = _merge_meshes(primary.meshes, authored.meshes)
        texture_map, textures = _merge_textures(primary.textures, authored.textures)
        materials = [
            *primary.materials,
            *(
                replace(item, texture=texture_map.get(item.texture, item.texture))
                for item in authored.materials
            ),
        ]

        self._node_to_scene.clear()
        self._object_to_scene.clear()
        self._light_to_scene.clear()
        self._camera_to_scene.clear()
        nodes = [replace(node, children=list(node.children)) for node in primary.nodes]
        authored_nodes: dict[int, int] = {}
        for raw in authored.nodes[1:]:
            node_id = node_offset + raw.node_id
            parent = 0 if raw.parent == 0 else node_offset + raw.parent
            object_id = self._map_object(raw)
            body = 0 if raw.body_index == 0 else body_offset + raw.body_index - 1
            light = len(primary.lights.lights) + raw.light_index if raw.light_index >= 0 else -1
            camera = len(primary.cameras) + raw.camera_index if raw.camera_index >= 0 else -1
            node = replace(
                raw,
                node_id=node_id,
                parent=parent,
                children=[node_offset + child for child in raw.children],
                object_id=object_id,
                body_index=body,
                light_index=light,
                camera_index=camera,
            )
            nodes.append(node)
            nodes[parent].children.append(node_id)
            authored_nodes[raw.node_id] = node_id
            self._node_to_scene[node_id] = raw.node_id
            if object_id:
                self._object_to_scene[object_id] = raw.object_id

        geom_object = np.array(
            [
                _AUTHORED_OBJECT_BASE + int(value) if value else 0
                for value in authored.geom_object_id
            ],
            np.uint32,
        )
        for mapped, raw in zip(geom_object, authored.geom_object_id, strict=True):
            if mapped:
                self._object_to_scene[int(mapped)] = int(raw)

        source = replace(
            primary,
            meshes=meshes,
            textures=textures,
            materials=materials,
            geom_mesh=[
                *primary.geom_mesh,
                *(_mesh_key(key, mesh_map) for key in authored.geom_mesh),
            ],
            geom_convex_mesh=[
                *primary.geom_convex_mesh,
                *(_mesh_key(key, mesh_map) for key in authored.geom_convex_mesh),
            ],
            geom_material=[
                *primary.geom_material,
                *(material_offset + int(i) for i in authored.geom_material),
            ],
            geom_size=_rows(primary.geom_size, authored.geom_size),
            geom_rgba=_rows(primary.geom_rgba, authored.geom_rgba),
            geom_object_id=np.concatenate((primary.geom_object_id, geom_object)),
            geom_body=np.concatenate(
                (primary.geom_body, body_offset + authored.geom_body.astype(np.int32) - 1)
            ),
            geom_source=np.concatenate(
                (primary.geom_source, geom_offset + authored.geom_source.astype(np.int32))
            ),
            geom_pose_source=np.concatenate((primary.geom_pose_source, authored.geom_pose_source)),
            geom_visual=np.concatenate((primary.geom_visual, authored.geom_visual)),
            geom_static=np.concatenate((primary.geom_static, authored.geom_static)),
            instance_island_body=np.concatenate(
                (primary.instance_island_body, np.full(authored.instance_count, -1, np.int32))
            ),
            geom_node=np.concatenate(
                (
                    primary.geom_node,
                    np.asarray([authored_nodes[int(i)] for i in authored.geom_node], np.int32),
                )
            ),
            geom_local=_rows(primary.geom_local, authored.geom_local),
            geom_infinite_plane=np.concatenate(
                (primary.geom_infinite_plane, authored.geom_infinite_plane)
            ),
            body_names=(
                *primary.body_names,
                *(
                    node.name
                    for node in authored.nodes
                    if node.body_index > 0 and node.kind.value == "link"
                ),
            ),
            lights=replace(
                authored.lights,
                lights=(*primary.lights.lights, *authored.lights.lights),
            ),
            cameras=(*primary.cameras, *authored.cameras),
            scene_extent=max(primary.scene_extent, authored.scene_extent),
            scene_center=(primary.scene_center + authored.scene_center) * 0.5,
            nodes=nodes,
        )
        return source

    def _map_object(self, node: SceneNode) -> int:
        if not node.object_id:
            return 0
        if CAMERA_OBJECT_BASE <= node.object_id < CAMERA_OBJECT_BASE + 0x01000000:
            value = _AUTHORED_CAMERA_BASE + node.object_id - CAMERA_OBJECT_BASE
            self._camera_to_scene[value] = node.object_id - CAMERA_OBJECT_BASE
            return value
        if node.object_id >= LIGHT_OBJECT_BASE:
            value = _AUTHORED_LIGHT_BASE + node.object_id - LIGHT_OBJECT_BASE
            self._light_to_scene[value] = node.object_id - LIGHT_OBJECT_BASE
            return value
        return _AUTHORED_OBJECT_BASE + node.object_id

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def _rows(first, second):
    if first is None:
        return None if second is None else np.asarray(second).copy()
    if second is None:
        return first
    return np.concatenate((first, second), axis=0)


def _body_rows(first, second):
    if second is None or len(second) <= 1:
        return first
    if first is None:
        return second
    return np.concatenate((first, second[1:]), axis=0)


def _body_count(nodes: list[SceneNode]) -> int:
    return max((node.body_index for node in nodes), default=0) + 1


def _merge_meshes(primary, authored):
    meshes = dict(primary)
    mapping: dict[MeshKey, MeshKey] = {}
    next_asset = max((key.index for key in meshes if key.shape is MeshShape.ASSET), default=-1) + 1
    for key, mesh in authored.items():
        target = key
        if target in meshes:
            target = MeshKey(MeshShape.ASSET, next_asset)
            next_asset += 1
        meshes[target] = mesh
        mapping[key] = target
    return mapping, meshes


def _merge_textures(primary, authored):
    textures = dict(primary)
    mapping: dict[str, str] = {}
    for name, texture in authored.items():
        target = name
        suffix = 2
        while target in textures:
            target = f"{name}.{suffix}"
            suffix += 1
        mapping[name] = target
        textures[target] = replace(texture, name=target)
    return mapping, textures


def _mesh_key(key: MeshKey, mapping: dict[MeshKey, MeshKey]) -> MeshKey:
    return mapping.get(key, key)
