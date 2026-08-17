from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .adapters.base import (
    CAMERA_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    CameraInfo,
    NodeKind,
    SceneFrame,
    SceneNode,
    SceneSource,
)
from .types import (
    DEFAULT_HEADLIGHT,
    DEFAULT_MATERIAL,
    CameraView,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    TextureData,
)


@dataclass(frozen=True)
class SceneObject:
    scene: Scene = field(repr=False, compare=False)
    object_id: int

    def set_pose(self, position, rotation=None) -> None:
        self.scene.set_pose(self.object_id, position, rotation)

    def remove(self) -> None:
        self.scene.remove(self.object_id)


@dataclass
class _Item:
    object_id: int
    name: str
    mesh: MeshKey
    size: np.ndarray
    color: np.ndarray
    material: Material
    position: np.ndarray
    rotation: np.ndarray
    initial_position: np.ndarray
    initial_rotation: np.ndarray


@dataclass
class _CameraItem:
    camera_id: int
    object_id: int
    name: str
    view: CameraView


class Scene:
    def __init__(self, *, camera: CameraView | None = None, lights: LightSet | None = None) -> None:
        self.camera = camera
        self.lights = lights or LightSet(headlight=DEFAULT_HEADLIGHT)
        self.textures: dict[str, TextureData] = {}
        self._items: list[_Item] = []
        self._meshes: dict[MeshKey, MeshData] = {}
        self._next_object_id = 1
        self._next_mesh_id = 0
        self._next_camera_id = 0
        self._cameras: list[_CameraItem] = []
        self._revision = 0
        if camera is not None:
            self.add_camera("camera", camera)
        self._built_revision = -1
        self._source = SceneSource()
        self._frame = SceneFrame()
        self._oid_to_index: dict[int, int] = {}
        self._node_to_oid: dict[int, int] = {}

    @property
    def structure_revision(self) -> int:
        return self._revision

    @property
    def source(self) -> SceneSource:
        self._rebuild()
        return self._source

    @property
    def frame(self) -> SceneFrame:
        self._rebuild()
        return self._frame

    def add(
        self,
        shape: MeshShape | MeshKey,
        *,
        name: str = "object",
        size=(0.5, 0.5, 0.5),
        position=(0.0, 0.0, 0.0),
        rotation=None,
        color=(0.65, 0.68, 0.72, 1.0),
        material: Material = DEFAULT_MATERIAL,
    ) -> SceneObject:
        key = shape if isinstance(shape, MeshKey) else MeshKey(shape)
        oid = self._next_object_id
        self._next_object_id += 1
        pos = _vec3(position)
        rot = _mat3(rotation)
        self._items.append(
            _Item(
                object_id=oid,
                name=name,
                mesh=key,
                size=_vec3(size),
                color=np.asarray(color, np.float32).reshape(4).copy(),
                material=material,
                position=pos,
                rotation=rot,
                initial_position=pos.copy(),
                initial_rotation=rot.copy(),
            )
        )
        self._revision += 1
        return SceneObject(self, oid)

    def box(self, **kwargs) -> SceneObject:
        return self.add(MeshShape.BOX, **kwargs)

    def sphere(self, **kwargs) -> SceneObject:
        return self.add(MeshShape.SPHERE, **kwargs)

    def cylinder(self, **kwargs) -> SceneObject:
        return self.add(MeshShape.CYLINDER, **kwargs)

    def plane(self, **kwargs) -> SceneObject:
        return self.add(MeshShape.PLANE, **kwargs)

    def mesh(self, data: MeshData, **kwargs) -> SceneObject:
        key = MeshKey(MeshShape.ASSET, self._next_mesh_id)
        self._next_mesh_id += 1
        self._meshes[key] = data
        return self.add(key, **kwargs)

    def add_texture(self, texture: TextureData) -> None:
        self.textures[texture.name] = texture
        self._revision += 1

    def add_camera(self, name: str, view: CameraView) -> int:
        camera_id = self._next_camera_id
        self._next_camera_id += 1
        self._cameras.append(
            _CameraItem(
                camera_id=camera_id,
                object_id=CAMERA_OBJECT_BASE + camera_id,
                name=str(name),
                view=view,
            )
        )
        if self.camera is None:
            self.camera = view
        self._revision += 1
        return camera_id

    def set_camera(self, camera_id: int, view: CameraView) -> bool:
        item = next((item for item in self._cameras if item.camera_id == int(camera_id)), None)
        if item is None:
            return False
        item.view = view
        if item.camera_id == 0:
            self.camera = view
        if self._built_revision == self._revision:
            self._source.cameras = tuple(camera.view for camera in self._cameras)
            self._frame.cameras = self._source.cameras
        return True

    def camera_view(self, camera_id: int) -> CameraView | None:
        item = next((item for item in self._cameras if item.camera_id == int(camera_id)), None)
        return item.view if item is not None else None

    def camera_infos(self) -> list[CameraInfo]:
        return [CameraInfo(item.camera_id, item.name, item.object_id) for item in self._cameras]

    def object(self, name: str) -> SceneObject:
        item = next((x for x in self._items if x.name == name), None)
        if item is None:
            raise KeyError(f"Unknown object name {name!r}")
        return SceneObject(self, item.object_id)

    def set_pose(self, object_id: int, position, rotation=None) -> None:
        item = self._item(object_id)
        item.position[:] = _vec3(position)
        if rotation is not None:
            item.rotation[:] = _mat3(rotation)
        if self._built_revision == self._revision:
            self._write_pose(self._oid_to_index[object_id], item)

    def set_pose_by_node(self, node_id: int, position, rotation) -> bool:
        oid = self._node_to_oid.get(int(node_id))
        if oid is None:
            return False
        self.set_pose(oid, position, rotation)
        return True

    def set_light(self, light_id: int, light) -> bool:
        i = int(light_id)
        if not 0 <= i < len(self.lights.lights):
            return False
        lights = list(self.lights.lights)
        lights[i] = light
        self.lights = replace(self.lights, lights=tuple(lights))
        if self._built_revision == self._revision:
            self._source.lights = self.lights
        return True

    def remove(self, object_id: int) -> None:
        before = len(self._items)
        self._items = [x for x in self._items if x.object_id != int(object_id)]
        if len(self._items) == before:
            raise KeyError(f"Unknown object_id={object_id}")
        self._revision += 1

    def reset_poses(self) -> None:
        for i, item in enumerate(self._items):
            item.position[:] = item.initial_position
            item.rotation[:] = item.initial_rotation
            if self._built_revision == self._revision:
                self._write_pose(i, item)

    def _item(self, object_id: int) -> _Item:
        for item in self._items:
            if item.object_id == int(object_id):
                return item
        raise KeyError(f"Unknown object_id={object_id}")

    def _rebuild(self) -> None:
        if self._built_revision == self._revision:
            return
        n = len(self._items)
        materials: list[Material] = []
        material_ids: dict[int, int] = {}
        nodes = [SceneNode(0, "world", NodeKind.WORLD, body_index=0)]
        geom_material: list[int] = []
        self._oid_to_index = {}
        self._node_to_oid = {}

        for i, item in enumerate(self._items):
            token = id(item.material)
            if token not in material_ids:
                material_ids[token] = len(materials)
                materials.append(item.material)
            geom_material.append(material_ids[token])
            body_index = i + 1
            link_id = len(nodes)
            geom_id = link_id + 1
            nodes.append(
                SceneNode(
                    link_id,
                    item.name,
                    NodeKind.LINK,
                    parent=0,
                    children=[geom_id],
                    object_id=item.object_id,
                    posable=True,
                    body_index=body_index,
                )
            )
            nodes.append(
                SceneNode(
                    geom_id,
                    f"{item.name}.geom",
                    NodeKind.GEOM,
                    parent=link_id,
                    body_index=body_index,
                )
            )
            nodes[0].children.append(link_id)
            self._oid_to_index[item.object_id] = i
            self._node_to_oid[link_id] = item.object_id

        for i, light in enumerate(self.lights.lights):
            node_id = len(nodes)
            nodes.append(
                SceneNode(
                    node_id,
                    f"{light.kind.name.lower()} light {i}",
                    NodeKind.LIGHT,
                    parent=0,
                    object_id=LIGHT_OBJECT_BASE + i,
                    visible=light.active,
                    body_index=0,
                    light_index=i,
                )
            )
            nodes[0].children.append(node_id)

        for camera_index, camera in enumerate(self._cameras):
            node_id = len(nodes)
            nodes.append(
                SceneNode(
                    node_id,
                    camera.name,
                    NodeKind.CAMERA,
                    parent=0,
                    object_id=camera.object_id,
                    body_index=0,
                    camera_index=camera_index,
                )
            )
            nodes[0].children.append(node_id)

        positions = np.zeros((n, 3), np.float32)
        rotations = np.zeros((n, 3, 3), np.float32)
        body_positions = np.zeros((n + 1, 3), np.float32)
        body_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], n + 1, axis=0)
        self._frame = SceneFrame(
            geom_xpos=positions,
            geom_xmat=rotations,
            body_xpos=body_positions,
            body_xmat=body_rotations,
            cameras=tuple(camera.view for camera in self._cameras),
        )

        for i, item in enumerate(self._items):
            self._write_pose(i, item)

        if n:
            radii = np.array([float(np.max(x.size)) for x in self._items], np.float32)
            lo = positions - radii[:, None]
            hi = positions + radii[:, None]
            center = ((lo.min(axis=0) + hi.max(axis=0)) * 0.5).astype(np.float32)
            extent = max(float(np.linalg.norm(hi.max(axis=0) - lo.min(axis=0)) * 0.5), 0.5)
        else:
            center = np.zeros(3, np.float32)
            extent = 1.0

        self._source = SceneSource(
            meshes=dict(self._meshes),
            textures=dict(self.textures),
            materials=materials or [DEFAULT_MATERIAL],
            geom_mesh=[x.mesh for x in self._items],
            geom_material=geom_material,
            geom_size=np.array([x.size for x in self._items], np.float32).reshape(n, 3),
            geom_rgba=np.array([x.color for x in self._items], np.float32).reshape(n, 4),
            geom_object_id=np.array([x.object_id for x in self._items], np.uint32),
            geom_body=np.arange(1, n + 1, dtype=np.int32),
            geom_source=np.arange(n, dtype=np.int32),
            geom_pose_source=np.zeros(n, np.uint8),
            geom_node=np.arange(2, 2 * n + 1, 2, dtype=np.int32),
            geom_local=np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0),
            geom_infinite_plane=np.zeros(n, bool),
            lights=self.lights,
            cameras=tuple(camera.view for camera in self._cameras),
            scene_extent=extent,
            scene_center=center,
            nodes=nodes,
        )
        self._built_revision = self._revision

    def _write_pose(self, index: int, item: _Item) -> None:
        assert self._frame.geom_xpos is not None and self._frame.geom_xmat is not None
        assert self._frame.body_xpos is not None and self._frame.body_xmat is not None
        self._frame.geom_xpos[index] = item.position
        self._frame.geom_xmat[index] = item.rotation
        self._frame.body_xpos[index + 1] = item.position
        self._frame.body_xmat[index + 1] = item.rotation


def _vec3(value) -> np.ndarray:
    return np.asarray(value, np.float32).reshape(3).copy()


def _mat3(value) -> np.ndarray:
    return (
        np.eye(3, dtype=np.float32)
        if value is None
        else np.asarray(value, np.float32).reshape(3, 3).copy()
    )
