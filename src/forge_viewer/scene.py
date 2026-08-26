"""Authored scene entities for backend-neutral workflows."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .adapters.base import (
    CAMERA_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    CameraInfo,
    NodeType,
    SceneFrame,
    SceneNode,
    SceneSource,
)
from .types import (
    DEFAULT_HEADLIGHT,
    DEFAULT_MATERIAL,
    CameraView,
    Environment,
    Light,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    TextureData,
    TextureType,
)


@dataclass(frozen=True)
class SceneObject:
    """Handle for editing one object owned by a :class:`Scene`."""

    scene: Scene = field(repr=False, compare=False)
    object_id: int

    def set_pose(self, position, rotation=None) -> None:
        """Set world position and optional 3x3 world rotation."""
        self.scene.set_pose(self.object_id, position, rotation)

    def set_material(self, material: Material) -> None:
        """Replace the object's material."""
        self.scene.set_object_material(self.object_id, material)

    def set_color(self, rgba) -> None:
        """Set the object's RGBA color."""
        self.scene.set_object_color(self.object_id, rgba)

    def set_size(self, size) -> None:
        """Set three positive shape dimensions."""
        self.scene.set_object_size(self.object_id, size)

    def remove(self) -> None:
        """Remove this object from its scene."""
        self.scene.remove(self.object_id)


@dataclass(frozen=True)
class SceneLight:
    """Handle for editing one light owned by a :class:`Scene`."""

    scene: Scene = field(repr=False, compare=False)
    light_id: int

    @property
    def value(self) -> Light:
        """Return the current immutable light value."""
        return self.scene.light_value(self.light_id)

    def set(self, light: Light) -> None:
        """Replace the current light value."""
        self.scene.set_light_by_id(self.light_id, light)

    def remove(self) -> None:
        """Remove this light from its scene."""
        self.scene.remove_light(self.light_id)


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


@dataclass
class _LightItem:
    light_id: int
    name: str


class Scene:
    """Mutable backend-neutral scene for programmatic geometry, cameras, and lights."""

    def __init__(self, *, camera: CameraView | None = None, lights: LightSet | None = None) -> None:
        self.camera = camera
        self.lights = lights or LightSet(headlight=DEFAULT_HEADLIGHT)
        self.textures: dict[str, TextureData] = {}
        self.skybox: str | None = None
        self._items: list[_Item] = []
        self._meshes: dict[MeshKey, MeshData] = {}
        self._next_object_id = 1
        self._next_mesh_id = 0
        self._next_camera_id = 0
        self._next_light_id = len(self.lights.lights)
        self._lights = [
            _LightItem(i, f"{light.type.name.lower()} light {i}")
            for i, light in enumerate(self.lights.lights)
        ]
        self._cameras: list[_CameraItem] = []
        self._revision = 0
        if camera is not None:
            self.add_camera("camera", camera)
        self._built_revision = -1
        self._source = SceneSource()
        self._frame = SceneFrame()
        self._oid_to_index: dict[int, int] = {}
        self._node_to_oid: dict[int, int] = {}
        self._geom_node_to_oid: dict[int, int] = {}

    @property
    def structure_revision(self) -> int:
        """Return the revision used by adapters to detect structural changes."""
        return self._revision

    def save(self, path: str | Path) -> Path:
        """Serialize the scene as a Forge JSON document and return its path."""
        from .scene_io import save_scene

        return save_scene(self, path)

    def clone(self) -> Scene:
        """Return a deep copy suitable for editor undo state."""
        return copy.deepcopy(self)

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        """Load a Forge JSON scene document."""
        from .scene_io import load_scene

        return load_scene(path)

    @property
    def source(self) -> SceneSource:
        """Return stable render structure for the current revision."""
        self._rebuild()
        return self._source

    @property
    def frame(self) -> SceneFrame:
        """Return current object, camera, and light state."""
        self._rebuild()
        return self._frame

    def add(
        self,
        shape: MeshShape | MeshKey,
        *,
        name: str = "object",
        size: ArrayLike = (0.5, 0.5, 0.5),
        position: ArrayLike = (0.0, 0.0, 0.0),
        rotation: ArrayLike | None = None,
        color: ArrayLike = (0.65, 0.68, 0.72, 1.0),
        material: Material = DEFAULT_MATERIAL,
    ) -> SceneObject:
        """Add one render object and return an editing handle.

        Args:
            shape: Built-in shape or registered mesh key.
            name: Hierarchy label.
            size: Three shape dimensions in world units.
            position: World-space XYZ position.
            rotation: Optional row-major 3x3 world rotation.
            color: Linear RGBA object multiplier.
            material: Immutable material parameters.
        """
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
        """Add a box using :meth:`add` keyword arguments."""
        return self.add(MeshShape.BOX, **kwargs)

    def sphere(self, **kwargs) -> SceneObject:
        """Add a sphere using :meth:`add` keyword arguments."""
        return self.add(MeshShape.SPHERE, **kwargs)

    def cylinder(self, **kwargs) -> SceneObject:
        """Add a cylinder using :meth:`add` keyword arguments."""
        return self.add(MeshShape.CYLINDER, **kwargs)

    def plane(self, **kwargs) -> SceneObject:
        """Add a plane using :meth:`add` keyword arguments."""
        return self.add(MeshShape.PLANE, **kwargs)

    def mesh(self, data: MeshData, **kwargs) -> SceneObject:
        """Register an indexed mesh and add one instance."""
        key = MeshKey(MeshShape.ASSET, self._next_mesh_id)
        self._next_mesh_id += 1
        self._meshes[key] = data
        return self.add(key, **kwargs)

    def add_texture(self, texture: TextureData) -> None:
        """Add or replace a named texture."""
        self.textures[texture.name] = texture
        self._revision += 1

    def set_skybox(self, texture: str | None) -> bool:
        """Select a cube texture as the environment skybox."""
        if texture is not None:
            item = self.textures.get(texture)
            if item is None or item.type not in (TextureType.CUBE, TextureType.SKYBOX):
                return False
        self.skybox = texture
        self._revision += 1
        return True

    def add_camera(self, name: str, view: CameraView) -> int:
        """Add a selectable scene camera and return its camera ID."""
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

    def remove_camera(self, camera_id: int) -> None:
        """Remove a camera by ID."""
        item = next((item for item in self._cameras if item.camera_id == int(camera_id)), None)
        if item is None:
            raise KeyError(f"Unknown camera_id={camera_id}")
        self._cameras.remove(item)
        self.camera = self._cameras[0].view if self._cameras else None
        self._revision += 1

    def set_camera(self, camera_id: int, view: CameraView) -> bool:
        """Replace a camera view and report whether the ID exists."""
        item = next((item for item in self._cameras if item.camera_id == int(camera_id)), None)
        if item is None:
            return False
        item.view = view
        if item is self._cameras[0]:
            self.camera = view
        if self._built_revision == self._revision:
            self._source.cameras = tuple(camera.view for camera in self._cameras)
            self._frame.cameras = self._source.cameras
        return True

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Return a camera view by ID."""
        item = next((item for item in self._cameras if item.camera_id == int(camera_id)), None)
        return item.view if item is not None else None

    def camera_infos(self) -> list[CameraInfo]:
        """Return stable IDs and names for every scene camera."""
        return [CameraInfo(item.camera_id, item.name, item.object_id) for item in self._cameras]

    def add_light(self, name: str, light: Light) -> SceneLight:
        """Add a selectable light and return an editing handle."""
        self._sync_light_items()
        light_id = self._next_light_id
        self._next_light_id += 1
        self._lights.append(_LightItem(light_id, str(name)))
        self.lights = replace(self.lights, lights=(*self.lights.lights, light))
        self._revision += 1
        return SceneLight(self, light_id)

    def light(self, name: str) -> SceneLight:
        """Return a light handle by name."""
        self._sync_light_items()
        item = next((item for item in self._lights if item.name == name), None)
        if item is None:
            raise KeyError(f"Unknown light name {name!r}")
        return SceneLight(self, item.light_id)

    def light_value(self, light_id: int) -> Light:
        """Return an immutable light value by stable ID."""
        index = self._light_index(light_id)
        return self.lights.lights[index]

    def set_light_by_id(self, light_id: int, light: Light) -> None:
        """Replace a light by stable ID."""
        index = self._light_index(light_id)
        if not self.set_light_at(index, light):
            raise KeyError(f"Unknown light_id={light_id}")

    def remove_light(self, light_id: int) -> None:
        """Remove a light by stable ID."""
        index = self._light_index(light_id)
        lights = list(self.lights.lights)
        del lights[index]
        del self._lights[index]
        self.lights = replace(self.lights, lights=tuple(lights))
        self._revision += 1

    def object(self, name: str) -> SceneObject:
        """Return an object handle by name."""
        item = next((x for x in self._items if x.name == name), None)
        if item is None:
            raise KeyError(f"Unknown object name {name!r}")
        return SceneObject(self, item.object_id)

    def set_pose(self, object_id: int, position, rotation=None) -> None:
        """Set an object's world pose by object ID."""
        item = self._item(object_id)
        item.position[:] = _vec3(position)
        if rotation is not None:
            item.rotation[:] = _mat3(rotation)
        if self._built_revision == self._revision:
            self._write_pose(self._oid_to_index[object_id], item)

    def set_pose_by_node(self, node_id: int, position, rotation) -> bool:
        """Set an object's world pose through its hierarchy node ID."""
        oid = self._node_to_oid.get(int(node_id))
        if oid is None:
            return False
        self.set_pose(oid, position, rotation)
        return True

    def set_object_material(self, object_id: int, material: Material) -> None:
        """Replace an object's material by object ID."""
        self._item(object_id).material = material
        self._revision += 1

    def set_object_color(self, object_id: int, rgba) -> None:
        """Set an object's RGBA color by object ID."""
        item = self._item(object_id)
        item.color[:] = np.asarray(rgba, np.float32).reshape(4)
        self._revision += 1

    def set_object_size(self, object_id: int, size) -> None:
        """Set an object's positive shape dimensions by object ID."""
        value = _vec3(size)
        if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
            raise ValueError("Geometry size must contain three positive finite values")
        self._item(object_id).size[:] = value
        self._revision += 1

    def set_geometry_size(self, node_id: int, size) -> bool:
        """Set geometry size through a hierarchy node ID."""
        object_id = self._geom_node_to_oid.get(int(node_id))
        if object_id is None:
            return False
        try:
            self.set_object_size(object_id, size)
        except ValueError:
            return False
        return True

    def set_light(self, light_id: int, light) -> bool:
        """Replace a light by stable ID."""
        try:
            index = self._light_index(light_id)
        except KeyError:
            return False
        return self.set_light_at(index, light)

    def set_light_at(self, light_index: int, light) -> bool:
        """Replace a light by its current zero-based render array index."""
        i = int(light_index)
        if not 0 <= i < len(self.lights.lights):
            return False
        lights = list(self.lights.lights)
        lights[i] = light
        self.lights = replace(self.lights, lights=tuple(lights))
        if self._built_revision == self._revision:
            self._source.lights = self.lights
            self._frame.lights = self.lights
        return True

    def set_environment(self, environment: Environment) -> bool:
        """Replace the scene environment and update the current frame."""
        self.lights = self.lights.with_environment(environment)
        if self._built_revision == self._revision:
            self._source.lights = self.lights
            self._frame.lights = self.lights
        return True

    def set_material(self, material_index: int, material: Material) -> bool:
        """Replace a material by its current scene-source array index."""
        self._rebuild()
        i = int(material_index)
        if not 0 <= i < len(self._source.materials):
            return False
        current = self._source.materials[i]
        for item in self._items:
            if item.material is current:
                item.material = material
        self._source.materials[i] = material
        return True

    def set_geometry_color(self, node_id: int, rgba) -> bool:
        """Set geometry color through a hierarchy node ID."""
        object_id = self._geom_node_to_oid.get(int(node_id))
        if object_id is None:
            return False
        item = self._item(object_id)
        item.color[:] = np.asarray(rgba, np.float32).reshape(4)
        if self._built_revision == self._revision:
            self._source.geom_rgba[self._oid_to_index[object_id]] = item.color
        return True

    def remove(self, object_id: int) -> None:
        """Remove a geometry object by object ID."""
        before = len(self._items)
        self._items = [x for x in self._items if x.object_id != int(object_id)]
        if len(self._items) == before:
            raise KeyError(f"Unknown object_id={object_id}")
        self._revision += 1

    def duplicate_entity(self, object_id: int) -> int:
        """Duplicate an object, light, or camera and return its new object ID."""
        oid = int(object_id)
        if oid >= CAMERA_OBJECT_BASE:
            camera_id = oid - CAMERA_OBJECT_BASE
            item = next((item for item in self._cameras if item.camera_id == camera_id), None)
            if item is None:
                raise KeyError(f"Unknown object_id={object_id}")
            return CAMERA_OBJECT_BASE + self.add_camera(f"{item.name} Copy", item.view)
        if oid >= LIGHT_OBJECT_BASE:
            light_id = oid - LIGHT_OBJECT_BASE
            index = self._light_index(light_id)
            item = self._lights[index]
            duplicate = self.add_light(f"{item.name} Copy", self.lights.lights[index])
            return LIGHT_OBJECT_BASE + duplicate.light_id
        item = self._item(oid)
        duplicate = self.add(
            item.mesh,
            name=f"{item.name} Copy",
            size=item.size,
            position=item.position,
            rotation=item.rotation,
            color=item.color,
            material=item.material,
        )
        return duplicate.object_id

    def remove_entity(self, object_id: int) -> None:
        """Remove an object, light, or camera by selectable object ID."""
        oid = int(object_id)
        if oid >= CAMERA_OBJECT_BASE:
            self.remove_camera(oid - CAMERA_OBJECT_BASE)
        elif oid >= LIGHT_OBJECT_BASE:
            self.remove_light(oid - LIGHT_OBJECT_BASE)
        else:
            self.remove(oid)

    def rename_entity(self, object_id: int, name: str) -> None:
        """Rename an object, light, or camera by selectable object ID."""
        value = str(name).strip()
        if not value:
            raise ValueError("Entity name cannot be empty")
        oid = int(object_id)
        if oid >= CAMERA_OBJECT_BASE:
            camera_id = oid - CAMERA_OBJECT_BASE
            item = next((item for item in self._cameras if item.camera_id == camera_id), None)
        elif oid >= LIGHT_OBJECT_BASE:
            light_id = oid - LIGHT_OBJECT_BASE
            item = next((item for item in self._lights if item.light_id == light_id), None)
        else:
            item = next((item for item in self._items if item.object_id == oid), None)
        if item is None:
            raise KeyError(f"Unknown object_id={object_id}")
        item.name = value
        self._revision += 1

    def reset_poses(self) -> None:
        """Restore every object's pose recorded when it was added."""
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

    def _light_index(self, light_id: int) -> int:
        self._sync_light_items()
        for index, item in enumerate(self._lights):
            if item.light_id == int(light_id):
                return index
        raise KeyError(f"Unknown light_id={light_id}")

    def _sync_light_items(self) -> None:
        count = len(self.lights.lights)
        del self._lights[count:]
        while len(self._lights) < count:
            index = len(self._lights)
            light = self.lights.lights[index]
            light_id = self._next_light_id
            self._next_light_id += 1
            self._lights.append(_LightItem(light_id, f"{light.type.name.lower()} light {index}"))

    def _rebuild(self) -> None:
        if self._built_revision == self._revision:
            return
        n = len(self._items)
        materials: list[Material] = []
        material_ids: dict[int, int] = {}
        nodes = [SceneNode(0, "world", NodeType.WORLD, body_index=0)]
        geom_material: list[int] = []
        self._oid_to_index = {}
        self._node_to_oid = {}
        self._geom_node_to_oid = {}

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
                    NodeType.LINK,
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
                    NodeType.GEOM,
                    parent=link_id,
                    body_index=body_index,
                )
            )
            nodes[0].children.append(link_id)
            self._oid_to_index[item.object_id] = i
            self._node_to_oid[link_id] = item.object_id
            self._geom_node_to_oid[geom_id] = item.object_id

        self._sync_light_items()
        for i, (light, item) in enumerate(zip(self.lights.lights, self._lights, strict=True)):
            node_id = len(nodes)
            nodes.append(
                SceneNode(
                    node_id,
                    item.name,
                    NodeType.LIGHT,
                    parent=0,
                    object_id=LIGHT_OBJECT_BASE + item.light_id,
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
                    NodeType.CAMERA,
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
            geom_visual=np.zeros(n, np.uint8),
            geom_static=np.zeros(n, bool),
            geom_node=np.arange(2, 2 * n + 1, 2, dtype=np.int32),
            geom_local=np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0),
            geom_infinite_plane=np.zeros(n, bool),
            lights=self.lights,
            cameras=tuple(camera.view for camera in self._cameras),
            skybox=self.skybox,
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
