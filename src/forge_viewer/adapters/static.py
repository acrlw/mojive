"""Scene adapter for authored, backend-neutral scenes."""

from __future__ import annotations

from pathlib import Path

from ..scene import Scene
from .base import (
    AdapterCaps,
    CameraInfo,
    FrameNeeds,
    SceneAdapterBase,
    SceneFrame,
    SceneNode,
    SceneSource,
)


class StaticSceneAdapter(SceneAdapterBase):
    caps = AdapterCaps(
        name="static",
        simulation=False,
        write_pose=True,
        model_cameras=True,
        reload=True,
        scene_authoring=True,
        scene_files=True,
        edit_history=True,
    )

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._path: Path | None = None

    def new_scene(self) -> None:
        self.scene = Scene()
        self._path = None

    def open_scene(self, path: Path) -> None:
        self.scene = Scene.load(path)
        self._path = Path(path).expanduser().resolve()

    def save_scene(self, path: Path) -> None:
        self._path = self.scene.save(path)

    def capture_edit_state(self) -> object:
        return self.scene.clone()

    def restore_edit_state(self, state: object) -> bool:
        if not isinstance(state, Scene):
            return False
        self.scene = state.clone()
        return True

    def reload(self) -> None:
        if self._path is None:
            raise RuntimeError("Scene has not been saved")
        self.scene = Scene.load(self._path)

    @property
    def structure_revision(self) -> int:
        return self.scene.structure_revision

    def scene_source(self) -> SceneSource:
        return self.scene.source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        return self.scene.frame

    def nodes(self) -> list[SceneNode]:
        return self.scene.source.nodes

    def reset(self) -> None:
        self.scene.reset_poses()

    def set_pose(self, node_id: int, position, rotation) -> bool:
        return self.scene.set_pose_by_node(node_id, position, rotation)

    def set_light(self, light_id: int, light) -> bool:
        return self.scene.set_light(light_id, light)

    def set_environment(self, environment) -> bool:
        return self.scene.set_environment(environment)

    def set_material(self, material_id, material) -> bool:
        return self.scene.set_material(material_id, material)

    def set_geometry_color(self, node_id, rgba) -> bool:
        return self.scene.set_geometry_color(node_id, rgba)

    def set_geometry_size(self, node_id, size) -> bool:
        return self.scene.set_geometry_size(node_id, size)

    def cameras(self) -> list[CameraInfo]:
        return self.scene.camera_infos()

    def camera_view(self, camera_id: int):
        return self.scene.camera_view(camera_id)

    def set_camera_view(self, camera_id: int, camera) -> bool:
        return self.scene.set_camera(camera_id, camera)

    def add_scene_object(self, shape, name, size, position, rotation, color, material) -> int:
        return self.scene.add(
            shape,
            name=name,
            size=size,
            position=position,
            rotation=rotation,
            color=color,
            material=material,
        ).object_id

    def remove_scene_object(self, object_id: int) -> bool:
        try:
            self.scene.remove(object_id)
        except KeyError:
            return False
        return True

    def add_scene_light(self, name: str, light) -> int:
        return self.scene.add_light(name, light).light_id

    def remove_scene_light(self, light_id: int) -> bool:
        try:
            self.scene.remove_light(light_id)
        except KeyError:
            return False
        return True

    def add_scene_camera(self, name: str, camera) -> int:
        return self.scene.add_camera(name, camera)

    def remove_scene_camera(self, camera_id: int) -> bool:
        try:
            self.scene.remove_camera(camera_id)
        except KeyError:
            return False
        return True

    def duplicate_scene_entity(self, object_id: int) -> int:
        try:
            return self.scene.duplicate_entity(object_id)
        except KeyError:
            return 0

    def remove_scene_entity(self, object_id: int) -> bool:
        try:
            self.scene.remove_entity(object_id)
        except KeyError:
            return False
        return True

    def rename_scene_entity(self, object_id: int, name: str) -> bool:
        try:
            self.scene.rename_entity(object_id, name)
        except (KeyError, ValueError):
            return False
        return True

    def camera_hint(self):
        return self.scene.camera
