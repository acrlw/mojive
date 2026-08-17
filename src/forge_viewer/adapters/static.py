from __future__ import annotations

from dataclasses import replace

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
    caps = AdapterCaps(name="static", simulation=False, write_pose=True)

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.caps = replace(self.caps, model_cameras=True)

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

    def cameras(self) -> list[CameraInfo]:
        return self.scene.camera_infos()

    def camera_view(self, camera_id: int):
        return self.scene.camera_view(camera_id)

    def set_camera_view(self, camera_id: int, camera) -> bool:
        return self.scene.set_camera(camera_id, camera)

    def camera_hint(self):
        return self.scene.camera
