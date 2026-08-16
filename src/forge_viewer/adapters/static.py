from __future__ import annotations

from ..scene import Scene
from .base import AdapterCaps, FrameNeeds, SceneAdapterBase, SceneFrame, SceneNode, SceneSource


class StaticSceneAdapter(SceneAdapterBase):
    caps = AdapterCaps(name="static", simulation=False, write_pose=True)

    def __init__(self, scene: Scene) -> None:
        self.scene = scene

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

    def camera_hint(self):
        return self.scene.camera
