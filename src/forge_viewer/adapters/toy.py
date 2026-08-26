"""Minimal dependency-free physics adapter for Forge."""

from __future__ import annotations

import numpy as np

from ..scene import Scene
from ..types import CameraView, Light, LightSet, LightType
from .base import AdapterCaps, CameraInfo, FrameNeeds, SceneAdapterBase, SceneFrame, SceneSource


class ToyPhysicsAdapter(SceneAdapterBase):
    """Three ballistic bodies with gravity, floor contact and editable poses."""

    caps = AdapterCaps(
        name="toy",
        simulation=True,
        write_pose=True,
        model_cameras=True,
        notes=("dependency-free reference adapter",),
    )

    def __init__(self) -> None:
        light = Light(
            type=LightType.DIRECTIONAL,
            direction=np.array([-0.6, 0.4, -1.0], np.float32),
            diffuse=np.full(3, 0.8, np.float32),
        )
        self.scene = Scene(
            camera=CameraView(
                eye=np.array([6.0, -7.0, 4.5], np.float32),
                target=np.array([0.0, 0.0, 0.8], np.float32),
            ),
            lights=LightSet(lights=(light,), ambient=np.full(3, 0.16, np.float32)),
        )
        self.scene.plane(
            name="ground",
            size=(6.0, 6.0, 0.04),
            position=(0.0, 0.0, -0.04),
            color=(0.22, 0.25, 0.29, 1.0),
        )
        self._objects = (
            self.scene.sphere(
                name="red ball",
                size=(0.38, 0.38, 0.38),
                color=(0.95, 0.28, 0.25, 1.0),
            ),
            self.scene.box(
                name="green box",
                size=(0.34, 0.34, 0.34),
                color=(0.25, 0.78, 0.42, 1.0),
            ),
            self.scene.sphere(
                name="blue ball",
                size=(0.3, 0.3, 0.3),
                color=(0.25, 0.52, 0.96, 1.0),
            ),
        )
        self._initial = np.array([[-1.6, 0.0, 2.4], [0.0, 0.0, 1.4], [1.5, 0.0, 3.1]], np.float32)
        self._positions = self._initial.copy()
        self._velocities = np.array(
            [[1.2, 0.3, 0.0], [-0.4, 0.2, 0.0], [-1.0, -0.25, 0.0]], np.float32
        )
        self._initial_velocities = self._velocities.copy()
        self._radii = np.array([0.38, 0.34, 0.30], np.float32)
        self._steps = 0
        self._sync_scene()
        source = self.scene.source
        source.nodes[1].posable = False
        self._node_to_body = {
            node.node_id: i
            for i, obj in enumerate(self._objects)
            for node in source.nodes
            if node.object_id == obj.object_id
        }

    @property
    def structure_revision(self) -> int:
        return self.scene.structure_revision

    def scene_source(self) -> SceneSource:
        return self.scene.source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        frame = self.scene.frame
        frame.time = self._steps * self.timestep()
        frame.step = self._steps
        frame.debug_commands = (
            {
                "op": "text",
                "id": "toy-time",
                "anchor": (0.0, 0.0, 2.8),
                "text": (
                    "toy physics backend (no MuJoCo)\n"
                    f"gravity + floor collision  |  t = {frame.time:5.2f} s"
                ),
                "align": (0.5, 0.5),
            },
        )
        return frame

    def timestep(self) -> float:
        return 1.0 / 240.0

    def step(self, count: int = 1) -> None:
        dt = self.timestep()
        for _ in range(max(0, int(count))):
            self._velocities[:, 2] -= 9.81 * dt
            self._positions += self._velocities * dt
            hit = self._positions[:, 2] < self._radii
            self._positions[hit, 2] = self._radii[hit]
            self._velocities[hit, 2] = np.abs(self._velocities[hit, 2]) * 0.78
            side = np.abs(self._positions[:, 0]) > 2.8
            self._positions[side, 0] = np.sign(self._positions[side, 0]) * 2.8
            self._velocities[side, 0] *= -1.0
            self._steps += 1
        self._sync_scene()

    def reset(self) -> None:
        self._positions[:] = self._initial
        self._velocities[:] = self._initial_velocities
        self._steps = 0
        self.scene.reset_poses()
        self._sync_scene()

    def set_pose(self, node_id: int, position, rotation) -> bool:
        index = self._node_to_body.get(int(node_id))
        if index is None:
            return False
        self._positions[index] = np.asarray(position, np.float32)
        self._velocities[index] = 0.0
        self._objects[index].set_pose(position, rotation)
        return True

    def camera_hint(self) -> CameraView | None:
        return self.scene.camera

    def cameras(self) -> list[CameraInfo]:
        return self.scene.camera_infos()

    def camera_view(self, camera_id: int) -> CameraView | None:
        return self.scene.camera_view(camera_id)

    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool:
        return self.scene.set_camera(camera_id, camera)

    def _sync_scene(self) -> None:
        for obj, position in zip(self._objects, self._positions, strict=True):
            obj.set_pose(position)
