from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import commands as cmd
from .adapters.base import (
    ENVIRONMENT_OBJECT_ID,
    ActuatorInfo,
    CameraInfo,
    EqualityConstraintInfo,
    FrameNeeds,
    JointInfo,
    KeyframeInfo,
    NodeKind,
    SceneAdapter,
    SceneFrame,
    SceneNode,
    SceneSource,
    SensorInfo,
)
from .commands import Command, CommandResult, Query
from .types import CameraView, Environment, Light


@dataclass
class PerturbState:
    active: bool = False
    node_id: int = -1
    object_id: int = 0
    mode: str = "translate"  # translate / rotate
    grab_point: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    start_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    start_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    target_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    plane_depth: float = 0.0

    body_radius: float = 0.1


class Session:
    def __init__(self, adapter: SceneAdapter, asset_path: Path | None = None) -> None:
        self._adapter = adapter
        self._asset_path = asset_path
        self._paused = not adapter.caps.simulation
        self._speed = 1.0
        self._sim_time_credit = 0.0
        self._selected = 0
        self._step_counter = 0
        self._pending_steps = 0
        self._frame = SceneFrame()
        self._source: SceneSource | None = None
        self._light_overrides: dict[int, Light] = {}
        self._environment_override: Environment | None = None
        self._camera_overrides: dict[int, CameraView] = {}
        self._nodes: list[SceneNode] = []
        self._by_object_id: dict[int, SceneNode] = {}
        self._joints: list[JointInfo] = []
        self._actuators: list[ActuatorInfo] = []
        self._cameras: list[CameraInfo] = []
        self._keyframes: list[KeyframeInfo] = []
        self._sensor_infos: list[SensorInfo] = []
        self._equality_constraints: list[EqualityConstraintInfo] = []
        self._active_keyframe = -1
        self._perturb = PerturbState()
        self._camera = CameraView()
        self._last_message = ""
        self._structure_generation = 0
        self._adapter_revision = -1
        self._refresh_structure()

    @property
    def adapter(self) -> SceneAdapter:
        return self._adapter

    @property
    def paused(self) -> bool:

        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def selected(self) -> int:
        return self._selected

    @property
    def selected_node(self) -> SceneNode | None:
        return self._by_object_id.get(self._selected)

    @property
    def frame(self) -> SceneFrame:
        return self._frame

    @property
    def source(self) -> SceneSource | None:
        return self._source

    @property
    def nodes(self) -> list[SceneNode]:
        return self._nodes

    @property
    def joints(self) -> list[JointInfo]:
        return self._joints

    @property
    def actuators(self) -> list[ActuatorInfo]:
        return self._actuators

    @property
    def cameras(self) -> list[CameraInfo]:
        return self._cameras

    @property
    def keyframes(self) -> list[KeyframeInfo]:
        return self._keyframes

    @property
    def active_keyframe(self) -> int:
        return self._active_keyframe

    @property
    def sensor_infos(self) -> list[SensorInfo]:
        return self._sensor_infos

    @property
    def equality_constraints(self) -> list[EqualityConstraintInfo]:
        return self._equality_constraints

    @property
    def perturb(self) -> PerturbState:
        return self._perturb

    @property
    def camera(self) -> CameraView:

        return self._camera

    @property
    def asset_path(self) -> Path | None:
        return self._asset_path

    @property
    def last_message(self) -> str:
        return self._last_message

    @property
    def structure_generation(self) -> int:

        return self._structure_generation

    def node(self, node_id: int) -> SceneNode | None:
        for n in self._nodes:
            if n.node_id == node_id:
                return n
        return None

    def node_by_object_id(self, object_id: int) -> SceneNode | None:
        return self._by_object_id.get(int(object_id))

    def tick(self, needs: FrameNeeds, wall_dt: float | None = None) -> SceneFrame:

        if not self._paused and not self._adapter.caps.external_clock:
            timestep = self._adapter.timestep()
            if wall_dt is not None and timestep > 0.0:
                self._sim_time_credit += float(wall_dt) * self._speed
                n = int(self._sim_time_credit / timestep + 1e-9)
                self._sim_time_credit -= n * timestep
            else:
                n = max(1, round(self._speed))
            if n:
                self._adapter.step(n)
                self._step_counter += n
        elif self._pending_steps > 0:
            self._adapter.step(self._pending_steps)
            self._step_counter += self._pending_steps
            self._pending_steps = 0

        if self._adapter.structure_revision != self._adapter_revision:
            self._refresh_structure()

        self._frame = self._adapter.frame(needs)
        self._sync_equality_state()
        self._compose_lights()
        self._compose_cameras()
        if self._adapter.caps.external_clock:
            self._paused = bool(self._frame.paused)
        else:
            self._frame.paused = self._paused
            self._frame.step = self._step_counter
        return self._frame

    def submit(self, command: Command) -> CommandResult:

        result = self._dispatch(command)
        self._last_message = result.message
        return result

    def _dispatch(self, c: Command) -> CommandResult:
        caps = self._adapter.caps

        if isinstance(c, cmd.Pause):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation to pause")
            if self._paused:
                return CommandResult.good("Simulation is already paused")
            if not self._adapter.set_paused(True):
                return CommandResult.bad("physics backend rejected pause")
            self._paused = True
            self._sim_time_credit = 0.0
            return CommandResult.good("Simulation paused")

        if isinstance(c, cmd.Play):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation to resume")
            if not self._paused:
                return CommandResult.good("Simulation is already running")
            if not self._adapter.set_paused(False):
                return CommandResult.bad("physics backend rejected play")
            self._paused = False
            self._sim_time_credit = 0.0
            self._perturb = PerturbState()
            return CommandResult.good("Simulation resumed")

        if isinstance(c, cmd.Step):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation to step")
            if not self._paused:
                return CommandResult.bad("Pause the simulation before stepping")
            self._pending_steps += max(1, c.count)
            return CommandResult.good(f"Stepped {c.count} frame(s)")

        if isinstance(c, cmd.Reset):
            self._adapter.reset()
            self._step_counter = 0
            self._sim_time_credit = 0.0
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._equality_constraints = (
                self._adapter.equality_constraints() if caps.equality_constraints else []
            )

            return CommandResult.good("Scene reset")

        if isinstance(c, cmd.Reload):
            if not caps.reload:
                return CommandResult.bad(f"{caps.name} does not support reload")
            try:
                self._adapter.reload()
            except Exception as exc:
                return CommandResult.bad(str(exc))
            self._step_counter = 0
            self._sim_time_credit = 0.0
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._light_overrides.clear()
            self._environment_override = None
            self._camera_overrides.clear()
            self._refresh_structure()
            return CommandResult.good("Scene reloaded")

        if isinstance(c, cmd.LoadAsset):
            if not caps.asset_loading:
                return CommandResult.bad(f"{caps.name} does not support model loading")
            path = Path(c.path).expanduser().resolve()
            try:
                self._adapter.load(path)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            self._asset_path = path
            self._step_counter = 0
            self._sim_time_credit = 0.0
            self._pending_steps = 0
            self._selected = 0
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._light_overrides.clear()
            self._environment_override = None
            self._camera_overrides.clear()
            self._refresh_structure()
            return CommandResult.good(f"Loaded {c.path.name}")

        if isinstance(c, cmd.LoadKeyframe):
            if not caps.keyframes:
                return CommandResult.bad(f"{caps.name} does not expose keyframes")
            if not self._paused:
                return CommandResult.bad("physics is running; pause to load a keyframe")
            i = int(c.keyframe_id)
            if not 0 <= i < len(self._keyframes):
                return CommandResult.bad(f"keyframe {i} is unavailable")
            if not self._adapter.load_keyframe(i):
                return CommandResult.bad(f"failed to load keyframe {i}")
            self._step_counter = 0
            self._sim_time_credit = 0.0
            self._pending_steps = 0
            self._perturb = PerturbState()
            self._active_keyframe = i
            return CommandResult.good(f"loaded {self._keyframes[i].name}")

        if isinstance(c, cmd.Select):
            node = self._by_object_id.get(int(c.object_id))
            if c.object_id and node is None:
                return CommandResult.bad(f"Unknown object_id={c.object_id}")
            self._selected = int(c.object_id)
            return CommandResult.good(node.name if node else "Selection cleared")

        if isinstance(c, cmd.SetVisible):
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            node.visible = c.visible
            if self._source is not None:
                source_node = next(
                    (item for item in self._source.nodes if item.node_id == c.node_id), None
                )
                if source_node is not None:
                    source_node.visible = c.visible
            self._structure_generation += 1
            return CommandResult.good("")

        if isinstance(c, cmd.SetVisualGroup):
            if not caps.visual_groups:
                return CommandResult.bad(f"{caps.name} does not expose visual groups")
            ok = self._adapter.set_visual_group(c.category, c.group, c.visible)
            return (
                CommandResult.good("")
                if ok
                else CommandResult.bad(f"visual group {c.category}:{c.group} is unavailable")
            )

        if isinstance(c, cmd.SetPose):
            if not caps.write_pose:
                return CommandResult.bad(f"{caps.name} does not support pose editing")
            if not self._paused:
                return CommandResult.bad("physics is running; pause to move things")
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            if not node.posable:
                return CommandResult.bad("this link is driven by joints; use the Joints panel")
            ok = self._adapter.set_pose(c.node_id, c.position, c.rotation)
            return CommandResult.good("") if ok else CommandResult.bad("Pose update failed")

        if isinstance(c, cmd.SetQpos):
            if not caps.write_qpos:
                return CommandResult.bad(f"{caps.name} does not support joint editing")
            ok = self._adapter.set_qpos(c.index, c.value)
            return (
                CommandResult.good("")
                if ok
                else CommandResult.bad(f"Joint {c.index} update failed")
            )

        if isinstance(c, cmd.SetEqualityEnabled):
            if not caps.equality_constraints:
                return CommandResult.bad(f"{caps.name} does not expose equality constraints")
            i = int(c.constraint_id)
            if not 0 <= i < len(self._equality_constraints):
                return CommandResult.bad(f"equality constraint {i} is unavailable")
            if not self._adapter.set_equality_enabled(i, c.enabled):
                return CommandResult.bad(f"equality constraint {i} update failed")
            self._equality_constraints[i] = replace(
                self._equality_constraints[i], enabled=bool(c.enabled)
            )
            return CommandResult.good("")

        if isinstance(c, cmd.SetCtrl):
            ok = self._adapter.set_ctrl(c.index, c.value)
            return (
                CommandResult.good("")
                if ok
                else CommandResult.bad(f"Actuator {c.index} update failed")
            )

        if isinstance(c, cmd.Perturb):
            if not caps.perturb:
                return CommandResult.bad(f"{caps.name} does not support perturbation")
            ok = self._adapter.apply_perturb(
                c.node_id, c.target_position, c.target_rotation, c.mode
            )
            return CommandResult.good("") if ok else CommandResult.bad("Perturbation failed")

        if isinstance(c, cmd.ClearPerturb):
            self._adapter.clear_perturb()
            self._perturb = PerturbState()
            return CommandResult.good("")

        if isinstance(c, cmd.SetLight):
            if self._source is None or not 0 <= c.light_id < len(self._source.lights.lights):
                return CommandResult.bad(f"light {c.light_id} is unavailable")
            writeback = self._adapter.set_light(c.light_id, c.light)
            lights = list(self._source.lights.lights)
            lights[c.light_id] = c.light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
            self._light_overrides[c.light_id] = c.light
            for node in self._nodes:
                if node.light_index == c.light_id:
                    node.visible = c.light.active
                    break
            self._compose_lights()
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetEnvironment):
            if self._source is None:
                return CommandResult.bad("environment is unavailable")
            writeback = self._adapter.set_environment(c.environment)
            self._source.lights = self._source.lights.with_environment(c.environment)
            self._environment_override = c.environment
            self._compose_lights()
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetSceneCamera):
            camera_id = int(c.camera_id)
            slot = self._camera_slot(camera_id)
            if self._source is None or slot < 0 or slot >= len(self._source.cameras):
                return CommandResult.bad(f"camera {camera_id} is unavailable")
            writeback = self._adapter.set_camera_view(camera_id, c.camera)
            cameras = list(self._source.cameras)
            cameras[slot] = c.camera
            self._source.cameras = tuple(cameras)
            self._camera_overrides[camera_id] = c.camera
            self._compose_cameras()
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetSpeed):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation speed")
            self._speed = max(0.05, float(c.factor))
            return CommandResult.good(f"Speed ×{self._speed:g}")

        if isinstance(c, cmd.SetCamera):
            self._camera = c.camera
            return CommandResult.good("")

        return CommandResult.bad(f"Unknown command: {type(c).__name__}")

    def query(self, q: Query):

        if isinstance(q, cmd.Pick):
            if not self._adapter.caps.raycast:
                return (0, float("inf"))
            return self._adapter.raycast(q.origin, q.direction)
        if isinstance(q, cmd.NodeAt):
            return self._by_object_id.get(int(q.object_id))
        if isinstance(q, cmd.Bounds):
            return self.bounds()
        raise TypeError(f"Unknown query: {type(q).__name__}")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:

        src = self._source
        frame = self._frame
        if src is None:
            return np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32)

        lo: np.ndarray | None = None
        hi: np.ndarray | None = None
        if frame.geom_xpos is not None and len(frame.geom_xpos):
            pos = np.asarray(frame.geom_xpos, np.float32)
            size = src.geom_size[: len(pos)] if len(src.geom_size) >= len(pos) else None
            finite = np.isfinite(pos).all(axis=1)

            if len(src.geom_infinite_plane) == len(pos):
                finite &= ~src.geom_infinite_plane
            if finite.any():
                p = pos[finite]
                r = np.max(size[finite], axis=1, keepdims=True) if size is not None else 0.0
                lo, hi = (p - r).min(axis=0), (p + r).max(axis=0)

        for key in src.dynamic_meshes:
            points = np.asarray(src.meshes[key].positions, np.float32)
            points = points[np.isfinite(points).all(axis=1)]
            if not len(points):
                continue
            mesh_lo, mesh_hi = points.min(axis=0), points.max(axis=0)
            lo = mesh_lo if lo is None else np.minimum(lo, mesh_lo)
            hi = mesh_hi if hi is None else np.maximum(hi, mesh_hi)

        if lo is None or hi is None:
            return np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32)
        return lo.astype(np.float32), hi.astype(np.float32)

    def camera_hint(self) -> CameraView | None:
        return self._adapter.camera_hint()

    def camera_view(self, camera_id: int) -> CameraView | None:
        i = int(camera_id)
        if i in self._camera_overrides:
            return self._camera_overrides[i]
        return self._adapter.camera_view(i) if self._adapter.caps.model_cameras else None

    def visual_groups(self):
        return self._adapter.visual_groups() if self._adapter.caps.visual_groups else ()

    def _refresh_structure(self) -> None:
        self._source = self._adapter.scene_source()
        if self._environment_override is not None:
            self._source.lights = self._source.lights.with_environment(self._environment_override)
        if self._light_overrides:
            lights = list(self._source.lights.lights)
            for i, light in self._light_overrides.items():
                if i < len(lights):
                    lights[i] = light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
        self._nodes = [
            replace(node, children=list(node.children)) for node in self._adapter.nodes()
        ]
        if not any(node.kind is NodeKind.ENVIRONMENT for node in self._nodes):
            parent = next(
                (node for node in self._nodes if node.kind is NodeKind.WORLD and node.parent < 0),
                None,
            )
            node_id = max((node.node_id for node in self._nodes), default=-1) + 1
            environment = SceneNode(
                node_id,
                "environment",
                NodeKind.ENVIRONMENT,
                parent=parent.node_id if parent is not None else -1,
                object_id=ENVIRONMENT_OBJECT_ID,
            )
            self._nodes.append(environment)
            if parent is not None:
                parent.children.append(node_id)
        for node in self._nodes:
            if 0 <= node.light_index < len(self._source.lights.lights):
                node.visible = self._source.lights.lights[node.light_index].active
        self._joints = self._adapter.joints()
        self._actuators = self._adapter.actuators()
        self._cameras = self._adapter.cameras() if self._adapter.caps.model_cameras else []
        if self._camera_overrides:
            cameras = list(self._source.cameras)
            for camera_id, camera in self._camera_overrides.items():
                slot = self._camera_slot(camera_id)
                if 0 <= slot < len(cameras):
                    cameras[slot] = camera
            self._source.cameras = tuple(cameras)
        self._keyframes = self._adapter.keyframes() if self._adapter.caps.keyframes else []
        self._sensor_infos = self._adapter.sensors() if self._adapter.caps.sensors else []
        self._equality_constraints = (
            self._adapter.equality_constraints() if self._adapter.caps.equality_constraints else []
        )
        if self._active_keyframe >= len(self._keyframes):
            self._active_keyframe = -1
        self._by_object_id = {n.object_id: n for n in self._nodes if n.object_id}
        self._adapter_revision = self._adapter.structure_revision
        self._structure_generation += 1
        self._frame = self._adapter.frame(FrameNeeds())
        self._sync_equality_state()
        self._compose_lights()
        self._compose_cameras()

    def _compose_lights(self) -> None:
        """Combine Forge-authored light settings with backend-driven transforms.

        A physics backend may move a body-attached light and publish its world
        position/direction in ``SceneFrame``.  Color, intensity, range, fog and
        every other render setting still come from the Forge scene.
        """
        if self._source is None:
            return
        authored = self._source.lights
        driven = self._frame.lights
        if driven is None or len(driven.lights) != len(authored.lights):
            self._frame.lights = authored
            return
        lights = tuple(
            replace(light, position=dynamic.position, direction=dynamic.direction)
            for light, dynamic in zip(authored.lights, driven.lights, strict=True)
        )
        self._frame.lights = replace(authored, lights=lights)

    def _sync_equality_state(self) -> None:
        values = self._frame.equality_enabled
        if values is None or len(values) != len(self._equality_constraints):
            return
        for i, enabled in enumerate(values):
            if self._equality_constraints[i].enabled != bool(enabled):
                self._equality_constraints[i] = replace(
                    self._equality_constraints[i], enabled=bool(enabled)
                )

    def _compose_cameras(self) -> None:
        if self._source is None:
            return
        driven = self._frame.cameras
        cameras = list(driven if driven is not None else self._source.cameras)
        if len(cameras) != len(self._source.cameras):
            cameras = list(self._source.cameras)
        for camera_id, camera in self._camera_overrides.items():
            slot = self._camera_slot(camera_id)
            if 0 <= slot < len(cameras):
                cameras[slot] = camera
        self._frame.cameras = tuple(cameras)

    def _camera_slot(self, camera_id: int) -> int:
        return next(
            (slot for slot, camera in enumerate(self._cameras) if camera.camera_id == camera_id),
            -1,
        )

    def release(self) -> None:
        self._adapter.release()
