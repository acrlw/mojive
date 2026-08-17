from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..types import CameraView, LightSet, Material, MeshData, MeshKey, MeshUpdate, TextureData

if TYPE_CHECKING:
    from pathlib import Path


class NodeKind(enum.StrEnum):
    WORLD = "world"
    ROBOT = "robot"
    LINK = "link"
    GEOM = "geom"
    JOINT = "joint"
    LIGHT = "light"
    CAMERA = "camera"
    SITE = "site"
    FLEX = "flex"
    SKIN = "skin"


@dataclass
class SceneNode:
    node_id: int
    name: str
    kind: NodeKind
    parent: int = -1
    children: list[int] = field(default_factory=list)
    object_id: int = 0

    posable: bool = False

    visible: bool = True
    body_index: int = -1
    light_index: int = -1


@dataclass
class JointInfo:
    joint_id: int
    name: str
    kind: str  # free / ball / slide / hinge
    limited: bool
    range: tuple[float, float]
    qpos_adr: int
    qvel_adr: int
    dof: int
    body: int = -1


@dataclass
class ActuatorInfo:
    actuator_id: int
    name: str
    ctrl_range: tuple[float, float]
    ctrl_limited: bool
    ctrl_address: int = 0
    ctrl_count: int = 1
    gain: float = 1.0
    joint: int = -1


@dataclass(frozen=True)
class CameraInfo:
    """A named camera supplied by the scene source."""

    camera_id: int
    name: str


@dataclass(frozen=True)
class KeyframeInfo:
    """A named MuJoCo state preset; unnamed motion frames get a stable fallback label."""

    keyframe_id: int
    name: str
    time: float


@dataclass(frozen=True)
class SensorInfo:
    sensor_id: int
    name: str
    kind: str
    data_adr: int
    dim: int


@dataclass(frozen=True)
class VisualGroupInfo:
    """One independently switchable family of numbered visibility groups."""

    category: str
    visible: tuple[bool, ...]


@dataclass(frozen=True)
class AdapterCaps:
    name: str = "?"
    simulation: bool = False

    external_clock: bool = False

    write_pose: bool = False
    write_qpos: bool = False
    perturb: bool = False
    raycast: bool = False
    inverse_kinematics: bool = False
    contacts: bool = False
    model_cameras: bool = False
    keyframes: bool = False
    sensors: bool = False
    visual_groups: bool = False
    reload: bool = False
    notes: tuple[str, ...] = ()


class JointVisualKind(enum.IntEnum):
    FREE = 0
    BALL = 1
    SLIDE = 2
    HINGE = 3


class ActuatorVisualKind(enum.IntEnum):
    SLIDE = 0
    HINGE = 1
    BALL = 2
    FREE = 3
    SPHERE = 4
    ELLIPSOID = 5
    CAPSULE = 6
    CYLINDER = 7
    BOX = 8


@dataclass(frozen=True)
class DiagnosticSource:
    joint_kinds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    joint_visible: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    joint_length: float = 0.0
    joint_width: float = 0.0
    joint_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.6, 0.8, 1.0], np.float32)
    )

    com_bodies: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    com_radius: float = 0.0
    com_rgba: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.9, 0.9, 1.0], np.float32))

    inertia_bodies: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    inertia_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    scaled_inertia_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    inertia_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.8, 0.2, 0.2, 0.6], np.float32)
    )

    actuator_visual_kinds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    actuator_visual_actuators: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    actuator_visual_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))


@dataclass
class DiagnosticFrame:
    joint_xpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    joint_xaxis: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    subtree_com: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    body_xipos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    body_ximat: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    actuator_xpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    actuator_xmat: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))


@dataclass
class FrameNeeds:
    poses: bool = True
    qpos: bool = False
    qvel: bool = False
    contacts: bool = False
    tendons: bool = False
    actuator: bool = False
    sensors: bool = False
    deformables: bool = False
    diagnostics: bool = False

    def merge(self, other: FrameNeeds) -> FrameNeeds:
        return FrameNeeds(
            poses=self.poses or other.poses,
            qpos=self.qpos or other.qpos,
            qvel=self.qvel or other.qvel,
            contacts=self.contacts or other.contacts,
            tendons=self.tendons or other.tendons,
            actuator=self.actuator or other.actuator,
            sensors=self.sensors or other.sensors,
            deformables=self.deformables or other.deformables,
            diagnostics=self.diagnostics or other.diagnostics,
        )

    @staticmethod
    def none() -> FrameNeeds:
        return FrameNeeds(poses=False)


@dataclass
class SceneFrame:
    time: float = 0.0
    step: int = 0
    paused: bool = False

    geom_xpos: np.ndarray | None = None
    geom_xmat: np.ndarray | None = None
    site_xpos: np.ndarray | None = None  # (S, 3) f32
    site_xmat: np.ndarray | None = None  # (S, 3, 3) f32
    body_xpos: np.ndarray | None = None
    body_xmat: np.ndarray | None = None

    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    ctrl: np.ndarray | None = None
    actuator_activation: np.ndarray | None = None

    contacts: np.ndarray | None = None  # (C, 7)：pos(3) + normal(3) + force
    tendon_segments: np.ndarray | None = None  # (W, 2, 3) f32
    tendon_ids: np.ndarray | None = None
    tendon_widths: np.ndarray | None = None
    sensors: np.ndarray | None = None
    mesh_updates: dict[MeshKey, MeshUpdate] | None = None
    diagnostics: DiagnosticFrame | None = None

    debug_commands: tuple[dict, ...] | None = None

    lights: LightSet | None = None


@dataclass
class SceneSource:
    meshes: dict[MeshKey, MeshData] = field(default_factory=dict)
    dynamic_meshes: frozenset[MeshKey] = frozenset()

    textures: dict[str, TextureData] = field(default_factory=dict)
    materials: list[Material] = field(default_factory=list)

    geom_mesh: list[MeshKey] = field(default_factory=list)
    geom_material: list[int] = field(default_factory=list)
    geom_size: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    geom_rgba: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    geom_object_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))
    geom_body: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    geom_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    geom_pose_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))

    geom_node: np.ndarray = field(default_factory=lambda: np.full(0, -1, np.int32))

    geom_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), np.float32))

    geom_infinite_plane: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    initial_qpos: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    initial_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    tendon_rgba: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    tendon_material: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    tendon_visible: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    actuator_tendon_scale: float = 1.0

    actuator_tendon: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    actuator_visible: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    actuator_ctrl_address: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    actuator_ctrl_limited: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    actuator_ctrl_range: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    actuator_act_limited: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    actuator_act_range: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    actuator_dynamic: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    actuator_rgba: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.175, 0.1, 0.1, 1.0], [0.7, 0.4, 0.4, 1.0], [1.0, 0.0, 0.0, 1.0]],
            np.float32,
        )
    )

    diagnostics: DiagnosticSource = field(default_factory=DiagnosticSource)

    lights: LightSet = field(default_factory=LightSet)
    skybox: str | None = None
    shadow_clip: float = 1.0

    scene_extent: float = 1.0
    scene_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    nodes: list[SceneNode] = field(default_factory=list)

    @property
    def instance_count(self) -> int:
        return len(self.geom_mesh)


class SceneAdapterBase:
    caps = AdapterCaps(name="custom")

    @property
    def structure_revision(self) -> int:
        return 0

    def load(self, path: Path) -> None:
        raise RuntimeError(f"{self.caps.name} does not support asset loading")

    def reload(self) -> None:
        raise RuntimeError(f"{self.caps.name} does not support reload")

    def reset(self) -> None: ...
    def step(self, count: int = 1) -> None: ...

    def set_paused(self, paused: bool) -> bool:
        return True

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        raise NotImplementedError

    def scene_source(self) -> SceneSource:
        raise NotImplementedError

    def nodes(self) -> list[SceneNode]:
        return self.scene_source().nodes

    def joints(self) -> list[JointInfo]:
        return []

    def actuators(self) -> list[ActuatorInfo]:
        return []

    def cameras(self) -> list[CameraInfo]:
        return []

    def keyframes(self) -> list[KeyframeInfo]:
        return []

    def sensors(self) -> list[SensorInfo]:
        return []

    def load_keyframe(self, keyframe_id: int) -> bool:
        return False

    def camera_view(self, camera_id: int) -> CameraView | None:
        return None

    def visual_groups(self) -> tuple[VisualGroupInfo, ...]:
        return ()

    def set_visual_group(self, category: str, group: int, visible: bool) -> bool:
        return False

    def set_qpos(self, index: int, value: float) -> bool:
        return False

    def set_ctrl(self, index: int, value: float) -> bool:
        return False

    def set_pose(self, node_id: int, position, rotation) -> bool:
        return False

    def set_light(self, light_id: int, light) -> bool:
        """Edit a Forge scene light.

        Lights belong to ``SceneSource``, not to the physics backend.  Adapters
        with a native light representation may override this to write through;
        the default keeps custom and render-only adapters editable.
        """
        source = self.scene_source()
        i = int(light_id)
        if not 0 <= i < len(source.lights.lights):
            return False
        lights = list(source.lights.lights)
        lights[i] = light
        source.lights = replace(source.lights, lights=tuple(lights))
        return True

    def apply_perturb(
        self, node_id: int, target_position: np.ndarray, target_rotation: np.ndarray, mode: str
    ) -> bool:
        return False

    def clear_perturb(self) -> None: ...

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]:
        return (0, float("inf"))

    def camera_hint(self) -> CameraView | None:
        return None

    def timestep(self) -> float:
        return 0.0

    def release(self) -> None: ...


@runtime_checkable
class SceneAdapter(Protocol):
    caps: AdapterCaps

    @property
    def structure_revision(self) -> int: ...

    def load(self, path: Path) -> None: ...
    def reload(self) -> None: ...
    def reset(self) -> None: ...
    def step(self, count: int = 1) -> None: ...
    def set_paused(self, paused: bool) -> bool: ...
    def frame(self, needs: FrameNeeds) -> SceneFrame: ...
    def scene_source(self) -> SceneSource: ...
    def nodes(self) -> list[SceneNode]: ...
    def joints(self) -> list[JointInfo]: ...
    def actuators(self) -> list[ActuatorInfo]: ...
    def cameras(self) -> list[CameraInfo]: ...
    def keyframes(self) -> list[KeyframeInfo]: ...
    def sensors(self) -> list[SensorInfo]: ...
    def load_keyframe(self, keyframe_id: int) -> bool: ...
    def camera_view(self, camera_id: int) -> CameraView | None: ...
    def visual_groups(self) -> tuple[VisualGroupInfo, ...]: ...
    def set_visual_group(self, category: str, group: int, visible: bool) -> bool: ...
    def set_qpos(self, index: int, value: float) -> bool: ...
    def set_ctrl(self, index: int, value: float) -> bool: ...
    def set_pose(self, node_id: int, position, rotation) -> bool: ...
    def set_light(self, light_id: int, light) -> bool: ...
    def apply_perturb(
        self, node_id: int, target_position: np.ndarray, target_rotation: np.ndarray, mode: str
    ) -> bool: ...
    def clear_perturb(self) -> None: ...
    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]: ...

    def camera_hint(self) -> CameraView | None: ...
    def timestep(self) -> float: ...
    def release(self) -> None: ...
