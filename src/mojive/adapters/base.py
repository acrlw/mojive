"""Shared scene adapter contracts and capability metadata."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..types import (
    CameraView,
    Environment,
    Light,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    MeshUpdate,
    ShadingModel,
    TextureData,
    TextureType,
)

if TYPE_CHECKING:
    from ..commands import ModelEdit

GEOMETRY_OBJECT_BASE = 0x50000000
LIGHT_OBJECT_BASE = 0x70000000
CAMERA_OBJECT_BASE = 0x71000000
ENVIRONMENT_OBJECT_ID = 0x72000000
MODEL_OBJECT_BASE = 0x73000000


class NodeType(enum.StrEnum):
    """Semantic node categories shown by the hierarchy and selection system."""

    WORLD = "world"
    MODEL = "model"
    ROBOT = "robot"
    LINK = "link"
    GEOM = "geom"
    JOINT = "joint"
    LIGHT = "light"
    CAMERA = "camera"
    ENVIRONMENT = "environment"
    SITE = "site"
    FLEX = "flex"
    SKIN = "skin"


@dataclass
class SceneNode:
    """One stable hierarchy node exposed by a scene adapter.

    ``node_id`` identifies the hierarchy entry and write-back target. ``object_id`` identifies
    the rendered selection object. Physics-specific indices are optional lookup accelerators.
    """

    node_id: int
    name: str
    type: NodeType
    parent: int = -1
    children: list[int] = field(default_factory=list)
    object_id: int = 0

    posable: bool = False

    visible: bool = True
    body_index: int = -1
    geom_index: int = -1
    light_index: int = -1
    camera_index: int = -1
    site_index: int = -1
    joint_index: int = -1
    model_id: int = -1

    # True only when this compiled node has a stable element in the editable
    # scene source. Runtime pose control (``posable``) is intentionally
    # separate: a free body can be movable even when no authored source exists.
    source_editable: bool = False


@dataclass
class JointInfo:
    """Joint metadata used by the Joints panel and qpos editors."""

    joint_id: int
    name: str
    type: str  # free / ball / slide / hinge
    limited: bool
    range: tuple[float, float]
    qpos_adr: int
    qvel_adr: int
    dof: int
    body: int = -1
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    damping: float = 0.0
    stiffness: float = 0.0


@dataclass
class ActuatorInfo:
    """Actuator control metadata exposed by a simulation adapter."""

    actuator_id: int
    name: str
    ctrl_range: tuple[float, float]
    ctrl_limited: bool
    ctrl_address: int = 0
    ctrl_count: int = 1
    act_address: int = -1
    act_count: int = 0
    gain: float = 1.0
    joint: int = -1


@dataclass(frozen=True)
class CameraInfo:
    """Stable camera identity and selectable scene object ID."""

    camera_id: int
    name: str
    object_id: int = 0


@dataclass(frozen=True)
class SceneModelInfo:
    """One file-backed model participating in an adapter scene."""

    model_id: int
    name: str
    path: Path
    removable: bool
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class SceneSaveOptions:
    """Optional format-specific behavior for a scene save operation."""

    current_pose_keyframe: str | None = None


@dataclass(frozen=True)
class ModelComponentField:
    """One editable MJCF attribute with optional model-local reference choices."""

    name: str
    value: str
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelComponentPathItem:
    """One nested tendon path element such as a site or joint reference."""

    type: str
    fields: tuple[ModelComponentField, ...] = ()


@dataclass(frozen=True)
class ModelComponentInfo:
    """One model-level contact, actuator, sensor, tendon, or equality declaration."""

    component_id: int
    model_id: int
    category: str
    subtype: str
    name: str
    fields: tuple[ModelComponentField, ...] = ()
    path: tuple[ModelComponentPathItem, ...] = ()
    path_presets: tuple[ModelComponentPathItem, ...] = ()


@dataclass(frozen=True)
class ModelAssetInfo:
    """One model-local MJCF asset and the elements that reference it."""

    model_id: int
    type: str
    name: str
    index: int
    file: str = ""
    fields: tuple[ModelComponentField, ...] = ()
    references: tuple[str, ...] = ()
    data_shape: tuple[int, int] = (0, 0)
    preview_shape: tuple[int, int] = (0, 0)
    preview_values: tuple[float, ...] = ()
    preview_range: tuple[float, float] = (0.0, 0.0)
    runtime_index: int = -1
    texture_layers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GeometryProperties:
    """Editable MuJoCo contact parameters for one model geometry."""

    node_id: int
    friction: tuple[float, float, float]
    collision_type_mask: int
    collision_affinity_mask: int
    contact_dimension: int
    contact_priority: int
    margin: float
    gap: float
    solver_mix: float
    solver_reference: tuple[float, float] = (0.02, 1.0)
    solver_impedance: tuple[float, float, float, float, float] = (
        0.9,
        0.95,
        0.001,
        0.5,
        2.0,
    )
    adhesion: float = 0.0
    surface_velocity: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


@dataclass(frozen=True)
class GeometryAdvancedProperties:
    """Geometry properties that require rebuilding MuJoCo-derived constants."""

    node_id: int
    visual_group: int
    mass_mode: str
    mass: float
    density: float
    inertia_mode: str
    fluid_ellipsoid: bool
    fluid_coefficients: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class GeometryShapeProperties:
    """Editable geometry type and model-local mesh or height-field binding."""

    node_id: int
    type: str
    resource_name: str
    mesh_names: tuple[str, ...] = ()
    height_field_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointAdvancedProperties:
    """Joint properties that require rebuilding MuJoCo-derived constants."""

    joint_id: int
    group: int
    armature: float
    friction_loss: float
    reference: float
    spring_reference: float
    margin: float
    limit_solver_reference: tuple[float, float]
    limit_solver_impedance: tuple[float, float, float, float, float]
    friction_solver_reference: tuple[float, float]
    friction_solver_impedance: tuple[float, float, float, float, float]
    actuator_force_limit_mode: str
    actuator_force_range: tuple[float, float]
    actuator_gravity_compensation: bool


@dataclass(frozen=True)
class SiteProperties:
    """Editable site shape, visual group, and endpoint representation."""

    node_id: int
    type: str
    group: int
    use_from_to: bool
    from_to: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class BodyProperties:
    """Editable inertial and dynamic properties for one model body."""

    node_id: int
    inertia_mode: str
    mass: float
    inertial_position: tuple[float, float, float]
    inertial_quaternion: tuple[float, float, float, float]
    diagonal_inertia: tuple[float, float, float]
    full_inertia: tuple[float, float, float, float, float, float]
    gravity_compensation: float
    mocap: bool
    sleep_policy: str


@dataclass(frozen=True)
class KeyframeInfo:
    """A named MuJoCo state preset; unnamed motion frames get a stable fallback label."""

    keyframe_id: int
    name: str
    time: float
    model_id: int = -1


@dataclass(frozen=True)
class KeyframeProperties:
    """Complete editable state stored by one model-local keyframe."""

    keyframe_id: int
    model_id: int
    name: str
    time: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    act: tuple[float, ...]
    ctrl: tuple[float, ...]
    mocap_position: tuple[float, ...]
    mocap_quaternion: tuple[float, ...]


@dataclass(frozen=True)
class SensorInfo:
    """Sensor metadata describing a slice of the current sensor frame."""

    sensor_id: int
    name: str
    type: str
    data_adr: int
    dim: int


@dataclass(frozen=True)
class EqualityConstraintInfo:
    """Editable equality-constraint state."""

    constraint_id: int
    name: str
    type: str
    enabled: bool


@dataclass(frozen=True)
class VisualGroupInfo:
    """One independently switchable family of numbered visibility groups."""

    category: str
    visible: tuple[bool, ...]


@dataclass(frozen=True)
class PhysicsState:
    """Complete simulation state used by snapshots, reset, and reproduction tools."""

    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: float
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray


@dataclass(frozen=True)
class AdapterCaps:
    """Capabilities used by UI and command routing to expose supported operations.

    Adapters declare behavior here instead of relying on type checks. A capability set to
    ``True`` means the corresponding adapter method provides functional write-back.
    """

    name: str = "?"
    simulation: bool = False
    asset_loading: bool = False

    external_clock: bool = False

    write_pose: bool = False
    write_qpos: bool = False
    perturb: bool = False
    raycast: bool = False
    state_snapshots: bool = False
    contacts: bool = False
    model_cameras: bool = False
    keyframes: bool = False
    sensors: bool = False
    equality_constraints: bool = False
    visual_groups: bool = False
    reload: bool = False
    scene_authoring: bool = False
    scene_files: bool = False
    edit_history: bool = False
    model_composition: bool = False
    topology_editing: bool = False
    model_properties: bool = False
    model_assets: bool = False
    notes: tuple[str, ...] = ()


class JointVisualType(enum.IntEnum):
    """Debug-draw representation selected for each joint type."""

    FREE = 0
    BALL = 1
    SLIDE = 2
    HINGE = 3


class ActuatorVisualType(enum.IntEnum):
    """Debug-draw representation selected for actuator transmissions."""

    SLIDE = 0
    HINGE = 1
    BALL = 2
    FREE = 3
    SPHERE = 4
    ELLIPSOID = 5
    CAPSULE = 6
    CYLINDER = 7
    BOX = 8


class BvhType(enum.IntEnum):
    """Bounding-volume hierarchy source used by diagnostic overlays."""

    BODY = 0
    FLEX = 1
    MESH = 2
    OCTREE = 3


@dataclass(frozen=True)
class DiagnosticSource:
    """Stable metadata required to construct physics diagnostic overlays."""

    joint_types: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
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

    actuator_visual_types: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    actuator_visual_actuators: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    actuator_visual_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    slider_crank_actuators: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    slider_crank_width: float = 0.0
    slider_crank_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.4, 0.8, 1.0], np.float32)
    )
    slider_crank_broken_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 1.0], np.float32)
    )
    camera_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.8, 1.0, 1.0], np.float32)
    )
    light_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.85, 0.3, 1.0], np.float32)
    )
    rangefinder_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.4, 0.2, 1.0], np.float32)
    )
    rangefinder_normal_length: float = 0.0
    constraint_radius: float = 0.0
    constraint_connect_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.8, 1.0], np.float32)
    )
    constraint_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.9, 0.0, 0.0, 1.0], np.float32)
    )
    contact_point_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.3, 0.1, 1.0], np.float32)
    )
    contact_force_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.9, 0.3, 1.0], np.float32)
    )
    contact_friction_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.9, 0.5, 0.1, 1.0], np.float32)
    )
    contact_force_scale: float = 1.0
    autoconnect_width: float = 0.0
    autoconnect_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.8, 1.0], np.float32)
    )
    bvh_type: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    bvh_depth: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    bvh_leaf: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    bvh_active_highlight: bool = False
    bvh_rgba: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0, 0.5], np.float32))
    bvh_active_rgba: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.5], np.float32)
    )
    bvh_control_count: int = 0
    bvh_control_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0], np.float32)
    )


@dataclass
class DiagnosticFrame:
    """Dynamic positions, transforms, and visibility for diagnostic overlays."""

    joint_xpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    joint_xaxis: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    subtree_com: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    body_xipos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    body_ximat: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    actuator_xpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    actuator_xmat: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    slider_crank_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    slider_crank_broken: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    autoconnect_segments: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2, 3), np.float32)
    )
    rangefinder_starts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    rangefinder_ends: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    rangefinder_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    rangefinder_lines: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    rangefinder_points: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    rangefinder_normal_arrows: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    constraint_starts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    constraint_ends: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    constraint_visible: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    bvh_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    bvh_matrices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    bvh_sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    bvh_active: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    bvh_control_segments: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2, 3), np.float32)
    )


@dataclass
class FrameNeeds:
    """Requests optional dynamic arrays for one adapter frame.

    Consumers merge their requirements before calling :meth:`SceneAdapter.frame`. Adapters can
    avoid work and allocation for every field left ``False``.
    """

    poses: bool = True
    qpos: bool = False
    qvel: bool = False
    contacts: bool = False
    tendons: bool = False
    actuator: bool = False
    sensors: bool = False
    deformables: bool = False
    joint_frames: bool = False
    diagnostics: bool = False
    islands: bool = False
    bvh: bool = False

    def merge(self, other: FrameNeeds) -> FrameNeeds:
        """Return the union of two frame requirement sets."""
        return FrameNeeds(
            poses=self.poses or other.poses,
            qpos=self.qpos or other.qpos,
            qvel=self.qvel or other.qvel,
            contacts=self.contacts or other.contacts,
            tendons=self.tendons or other.tendons,
            actuator=self.actuator or other.actuator,
            sensors=self.sensors or other.sensors,
            deformables=self.deformables or other.deformables,
            joint_frames=self.joint_frames or other.joint_frames,
            diagnostics=self.diagnostics or other.diagnostics,
            islands=self.islands or other.islands,
            bvh=self.bvh or other.bvh,
        )

    @staticmethod
    def none() -> FrameNeeds:
        """Return a request with every optional field disabled."""
        return FrameNeeds(poses=False)


@dataclass
class SceneFrame:
    """Dynamic scene data for one simulation or authored-scene frame.

    Arrays correspond to indices stored in :class:`SceneSource`. Optional arrays are ``None``
    when the matching :class:`FrameNeeds` flag was disabled or the adapter lacks that feature.
    Adapters may reuse these arrays between calls; consumers copy data that must outlive a frame.
    """

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

    contacts: np.ndarray | None = None  # (C, 7): position(3), normal(3), force magnitude
    contact_forces: np.ndarray | None = None  # (C, 2, 3): normal and friction force
    contact_island_rgba: np.ndarray | None = None
    tendon_segments: np.ndarray | None = None  # (W, 2, 3) f32
    tendon_ids: np.ndarray | None = None
    tendon_widths: np.ndarray | None = None
    tendon_island_rgba: np.ndarray | None = None
    sensors: np.ndarray | None = None
    equality_enabled: np.ndarray | None = None
    mesh_updates: dict[MeshKey, MeshUpdate] | None = None
    flex_vertices: np.ndarray | None = None
    flex_island_rgba: np.ndarray | None = None
    island_rgba: np.ndarray | None = None
    diagnostics: DiagnosticFrame | None = None

    debug_commands: tuple[dict, ...] | None = None

    lights: LightSet | None = None
    cameras: tuple[CameraView, ...] | None = None


@dataclass
class SceneSource:
    """Stable scene structure uploaded when ``structure_revision`` changes.

    The source owns meshes, textures, materials, hierarchy nodes, instance metadata, cameras,
    lights, and diagnostic descriptions. Per-frame transforms and simulation values live in
    :class:`SceneFrame`.
    """

    meshes: dict[MeshKey, MeshData] = field(default_factory=dict)
    dynamic_meshes: frozenset[MeshKey] = frozenset()

    textures: dict[str, TextureData] = field(default_factory=dict)
    materials: list[Material] = field(default_factory=list)

    geom_mesh: list[MeshKey] = field(default_factory=list)
    geom_convex_mesh: list[MeshKey] = field(default_factory=list)
    geom_material: list[int] = field(default_factory=list)
    geom_size: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    geom_rgba: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    geom_object_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))
    geom_body: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    geom_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    geom_pose_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    geom_visual: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    geom_static: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    instance_island_body: np.ndarray = field(default_factory=lambda: np.full(0, -1, np.int32))

    geom_node: np.ndarray = field(default_factory=lambda: np.full(0, -1, np.int32))

    geom_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), np.float32))

    geom_infinite_plane: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    body_names: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()
    geom_names: tuple[str, ...] = ()
    site_names: tuple[str, ...] = ()
    camera_names: tuple[str, ...] = ()
    light_names: tuple[str, ...] = ()
    tendon_names: tuple[str, ...] = ()
    actuator_names: tuple[str, ...] = ()
    constraint_names: tuple[str, ...] = ()
    flex_names: tuple[str, ...] = ()

    flex_vertex_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    flex_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    flex_vertex_rgba: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    flex_edge_rgba: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    flex_vertex_ranges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    flex_vertex_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    flex_edge_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    debug_frame_length: float = 0.1

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
    cameras: tuple[CameraView, ...] = ()
    skybox: str | None = None
    shadow_clip: float = 1.0

    scene_extent: float = 1.0
    scene_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    nodes: list[SceneNode] = field(default_factory=list)
    shading_model: ShadingModel = ShadingModel.LINEAR

    # Appended to preserve the positional constructor used by older adapters.
    geom_segmentation: np.ndarray = field(default_factory=lambda: np.full((0, 2), -1, np.int32))

    @property
    def instance_count(self) -> int:
        """Return the number of render instances."""
        return len(self.geom_mesh)


class SceneAdapterBase:
    """Default implementation of the scene adapter contract.

    A custom adapter must implement :meth:`scene_source`, :meth:`frame`, and
    :attr:`structure_revision`. Simulation adapters also implement :meth:`step` and
    :meth:`reset`. Optional editor and physics operations return an unsupported result until the
    adapter advertises and implements the matching :class:`AdapterCaps` field.
    """

    caps = AdapterCaps(name="custom")

    @property
    def structure_revision(self) -> int:
        """Return a monotonically increasing stable-structure revision."""
        return 0

    def load(self, path: Path) -> None:
        """Replace the current source with the model at ``path``."""
        raise RuntimeError(f"{self.caps.name} does not support asset loading")

    def reload(self) -> None:
        """Reload the current file-backed source."""
        raise RuntimeError(f"{self.caps.name} does not support reload")

    def new_scene(self) -> None:
        """Replace the current source with an empty authored scene."""
        raise RuntimeError(f"{self.caps.name} does not support scene files")

    def open_scene(self, path: Path) -> None:
        """Open an authored scene document."""
        raise RuntimeError(f"{self.caps.name} does not support scene files")

    def save_scene(self, path: Path, options: SceneSaveOptions | None = None) -> None:
        """Save the current authored scene document."""
        raise RuntimeError(f"{self.caps.name} does not support scene files")

    def current_pose_modified(self) -> bool:
        """Return whether dynamic pose state differs from its authored state."""
        return False

    def export_mjcf(
        self,
        path: Path,
        source: SceneSource,
        frame: SceneFrame,
        options: SceneSaveOptions | None = None,
    ) -> Path:
        """Export the current source and frame as a portable MJCF document."""
        raise RuntimeError(f"{self.caps.name} does not support MJCF export")

    @property
    def resource_roots(self) -> tuple[Path, ...]:
        """Return directories searched for document resources."""
        return ()

    def add_resource_root(self, path: Path) -> bool:
        """Append a resource search directory."""
        return False

    def remove_resource_root(self, path: Path) -> bool:
        """Remove a resource search directory."""
        return False

    def set_resource_roots(self, paths: tuple[Path, ...]) -> None:
        """Replace the ordered resource search directories."""
        pass

    def capture_edit_state(self) -> object | None:
        """Capture adapter-owned state for undo and redo."""
        return None

    def restore_edit_state(self, state: object) -> bool:
        """Restore a state returned by :meth:`capture_edit_state`."""
        return False

    def scene_models(self) -> tuple[SceneModelInfo, ...]:
        """Return file-backed models composed into the current scene."""
        return ()

    def add_scene_model(self, path: Path, position, rotation) -> int:
        """Add a model with a world transform and return its model ID."""
        return -1

    def remove_scene_model(self, model_id: int) -> bool:
        """Remove a composed model by ID."""
        return False

    def set_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        """Set the world transform of a composed model."""
        return False

    def preview_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        """Set a transient render-only model transform without changing topology."""
        return False

    def clear_scene_model_transform_preview(self, model_id: int) -> bool:
        """Discard a transient model transform preview."""
        return False

    def add_model_element(self, parent_node_id: int, element_type: str, name: str) -> int:
        """Add a supported topology element below a hierarchy node."""
        return -1

    def duplicate_model_element(self, node_id: int) -> int:
        """Duplicate a topology element or body subtree by hierarchy node ID."""
        return -1

    def remove_model_element(self, node_id: int) -> bool:
        """Remove a topology element by hierarchy node ID."""
        return False

    def rename_model_element(self, node_id: int, name: str) -> bool:
        """Rename a topology element by hierarchy node ID."""
        return False

    def apply_model_edit_batch(self, edits: tuple[ModelEdit, ...]) -> tuple[int, ...]:
        """Apply model-element topology edits atomically and return per-edit node IDs."""
        return ()

    def scene_model_xml(self, model_id: int) -> str | None:
        """Return the normalized editable MJCF text for one model."""
        return None

    def scene_model_source(self, model_id: int) -> str | None:
        """Return the original source text for one model."""
        return None

    def set_scene_model_xml(self, model_id: int, xml: str) -> bool:
        """Compile and apply replacement MJCF text for one model."""
        return False

    def model_components(self, model_id: int, category: str) -> tuple[ModelComponentInfo, ...]:
        """Return editable model-level component or custom declarations in a category."""
        return ()

    def model_component_presets(self, model_id: int, category: str) -> tuple[str, ...]:
        """Return supported declaration subtypes for a model category."""
        return ()

    def add_model_component(self, model_id: int, category: str, subtype: str, name: str) -> int:
        """Add a model-level declaration and return its component ID."""
        return -1

    def update_model_component(
        self,
        model_id: int,
        category: str,
        component_id: int,
        name: str,
        fields: tuple[tuple[str, str], ...],
        path: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> bool:
        """Replace the editable fields and path of a model-level declaration."""
        return False

    def remove_model_component(self, model_id: int, category: str, component_id: int) -> bool:
        """Remove a model-level declaration."""
        return False

    def model_assets(self, model_id: int) -> tuple[ModelAssetInfo, ...]:
        """Return the model-local asset inventory and reference summary."""
        return ()

    def import_model_asset(
        self,
        model_id: int,
        asset_type: str,
        path: Path,
        name: str,
        fields: tuple[tuple[str, str], ...] = (),
    ) -> bool:
        """Import one file-backed model asset without binding it to an element."""
        return False

    def set_height_field_size(
        self,
        model_id: int,
        name: str,
        size: tuple[float, float, float, float],
    ) -> bool:
        """Set physical dimensions for one model-local height field."""
        return False

    def rename_model_asset(self, model_id: int, asset_type: str, name: str, new_name: str) -> bool:
        """Rename one asset and repair its model-local references."""
        return False

    def duplicate_model_asset(
        self, model_id: int, asset_type: str, name: str, new_name: str
    ) -> bool:
        """Duplicate one asset while keeping the same external source."""
        return False

    def replace_model_asset_file(
        self, model_id: int, asset_type: str, name: str, path: Path
    ) -> bool:
        """Replace the external source of one file-backed model asset."""
        return False

    def remove_model_asset(self, model_id: int, asset_type: str, name: str) -> bool:
        """Remove one unreferenced model asset."""
        return False

    def reset(self) -> None:
        """Restore the adapter's initial dynamic state."""

    def step(self, count: int = 1) -> None:
        """Advance simulation state by ``count`` fixed steps."""

    def set_paused(self, paused: bool) -> bool:
        """Set adapter-owned pause state and report whether it was accepted."""
        return True

    def prepare_frame(self, needs: FrameNeeds) -> bool:
        """Materialize optional stable data and report whether structure changed."""
        return False

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        """Return the latest dynamic frame containing the requested optional data."""
        raise NotImplementedError

    def scene_source(self) -> SceneSource:
        """Return stable scene structure for the current revision."""
        raise NotImplementedError

    def nodes(self) -> list[SceneNode]:
        """Return hierarchy nodes in stable node order."""
        return self.scene_source().nodes

    def joints(self) -> list[JointInfo]:
        """Return editable joint metadata in stable model order."""
        return []

    def actuators(self) -> list[ActuatorInfo]:
        """Return editable actuator metadata in stable model order."""
        return []

    def cameras(self) -> list[CameraInfo]:
        """Return model camera identities in stable model order."""
        return []

    def keyframes(self) -> list[KeyframeInfo]:
        """Return named simulation keyframes."""
        return []

    def sensors(self) -> list[SensorInfo]:
        """Return sensor metadata for the current source."""
        return []

    def equality_constraints(self) -> list[EqualityConstraintInfo]:
        """Return editable equality-constraint metadata."""
        return []

    def load_keyframe(self, keyframe_id: int) -> bool:
        """Replace dynamic state with a model keyframe."""
        return False

    def keyframe_properties(self, keyframe_id: int) -> KeyframeProperties | None:
        """Return complete editable state for one model-local keyframe."""
        return None

    def add_model_keyframe(self, model_id: int, name: str) -> int:
        """Capture current model-local state in a new keyframe."""
        return -1

    def set_keyframe_properties(self, properties: KeyframeProperties) -> bool:
        """Replace one model-local keyframe."""
        return False

    def remove_model_keyframe(self, keyframe_id: int) -> bool:
        """Remove one model-local keyframe."""
        return False

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Resolve a model camera into a backend-neutral world view."""
        return None

    def visual_groups(self) -> tuple[VisualGroupInfo, ...]:
        """Return numbered visibility groups exposed by the source."""
        return ()

    def set_visual_group(self, category: str, group: int, visible: bool) -> bool:
        """Set one numbered visibility group."""
        return False

    def set_qpos(self, index: int, value: float) -> bool:
        """Set one generalized position coordinate."""
        return False

    def set_qpos_batch(self, indices: np.ndarray, values: np.ndarray) -> bool:
        """Atomically set generalized position coordinates."""
        return False

    def set_joint_properties(
        self,
        joint_id: int,
        axis: np.ndarray,
        limited: bool,
        value_range: tuple[float, float],
        damping: float,
        stiffness: float,
    ) -> bool:
        """Set authored numeric properties for one model joint."""
        return False

    def joint_advanced_properties(self, joint_id: int) -> JointAdvancedProperties | None:
        """Return joint properties backed by rebuilt MuJoCo constants."""
        return None

    def set_joint_advanced_properties(self, properties: JointAdvancedProperties) -> bool:
        """Set joint properties that require rebuilding MuJoCo constants."""
        return False

    def site_properties(self, node_id: int) -> SiteProperties | None:
        """Return editable shape and endpoint properties for one model site."""
        return None

    def set_site_properties(self, properties: SiteProperties) -> bool:
        """Set shape and endpoint properties for one model site."""
        return False

    def geometry_properties(self, node_id: int) -> GeometryProperties | None:
        """Return editable contact parameters for one model geometry."""
        return None

    def set_geometry_properties(self, properties: GeometryProperties) -> bool:
        """Set authored contact parameters for one model geometry."""
        return False

    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None:
        """Return geometry properties backed by rebuilt MuJoCo constants."""
        return None

    def set_geometry_advanced_properties(self, properties: GeometryAdvancedProperties) -> bool:
        """Set geometry properties that require rebuilding MuJoCo constants."""
        return False

    def geometry_shape_properties(self, node_id: int) -> GeometryShapeProperties | None:
        """Return geometry type and model-local resource choices."""
        return None

    def set_geometry_shape(self, node_id: int, geom_type: str, resource_name: str) -> bool:
        """Set a geometry type and optional model-local resource binding."""
        return False

    def import_model_geometry_resource(
        self, node_id: int, resource_type: str, path: Path, name: str
    ) -> bool:
        """Import and bind one mesh or height-field resource atomically."""
        return False

    def body_properties(self, node_id: int) -> BodyProperties | None:
        """Return editable inertial and dynamic properties for one model body."""
        return None

    def set_body_properties(self, properties: BodyProperties) -> bool:
        """Set authored inertial and dynamic properties for one model body."""
        return False

    def model_material_indices(self, model_id: int) -> tuple[int, ...]:
        """Return render material indices owned by one editable model."""
        return ()

    def model_texture_names(self, model_id: int) -> tuple[str, ...]:
        """Return compiled texture names owned by one editable model."""
        return ()

    def create_model_material(self, model_id: int, name: str) -> int:
        """Create one unbound model-local material and return its render index."""
        return -1

    def add_model_material(self, node_id: int, name: str, copy_from: int = -1) -> int:
        """Create and bind a model-local material to one geometry or site."""
        return -1

    def import_model_texture(
        self,
        model_id: int,
        path: Path,
        name: str,
        material_index: int = -1,
        texture_type: str = "2d",
    ) -> bool:
        """Import one 2D, cube, or skybox image and optionally bind a 2D image."""
        return False

    def set_geometry_material(self, node_id: int, material_index: int) -> bool:
        """Bind a model-local material, or -1 for inline appearance."""
        return False

    def set_equality_enabled(self, constraint_id: int, enabled: bool) -> bool:
        """Enable or disable an equality constraint."""
        return False

    def set_ctrl(self, index: int, value: float) -> bool:
        """Set one actuator control coordinate."""
        return False

    def set_pose(self, node_id: int, position, rotation) -> bool:
        """Write a world pose to a posable hierarchy node."""
        return False

    def capture_state(self) -> PhysicsState | None:
        """Capture a complete restorable physics state when supported."""
        return None

    def restore_state(self, state: PhysicsState) -> bool:
        """Restore a previously captured physics state."""
        return False

    def set_light(self, light_index: int, light) -> bool:
        """Edit a Mojive scene light.

        Lights belong to ``SceneSource``, not to the physics backend.  Adapters
        with a native light representation may override this to write through;
        the default keeps custom and render-only adapters editable.
        """
        source = self.scene_source()
        i = int(light_index)
        if not 0 <= i < len(source.lights.lights):
            return False
        lights = list(source.lights.lights)
        lights[i] = light
        source.lights = replace(source.lights, lights=tuple(lights))
        return True

    def set_environment(self, environment: Environment) -> bool:
        """Replace the Mojive environment values exposed by this adapter."""
        source = self.scene_source()
        source.lights = source.lights.with_environment(environment)
        return True

    def set_skybox(self, texture: str | None) -> bool:
        """Select one cube texture as the rendered environment."""
        source = self.scene_source()
        if texture is not None:
            item = source.textures.get(texture)
            if item is None or item.type not in (TextureType.CUBE, TextureType.SKYBOX):
                return False
        source.skybox = texture
        return True

    def set_material(self, material_index: int, material: Material) -> bool:
        """Replace one Mojive material by source index."""
        source = self.scene_source()
        i = int(material_index)
        if not 0 <= i < len(source.materials):
            return False
        source.materials[i] = material
        return True

    def set_geometry_color(self, node_id: int, rgba: np.ndarray) -> bool:
        """Set the instance color of all geometry owned by a hierarchy node."""
        source = self.scene_source()
        instances = np.flatnonzero(source.geom_node == int(node_id))
        if not len(instances):
            return False
        source.geom_rgba[instances] = np.asarray(rgba, np.float32)
        return True

    def set_geometry_size(self, node_id: int, size: np.ndarray) -> bool:
        """Set authored primitive dimensions for a geometry or site node."""
        return False

    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool:
        """Write a backend-neutral view to a model camera."""
        return False

    def add_scene_object(
        self,
        shape: MeshShape | MeshKey,
        name: str,
        size,
        position,
        rotation,
        color,
        material: Material,
    ) -> int:
        """Create an authored render object and return its object ID."""
        return -1

    def remove_scene_object(self, object_id: int) -> bool:
        """Remove an authored render object."""
        return False

    def add_scene_light(self, name: str, light: Light) -> int:
        """Create an authored light and return its light index."""
        return -1

    def remove_scene_light(self, light_id: int) -> bool:
        """Remove an authored light by source index."""
        return False

    def add_scene_camera(self, name: str, camera: CameraView) -> int:
        """Create an authored camera and return its camera index."""
        return -1

    def remove_scene_camera(self, camera_id: int) -> bool:
        """Remove an authored camera by source index."""
        return False

    def duplicate_scene_entity(self, object_id: int) -> int:
        """Duplicate an authored entity and return its new object ID."""
        return 0

    def remove_scene_entity(self, object_id: int) -> bool:
        """Remove an authored entity by selection object ID."""
        return False

    def rename_scene_entity(self, object_id: int, name: str) -> bool:
        """Rename an authored entity by selection object ID."""
        return False

    def apply_perturb(
        self, node_id: int, target_position: np.ndarray, target_rotation: np.ndarray, mode: str
    ) -> bool:
        """Apply a translation or rotation perturbation to a hierarchy node."""
        return False

    def clear_perturb(self) -> None:
        """Clear the active physical perturbation."""

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]:
        """Return the selected object ID and ray distance, or ``(0, inf)`` for no hit."""
        return (0, float("inf"))

    def camera_hint(self) -> CameraView | None:
        """Return the adapter's preferred initial editor camera."""
        return None

    def timestep(self) -> float:
        """Return the simulation step duration in seconds."""
        return 0.0

    def release(self) -> None:
        """Release adapter-owned resources."""


@runtime_checkable
class SceneProvider(Protocol):
    """Minimal read-only scene stream consumed by offscreen rendering.

    Structure revisions change after topology or resource updates. Frame buffers
    may be reused until the next ``frame`` call. Simulation, selection and authoring
    are optional extensions supplied by the full ``SceneAdapter`` interface.
    """

    @property
    def structure_revision(self) -> int: ...

    def scene_source(self) -> SceneSource: ...
    def frame(self, needs: FrameNeeds) -> SceneFrame: ...


@runtime_checkable
class SceneAdapter(SceneProvider, Protocol):
    """Complete editor interface, including capability-specific extensions.

    Derive from ``SceneAdapterBase`` to inherit defaults for unsupported editor
    operations. Read-only rendering consumers need only ``SceneProvider``.
    """

    caps: AdapterCaps

    def load(self, path: Path) -> None: ...
    def reload(self) -> None: ...
    def new_scene(self) -> None: ...
    def open_scene(self, path: Path) -> None: ...
    def save_scene(self, path: Path, options: SceneSaveOptions | None = None) -> None: ...
    def current_pose_modified(self) -> bool: ...
    def export_mjcf(
        self,
        path: Path,
        source: SceneSource,
        frame: SceneFrame,
        options: SceneSaveOptions | None = None,
    ) -> Path: ...
    @property
    def resource_roots(self) -> tuple[Path, ...]: ...
    def add_resource_root(self, path: Path) -> bool: ...
    def remove_resource_root(self, path: Path) -> bool: ...
    def set_resource_roots(self, paths: tuple[Path, ...]) -> None: ...
    def capture_edit_state(self) -> object | None: ...
    def restore_edit_state(self, state: object) -> bool: ...
    def scene_models(self) -> tuple[SceneModelInfo, ...]: ...
    def add_scene_model(self, path: Path, position, rotation) -> int: ...
    def remove_scene_model(self, model_id: int) -> bool: ...
    def set_scene_model_transform(self, model_id: int, position, rotation) -> bool: ...
    def preview_scene_model_transform(self, model_id: int, position, rotation) -> bool: ...
    def clear_scene_model_transform_preview(self, model_id: int) -> bool: ...
    def add_model_element(self, parent_node_id: int, element_type: str, name: str) -> int: ...
    def duplicate_model_element(self, node_id: int) -> int: ...
    def remove_model_element(self, node_id: int) -> bool: ...
    def rename_model_element(self, node_id: int, name: str) -> bool: ...
    def apply_model_edit_batch(self, edits: tuple[ModelEdit, ...]) -> tuple[int, ...]: ...
    def scene_model_xml(self, model_id: int) -> str | None: ...
    def scene_model_source(self, model_id: int) -> str | None: ...
    def set_scene_model_xml(self, model_id: int, xml: str) -> bool: ...
    def model_components(self, model_id: int, category: str) -> tuple[ModelComponentInfo, ...]: ...
    def model_component_presets(self, model_id: int, category: str) -> tuple[str, ...]: ...
    def add_model_component(self, model_id: int, category: str, subtype: str, name: str) -> int: ...
    def update_model_component(
        self,
        model_id: int,
        category: str,
        component_id: int,
        name: str,
        fields: tuple[tuple[str, str], ...],
        path: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> bool: ...
    def remove_model_component(self, model_id: int, category: str, component_id: int) -> bool: ...
    def model_assets(self, model_id: int) -> tuple[ModelAssetInfo, ...]: ...
    def import_model_asset(
        self,
        model_id: int,
        asset_type: str,
        path: Path,
        name: str,
        fields: tuple[tuple[str, str], ...] = (),
    ) -> bool: ...
    def set_height_field_size(
        self,
        model_id: int,
        name: str,
        size: tuple[float, float, float, float],
    ) -> bool: ...
    def rename_model_asset(
        self, model_id: int, asset_type: str, name: str, new_name: str
    ) -> bool: ...
    def duplicate_model_asset(
        self, model_id: int, asset_type: str, name: str, new_name: str
    ) -> bool: ...
    def replace_model_asset_file(
        self, model_id: int, asset_type: str, name: str, path: Path
    ) -> bool: ...
    def remove_model_asset(self, model_id: int, asset_type: str, name: str) -> bool: ...
    def reset(self) -> None: ...
    def step(self, count: int = 1) -> None: ...
    def set_paused(self, paused: bool) -> bool: ...
    def nodes(self) -> list[SceneNode]: ...
    def joints(self) -> list[JointInfo]: ...
    def actuators(self) -> list[ActuatorInfo]: ...
    def cameras(self) -> list[CameraInfo]: ...
    def keyframes(self) -> list[KeyframeInfo]: ...
    def sensors(self) -> list[SensorInfo]: ...
    def equality_constraints(self) -> list[EqualityConstraintInfo]: ...
    def load_keyframe(self, keyframe_id: int) -> bool: ...
    def keyframe_properties(self, keyframe_id: int) -> KeyframeProperties | None: ...
    def add_model_keyframe(self, model_id: int, name: str) -> int: ...
    def set_keyframe_properties(self, properties: KeyframeProperties) -> bool: ...
    def remove_model_keyframe(self, keyframe_id: int) -> bool: ...
    def camera_view(self, camera_id: int) -> CameraView | None: ...
    def visual_groups(self) -> tuple[VisualGroupInfo, ...]: ...
    def set_visual_group(self, category: str, group: int, visible: bool) -> bool: ...
    def set_qpos(self, index: int, value: float) -> bool: ...
    def set_qpos_batch(self, indices: np.ndarray, values: np.ndarray) -> bool: ...
    def set_joint_properties(
        self,
        joint_id: int,
        axis: np.ndarray,
        limited: bool,
        value_range: tuple[float, float],
        damping: float,
        stiffness: float,
    ) -> bool: ...
    def joint_advanced_properties(self, joint_id: int) -> JointAdvancedProperties | None: ...
    def set_joint_advanced_properties(self, properties: JointAdvancedProperties) -> bool: ...
    def site_properties(self, node_id: int) -> SiteProperties | None: ...
    def set_site_properties(self, properties: SiteProperties) -> bool: ...
    def geometry_properties(self, node_id: int) -> GeometryProperties | None: ...
    def set_geometry_properties(self, properties: GeometryProperties) -> bool: ...
    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None: ...
    def set_geometry_advanced_properties(self, properties: GeometryAdvancedProperties) -> bool: ...
    def geometry_shape_properties(self, node_id: int) -> GeometryShapeProperties | None: ...
    def set_geometry_shape(self, node_id: int, geom_type: str, resource_name: str) -> bool: ...
    def import_model_geometry_resource(
        self, node_id: int, resource_type: str, path: Path, name: str
    ) -> bool: ...
    def body_properties(self, node_id: int) -> BodyProperties | None: ...
    def set_body_properties(self, properties: BodyProperties) -> bool: ...
    def model_material_indices(self, model_id: int) -> tuple[int, ...]: ...
    def model_texture_names(self, model_id: int) -> tuple[str, ...]: ...
    def create_model_material(self, model_id: int, name: str) -> int: ...
    def add_model_material(self, node_id: int, name: str, copy_from: int = -1) -> int: ...
    def import_model_texture(
        self,
        model_id: int,
        path: Path,
        name: str,
        material_index: int = -1,
        texture_type: str = "2d",
    ) -> bool: ...
    def set_geometry_material(self, node_id: int, material_index: int) -> bool: ...
    def set_equality_enabled(self, constraint_id: int, enabled: bool) -> bool: ...
    def set_ctrl(self, index: int, value: float) -> bool: ...
    def set_pose(self, node_id: int, position, rotation) -> bool: ...
    def capture_state(self) -> PhysicsState | None: ...
    def restore_state(self, state: PhysicsState) -> bool: ...
    def set_light(self, light_index: int, light) -> bool: ...
    def set_environment(self, environment: Environment) -> bool: ...
    def set_skybox(self, texture: str | None) -> bool: ...
    def set_material(self, material_index: int, material: Material) -> bool: ...
    def set_geometry_color(self, node_id: int, rgba: np.ndarray) -> bool: ...
    def set_geometry_size(self, node_id: int, size: np.ndarray) -> bool: ...
    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool: ...
    def add_scene_object(
        self,
        shape: MeshShape | MeshKey,
        name: str,
        size,
        position,
        rotation,
        color,
        material: Material,
    ) -> int: ...
    def remove_scene_object(self, object_id: int) -> bool: ...
    def add_scene_light(self, name: str, light: Light) -> int: ...
    def remove_scene_light(self, light_id: int) -> bool: ...
    def add_scene_camera(self, name: str, camera: CameraView) -> int: ...
    def remove_scene_camera(self, camera_id: int) -> bool: ...
    def duplicate_scene_entity(self, object_id: int) -> int: ...
    def remove_scene_entity(self, object_id: int) -> bool: ...
    def rename_scene_entity(self, object_id: int, name: str) -> bool: ...
    def apply_perturb(
        self, node_id: int, target_position: np.ndarray, target_rotation: np.ndarray, mode: str
    ) -> bool: ...
    def clear_perturb(self) -> None: ...
    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]: ...

    def camera_hint(self) -> CameraView | None: ...
    def timestep(self) -> float: ...
    def release(self) -> None: ...
