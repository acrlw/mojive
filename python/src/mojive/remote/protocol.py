"""Remote: protocol."""

from __future__ import annotations

import contextlib
import socket
from dataclasses import dataclass
from multiprocessing.connection import Connection

from mojive.adapters.base import (
    ActuatorInfo,
    AdapterCaps,
    BodyProperties,
    CameraInfo,
    EqualityConstraintInfo,
    GeometryAdvancedProperties,
    GeometryProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    JointInfo,
    KeyframeInfo,
    SceneFrame,
    SceneNode,
    SceneSource,
    SensorInfo,
    SiteProperties,
    VisualGroupInfo,
)
from mojive.types import CameraView

DEFAULT_PORT = 47650
AUTHKEY = b"mojive-local"
STREAM_PROTOCOL_VERSION = 1
_REMOTE_REQUIREMENTS = {
    **dict.fromkeys(("pause", "play"), "clock_control"),
    "step": "simulation",
    **dict.fromkeys(("reset", "light", "environment", "skybox", "material", "clear_perturb"), ""),
    "reload": "reload",
    "keyframe": "keyframes",
    "raycast": "raycast",
    "visual_group": "visual_groups",
    "pose": "write_pose",
    "perturb": "perturb",
    **dict.fromkeys(("qpos", "qpos_batch"), "write_qpos"),
    "equality": "equality_constraints",
    **dict.fromkeys(("ctrl", "ctrl_vector"), "write_ctrl"),
    **dict.fromkeys(
        (
            "joint_properties",
            "joint_advanced_properties",
            "site_properties",
            "geometry_properties",
            "geometry_advanced_properties",
            "geometry_shape",
            "body_properties",
        ),
        "model_properties",
    ),
    **dict.fromkeys(("geometry_color", "geometry_size", "scene_camera"), ""),
    **dict.fromkeys(
        (
            "add_scene_object",
            "remove_scene_object",
            "add_scene_light",
            "remove_scene_light",
            "add_scene_camera",
            "remove_scene_camera",
            "duplicate_scene_entity",
            "remove_scene_entity",
            "rename_scene_entity",
        ),
        "scene_authoring",
    ),
}


def remote_command_versions(caps: AdapterCaps) -> tuple[tuple[str, int], ...]:
    """Advertise only wire commands supported by this publisher's adapter."""
    return tuple(
        (name, 1)
        for name, feature in _REMOTE_REQUIREMENTS.items()
        if (not feature or caps.supports(feature))
        and not (
            name in {"pause", "play", "step", "reset"}
            and caps.simulation
            and not caps.clock_control
        )
    )


def _close_connection(connection: Connection) -> None:
    """Interrupt pending TCP reads/writes before releasing the Connection's descriptor."""
    with contextlib.suppress(OSError):
        transport = socket.socket(fileno=connection.fileno())
        try:
            transport.shutdown(socket.SHUT_RDWR)
        finally:
            # Connection owns the descriptor; the temporary socket only shuts it down.
            transport.detach()
    connection.close()


@dataclass(frozen=True)
class RemoteStructure:
    """Reliable stable structure sent when a scene revision changes."""

    structure_revision: int
    source: SceneSource
    caps: AdapterCaps
    nodes: list[SceneNode]
    joints: list[JointInfo]
    actuators: list[ActuatorInfo]
    cameras: list[CameraInfo]
    keyframes: list[KeyframeInfo]
    sensors: list[SensorInfo]
    equality_constraints: list[EqualityConstraintInfo]
    camera_hint: CameraView | None
    timestep: float
    visual_groups: tuple[VisualGroupInfo, ...] = ()
    geometry_properties: tuple[GeometryProperties, ...] = ()
    body_properties: tuple[BodyProperties, ...] = ()
    geometry_advanced_properties: tuple[GeometryAdvancedProperties, ...] = ()
    geometry_shape_properties: tuple[GeometryShapeProperties, ...] = ()
    joint_advanced_properties: tuple[JointAdvancedProperties, ...] = ()
    site_properties: tuple[SiteProperties, ...] = ()
    protocol_version: int = STREAM_PROTOCOL_VERSION
    command_versions: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RemoteFrame:
    """Latest-only dynamic scene frame and its debug commands."""

    frame_sequence: int
    frame: SceneFrame
    debug_commands: tuple[dict, ...] = ()
    structure_revision: int = -1


def snapshot_structure(session) -> RemoteStructure:
    """Capture adapter structure and metadata from a session."""
    return RemoteStructure(
        session.structure_generation,
        session.source,
        session.adapter.caps,
        session.nodes,
        session.joints,
        session.actuators,
        session.cameras,
        session.keyframes,
        session.sensor_infos,
        session.equality_constraints,
        session.camera_hint(),
        session.adapter.timestep(),
        session.visual_groups(),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.geometry_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.body_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.geometry_advanced_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.geometry_shape_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for joint in session.joints
            if (properties := session.joint_advanced_properties(joint.joint_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.site_properties(node.node_id)) is not None
        ),
        command_versions=remote_command_versions(session.adapter.caps),
    )
