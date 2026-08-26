"""Scene adapter interfaces and built-in adapters."""

from typing import TYPE_CHECKING

from .base import (
    ActuatorInfo,
    ActuatorVisualType,
    AdapterCaps,
    CameraInfo,
    DiagnosticFrame,
    DiagnosticSource,
    EqualityConstraintInfo,
    FrameNeeds,
    JointInfo,
    JointVisualType,
    KeyframeInfo,
    PhysicsState,
    SceneAdapter,
    SceneAdapterBase,
    SceneFrame,
    SceneModelInfo,
    SceneNode,
    SceneSource,
    SensorInfo,
    VisualGroupInfo,
)
from .workspace import WorkspaceAdapter

if TYPE_CHECKING:
    from .mujoco_adapter import MuJoCoAdapter


def __getattr__(name: str):
    if name != "MuJoCoAdapter":
        raise AttributeError(name)
    from .mujoco_adapter import MuJoCoAdapter

    globals()[name] = MuJoCoAdapter
    return MuJoCoAdapter


__all__ = [
    "ActuatorInfo",
    "ActuatorVisualType",
    "AdapterCaps",
    "CameraInfo",
    "DiagnosticFrame",
    "DiagnosticSource",
    "EqualityConstraintInfo",
    "FrameNeeds",
    "JointInfo",
    "JointVisualType",
    "KeyframeInfo",
    "MuJoCoAdapter",
    "PhysicsState",
    "SceneAdapter",
    "SceneAdapterBase",
    "SceneFrame",
    "SceneModelInfo",
    "SceneNode",
    "SceneSource",
    "SensorInfo",
    "VisualGroupInfo",
    "WorkspaceAdapter",
]
