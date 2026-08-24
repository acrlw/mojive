"""Scene adapter interfaces and built-in adapters."""

from typing import TYPE_CHECKING

from .base import (
    ActuatorInfo,
    ActuatorVisualKind,
    AdapterCaps,
    CameraInfo,
    DiagnosticFrame,
    DiagnosticSource,
    EqualityConstraintInfo,
    FrameNeeds,
    IkOptions,
    IkResult,
    JointInfo,
    JointVisualKind,
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
    "ActuatorVisualKind",
    "AdapterCaps",
    "CameraInfo",
    "DiagnosticFrame",
    "DiagnosticSource",
    "EqualityConstraintInfo",
    "FrameNeeds",
    "IkOptions",
    "IkResult",
    "JointInfo",
    "JointVisualKind",
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
