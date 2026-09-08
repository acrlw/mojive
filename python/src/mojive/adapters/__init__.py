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

if TYPE_CHECKING:
    from .mujoco_adapter import MuJoCoAdapter
    from .workspace import WorkspaceAdapter


def __getattr__(name: str):
    if name == "MuJoCoAdapter":
        from .mujoco_adapter import MuJoCoAdapter

        value = MuJoCoAdapter
    elif name == "WorkspaceAdapter":
        from .workspace import WorkspaceAdapter

        value = WorkspaceAdapter
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


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
