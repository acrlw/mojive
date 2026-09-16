"""Scene adapter interfaces and built-in adapters."""

from typing import TYPE_CHECKING

from .base import (
    ActuatorInfo,
    ActuatorVisualType,
    AdapterCaps,
    CameraInfo,
    DiagnosticFrame,
    DiagnosticSource,
    DocumentCheckpoint,
    EqualityConstraintInfo,
    FrameNeeds,
    JointInfo,
    JointVisualType,
    KeyframeCatalog,
    KeyframeEditing,
    KeyframeInfo,
    KeyframePlayback,
    ModelAssets,
    ModelComposition,
    ModelEditing,
    ModelKeyframes,
    ModelProperties,
    ModelTopology,
    PhysicsState,
    PointPerturbation,
    SceneAdapter,
    SceneAdapterBase,
    SceneAppearance,
    SceneAuthoring,
    SceneDocuments,
    SceneEditing,
    SceneFrame,
    SceneInspection,
    SceneModelInfo,
    SceneNode,
    ScenePersistence,
    SceneProvider,
    SceneRuntime,
    SceneSource,
    SensorInfo,
    SimulationAccess,
    SimulationControl,
    VisualGroupInfo,
)
from .worlds import WorldInstances

if TYPE_CHECKING:
    from .mujoco import MuJoCoAdapter
    from .workspace import WorkspaceAdapter


def __getattr__(name: str):
    if name == "MuJoCoAdapter":
        from .mujoco import MuJoCoAdapter

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
    "DocumentCheckpoint",
    "EqualityConstraintInfo",
    "FrameNeeds",
    "JointInfo",
    "JointVisualType",
    "KeyframeCatalog",
    "KeyframeEditing",
    "KeyframeInfo",
    "KeyframePlayback",
    "ModelAssets",
    "ModelComposition",
    "ModelEditing",
    "ModelKeyframes",
    "ModelProperties",
    "ModelTopology",
    "MuJoCoAdapter",
    "PhysicsState",
    "PointPerturbation",
    "SceneAdapter",
    "SceneAdapterBase",
    "SceneAppearance",
    "SceneAuthoring",
    "SceneDocuments",
    "SceneEditing",
    "SceneFrame",
    "SceneInspection",
    "SceneModelInfo",
    "SceneNode",
    "ScenePersistence",
    "SceneProvider",
    "SceneRuntime",
    "SceneSource",
    "SensorInfo",
    "SimulationAccess",
    "SimulationControl",
    "VisualGroupInfo",
    "WorkspaceAdapter",
    "WorldInstances",
]
