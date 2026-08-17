from typing import TYPE_CHECKING

from .adapters.base import (
    ActuatorInfo,
    ActuatorVisualKind,
    AdapterCaps,
    CameraInfo,
    DiagnosticFrame,
    DiagnosticSource,
    EqualityConstraintInfo,
    FrameNeeds,
    JointInfo,
    JointVisualKind,
    KeyframeInfo,
    NodeKind,
    SceneAdapter,
    SceneAdapterBase,
    SceneFrame,
    SceneNode,
    SceneSource,
    SensorInfo,
    VisualGroupInfo,
)
from .adapters.conformance import ConformanceCheck, ConformanceReport, check_adapter
from .adapters.toy import ToyPhysicsAdapter
from .backends import make_adapter
from .composition import build_from_adapter, build_scene
from .recording import SnapshotWriter, VideoRecorder, read_snapshots
from .remote import RemoteSceneAdapter, SnapshotPublisher
from .render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from .render.debugdraw import DebugDraw, Layer, Occlusion
from .scene import Scene, SceneObject
from .types import (
    CameraView,
    Light,
    LightKind,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    MeshUpdate,
)

if TYPE_CHECKING:
    from .adapters.mujoco_adapter import MuJoCoAdapter


def __getattr__(name: str):
    """Keep the optional MuJoCo package out of physics-free imports."""
    if name == "MuJoCoAdapter":
        from .adapters.mujoco_adapter import MuJoCoAdapter

        value = MuJoCoAdapter
    elif name in {"audit_model", "visual_coverage"}:
        from . import mujoco_audit

        value = getattr(mujoco_audit, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "ActuatorInfo",
    "ActuatorVisualKind",
    "AdapterCaps",
    "CameraInfo",
    "CameraView",
    "ConformanceCheck",
    "ConformanceReport",
    "DebugDraw",
    "DebugView",
    "DiagnosticFrame",
    "DiagnosticSource",
    "EqualityConstraintInfo",
    "FrameMode",
    "FrameNeeds",
    "JointInfo",
    "JointVisualKind",
    "KeyframeInfo",
    "LabelMode",
    "Layer",
    "Light",
    "LightKind",
    "LightSet",
    "Material",
    "MeshData",
    "MeshKey",
    "MeshShape",
    "MeshUpdate",
    "MuJoCoAdapter",
    "NodeKind",
    "Occlusion",
    "RemoteSceneAdapter",
    "RenderFlag",
    "Scene",
    "SceneAdapter",
    "SceneAdapterBase",
    "SceneFrame",
    "SceneNode",
    "SceneObject",
    "SceneSource",
    "SensorInfo",
    "SnapshotPublisher",
    "SnapshotWriter",
    "ToyPhysicsAdapter",
    "VideoRecorder",
    "VisualGroupInfo",
    "audit_model",
    "build_from_adapter",
    "build_scene",
    "check_adapter",
    "make_adapter",
    "read_snapshots",
    "visual_coverage",
]

__version__ = "0.1.0"
