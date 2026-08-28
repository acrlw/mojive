"""Public package exports for forge-viewer."""

from typing import TYPE_CHECKING

from .adapters.base import (
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
    NodeType,
    PhysicsState,
    SceneAdapter,
    SceneAdapterBase,
    SceneFrame,
    SceneModelInfo,
    SceneNode,
    SceneSaveOptions,
    SceneSource,
    SensorInfo,
    VisualGroupInfo,
)
from .adapters.conformance import ConformanceCheck, ConformanceReport, check_adapter
from .adapters.toy import ToyPhysicsAdapter
from .backends import make_adapter
from .composition import build_from_adapter, build_scene, build_workspace
from .recording import SnapshotWriter, VideoRecorder, read_snapshots
from .remote import RemoteSceneAdapter, SnapshotPublisher
from .render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from .render.debugdraw import DebugDraw, Layer, Occlusion
from .scene import Scene, SceneLight, SceneObject
from .types import (
    CameraView,
    Environment,
    Light,
    LightSet,
    LightType,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    MeshUpdate,
    ShadingModel,
)

if TYPE_CHECKING:
    from .adapters.mujoco_adapter import MuJoCoAdapter
    from .renderer import Renderer


def __getattr__(name: str):
    """Keep the optional MuJoCo package out of physics-free imports."""
    if name == "MuJoCoAdapter":
        from .adapters.mujoco_adapter import MuJoCoAdapter

        value = MuJoCoAdapter
    elif name == "Renderer":
        from .renderer import Renderer

        value = Renderer
    elif name in {"audit_model", "schema_coverage", "visual_coverage"}:
        from . import mujoco_audit

        value = getattr(mujoco_audit, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "ActuatorInfo",
    "ActuatorVisualType",
    "AdapterCaps",
    "CameraInfo",
    "CameraView",
    "ConformanceCheck",
    "ConformanceReport",
    "DebugDraw",
    "DebugView",
    "DiagnosticFrame",
    "DiagnosticSource",
    "Environment",
    "EqualityConstraintInfo",
    "FrameMode",
    "FrameNeeds",
    "JointInfo",
    "JointVisualType",
    "KeyframeInfo",
    "LabelMode",
    "Layer",
    "Light",
    "LightSet",
    "LightType",
    "Material",
    "MeshData",
    "MeshKey",
    "MeshShape",
    "MeshUpdate",
    "MuJoCoAdapter",
    "NodeType",
    "Occlusion",
    "PhysicsState",
    "RemoteSceneAdapter",
    "RenderFlag",
    "Renderer",
    "Scene",
    "SceneAdapter",
    "SceneAdapterBase",
    "SceneFrame",
    "SceneLight",
    "SceneModelInfo",
    "SceneNode",
    "SceneObject",
    "SceneSaveOptions",
    "SceneSource",
    "SensorInfo",
    "ShadingModel",
    "SnapshotPublisher",
    "SnapshotWriter",
    "ToyPhysicsAdapter",
    "VideoRecorder",
    "VisualGroupInfo",
    "audit_model",
    "build_from_adapter",
    "build_scene",
    "build_workspace",
    "check_adapter",
    "make_adapter",
    "read_snapshots",
    "schema_coverage",
    "visual_coverage",
]

__version__ = "0.1.0"
