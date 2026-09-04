"""Public package exports for mojive."""

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
from .canvas2d import Canvas2D, CanvasLayer2D
from .capture import CaptureSurface, RecordingInfo, RecordingPhase
from .composition import (
    Viewer,
    build,
    build_editor,
    build_from_adapter,
    build_scene,
    build_workspace,
)
from .config import (
    CameraInputConfig,
    InteractionConfig,
    LayoutConfig,
    PanelConfig,
    SelectionInputConfig,
    SelectionStyle,
    ViewerConfig,
    ViewportOverlayConfig,
)
from .input import InputClaim, InputContext
from .recording import SnapshotWriter, VideoRecorder, read_snapshots
from .remote import RemoteSceneAdapter, SnapshotPublisher
from .render.backend import (
    DebugView,
    FrameMode,
    LabelMode,
    RenderFlag,
    RenderProduct,
    RenderRequest,
    ShadowQuality,
)
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
    "CameraInputConfig",
    "CameraView",
    "Canvas2D",
    "CanvasLayer2D",
    "CaptureSurface",
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
    "InputClaim",
    "InputContext",
    "InteractionConfig",
    "JointInfo",
    "JointVisualType",
    "KeyframeInfo",
    "LabelMode",
    "Layer",
    "LayoutConfig",
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
    "PanelConfig",
    "PhysicsState",
    "RecordingInfo",
    "RecordingPhase",
    "RemoteSceneAdapter",
    "RenderFlag",
    "RenderProduct",
    "RenderRequest",
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
    "SelectionInputConfig",
    "SelectionStyle",
    "SensorInfo",
    "ShadingModel",
    "ShadowQuality",
    "SnapshotPublisher",
    "SnapshotWriter",
    "ToyPhysicsAdapter",
    "VideoRecorder",
    "Viewer",
    "ViewerConfig",
    "ViewportOverlayConfig",
    "VisualGroupInfo",
    "audit_model",
    "build",
    "build_editor",
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
