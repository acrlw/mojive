"""Public API with lazy exports for UI-free scene and adapter imports."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        SceneProvider,
        SceneSaveOptions,
        SceneSource,
        SensorInfo,
        VisualGroupInfo,
    )
    from .adapters.conformance import ConformanceCheck, ConformanceReport, check_adapter
    from .adapters.mujoco_adapter import MuJoCoAdapter
    from .adapters.registry import register_adapter, unregister_adapter
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
    from .mujoco_audit import audit_model, schema_coverage, visual_coverage
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
    from .renderer import Renderer
    from .scene import Scene, SceneLight, SceneObject
    from .scene_renderer import SceneRenderer
    from .types import (
        Bounds,
        CameraView,
        CenteredBounds,
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
    from .ui.input_bindings import InputAction


_EXPORT_MODULES = {
    ".scene_renderer": ("SceneRenderer",),
    ".adapters.registry": ("register_adapter", "unregister_adapter"),
    ".adapters.base": (
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
        "NodeType",
        "PhysicsState",
        "SceneAdapter",
        "SceneAdapterBase",
        "SceneFrame",
        "SceneModelInfo",
        "SceneNode",
        "SceneProvider",
        "SceneSaveOptions",
        "SceneSource",
        "SensorInfo",
        "VisualGroupInfo",
    ),
    ".adapters.conformance": ("ConformanceCheck", "ConformanceReport", "check_adapter"),
    ".adapters.toy": ("ToyPhysicsAdapter",),
    ".backends": ("make_adapter",),
    ".canvas2d": ("Canvas2D", "CanvasLayer2D"),
    ".capture": ("CaptureSurface", "RecordingInfo", "RecordingPhase"),
    ".composition": (
        "Viewer",
        "build",
        "build_editor",
        "build_from_adapter",
        "build_scene",
        "build_workspace",
    ),
    ".config": (
        "CameraInputConfig",
        "InteractionConfig",
        "LayoutConfig",
        "PanelConfig",
        "SelectionInputConfig",
        "SelectionStyle",
        "ViewerConfig",
        "ViewportOverlayConfig",
    ),
    ".input": ("InputClaim", "InputContext"),
    ".recording": ("SnapshotWriter", "VideoRecorder", "read_snapshots"),
    ".remote": ("RemoteSceneAdapter", "SnapshotPublisher"),
    ".render.backend": (
        "DebugView",
        "FrameMode",
        "LabelMode",
        "RenderFlag",
        "RenderProduct",
        "RenderRequest",
        "ShadowQuality",
    ),
    ".render.debugdraw": ("DebugDraw", "Layer", "Occlusion"),
    ".scene": ("Scene", "SceneLight", "SceneObject"),
    ".types": (
        "Bounds",
        "CameraView",
        "CenteredBounds",
        "Environment",
        "Light",
        "LightSet",
        "LightType",
        "Material",
        "MeshData",
        "MeshKey",
        "MeshShape",
        "MeshUpdate",
        "ShadingModel",
    ),
    ".ui.input_bindings": ("InputAction",),
    ".adapters.mujoco_adapter": ("MuJoCoAdapter",),
    ".renderer": ("Renderer",),
    ".mujoco_audit": ("audit_model", "schema_coverage", "visual_coverage"),
}
_EXPORTS = {name: module for module, names in _EXPORT_MODULES.items() for name in names}


def __getattr__(name: str):
    """Load each public component only when it is requested."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ActuatorInfo",
    "ActuatorVisualType",
    "AdapterCaps",
    "Bounds",
    "CameraInfo",
    "CameraInputConfig",
    "CameraView",
    "Canvas2D",
    "CanvasLayer2D",
    "CaptureSurface",
    "CenteredBounds",
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
    "InputAction",
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
    "SceneProvider",
    "SceneRenderer",
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
    "register_adapter",
    "schema_coverage",
    "unregister_adapter",
    "visual_coverage",
]

__version__ = "0.1.0"
