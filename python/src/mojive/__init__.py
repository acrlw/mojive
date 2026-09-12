"""Public API with lazy exports for UI-free scene and adapter imports."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mojive.adapters.mujoco.audit import audit_model, schema_coverage, visual_coverage
    from mojive.application.backends import make_adapter
    from mojive.application.composition import (
        Viewer,
        build,
        build_editor,
        build_from_adapter,
        build_scene,
        build_workspace,
    )
    from mojive.application.passive import PassiveViewer, launch_passive
    from mojive.application.renderer import Renderer
    from mojive.capture import CaptureSurface, RecordingInfo, RecordingPhase
    from mojive.capture.recording import SnapshotWriter, VideoRecorder, read_snapshots
    from mojive.capture.shared_image import SharedImage
    from mojive.drawing.canvas import Canvas2D, CanvasLayer2D
    from mojive.interaction.input import InputClaim, InputContext
    from mojive.render.offscreen import SceneRenderer
    from mojive.scene import Scene, SceneLight, SceneObject

    from .adapters.base import (
        ActuatorInfo,
        ActuatorVisualType,
        AdapterCaps,
        CameraInfo,
        ContactObservation,
        DiagnosticFrame,
        DiagnosticSource,
        EqualityConstraintInfo,
        FrameNeeds,
        JointInfo,
        JointVisualType,
        KeyframeInfo,
        NodeType,
        PhysicsObservation,
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
    from .adapters.mujoco import MuJoCoAdapter
    from .adapters.registry import register_adapter, unregister_adapter
    from .adapters.toy import ToyPhysicsAdapter
    from .config import (
        CameraInputConfig,
        CameraTrackingConfig,
        InteractionConfig,
        LayoutConfig,
        PanelConfig,
        RecordingConfig,
        SelectionInputConfig,
        SelectionStyle,
        ViewerConfig,
        ViewportLayers,
        ViewportOverlayConfig,
    )
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
    from .ui.theme import THEME, Theme, ViewportChromeColors, rgb8


_EXPORT_MODULES = {
    ".render.offscreen": ("SceneRenderer",),
    ".adapters.registry": ("register_adapter", "unregister_adapter"),
    ".adapters.base": (
        "ActuatorInfo",
        "ActuatorVisualType",
        "AdapterCaps",
        "CameraInfo",
        "ContactObservation",
        "DiagnosticFrame",
        "DiagnosticSource",
        "EqualityConstraintInfo",
        "FrameNeeds",
        "JointInfo",
        "JointVisualType",
        "KeyframeInfo",
        "NodeType",
        "PhysicsObservation",
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
    ".application.backends": ("make_adapter",),
    ".drawing.canvas": ("Canvas2D", "CanvasLayer2D"),
    ".capture": ("CaptureSurface", "RecordingInfo", "RecordingPhase"),
    ".application.composition": (
        "Viewer",
        "build",
        "build_editor",
        "build_from_adapter",
        "build_scene",
        "build_workspace",
    ),
    ".config": (
        "CameraInputConfig",
        "CameraTrackingConfig",
        "InteractionConfig",
        "LayoutConfig",
        "PanelConfig",
        "RecordingConfig",
        "SelectionInputConfig",
        "SelectionStyle",
        "ViewerConfig",
        "ViewportLayers",
        "ViewportOverlayConfig",
    ),
    ".interaction.input": ("InputClaim", "InputContext"),
    ".application.passive": ("PassiveViewer", "launch_passive"),
    ".capture.shared_image": ("SharedImage",),
    ".capture.recording": ("SnapshotWriter", "VideoRecorder", "read_snapshots"),
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
    ".ui.theme": ("THEME", "Theme", "ViewportChromeColors", "rgb8"),
    ".adapters.mujoco": ("MuJoCoAdapter",),
    ".application.renderer": ("Renderer",),
    ".adapters.mujoco.audit": ("audit_model", "schema_coverage", "visual_coverage"),
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
    "THEME",
    "ActuatorInfo",
    "ActuatorVisualType",
    "AdapterCaps",
    "Bounds",
    "CameraInfo",
    "CameraInputConfig",
    "CameraTrackingConfig",
    "CameraView",
    "Canvas2D",
    "CanvasLayer2D",
    "CaptureSurface",
    "CenteredBounds",
    "ConformanceCheck",
    "ConformanceReport",
    "ContactObservation",
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
    "PassiveViewer",
    "PhysicsObservation",
    "PhysicsState",
    "RecordingConfig",
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
    "SharedImage",
    "SnapshotPublisher",
    "SnapshotWriter",
    "Theme",
    "ToyPhysicsAdapter",
    "VideoRecorder",
    "Viewer",
    "ViewerConfig",
    "ViewportChromeColors",
    "ViewportLayers",
    "ViewportOverlayConfig",
    "VisualGroupInfo",
    "audit_model",
    "build",
    "build_editor",
    "build_from_adapter",
    "build_scene",
    "build_workspace",
    "check_adapter",
    "launch_passive",
    "make_adapter",
    "read_snapshots",
    "register_adapter",
    "rgb8",
    "schema_coverage",
    "unregister_adapter",
    "visual_coverage",
]

__version__ = "0.1.0"
