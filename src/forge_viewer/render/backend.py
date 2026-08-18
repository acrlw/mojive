"""Renderer protocols, feature flags, and frame statistics."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..types import CameraView, ViewportImage

if TYPE_CHECKING:
    from ..adapters.base import SceneFrame, SceneSource
    from ..gizmo import GizmoFrame
    from .debugdraw import DebugDraw


class RenderFlag(enum.StrEnum):
    # --- mjtRndFlag
    SHADOW = "shadow"
    WIREFRAME = "wireframe"
    REFLECTION = "reflection"
    ADDITIVE = "additive"
    SKYBOX = "skybox"
    FOG = "fog"
    HAZE = "haze"
    SEGMENT = "segment"
    IDCOLOR = "idcolor"
    CULL_FACE = "cull_face"

    CONVEXHULL = "convexhull"
    TEXTURE = "texture"
    JOINT = "joint"
    ACTUATOR = "actuator"
    ACTIVATION = "activation"
    CAMERA = "camera"
    LIGHT = "light"
    RANGEFINDER = "rangefinder"
    CONSTRAINT = "constraint"
    STATIC = "static"
    SKIN = "skin"
    FLEXFACE = "flex_face"
    FLEXSKIN = "flex_skin"
    FLEXVERT = "flex_vertex"
    FLEXEDGE = "flex_edge"
    CONTACTPOINT = "contactpoint"
    CONTACTFORCE = "contactforce"
    CONTACTSPLIT = "contactsplit"
    ISLAND = "island"
    AUTOCONNECT = "autoconnect"
    TENDON = "tendon"
    TRANSPARENT = "transparent"
    COM = "com"
    INERTIA = "inertia"
    SCLINERTIA = "scaled_inertia"
    BODYBVH = "body_bvh"
    MESHBVH = "mesh_bvh"

    OUTLINE = "outline"
    ALBEDO = "albedo"
    NORMAL = "normal"
    OVERDRAW = "overdraw"
    DEPTH = "depth"
    TONEMAP = "tonemap"
    MSAA = "msaa"


class DebugView(enum.StrEnum):
    SHADED = "shaded"
    ALBEDO = "albedo"
    NORMAL = "normal"
    DEPTH = "depth"
    SEGMENT = "segment"
    IDCOLOR = "idcolor"
    OVERDRAW = "overdraw"
    WIREFRAME = "wireframe"


class LabelMode(enum.StrEnum):
    NONE = "none"
    BODY = "body"
    JOINT = "joint"
    GEOM = "geom"
    SITE = "site"
    CAMERA = "camera"
    LIGHT = "light"
    TENDON = "tendon"
    ACTUATOR = "actuator"
    CONSTRAINT = "constraint"
    FLEX = "flex"
    CONTACT_POINT = "contact point"
    CONTACT_FORCE = "contact force"
    SELECTION = "selection"


class FrameMode(enum.StrEnum):
    NONE = "none"
    BODY = "body"
    GEOM = "geom"
    SITE = "site"
    CAMERA = "camera"
    LIGHT = "light"
    CONTACT = "contact"
    WORLD = "world"


@dataclass(frozen=True)
class BackendCaps:
    name: str = "?"
    gpu_pick: bool = False
    debug_draw: bool = False
    render_flags: frozenset[RenderFlag] = frozenset()
    debug_views: frozenset[DebugView] = frozenset()
    label_modes: frozenset[LabelMode] = frozenset()
    frame_modes: frozenset[FrameMode] = frozenset()
    capture: bool = False
    orthographic: bool = False
    shadows: bool = False
    outline: bool = False
    gizmo: bool = False
    pass_timing: bool = False
    gpu_timing: bool = False
    msaa_samples: int = 0
    id_msaa: bool = False

    gl_version: str = ""
    renderer: str = ""
    notes: tuple[str, ...] = ()

    def supports(self, flag: RenderFlag) -> bool:
        return flag in self.render_flags


@dataclass
class RenderStats:
    draw_calls: int = 0
    instances: int = 0
    triangles: int = 0
    buckets: int = 0
    cpu_ms: dict[str, float] = field(default_factory=dict)
    gpu_ms: dict[str, float] = field(default_factory=dict)
    frame_cpu_ms: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class RenderBackend(Protocol):
    caps: BackendCaps
    debug: DebugDraw | None
    stats: RenderStats

    def set_scene(self, source: SceneSource) -> None: ...

    def update(self, frame: SceneFrame) -> None: ...

    def set_camera(self, camera: CameraView) -> None: ...

    def render(self, frame: SceneFrame) -> ViewportImage | None: ...

    def resize(self, width: int, height: int) -> None: ...

    def capture(self, path: Path, camera: CameraView | None = None) -> bool: ...

    def pick(self, x: int, y: int) -> int: ...

    def highlight(self, object_id: int) -> None: ...

    def set_gizmo(self, gizmo: GizmoFrame | None) -> bool: ...

    def set_flag(self, flag: RenderFlag, value: bool) -> bool: ...

    def get_flag(self, flag: RenderFlag) -> bool: ...

    def set_debug_view(self, view: DebugView) -> bool: ...

    def get_debug_view(self) -> DebugView: ...

    def set_label_mode(self, mode: LabelMode) -> bool: ...

    def get_label_mode(self) -> LabelMode: ...

    def set_frame_mode(self, mode: FrameMode) -> bool: ...

    def get_frame_mode(self) -> FrameMode: ...

    def set_bvh_depth(self, depth: int) -> bool: ...

    def get_bvh_depth(self) -> int: ...

    def render_options(self) -> tuple[RenderFlag, ...]: ...

    def release(self) -> None: ...


class NullBackend:
    def __init__(self, reason: str = "No render backend is available") -> None:
        self.reason = reason
        self.caps = BackendCaps(name="null", notes=(reason,))
        self.debug = None
        self.stats = RenderStats()
        self._flags: dict[RenderFlag, bool] = {}
        self._view = DebugView.SHADED
        self._label_mode = LabelMode.NONE
        self._frame_mode = FrameMode.NONE

    def set_scene(self, source) -> None: ...
    def update(self, frame) -> None: ...
    def set_camera(self, camera) -> None: ...
    def render(self, frame) -> ViewportImage | None:
        return None

    def resize(self, width: int, height: int) -> None: ...
    def capture(self, path, camera=None) -> bool:
        return False

    def pick(self, x: int, y: int) -> int:
        return 0

    def highlight(self, object_id: int) -> None: ...
    def set_gizmo(self, gizmo) -> bool:
        return False

    def set_flag(self, flag: RenderFlag, value: bool) -> bool:
        return False

    def get_flag(self, flag: RenderFlag) -> bool:
        return self._flags.get(flag, False)

    def set_debug_view(self, view: DebugView) -> bool:
        return False

    def get_debug_view(self) -> DebugView:
        return self._view

    def set_label_mode(self, mode: LabelMode) -> bool:
        return False

    def get_label_mode(self) -> LabelMode:
        return self._label_mode

    def set_frame_mode(self, mode: FrameMode) -> bool:
        return False

    def get_frame_mode(self) -> FrameMode:
        return self._frame_mode

    def set_bvh_depth(self, depth: int) -> bool:
        return False

    def get_bvh_depth(self) -> int:
        return 0

    def render_options(self) -> tuple[RenderFlag, ...]:
        return ()

    def release(self) -> None: ...
