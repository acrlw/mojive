"""Renderer protocols, feature flags, and frame statistics."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..types import CameraView, ViewportImage

if TYPE_CHECKING:
    from ..adapters.base import SceneFrame, SceneSource
    from ..gizmo import GizmoFrame
    from .debugdraw import DebugDraw


class RenderFlag(enum.StrEnum):
    """Independently switchable renderer features and MuJoCo visual semantics."""

    # --- mjtRndFlag
    SHADOW = "shadow"
    WIREFRAME = "wireframe"
    REFLECTION = "reflection"
    ADDITIVE = "additive"
    SKYBOX = "skybox"
    FOG = "fog"
    HAZE = "haze"
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
    TONEMAP = "tonemap"
    MSAA = "msaa"


class DebugView(enum.StrEnum):
    """Viewport output selected for renderer inspection."""

    SHADED = "shaded"
    ALBEDO = "albedo"
    NORMAL = "normal"
    DEPTH = "depth"
    SEGMENT = "segment"
    IDCOLOR = "idcolor"
    OVERDRAW = "overdraw"
    WIREFRAME = "wireframe"


class LabelMode(enum.StrEnum):
    """Scene entity category rendered as text labels."""

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
    """Scene entity category rendered with coordinate frames."""

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
    """Immutable feature and platform capabilities reported by a render backend."""

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
        """Return whether the backend implements ``flag``."""

        return flag in self.render_flags


@dataclass
class RenderStats:
    """Per-frame workload counters and named CPU/GPU render-pass timings."""

    draw_calls: int = 0
    instances: int = 0
    triangles: int = 0
    buckets: int = 0
    cpu_ms: dict[str, float] = field(default_factory=dict)
    gpu_ms: dict[str, float] = field(default_factory=dict)
    frame_cpu_ms: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)


class ReadbackTarget(Protocol):
    """Minimal render-target surface exposed to capture and compatibility APIs."""

    width: int
    height: int
    samples: int

    def read_color(self, flip: bool = True) -> np.ndarray: ...
    def read_depth(self, flip: bool = True) -> np.ndarray: ...
    def read_ids(self, flip: bool = False) -> np.ndarray: ...


@runtime_checkable
class RenderBackend(Protocol):
    """Renderer contract consumed by the viewer and offscreen composition layer.

    A backend receives stable data through :meth:`set_scene` and dynamic data
    through :meth:`update`. Image orientation is declared by the returned
    :class:`~forge_viewer.types.ViewportImage` metadata.
    """

    caps: BackendCaps
    debug: DebugDraw | None
    stats: RenderStats
    target: ReadbackTarget

    def set_background(self, rgba: tuple[float, float, float, float]) -> None:
        """Set the clear color used by subsequent render calls."""

        ...

    def set_transparent_id_rendering(self, enabled: bool) -> None:
        """Include or exclude transparent objects from ID output."""

        ...

    def set_scene(self, source: SceneSource) -> None:
        """Upload stable scene structure after its revision changes."""

        ...

    def update(self, frame: SceneFrame) -> None:
        """Upload one dynamic scene frame."""

        ...

    def set_camera(self, camera: CameraView) -> None:
        """Set the camera used by the next render call."""

        ...

    def render(self, frame: SceneFrame | None = None) -> ViewportImage | None:
        """Optionally update from ``frame``, then return a viewport image handle."""

        ...

    def resize(self, width: int, height: int) -> None:
        """Resize color, depth, ID, and post-process targets in physical pixels."""

        ...

    def capture(
        self,
        path: Path,
        camera: CameraView | None = None,
        size: tuple[int, int] | None = None,
    ) -> bool:
        """Write the current scene to ``path``, optionally from another camera."""

        ...

    def pick(self, x: int, y: int) -> int:
        """Return the object ID at a physical-pixel viewport coordinate."""

        ...

    def highlight(self, object_id: int) -> None:
        """Set the object rendered by the selection outline pass."""

        ...

    def set_gizmo(self, gizmo: GizmoFrame | None) -> bool:
        """Set or hide the optional GPU-rendered transform gizmo."""

        ...

    def set_flag(self, flag: RenderFlag, value: bool) -> bool:
        """Set a render flag and report whether the backend accepted it."""

        ...

    def get_flag(self, flag: RenderFlag) -> bool:
        """Return the effective value of a render flag."""

        ...

    def set_debug_view(self, view: DebugView) -> bool:
        """Select a debug output and report whether it is supported."""

        ...

    def get_debug_view(self) -> DebugView:
        """Return the active debug output."""

        ...

    def set_label_mode(self, mode: LabelMode) -> bool:
        """Select the scene label category."""

        ...

    def get_label_mode(self) -> LabelMode:
        """Return the active scene label category."""

        ...

    def set_frame_mode(self, mode: FrameMode) -> bool:
        """Select the coordinate-frame overlay category."""

        ...

    def get_frame_mode(self) -> FrameMode:
        """Return the active coordinate-frame overlay category."""

        ...

    def set_bvh_depth(self, depth: int) -> bool:
        """Select the diagnostic BVH depth."""

        ...

    def get_bvh_depth(self) -> int:
        """Return the active diagnostic BVH depth."""

        ...

    def render_options(self) -> tuple[RenderFlag, ...]:
        """Return render flags exposed by the backend settings UI."""

        ...

    def create_peer(self, width: int, height: int) -> RenderBackend:
        """Create an independent render target sharing immutable GPU resources."""

        ...

    def describe(self) -> str:
        """Return a compact renderer and platform description for diagnostics."""

        ...

    def release(self) -> None:
        """Release GPU resources owned by this backend instance."""

        ...


@dataclass
class _NullTarget:
    width: int = 1
    height: int = 1
    samples: int = 0

    def read_color(self, flip: bool = True) -> np.ndarray:
        del flip
        return np.zeros((self.height, self.width, 4), np.uint8)

    def read_depth(self, flip: bool = True) -> np.ndarray:
        del flip
        return np.ones((self.height, self.width), np.float32)

    def read_ids(self, flip: bool = False) -> np.ndarray:
        del flip
        return np.zeros((self.height, self.width), np.uint32)


class NullBackend:
    """No-op backend used when rendering is unavailable or intentionally disabled."""

    def __init__(self, reason: str = "No render backend is available") -> None:
        self.reason = reason
        self.caps = BackendCaps(name="null", notes=(reason,))
        self.debug = None
        self.stats = RenderStats()
        self.target = _NullTarget()
        self._flags: dict[RenderFlag, bool] = {}
        self._view = DebugView.SHADED
        self._label_mode = LabelMode.NONE
        self._frame_mode = FrameMode.NONE

    def set_scene(self, source) -> None: ...
    def update(self, frame) -> None: ...
    def set_camera(self, camera) -> None: ...
    def set_background(self, rgba) -> None: ...
    def set_transparent_id_rendering(self, enabled: bool) -> None: ...
    def render(self, frame=None) -> ViewportImage | None:
        return None

    def resize(self, width: int, height: int) -> None:
        self.target.width = max(1, int(width))
        self.target.height = max(1, int(height))

    def capture(self, path, camera=None, size=None) -> bool:
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

    def create_peer(self, width: int, height: int) -> NullBackend:
        return NullBackend(self.reason)

    def describe(self) -> str:
        return f"null renderer: {self.reason}"

    def release(self) -> None: ...
