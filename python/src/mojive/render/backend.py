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


SELECTION_XRAY_ALPHA = 0.12


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


class ShadowQuality(enum.StrEnum):
    """Shadow-map sampling and near-cascade density preset."""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    HIGH = "high"

    @property
    def level(self) -> int:
        """Return the compact shader value used by both render backends."""

        return tuple(type(self)).index(self)

    @property
    def cascade_divisors(self) -> tuple[float, float, float]:
        """Return near-to-far cascade density without changing atlas memory."""

        return {
            type(self).PERFORMANCE: (6.0, 2.0, 1.0),
            type(self).BALANCED: (9.0, 3.0, 1.0),
            type(self).HIGH: (12.0, 4.0, 1.0),
        }[self]


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


class RenderProduct(enum.IntFlag):
    """GPU products that a render invocation must make available.

    Products describe observable outputs, not implementation passes. Backends
    remain free to combine products in one pass or schedule dedicated passes.
    """

    COLOR = 1 << 0
    METRIC_DEPTH = 1 << 1
    OBJECT_ID = 1 << 2
    SEGMENTATION = 1 << 3


@dataclass(frozen=True)
class RenderRequest:
    """Backend-neutral output contract for one render invocation.

    The default is the interactive viewport contract: resolved color plus a
    current object-ID target for picking. Compatibility renderers should use
    one of the named constructors so a backend can prune unrelated work.
    """

    products: RenderProduct = RenderProduct.COLOR | RenderProduct.OBJECT_ID

    def __post_init__(self) -> None:
        products = RenderProduct(self.products)
        if not products:
            raise ValueError("A render request must contain at least one product")
        object.__setattr__(self, "products", products)

    def needs(self, product: RenderProduct) -> bool:
        """Return whether all bits in ``product`` are requested."""

        return self.products & product == product

    @classmethod
    def viewport(cls) -> RenderRequest:
        """Request resolved color and a current picking buffer."""

        return cls(RenderProduct.COLOR | RenderProduct.OBJECT_ID)

    @classmethod
    def color(cls) -> RenderRequest:
        """Request resolved color only."""

        return cls(RenderProduct.COLOR)

    @classmethod
    def metric_depth(cls) -> RenderRequest:
        """Request linear metric depth only."""

        return cls(RenderProduct.METRIC_DEPTH)

    @classmethod
    def segmentation(cls) -> RenderRequest:
        """Request semantic object and object-type IDs only."""

        return cls(RenderProduct.SEGMENTATION)


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
    def read_rgb(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray: ...
    def read_depth(self, flip: bool = True) -> np.ndarray: ...
    def read_metric_depth(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray: ...
    def read_ids(self, flip: bool = False) -> np.ndarray: ...
    def read_segmentation(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray: ...


@runtime_checkable
class RenderBackend(Protocol):
    """Renderer contract consumed by the viewer and offscreen composition layer.

    A backend receives stable data through :meth:`set_scene` and dynamic data
    through :meth:`update`. Image orientation is declared by the returned
    :class:`~mojive.types.ViewportImage` metadata.
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

    def render(
        self,
        frame: SceneFrame | None = None,
        request: RenderRequest | None = None,
    ) -> ViewportImage | None:
        """Render requested products, defaulting to the interactive viewport."""

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

    def highlight(
        self,
        object_id: int,
        *,
        xray: bool = False,
        fill: bool = True,
        outline: bool = True,
    ) -> None:
        """Set independently filled and outlined selection presentation."""

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

    def set_shadow_quality(self, quality: ShadowQuality) -> bool:
        """Set the shadow quality preset and report whether it is supported."""

        ...

    def get_shadow_quality(self) -> ShadowQuality:
        """Return the active shadow quality preset."""

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
        """Create an independent render target on the same graphics device or context."""

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

    def read_rgb(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        del flip
        shape = (self.height, self.width, 3)
        if out is None:
            return np.zeros(shape, np.uint8)
        if out.shape != shape or out.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 destination with shape {shape}, got {out.dtype} {out.shape}"
            )
        out.fill(0)
        return out

    def read_depth(self, flip: bool = True) -> np.ndarray:
        del flip
        return np.ones((self.height, self.width), np.float32)

    def read_metric_depth(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        del flip
        shape = (self.height, self.width)
        if out is None:
            return np.ones(shape, np.float32)
        if out.shape != shape or out.dtype != np.float32:
            raise ValueError(
                f"Expected float32 destination with shape {shape}, got {out.dtype} {out.shape}"
            )
        out.fill(1.0)
        return out

    def read_ids(self, flip: bool = False) -> np.ndarray:
        del flip
        return np.zeros((self.height, self.width), np.uint32)

    def read_segmentation(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        del flip
        shape = (self.height, self.width, 2)
        if out is None:
            return np.full(shape, -1, np.int32)
        if out.shape != shape or out.dtype != np.int32:
            raise ValueError(
                f"Expected int32 destination with shape {shape}, got {out.dtype} {out.shape}"
            )
        out.fill(-1)
        return out


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
        self._shadow_quality = ShadowQuality.BALANCED

    def set_scene(self, source) -> None: ...
    def update(self, frame) -> None: ...
    def set_camera(self, camera) -> None: ...
    def set_background(self, rgba) -> None: ...
    def set_transparent_id_rendering(self, enabled: bool) -> None: ...
    def render(self, frame=None, request: RenderRequest | None = None) -> ViewportImage | None:
        del frame, request
        return None

    def resize(self, width: int, height: int) -> None:
        self.target.width = max(1, int(width))
        self.target.height = max(1, int(height))

    def capture(self, path, camera=None, size=None) -> bool:
        return False

    def pick(self, x: int, y: int) -> int:
        return 0

    def highlight(
        self,
        object_id: int,
        *,
        xray: bool = False,
        fill: bool = True,
        outline: bool = True,
    ) -> None: ...
    def set_gizmo(self, gizmo) -> bool:
        return False

    def set_flag(self, flag: RenderFlag, value: bool) -> bool:
        return False

    def get_flag(self, flag: RenderFlag) -> bool:
        return self._flags.get(flag, False)

    def set_shadow_quality(self, quality: ShadowQuality) -> bool:
        del quality
        return False

    def get_shadow_quality(self) -> ShadowQuality:
        return self._shadow_quality

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
