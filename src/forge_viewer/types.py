"""Shared cameras, lights, materials, meshes, and instance types."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import math3d


@dataclass(frozen=True)
class CameraView:
    """Backend-neutral camera definition in world coordinates.

    ``eye``, ``target``, and ``up`` define a Z-up look-at camera. Perspective cameras use
    ``fov_y`` unless positive ``focal_length`` and ``sensor_size`` values provide physical
    intrinsics. Orthographic cameras use ``ortho_height`` as their vertical world extent.

    Angles are radians and clipping distances are world units.
    """

    eye: np.ndarray = field(default_factory=lambda: np.array([4.0, 0.0, 0.0], np.float32))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], np.float32))
    fov_y: float = np.deg2rad(45.0)
    near: float = 0.02
    far: float = 200.0
    aspect: float = 1.0
    orthographic: bool = False
    ortho_height: float = 4.0
    focal_length: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    sensor_size: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    principal_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))

    def view_matrix(self) -> np.ndarray:
        """Return the row-major world-to-camera matrix."""
        return math3d.look_at(self.eye, self.target, self.up)

    def proj_matrix(self) -> np.ndarray:
        """Return the row-major projection matrix selected by the camera parameters."""
        if self.orthographic:
            return math3d.orthographic(self.ortho_height, self.aspect, self.near, self.far)
        if self.uses_intrinsics():
            return math3d.perspective_intrinsics(
                self.focal_length,
                self.sensor_size,
                self.principal_offset,
                self.near,
                self.far,
            )
        return math3d.perspective(self.fov_y, self.aspect, self.near, self.far)

    def uses_intrinsics(self) -> bool:
        """Return whether physical focal length and sensor size define the projection."""
        return bool(np.all(np.asarray(self.focal_length) > 0.0)) and bool(
            np.all(np.asarray(self.sensor_size) > 0.0)
        )

    def forward(self) -> np.ndarray:
        """Return the normalized world-space viewing direction."""
        return math3d.normalize(np.asarray(self.target) - np.asarray(self.eye))

    def distance(self) -> float:
        """Return the distance from ``eye`` to ``target``."""
        return float(np.linalg.norm(np.asarray(self.target) - np.asarray(self.eye)))

    def with_aspect(self, aspect: float) -> CameraView:
        """Return a copy configured for a viewport aspect ratio."""
        return replace(self, aspect=float(aspect))

    def matched_ortho_height(self) -> float:
        """Return an orthographic height matching the perspective span at the target."""
        return 2.0 * self.distance() * float(np.tan(self.fov_y * 0.5))


class LightType(enum.IntEnum):
    """Supported light source models."""

    DIRECTIONAL = 0
    POINT = 1
    SPOT = 2
    AREA = 3
    IMAGE = 4


@dataclass(frozen=True)
class Light:
    """One render light expressed in world coordinates.

    Directional and image lights use ``direction``. Point, spot, and area lights use
    ``position``. ``cutoff`` is the spot half-angle in degrees; ``range=0`` selects automatic
    range estimation from attenuation.
    """

    type: LightType = LightType.DIRECTIONAL
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.0], np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], np.float32))
    diffuse: np.ndarray = field(default_factory=lambda: np.full(3, 0.7, np.float32))
    specular: np.ndarray = field(default_factory=lambda: np.full(3, 0.3, np.float32))
    ambient: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    attenuation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], np.float32))
    range: float = 0.0
    area_radius: float = 0.0
    cutoff: float = 45.0
    exponent: float = 10.0
    texture: str | None = None
    intensity: float = 1.0
    cast_shadow: bool = True
    active: bool = True


@dataclass(frozen=True)
class Environment:
    """Global illumination, fog, and horizon-haze parameters."""

    headlight: Light | None = None
    ambient: np.ndarray = field(default_factory=lambda: np.full(3, 0.2, np.float32))
    fog_color: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    fog_start: float = 0.0
    fog_end: float = 0.0
    haze_color: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    haze_density: float = 0.0
    horizon_haze: bool = False
    horizon_haze_slices: int = 64


@dataclass(frozen=True)
class LightSet:
    """Scene lights and the environment values consumed by a render backend."""

    lights: tuple[Light, ...] = ()
    headlight: Light | None = None
    ambient: np.ndarray = field(default_factory=lambda: np.full(3, 0.2, np.float32))

    fog_color: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    fog_start: float = 0.0
    fog_end: float = 0.0
    haze_color: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    haze_density: float = 0.0
    horizon_haze: bool = False
    horizon_haze_slices: int = 64

    def shadow_casters(self) -> tuple[Light, ...]:
        """Return active non-image lights that request shadow maps."""
        return tuple(
            light
            for light in self.lights
            if light.active and light.cast_shadow and light.type is not LightType.IMAGE
        )

    def environment(self) -> Environment:
        """Extract the environment portion of this light set."""
        return Environment(
            headlight=self.headlight,
            ambient=self.ambient,
            fog_color=self.fog_color,
            fog_start=self.fog_start,
            fog_end=self.fog_end,
            haze_color=self.haze_color,
            haze_density=self.haze_density,
            horizon_haze=self.horizon_haze,
            horizon_haze_slices=self.horizon_haze_slices,
        )

    def with_environment(self, environment: Environment) -> LightSet:
        """Return a copy with environment values replaced."""
        return replace(
            self,
            headlight=environment.headlight,
            ambient=environment.ambient,
            fog_color=environment.fog_color,
            fog_start=environment.fog_start,
            fog_end=environment.fog_end,
            haze_color=environment.haze_color,
            haze_density=environment.haze_density,
            horizon_haze=environment.horizon_haze,
            horizon_haze_slices=environment.horizon_haze_slices,
        )


DEFAULT_HEADLIGHT = Light(
    type=LightType.DIRECTIONAL,
    diffuse=np.full(3, 0.4, np.float32),
    specular=np.full(3, 0.5, np.float32),
    ambient=np.full(3, 0.1, np.float32),
    cast_shadow=False,
)


class MeshShape(enum.StrEnum):
    """Built-in and adapter-provided mesh families."""

    SPHERE = "sphere"
    BOX = "box"
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    DISK = "disk"
    TUBE = "tube"
    CAPSULE_SHAFT = "capsule_shaft"
    CAPSULE_CAP = "capsule_cap"
    ARROW_SHAFT = "arrow_shaft"
    ARROW_HEAD = "arrow_head"
    ARROW = "arrow"
    DOUBLE_ARROW = "double_arrow"
    HEIGHTFIELD = "heightfield"
    FLEX = "flex"
    FLEX_FACE = "flex_face"
    SKIN = "skin"
    ASSET = "asset"
    CONVEX_HULL = "convex_hull"


class InstancePoseSource(enum.IntEnum):
    """Index space used to resolve an instance pose from a scene frame."""

    GEOM = 0
    SITE = 1
    WORLD = 2


class InstanceVisual(enum.IntEnum):
    """Semantic rendering treatment assigned to one scene instance."""

    DEFAULT = 0
    FLEX_EDGE = 1
    FLEX_FACE = 2
    FLEX_SKIN = 3
    SKIN = 4


@dataclass(frozen=True)
class MeshKey:
    """Stable mesh identity consisting of a shape family and optional asset index."""

    shape: MeshShape = MeshShape.BOX
    index: int = -1

    def __str__(self) -> str:
        return f"{self.shape}" if self.index < 0 else f"{self.shape}[{self.index}]"


@dataclass(frozen=True)
class MeshData:
    """Indexed triangle mesh uploaded when a scene source changes.

    Arrays use float32 positions, normals, and UVs plus uint32 triangle indices. Vertex arrays
    have equal length; indices are a flat sequence of triangle vertex indices.
    """

    positions: np.ndarray  # (V, 3) f32
    normals: np.ndarray  # (V, 3) f32
    uvs: np.ndarray  # (V, 2) f32
    indices: np.ndarray  # (I,)  u32

    def __post_init__(self) -> None:
        v = len(self.positions)
        assert self.normals.shape == (v, 3)
        assert self.uvs.shape == (v, 2)
        assert self.indices.dtype == np.uint32

    @property
    def triangle_count(self) -> int:
        """Return the number of indexed triangles."""
        return len(self.indices) // 3


@dataclass(frozen=True)
class MeshUpdate:
    """Dynamic position and normal replacement for an existing mesh."""

    positions: np.ndarray  # (V, 3) f32
    normals: np.ndarray  # (V, 3) f32


class TextureType(enum.StrEnum):
    """Texture dimensionality and environment role."""

    TWO_D = "2d"
    CUBE = "cube"
    SKYBOX = "skybox"


class ShadingModel(enum.StrEnum):
    """Color-space and material-lighting convention used by a scene source."""

    LINEAR = "linear"
    MUJOCO_CLASSIC = "mujoco-classic"


MATERIAL_TEXTURE_ROLES = (
    "user",
    "rgb",
    "occlusion",
    "roughness",
    "metallic",
    "normal",
    "opacity",
    "emissive",
    "rgba",
    "orm",
)


@dataclass(frozen=True)
class TextureData:
    """Named uint8 texture pixels stored in a scene source."""

    name: str
    type: TextureType
    pixels: np.ndarray  # 2D: (H, W, C) u8; cube/skybox: (6, S, S, C) u8
    srgb: bool = True

    @property
    def size(self) -> tuple[int, int]:
        """Return texture width and height."""
        if self.type is TextureType.TWO_D:
            return int(self.pixels.shape[1]), int(self.pixels.shape[0])
        return int(self.pixels.shape[2]), int(self.pixels.shape[1])


@dataclass(frozen=True)
class Material:
    """Forge material parameters shared by OpenGL and WebGPU backends.

    Values follow MuJoCo's Phong-style material model. ``texture`` refers to a
    :class:`TextureData` name in the same scene source.
    """

    name: str = ""
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0], np.float32))
    emission: float = 0.0
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0.0
    metallic: float = -1.0
    roughness: float = -1.0
    texture: str | None = None  # TextureData.name
    tex_repeat: np.ndarray = field(default_factory=lambda: np.ones(2, np.float32))
    tex_uniform: bool = False

    @property
    def opaque(self) -> bool:
        """Return whether the alpha channel selects the opaque render pass."""
        return float(self.rgba[3]) >= 1.0


DEFAULT_MATERIAL = Material(name="__default__")


@dataclass(frozen=True)
class ViewportImage:
    """Resolved render target presented inside the editor viewport.

    ``texture_id`` identifies an OpenGL texture. WebGPU backends store the resolved texture view
    in ``payload``. ``flip_y`` describes the image orientation expected by presentation code.
    """

    texture_id: int
    width: int
    height: int
    flip_y: bool = True
    # Backend-specific presentation payload: the GL path presents texture_id
    # directly and leaves this None; the wgpu path carries the GPUTextureView
    # of the resolved color target for the window's imgui renderer.
    payload: Any = None

    @property
    def aspect(self) -> float:
        """Return the image width-to-height ratio."""
        return self.width / max(self.height, 1)

    def pixel_from_viewport_point(
        self, point: tuple[float, float], rect: tuple[float, float, float, float]
    ) -> tuple[int, int] | None:
        """Map a UI point to a bottom-left-origin render-target pixel.

        Args:
            point: UI-space point in framebuffer coordinates.
            rect: Viewport rectangle as ``(x, y, width, height)``.

        Returns:
            The clamped pixel coordinate, or ``None`` when the point lies outside the viewport.
        """
        rx, ry, rw, rh = rect
        if rw <= 0.0 or rh <= 0.0 or self.width <= 0 or self.height <= 0:
            return None
        u = (float(point[0]) - rx) / rw
        v = (float(point[1]) - ry) / rh
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        col = min(int(u * self.width), self.width - 1)
        row_from_top = min(int(v * self.height), self.height - 1)

        return max(col, 0), self.height - 1 - max(row_from_top, 0)
