from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace

import numpy as np

from . import math3d


@dataclass(frozen=True)
class CameraView:
    eye: np.ndarray = field(default_factory=lambda: np.array([3.0, -3.0, 2.0], np.float32))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], np.float32))
    fov_y: float = np.deg2rad(45.0)
    near: float = 0.02
    far: float = 200.0
    aspect: float = 1.0
    orthographic: bool = False
    ortho_height: float = 4.0

    def view_matrix(self) -> np.ndarray:
        return math3d.look_at(self.eye, self.target, self.up)

    def proj_matrix(self) -> np.ndarray:
        if self.orthographic:
            return math3d.orthographic(self.ortho_height, self.aspect, self.near, self.far)
        return math3d.perspective(self.fov_y, self.aspect, self.near, self.far)

    def forward(self) -> np.ndarray:
        return math3d.normalize(np.asarray(self.target) - np.asarray(self.eye))

    def distance(self) -> float:
        return float(np.linalg.norm(np.asarray(self.target) - np.asarray(self.eye)))

    def with_aspect(self, aspect: float) -> CameraView:
        return replace(self, aspect=float(aspect))

    def matched_ortho_height(self) -> float:

        return 2.0 * self.distance() * float(np.tan(self.fov_y * 0.5))


class LightKind(enum.IntEnum):
    DIRECTIONAL = 0
    POINT = 1
    SPOT = 2
    AREA = 3


@dataclass(frozen=True)
class Light:
    kind: LightKind = LightKind.DIRECTIONAL
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
    cast_shadow: bool = True
    active: bool = True


@dataclass(frozen=True)
class LightSet:
    lights: tuple[Light, ...] = ()
    headlight: Light | None = None
    ambient: np.ndarray = field(default_factory=lambda: np.full(3, 0.2, np.float32))

    fog_color: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    fog_start: float = 0.0
    fog_end: float = 0.0
    haze_color: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    haze_density: float = 0.0

    def shadow_casters(self) -> tuple[Light, ...]:
        return tuple(x for x in self.lights if x.active and x.cast_shadow)


DEFAULT_HEADLIGHT = Light(
    kind=LightKind.DIRECTIONAL,
    diffuse=np.full(3, 0.4, np.float32),
    specular=np.full(3, 0.5, np.float32),
    ambient=np.full(3, 0.1, np.float32),
    cast_shadow=False,
)


class MeshShape(enum.StrEnum):
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
    SKIN = "skin"
    ASSET = "asset"


class InstancePoseSource(enum.IntEnum):
    GEOM = 0
    SITE = 1
    WORLD = 2


@dataclass(frozen=True)
class MeshKey:
    shape: MeshShape = MeshShape.BOX
    index: int = -1

    def __str__(self) -> str:
        return f"{self.shape}" if self.index < 0 else f"{self.shape}[{self.index}]"


@dataclass(frozen=True)
class MeshData:
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
        return len(self.indices) // 3


@dataclass(frozen=True)
class MeshUpdate:
    positions: np.ndarray  # (V, 3) f32
    normals: np.ndarray  # (V, 3) f32


class TextureKind(enum.StrEnum):
    TWO_D = "2d"
    CUBE = "cube"
    SKYBOX = "skybox"


@dataclass(frozen=True)
class TextureData:
    name: str
    kind: TextureKind
    pixels: np.ndarray  # 2d: (H, W, C) u8；cube/skybox: (6, S, S, C) u8
    srgb: bool = True

    @property
    def size(self) -> tuple[int, int]:
        if self.kind is TextureKind.TWO_D:
            return int(self.pixels.shape[1]), int(self.pixels.shape[0])
        return int(self.pixels.shape[2]), int(self.pixels.shape[1])


@dataclass(frozen=True)
class Material:
    name: str = ""
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0], np.float32))
    emission: float = 0.0
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0.0
    texture: str | None = None  # TextureData.name
    tex_repeat: np.ndarray = field(default_factory=lambda: np.ones(2, np.float32))
    tex_uniform: bool = False

    @property
    def opaque(self) -> bool:
        return float(self.rgba[3]) >= 1.0


DEFAULT_MATERIAL = Material(name="__default__")


@dataclass(frozen=True)
class ViewportImage:
    texture_id: int
    width: int
    height: int
    flip_y: bool = True

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    def pixel_from_viewport_point(
        self, point: tuple[float, float], rect: tuple[float, float, float, float]
    ) -> tuple[int, int] | None:

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
