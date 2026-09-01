"""Light scheduling and uniform packing for wgpu render passes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import wgpu

from ...types import Light, LightSet, LightType

MAX_SCENE_LIGHTS = 100
LOCAL_SHADOW_SLOTS = 8

# MuJoCo's image-light intensity scale, mirroring opengl passes/opaque.py.
IMAGE_LIGHT_REFERENCE_INTENSITY = 5000.0

# Slope-scaled bias factors, mirroring opengl passes/shadow.py SHADOW_BIAS.
SHADOW_BIAS = (1.0, 2.5)

LIGHTS_DTYPE = np.dtype(
    [
        ("pos", "(100,4)f4"),  # xyz position, w light type
        ("dir", "(100,4)f4"),  # xyz direction, w cutoff cosine
        ("diffuse", "(100,4)f4"),  # rgb linear, w spot exponent
        ("specular", "(100,4)f4"),
        ("atten", "(100,4)f4"),  # constant, linear, quadratic, range
        # Shadow block, mirroring the shadow_sample.glsl uniform set.
        ("shadow_matrix", "(3,4,4)f4"),  # u_shadow_matrix (glsl:17), column-major
        ("shadow_tile", "(3,4)f4"),  # u_shadow_tile (glsl:20)
        ("shadow_splits", "(4,)f4"),  # u_shadow_splits xyz (glsl:19)
        ("shadow_texel", "(4,)f4"),  # u_shadow_texel xyz (glsl:19)
        ("shadow_bias", "(4,)f4"),  # u_shadow_bias xy (glsl:22)
        ("shadow_counts", "(4,)f4"),  # x u_shadow_count (glsl:21), y u_local_count
        # (glsl:30), z u_shadow_light (lighting.glsl:25)
        ("local_matrix", "(8,4,4)f4"),  # u_local_matrix (glsl:25), column-major
        ("local_pos", "(8,4)f4"),  # u_local_pos xyz position, w range (glsl:26)
        ("local_texel", "(8,)f4"),  # u_local_texel (glsl:27)
        ("local_radius", "(8,)f4"),  # u_local_radius (glsl:28)
        ("local_layer", "(8,)i4"),  # packed base layer for each local shadow
        ("local_slot", "(100,)i4"),  # u_local_slot (glsl:29)
    ]
)
LIGHTS_BYTES = LIGHTS_DTYPE.itemsize
assert LIGHTS_BYTES == 9440


def srgb_to_linear(x) -> np.ndarray:
    """sRGB EOTF, matching the GLSL srgb_to_linear (used for light colors)."""
    x = np.asarray(x, np.float32)
    lo = x / 12.92
    hi = ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4
    return np.where(x <= 0.04045, lo, hi)


@dataclass(frozen=True)
class LightSchedule:
    """Port of ``render.opengl.passes.base.LightSchedule``."""

    lights: tuple[Light, ...]
    active_count: int
    directional_shadow: int
    local_shadows: tuple[int, ...]
    shadow_candidate_count: int

    @property
    def deferred_lights(self) -> int:
        return self.active_count - len(self.lights)

    @property
    def selected_shadow_count(self) -> int:
        return (self.directional_shadow >= 0) + len(self.local_shadows)

    @property
    def deferred_shadows(self) -> int:
        return self.shadow_candidate_count - self.selected_shadow_count


def schedule_lights(lights: LightSet) -> LightSchedule:
    """Active lights plus shadow-caster selection, mirroring opengl base.py:40."""
    active = tuple(
        light for light in lights.lights if light.active and light.type is not LightType.IMAGE
    )
    selected = active[:MAX_SCENE_LIGHTS]
    directional = next(
        (
            index
            for index, light in enumerate(selected)
            if light.cast_shadow and light.type is LightType.DIRECTIONAL
        ),
        -1,
    )
    local = tuple(
        index
        for index, light in enumerate(selected)
        if light.cast_shadow and light.type in (LightType.POINT, LightType.SPOT, LightType.AREA)
    )[:LOCAL_SHADOW_SLOTS]
    shadow_candidates = sum(
        light.cast_shadow
        and light.type in (LightType.DIRECTIONAL, LightType.POINT, LightType.SPOT, LightType.AREA)
        for light in active
    )
    return LightSchedule(selected, len(active), directional, local, shadow_candidates)


def active_image_light(lights: LightSet) -> Light | None:
    """The last active image light, mirroring ``OpaquePass._light_uniforms``."""
    return next(
        (
            light
            for light in reversed(lights.lights)
            if light.active and light.type is LightType.IMAGE
        ),
        None,
    )


@dataclass
class ShadowState:
    """Per-frame shadow caster payload, mirroring opengl passes/base.py ShadowResult.

    Produced by ``passes.shadow.ShadowPass.prepare`` and consumed by
    ``LightUniforms.upload`` (uniform packing) and the scene draw (bind-group
    selection).  ``enabled=False`` zeroes the counts so the shader never
    samples the fallback textures.
    """

    enabled: bool = False
    cascade_count: int = 0
    shadow_light: int = -1  # scheduled index of the directional caster
    matrices: np.ndarray = field(default_factory=lambda: np.zeros((3, 4, 4), np.float32))
    splits: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    texel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    tile_uv: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), np.float32))
    local_count: int = 0
    local_light_indices: np.ndarray = field(
        default_factory=lambda: np.full(LOCAL_SHADOW_SLOTS, -1, np.int32)
    )
    local_matrices: np.ndarray = field(
        default_factory=lambda: np.zeros((LOCAL_SHADOW_SLOTS, 4, 4), np.float32)
    )
    local_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((LOCAL_SHADOW_SLOTS, 4), np.float32)
    )
    local_texel: np.ndarray = field(
        default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.float32)
    )
    local_radius: np.ndarray = field(
        default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.float32)
    )
    local_layers: np.ndarray = field(default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.int32))


class LightUniforms:
    """Packs the scheduled lights and shadow payload into the storage buffer."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._block = np.zeros((), LIGHTS_DTYPE)
        self.buffer = device.create_buffer(
            size=LIGHTS_BYTES, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )

    def upload(self, lights: LightSet, shadow: ShadowState | None = None) -> LightSchedule:
        schedule = schedule_lights(lights)
        block = self._block
        for n, light in enumerate(schedule.lights):
            d = np.asarray(light.direction, np.float64)
            norm = float(np.linalg.norm(d))
            block["pos"][n, :3] = light.position
            block["pos"][n, 3] = float(int(light.type))
            block["dir"][n, :3] = d / norm if norm > 1e-9 else (0.0, 0.0, -1.0)
            block["dir"][n, 3] = float(np.cos(np.deg2rad(min(max(light.cutoff, 0.0), 180.0))))
            block["diffuse"][n, :3] = srgb_to_linear(light.diffuse)
            block["diffuse"][n, 3] = float(light.exponent)
            block["specular"][n, :3] = srgb_to_linear(light.specular)
            block["atten"][n, :3] = light.attenuation
            block["atten"][n, 3] = float(light.range)
        self._pack_shadow(block, shadow)
        # The counts gate shadow sampling, so the block is rewritten even for
        # an empty light list (stale counts would re-enable stale matrices).
        self._device.queue.write_buffer(self.buffer, 0, self._block.tobytes())
        return schedule

    @staticmethod
    def _pack_shadow(block: np.ndarray, shadow: ShadowState | None) -> None:
        if shadow is None or not shadow.enabled:
            block["shadow_counts"][:] = (0.0, 0.0, -1.0, 0.0)
            block["local_slot"].fill(-1)
            return
        k = min(shadow.cascade_count, 3)
        block["shadow_matrix"][:] = 0.0
        block["shadow_matrix"][:k] = shadow.matrices[:k].transpose(0, 2, 1)
        block["shadow_tile"][:] = shadow.tile_uv
        block["shadow_splits"][:3] = shadow.splits
        block["shadow_texel"][:3] = shadow.texel_world
        block["shadow_bias"][:] = (*SHADOW_BIAS, 0.0, 0.0)
        block["shadow_counts"][:] = (
            float(shadow.cascade_count),
            float(shadow.local_count),
            float(shadow.shadow_light),
            0.0,
        )
        m = min(shadow.local_count, LOCAL_SHADOW_SLOTS)
        block["local_matrix"][:] = 0.0
        block["local_matrix"][:m] = shadow.local_matrices[:m].transpose(0, 2, 1)
        block["local_pos"][:] = shadow.local_positions
        block["local_texel"][:] = shadow.local_texel
        block["local_radius"][:] = shadow.local_radius
        block["local_layer"][:] = shadow.local_layers
        block["local_slot"].fill(-1)
        for slot in range(m):
            index = int(shadow.local_light_indices[slot])
            if 0 <= index < MAX_SCENE_LIGHTS:
                block["local_slot"][index] = slot

    def headlight_terms(self, lights: LightSet) -> tuple[np.ndarray, np.ndarray]:
        hl = lights.headlight
        if hl is None or not hl.active:
            return np.zeros(4, np.float32), np.zeros(4, np.float32)
        diffuse = np.zeros(4, np.float32)
        diffuse[:3] = srgb_to_linear(hl.diffuse)
        diffuse[3] = 1.0
        specular = np.zeros(4, np.float32)
        specular[:3] = srgb_to_linear(hl.specular)
        return diffuse, specular

    def release(self) -> None:
        self.buffer.destroy()
