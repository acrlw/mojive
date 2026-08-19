"""Light scheduling and uniform packing for the webgpu backend.

Faithful port of ``render.forge.passes.base.schedule_lights`` (minus the shadow
selection, which this backend does not implement yet) and of the
``OpaquePass._light_uniforms`` conversion: light colors are sRGB-decoded on the
CPU, the ambient term stays raw and is decoded in the shader.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...types import Light, LightKind, LightSet

MAX_SCENE_LIGHTS = 100

LIGHTS_DTYPE = np.dtype(
    [
        ("pos", "(100,4)f4"),  # xyz position, w kind
        ("dir", "(100,4)f4"),  # xyz direction, w cutoff cosine
        ("diffuse", "(100,4)f4"),  # rgb linear, w spot exponent
        ("specular", "(100,4)f4"),
        ("atten", "(100,4)f4"),  # constant, linear, quadratic, range
    ]
)
LIGHTS_BYTES = LIGHTS_DTYPE.itemsize
assert LIGHTS_BYTES == 5 * 100 * 16


def srgb_to_linear(x) -> np.ndarray:
    """sRGB EOTF, matching the GLSL srgb_to_linear (used for light colors)."""
    x = np.asarray(x, np.float32)
    lo = x / 12.92
    hi = ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4
    return np.where(x <= 0.04045, lo, hi)


def schedule_lights(lights: LightSet) -> tuple[Light, ...]:
    active = tuple(
        light for light in lights.lights if light.active and light.kind is not LightKind.IMAGE
    )
    return active[:MAX_SCENE_LIGHTS]


class LightUniforms:
    """Packs the scheduled lights into the storage-buffer block each frame."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._block = np.zeros((), LIGHTS_DTYPE)
        self.buffer = device.create_buffer(
            size=LIGHTS_BYTES, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )

    def upload(self, lights: LightSet) -> int:
        block = self._block
        n = 0
        for light in schedule_lights(lights):
            d = np.asarray(light.direction, np.float64)
            norm = float(np.linalg.norm(d))
            block["pos"][n, :3] = light.position
            block["pos"][n, 3] = float(int(light.kind))
            block["dir"][n, :3] = d / norm if norm > 1e-9 else (0.0, 0.0, -1.0)
            block["dir"][n, 3] = float(np.cos(np.deg2rad(min(max(light.cutoff, 0.0), 180.0))))
            block["diffuse"][n, :3] = srgb_to_linear(light.diffuse)
            block["diffuse"][n, 3] = float(light.exponent)
            block["specular"][n, :3] = srgb_to_linear(light.specular)
            block["atten"][n, :3] = light.attenuation
            block["atten"][n, 3] = float(light.range)
            n += 1
        if n:
            self._device.queue.write_buffer(self.buffer, 0, self._block.tobytes())
        return n

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
