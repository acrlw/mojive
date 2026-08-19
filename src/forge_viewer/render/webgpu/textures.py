"""Texture storage for the webgpu backend.

2D textures only, matching the sRGB semantics of
``render.forge.resources.TextureStore``: sRGB sources use an sRGB view so the
hardware decode matches GL ``SRGB8_ALPHA8``; other formats are linearized on
the CPU the same way forge does when no sRGB internal format exists.  Cube
maps and image lights are out of scope for now.
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...types import TextureData, TextureKind

_BYTES_PER_CHANNEL = {1: (wgpu.TextureFormat.r8unorm, 1), 2: (wgpu.TextureFormat.rg8unorm, 2)}


def _srgb_to_linear_u8(pixels: np.ndarray) -> np.ndarray:
    x = pixels.astype(np.float32) / 255.0
    lo = x / 12.92
    hi = ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4
    return (np.where(x <= 0.04045, lo, hi) * 255.0 + 0.5).astype(np.uint8)


class TextureStore:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._textures: dict[str, wgpu.GPUTextureView] = {}
        self._white: wgpu.GPUTextureView | None = None
        self.sampler = device.create_sampler(
            mag_filter="linear", min_filter="linear", mipmap_filter="nearest"
        )

    @property
    def white(self) -> wgpu.GPUTextureView:
        if self._white is None:
            tex = self._device.create_texture(
                size=(1, 1, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self._device.queue.write_texture(
                {"texture": tex},
                np.array([[[255, 255, 255, 255]]], np.uint8),
                {"bytes_per_row": 4, "rows_per_image": 1},
                (1, 1, 1),
            )
            self._white = tex.create_view()
        return self._white

    def sync(self, textures: dict[str, TextureData]) -> None:
        for name, data in textures.items():
            if name in self._textures or data.kind is not TextureKind.TWO_D:
                continue
            self._textures[name] = self._upload(data)
        for name in [k for k in self._textures if k not in textures]:
            del self._textures[name]

    def _upload(self, data: TextureData) -> wgpu.GPUTextureView:
        pixels = np.ascontiguousarray(data.pixels)
        h, w, comps = pixels.shape
        if comps in (3, 4):
            if data.srgb:
                fmt = wgpu.TextureFormat.rgba8unorm_srgb
                linearize = False
            else:
                fmt = wgpu.TextureFormat.rgba8unorm
                linearize = True
            if comps == 3:
                alpha = np.full((h, w, 1), 255, np.uint8)
                pixels = np.concatenate([pixels, alpha], axis=2)
            bpp = 4
        elif comps in _BYTES_PER_CHANNEL:
            fmt, bpp = _BYTES_PER_CHANNEL[comps]
            linearize = data.srgb  # no single/dual-channel sRGB formats in WebGPU
        else:
            raise ValueError(f"unsupported texture channel count: {comps}")
        if linearize:
            pixels = np.ascontiguousarray(_srgb_to_linear_u8(pixels))
        tex = self._device.create_texture(
            size=(w, h, 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        row_bytes = (w * bpp + 255) // 256 * 256
        if row_bytes == w * bpp:
            payload = pixels
            layout = {"bytes_per_row": w * bpp, "rows_per_image": h}
        else:
            padded = np.zeros((h, row_bytes), np.uint8)
            padded[:, : w * bpp] = pixels.reshape(h, w * bpp)
            payload = padded
            layout = {"bytes_per_row": row_bytes, "rows_per_image": h}
        self._device.queue.write_texture({"texture": tex}, payload, layout, (w, h, 1))
        return tex.create_view()

    def get(self, name: str | None) -> wgpu.GPUTextureView | None:
        if name is None:
            return None
        return self._textures.get(name)

    def release(self) -> None:
        self._textures.clear()
        self._white = None
