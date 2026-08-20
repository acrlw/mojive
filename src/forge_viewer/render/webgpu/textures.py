"""Texture storage for the webgpu backend.

2D and cube textures, matching the sRGB semantics of
``render.forge.resources.TextureStore``: sRGB sources use an sRGB view so the
hardware decode matches GL ``SRGB8_ALPHA8``; other formats are linearized on
the CPU the same way forge does when no sRGB internal format exists.

Cube textures (``TextureKind.CUBE``/``SKYBOX``) upload six faces as a 2D array
exposed through a ``cube`` view.  WebGPU has no mipmap generation, so mip
chains (2D and cube alike) are box-filtered on the CPU in stored (possibly
sRGB-encoded) space — the same domain ``glGenerateMipmap`` averages in for
forge.  2D filtering matches forge: trilinear with anisotropy 8.0.  The IBL
roughness LOD depends on the full cube chain.  A 1x1 ``black_cube`` stands in
when no image light or skybox cube is bound.
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


def _mip_chain(pixels: np.ndarray) -> list[np.ndarray]:
    """Box-filtered mip levels of a (layers, h, w, comps) u8 array, in stored space."""
    levels = [pixels]
    h, w = pixels.shape[1], pixels.shape[2]
    while (h > 1 or w > 1) and h % 2 == 0 and w % 2 == 0:
        h, w = h // 2, w // 2
        layers, _, _, comps = levels[-1].shape
        level = levels[-1].reshape(layers, h, 2, w, 2, comps).mean(axis=(2, 4))
        levels.append(np.ascontiguousarray((level + 0.5).astype(np.uint8)))
    return levels


class TextureStore:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self._textures: dict[str, wgpu.GPUTextureView] = {}
        self._cubes: dict[str, tuple[wgpu.GPUTextureView, int]] = {}
        self._skybox_name: str | None = None
        self._white: wgpu.GPUTextureView | None = None
        self._black_cube: wgpu.GPUTextureView | None = None
        # forge 2D textures wrap (repeat_x/repeat_y = True); tiled planes and
        # box face-axis mapping rely on uv outside [0,1] repeating.  Trilinear
        # plus anisotropy 8.0 matches forge (resources.py builds mipmaps with
        # aniso 8.0); WebGPU has no mipmap generation, so _upload builds the
        # chain on the CPU.
        self.sampler = device.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            mipmap_filter="linear",
            address_mode_u="repeat",
            address_mode_v="repeat",
            max_anisotropy=8,
        )
        self.cube_sampler = device.create_sampler(
            mag_filter="linear", min_filter="linear", mipmap_filter="linear"
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

    def sync(self, textures: dict[str, TextureData], skybox: str | None = None) -> None:
        for name, data in textures.items():
            if data.kind is TextureKind.TWO_D:
                if name not in self._textures:
                    self._textures[name] = self._upload(data)
            elif name not in self._cubes:
                self._cubes[name] = self._upload_cube(data)
        for name in [k for k in self._textures if k not in textures]:
            del self._textures[name]
        for name in [k for k in self._cubes if k not in textures]:
            del self._cubes[name]
        self._skybox_name = skybox

    def _format_for(self, comps: int, srgb: bool) -> tuple[str, int, bool]:
        """Texture format, bytes per pixel, and CPU-linearization flag."""
        if comps in (3, 4):
            if srgb:
                return wgpu.TextureFormat.rgba8unorm_srgb, 4, False
            return wgpu.TextureFormat.rgba8unorm, 4, True
        if comps in _BYTES_PER_CHANNEL:
            fmt, bpp = _BYTES_PER_CHANNEL[comps]
            return fmt, bpp, srgb  # no single/dual-channel sRGB formats in WebGPU
        raise ValueError(f"unsupported texture channel count: {comps}")

    def _write_payload(
        self, tex: wgpu.GPUTexture, pixels: np.ndarray, bpp: int, mip_level: int
    ) -> None:
        """Upload one mip level; pixels is a contiguous (layers, h, w, comps) u8 array."""
        layers, h, w, _ = pixels.shape
        row_bytes = (w * bpp + 255) // 256 * 256
        payload = pixels
        layout = {"bytes_per_row": w * bpp, "rows_per_image": h}
        if row_bytes != w * bpp:
            padded = np.zeros((layers, h, row_bytes), np.uint8)
            padded[:, :, : w * bpp] = pixels.reshape(layers, h, w * bpp)
            payload = padded
            layout = {"bytes_per_row": row_bytes, "rows_per_image": h}
        self._device.queue.write_texture(
            {"texture": tex, "mip_level": mip_level}, payload, layout, (w, h, layers)
        )

    def _upload(self, data: TextureData) -> wgpu.GPUTextureView:
        pixels = np.ascontiguousarray(data.pixels)
        h, w, comps = pixels.shape
        fmt, bpp, linearize = self._format_for(comps, data.srgb)
        if comps == 3:
            alpha = np.full((h, w, 1), 255, np.uint8)
            pixels = np.concatenate([pixels, alpha], axis=2)
        if linearize:
            pixels = np.ascontiguousarray(_srgb_to_linear_u8(pixels))
        levels = _mip_chain(pixels[None])
        tex = self._device.create_texture(
            size=(w, h, 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            mip_level_count=len(levels),
        )
        for mip_level, level in enumerate(levels):
            self._write_payload(tex, level, bpp, mip_level)
        return tex.create_view()

    def _upload_cube(self, data: TextureData) -> tuple[wgpu.GPUTextureView, int]:
        pixels = np.ascontiguousarray(data.pixels)
        _, size, _, comps = pixels.shape  # (6, S, S, C) u8
        fmt, bpp, linearize = self._format_for(comps, data.srgb)
        if comps == 3:
            alpha = np.full((6, size, size, 1), 255, np.uint8)
            pixels = np.concatenate([pixels, alpha], axis=3)
        if linearize:
            pixels = np.ascontiguousarray(_srgb_to_linear_u8(pixels))
        levels = _mip_chain(pixels)
        tex = self._device.create_texture(
            size=(size, size, 6),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            mip_level_count=len(levels),
        )
        for mip_level, level in enumerate(levels):
            self._write_payload(tex, level, bpp, mip_level)
        return tex.create_view(dimension="cube"), size

    def get(self, name: str | None) -> wgpu.GPUTextureView | None:
        if name is None:
            return None
        return self._textures.get(name)

    def cube(self, name: str | None) -> tuple[wgpu.GPUTextureView, int] | None:
        """Cube view and face size for an image-light or skybox texture."""
        if name is None:
            return None
        return self._cubes.get(name)

    @property
    def skybox(self) -> wgpu.GPUTextureView | None:
        entry = self._cubes.get(self._skybox_name) if self._skybox_name else None
        return entry[0] if entry is not None else None

    @property
    def black_cube(self) -> wgpu.GPUTextureView:
        if self._black_cube is None:
            tex = self._device.create_texture(
                size=(1, 1, 6),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self._write_payload(tex, np.zeros((6, 1, 1, 4), np.uint8), 4, 0)
            self._black_cube = tex.create_view(dimension="cube")
        return self._black_cube

    def release(self) -> None:
        self._textures.clear()
        self._cubes.clear()
        self._skybox_name = None
        self._white = None
        self._black_cube = None
