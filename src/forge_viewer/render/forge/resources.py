"""GPU mesh and texture resource stores."""

from __future__ import annotations

import moderngl
import numpy as np

from ...types import MeshData, MeshKey, MeshUpdate, TextureData, TextureKind
from .instances import GpuMesh


class MeshStore:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._meshes: dict[MeshKey, GpuMesh] = {}
        self._sources: dict[MeshKey, MeshData] = {}

    def sync(self, meshes: dict[MeshKey, MeshData]) -> None:
        for key, data in meshes.items():
            if key not in self._meshes or self._sources.get(key) is not data:
                old = self._meshes.pop(key, None)
                if old is not None:
                    old.release()
                self._meshes[key] = GpuMesh(
                    self.ctx, data.positions, data.normals, data.uvs, data.indices
                )
                self._sources[key] = data
        for key in [k for k in self._meshes if k not in meshes]:
            self._meshes.pop(key).release()
            self._sources.pop(key, None)

    def update(self, meshes: dict[MeshKey, MeshUpdate] | None) -> None:
        if not meshes:
            return
        for key, data in meshes.items():
            mesh = self._meshes.get(key)
            if mesh is not None:
                mesh.update(data.positions, data.normals)

    def get(self, key: MeshKey) -> GpuMesh | None:
        return self._meshes.get(key)

    def triangle_counts(self) -> dict[MeshKey, int]:
        return {k: m.triangle_count for k, m in self._meshes.items()}

    def release(self) -> None:
        for m in self._meshes.values():
            m.release()
        self._meshes.clear()
        self._sources.clear()


GL_SRGB8 = 0x8C41
GL_SRGB8_ALPHA8 = 0x8C43


def _srgb_internal_format(components: int) -> int | None:
    if components == 4:
        return GL_SRGB8_ALPHA8
    if components == 3:
        return GL_SRGB8
    return None


def srgb_to_linear_u8(pixels: np.ndarray) -> np.ndarray:
    out = pixels.astype(np.float32) / 255.0
    rgb = out[..., :3]
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    out[..., :3] = linear
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


class TextureStore:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._textures: dict[str, moderngl.Texture | moderngl.TextureCube] = {}
        self._skybox_name: str | None = None
        self._white: moderngl.Texture | None = None
        self._black_cube: moderngl.TextureCube | None = None

    @property
    def white(self) -> moderngl.Texture:
        if self._white is None:
            self._white = self.ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")
        return self._white

    @property
    def black_cube(self) -> moderngl.TextureCube:
        if self._black_cube is None:
            self._black_cube = self.ctx.texture_cube((1, 1), 3, bytes(18))
        return self._black_cube

    def sync(self, textures: dict[str, TextureData], skybox: str | None = None) -> None:
        for name, data in textures.items():
            if name in self._textures:
                continue
            pixels = data.pixels
            comps = pixels.shape[-1]

            fmt = _srgb_internal_format(comps) if data.srgb else None
            if data.srgb and fmt is None:
                pixels = srgb_to_linear_u8(pixels)

            if data.kind is TextureKind.TWO_D:
                h, w = pixels.shape[0], pixels.shape[1]
                tex = self._make_2d(w, h, comps, pixels, fmt)
                tex.repeat_x = tex.repeat_y = True
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex.anisotropy = 8.0
            else:
                size = pixels.shape[1]
                tex = self._make_cube(size, comps, pixels, fmt)
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self._textures[name] = tex
        for name in [k for k in self._textures if k not in textures]:
            self._textures.pop(name).release()
        self._skybox_name = skybox

    def _make_2d(self, w, h, comps, pixels, fmt):
        blob = np.ascontiguousarray(pixels).tobytes()
        if fmt is not None:
            try:
                return self.ctx.texture((w, h), comps, blob, internal_format=fmt)
            except Exception:
                return self.ctx.texture((w, h), comps, srgb_to_linear_u8(pixels).tobytes())
        return self.ctx.texture((w, h), comps, blob)

    def _make_cube(self, size, comps, pixels, fmt):
        blob = np.ascontiguousarray(pixels).tobytes()
        if fmt is not None:
            try:
                return self.ctx.texture_cube((size, size), comps, blob, internal_format=fmt)
            except Exception:
                return self.ctx.texture_cube(
                    (size, size), comps, srgb_to_linear_u8(pixels).tobytes()
                )
        return self.ctx.texture_cube((size, size), comps, blob)

    def get(self, name: str | None) -> moderngl.Texture | moderngl.TextureCube | None:
        return self._textures.get(name) if name else None

    @property
    def skybox(self) -> moderngl.TextureCube | None:
        tex = self._textures.get(self._skybox_name) if self._skybox_name else None
        return tex if isinstance(tex, moderngl.TextureCube) else None

    def release(self) -> None:
        for t in self._textures.values():
            t.release()
        self._textures.clear()
        if self._white is not None:
            self._white.release()
            self._white = None
        if self._black_cube is not None:
            self._black_cube.release()
            self._black_cube = None
