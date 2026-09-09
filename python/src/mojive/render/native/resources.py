"""Weak identity cache for immutable public scene resources."""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass

import numpy as np

from ...types import TextureType
from ..texture import srgb_to_linear_u8


@dataclass
class PreparedResource:
    value: object


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
        if array.size:
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.digest()


class ResourceCache:
    """Share unchanged assets across source recompiles and independent scenes.

    Identity hits avoid hashing entirely. SHA-256 content keys cover adapters that
    recreate equivalent arrays. Backend leases keep entries alive during atomic
    replacement; weak caches release unused CPU storage without a growing LRU.
    MeshUpdate supplies scene-local deformation, independently of these assets.
    """

    def __init__(self, api):
        self.api = api
        self._meshes = {}
        self._textures = {}
        self._content = weakref.WeakValueDictionary()

    def _get(self, cache, source, fingerprint, prepare):
        key = id(source)
        entry = cache.get(key)
        if entry is not None and entry[0]() is source:
            return entry[1]
        content_key = fingerprint()
        value = self._content.get(content_key)
        if value is None:
            value = PreparedResource(prepare())
            self._content[content_key] = value

        def discard(reference):
            if cache.get(key, (None,))[0] is reference:
                cache.pop(key, None)

        cache[key] = (weakref.ref(source, discard), value)
        return value

    def clear(self):
        self._meshes.clear()
        self._textures.clear()
        self._content.clear()

    def mesh(self, mesh):
        return self._get(
            self._meshes,
            mesh,
            lambda: ("mesh", _digest(mesh.positions, mesh.normals, mesh.indices, mesh.uvs)),
            lambda: self.api.prepare_mesh(mesh.positions, mesh.normals, mesh.indices, mesh.uvs),
        )

    def texture(self, texture):
        def prepare():
            pixels, srgb = texture.pixels, texture.srgb
            if srgb and pixels.shape[-1] < 3:
                pixels = srgb_to_linear_u8(pixels)
                srgb = False
            cube = texture.type is not TextureType.TWO_D
            if not cube:
                pixels = pixels[None]
            return self.api.prepare_texture(np.ascontiguousarray(pixels), cube, srgb)

        return self._get(
            self._textures,
            texture,
            lambda: (
                "texture",
                texture.type is not TextureType.TWO_D,
                texture.srgb,
                _digest(texture.pixels),
            ),
            prepare,
        )
