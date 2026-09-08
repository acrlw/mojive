"""Backend-neutral spatial tendon geometry and actuator publishing."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np

from ..adapters.base import SceneFrame
from ..types import DEFAULT_MATERIAL, MeshKey, MeshShape
from .backend import RenderFlag
from .scene import RenderScene

_SHAFT = MeshKey(MeshShape.CAPSULE_SHAFT)
_CAP = MeshKey(MeshShape.CAPSULE_CAP)


class TendonScene:
    def __init__(self):
        self._scene = RenderScene()
        self._count = 0
        self._material_ids = np.zeros(0, np.int32)
        self._transparent = np.zeros(0, bool)

    @property
    def scene(self):
        return self._scene

    @property
    def capsule_count(self):
        return self._count

    def update(
        self,
        segments,
        widths,
        colors,
        materials,
        material_ids,
        transparent,
        material_table,
    ) -> None:
        n = len(segments)
        material_ids = np.asarray(material_ids, np.int32)
        transparent = np.asarray(transparent, bool)
        if (
            n != self._count
            or material_table is not self._scene.materials
            or not np.array_equal(material_ids, self._material_ids)
            or not np.array_equal(transparent, self._transparent)
        ):
            self._resize(n, material_ids, transparent, material_table)
        if not n:
            return

        segments = np.asarray(segments, np.float32)
        widths = np.asarray(widths, np.float32)
        delta = segments[:, 1] - segments[:, 0]
        length = np.linalg.norm(delta, axis=1)
        z = delta / np.maximum(length[:, None], 1e-12)
        helper = np.zeros_like(z)
        helper[:, 2] = 1.0
        steep = np.abs(z[:, 2]) > 0.9
        helper[steep] = (0.0, 1.0, 0.0)
        x = np.cross(helper, z)
        x /= np.maximum(np.linalg.norm(x, axis=1)[:, None], 1e-12)
        y = np.cross(z, x)

        transforms = self._scene.transforms
        transforms.fill(0.0)
        transforms[:, 3, 3] = 1.0
        half = 0.5 * length

        transforms[:n, :3, 0] = x * widths[:, None]
        transforms[:n, :3, 1] = y * widths[:, None]
        transforms[:n, :3, 2] = z * half[:, None]
        transforms[:n, :3, 3] = 0.5 * (segments[:, 0] + segments[:, 1])

        transforms[n : 2 * n, :3, 0] = x * widths[:, None]
        transforms[n : 2 * n, :3, 1] = y * widths[:, None]
        transforms[n : 2 * n, :3, 2] = z * widths[:, None]
        transforms[n : 2 * n, :3, 3] = segments[:, 1]

        transforms[2 * n :, :3, 0] = x * widths[:, None]
        transforms[2 * n :, :3, 1] = -y * widths[:, None]
        transforms[2 * n :, :3, 2] = -z * widths[:, None]
        transforms[2 * n :, :3, 3] = segments[:, 0]

        rgba = np.asarray(colors, np.float32)
        for start in (0, n, 2 * n):
            dst = self._scene.colors[start : start + n]
            np.power(np.clip(rgba[:, :3], 0.0, 1.0), 2.2, out=dst[:, :3])
            dst[:, 3] = rgba[:, 3]
            self._scene.material[start : start + n] = materials
            self._scene.material[start : start + n, 3] = 0.0

    def clear(self) -> None:
        if self._count:
            self._resize(0, np.zeros(0, np.int32), np.zeros(0, bool), self._scene.materials)

    def _resize(self, n: int, material_ids, transparent, material_table) -> None:
        count = 3 * n
        scene = RenderScene(count=count)
        scene.transforms = np.zeros((count, 4, 4), np.float32)
        scene.colors = np.ones((count, 4), np.float32)
        scene.material = np.tile(np.array((0.0, 0.5, 0.5, 0.0), np.float32), (count, 1))
        scene.tex_coef = np.tile(np.array((1.0, 1.0, 0.0, 0.0), np.float32), (count, 1))
        scene.cube_coef = np.zeros((count, 4), np.float32)
        scene.object_id = np.zeros(count, np.uint32)
        scene.bucket = np.zeros(count, np.int32)
        scene.materials = material_table or (DEFAULT_MATERIAL,)
        boundaries = [0]
        for i in range(1, n):
            if material_ids[i] != material_ids[i - 1] or transparent[i] != transparent[i - 1]:
                boundaries.append(i)
        boundaries.append(n)
        runs = list(pairwise(boundaries)) if n else []
        keys = []
        ranges = []
        opaque = []
        translucent = []
        for mesh, offset in ((_SHAFT, 0), (_CAP, n), (_CAP, 2 * n)):
            for start, stop in runs:
                bucket = len(keys)
                matid = int(material_ids[start])
                keys.append((mesh, matid))
                ranges.append((offset + start, offset + stop))
                scene.bucket[offset + start : offset + stop] = bucket
                (translucent if transparent[start] else opaque).append(bucket)
                mat = scene.materials[matid]
                scene.tex_coef[offset + start : offset + stop, :2] = mat.tex_repeat
        scene.bucket_keys = tuple(keys)
        scene.bucket_ranges = tuple(ranges)
        scene.opaque_buckets = tuple(opaque)
        scene.transparent_buckets = tuple(translucent)
        self._scene = scene
        self._count = n
        self._material_ids = material_ids.copy()
        self._transparent = transparent.copy()


class TendonPublisher:
    def __init__(self, output, overlay, get_flag):
        self._tendons, self._overlay, self.get_flag = output, overlay, get_flag
        self._source = None
        self._tendon_visible = np.zeros(0, bool)
        self._actuator_visible = np.zeros(0, bool)
        self._material_values = np.zeros((0, 4), np.float32)
        self._tendon_material_table: tuple = ()
        self._island_tendon_material_table: tuple = ()
        self._tendon_actuator = np.zeros(0, np.int32)
        self._capsule_segments = np.zeros((0, 2, 3), np.float32)
        self._capsule_widths = np.zeros(0, np.float32)
        self._capsule_colors = np.zeros((0, 4), np.float32)
        self._capsule_materials = np.zeros((0, 4), np.float32)
        self._capsule_material_ids = np.zeros(0, np.int32)
        self._capsule_transparent = np.zeros(0, bool)

    def set_scene(self, source):
        self._source = source
        self._tendon_visible = source.tendon_visible
        self._actuator_visible = source.actuator_visible
        self._material_values = np.asarray(
            [
                (mat.emission, mat.specular, mat.shininess, mat.reflectance)
                for mat in source.materials
            ],
            np.float32,
        )
        self._tendon_material_table = tuple(source.materials)
        self._island_tendon_material_table = tuple(
            replace(material, texture=None) for material in source.materials
        )
        self._tendon_actuator = np.full(len(source.tendon_rgba), -1, np.int32)
        for actuator, tendon in enumerate(source.actuator_tendon):
            if 0 <= tendon < len(self._tendon_actuator):
                self._tendon_actuator[tendon] = actuator

    def update(self, frame: SceneFrame) -> None:
        """Pack visible tendon segments into capsule instances (opengl parity)."""
        if self._tendons is None:
            return
        segments, ids, widths = (
            frame.tendon_segments,
            frame.tendon_ids,
            frame.tendon_widths,
        )
        if segments is None or ids is None or widths is None or not len(segments):
            self._tendons.clear()
            return

        base_indices = (
            np.flatnonzero(self._tendon_visible[ids])
            if self.get_flag(RenderFlag.TENDON)
            else np.zeros(0, np.intp)
        )
        base_count = len(base_indices)
        actuator_indices = np.zeros(0, np.intp)
        segment_actuators = np.zeros(0, np.int32)
        if self.get_flag(RenderFlag.ACTUATOR) and frame.ctrl is not None:
            segment_actuators = self._tendon_actuator[ids]
            available = segment_actuators >= 0
            available[available] &= self._actuator_visible[segment_actuators[available]]
            actuator_indices = np.flatnonzero(available)
        total = base_count + len(actuator_indices)
        if not total:
            self._tendons.clear()
            return

        if total > len(self._capsule_widths):
            capacity = max(total, 2 * len(self._capsule_widths), 64)
            self._capsule_segments = np.zeros((capacity, 2, 3), np.float32)
            self._capsule_widths = np.zeros(capacity, np.float32)
            self._capsule_colors = np.zeros((capacity, 4), np.float32)
            self._capsule_materials = np.zeros((capacity, 4), np.float32)
            self._capsule_material_ids = np.zeros(capacity, np.int32)
            self._capsule_transparent = np.zeros(capacity, bool)

        assert self._source is not None
        if base_count:
            self._capsule_segments[:base_count] = segments[base_indices]
            self._capsule_widths[:base_count] = widths[base_indices]
            tendon_rgba = self._source.tendon_rgba
            if self.get_flag(RenderFlag.ISLAND) and frame.tendon_island_rgba is not None:
                tendon_rgba = frame.tendon_island_rgba
            np.take(
                tendon_rgba,
                ids[base_indices],
                axis=0,
                out=self._capsule_colors[:base_count],
                mode="clip",
            )
            np.take(
                self._material_values,
                self._source.tendon_material[ids[base_indices]],
                axis=0,
                out=self._capsule_materials[:base_count],
                mode="clip",
            )
            self._capsule_material_ids[:base_count] = self._source.tendon_material[
                ids[base_indices]
            ]

        if len(actuator_indices):
            palette = self._overlay.fill_actuator_palette(frame)
            start = base_count
            stop = start + len(actuator_indices)
            self._capsule_segments[start:stop] = segments[actuator_indices]
            self._capsule_widths[start:stop] = (
                widths[actuator_indices] * self._source.actuator_tendon_scale
            )
            np.take(
                palette,
                segment_actuators[actuator_indices],
                axis=0,
                out=self._capsule_colors[start:stop],
            )
            np.take(
                self._material_values,
                self._source.tendon_material[ids[actuator_indices]],
                axis=0,
                out=self._capsule_materials[start:stop],
                mode="clip",
            )
            self._capsule_material_ids[start:stop] = self._source.tendon_material[
                ids[actuator_indices]
            ]

        np.less(
            self._capsule_colors[:total, 3],
            1.0,
            out=self._capsule_transparent[:total],
        )

        material_table = (
            self._island_tendon_material_table
            if self.get_flag(RenderFlag.ISLAND)
            else self._tendon_material_table
        )
        self._tendons.update(
            self._capsule_segments[:total],
            self._capsule_widths[:total],
            self._capsule_colors[:total],
            self._capsule_materials[:total],
            self._capsule_material_ids[:total],
            self._capsule_transparent[:total],
            material_table,
        )
