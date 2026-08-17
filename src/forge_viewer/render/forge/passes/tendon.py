"""Spatial tendon / actuator paths rendered as instanced 3D capsules."""

from __future__ import annotations

from itertools import pairwise

import moderngl
import numpy as np

from ....types import DEFAULT_MATERIAL, MeshKey, MeshShape
from ...backend import RenderFlag
from ...scene import RenderScene
from ..instances import GpuMesh, InstanceStore
from ..registry import register_pass
from ..targets import IdLayout
from .base import BasePass, PassContext, state_opaque, state_transparent

_SHAFT = MeshKey(MeshShape.CAPSULE_SHAFT)
_CAP = MeshKey(MeshShape.CAPSULE_CAP)


class TendonPass(BasePass):
    name = "tendon"

    def __init__(self) -> None:
        self._scene = RenderScene()
        self._count = 0
        self._store: InstanceStore | None = None
        self._shaft_mesh: GpuMesh | None = None
        self._cap_mesh: GpuMesh | None = None
        self._bucket_meshes: list[GpuMesh] = []
        self._program: moderngl.Program | None = None
        self._material_ids = np.zeros(0, np.int32)
        self._transparent = np.zeros(0, bool)

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

    @property
    def capsule_count(self) -> int:
        return self._count

    def _resize(self, n: int, material_ids, transparent, material_table) -> None:
        count = 3 * n
        scene = RenderScene(count=count)
        scene.transforms = np.zeros((count, 4, 4), np.float32)
        scene.colors = np.ones((count, 4), np.float32)
        scene.material = np.tile(np.array((0.0, 0.5, 0.5, 0.0), np.float32), (count, 1))
        scene.tex_coef = np.tile(np.array((1.0, 1.0, 0.0, 0.0), np.float32), (count, 1))
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

    def prepare(self, ctx: PassContext) -> bool:
        return bool(self._count and ctx.scene_program is not None)

    def execute(self, ctx: PassContext) -> None:
        program = ctx.scene_program
        if program is None:
            return
        self._sync(ctx, program)
        assert self._store is not None

        ctx.target.use_main()
        shared_id = ctx.target.id_layout is IdLayout.SHARED
        if shared_id:
            ctx.target.fbo.color_mask = (
                (True, True, True, True),
                (False, False, False, False),
            )
        self._store.upload(self._scene)
        state_opaque(ctx.ctx)
        if not ctx.flag(RenderFlag.CULL_FACE):
            ctx.ctx.disable(moderngl.CULL_FACE)
        ctx.ctx.multisample = bool(ctx.flag(RenderFlag.MSAA))
        ctx.target.fbo.depth_mask = True
        self._draw(ctx, self._scene.opaque_buckets)
        if self._scene.transparent_buckets and ctx.flag(RenderFlag.TRANSPARENT):
            state_transparent(
                ctx.ctx,
                additive=ctx.flag(RenderFlag.ADDITIVE, False),
            )
            if not ctx.flag(RenderFlag.CULL_FACE):
                ctx.ctx.disable(moderngl.CULL_FACE)
            ctx.target.fbo.depth_mask = False
            self._draw(ctx, self._scene.transparent_draw_order())
            ctx.target.fbo.depth_mask = True
        if shared_id:
            ctx.target.fbo.color_mask = (
                (True, True, True, True),
                (True, True, True, True),
            )

    def _sync(self, ctx: PassContext, program: moderngl.Program) -> None:
        if self._store is None:
            from ...mesh import builtin_mesh

            self._store = InstanceStore(ctx.ctx)
            shaft, cap = builtin_mesh(_SHAFT), builtin_mesh(_CAP)
            self._shaft_mesh = GpuMesh(
                ctx.ctx, shaft.positions, shaft.normals, shaft.uvs, shaft.indices
            )
            self._cap_mesh = GpuMesh(ctx.ctx, cap.positions, cap.normals, cap.uvs, cap.indices)
        self._bucket_meshes = [
            self._shaft_mesh if key[0] == _SHAFT else self._cap_mesh
            for key in self._scene.bucket_keys
        ]
        if self._program is not program or self._store.needs_rebuild(
            self._scene, ctx.programs.generation
        ):
            self._store.rebuild(self._scene, program, self._bucket_meshes, ctx.programs.generation)
            self._program = program

    def _draw(self, ctx: PassContext, buckets) -> None:
        assert self._store is not None
        for bucket in buckets:
            mat = self._scene.materials[self._scene.bucket_keys[bucket][1]]
            texture = ctx.textures.get(mat.texture) if ctx.flag(RenderFlag.TEXTURE) else None
            (texture if isinstance(texture, moderngl.Texture) else ctx.textures.white).use(0)
            drawn = self._store.draw(bucket)
            if drawn:
                ctx.draw_calls += 1
                ctx.instance_count += drawn

    def release(self) -> None:
        if self._store is not None:
            self._store.release()
        if self._shaft_mesh is not None:
            self._shaft_mesh.release()
        if self._cap_mesh is not None:
            self._cap_mesh.release()
        self._shaft_mesh = self._cap_mesh = None
        self._bucket_meshes.clear()
        self._store = None
        self._program = None


register_pass("tendon", TendonPass)
