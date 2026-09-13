"""Spatial tendon / actuator paths rendered as instanced 3D capsules."""

from __future__ import annotations

import moderngl

from ....types import MeshKey, MeshShape
from ...backend import RenderFlag
from ...tendon import TendonScene
from ..instances import GpuMesh, InstanceStore
from ..registry import register_pass
from ..targets import IdLayout
from .base import BasePass, PassContext, state_opaque, state_transparent

_SHAFT = MeshKey(MeshShape.CAPSULE_SHAFT)
_CAP = MeshKey(MeshShape.CAPSULE_CAP)


class TendonPass(TendonScene, BasePass):
    name = "tendon"

    def __init__(self) -> None:
        super().__init__()
        self._store: InstanceStore | None = None
        self._shaft_mesh: GpuMesh | None = None
        self._cap_mesh: GpuMesh | None = None
        self._bucket_meshes: list[GpuMesh] = []
        self._program: moderngl.Program | None = None

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
