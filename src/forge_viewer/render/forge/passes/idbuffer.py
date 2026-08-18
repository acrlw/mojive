"""Object ID buffer generation for picking and outlines."""

from __future__ import annotations

import moderngl

from .... import math3d as M
from ....log import get_logger
from ...backend import RenderFlag
from .. import gl_native as G
from ..instances import (
    INSTANCE_ATTRIBUTES,
    INSTANCE_BYTES,
    MESH_ATTRIBUTES,
    Strategy,
    build_layout,
)
from ..programs import ProgramSpec
from ..registry import register_pass
from .base import BasePass, PassContext, state_opaque

log = get_logger("id")


class IdGeometry:
    def __init__(self, only_selected: bool = False, float_mask: bool = False) -> None:
        self._only_selected = bool(only_selected)
        self._float_mask = bool(float_mask)
        self.program: moderngl.Program | None = None
        self._spec: ProgramSpec | None = None
        self._attachment = -1
        self._generation = -1
        self._vaos: list[moderngl.VertexArray | None] = []
        self._own: list[moderngl.Buffer | None] = []

        self._buffer_glo = -1
        self._ranges: object = None
        self._meshes: object = None
        self._strategy: Strategy | None = None
        self._broken = False

    def ensure(self, ctx: PassContext, attachment: int) -> bool:
        fresh = False
        if (
            self.program is None
            or self._attachment != attachment
            or self._generation != ctx.programs.generation
        ):
            self._attachment = int(attachment)
            self._spec = self._make_spec(self._attachment)
            self.program = ctx.programs.get(self._spec)
            self._generation = ctx.programs.generation
            fresh = True

        store = ctx.instances
        buf = store.buffer
        if buf is None:
            return False
        if (
            fresh
            or self._buffer_glo != buf.glo
            or self._ranges is not ctx.scene.bucket_ranges
            or self._meshes is not ctx.meshes
            or self._strategy is not store.strategy
        ):
            self._rebuild(ctx)
        return bool(self._vaos) and not self._broken

    def _make_spec(self, attachment: int) -> ProgramSpec:
        defines: dict[str, object] = {"ID_ATTACHMENT": attachment}
        if self._only_selected:
            defines["ID_ONLY_SELECTED"] = 1
        if self._float_mask:
            defines["ID_MASK_FLOAT"] = 1
        return ProgramSpec(name="id", vertex="id.vert", fragment="id.frag", defines=defines)

    def _rebuild(self, ctx: PassContext) -> None:
        self._release_gl()
        prog, store, scene = self.program, ctx.instances, ctx.scene
        buf = store.buffer
        assert prog is not None and buf is not None
        self._buffer_glo = buf.glo
        self._ranges = scene.bucket_ranges
        self._meshes = ctx.meshes
        self._strategy = store.strategy
        self._broken = False

        mesh_layout, mesh_names = build_layout(prog, MESH_ATTRIBUTES)
        inst_layout, inst_names = build_layout(prog, INSTANCE_ATTRIBUTES, per_instance=True)
        attrs = tuple(
            (self._location(prog, name), comps, off, gl_type)
            for name, _fmt, _nbytes, comps, off, gl_type in INSTANCE_ATTRIBUTES
        )
        shared = store.strategy is Strategy.SHARED

        for b, (start, stop) in enumerate(scene.bucket_ranges):
            mesh = ctx.meshes[b] if b < len(ctx.meshes) else None
            if mesh is None:
                self._vaos.append(None)
                self._own.append(None)
                continue
            own = None
            src = buf
            if not shared:
                own = ctx.ctx.buffer(reserve=max(1, stop - start) * INSTANCE_BYTES)
                src = own
            vao = ctx.ctx.vertex_array(
                prog,
                [(mesh.vbo, mesh_layout, *mesh_names), (src, inst_layout, *inst_names)],
                mesh.ibo,
                index_element_size=4,
            )
            if shared and not G.native().rebind_instance_attributes(
                vao.glo, buf.glo, INSTANCE_BYTES, start * INSTANCE_BYTES, attrs
            ):
                vao.release()
                self._release_gl()
                self._broken = True
                log.error(
                    "ID pass could not bind the instance offset; GPU picking is disabled for this frame"
                )
                return
            self._vaos.append(vao)
            self._own.append(own)

    @staticmethod
    def _location(program: moderngl.Program, name: str) -> int:
        try:
            return int(program[name].location)
        except KeyError:
            return -1

    def upload(self, ctx: PassContext) -> None:
        if self._strategy is not Strategy.PER_BUCKET:
            return
        data = ctx.instances.pack(ctx.scene)
        for b, (start, stop) in enumerate(ctx.scene.bucket_ranges):
            buf = self._own[b] if b < len(self._own) else None
            if buf is not None and stop > start:
                buf.write(data[start:stop])

    def set_view_proj(self, ctx: PassContext) -> None:
        assert self.program is not None

        self.program["u_view_proj"].write(M.to_gl(ctx.view_proj))

    def draw(self, ctx: PassContext, buckets) -> int:
        ranges = ctx.scene.bucket_ranges
        calls = 0
        for b in buckets:
            vao = self._vaos[b] if 0 <= b < len(self._vaos) else None
            if vao is None:
                continue
            start, stop = ranges[b]
            if stop <= start:
                continue
            vao.render(moderngl.TRIANGLES, instances=stop - start)
            calls += 1
        return calls

    def _release_gl(self) -> None:
        for vao in self._vaos:
            if vao is not None:
                vao.release()
        self._vaos.clear()
        for buf in self._own:
            if buf is not None:
                buf.release()
        self._own.clear()

    def release(self) -> None:
        self._release_gl()
        self.program = None
        self._buffer_glo = -1
        self._ranges = None
        self._meshes = None
        self._strategy = None


class IdBufferPass(BasePass):
    name = "id"

    def __init__(self) -> None:
        self._geom = IdGeometry()

    def prepare(self, ctx: PassContext) -> bool:
        ctx.target.clear_id(0)
        if not self._geom.ensure(ctx, ctx.target.id_draw_buffer):
            return False
        self._geom.upload(ctx)
        return bool(
            ctx.scene.opaque_buckets
            or (
                ctx.include_transparent_ids
                and ctx.scene.transparent_buckets
                and ctx.flag(RenderFlag.TRANSPARENT)
            )
        )

    def execute(self, ctx: PassContext) -> None:
        target = ctx.target
        fbo = target.id_fbo
        shared = fbo is target.fbo

        if shared:
            fbo.depth_mask = False

            fbo.color_mask = ((False, False, False, False), (True, True, True, True))
        else:
            fbo.depth_mask = True
        target.use_id()
        state_opaque(ctx.ctx)
        ctx.ctx.depth_func = "<=" if shared else "<"

        self._geom.set_view_proj(ctx)
        ctx.draw_calls += self._geom.draw(ctx, ctx.scene.opaque_buckets)
        if ctx.include_transparent_ids and ctx.flag(RenderFlag.TRANSPARENT):
            ctx.draw_calls += self._geom.draw(ctx, ctx.scene.transparent_draw_order())

        if shared:
            fbo.color_mask = ((True, True, True, True), (True, True, True, True))
            fbo.depth_mask = True
        ctx.ctx.depth_func = "<"

    def release(self) -> None:
        self._geom.release()


# Importing through the package here would create a register_pass initialization cycle.

register_pass("id", IdBufferPass)
