"""Debug primitive and world-space text render pass."""

from __future__ import annotations

import moderngl
import numpy as np

from ....log import get_logger
from ....types import MeshKey
from ...debugdraw import RECORD_FLOATS, DebugDraw, DrawPath, Occlusion, PackedFrame
from .. import gl_native as G
from ..backend import register_pass
from ..instances import GpuMesh
from ..programs import ProgramSpec
from ..targets import IdLayout
from ..text import TextRenderer
from .base import BasePass, PassContext, state_overlay

log = get_logger("debug")

SECTOR_SEGMENTS = 32


STROKE_JOIN_SEGMENTS = 6


GHOST_ALPHA = 0.28


_SPECS = {
    DrawPath.SEGMENT: ProgramSpec("debug_line", "debug_line.vert", "debug_line.frag"),
    DrawPath.STROKE: ProgramSpec(
        "debug_stroke",
        "debug_stroke.vert",
        "debug_line.frag",
        defines={"STROKE_JOIN_SEGMENTS": STROKE_JOIN_SEGMENTS},
    ),
    DrawPath.POINT: ProgramSpec("debug_point", "debug_point.vert", "debug_point.frag"),
    DrawPath.DRAG_LINK: ProgramSpec(
        "debug_drag_link", "debug_drag_link.vert", "debug_drag_link.frag"
    ),
    DrawPath.SOLID: ProgramSpec("debug_solid", "debug_solid.vert", "debug_solid.frag"),
    DrawPath.SECTOR: ProgramSpec(
        "debug_sector",
        "debug_sector.vert",
        "debug_sector.frag",
        defines={"SECTOR_SEGMENTS": SECTOR_SEGMENTS},
    ),
}
_LAYOUT: dict[DrawPath, str] = {
    DrawPath.SEGMENT: "3f 3f 4f 1f 1f 1f/i",
    DrawPath.STROKE: "3f 3f 3f 4f 1f/i",
    DrawPath.POINT: "3f 4f 1f/i",
    DrawPath.DRAG_LINK: "3f 3f 4f 4f 1f 1f 1f/i",
    DrawPath.SOLID: "4f 4f 4f 4f 4f/i",
    DrawPath.SECTOR: "3f 3f 3f 4f 1f/i",
}
_ATTRS: dict[DrawPath, tuple[tuple[str, int, int], ...]] = {
    DrawPath.SEGMENT: (
        ("in_a", 3, 0),
        ("in_b", 3, 12),
        ("in_color", 4, 24),
        ("in_width", 1, 40),
        ("in_head", 1, 44),
        ("in_start_mask", 1, 48),
    ),
    DrawPath.STROKE: (
        ("in_prev", 3, 0),
        ("in_a", 3, 12),
        ("in_b", 3, 24),
        ("in_color", 4, 36),
        ("in_width", 1, 52),
    ),
    DrawPath.POINT: (("in_p", 3, 0), ("in_color", 4, 12), ("in_radius", 1, 28)),
    DrawPath.DRAG_LINK: (
        ("in_a", 3, 0),
        ("in_b", 3, 12),
        ("in_core_color", 4, 24),
        ("in_edge_color", 4, 40),
        ("in_width", 1, 56),
        ("in_radius", 1, 60),
        ("in_edge", 1, 64),
    ),
    DrawPath.SOLID: (
        ("in_model0", 4, 0),
        ("in_model1", 4, 16),
        ("in_model2", 4, 32),
        ("in_model3", 4, 48),
        ("in_color", 4, 64),
    ),
    DrawPath.SECTOR: (
        ("in_center", 3, 0),
        ("in_rot_end", 3, 12),
        ("in_ref_end", 3, 24),
        ("in_color", 4, 36),
        ("in_radius", 1, 52),
    ),
}
_VERTICES: dict[DrawPath, int] = {
    DrawPath.SEGMENT: 15,
    DrawPath.STROKE: 6 + 3 * STROKE_JOIN_SEGMENTS,
    DrawPath.POINT: 6,
    DrawPath.DRAG_LINK: 6,
    DrawPath.SECTOR: 3 * SECTOR_SEGMENTS,
}
_MESH_LAYOUT = ("3f 3f 8x", ("in_position", "in_normal"))


class DebugPass(BasePass):
    name = "debug"

    def __init__(self) -> None:
        self.draw = DebugDraw()
        self.draw_calls = 0
        self._gl = G.native()
        self._buffers: dict[DrawPath, moderngl.Buffer | None] = dict.fromkeys(DrawPath, None)
        self._vaos: dict[tuple[DrawPath, MeshKey | None], moderngl.VertexArray] = {}
        self._progs: dict[DrawPath, moderngl.Program] = {}
        self._members: dict[DrawPath, frozenset[str]] = {}
        self._locs: dict[DrawPath, tuple[tuple[int, int, int], ...]] = {}
        self._meshes: dict[MeshKey, GpuMesh | None] = {}
        self._generation = -1
        self._broken = ""

        self._mats = np.zeros((3, 4, 4), np.float32)
        self._ctx: PassContext | None = None
        self._emit_bound = self._emit
        self._text = TextRenderer()

    def prepare(self, ctx: PassContext) -> bool:
        self.draw_calls = 0
        if self._broken:
            return False
        return self.draw.primitives > 0

    def execute(self, ctx: PassContext) -> None:
        if not self._sync_programs(ctx):
            return
        ctx.target.use_main()
        self._ctx = ctx

        self.draw.render_frame(self._emit_bound, now=ctx.time or None)

    def configure_text(
        self,
        primary: str = "",
        primary_index: int = 0,
        fallback: str = "",
        fallback_index: int = 0,
        size_px: float = 14.0,
    ) -> None:
        self._text.configure(primary, primary_index, fallback, fallback_index, size_px)

    def _sync_programs(self, ctx: PassContext) -> bool:
        if self._progs and self._generation == ctx.programs.generation:
            return True
        try:
            progs = {p: ctx.programs.get(spec) for p, spec in _SPECS.items()}
        except Exception as e:
            self._broken = str(e)
            log.error(
                "Debug draw shader compilation failed; the pass is disabled:\n{}", self._broken
            )
            return False
        self._progs = progs
        self._members = {p: frozenset(prog) for p, prog in progs.items()}

        self._locs = {
            p: tuple(
                (int(progs[p][name].location), comps, off, G.GL_FLOAT)
                for name, comps, off in attrs
                if name in self._members[p]
            )
            for p, attrs in _ATTRS.items()
        }
        self._release_vaos()
        self._generation = ctx.programs.generation
        return True

    def _ensure_buffer(self, ctx: PassContext, path: DrawPath, records: int) -> moderngl.Buffer:
        stride = RECORD_FLOATS[path] * 4
        buf = self._buffers[path]
        if buf is not None and buf.size >= records * stride:
            return buf
        need = max(records, (buf.size // stride * 2) if buf is not None else 0, 64)
        if buf is not None:
            buf.release()
        buf = ctx.ctx.buffer(reserve=need * stride)
        self._buffers[path] = buf
        for key in [k for k in self._vaos if k[0] is path]:
            self._vaos.pop(key).release()
        return buf

    def _vao(
        self, ctx: PassContext, path: DrawPath, mesh: MeshKey | None
    ) -> moderngl.VertexArray | None:
        vao = self._vaos.get((path, mesh))
        if vao is not None:
            return vao
        prog = self._progs[path]
        buf = self._buffers[path]
        if buf is None:
            return None
        names = tuple(n for n, _c, _o in _ATTRS[path] if n in self._members[path])
        content: list = []
        if mesh is not None:
            gpu = self._mesh(ctx, mesh)
            if gpu is None:
                return None
            content.append((gpu.vbo, _MESH_LAYOUT[0], *_MESH_LAYOUT[1]))
            content.append((buf, _LAYOUT[path], *names))
            vao = ctx.ctx.vertex_array(prog, content, gpu.ibo, index_element_size=4)
        else:
            vao = ctx.ctx.vertex_array(prog, [(buf, _LAYOUT[path], *names)])
        self._vaos[(path, mesh)] = vao
        return vao

    def _mesh(self, ctx: PassContext, key: MeshKey) -> GpuMesh | None:
        if key in self._meshes:
            return self._meshes[key]
        gpu: GpuMesh | None = None
        try:
            from ...mesh import builtin_mesh

            data = builtin_mesh(key)
            gpu = GpuMesh(ctx.ctx, data.positions, data.normals, data.uvs, data.indices)
        except Exception as e:
            log.error(
                "Built-in mesh {} is unavailable; solid annotations cannot be drawn: {}", key, e
            )
        self._meshes[key] = gpu
        return gpu

    def _emit(self, frame: PackedFrame) -> None:
        ctx = self._ctx
        assert ctx is not None
        text_ready = self._text.prepare(ctx, frame.texts, frame.text_count)

        for path in DrawPath:
            n = frame.counts[path]
            if n:
                self._ensure_buffer(ctx, path, n).write(frame.stream(path))
                self._set_common(ctx, path)

        fbo = ctx.target.fbo
        shared_id = ctx.target.id_layout is IdLayout.SHARED and ctx.target.id_fbo is fbo
        if shared_id:
            fbo.color_mask = ((True, True, True, True), (False, False, False, False))

        i = 0
        while i < frame.batch_count:
            occ = frame.batches[i].occlusion
            j = i
            while j < frame.batch_count and frame.batches[j].occlusion is occ:
                j += 1
            if occ is Occlusion.GHOST:
                self._state(ctx, depth_test=True, depth_func="<")
                self._draw_range(ctx, frame, i, j, 1.0)
                self._state(ctx, depth_test=True, depth_func=">")
                self._draw_range(ctx, frame, i, j, GHOST_ALPHA)
            else:
                self._state(ctx, depth_test=occ is Occlusion.DEPTH, depth_func="<")
                self._draw_range(ctx, frame, i, j, 1.0)
            i = j

        if text_ready:
            for batch in self._text.batches():
                if batch.occlusion is Occlusion.GHOST:
                    self._state(ctx, depth_test=True, depth_func="<")
                    self._text.draw(ctx, batch, 1.0)
                    self._state(ctx, depth_test=True, depth_func=">")
                    self._text.draw(ctx, batch, GHOST_ALPHA)
                    calls = 2
                else:
                    self._state(ctx, depth_test=batch.occlusion is Occlusion.DEPTH, depth_func="<")
                    self._text.draw(ctx, batch, 1.0)
                    calls = 1
                self.draw_calls += calls
                ctx.draw_calls += calls

        if shared_id:
            fbo.color_mask = ((True, True, True, True), (True, True, True, True))

    def _state(self, ctx: PassContext, depth_test: bool, depth_func: str) -> None:
        state_overlay(ctx.ctx, depth_test)
        ctx.ctx.depth_func = depth_func

        ctx.target.fbo.depth_mask = False

        ctx.ctx.wireframe = False

    def _draw_range(
        self, ctx: PassContext, frame: PackedFrame, i0: int, i1: int, alpha: float
    ) -> None:
        for i in range(i0, i1):
            b = frame.batches[i]
            prog = self._progs[b.path]
            vao = self._vao(ctx, b.path, b.mesh)
            if vao is None:
                self.draw.drop(b.count, f"{b.path} batch is missing a mesh or VAO")
                continue
            if "u_alpha" in self._members[b.path]:
                prog["u_alpha"].value = alpha
            self._bind_base(frame, vao, b)
            vao.render(moderngl.TRIANGLES, vertices=_VERTICES.get(b.path, -1), instances=b.count)
            self.draw_calls += 1
            ctx.draw_calls += 1

    def _bind_base(self, frame: PackedFrame, vao: moderngl.VertexArray, b) -> None:
        buf = self._buffers[b.path]
        if buf is None:
            return
        stride = RECORD_FLOATS[b.path] * 4
        if self._gl.rebind_instance_attributes(
            vao.glo, buf.glo, stride, b.start * stride, self._locs[b.path]
        ):
            return

        buf.write(frame.stream(b.path)[b.start : b.start + b.count])

    def _set_common(self, ctx: PassContext, path: DrawPath) -> None:
        prog = self._progs[path]
        members = self._members[path]

        np.copyto(self._mats[0], ctx.view.T)
        np.copyto(self._mats[1], ctx.proj.T)
        np.copyto(self._mats[2], ctx.view_proj.T)
        for name, idx in (("u_view", 0), ("u_proj", 1), ("u_view_proj", 2)):
            if name in members:
                prog[name].write(self._mats[idx])
        if "u_px_scale" in members:
            prog["u_px_scale"].value = ctx.px_scale
        if "u_viewport" in members:
            prog["u_viewport"].value = (ctx.target.width, ctx.target.height)

    def _release_vaos(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        self._vaos.clear()

    def release(self) -> None:
        self._release_vaos()
        for buf in self._buffers.values():
            if buf is not None:
                buf.release()
        self._buffers = dict.fromkeys(DrawPath, None)
        for mesh in self._meshes.values():
            if mesh is not None:
                mesh.release()
        self._meshes.clear()
        self._text.release()


register_pass("debug", DebugPass)
