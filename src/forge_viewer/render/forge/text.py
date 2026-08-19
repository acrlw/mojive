"""GL upload and draw for world-space labels; layout lives in render.text."""

from __future__ import annotations

import moderngl
import numpy as np

from ...log import get_logger
from ..debugdraw import TextLabel
from ..text import RECORD_FLOATS, TextBatch, TextLayout
from . import gl_native as G
from .programs import ProgramCache, ProgramSpec

log = get_logger("text")

_SPEC = ProgramSpec("debug_text", "debug_text.vert", "debug_text.frag")
_LAYOUT = "3f 2f 4f 4f 4f/i"
_ATTRS = (
    ("in_anchor", 3, 0),
    ("in_offset", 2, 12),
    ("in_rect", 4, 20),
    ("in_uv_rect", 4, 36),
    ("in_color", 4, 52),
)


class TextRenderer:
    """GL counterpart of the shared TextLayout: atlas texture + glyph draw."""

    def __init__(self) -> None:
        self._layout = TextLayout()
        self._buffer: moderngl.Buffer | None = None
        self._texture: moderngl.Texture | None = None
        self._vao: moderngl.VertexArray | None = None
        self._program: moderngl.Program | None = None
        self._generation = -1
        self._gl = G.native()
        self._locs: tuple[tuple[int, int, int, int], ...] = ()
        self._matrix = np.zeros((4, 4), np.float32)

    def configure(
        self,
        primary: str = "",
        primary_index: int = 0,
        fallback: str = "",
        fallback_index: int = 0,
        size_px: float = 14.0,
    ) -> None:
        self._layout.configure(primary, primary_index, fallback, fallback_index, size_px)
        self._release_texture()

    def prepare(self, ctx, labels: list[TextLabel], count: int) -> bool:
        if not self._sync_program(ctx.programs):
            return False
        if not self._layout.prepare(labels, count):
            return False
        if self._layout.atlas_dirty or self._texture is None:
            self._upload_atlas(ctx.ctx)
            self._layout.mark_uploaded()
        self._ensure_buffer(ctx.ctx, self._layout.count).write(
            self._layout.records[: self._layout.count]
        )
        return True

    def batches(self) -> list[TextBatch]:
        return self._layout.batches()

    def draw(self, ctx, batch: TextBatch, alpha: float) -> None:
        if self._vao is None or self._program is None or self._texture is None:
            return
        stride = RECORD_FLOATS * 4
        if not self._gl.rebind_instance_attributes(
            self._vao.glo, self._buffer.glo, stride, batch.start * stride, self._locs
        ):
            self._buffer.write(self._layout.records[batch.start : batch.start + batch.count])
        np.copyto(self._matrix, ctx.view_proj.T)
        self._program["u_view_proj"].write(self._matrix)
        self._program["u_viewport"].value = (float(ctx.target.width), float(ctx.target.height))
        self._program["u_alpha"].value = float(alpha)
        self._program["u_atlas"].value = 0
        self._texture.use(0)
        self._vao.render(moderngl.TRIANGLES, vertices=6, instances=batch.count)

    def _sync_program(self, programs: ProgramCache) -> bool:
        if self._program is not None and self._generation == programs.generation:
            return True
        try:
            self._program = programs.get(_SPEC)
        except Exception as exc:
            log.error("World text shader compilation failed: {}", exc)
            return False
        members = frozenset(self._program)
        self._locs = tuple(
            (int(self._program[name].location), comps, offset, G.GL_FLOAT)
            for name, comps, offset in _ATTRS
            if name in members
        )
        self._generation = programs.generation
        self._release_vao()
        return True

    def _ensure_buffer(self, ctx: moderngl.Context, records: int) -> moderngl.Buffer:
        stride = RECORD_FLOATS * 4
        if self._buffer is None or self._buffer.size < records * stride:
            size = max(records, self._buffer.size // stride * 2 if self._buffer else 0, 64) * stride
            if self._buffer is not None:
                self._buffer.release()
            self._release_vao()
            self._buffer = ctx.buffer(reserve=size)
        if self._vao is None:
            names = tuple(name for name, _c, _o in _ATTRS)
            self._vao = ctx.vertex_array(self._program, [(self._buffer, _LAYOUT, *names)])
        return self._buffer

    def _upload_atlas(self, ctx: moderngl.Context) -> None:
        self._release_texture()
        self._texture = ctx.texture(self._layout.atlas_size, 1, self._layout.pixels)
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def _release_vao(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None

    def _release_texture(self) -> None:
        if self._texture is not None:
            self._texture.release()
            self._texture = None

    def release(self) -> None:
        self._release_vao()
        self._release_texture()
        if self._buffer is not None:
            self._buffer.release()
            self._buffer = None
