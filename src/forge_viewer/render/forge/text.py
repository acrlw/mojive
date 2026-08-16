"""JetBrains Mono/CJK glyph atlas and GPU layout for world-space labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...log import get_logger
from ..debugdraw import Occlusion, TextLabel
from . import gl_native as G
from .programs import ProgramCache, ProgramSpec

log = get_logger("text")

_SPEC = ProgramSpec("debug_text", "debug_text.vert", "debug_text.frag")
_FLOATS = 17
_LAYOUT = "3f 2f 4f 4f 4f/i"
_ATTRS = (
    ("in_anchor", 3, 0),
    ("in_offset", 2, 12),
    ("in_rect", 4, 20),
    ("in_uv_rect", 4, 36),
    ("in_color", 4, 52),
)


@dataclass(frozen=True)
class _Glyph:
    rect: tuple[float, float, float, float]
    uv: tuple[float, float, float, float]
    advance: float


@dataclass
class TextBatch:
    occlusion: Occlusion = Occlusion.DEPTH
    start: int = 0
    count: int = 0


class TextRenderer:
    def __init__(self) -> None:
        self._primary = ("", 0)
        self._fallback = ("", 0)
        self._size_px = 14
        self._fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont | None] | None = None
        self._chars: set[str] = set()
        self._glyphs: dict[str, _Glyph] = {}
        self._pixels = b""
        self._atlas_size = (1, 1)
        self._records = np.zeros((0, _FLOATS), np.float32)
        self._count = 0
        self._batches: list[TextBatch] = []
        self._batch_count = 0
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
        spec = (
            (primary, int(primary_index)),
            (fallback, int(fallback_index)),
            max(6, round(size_px)),
        )
        if spec == (self._primary, self._fallback, self._size_px):
            return
        self._primary, self._fallback, self._size_px = spec
        self._fonts = None
        self._chars.clear()
        self._glyphs.clear()
        self._pixels = b""
        self._release_texture()

    def prepare(self, ctx, labels: list[TextLabel], count: int) -> bool:
        if count == 0 or not self._sync_program(ctx.programs):
            return False
        wanted = {ch for label in labels[:count] for ch in label.text if ch != "\n"}
        if not wanted.issubset(self._chars):
            self._chars |= wanted
            self._build_atlas()
            self._upload_atlas(ctx.ctx)
        elif self._texture is None:
            self._upload_atlas(ctx.ctx)
        self._layout(labels, count)
        if self._count == 0:
            return False
        self._ensure_buffer(ctx.ctx, self._count).write(self._records[: self._count])
        return True

    def batches(self) -> list[TextBatch]:
        return self._batches[: self._batch_count]

    def draw(self, ctx, batch: TextBatch, alpha: float) -> None:
        if self._vao is None or self._program is None or self._texture is None:
            return
        stride = _FLOATS * 4
        if not self._gl.rebind_instance_attributes(
            self._vao.glo, self._buffer.glo, stride, batch.start * stride, self._locs
        ):
            self._buffer.write(self._records[batch.start : batch.start + batch.count])
        np.copyto(self._matrix, ctx.view_proj.T)
        self._program["u_view_proj"].write(self._matrix)
        self._program["u_viewport"].value = (float(ctx.target.width), float(ctx.target.height))
        self._program["u_alpha"].value = float(alpha)
        self._program["u_atlas"].value = 0
        self._texture.use(0)
        self._vao.render(moderngl.TRIANGLES, vertices=6, instances=batch.count)

    def _fonts_for_atlas(self) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont | None]:
        if self._fonts is not None:
            return self._fonts

        def load(spec: tuple[str, int]):
            path, index = spec
            return (
                ImageFont.truetype(str(Path(path)), self._size_px, index=index)
                if path and Path(path).is_file()
                else None
            )

        primary = load(self._primary) or ImageFont.load_default(size=self._size_px)
        self._fonts = primary, load(self._fallback)
        return self._fonts

    def _font(self, ch: str) -> ImageFont.ImageFont:
        primary, fallback = self._fonts_for_atlas()
        return fallback if fallback is not None and ord(ch) >= 0x2E80 else primary

    def _build_atlas(self) -> None:
        entries = []
        for ch in sorted(self._chars):
            font = self._font(ch)
            bbox = font.getbbox(ch, anchor="ls")
            entries.append((ch, font, bbox, float(font.getlength(ch))))
        width = 512
        x = y = 2
        row_h = 0
        placed = []
        for ch, font, bbox, advance in entries:
            w, h = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
            if x + w + 2 > width:
                x, y, row_h = 2, y + row_h + 2, 0
            placed.append((ch, font, bbox, advance, x, y, w, h))
            x += w + 2
            row_h = max(row_h, h)
        height = 1
        used_h = y + row_h + 2
        while height < used_h:
            height *= 2
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        glyphs: dict[str, _Glyph] = {}
        for ch, font, bbox, advance, x, y, w, h in placed:
            if not ch.isspace():
                draw.text((x - bbox[0], y - bbox[1]), ch, font=font, fill=255, anchor="ls")
            glyphs[ch] = _Glyph(
                tuple(float(v) for v in bbox),
                (x / width, y / height, (x + w) / width, (y + h) / height),
                advance,
            )
        self._glyphs = glyphs
        self._pixels = image.tobytes()
        self._atlas_size = image.size

    def _layout(self, labels: list[TextLabel], count: int) -> None:
        glyph_count = sum(sum(not ch.isspace() for ch in label.text) for label in labels[:count])
        if glyph_count > len(self._records):
            self._records = np.zeros(
                (max(glyph_count, len(self._records) * 2, 64), _FLOATS), np.float32
            )
        self._count = 0
        self._batch_count = 0
        primary, _ = self._fonts_for_atlas()
        ascent, descent = primary.getmetrics()
        line_h = float(ascent + descent)
        last_occ = None
        for label in labels[:count]:
            lines = label.text.split("\n")
            widths = [sum(self._glyphs[ch].advance for ch in line) for line in lines]
            block_h = line_h * len(lines)
            start = self._count
            for row, line in enumerate(lines):
                pen_x = float(label.offset_px[0] - widths[row] * label.align[0])
                base_y = float(
                    label.offset_px[1] - block_h * label.align[1] + ascent + row * line_h
                )
                for ch in line:
                    glyph = self._glyphs[ch]
                    if not ch.isspace():
                        record = self._records[self._count]
                        record[0:3] = label.anchor
                        record[3:5] = (pen_x, base_y)
                        record[5:9] = glyph.rect
                        record[9:13] = glyph.uv
                        record[13:17] = label.color
                        self._count += 1
                    pen_x += glyph.advance
            if self._count == start:
                continue
            if label.occlusion is last_occ:
                self._batches[self._batch_count - 1].count += self._count - start
            else:
                if self._batch_count == len(self._batches):
                    self._batches.append(TextBatch())
                batch = self._batches[self._batch_count]
                batch.occlusion, batch.start, batch.count = (
                    label.occlusion,
                    start,
                    self._count - start,
                )
                self._batch_count += 1
                last_occ = label.occlusion

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
        stride = _FLOATS * 4
        if self._buffer is None or self._buffer.size < records * stride:
            size = max(records, self._buffer.size // stride * 2 if self._buffer else 0, 64) * stride
            if self._buffer is not None:
                self._buffer.release()
            self._buffer = ctx.buffer(reserve=size)
            self._release_vao()
        if self._vao is None:
            names = tuple(name for name, _c, _o in _ATTRS)
            self._vao = ctx.vertex_array(self._program, [(self._buffer, _LAYOUT, *names)])
        return self._buffer

    def _upload_atlas(self, ctx: moderngl.Context) -> None:
        self._release_texture()
        self._texture = ctx.texture(self._atlas_size, 1, self._pixels)
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
