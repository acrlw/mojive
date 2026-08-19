"""Backend-neutral glyph atlas and label layout for world-space text.

Split of ``render.forge.text``: the PIL atlas building (JetBrains Mono primary
with a CJK fallback for codepoints >= U+2E80) and the per-frame glyph record
layout are pure CPU work shared by both render backends.  GPU upload and draw
stay per-backend (``render.forge.text.TextRenderer`` and the webgpu debug
pass); both consume the same 17-float records:

    anchor(3) offset_px(2) rect(4) uv_rect(4) color(4)

``rect`` is the PIL glyph bounding box relative to the anchor-baseline origin
(``anchor="ls"``), ``uv_rect`` the normalized atlas window.  Records are
batched by occlusion mode in label order (``TextBatch``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..log import get_logger
from .debugdraw import Occlusion, TextLabel

log = get_logger("text")

RECORD_FLOATS = 17


@dataclass(frozen=True)
class Glyph:
    rect: tuple[float, float, float, float]
    uv: tuple[float, float, float, float]
    advance: float


@dataclass
class TextBatch:
    occlusion: Occlusion = Occlusion.DEPTH
    start: int = 0
    count: int = 0


class TextLayout:
    """PIL glyph atlas plus per-frame label layout; owns no GPU state."""

    def __init__(self) -> None:
        self._primary = ("", 0)
        self._fallback = ("", 0)
        self._size_px = 14
        self._fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont | None] | None = None
        self._chars: set[str] = set()
        self._glyphs: dict[str, Glyph] = {}
        self.pixels = b""
        self.atlas_size = (1, 1)
        self.atlas_dirty = True
        self.records = np.zeros((0, RECORD_FLOATS), np.float32)
        self.count = 0
        self._batches: list[TextBatch] = []
        self._batch_count = 0

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
        self.pixels = b""
        # The atlas content is gone; the backend must re-upload before drawing.
        self.atlas_dirty = True

    def prepare(self, labels: list[TextLabel], count: int) -> bool:
        """Layout glyph records; True when there is anything to draw."""
        if count == 0:
            return False
        wanted = {ch for label in labels[:count] for ch in label.text if ch != "\n"}
        if not wanted.issubset(self._chars):
            self._chars |= wanted
            self._build_atlas()
            self.atlas_dirty = True
        self._layout(labels, count)
        return self.count > 0

    def mark_uploaded(self) -> None:
        self.atlas_dirty = False

    def batches(self) -> list[TextBatch]:
        return self._batches[: self._batch_count]

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
        glyphs: dict[str, Glyph] = {}
        for ch, font, bbox, advance, x, y, w, h in placed:
            if not ch.isspace():
                draw.text((x - bbox[0], y - bbox[1]), ch, font=font, fill=255, anchor="ls")
            glyphs[ch] = Glyph(
                tuple(float(v) for v in bbox),
                (x / width, y / height, (x + w) / width, (y + h) / height),
                advance,
            )
        self._glyphs = glyphs
        self.pixels = image.tobytes()
        self.atlas_size = image.size

    def _layout(self, labels: list[TextLabel], count: int) -> None:
        glyph_count = sum(sum(not ch.isspace() for ch in label.text) for label in labels[:count])
        if glyph_count > len(self.records):
            self.records = np.zeros(
                (max(glyph_count, len(self.records) * 2, 64), RECORD_FLOATS), np.float32
            )
        self.count = 0
        self._batch_count = 0
        primary, _ = self._fonts_for_atlas()
        ascent, descent = primary.getmetrics()
        line_h = float(ascent + descent)
        last_occ = None
        for label in labels[:count]:
            lines = label.text.split("\n")
            widths = [sum(self._glyphs[ch].advance for ch in line) for line in lines]
            block_h = line_h * len(lines)
            start = self.count
            for row, line in enumerate(lines):
                pen_x = float(label.offset_px[0] - widths[row] * label.align[0])
                base_y = float(
                    label.offset_px[1] - block_h * label.align[1] + ascent + row * line_h
                )
                for ch in line:
                    glyph = self._glyphs[ch]
                    if not ch.isspace():
                        record = self.records[self.count]
                        record[0:3] = label.anchor
                        record[3:5] = (pen_x, base_y)
                        record[5:9] = glyph.rect
                        record[9:13] = glyph.uv
                        record[13:17] = label.color
                        self.count += 1
                    pen_x += glyph.advance
            if self.count == start:
                continue
            if label.occlusion is last_occ:
                self._batches[self._batch_count - 1].count += self.count - start
            else:
                if self._batch_count == len(self._batches):
                    self._batches.append(TextBatch())
                batch = self._batches[self._batch_count]
                batch.occlusion, batch.start, batch.count = (
                    label.occlusion,
                    start,
                    self.count - start,
                )
                self._batch_count += 1
                last_occ = label.occlusion
