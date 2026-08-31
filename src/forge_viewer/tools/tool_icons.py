"""Export production Tool Column glyph geometry to transparent PNG files."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..ui.draw2d import _open_polyline_ribbon
from ..ui.viewport_widgets import (
    FILLED_GLYPH_STROKE_SCALE,
    OVERLAY_GEOMETRY,
    TOOL_GLYPH_SCALE,
    _rotate_visible_ring_polygons,
    _transform_path,
    draw_tool_glyph,
)

_SUPERSAMPLE = 4
_GLYPH_SCALE_FOR_CANVAS = 0.03125
_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
_FOREGROUND = (220, 223, 227, 255)
_TRANSPARENT = (0, 0, 0, 0)
_BLACK = (0, 0, 0, 255)


def _rgba(color) -> tuple[int, int, int, int]:
    return tuple(max(0, min(255, round(float(channel) * 255.0))) for channel in color)


class _PillowDraw2D:
    """Small Draw2D adapter used only for transparent design exports."""

    def __init__(self, image: Image.Image, scale: float) -> None:
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.scale = scale

    @staticmethod
    def _points(points):
        return tuple((float(x), float(y)) for x, y in points)

    @staticmethod
    def _width(width: float) -> int:
        return max(1, round(float(width)))

    def line(self, a, b, color, width: float) -> None:
        self.draw.line((tuple(a), tuple(b)), fill=_rgba(color), width=self._width(width))

    def polyline(self, points, color, width: float, *, closed: bool = False) -> None:
        path = self._points(points)
        if closed and path:
            path = (*path, path[0])
        self.capped_polyline(path, color, width, cap="butt")

    def capped_polyline(self, points, color, width: float, *, cap: str) -> None:
        path = self._points(points)
        left, right, outline = _open_polyline_ribbon(path, float(width))
        if not outline:
            return
        fill = _rgba(color)
        self.draw.polygon((*left, *reversed(right)), fill=fill)
        if cap == "round":
            radius = float(width) * 0.5
            for x, y in (path[0], path[-1]):
                self.draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=fill,
                )
        elif cap != "butt":
            raise ValueError(f"unknown polyline cap: {cap!r}")

    def convex_fill(self, points, color) -> None:
        self.draw.polygon(self._points(points), fill=_rgba(color))

    def fringed_concave_fill(self, points, color) -> None:
        self.draw.polygon(self._points(points), fill=_rgba(color))

    def circle(
        self,
        center,
        radius: float,
        color,
        width: float = 1.0,
        *,
        segments: int = 0,
    ) -> None:
        del segments
        x, y = center
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self.draw.ellipse(bounds, outline=_rgba(color), width=self._width(width))

    def circle_filled(self, center, radius: float, color, *, segments: int = 0) -> None:
        del segments
        x, y = center
        self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_rgba(color))

    def centered_label(self, text: str, center, color, max_width: float) -> None:
        font_size = max(1, round(6.4 * self.scale))
        font = ImageFont.truetype(str(_FONT_PATH), font_size)
        bounds = self.draw.textbbox((0, 0), text, font=font)
        width = bounds[2] - bounds[0]
        if width > max_width:
            font_size = max(1, round(font_size * max_width / width))
            font = ImageFont.truetype(str(_FONT_PATH), font_size)
            bounds = self.draw.textbbox((0, 0), text, font=font)
        x = center[0] - (bounds[2] + bounds[0]) * 0.5
        y = center[1] - (bounds[3] + bounds[1]) * 0.5
        self.draw.text((x, y), text, fill=_rgba(color), font=font)


def export_icons(output: Path, size: int = 1024) -> tuple[Path, ...]:
    """Write each production glyph to its own transparent square canvas."""

    output.mkdir(parents=True, exist_ok=True)
    working_size = int(size) * _SUPERSAMPLE
    scale = working_size * _GLYPH_SCALE_FOR_CANVAS
    center = (working_size * 0.5, working_size * 0.5)
    foreground = tuple(channel / 255.0 for channel in _FOREGROUND)
    transparent = tuple(channel / 255.0 for channel in _TRANSPARENT)
    variants = (
        ("move", "move", "world", OVERLAY_GEOMETRY),
        ("rotate", "rotate", "world", OVERLAY_GEOMETRY),
        (
            "rotate-butt",
            "rotate",
            "world",
            replace(OVERLAY_GEOMETRY, rotate_ring_cap="butt"),
        ),
        (
            "rotate-round",
            "rotate",
            "world",
            replace(OVERLAY_GEOMETRY, rotate_ring_cap="round"),
        ),
        ("frame-world", "frame", "world", OVERLAY_GEOMETRY),
        ("frame-body", "frame", "body", OVERLAY_GEOMETRY),
        ("snap", "snap", "world", OVERLAY_GEOMETRY),
    )
    written = []
    for filename, kind, space, geometry in variants:
        image = Image.new("RGBA", (working_size, working_size), _TRANSPARENT)
        draw_tool_glyph(
            _PillowDraw2D(image, scale),
            center,
            foreground,
            scale,
            kind,
            transparent,
            space,
            geometry,
        )
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        path = output / f"{filename}.png"
        image.save(path)
        written.append(path)

    cell = max(8, size // 16)
    checker_colors = ((92, 96, 104, 255), (132, 137, 147, 255))

    def checker_preview(icon: Image.Image) -> Image.Image:
        checker = Image.new("RGBA", (size, size), (0, 0, 0, 255))
        checker_draw = ImageDraw.Draw(checker)
        for y in range(0, size, cell):
            for x in range(0, size, cell):
                checker_draw.rectangle(
                    (x, y, min(size, x + cell), min(size, y + cell)),
                    fill=checker_colors[(x // cell + y // cell) & 1],
                )
        checker.alpha_composite(icon)
        return checker

    # Design probes: show the invisible shell in black around the same cyclic
    # production polygons, then key black to alpha before downsampling.
    glyph_scale = scale * TOOL_GLYPH_SCALE
    core_width = OVERLAY_GEOMETRY.tool_stroke * scale
    shell_outline = max(1, round(2.0 * OVERLAY_GEOMETRY.rotate_ring_gap * scale))
    black = tuple(channel / 255.0 for channel in _BLACK)
    for cap in ("butt", "round"):
        shell = Image.new("RGBA", (working_size, working_size), _TRANSPARENT)
        draw = _PillowDraw2D(shell, scale)
        draw.circle(center, 10.0 * glyph_scale, foreground, core_width, segments=48)
        paths = tuple(
            _transform_path(local, center[0], center[1], glyph_scale)
            for ring in _rotate_visible_ring_polygons(
                OVERLAY_GEOMETRY.tool_stroke * FILLED_GLYPH_STROKE_SCALE,
                OVERLAY_GEOMETRY.rotate_ring_gap,
                cap,
            )
            for local in ring
        )
        for path in paths:
            draw.draw.line((*path, path[0]), fill=_rgba(black), width=shell_outline, joint="curve")
        for path in paths:
            draw.draw.polygon(path, fill=_rgba(foreground))

        keyed = np.asarray(shell).copy()
        black_pixels = (keyed[:, :, 3] > 0) & np.all(keyed[:, :, :3] == 0, axis=2)
        keyed[black_pixels] = 0
        keyed_image = Image.fromarray(keyed, "RGBA")
        shell = shell.resize((size, size), Image.Resampling.LANCZOS)
        keyed_image = keyed_image.resize((size, size), Image.Resampling.LANCZOS)
        for stem, icon in (
            (f"rotate-black-shell-{cap}", shell),
            (f"rotate-transparent-shell-{cap}", keyed_image),
        ):
            path = output / f"{stem}.png"
            icon.save(path)
            written.append(path)
            preview_path = output / f"{stem}-preview.png"
            checker_preview(icon).save(preview_path)
            written.append(preview_path)
    return tuple(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/tool-icons-1024"))
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    for path in export_icons(args.output, args.size):
        print(path)


if __name__ == "__main__":
    main()
