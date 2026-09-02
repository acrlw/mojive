"""Export production Tool Column glyph geometry to transparent PNG files."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..ui.draw2d import _capped_polyline_outline, _open_polyline_ribbon
from ..ui.viewport_widgets import (
    _ROTATE_HALF_RINGS,
    OVERLAY_GEOMETRY,
    TOOL_GLYPH_SCALE,
    OverlayGeometry,
    _rotate_stroke_outline,
    _transform_path,
    draw_tool_glyph,
)

_SUPERSAMPLE = 4
_GLYPH_SCALE_FOR_CANVAS = 0.03125
_FONT_PATHS = (
    Path("/System/Library/Fonts/SFNSMono.ttf"),
    Path("/System/Library/Fonts/Menlo.ttc"),
    Path("/System/Library/Fonts/Monaco.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("C:/Windows/Fonts/consola.ttf"),
)
_FOREGROUND = (220, 223, 227, 255)
_TRANSPARENT = (0, 0, 0, 0)
_BLACK = (0, 0, 0, 255)
_TOOL_SHELL_GAP_RATIO = 0.5


def _mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


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

    def line(self, a, b, color, width: float, *, cap: str = "butt") -> None:
        self.polyline((a, b), color, width, cap=cap)

    def polyline(
        self,
        points,
        color,
        width: float,
        *,
        closed: bool = False,
        cap: str = "butt",
    ) -> None:
        path = self._points(points)
        if closed and path:
            path = (*path, path[0])
        self.capped_polyline(path, color, width, cap=cap)

    def capped_polyline(self, points, color, width: float, *, cap: str) -> None:
        path = self._points(points)
        if cap in {"round", "round_start", "round_end"}:
            outline = _capped_polyline_outline(
                path,
                float(width),
                round_start=cap in {"round", "round_start"},
                round_end=cap in {"round", "round_end"},
            )
            if outline:
                self.draw.polygon(outline, fill=_rgba(color))
            return
        left, right, outline = _open_polyline_ribbon(path, float(width))
        if not outline:
            return
        fill = _rgba(color)
        self.draw.polygon((*left, *reversed(right)), fill=fill)
        if cap != "butt":
            raise ValueError(f"unknown polyline cap: {cap!r}")

    def convex_fill(self, points, color) -> None:
        self.draw.polygon(self._points(points), fill=_rgba(color))

    def concave_fill(self, points, color) -> None:
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
        half_width = float(width) * 0.5
        outer = radius + half_width
        inner = max(0.0, radius - half_width)
        self.draw.ellipse(
            (x - outer, y - outer, x + outer, y + outer),
            fill=_rgba(color),
        )
        self.draw.ellipse(
            (x - inner, y - inner, x + inner, y + inner),
            fill=_TRANSPARENT,
        )

    def circle_filled(self, center, radius: float, color, *, segments: int = 0) -> None:
        del segments
        x, y = center
        self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_rgba(color))

    def rect(self, lo, hi, color, width: float = 1.0, *, rounding: float = 0.0) -> None:
        half_width = float(width) * 0.5
        self.draw.rounded_rectangle(
            (
                float(lo[0]) - half_width,
                float(lo[1]) - half_width,
                float(hi[0]) + half_width,
                float(hi[1]) + half_width,
            ),
            radius=float(rounding) + half_width,
            fill=_rgba(color),
        )
        self.draw.rounded_rectangle(
            (
                float(lo[0]) + half_width,
                float(lo[1]) + half_width,
                float(hi[0]) - half_width,
                float(hi[1]) - half_width,
            ),
            radius=max(0.0, float(rounding) - half_width),
            fill=_TRANSPARENT,
        )

    def rect_filled(self, lo, hi, color, *, rounding: float = 0.0) -> None:
        self.draw.rounded_rectangle(
            (*lo, *hi),
            radius=float(rounding),
            fill=_rgba(color),
        )

    def text(self, pos, color, text: str, *, pixel_snap: bool = True) -> None:
        del pixel_snap  # Pillow consumes the export's authored coordinates directly.
        self.draw.text(
            tuple(pos), text, fill=_rgba(color), font=_mono_font(round(6.4 * self.scale))
        )

    def text_size(self, text: str) -> tuple[float, float]:
        bounds = self.draw.textbbox((0, 0), text, font=_mono_font(round(6.4 * self.scale)))
        return float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1])

    def text_ink_bounds(self, text: str) -> tuple[float, float, float, float] | None:
        if not text.strip():
            return None
        return tuple(
            float(value)
            for value in self.draw.textbbox((0, 0), text, font=_mono_font(round(6.4 * self.scale)))
        )

    def centered_label(self, text: str, center, color, max_width: float) -> None:
        font_size = max(1, round(6.4 * self.scale))
        font = _mono_font(font_size)
        bounds = self.draw.textbbox((0, 0), text, font=font)
        width = bounds[2] - bounds[0]
        if width > max_width:
            font_size = max(1, round(font_size * max_width / width))
            font = _mono_font(font_size)
            bounds = self.draw.textbbox((0, 0), text, font=font)
        x = center[0] - (bounds[2] + bounds[0]) * 0.5
        y = center[1] - (bounds[3] + bounds[1]) * 0.5
        self.draw.text((x, y), text, fill=_rgba(color), font=font)


class _ToolShellMaskDraw2D:
    """Rasterize the union of Tool glyph primitives expanded by one shell gap."""

    def __init__(self, mask: Image.Image, gap: float) -> None:
        self.draw = ImageDraw.Draw(mask)
        self.gap = float(gap)

    @staticmethod
    def _points(points):
        return tuple((float(x), float(y)) for x, y in points)

    @staticmethod
    def _width(width: float) -> int:
        return max(1, round(float(width)))

    def line(self, a, b, _color, width: float, *, cap: str = "butt") -> None:
        expanded = self._width(float(width) + 2.0 * self.gap)
        self.draw.line((tuple(a), tuple(b)), fill=255, width=expanded)
        if cap in {"round", "round_start", "round_end"}:
            radius = expanded * 0.5
            endpoints = (a,) if cap == "round_start" else (b,) if cap == "round_end" else (a, b)
            for x, y in endpoints:
                self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    def _filled_polygon(self, points) -> None:
        path = self._points(points)
        if not path:
            return
        self.draw.polygon(path, fill=255)
        width = self._width(2.0 * self.gap)
        radius = width * 0.5
        for start, end in zip(path, (*path[1:], path[0]), strict=True):
            self.draw.line((start, end), fill=255, width=width)
        # Pillow's closed-path curve join can leave a one-segment slit at the
        # closing vertex.  Explicit vertex disks form the actual rounded
        # Minkowski shell and keep this diagnostic free of rasterizer seams.
        for x, y in path:
            self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    def convex_fill(self, points, _color) -> None:
        self._filled_polygon(points)

    def concave_fill(self, points, _color) -> None:
        self._filled_polygon(points)

    def fringed_concave_fill(self, points, _color) -> None:
        self._filled_polygon(points)

    def circle_filled(self, center, radius: float, _color, *, segments: int = 0) -> None:
        del segments
        x, y = center
        expanded = float(radius) + self.gap
        self.draw.ellipse(
            (x - expanded, y - expanded, x + expanded, y + expanded),
            fill=255,
        )

    def centered_label(self, *_args, **_kwargs) -> None:
        # The W/B label is typography, not part of the arrow-shell diagnostic.
        pass


def render_tool_shell_icon(
    size: int,
    kind: str,
    space: str,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
    *,
    foreground: tuple[int, int, int, int] = _FOREGROUND,
    shell: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Render a filled Tool glyph with a relative black or transparent shell."""

    working_size = int(size) * _SUPERSAMPLE
    scale = working_size * _GLYPH_SCALE_FOR_CANVAS
    center = (working_size * 0.5, working_size * 0.5)
    color = tuple(channel / 255.0 for channel in foreground)
    transparent = tuple(channel / 255.0 for channel in _TRANSPARENT)
    core = Image.new("RGBA", (working_size, working_size), _TRANSPARENT)
    draw_tool_glyph(
        _PillowDraw2D(core, scale),
        center,
        color,
        scale,
        kind,
        transparent,
        space,
        geometry,
    )
    image = Image.new("RGBA", (working_size, working_size), _TRANSPARENT)
    if shell is not None:
        gap_ratio = geometry.frame_center_gap_ratio if kind == "frame" else _TOOL_SHELL_GAP_RATIO
        shell_gap = geometry.tool_stroke * gap_ratio * scale
        shell_mask = Image.new("L", (working_size, working_size), 0)
        draw_tool_glyph(
            _ToolShellMaskDraw2D(
                shell_mask,
                shell_gap,
            ),
            center,
            color,
            scale,
            kind,
            transparent,
            space,
            geometry,
        )
        image.paste(shell, (0, 0, working_size, working_size), shell_mask)
    image.alpha_composite(core)
    if shell is not None and kind == "frame":
        # Keep the center dot legible as an independently outlined origin.
        # The arrow shafts pass underneath it, so their visible starts still
        # meet the outside of this ring without exposing background wedges.
        glyph_scale = scale * TOOL_GLYPH_SCALE
        radius = geometry.frame_center_radius * glyph_scale
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (
                center[0] - radius - shell_gap,
                center[1] - radius - shell_gap,
                center[0] + radius + shell_gap,
                center[1] + radius + shell_gap,
            ),
            fill=shell,
        )
        draw.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            fill=foreground,
        )
    return image.resize((size, size), Image.Resampling.LANCZOS)


def render_rotate_shell_icon(
    size: int,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
    *,
    foreground: tuple[int, int, int, int] = _FOREGROUND,
    shell: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Render the cyclic rotate glyph with either visible or transparent shells."""

    working_size = int(size) * _SUPERSAMPLE
    scale = working_size * _GLYPH_SCALE_FOR_CANVAS
    center = (working_size * 0.5, working_size * 0.5)
    glyph_scale = scale * TOOL_GLYPH_SCALE
    core_local_width = geometry.tool_stroke / TOOL_GLYPH_SCALE
    shell_local_width = (geometry.tool_stroke + 2.0 * geometry.rotate_ring_gap) / TOOL_GLYPH_SCALE

    states = []
    for path in _ROTATE_HALF_RINGS:
        core_mask = Image.new("L", (working_size, working_size), 0)
        shell_mask = Image.new("L", (working_size, working_size), 0)
        core_path = _transform_path(
            _rotate_stroke_outline(path, core_local_width, geometry.rotate_ring_cap),
            center[0],
            center[1],
            glyph_scale,
        )
        shell_path = _transform_path(
            _rotate_stroke_outline(path, shell_local_width, geometry.rotate_ring_cap),
            center[0],
            center[1],
            glyph_scale,
        )
        ImageDraw.Draw(core_mask).polygon(core_path, fill=255)
        ImageDraw.Draw(shell_mask).polygon(shell_path, fill=255)
        core_pixels = np.asarray(core_mask) > 0
        shell_pixels = np.asarray(shell_mask) > 0
        states.append(np.where(core_pixels, 2, shell_pixels).astype(np.uint8))

    ring_states = np.stack(states)
    active = ring_states > 0
    active_count = np.sum(active, axis=0)
    composed = np.zeros((working_size, working_size), np.uint8)
    for index in range(3):
        only_ring = (active_count == 1) & active[index]
        composed[only_ring] = ring_states[index][only_ring]
    # Path order is Y, X, Z. Each tuple is (front, back).
    for front, back in ((0, 1), (1, 2), (2, 0)):
        crossing = (active_count == 2) & active[front] & active[back]
        composed[crossing] = ring_states[front][crossing]
    triple_overlap = active_count == 3
    composed[triple_overlap] = np.max(ring_states[:, triple_overlap], axis=0)

    image = Image.new("RGBA", (working_size, working_size), _TRANSPARENT)
    draw = _PillowDraw2D(image, scale)
    color = tuple(channel / 255.0 for channel in foreground)
    draw.circle(
        center,
        10.0 * glyph_scale,
        color,
        geometry.tool_stroke * scale,
        segments=48,
    )
    image_bounds = (0, 0, working_size, working_size)
    if shell is not None:
        shell_mask = Image.fromarray(
            np.where(composed == 1, 255, 0).astype(np.uint8),
            "L",
        )
        image.paste(shell, image_bounds, shell_mask)
    visible_mask = Image.fromarray(
        np.where(composed == 2, 255, 0).astype(np.uint8),
        "L",
    )
    image.paste(foreground, image_bounds, visible_mask)
    return image.resize((size, size), Image.Resampling.LANCZOS)


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

    for stem, kind, space in (
        ("move", "move", "world"),
        ("frame-world", "frame", "world"),
        ("frame-body", "frame", "body"),
    ):
        black_shell = render_tool_shell_icon(size, kind, space, shell=_BLACK)
        transparent_shell = render_tool_shell_icon(size, kind, space)
        for suffix, icon in (
            ("black-shell", black_shell),
            ("transparent-shell", transparent_shell),
        ):
            path = output / f"{stem}-{suffix}.png"
            icon.save(path)
            written.append(path)
            preview_path = output / f"{stem}-{suffix}-preview.png"
            checker_preview(icon).save(preview_path)
            written.append(preview_path)

    # Design probes: show the same local cyclic shell composition with the
    # shell pixels either black or transparent.
    for cap in ("butt", "round"):
        geometry = replace(OVERLAY_GEOMETRY, rotate_ring_cap=cap)
        shell = render_rotate_shell_icon(size, geometry, shell=_BLACK)
        keyed_image = render_rotate_shell_icon(size, geometry)
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
