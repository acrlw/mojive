"""Cache local glyph commands and submit runtime icons and helper strokes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from mojive.geometry2d.curves import (
    CORNER_SMOOTHING,
)

from .glyphs import (
    _draw_glyph,
)
from .layout import (
    _icon_layout,
    _production_icon_layout,
    icon_metrics,
    production_icon_metrics,
)
from .model import (
    ICON_GRID,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_STROKE,
    ICON_TUNING_DEFAULTS,
    REVIEW_LOCKED_ICONS,
    STATUS_MOUSE_DEFAULT_WIDTH,
    STROKE_SCALE_LOCKED_ICONS,
    IconStyle,
    IconTuning,
    production_icon_offset,
    production_icon_style,
)


def draw_concept_icon(
    draw,
    center,
    size: float,
    name: str,
    color,
    *,
    padding: float | None = None,
    accent_color=None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
) -> None:
    """Draw one declared-anchor candidate fitted to a padded circular slot."""

    style = IconStyle(
        padding=padding,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
        alignment=alignment,
    )
    _draw_cached_icon(draw, center, size, name, color, accent_color, style)


class _IconCommands:
    """Record immutable Draw2D geometry with late-bound foreground/accent colors."""

    _COLOR_ARGUMENT: ClassVar[dict[str, int]] = {
        "line": 2,
        "arrow": 2,
        "polyline": 1,
        "convex_fill": 1,
        "fringed_concave_fill": 1,
        "indexed_fill": 2,
        "circle": 2,
        "circle_filled": 2,
        "rect": 2,
        "rect_filled": 2,
    }

    def __init__(self):
        self.commands = []

    def __getattr__(self, method):
        if method not in self._COLOR_ARGUMENT:
            raise AttributeError(method)
        color_index = self._COLOR_ARGUMENT[method]

        def record(*args, **kwargs):
            self.commands.append(
                (
                    method,
                    args[:color_index],
                    args[color_index],
                    args[color_index + 1 :],
                    kwargs,
                )
            )

        return record


@lru_cache(maxsize=512)
def _icon_draw_commands(name: str, size: float, style: IconStyle | None):
    """Compile local geometry once per shape and size, independent of placement or color."""

    if style is None:
        style, layout = _production_icon_layout(name)
    else:
        layout = _icon_layout(
            name,
            style.padding,
            style.mouse_width,
            style.stroke_width,
            style.rotate_ring_gap_ratio,
            style.rotate_ring_cap,
            style.tuning,
            style.alignment,
        )
    layout_scale, offset, stroke_compensation = layout
    unit_scale = size / ICON_GRID
    adjusted_center = (
        offset[0] * unit_scale,
        offset[1] * unit_scale,
    )
    recording = _IconCommands()
    _draw_glyph(
        recording,
        adjusted_center,
        size * layout_scale,
        name,
        0,
        accent_color=1,
        mouse_width=style.mouse_width,
        stroke_width=style.stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=style.rotate_ring_gap_ratio,
        rotate_ring_cap=style.rotate_ring_cap,
        tuning=style.tuning,
    )
    return tuple(recording.commands)


def _place_icon_command(method, before, options, center):
    """Translate recorded primitives for Draw2D consumers without native vertex transforms."""

    def point(value):
        return value[0] + center[0], value[1] + center[1]

    if method in {"indexed_fill", "fringed_concave_fill"}:
        if options.get("origin") is not None:
            return before, {**options, "origin": point(options["origin"])}
        options = {
            key: tuple(map(point, value)) if key in {"outline", "hole"} else value
            for key, value in options.items()
        }
    if method in {"line", "arrow", "rect", "rect_filled"}:
        return (point(before[0]), point(before[1]), *before[2:]), options
    if method in {"circle", "circle_filled"}:
        return (point(before[0]), *before[1:]), options
    return (tuple(map(point, before[0])), *before[1:]), options


def draw_icon(draw, center, size: float, name: str, color, *, accent_color=None) -> None:
    """Submit cached production geometry with the current interaction colors."""

    _draw_cached_icon(draw, center, size, name, color, accent_color, None)


def draw_icon_label(
    draw,
    lo,
    hi,
    color,
    scale: float,
    name: str,
    label: str,
    *,
    style: IconStyle | None = None,
    icon_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Center a measured pair; optional pixel offsets move only the icon, not its label."""
    size = 16.0 * scale
    if not label:
        center = ((lo[0] + hi[0]) * 0.5 + icon_offset[0], (lo[1] + hi[1]) * 0.5 + icon_offset[1])
        _draw_cached_icon(draw, center, size, name, color, None, style)
        return
    if style is None:
        bounds = production_icon_metrics(name).bounds
    else:
        bounds = icon_metrics(
            name,
            style.padding,
            style.mouse_width,
            style.stroke_width,
            style.rotate_ring_gap_ratio,
            style.rotate_ring_cap,
            style.tuning,
            style.alignment,
        ).bounds
    unit = size / ICON_GRID
    glyph_width = (bounds[2] - bounds[0]) * unit
    gap = 7.0 * scale
    label_width, line_height = draw.text_size(label)
    ink = draw.text_ink_bounds(label) or (0.0, 0.0, label_width, line_height)
    left = (lo[0] + hi[0] - glyph_width - gap - (ink[2] - ink[0])) * 0.5
    # Use one body line for related words, independent of ascenders/descenders.
    body = draw.text_ink_bounds("x" if label.isascii() else "田")
    center_y = (lo[1] + hi[1]) * 0.5
    text_y = center_y - ((body[1] + body[3]) * 0.5 if body else line_height * 0.5)
    _draw_cached_icon(
        draw,
        (left - bounds[0] * unit + icon_offset[0], center_y + icon_offset[1]),
        size,
        name,
        color,
        None,
        style,
    )
    draw.text((left + glyph_width + gap - ink[0], text_y), color, label, pixel_snap=False)


def _draw_cached_icon(draw, center, size, name, color, accent_color, style):
    from ..icon_draw import ImguiIconDraw
    from ..imgui_draw import ImguiDraw2D

    if style is None:
        dx, dy = production_icon_offset(name)
        unit = size / ICON_GRID
        center = (center[0] + dx * unit, center[1] + dy * unit)
    if isinstance(draw, ImguiDraw2D):
        if name in REVIEW_LOCKED_ICONS:
            density = draw._imgui.get_io().display_framebuffer_scale
            draw = ImguiIconDraw(draw, 1.0 / max(1.0, density.x, density.y))
        elif name not in STROKE_SCALE_LOCKED_ICONS:
            resolved = production_icon_style(name) if style is None else style
            fringe = min(1.0, size / ICON_GRID * resolved.stroke_width * 0.5)
            draw = ImguiIconDraw(draw, fringe)
    colors = (color, color if accent_color is None else accent_color)
    native = isinstance(draw, ImguiDraw2D)
    first = len(draw._dl.vtx_buffer) if native else 0
    for method, before, slot, after, options in _icon_draw_commands(name, float(size), style):
        if not native:
            before, options = _place_icon_command(method, before, options, center)
        getattr(draw, method)(*before, colors[slot], *after, **options)
    if native:
        # Include AA vertices in the same native translation. Geometry caches
        # then survive scrolling, dock resizing, and subpixel timeline motion.
        draw._imgui.internal.shade_verts_transform_pos(
            draw._dl, first, len(draw._dl.vtx_buffer), (0.0, 0.0), 1.0, 0.0, draw._vec(center)
        )


def draw_control_icon(draw, center, size: float, kind: str, color) -> None:
    """Adapt reusable panel-control names to production icon names."""

    draw_icon(draw, center, size, f"panel-{kind}", color)


@dataclass(frozen=True)
class IconStrokePath:
    """One cached centerline path for a stroked production helper icon."""

    points: tuple[tuple[float, float], ...]
    width: float
    closed: bool


def _simplify_stroke_points(points, tolerance: float, *, closed: bool) -> tuple:
    """Reduce cached helper centerlines below a subpixel chord-error bound."""

    values = tuple((float(point[0]), float(point[1])) for point in points)

    def simplify(path: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        if len(path) <= 2:
            return path
        ax, ay = path[0]
        bx, by = path[-1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        greatest = -1.0
        split = 0
        for index, (x, y) in enumerate(path[1:-1], start=1):
            if length2 <= 1e-20:
                distance2 = (x - ax) ** 2 + (y - ay) ** 2
            else:
                fraction = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length2))
                distance2 = (x - ax - fraction * dx) ** 2 + (y - ay - fraction * dy) ** 2
            if distance2 > greatest:
                greatest, split = distance2, index
        if greatest <= tolerance * tolerance:
            return path[0], path[-1]
        return (*simplify(path[: split + 1])[:-1], *simplify(path[split:]))

    if not closed or len(values) < 4:
        return simplify(values)
    split = len(values) // 2
    first = simplify(values[: split + 1])
    second = simplify((*values[split:], values[0]))
    return (*first[:-1], *second[:-1])


class _StrokePathDraw:
    """Record outline-only icons once for the 3D debug helper renderer."""

    corner_smoothing = CORNER_SMOOTHING

    def __init__(self) -> None:
        self.paths: list[IconStrokePath] = []

    def line(self, a, b, _color, width: float, **_kwargs) -> None:
        self.paths.append(
            IconStrokePath(
                ((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))),
                float(width),
                False,
            )
        )

    def polyline(self, points, _color, width: float, *, closed: bool = False, **_kwargs) -> None:
        self.paths.append(
            IconStrokePath(
                _simplify_stroke_points(points, 0.06, closed=bool(closed)),
                float(width),
                bool(closed),
            )
        )

    def circle(
        self, center, radius: float, _color, width: float = 1.0, *, segments: int = 0
    ) -> None:
        count = max(12, int(segments))
        self.paths.append(
            IconStrokePath(
                tuple(
                    (
                        float(center[0]) + float(radius) * math.cos(index * math.tau / count),
                        float(center[1]) + float(radius) * math.sin(index * math.tau / count),
                    )
                    for index in range(count)
                ),
                float(width),
                True,
            )
        )


@lru_cache(maxsize=2)
def production_helper_strokes(name: str) -> tuple[IconStrokePath, ...]:
    """Return cached 24 pt centerlines for a scene helper icon."""

    if name not in {"helper-camera", "helper-light"}:
        raise ValueError(f"not a scene helper icon: {name!r}")
    draw = _StrokePathDraw()
    draw_icon(draw, (0.0, 0.0), ICON_GRID, name, (1.0, 1.0, 1.0, 1.0))
    return tuple(draw.paths)
