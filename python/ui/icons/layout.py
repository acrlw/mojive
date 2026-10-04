"""Measure authored glyphs and fit their declared anchors to icon slots."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from functools import cache, lru_cache
from pathlib import Path

from mojive.geometry2d.curves import (
    CORNER_SMOOTHING,
    arrow_mesh,
    smooth_rect_points,
)

from ..overlay_geometry import (
    TOOL_GLYPH_SCALE,
)
from .glyphs import (
    _draw_glyph,
)
from .model import (
    BOX_CENTERED_ICONS,
    ICON_ALIGNMENT_CHOICES,
    ICON_BOUND_DIAMETER,
    ICON_GLYPH_ALIGNMENT_DEFAULTS,
    ICON_GLYPH_PADDING_DEFAULTS,
    ICON_GRID,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_LAYOUT_REFERENCES,
    ICON_MAX_PADDING,
    ICON_MAX_STROKE,
    ICON_MIN_CLEARANCE,
    ICON_MIN_STROKE,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_STROKE,
    ICON_TUNING_DEFAULTS,
    RESET_RING_CENTER,
    REVIEW_LOCKED_ICONS,
    REVIEW_LOCKED_PADDING,
    RING_CENTERED_ICONS,
    STATUS_MOUSE_DEFAULT_WIDTH,
    STROKE_SCALE_LOCKED_ICONS,
    IconStyle,
    IconTuning,
    icon_component_group,
    production_icon_style,
)


@dataclass(frozen=True)
class IconMetrics:
    bounds: tuple[float, float, float, float]
    enclosing_center: tuple[float, float]
    enclosing_radius: float
    origin_extent: float
    anchor_extent: float

    @property
    def center_offset(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5

    @property
    def circular_clearance(self) -> float:
        return ICON_BOUND_DIAMETER * 0.5 - self.origin_extent


def minimum_enclosing_circle(points) -> tuple[tuple[float, float], float]:
    """Return the exact smallest circle for a finite set of sampled boundary points."""

    values = sorted({(float(x), float(y)) for x, y in points})
    if not values:
        return (0.0, 0.0), 0.0
    if len(values) > 2:
        # Interior samples can never define the minimum enclosing circle.  A
        # stroked polyline contributes many overlapping round-cap samples, so
        # reducing them to their convex hull keeps the exact result while
        # avoiding millions of repeated containment checks during a slider drag.
        def cross(origin, a, b) -> float:
            return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

        lower = []
        for point in values:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(values):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        values = lower[:-1] + upper[:-1]
    random.Random(0).shuffle(values)

    def contains(circle, point) -> bool:
        center, radius = circle
        dx, dy = center[0] - point[0], center[1] - point[1]
        return dx * dx + dy * dy <= (radius + 1e-7) ** 2

    def diameter(a, b):
        center = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        return center, math.dist(a, b) * 0.5

    def through_three(a, b, c):
        cross = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(cross) <= 1e-12:
            candidates = (diameter(a, b), diameter(a, c), diameter(b, c))
            return min(
                (
                    candidate
                    for candidate in candidates
                    if all(contains(candidate, point) for point in (a, b, c))
                ),
                key=lambda candidate: candidate[1],
            )
        a2 = a[0] * a[0] + a[1] * a[1]
        b2 = b[0] * b[0] + b[1] * b[1]
        c2 = c[0] * c[0] + c[1] * c[1]
        center = (
            (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / cross,
            (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / cross,
        )
        return center, math.dist(center, a)

    circle = (values[0], 0.0)
    for index, point in enumerate(values):
        if contains(circle, point):
            continue
        circle = (point, 0.0)
        for second_index, second in enumerate(values[:index]):
            if contains(circle, second):
                continue
            circle = diameter(point, second)
            for third in values[:second_index]:
                if not contains(circle, third):
                    circle = through_three(point, second, third)
    return circle


class _MetricsDraw:
    """Measure visible bounds and the minimum enclosing circle."""

    _ROUND_SUPPORT_SAMPLES = 64

    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.boundary_points: list[tuple[float, float]] = []

    def _add(self, points, pad: float = 0.0) -> None:
        for raw_x, raw_y in points:
            x, y = float(raw_x), float(raw_y)
            self.bounds[0] = min(self.bounds[0], x - pad)
            self.bounds[1] = min(self.bounds[1], y - pad)
            self.bounds[2] = max(self.bounds[2], x + pad)
            self.bounds[3] = max(self.bounds[3], y + pad)
            if pad > 0.0:
                self.boundary_points.extend(
                    (
                        x + pad * math.cos(index * math.tau / self._ROUND_SUPPORT_SAMPLES),
                        y + pad * math.sin(index * math.tau / self._ROUND_SUPPORT_SAMPLES),
                    )
                    for index in range(self._ROUND_SUPPORT_SAMPLES)
                )
            else:
                self.boundary_points.append((x, y))

    def line(self, a, b, _color, width, **_kwargs) -> None:
        width = float(width)
        self._add((a, b), width * 0.5)

    def arrow(
        self,
        a,
        b,
        _color,
        width,
        *,
        head_length,
        head_width,
        corner_radius,
        join_radius,
        **_kwargs,
    ) -> None:
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        _vertices, _indices, outline = arrow_mesh(
            length,
            float(width),
            head_length=float(head_length),
            head_width=float(head_width),
            corner_radius=float(corner_radius),
            join_radius=float(join_radius),
            round_tail=bool(_kwargs.get("round_tail", False)),
        )
        points = tuple((a[0] + x * ux - y * uy, a[1] + x * uy + y * ux) for x, y in outline)
        self._add(points)

    def polyline(self, points, _color, width, **kwargs) -> None:
        points = tuple(points)
        width = float(width)
        self._add(points, width * 0.5)

    def fringed_concave_fill(self, points, _color, **_kwargs) -> None:
        points = tuple(points)
        self._add(points)

    def convex_fill(self, points, _color, **_kwargs) -> None:
        self._add(tuple(points))

    def indexed_fill(
        self,
        points,
        _indices,
        _color,
        *,
        outline=(),
        hole=(),
        **_kwargs,
    ) -> None:
        points = tuple(points)
        self._add(tuple(outline) or points)

    def concave_fill(self, points, color) -> None:
        self.fringed_concave_fill(points, color)

    def circle(self, center, radius, _color, width, **_kwargs) -> None:
        radius, width = float(radius), float(width)
        self._add((center,), radius + width * 0.5)

    def circle_filled(self, center, radius, _color, **_kwargs) -> None:
        radius = float(radius)
        self._add((center,), radius)

    def rect(self, lo, hi, _color, width, **kwargs) -> None:
        width = float(width)
        rounding = float(kwargs.get("rounding", 0.0))
        if rounding > 0.0:
            points = smooth_rect_points(
                float(lo[0]),
                float(lo[1]),
                float(hi[0]),
                float(hi[1]),
                rounding,
                smoothing=(
                    CORNER_SMOOTHING
                    if kwargs.get("smoothing") is None
                    else float(kwargs["smoothing"])
                ),
            )
        else:
            points = ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))
        self._add(points, width * 0.5)

    def rect_filled(self, lo, hi, _color, **_kwargs) -> None:
        self._add(((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])))


@lru_cache(maxsize=2048)
def _measure_raw_icon(
    name: str,
    center=(0.0, 0.0),
    size: float = ICON_GRID,
    *,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    stroke_compensation: float = 1.0,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
) -> IconMetrics:
    draw = _MetricsDraw()
    _draw_glyph(
        draw,
        center,
        size,
        name,
        (1.0, 1.0, 1.0, 1.0),
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
    )
    bounds = tuple(draw.bounds)
    enclosing_center, enclosing_radius = minimum_enclosing_circle(draw.boundary_points)
    anchor_center = _alignment_center_values(name, bounds, enclosing_center, alignment)
    origin_extent = max(math.hypot(x, y) for x, y in draw.boundary_points)
    anchor_extent = max(
        math.hypot(x - anchor_center[0], y - anchor_center[1]) for x, y in draw.boundary_points
    )
    return IconMetrics(bounds, enclosing_center, enclosing_radius, origin_extent, anchor_extent)


def _alignment_center_values(
    name: str,
    bounds: tuple[float, float, float, float],
    enclosing_center: tuple[float, float],
    alignment: str | None = None,
) -> tuple[float, float]:
    anchor = icon_alignment_anchor(name, alignment)
    if anchor == "box":
        x0, y0, x1, y1 = bounds
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5
    if anchor == "ring":
        return RESET_RING_CENTER
    return enclosing_center


def _alignment_center(
    name: str, metrics: IconMetrics, alignment: str | None = None
) -> tuple[float, float]:
    return _alignment_center_values(name, metrics.bounds, metrics.enclosing_center, alignment)


@lru_cache(maxsize=256)
def _icon_layout(
    name: str,
    padding: float | None = None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
) -> tuple[float, tuple[float, float], float]:
    """Center the declared anchor and fit every visible point to the requested padding."""

    default_padding = ICON_GLYPH_PADDING_DEFAULTS.get(
        name, ICON_GROUP_LAYOUT_DEFAULTS[icon_component_group(name)]
    )
    if padding is None:
        padding = default_padding
    if name in REVIEW_LOCKED_ICONS:
        padding = REVIEW_LOCKED_PADDING
    padding = float(padding)
    minimum_padding = 0.0 if name == "tool-rotate" else ICON_MIN_CLEARANCE
    if not minimum_padding <= padding <= ICON_MAX_PADDING:
        raise ValueError(
            f"icon padding must be between {minimum_padding:g} and {ICON_MAX_PADDING:g}"
        )
    stroke_width = float(stroke_width)
    if not ICON_MIN_STROKE <= stroke_width <= ICON_MAX_STROKE:
        raise ValueError(f"icon stroke must be between {ICON_MIN_STROKE:g} and {ICON_MAX_STROKE:g}")
    if name in REVIEW_LOCKED_ICONS:
        # The reviewed severity master already defines its frame and internal
        # alignment. Refitting it would diverge from Output and asset export.
        return 1.0, (0.0, 0.0), 1.0
    # Rotate's frame padding measures its outer screen-ring centerline, so the
    # visible edge extends beyond that radius by half of the canonical stroke.
    reference_name = ICON_LAYOUT_REFERENCES.get(name, name)
    target_padding = padding
    safe_radius = ICON_BOUND_DIAMETER * 0.5 - target_padding
    if name == "tool-rotate":
        safe_radius += stroke_width * 0.5

    normalize_stroke = reference_name not in STROKE_SCALE_LOCKED_ICONS

    def fitted_scale(layout_scale: float) -> float:
        compensation = 1.0 / layout_scale if normalize_stroke else 1.0
        reference_raw = _measure_raw_icon(
            reference_name,
            mouse_width=mouse_width,
            stroke_width=stroke_width,
            stroke_compensation=compensation,
            rotate_ring_gap_ratio=rotate_ring_gap_ratio,
            rotate_ring_cap=rotate_ring_cap,
            tuning=tuning,
            alignment=alignment,
        )
        return safe_radius / reference_raw.anchor_extent

    # Fitting each complete icon used to scale its stroke together with its
    # reach, which made nominally identical shared strokes land anywhere from
    # thin to very heavy. Counter-scale the construction stroke and solve only
    # for the envelope. A short fixed-point solve also handles the implicit
    # search mesh and filled G3 ribbons whose visible bounds depend on width.
    if name == "tool-rotate":
        # The outer ring centerline is the slot frame, so its scale is known
        # directly and does not need the expensive iterative ring construction.
        layout_scale = (ICON_BOUND_DIAMETER * 0.5 - target_padding) / (
            10.0 * 0.82 * TOOL_GLYPH_SCALE
        )
    else:
        layout_scale = 1.0
        for _ in range(16):
            updated_scale = fitted_scale(layout_scale)
            if math.isclose(updated_scale, layout_scale, rel_tol=0.0, abs_tol=1e-6):
                layout_scale = updated_scale
                break
            layout_scale = updated_scale
    stroke_compensation = 1.0 / layout_scale if normalize_stroke else 1.0
    raw = _measure_raw_icon(
        name,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
        alignment=alignment,
    )
    anchor_center = _alignment_center(name, raw, alignment)
    offset = (-anchor_center[0], -anchor_center[1])
    return (
        layout_scale,
        (offset[0] * layout_scale, offset[1] * layout_scale),
        stroke_compensation,
    )


def icon_alignment_anchor(name: str, alignment: str | None = None) -> str:
    """Return the explicit geometric placement anchor for one concept icon."""

    icon_component_group(name)
    if alignment is None:
        alignment = ICON_GLYPH_ALIGNMENT_DEFAULTS.get(name)
    if alignment is not None:
        if alignment not in ICON_ALIGNMENT_CHOICES:
            raise ValueError(f"icon alignment must be one of {ICON_ALIGNMENT_CHOICES!r}")
        return alignment
    if name in BOX_CENTERED_ICONS:
        return "box"
    if name in RING_CENTERED_ICONS:
        return "ring"
    return "circle"


@lru_cache(maxsize=256)
def icon_alignment_center(
    name: str,
    padding: float | None = None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
) -> tuple[float, float]:
    """Return the declared anchor center after candidate placement."""

    layout_scale, offset, stroke_compensation = _icon_layout(
        name,
        padding,
        mouse_width,
        stroke_width,
        rotate_ring_gap_ratio,
        rotate_ring_cap,
        tuning,
        alignment,
    )
    raw = _measure_raw_icon(
        name,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
        alignment=alignment,
    )
    source_center = _alignment_center(name, raw, alignment)
    return (
        source_center[0] * layout_scale + offset[0],
        source_center[1] * layout_scale + offset[1],
    )


@cache
def _production_icon_presets():
    return json.loads((Path(__file__).parent.parent / "icon_presets.json").read_text())


@cache
def _production_icon_preset(name: str):
    preset = _production_icon_presets().get(name)
    # Developer changes and custom styles remain correct before regeneration;
    # the preset conformance test requires the shipped defaults to be current.
    if preset is not None and preset["style"] == asdict(production_icon_style(name)):
        return preset
    return None


@cache
def _production_icon_layout(
    name: str,
) -> tuple[IconStyle, tuple[float, tuple[float, float], float]]:
    """Cache all measurement and centering work for the frozen runtime style."""

    style = production_icon_style(name)
    preset = _production_icon_preset(name)
    if preset is not None:
        scale, offset, compensation = preset["layout"]
        return style, (scale, tuple(offset), compensation)
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
    return style, layout


@lru_cache(maxsize=256)
def icon_metrics(
    name: str,
    padding: float | None = None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
) -> IconMetrics:
    """Return placement and geometry measurements for one 24-unit candidate."""

    layout_scale, offset, stroke_compensation = _icon_layout(
        name,
        padding,
        mouse_width,
        stroke_width,
        rotate_ring_gap_ratio,
        rotate_ring_cap,
        tuning,
        alignment,
    )
    draw = _MetricsDraw()
    _draw_glyph(
        draw,
        offset,
        ICON_GRID * layout_scale,
        name,
        (1.0, 1.0, 1.0, 1.0),
        stroke_compensation=stroke_compensation,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
    )
    bounds = tuple(draw.bounds)
    enclosing_center, enclosing_radius = minimum_enclosing_circle(draw.boundary_points)
    anchor_center = _alignment_center_values(name, bounds, enclosing_center, alignment)
    origin_extent = max(math.hypot(x, y) for x, y in draw.boundary_points)
    anchor_extent = max(
        math.hypot(x - anchor_center[0], y - anchor_center[1]) for x, y in draw.boundary_points
    )
    return IconMetrics(bounds, enclosing_center, enclosing_radius, origin_extent, anchor_extent)


@cache
def production_icon_metrics(name: str) -> IconMetrics:
    """Return fitted shape metrics before optical translation, for stable layout."""

    preset = _production_icon_preset(name)
    if preset is not None:
        bounds, center, radius, origin_extent, anchor_extent = preset["metrics"]
        return IconMetrics(tuple(bounds), tuple(center), radius, origin_extent, anchor_extent)
    style = production_icon_style(name)
    return icon_metrics(
        name,
        style.padding,
        mouse_width=style.mouse_width,
        stroke_width=style.stroke_width,
        rotate_ring_gap_ratio=style.rotate_ring_gap_ratio,
        rotate_ring_cap=style.rotate_ring_cap,
        tuning=style.tuning,
        alignment=style.alignment,
    )
