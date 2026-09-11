"""Concept-only Mojive icon families for the UI feasibility workbench.

These painters deliberately do not feed production UI.  They provide a shared
24-unit comparison grid so candidate families can be reviewed together before
any production glyph is replaced.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache

from mojive.curves2d import (
    CORNER_SMOOTHING,
    arrow_mesh,
    box_handle_points,
    polyline_ribbon,
    smooth_polygon_corners,
    smooth_rect_points,
)
from mojive.draglink2d import smooth_union
from mojive.ui.viewport_widgets import (
    CAPSULE_SMOOTHING,
    OVERLAY_GEOMETRY,
    TOOL_GLYPH_SCALE,
    _dimensions_glyph_geometry,
    _rotate_visible_ring_polygons,
    _snap_glyph_shape,
)

ICON_GRID = 24.0
ICON_STROKE = 1.5
# Match the complete circular placement boundary used by Diagnostics. The reviewed
# severity family leaves about 0.72 grid units at its outer frame; 0.75 keeps that
# near-boundary scale while preserving a measurable gap at every size.
ICON_BOUND_DIAMETER = ICON_GRID
ICON_MIN_CLEARANCE = 0.5
ICON_DEFAULT_PADDING = 0.75
ICON_MAX_PADDING = 4.0
ROTATE_FRAME_PADDING = 0.0
STATUS_MOUSE_DEFAULT_WIDTH = 11.8
REVIEW_LOCKED_ICONS = frozenset(("status-info", "status-warning", "status-error"))
# ``_dimensions_glyph_geometry`` narrows a source stroke to compensate for the
# DrawList fringe. This concept-only source value makes Scale's fitted shaft
# match Rotate's fitted ring at the shared 24-unit slot. Preserve the reviewed
# center clearance and handle size while changing that one visible weight.
_SCALE_CONCEPT_SOURCE_STROKE = 2.287
SCALE_CONCEPT_GEOMETRY = replace(
    OVERLAY_GEOMETRY,
    tool_stroke=_SCALE_CONCEPT_SOURCE_STROKE,
    frame_center_gap_ratio=(
        OVERLAY_GEOMETRY.tool_stroke
        * OVERLAY_GEOMETRY.frame_center_gap_ratio
        / _SCALE_CONCEPT_SOURCE_STROKE
    ),
)

ICON_FAMILIES = (
    (
        "Viewport tools",
        (
            ("Move", "tool-move"),
            ("Rotate", "tool-rotate"),
            ("Scale", "tool-scale"),
            ("World", "tool-world"),
            ("Body", "tool-body"),
            ("Snap", "tool-snap"),
        ),
    ),
    (
        "Viewport playback",
        (
            ("Previous", "playback-previous"),
            ("Play", "playback-play"),
            ("Pause", "playback-pause"),
            ("Next", "playback-next"),
            ("Reset", "playback-reset"),
            ("Record", "playback-record"),
            ("Stop", "playback-stop"),
            ("More", "playback-more"),
        ),
    ),
    (
        "Keyframe transport",
        (
            ("First", "transport-first"),
            ("Previous", "transport-previous"),
            ("Play", "transport-play"),
            ("Pause", "transport-pause"),
            ("Next", "transport-next"),
            ("Last", "transport-last"),
            ("Reset", "transport-reset"),
            ("Record", "transport-record"),
            ("Stop", "transport-stop"),
            ("More", "transport-more"),
        ),
    ),
    (
        "Keyframes",
        (
            ("Snapshot", "key-snapshot"),
            ("Keyframe", "key-keyframe"),
            ("Add", "key-add"),
            ("Clear", "key-clear"),
            ("Previous key", "key-previous"),
            ("Next key", "key-next"),
            ("Fit", "key-fit"),
            ("Follow", "key-follow"),
            ("View", "key-view"),
        ),
    ),
    (
        "Panels",
        (
            ("Search", "panel-search"),
            ("Sort", "panel-sort"),
            ("Clear", "panel-clear"),
            ("Visible", "panel-visible"),
            ("Hidden", "panel-hidden"),
            ("Perspective", "panel-perspective"),
            ("Orthographic", "panel-orthographic"),
            ("Disclosure right", "panel-right"),
            ("Disclosure down", "panel-down"),
        ),
    ),
    (
        "Scene helpers",
        (
            ("Camera", "helper-camera"),
            ("Light", "helper-light"),
        ),
    ),
    (
        "Status & input",
        (
            ("Information", "status-info"),
            ("Warning", "status-warning"),
            ("Error", "status-error"),
            ("Mouse left", "status-mouse-left"),
            ("Mouse right", "status-mouse-right"),
            ("Mouse wheel", "status-mouse-wheel"),
        ),
    ),
)

# Placement is owned by the component that consumes the glyph. Viewport tools
# use their authored/semantic centers and the tight half-unit margin requested
# for the vertical capsule. Playback and Keyframes keep the same visible-box
# baseline, but own independent values so either component can be tuned alone.
ICON_GROUP_LAYOUT_DEFAULTS = {
    "Viewport tools": (0.0, 0.5),
    "Viewport playback": (0.0, ICON_DEFAULT_PADDING),
    "Keyframe transport": (0.0, ICON_DEFAULT_PADDING),
    "Keyframes": (0.0, ICON_DEFAULT_PADDING),
    "Panels": (0.0, ICON_DEFAULT_PADDING),
    "Scene helpers": (0.0, ICON_DEFAULT_PADDING),
    "Status & input": (0.0, ICON_DEFAULT_PADDING),
}

# Open chevrons occupy more of a circular cell than Play/Pause at the same
# enclosing radius. Keep each correction with the component that consumes it.
ICON_PADDING_BIAS = {
    "playback-previous": 1.85,
    "playback-next": 1.85,
    "transport-previous": 1.7,
    "transport-next": 1.7,
}

ICON_LIBRARY_TABS = (
    "Overview",
    "UI context",
    "Capsules",
    *(label for label, _icons in ICON_FAMILIES),
)
ICON_GROUP_BY_SLUG = {
    label.casefold().replace(" ", "-").replace("&", "and"): label for label in ICON_LIBRARY_TABS
}


@dataclass(frozen=True)
class IconMetrics:
    bounds: tuple[float, float, float, float]
    radial_extent: float
    area_centroid: tuple[float, float]
    bounding_center: tuple[float, float]
    bounding_radius: float

    @property
    def center_offset(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5

    @property
    def radial_clearance(self) -> float:
        return ICON_BOUND_DIAMETER * 0.5 - self.radial_extent


def _signed_polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1], strict=True)
    )


def _triangle_cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _inside_counterclockwise_triangle(point, a, b, c, epsilon: float) -> bool:
    return (
        _triangle_cross(a, b, point) >= -epsilon
        and _triangle_cross(b, c, point) >= -epsilon
        and _triangle_cross(c, a, point) >= -epsilon
    )


@lru_cache(maxsize=128)
def _simple_polygon_indices(points: tuple[tuple[float, float], ...]) -> tuple[int, ...]:
    """Triangulate one simple screen-clockwise contour without an ImGui context."""

    if len(points) < 3:
        return ()
    vertices = list(range(len(points)))
    if _signed_polygon_area(points) < 0.0:
        vertices.reverse()
    scale = max(max(abs(value) for point in points for value in point), 1.0)
    epsilon = scale * scale * 1e-10
    indices: list[int] = []
    while len(vertices) > 3:
        for offset, current in enumerate(vertices):
            previous = vertices[offset - 1]
            following = vertices[(offset + 1) % len(vertices)]
            a, b, c = points[previous], points[current], points[following]
            if _triangle_cross(a, b, c) <= epsilon:
                continue
            if any(
                candidate not in (previous, current, following)
                and _inside_counterclockwise_triangle(points[candidate], a, b, c, epsilon)
                for candidate in vertices
            ):
                continue
            indices.extend((previous, current, following))
            del vertices[offset]
            break
        else:
            raise RuntimeError("icon contour could not be triangulated")
    indices.extend(vertices)
    return tuple(indices)


def minimum_enclosing_circle(points) -> tuple[tuple[float, float], float]:
    """Return the exact smallest circle for a finite set of sampled boundary points."""

    values = list(dict.fromkeys((float(x), float(y)) for x, y in points))
    if not values:
        return (0.0, 0.0), 0.0
    random.Random(0).shuffle(values)

    def contains(circle, point) -> bool:
        center, radius = circle
        return math.dist(center, point) <= radius + 1e-7

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
                    if all(contains(candidate, p) for p in (a, b, c))
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


class _Painter:
    def __init__(self, draw, center, size: float, color) -> None:
        self.draw = draw
        self.cx, self.cy = (float(center[0]), float(center[1]))
        self.scale = float(size) / ICON_GRID
        self.color = color
        self.stroke = ICON_STROKE * self.scale

    def point(self, x: float, y: float) -> tuple[float, float]:
        return self.cx + x * self.scale, self.cy + y * self.scale

    def points(self, values) -> tuple[tuple[float, float], ...]:
        return tuple(self.point(x, y) for x, y in values)

    def line(self, a, b, *, width: float = ICON_STROKE) -> None:
        self.draw.line(
            self.point(*a),
            self.point(*b),
            self.color,
            width * self.scale,
            cap="round",
        )

    def arrow(
        self,
        a,
        b,
        *,
        width: float = ICON_STROKE,
        head_length: float = 3.0,
        head_width: float = 4.8,
        corner_radius: float = 0.45,
        round_tail: bool = False,
    ) -> None:
        """Draw one continuous shaft-and-head silhouette."""

        self.draw.arrow(
            self.point(*a),
            self.point(*b),
            self.color,
            width * self.scale,
            head_length=head_length * self.scale,
            head_width=head_width * self.scale,
            corner_radius=corner_radius * self.scale,
            join_radius=corner_radius * 0.55 * self.scale,
            round_tail=round_tail,
        )

    def polyline(self, points, *, closed: bool = False, width: float = ICON_STROKE) -> None:
        path = self.points(points)
        self.draw.polyline(
            path,
            self.color,
            width * self.scale,
            closed=closed,
            cap="butt" if closed else "round",
        )

    def polygon(self, points, *, fringe_width: float | None = None) -> None:
        if fringe_width is None:
            self.draw.fringed_concave_fill(self.points(points), self.color)
            return
        local = tuple((float(x) * self.scale, float(y) * self.scale) for x, y in points)
        if _signed_polygon_area(local) < 0.0:
            local = local[::-1]
        screen = tuple((self.cx + x, self.cy + y) for x, y in local)
        self.draw.indexed_fill(
            screen,
            _simple_polygon_indices(local),
            self.color,
            outline=screen,
            fringe_width=fringe_width,
        )

    def indexed_fill(
        self,
        points,
        indices,
        *,
        outline,
        hole=(),
        fringe_width: float = 1.0,
    ) -> None:
        self.draw.indexed_fill(
            self.points(points),
            indices,
            self.color,
            outline=self.points(outline),
            hole=self.points(hole),
            fringe_width=fringe_width,
        )

    def smooth_polygon(
        self,
        points,
        radius: float,
        *,
        corners: tuple[int, ...] | None = None,
        convex_only: bool = True,
    ) -> None:
        """Fill one contour with G3-continuous selected corners."""

        source = tuple(points)
        selected = tuple(range(len(source))) if corners is None else corners
        path = smooth_polygon_corners(
            source,
            radius,
            selected,
            smoothing=CORNER_SMOOTHING,
            convex_only=convex_only,
        )
        self.polygon(path)

    def smooth_outline(
        self,
        points,
        radius: float,
        *,
        corners: tuple[int, ...] | None = None,
        convex_only: bool = True,
        width: float = ICON_STROKE,
    ) -> None:
        """Stroke one closed contour with G3-continuous selected corners."""

        source = tuple(points)
        selected = tuple(range(len(source))) if corners is None else corners
        path = smooth_polygon_corners(
            source,
            radius,
            selected,
            smoothing=CORNER_SMOOTHING,
            convex_only=convex_only,
        )
        self.polyline(path, closed=True, width=width)

    def box_handle(
        self,
        start,
        end,
        *,
        width: float,
        head_size: float,
        corner_radius: float,
    ) -> None:
        """Draw one shaft and square endpoint as a shared G3 contour."""

        path = box_handle_points(
            start,
            end,
            width,
            head_size,
            corner_radius=corner_radius,
            smoothing=CORNER_SMOOTHING,
        )
        self.polygon(path)

    def circle(self, x: float, y: float, radius: float, *, width: float = ICON_STROKE) -> None:
        self.draw.circle(
            self.point(x, y),
            radius * self.scale,
            self.color,
            width * self.scale,
            segments=max(24, round(radius * self.scale * 4.0)),
        )

    def circle_filled(self, x: float, y: float, radius: float) -> None:
        self.draw.circle_filled(
            self.point(x, y),
            radius * self.scale,
            self.color,
            segments=max(16, round(radius * self.scale * 4.0)),
        )

    def rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        rounding: float = 0.0,
        width: float = ICON_STROKE,
    ) -> None:
        self.draw.rect(
            self.point(x0, y0),
            self.point(x1, y1),
            self.color,
            width * self.scale,
            rounding=rounding * self.scale,
        )

    def rect_filled(
        self, x0: float, y0: float, x1: float, y1: float, *, rounding: float = 0.0
    ) -> None:
        self.draw.rect_filled(
            self.point(x0, y0),
            self.point(x1, y1),
            self.color,
            rounding=rounding * self.scale,
        )

    def arc(
        self,
        radius: float,
        start_degrees: float,
        end_degrees: float,
        *,
        center=(0.0, 0.0),
        width: float = ICON_STROKE,
    ) -> tuple[tuple[float, float], ...]:
        count = max(16, round(abs(end_degrees - start_degrees) / 12.0))
        points = tuple(
            (
                center[0] + radius * math.cos(math.radians(angle)),
                center[1] + radius * math.sin(math.radians(angle)),
            )
            for angle in (
                start_degrees + (end_degrees - start_degrees) * index / count
                for index in range(count + 1)
            )
        )
        self.polyline(points, width=width)
        return points


def _triangle(p: _Painter, direction: float, *, center_x: float = 0.0, scale: float = 1.0) -> None:
    p.smooth_polygon(
        (
            (center_x - direction * 6.4 * scale, -6.8 * scale),
            (center_x + direction * 6.4 * scale, 0.0),
            (center_x - direction * 6.4 * scale, 6.8 * scale),
        ),
        1.12 * scale,
    )


def _rounded_polyline(
    p: _Painter,
    points,
    *,
    width: float = ICON_STROKE,
    radius: float = 0.68,
) -> None:
    """Fill one open centerline as a joined G3 contour."""

    _left, _right, outline = polyline_ribbon(tuple(points), width)
    p.smooth_polygon(outline, radius, convex_only=False)


def _g3_rect(p: _Painter, x0: float, y0: float, x1: float, y1: float, radius: float) -> None:
    p.polygon(smooth_rect_points(x0, y0, x1, y1, radius, smoothing=CORNER_SMOOTHING))


def _filled_ring(
    p: _Painter,
    radius: float,
    width: float,
    *,
    fringe_width: float = 1.0,
    segments: int = 64,
) -> None:
    """Draw an annulus through the same filled-mesh path as the inner Rotate rings."""

    outer_radius = radius + width * 0.5
    inner_radius = radius - width * 0.5
    outer = tuple(
        (
            outer_radius * math.cos(index * math.tau / segments),
            outer_radius * math.sin(index * math.tau / segments),
        )
        for index in range(segments)
    )
    inner = tuple(
        (
            inner_radius * math.cos(index * math.tau / segments),
            inner_radius * math.sin(index * math.tau / segments),
        )
        for index in range(segments)
    )
    points = outer + inner
    indices = []
    for index in range(segments):
        following = (index + 1) % segments
        indices.extend((index, following, segments + following))
        indices.extend((index, segments + following, segments + index))
    p.indexed_fill(
        points,
        tuple(indices),
        outline=outer,
        hole=inner,
        fringe_width=fringe_width,
    )


def _rotate_fringe_width(p: _Painter, ring_width: float, gap: float) -> float:
    """Keep two AA ramps from consuming a subpixel Rotate crossing gap."""

    stroke_pixels = ring_width * p.scale
    gap_pixels = gap * p.scale
    return min(1.0, stroke_pixels * 0.5, gap_pixels * (3.0 / 8.0))


def _chevron(p: _Painter, direction: float, x: float, *, scale: float = 1.0) -> None:
    _rounded_polyline(
        p,
        (
            (x - direction * 2.6 * scale, -4.4 * scale),
            (x + direction * 2.2 * scale, 0.0),
            (x - direction * 2.6 * scale, 4.4 * scale),
        ),
        width=ICON_STROKE * scale,
        radius=0.68 * scale,
    )


def _arc_arrow(
    p: _Painter,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    *,
    center=(0.0, 0.0),
    stroke: float = ICON_STROKE,
    head_length: float = 3.0,
    head_half_width: float = 2.4,
) -> None:
    """Draw a circular arrow as one joined ribbon instead of two shapes."""

    direction = 1.0 if end_degrees >= start_degrees else -1.0
    count = max(24, round(abs(end_degrees - start_degrees) / 8.0))
    angles = tuple(
        math.radians(start_degrees + (end_degrees - start_degrees) * index / count)
        for index in range(count + 1)
    )
    half_width = stroke * 0.5
    outer = tuple(
        (
            center[0] + (radius + half_width) * math.cos(angle),
            center[1] + (radius + half_width) * math.sin(angle),
        )
        for angle in angles
    )
    inner = tuple(
        (
            center[0] + (radius - half_width) * math.cos(angle),
            center[1] + (radius - half_width) * math.sin(angle),
        )
        for angle in reversed(angles)
    )
    end_angle = angles[-1]
    end = (
        center[0] + radius * math.cos(end_angle),
        center[1] + radius * math.sin(end_angle),
    )
    tangent = (-math.sin(end_angle) * direction, math.cos(end_angle) * direction)
    normal = (math.cos(end_angle), math.sin(end_angle))
    base_center = (end[0] - tangent[0] * 0.8, end[1] - tangent[1] * 0.8)
    base_outer = (
        base_center[0] + normal[0] * head_half_width,
        base_center[1] + normal[1] * head_half_width,
    )
    base_inner = (
        base_center[0] - normal[0] * head_half_width,
        base_center[1] - normal[1] * head_half_width,
    )
    tip = (end[0] + tangent[0] * head_length, end[1] + tangent[1] * head_length)
    outline = (*outer, base_outer, tip, base_inner, *inner)
    head_start = len(outer)
    p.smooth_polygon(
        outline,
        0.42,
        corners=(head_start, head_start + 1, head_start + 2),
        convex_only=False,
    )
    start_angle = angles[0]
    p.circle_filled(
        center[0] + radius * math.cos(start_angle),
        center[1] + radius * math.sin(start_angle),
        half_width,
    )


def _draw_tool(p: _Painter, name: str) -> None:
    if name == "tool-move":
        # Keep the accepted Icon Library candidate as one connected G3 outline;
        # the wider central cross remains legible at the 14-point specimen.
        tip, base, wing, shaft = 8.7, 5.6, 2.65, 0.82
        p.smooth_polygon(
            (
                (0.0, -tip),
                (wing, -base),
                (shaft, -base),
                (shaft, -shaft),
                (base, -shaft),
                (base, -wing),
                (tip, 0.0),
                (base, wing),
                (base, shaft),
                (shaft, shaft),
                (shaft, base),
                (wing, base),
                (0.0, tip),
                (-wing, base),
                (-shaft, base),
                (-shaft, shaft),
                (-base, shaft),
                (-base, wing),
                (-tip, 0.0),
                (-base, -wing),
                (-base, -shaft),
                (-shaft, -shaft),
                (-shaft, -base),
                (-wing, -base),
            ),
            0.42,
            convex_only=False,
        )
    elif name == "tool-rotate":
        # Draw the established three cyclic half-rings locally. Reset keeps the
        # separate one-arrow grammar, and production geometry remains untouched.
        production_scale = 0.82
        glyph_scale = production_scale * TOOL_GLYPH_SCALE
        ring_width = OVERLAY_GEOMETRY.tool_stroke * production_scale
        ring_gap = OVERLAY_GEOMETRY.rotate_ring_gap * production_scale
        fringe_width = _rotate_fringe_width(p, ring_width, ring_gap)
        _filled_ring(
            p,
            10.0 * glyph_scale,
            ring_width,
            fringe_width=fringe_width,
        )
        for ring in _rotate_visible_ring_polygons(
            OVERLAY_GEOMETRY.tool_stroke,
            OVERLAY_GEOMETRY.rotate_ring_gap_ratio,
            OVERLAY_GEOMETRY.rotate_ring_cap,
            CAPSULE_SMOOTHING,
        ):
            for local in ring:
                p.polygon(
                    tuple((x * glyph_scale, y * glyph_scale) for x, y in local),
                    fringe_width=fringe_width,
                )
    elif name == "tool-scale":
        # Keep the accepted viewport Scale glyph as the source of truth. It
        # supplies the original reach, center clearance, joined G3 shafts and
        # endpoint blocks instead of maintaining a second approximation here.
        paths, dot_radius = _dimensions_glyph_geometry(
            (0.0, 0.0),
            1.0,
            SCALE_CONCEPT_GEOMETRY,
            smoothing=CAPSULE_SMOOTHING,
        )
        for path in paths:
            p.polygon(path)
        p.circle_filled(0.0, 0.0, dot_radius)
    elif name == "tool-world":
        # Sparse stroked symbols need a larger authored envelope than solid
        # tools to carry comparable visual weight in the same 24-unit slot.
        p.circle(0.0, 0.0, 8.75, width=1.45)
        # Centerline endpoints account for both strokes. The round caps meet
        # the globe's inner edge instead of painting through the outer ring.
        inner_reach = 8.75 - 1.45 * 0.5 - 1.35 * 0.5
        p.line((-inner_reach, 0.0), (inner_reach, 0.0), width=1.35)
        ellipse = tuple(
            (
                3.53 * math.cos(index * math.tau / 32),
                inner_reach * math.sin(index * math.tau / 32),
            )
            for index in range(32)
        )
        p.polyline(ellipse, closed=True, width=1.35)
    elif name == "tool-body":
        top = (0.0, -8.62)
        left = (-7.50, -4.31)
        right = (7.50, -4.31)
        bottom = (0.0, 8.62)
        lower_left = (-7.50, 4.31)
        lower_right = (7.50, 4.31)
        # A cube has three visible faces: the center joins the two rear side
        # corners and the lower vertex. The former top-to-center edge invented
        # a fourth face. Inset the spoke ends below the outer stroke and paint
        # the G3 shell last so no round cap protrudes through a cube vertex.
        junction = (0.0, 0.0)
        inset = 1.62
        for endpoint in (left, right, bottom):
            length = math.hypot(endpoint[0], endpoint[1])
            end = (
                endpoint[0] * (length - inset) / length,
                endpoint[1] * (length - inset) / length,
            )
            p.line(junction, end, width=1.4)
        p.circle_filled(*junction, 0.7)
        p.smooth_outline(
            (top, right, lower_right, bottom, lower_left, left),
            0.5,
            width=1.45,
        )
    else:
        # Snap is the production Tool Column contour. Its U-turn is generated
        # from the same G3 profile as Mojive capsules instead of approximating
        # the curve with a semicircle plus independently capped stems.
        production_scale = 0.95
        path = _snap_glyph_shape(
            production_scale * TOOL_GLYPH_SCALE,
            CAPSULE_SMOOTHING,
        )
        p.polyline(path, width=OVERLAY_GEOMETRY.tool_stroke * production_scale)
        for x, y in (path[0], path[-1]):
            p.smooth_polygon(
                (
                    (x - 1.35, y - 0.9),
                    (x + 1.35, y - 0.9),
                    (x + 1.35, y + 0.9),
                    (x - 1.35, y + 0.9),
                ),
                0.42,
            )


def _draw_transport(p: _Painter, name: str) -> None:
    kind = name.removeprefix("transport-").removeprefix("playback-")
    if kind == "play":
        _triangle(p, 1.0)
    elif kind == "pause":
        _g3_rect(p, -5.0, -6.8, -1.5, 6.8, 0.92)
        _g3_rect(p, 1.5, -6.8, 5.0, 6.8, 0.92)
    elif kind in {"previous", "next"}:
        direction = -1.0 if kind == "previous" else 1.0
        _chevron(p, direction, direction * 0.24, scale=1.18)
    elif kind in {"first", "last"}:
        direction = -1.0 if kind == "first" else 1.0
        # A skip-to-end symbol pairs one filled transport triangle with a bar.
        # The small separation is intentional and stays symmetrical when mirrored.
        _triangle(p, direction, center_x=-direction * 1.27, scale=0.72)
        x = direction * 5.13
        _g3_rect(p, x - 0.75, -5.8, x + 0.75, 5.8, 0.58)
    elif kind == "reset":
        _arc_arrow(p, 6.8, -52.0, 255.0, center=(0.0, 0.47))
    elif kind == "record":
        p.circle_filled(0.0, 0.0, 4.8)
    elif kind == "stop":
        _g3_rect(p, -4.8, -4.8, 4.8, 4.8, 1.05)
    else:
        _rounded_polyline(p, ((-5.6, -2.8), (0.0, 2.8), (5.6, -2.8)))


def _diamond(p: _Painter, center=(0.0, 0.0), radius: float = 5.4, *, filled: bool) -> None:
    points = (
        (center[0], center[1] - radius),
        (center[0] + radius, center[1]),
        (center[0], center[1] + radius),
        (center[0] - radius, center[1]),
    )
    if filled:
        p.smooth_polygon(points, 0.62)
    else:
        p.smooth_outline(points, 0.62)


def _draw_keyframe(p: _Painter, name: str) -> None:
    kind = name.removeprefix("key-")
    if kind == "snapshot":
        _draw_camera(p)
    elif kind == "keyframe":
        _diamond(p, radius=5.0, filled=True)
    elif kind == "add":
        p.line((-6.6, 0.0), (6.6, 0.0), width=1.65)
        p.line((0.0, -6.6), (0.0, 6.6), width=1.65)
    elif kind == "clear":
        p.line((-5.8, -5.8), (5.8, 5.8), width=1.65)
        p.line((5.8, -5.8), (-5.8, 5.8), width=1.65)
    elif kind in {"previous", "next"}:
        direction = -1.0 if kind == "previous" else 1.0
        _chevron(p, direction, direction * 4.4, scale=0.68)
        _diamond(p, center=(-direction * 3.2, 0.0), radius=3.5, filled=True)
    elif kind == "fit":
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            p.polyline(
                (
                    (sx * 3.5, sy * 6.5),
                    (sx * 6.5, sy * 6.5),
                    (sx * 6.5, sy * 3.5),
                )
            )
    elif kind == "follow":
        p.arrow((-7.38, 0.0), (4.42, 0.0), head_length=3.2, head_width=5.0)
        p.line((6.62, -5.8), (6.62, 5.8))
    else:
        p.rect(-7.5, -5.1, 7.5, 5.1, rounding=1.8)
        p.circle_filled(0.0, 0.0, 1.9)


def _eye_points() -> tuple[tuple[float, float], ...]:
    top = tuple((-8.2 + index * 2.05, -4.8 * math.sin(math.pi * index / 8)) for index in range(9))
    bottom = tuple((8.2 - index * 2.05, 4.8 * math.sin(math.pi * index / 8)) for index in range(9))
    return (*top, *bottom[1:-1])


@lru_cache(maxsize=1)
def _search_icon_mesh():
    """Return one hollow lens and handle joined by a G3 smooth union."""

    outer_radius = 5.55
    inner_radius = 4.05
    handle_half_width = 0.825
    handle_start = 3.65
    handle_end = 10.775
    handle_tip = handle_end + handle_half_width
    blend = 1.05

    def field(x: float, y: float) -> float:
        circle = math.hypot(x, y) - outer_radius
        along = min(handle_end, max(handle_start, x))
        handle = math.hypot(x - along, y) - handle_half_width
        return float(smooth_union(circle, handle, blend))

    # A uniform parameter avoids near-duplicate strip columns at the circle and
    # handle landmarks. Those tiny edges destabilize fringe normals and appear
    # as spikes on the 112-point lens even when the implicit field is smooth.
    count = 257
    xs = [
        -outer_radius + (handle_tip + outer_radius) * index / (count - 1) for index in range(count)
    ]
    # The fill strips and hole fringe must meet at the same two axis points.
    # Replace the nearest uniform columns instead of adding almost coincident
    # landmarks, which would recreate unstable fringe normals.
    for landmark in (-inner_radius, inner_radius):
        nearest = min(range(len(xs)), key=lambda index: abs(xs[index] - landmark))
        xs[nearest] = landmark
    xs = tuple(sorted(set(xs)))
    columns = []
    for x in xs:
        if field(x, 0.0) > 1e-7:
            outer_y = 0.0
        else:
            lo, hi = 0.0, outer_radius + blend + handle_half_width
            while field(x, hi) <= 0.0:
                hi *= 1.5
            for _ in range(36):
                mid = (lo + hi) * 0.5
                if field(x, mid) <= 0.0:
                    lo = mid
                else:
                    hi = mid
            outer_y = (lo + hi) * 0.5
        inner_y = math.sqrt(max(0.0, inner_radius * inner_radius - x * x))
        columns.append((x, outer_y, inner_y))

    vertices = tuple(
        point
        for x, outer_y, inner_y in columns
        for point in ((x, -outer_y), (x, -inner_y), (x, inner_y), (x, outer_y))
    )
    indices = []
    for column in range(len(columns) - 1):
        start = column * 4
        following = start + 4
        for low, high in ((0, 1), (2, 3)):
            for triangle in (
                (start + low, following + low, following + high),
                (start + low, following + high, start + high),
            ):
                a, b, c = (vertices[index] for index in triangle)
                twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if twice_area > 1e-9:
                    indices.extend(triangle)

    outline = tuple((x, -outer_y) for x, outer_y, _inner_y in columns) + tuple(
        (x, outer_y) for x, outer_y, _inner_y in reversed(columns[1:-1])
    )
    hole_columns = tuple(
        (x, inner_y) for x, _outer_y, inner_y in columns if -inner_radius <= x <= inner_radius
    )
    # Reuse the strip's inner vertices for AA. An independently sampled circle
    # leaves subpixel wedges of the opaque fill exposed inside the hole.
    hole = tuple((x, -inner_y) for x, inner_y in hole_columns) + tuple(
        (x, inner_y) for x, inner_y in reversed(hole_columns[1:-1])
    )

    cosine = math.sqrt(0.5)

    def place(path):
        return tuple(
            (-1.7 + x * cosine - y * cosine, -1.7 + x * cosine + y * cosine) for x, y in path
        )

    return place(vertices), tuple(indices), place(outline), place(hole)


def _draw_panel(p: _Painter, name: str) -> None:
    kind = name.removeprefix("panel-")
    if kind == "search":
        vertices, indices, outline, hole = _search_icon_mesh()
        p.indexed_fill(vertices, indices, outline=outline, hole=hole)
    elif kind == "sort":
        for y, end in ((-5.09, 2.24), (0.11, 0.04), (5.31, -2.16)):
            p.line((-6.76, y), (end, y), width=1.45)
        # The round tail aligns with the top bar centerline; the head tip aligns
        # with the visible lower edge of the bottom bar.
        p.arrow(
            (5.54, -5.09),
            (5.54, 5.31 + 1.45 * 0.5),
            width=1.4,
            head_length=3.0,
            head_width=4.6,
            round_tail=True,
        )
    elif kind == "clear":
        p.line((-5.8, -5.8), (5.8, 5.8), width=1.65)
        p.line((5.8, -5.8), (-5.8, 5.8), width=1.65)
    elif kind == "visible":
        p.polyline(_eye_points(), closed=True, width=1.45)
        p.circle_filled(0.0, 0.0, 2.05)
    elif kind == "hidden":
        # Match Mojive's hierarchy toggle: a closed curved lid with three
        # lashes. Its visible bounds are shifted onto the shared icon center.
        radius_x = 8.0
        lid_height = 3.2
        offset_y = -2.85
        lid = tuple(
            (
                -radius_x + radius_x * 2.0 * index / 8.0,
                offset_y + math.sin(math.pi * index / 8.0) * lid_height,
            )
            for index in range(9)
        )
        p.polyline(lid, width=1.45)
        for offset in (-0.52, 0.0, 0.52):
            lash_x = radius_x * offset
            lash_y = offset_y + lid_height * math.sqrt(max(0.0, 1.0 - offset**2))
            p.line(
                (lash_x, lash_y),
                (lash_x + offset * 1.95, lash_y + 2.65),
                width=1.15,
            )
    elif kind in {"perspective", "orthographic"}:
        near = 3.0 if kind == "perspective" else 6.1
        p.smooth_outline(
            ((-6.8, -near), (6.8, -6.1), (6.8, 6.1), (-6.8, near)),
            0.52,
        )
    elif kind == "right":
        p.smooth_polygon(((-3.8, -6.2), (3.8, 0.0), (-3.8, 6.2)), 0.62)
    else:
        p.smooth_polygon(((-6.2, -3.8), (0.0, 3.8), (6.2, -3.8)), 0.62)


def _draw_camera(p: _Painter) -> None:
    """Draw one camera outline with an integrated top and centered lens."""

    # The body and viewfinder form one contour, so no interior stroke crosses
    # the shell. The lens uses the geometric center of the rectangular body.
    p.smooth_outline(
        (
            (-7.3, -3.7),
            (-3.5, -3.7),
            (-2.2, -6.0),
            (2.2, -6.0),
            (3.5, -3.7),
            (7.3, -3.7),
            (7.3, 6.0),
            (-7.3, 6.0),
        ),
        0.48,
    )
    p.circle(0.0, 1.15, 3.05)


def _draw_light(p: _Painter) -> None:
    """Draw a continuous bulb shell with a clear gap before every ray."""

    count = 28
    arc = tuple(
        (
            4.2 * math.cos(math.radians(150.0 + 240.0 * index / count)),
            4.2 * math.sin(math.radians(150.0 + 240.0 * index / count)) + 0.3,
        )
        for index in range(count + 1)
    )
    p.smooth_outline(
        (
            *arc,
            (2.8, 3.9),
            (2.5, 5.1),
            (2.5, 6.0),
            (1.7, 7.3),
            (-1.7, 7.3),
            (-2.5, 6.0),
            (-2.5, 5.1),
            (-2.8, 3.9),
        ),
        0.42,
        convex_only=False,
    )
    p.line((-2.4, 5.65), (2.4, 5.65), width=1.3)
    for a, b in (
        ((0.0, -5.6), (0.0, -7.3)),
        ((-5.2, -4.9), (-6.3, -6.0)),
        ((5.2, -4.9), (6.3, -6.0)),
        ((-5.9, 0.3), (-7.6, 0.3)),
        ((5.9, 0.3), (7.6, 0.3)),
    ):
        p.line(a, b)


def _draw_helper(p: _Painter, name: str) -> None:
    if name == "helper-camera":
        _draw_camera(p)
    else:
        _draw_light(p)


def _draw_status(
    p: _Painter,
    name: str,
    accent_color=None,
    *,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> None:
    kind = name.removeprefix("status-")
    if kind in {"info", "warning", "error"}:
        p.circle(0.0, 0.0, 9.0, width=1.55)
        if kind == "error":
            p.line((-3.4, -3.4), (3.4, 3.4), width=1.95)
            p.line((3.4, -3.4), (-3.4, 3.4), width=1.95)
        else:
            dot_y = -4.4 if kind == "info" else 4.4
            stem_a = (-0.0, -1.7 if kind == "info" else -5.6)
            stem_b = (0.0, 5.2 if kind == "info" else 1.3)
            p.circle_filled(0.0, dot_y, 1.2)
            p.line(stem_a, stem_b, width=1.95)
        return

    accent = (
        _Painter(p.draw, (p.cx, p.cy), p.scale * ICON_GRID, accent_color)
        if accent_color is not None
        else p
    )
    half_width = float(mouse_width) * 0.5
    if not 5.0 <= mouse_width <= 22.0:
        raise ValueError("status mouse width must be between 5 and 22 grid units")
    p.rect(-half_width, -7.8, half_width, 7.8, rounding=4.4, width=1.45)
    p.line((-half_width + 0.3, -1.1), (half_width - 0.3, -1.1), width=1.2)
    control_x = max(0.0, half_width - 3.7)
    if kind == "mouse-left":
        accent.circle_filled(-control_x, -4.2, 1.25)
    elif kind == "mouse-right":
        accent.circle_filled(control_x, -4.2, 1.25)
    else:
        accent.rect_filled(-0.9, -5.8, 0.9, -2.5, rounding=0.9)


def _draw_concept_icon_raw(
    draw,
    center,
    size: float,
    name: str,
    color,
    *,
    accent_color=None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> None:
    """Draw authored geometry before shared optical placement is applied."""

    painter = _Painter(draw, center, size, color)
    if name.startswith("tool-"):
        _draw_tool(painter, name)
    elif name.startswith(("transport-", "playback-")):
        _draw_transport(painter, name)
    elif name.startswith("key-"):
        _draw_keyframe(painter, name)
    elif name.startswith("panel-"):
        _draw_panel(painter, name)
    elif name.startswith("helper-"):
        _draw_helper(painter, name)
    elif name.startswith("status-"):
        _draw_status(painter, name, accent_color, mouse_width=mouse_width)
    else:
        raise ValueError(f"unknown concept icon: {name!r}")


class _MetricsDraw:
    """Measure bounds and approximate filled area from authored primitives."""

    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.radial_extent = 0.0
        self.boundary_points: list[tuple[float, float]] = []
        self.filled_area = 0.0
        self.area_moment = [0.0, 0.0]

    def _add(self, points, pad: float = 0.0) -> None:
        for raw_x, raw_y in points:
            x, y = float(raw_x), float(raw_y)
            self.bounds[0] = min(self.bounds[0], x - pad)
            self.bounds[1] = min(self.bounds[1], y - pad)
            self.bounds[2] = max(self.bounds[2], x + pad)
            self.bounds[3] = max(self.bounds[3], y + pad)
            self.radial_extent = max(self.radial_extent, math.hypot(x, y) + pad)
            if pad > 0.0:
                self.boundary_points.extend(
                    (
                        x + pad * math.cos(index * math.tau / 32),
                        y + pad * math.sin(index * math.tau / 32),
                    )
                    for index in range(32)
                )
            else:
                self.boundary_points.append((x, y))

    def _add_mass(self, area: float, center) -> None:
        if area <= 0.0:
            return
        self.filled_area += area
        self.area_moment[0] += area * float(center[0])
        self.area_moment[1] += area * float(center[1])

    def _add_polygon_mass(self, points) -> None:
        path = tuple((float(point[0]), float(point[1])) for point in points)
        if len(path) < 3:
            return
        twice_area = 0.0
        moment_x = 0.0
        moment_y = 0.0
        for current, following in zip(path, (*path[1:], path[0]), strict=True):
            cross = current[0] * following[1] - following[0] * current[1]
            twice_area += cross
            moment_x += (current[0] + following[0]) * cross
            moment_y += (current[1] + following[1]) * cross
        if abs(twice_area) <= 1e-9:
            return
        area = abs(twice_area) * 0.5
        center = (moment_x / (3.0 * twice_area), moment_y / (3.0 * twice_area))
        self._add_mass(area, center)

    def _add_stroke_mass(self, points, width: float, *, closed: bool, round_caps: bool) -> None:
        path = tuple((float(point[0]), float(point[1])) for point in points)
        if len(path) < 2 or width <= 0.0:
            return
        following = (*path[1:], path[0]) if closed else path[1:]
        starts = path if closed else path[:-1]
        for start, end in zip(starts, following, strict=True):
            length = math.dist(start, end)
            self._add_mass(length * width, ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5))
        if round_caps:
            cap_area = math.pi * (width * 0.5) ** 2 * 0.5
            self._add_mass(cap_area, path[0])
            self._add_mass(cap_area, path[-1])

    def line(self, a, b, _color, width, **_kwargs) -> None:
        width = float(width)
        self._add((a, b), width * 0.5)
        self._add_stroke_mass((a, b), width, closed=False, round_caps=True)

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
        self._add_polygon_mass(points)

    def polyline(self, points, _color, width, **kwargs) -> None:
        points = tuple(points)
        width = float(width)
        closed = bool(kwargs.get("closed", False))
        self._add(points, width * 0.5)
        self._add_stroke_mass(
            points,
            width,
            closed=closed,
            round_caps=not closed and kwargs.get("cap", "butt") != "butt",
        )

    def fringed_concave_fill(self, points, _color, **_kwargs) -> None:
        points = tuple(points)
        self._add(points)
        self._add_polygon_mass(points)

    def indexed_fill(
        self,
        points,
        indices,
        _color,
        *,
        outline=(),
        hole=(),
        **_kwargs,
    ) -> None:
        points = tuple(points)
        self._add(tuple(outline) or points)
        for offset in range(0, len(indices), 3):
            triangle = tuple(points[index] for index in indices[offset : offset + 3])
            twice_area = abs(
                (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1])
                - (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[0][0])
            )
            if twice_area <= 1e-12:
                continue
            self._add_mass(
                twice_area * 0.5,
                (
                    sum(point[0] for point in triangle) / 3.0,
                    sum(point[1] for point in triangle) / 3.0,
                ),
            )

    def concave_fill(self, points, color) -> None:
        self.fringed_concave_fill(points, color)

    def circle(self, center, radius, _color, width, **_kwargs) -> None:
        radius, width = float(radius), float(width)
        self._add((center,), radius + width * 0.5)
        outer = radius + width * 0.5
        inner = max(0.0, radius - width * 0.5)
        self._add_mass(math.pi * (outer * outer - inner * inner), center)

    def circle_filled(self, center, radius, _color, **_kwargs) -> None:
        radius = float(radius)
        self._add((center,), radius)
        self._add_mass(math.pi * radius * radius, center)

    def rect(self, lo, hi, _color, width, **_kwargs) -> None:
        width = float(width)
        self._add((lo, hi), width * 0.5)
        outer_width = abs(float(hi[0]) - float(lo[0])) + width
        outer_height = abs(float(hi[1]) - float(lo[1])) + width
        inner_width = max(0.0, outer_width - 2.0 * width)
        inner_height = max(0.0, outer_height - 2.0 * width)
        self._add_mass(
            outer_width * outer_height - inner_width * inner_height,
            ((float(lo[0]) + float(hi[0])) * 0.5, (float(lo[1]) + float(hi[1])) * 0.5),
        )

    def rect_filled(self, lo, hi, _color, **_kwargs) -> None:
        self._add((lo, hi))
        width = abs(float(hi[0]) - float(lo[0]))
        height = abs(float(hi[1]) - float(lo[1]))
        self._add_mass(
            width * height,
            ((float(lo[0]) + float(hi[0])) * 0.5, (float(lo[1]) + float(hi[1])) * 0.5),
        )


def _measure_raw_icon(
    name: str,
    center=(0.0, 0.0),
    size: float = ICON_GRID,
    *,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> IconMetrics:
    draw = _MetricsDraw()
    _draw_concept_icon_raw(
        draw,
        center,
        size,
        name,
        (1.0, 1.0, 1.0, 1.0),
        mouse_width=mouse_width,
    )
    area_centroid = (
        draw.area_moment[0] / draw.filled_area,
        draw.area_moment[1] / draw.filled_area,
    )
    bounding_center, bounding_radius = minimum_enclosing_circle(draw.boundary_points)
    return IconMetrics(
        tuple(draw.bounds), draw.radial_extent, area_centroid, bounding_center, bounding_radius
    )


@lru_cache(maxsize=256)
def _icon_layout(
    name: str,
    radial_alignment: float | None = None,
    padding: float | None = None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> tuple[float, tuple[float, float]]:
    """Return the declared placement anchor and scale at the requested radial padding."""

    raw = _measure_raw_icon(name, mouse_width=mouse_width)
    default_radial, default_padding = ICON_GROUP_LAYOUT_DEFAULTS[icon_component_group(name)]
    if radial_alignment is None:
        radial_alignment = default_radial
    if padding is None:
        padding = default_padding
    if name in REVIEW_LOCKED_ICONS:
        radial_alignment = 0.0
        padding = ICON_DEFAULT_PADDING
    elif name == "tool-rotate":
        radial_alignment = 0.0
    anchor = icon_alignment_anchor(name)
    if anchor in {"arc", "hub"}:
        # Snap's lower-arc center and Scale's three-axis hub are authored at
        # the origin. Their semantic control center takes priority over a box.
        base_offset = (0.0, 0.0)
    else:
        base_offset = (-raw.center_offset[0], -raw.center_offset[1])
    radial_offset = (-raw.bounding_center[0], -raw.bounding_center[1])
    amount = min(1.0, max(0.0, float(radial_alignment)))
    offset = (
        base_offset[0] + (radial_offset[0] - base_offset[0]) * amount,
        base_offset[1] + (radial_offset[1] - base_offset[1]) * amount,
    )
    # Measure after placement so centering happens before the complete master
    # is fitted. Scaling first would preserve each source contour's old drift.
    shifted = _measure_raw_icon(name, offset, mouse_width=mouse_width)
    padding = float(padding)
    if not ICON_MIN_CLEARANCE <= padding <= ICON_MAX_PADDING:
        raise ValueError(
            f"icon padding must be between {ICON_MIN_CLEARANCE:g} and {ICON_MAX_PADDING:g}"
        )
    # Rotate's outer screen ring is itself the slot frame. Keep its visible
    # outside diameter on the orange guide while scaling all inner rings with it.
    target_padding = (
        ROTATE_FRAME_PADDING
        if name == "tool-rotate"
        else padding + ICON_PADDING_BIAS.get(name, 0.0)
    )
    safe_radius = ICON_BOUND_DIAMETER * 0.5 - target_padding
    layout_scale = safe_radius / shifted.radial_extent
    return layout_scale, (offset[0] * layout_scale, offset[1] * layout_scale)


def icon_alignment_anchor(name: str) -> str:
    """Return the reviewed placement anchor for one concept icon."""

    if name == "tool-snap":
        return "arc"
    if name == "tool-scale":
        return "hub"
    return "box"


def draw_concept_icon(
    draw,
    center,
    size: float,
    name: str,
    color,
    *,
    radial_alignment: float | None = None,
    padding: float | None = None,
    accent_color=None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> None:
    """Draw one anchor-centered candidate fitted to a padded circular slot."""

    layout_scale, offset = _icon_layout(name, radial_alignment, padding, mouse_width)
    unit_scale = float(size) / ICON_GRID
    adjusted_center = (
        float(center[0]) + offset[0] * unit_scale,
        float(center[1]) + offset[1] * unit_scale,
    )
    _draw_concept_icon_raw(
        draw,
        adjusted_center,
        size * layout_scale,
        name,
        color,
        accent_color=accent_color,
        mouse_width=mouse_width,
    )


@lru_cache(maxsize=256)
def icon_metrics(
    name: str,
    radial_alignment: float | None = None,
    padding: float | None = None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> IconMetrics:
    """Return placement and optical measurements for one 24-unit candidate."""

    draw = _MetricsDraw()
    draw_concept_icon(
        draw,
        (0.0, 0.0),
        ICON_GRID,
        name,
        (1.0, 1.0, 1.0, 1.0),
        radial_alignment=radial_alignment,
        padding=padding,
        mouse_width=mouse_width,
    )
    area_centroid = (
        draw.area_moment[0] / draw.filled_area,
        draw.area_moment[1] / draw.filled_area,
    )
    bounding_center, bounding_radius = minimum_enclosing_circle(draw.boundary_points)
    return IconMetrics(
        tuple(draw.bounds), draw.radial_extent, area_centroid, bounding_center, bounding_radius
    )


def icon_family(label: str):
    """Return one family by its user-facing label."""

    return next(icons for family, icons in ICON_FAMILIES if family == label)


def icon_component_group(name: str) -> str:
    """Return the actual UI component group that owns one candidate."""

    if name.startswith("tool-"):
        return "Viewport tools"
    if name.startswith("playback-"):
        return "Viewport playback"
    if name.startswith("transport-"):
        return "Keyframe transport"
    if name.startswith("key-"):
        return "Keyframes"
    if name.startswith("panel-"):
        return "Panels"
    if name.startswith("helper-"):
        return "Scene helpers"
    if name.startswith("status-"):
        return "Status & input"
    raise ValueError(f"unknown concept icon: {name!r}")
