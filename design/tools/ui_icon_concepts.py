"""Concept-only Mojive icon families for the UI feasibility workbench.

These painters deliberately do not feed production UI.  They provide a shared
24-unit comparison grid so candidate families can be reviewed together before
any production glyph is replaced.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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
    _rotate_visible_ring_polygons,
    _snap_glyph_shape,
)

ICON_GRID = 24.0
ICON_STROKE = 1.5
# Match the complete circular placement boundary used by Diagnostics. Glyph ink
# stays inside this guide and keeps its own optical padding within the circle.
ICON_BOUND_DIAMETER = ICON_GRID
ICON_MIN_CLEARANCE = 1.5

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
        "Transport",
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

ICON_LIBRARY_TABS = ("Overview", *(label for label, _icons in ICON_FAMILIES))
ICON_GROUP_BY_SLUG = {
    label.casefold().replace(" ", "-").replace("&", "and"): label for label in ICON_LIBRARY_TABS
}


@dataclass(frozen=True)
class IconMetrics:
    bounds: tuple[float, float, float, float]
    radial_extent: float
    ink_center: tuple[float, float]
    bounding_center: tuple[float, float]
    bounding_radius: float

    @property
    def center_offset(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5

    @property
    def radial_clearance(self) -> float:
        return ICON_BOUND_DIAMETER * 0.5 - self.radial_extent


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

    def polygon(self, points) -> None:
        self.draw.fringed_concave_fill(self.points(points), self.color)

    def indexed_fill(self, points, indices, *, outline, hole=()) -> None:
        self.draw.indexed_fill(
            self.points(points),
            indices,
            self.color,
            outline=self.points(outline),
            hole=self.points(hole),
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
        p.circle(
            0.0,
            0.0,
            10.0 * glyph_scale,
            width=OVERLAY_GEOMETRY.tool_stroke * production_scale,
        )
        for ring in _rotate_visible_ring_polygons(
            OVERLAY_GEOMETRY.tool_stroke,
            OVERLAY_GEOMETRY.rotate_ring_gap_ratio,
            OVERLAY_GEOMETRY.rotate_ring_cap,
            CAPSULE_SMOOTHING,
        ):
            for local in ring:
                p.polygon(tuple((x * glyph_scale, y * glyph_scale) for x, y in local))
    elif name == "tool-scale":
        # Three equal axes use joined G3 endpoint blocks. Starting each shaft
        # outside the center dot leaves the reviewed circular transparent shell.
        dot_radius = 1.2
        clear_radius = 2.85
        reach = 6.45
        for direction in ((0.0, -1.0), (0.866025, 0.5), (-0.866025, 0.5)):
            p.box_handle(
                (direction[0] * clear_radius, direction[1] * clear_radius),
                (direction[0] * reach, direction[1] * reach),
                width=1.4,
                head_size=2.4,
                corner_radius=0.45,
            )
        p.circle_filled(0.0, 0.0, dot_radius)
    elif name == "tool-world":
        p.circle(0.0, 0.0, 7.8, width=1.45)
        p.line((-7.65, 0.0), (7.65, 0.0), width=1.35)
        ellipse = tuple(
            (3.15 * math.cos(index * math.tau / 32), 7.65 * math.sin(index * math.tau / 32))
            for index in range(32)
        )
        p.polyline(ellipse, closed=True, width=1.35)
    elif name == "tool-body":
        top = (0.0, -7.7)
        left = (-6.7, -3.85)
        right = (6.7, -3.85)
        bottom = (0.0, 7.7)
        lower_left = (-6.7, 3.85)
        lower_right = (6.7, 3.85)
        # A cube has three visible faces: the center joins the two rear side
        # corners and the lower vertex. The former top-to-center edge invented
        # a fourth face. Inset the spoke ends below the outer stroke and paint
        # the G3 shell last so no round cap protrudes through a cube vertex.
        junction = (0.0, 0.0)
        inset = 0.72
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
    kind = name.removeprefix("transport-")
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
    xs = tuple(
        -outer_radius + (handle_tip + outer_radius) * index / (count - 1) for index in range(count)
    )
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
    hole = tuple(
        (
            inner_radius * math.cos(index * math.tau / 96),
            inner_radius * math.sin(index * math.tau / 96),
        )
        for index in range(96)
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


def _draw_status(p: _Painter, name: str) -> None:
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

    p.rect(-5.2, -7.8, 5.2, 7.8, rounding=4.4, width=1.45)
    p.line((-4.9, -1.1), (4.9, -1.1), width=1.2)
    if kind == "mouse-left":
        p.circle_filled(-2.2, -4.2, 1.25)
    elif kind == "mouse-right":
        p.circle_filled(2.2, -4.2, 1.25)
    else:
        p.rect_filled(-0.9, -5.8, 0.9, -2.5, rounding=0.9)


def _draw_concept_icon_raw(draw, center, size: float, name: str, color) -> None:
    """Draw authored geometry before shared optical placement is applied."""

    painter = _Painter(draw, center, size, color)
    if name.startswith("tool-"):
        _draw_tool(painter, name)
    elif name.startswith("transport-"):
        _draw_transport(painter, name)
    elif name.startswith("key-"):
        _draw_keyframe(painter, name)
    elif name.startswith("panel-"):
        _draw_panel(painter, name)
    elif name.startswith("helper-"):
        _draw_helper(painter, name)
    elif name.startswith("status-"):
        _draw_status(painter, name)
    else:
        raise ValueError(f"unknown concept icon: {name!r}")


class _MetricsDraw:
    """Measure bounds and approximate ink mass from the authored primitives."""

    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.radial_extent = 0.0
        self.boundary_points: list[tuple[float, float]] = []
        self.ink_area = 0.0
        self.ink_moment = [0.0, 0.0]

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
        self.ink_area += area
        self.ink_moment[0] += area * float(center[0])
        self.ink_moment[1] += area * float(center[1])

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


def _measure_raw_icon(name: str, center=(0.0, 0.0), size: float = ICON_GRID) -> IconMetrics:
    draw = _MetricsDraw()
    _draw_concept_icon_raw(draw, center, size, name, (1.0, 1.0, 1.0, 1.0))
    ink_center = (
        draw.ink_moment[0] / draw.ink_area,
        draw.ink_moment[1] / draw.ink_area,
    )
    bounding_center, bounding_radius = minimum_enclosing_circle(draw.boundary_points)
    return IconMetrics(
        tuple(draw.bounds), draw.radial_extent, ink_center, bounding_center, bounding_radius
    )


@lru_cache(maxsize=64)
def _icon_layout(name: str) -> tuple[float, tuple[float, float]]:
    """Return the declared visual anchor and scale inside the placement bound."""

    raw = _measure_raw_icon(name)
    anchor = icon_alignment_anchor(name)
    if anchor in {"hub", "arc"}:
        # Scale's hub and Snap's lower-arc center are authored at the origin.
        offset = (0.0, 0.0)
    elif anchor == "sphere":
        # Compound Keyframe and nondirectional Panel marks use the actual
        # minimum bounding circle instead of an asymmetric box or ink mass.
        offset = (-raw.bounding_center[0], -raw.bounding_center[1])
    elif anchor == "ink":
        # Optically unbalanced helpers read from their ink mass. Their
        # asymmetric boxes are therefore diagnostic.
        offset = (-raw.ink_center[0], -raw.ink_center[1])
    else:
        offset = (-raw.center_offset[0], -raw.center_offset[1])
    shifted = _measure_raw_icon(name, offset)
    safe_radius = ICON_BOUND_DIAMETER * 0.5 - ICON_MIN_CLEARANCE
    layout_scale = min(1.0, safe_radius / shifted.radial_extent)
    return layout_scale, (offset[0] * layout_scale, offset[1] * layout_scale)


def icon_alignment_anchor(name: str) -> str:
    """Return the reviewed placement anchor for one concept icon."""

    if name == "tool-scale":
        return "hub"
    if name == "tool-snap":
        return "arc"
    if name.startswith("key-") or (
        name.startswith("panel-") and name not in {"panel-right", "panel-down"}
    ):
        return "sphere"
    if name == "helper-camera":
        return "ink"
    return "box"


def draw_concept_icon(draw, center, size: float, name: str, color) -> None:
    """Draw one anchor-centered candidate from a proportional 24-unit master."""

    layout_scale, offset = _icon_layout(name)
    unit_scale = float(size) / ICON_GRID
    adjusted_center = (
        float(center[0]) + offset[0] * unit_scale,
        float(center[1]) + offset[1] * unit_scale,
    )
    _draw_concept_icon_raw(draw, adjusted_center, size * layout_scale, name, color)


@lru_cache(maxsize=64)
def icon_metrics(name: str) -> IconMetrics:
    """Return placement and optical measurements for one 24-unit candidate."""

    draw = _MetricsDraw()
    draw_concept_icon(draw, (0.0, 0.0), ICON_GRID, name, (1.0, 1.0, 1.0, 1.0))
    ink_center = (
        draw.ink_moment[0] / draw.ink_area,
        draw.ink_moment[1] / draw.ink_area,
    )
    bounding_center, bounding_radius = minimum_enclosing_circle(draw.boundary_points)
    return IconMetrics(
        tuple(draw.bounds), draw.radial_extent, ink_center, bounding_center, bounding_radius
    )


def icon_family(label: str):
    """Return one family by its user-facing label."""

    return next(icons for family, icons in ICON_FAMILIES if family == label)
