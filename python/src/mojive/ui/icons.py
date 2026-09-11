"""Shared vector icons used by Mojive's runtime UI and design workbench."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from itertools import pairwise

from ..curves2d import (
    CORNER_SMOOTHING,
    arrow_mesh,
    box_handle_points,
    polyline_ribbon,
    smooth_line_cap,
    smooth_polygon_corners,
    smooth_rect_points,
)
from ..draglink2d import smooth_union
from ..gizmo import DIMENSION_CORNER_RADIUS_RATIO
from .viewport_widgets import (
    CAPSULE_SMOOTHING,
    OVERLAY_GEOMETRY,
    TOOL_GLYPH_SCALE,
    _rotate_visible_ring_polygons,
    _snap_glyph_shape,
    mouse_button_geometry,
    mouse_wheel_geometry,
)

ICON_GRID = 24.0
ICON_STROKE = 1.75
ICON_MIN_STROKE = 1.0
ICON_MAX_STROKE = 2.5
# Keep ordinary non-tool families two grid units inside their circular slot.
# Viewport tools override this below so their glyphs remain close to the frame.
ICON_BOUND_DIAMETER = ICON_GRID
ICON_MIN_CLEARANCE = 0.5
ICON_DEFAULT_PADDING = 2.0
ICON_MAX_PADDING = 4.0
ROTATE_FRAME_PADDING = 0.0
ROTATE_FRAME_STROKE_OVERSHOOT = ICON_STROKE * 0.5
ROTATE_FRINGE_MAX = 1.0
ROTATE_FRINGE_GAP_FRACTION = 0.5
ICON_ROTATE_RING_GAP_RATIO = 0.8
ICON_ROTATE_RING_CAP = "round"
MORE_ARM_RATIO = 0.94
STATUS_MOUSE_DEFAULT_WIDTH = OVERLAY_GEOMETRY.hint_mouse_width
REVIEW_LOCKED_ICONS = frozenset(("status-info", "status-warning", "status-error"))
REVIEW_LOCKED_PADDING = 0.75
STROKE_SCALE_LOCKED_ICONS = REVIEW_LOCKED_ICONS | frozenset(
    ("status-mouse-left", "status-mouse-right", "status-mouse-wheel")
)
BOX_CENTERED_ICONS = frozenset(
    (
        "tool-snap",
        "playback-previous",
        "playback-next",
        "playback-more",
        "transport-first",
        "transport-previous",
        "transport-next",
        "transport-last",
        "transport-more",
    )
)
RING_CENTERED_ICONS = frozenset(("playback-reset", "transport-reset"))
ICON_ALIGNMENT_CHOICES = ("circle", "box")
ICON_ALIGNMENT_EDITABLE_ICONS = frozenset(("key-snapshot", "helper-camera", "helper-light"))
ICON_GLYPH_ALIGNMENT_DEFAULTS = dict.fromkeys(ICON_ALIGNMENT_EDITABLE_ICONS, "box")
RESET_RING_CENTER = (0.0, 0.47)
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

# Every group owns circular padding so component sizing can be tuned independently.
# Placement follows the explicit geometric anchor declared below.
ICON_GROUP_LAYOUT_DEFAULTS = {
    "Viewport tools": 0.5,
    "Viewport playback": ICON_DEFAULT_PADDING,
    "Keyframe transport": ICON_DEFAULT_PADDING,
    "Keyframes": ICON_DEFAULT_PADDING,
    "Panels": ICON_DEFAULT_PADDING,
    "Scene helpers": ICON_DEFAULT_PADDING,
    "Status & input": ICON_DEFAULT_PADDING,
}
ICON_GLYPH_PADDING_DEFAULTS = {
    "tool-rotate": ROTATE_FRAME_PADDING,
    "helper-camera": 0.5,
    "helper-light": 0.5,
    "playback-previous": 4.0,
    "playback-next": 4.0,
    "playback-more": 4.0,
    "transport-first": 4.0,
    "transport-previous": 4.0,
    "transport-play": 3.0,
    "transport-next": 4.0,
    "transport-more": 4.0,
    "key-keyframe": 4.0,
    "key-add": 4.0,
    "key-clear": 4.0,
    "key-previous": 4.0,
    "key-next": 4.0,
    "key-snapshot": 0.5,
}
ICON_GROUP_STROKE_DEFAULTS = dict.fromkeys(ICON_GROUP_LAYOUT_DEFAULTS, ICON_STROKE)
ICON_GLYPH_STROKE_DEFAULTS = {
    "tool-move": 1.2,
    "tool-rotate": 1.2,
    "tool-scale": 1.2,
    "tool-world": 1.25,
    "tool-body": 1.25,
    "tool-snap": 1.0,
    "playback-previous": 2.0,
    "playback-next": 2.0,
    "playback-reset": 1.25,
    "playback-more": 2.0,
    "transport-previous": 2.0,
    "transport-next": 2.0,
    "transport-reset": 1.25,
    "transport-more": 2.0,
    "key-snapshot": 1.5,
    "key-keyframe": 1.75,
    "key-previous": 1.5,
    "key-next": 1.5,
    "key-fit": 1.25,
    "key-follow": 1.5,
    "key-view": 1.5,
    "panel-search": 1.25,
    "panel-sort": 1.25,
    "panel-clear": 1.5,
    "panel-visible": 1.25,
    "panel-hidden": 1.25,
    "helper-camera": 1.5,
    "helper-light": 1.5,
}


@dataclass(frozen=True)
class IconTuning:
    """Authored shape controls that are independent of fit and stroke."""

    move_head_scale: float = 1.0
    scale_handle_scale: float = 1.0
    key_fit_arm_length: float = 4.0


ICON_TUNING_DEFAULTS = IconTuning()


@dataclass(frozen=True)
class IconStyle:
    """Frozen production inputs for one reviewed icon."""

    padding: float
    stroke_width: float
    alignment: str | None
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP
    tuning: IconTuning = ICON_TUNING_DEFAULTS


# Closely related marks share one fitted master so their authored dimensions
# remain comparable after placement. More is a shorter, rotated Previous at the
# same stroke scale. Record shares Stop's fitted scale and carries its requested
# two-percent reduction in the authored circle. All mouse states reuse the Left
# shell scale because their production outer rectangle has one fixed size.
ICON_LAYOUT_REFERENCES = {
    "playback-more": "playback-previous",
    "playback-record": "playback-stop",
    "transport-more": "transport-previous",
    "transport-record": "transport-stop",
    "status-mouse-right": "status-mouse-left",
    "status-mouse-wheel": "status-mouse-left",
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


class _Painter:
    def __init__(
        self,
        draw,
        center,
        size: float,
        color,
        *,
        stroke_width: float = ICON_STROKE,
        stroke_compensation: float = 1.0,
        rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
        rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
        tuning: IconTuning = ICON_TUNING_DEFAULTS,
    ) -> None:
        self.draw = draw
        self.cx, self.cy = (float(center[0]), float(center[1]))
        self.scale = float(size) / ICON_GRID
        self.stroke_width = float(stroke_width)
        self.stroke_compensation = float(stroke_compensation)
        self.rotate_ring_gap_ratio = float(rotate_ring_gap_ratio)
        self.rotate_ring_cap = str(rotate_ring_cap)
        self.tuning = tuning
        self.color = color
        self.stroke = self.stroke_width * self.stroke_compensation * self.scale

    def resolved_width(self, width: float | None) -> float:
        return self.stroke_width if width is None else float(width)

    def point(self, x: float, y: float) -> tuple[float, float]:
        return self.cx + x * self.scale, self.cy + y * self.scale

    def points(self, values) -> tuple[tuple[float, float], ...]:
        return tuple(self.point(x, y) for x, y in values)

    def line(self, a, b, *, width: float | None = None) -> None:
        self.draw.line(
            self.point(*a),
            self.point(*b),
            self.color,
            self.resolved_width(width) * self.stroke_compensation * self.scale,
            cap="round",
        )

    def arrow(
        self,
        a,
        b,
        *,
        width: float | None = None,
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
            self.resolved_width(width) * self.stroke_compensation * self.scale,
            head_length=head_length * self.scale,
            head_width=head_width * self.scale,
            corner_radius=corner_radius * self.scale,
            join_radius=corner_radius * 0.55 * self.scale,
            round_tail=round_tail,
        )

    def polyline(
        self,
        points,
        *,
        closed: bool = False,
        width: float | None = None,
        cap: str | None = None,
    ) -> None:
        path = self.points(points)
        self.draw.polyline(
            path,
            self.color,
            self.resolved_width(width) * self.stroke_compensation * self.scale,
            closed=closed,
            cap=cap or ("butt" if closed else "round"),
        )

    def convex_polygon(self, points) -> None:
        self.draw.convex_fill(self.points(points), self.color)

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
        width: float | None = None,
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

    def circle(self, x: float, y: float, radius: float, *, width: float | None = None) -> None:
        self.draw.circle(
            self.point(x, y),
            radius * self.scale,
            self.color,
            self.resolved_width(width) * self.stroke_compensation * self.scale,
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
        width: float | None = None,
        smoothing: float | None = None,
    ) -> None:
        self.draw.rect(
            self.point(x0, y0),
            self.point(x1, y1),
            self.color,
            self.resolved_width(width) * self.stroke_compensation * self.scale,
            rounding=rounding * self.scale,
            smoothing=smoothing,
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
        width: float | None = None,
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


def _triangle_source(
    direction: float, center_x: float = 0.0, scale: float = 1.0
) -> tuple[tuple[float, float], ...]:
    """Return an equilateral triangle centered on its geometric centroid."""

    half_side = 6.8 * scale
    centroid_to_rear = half_side / math.sqrt(3.0)
    return (
        (center_x - direction * centroid_to_rear, -half_side),
        (center_x + direction * 2.0 * centroid_to_rear, 0.0),
        (center_x - direction * centroid_to_rear, half_side),
    )


def _triangle_path(
    direction: float, center_x: float = 0.0, scale: float = 1.0
) -> tuple[tuple[float, float], ...]:
    """Return the equilateral triangle's final G3 contour."""

    path = smooth_polygon_corners(
        _triangle_source(direction, center_x, scale),
        1.12 * scale,
        tuple(range(3)),
        smoothing=CORNER_SMOOTHING,
    )
    visible_center_x = (float(path[:, 0].min()) + float(path[:, 0].max())) * 0.5
    path[:, 0] += center_x - visible_center_x
    return tuple(map(tuple, path.tolist()))


def _triangle(p: _Painter, direction: float, *, center_x: float = 0.0, scale: float = 1.0) -> None:
    p.polygon(_triangle_path(direction, center_x, scale))


def _rounded_polyline(
    p: _Painter,
    points,
    *,
    width: float | None = None,
    radius: float = 0.68,
) -> None:
    """Fill one open centerline as a joined G3 contour."""

    compensated_width = p.resolved_width(width) * p.stroke_compensation
    _left, _right, outline = polyline_ribbon(tuple(points), compensated_width)
    p.smooth_polygon(
        outline,
        radius * p.stroke_compensation,
        convex_only=False,
    )


def _g3_rect(p: _Painter, x0: float, y0: float, x1: float, y1: float, radius: float) -> None:
    p.polygon(smooth_rect_points(x0, y0, x1, y1, radius, smoothing=CORNER_SMOOTHING))


def _chevron_centerline(
    direction: float, x: float, scale: float = 1.0
) -> tuple[tuple[float, float], ...]:
    """Return the original right-angle transport chevron centerline."""

    half_height = 4.4 * scale
    run = half_height
    return (
        (x - direction * run * 0.5, -half_height),
        (x + direction * run * 0.5, 0.0),
        (x - direction * run * 0.5, half_height),
    )


def _chevron(p: _Painter, direction: float, x: float, *, scale: float = 1.0) -> None:
    _rounded_polyline(
        p,
        _chevron_centerline(direction, x, scale),
        radius=0.68,
    )


def _more_centerline(scale: float = 1.0) -> tuple[tuple[float, float], ...]:
    """Rotate a slightly shorter Previous centerline 90 degrees counterclockwise."""

    previous = _chevron_centerline(-1.0, 0.0, scale * MORE_ARM_RATIO)
    return tuple((y, -x) for x, y in previous)


def _arc_arrow(
    p: _Painter,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    *,
    center=(0.0, 0.0),
    stroke: float | None = None,
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
    half_width = p.resolved_width(stroke) * p.stroke_compensation * 0.5
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


@lru_cache(maxsize=512)
def _concept_rotate_ring_polygons(
    stroke_width: float,
    gap_ratio: float,
    cap: str,
):
    return _rotate_visible_ring_polygons(
        stroke_width,
        gap_ratio,
        cap,
        CAPSULE_SMOOTHING,
    )


def _draw_tool(p: _Painter, name: str) -> None:
    if name == "tool-move":
        # Restore the original compact arrowhead proportions. The head control
        # scales that accepted shape around each tip without moving its reach.
        tip = 8.7
        head_scale = p.tuning.move_head_scale
        base = tip - (tip - 5.6) * head_scale
        wing = 2.65 * head_scale
        shaft = p.stroke_width * p.stroke_compensation * 0.5
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
            0.42 * p.stroke_compensation,
            convex_only=False,
        )
    elif name == "tool-rotate":
        # Repeat the established production construction: the screen ring uses
        # ImGui's antialiased circle stroke. The three local half-rings use a
        # gap-limited fringe: two opposing AA ramps may meet at the center of a
        # knockout but must never overlap and fill it at compact sizes.
        production_scale = 0.82
        glyph_scale = production_scale * TOOL_GLYPH_SCALE
        p.circle(0.0, 0.0, 10.0 * glyph_scale)
        rendered_gap = p.stroke * p.rotate_ring_gap_ratio
        fringe_width = min(
            ROTATE_FRINGE_MAX,
            rendered_gap * ROTATE_FRINGE_GAP_FRACTION,
        )
        for ring in _concept_rotate_ring_polygons(
            p.stroke_width * p.stroke_compensation / production_scale,
            p.rotate_ring_gap_ratio,
            p.rotate_ring_cap,
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
        # Its DrawList contour narrows the supplied source stroke by one unit;
        # choose that source from the shared final weight and retain the
        # production transparent center-shell distance.
        source_core_width = p.stroke_width * p.stroke_compensation
        source_tool_stroke = source_core_width + 1.0
        scale_geometry = replace(
            OVERLAY_GEOMETRY,
            tool_stroke=source_tool_stroke,
            frame_center_gap_ratio=(
                OVERLAY_GEOMETRY.tool_stroke
                * OVERLAY_GEOMETRY.frame_center_gap_ratio
                * p.stroke_compensation
                / source_tool_stroke
            ),
        )
        glyph_scale = TOOL_GLYPH_SCALE
        handle_half = 1.5 * p.tuning.scale_handle_scale
        reach = math.sqrt((9.0 * glyph_scale) ** 2 - handle_half**2) - handle_half
        clear_radius = (
            scale_geometry.frame_center_radius * glyph_scale
            + scale_geometry.tool_stroke * scale_geometry.frame_center_gap_ratio
        )
        paths = tuple(
            box_handle_points(
                (ux * clear_radius, uy * clear_radius),
                (ux * reach, uy * reach),
                max(source_tool_stroke * 0.45, source_tool_stroke - 1.0),
                2.0 * handle_half,
                corner_radius=(2.0 * handle_half * DIMENSION_CORNER_RADIUS_RATIO),
                smoothing=CAPSULE_SMOOTHING,
            )
            for ux, uy in ((0.0, -1.0), (0.866025, 0.5), (-0.866025, 0.5))
        )
        for path in paths:
            p.polygon(path)
        p.circle_filled(0.0, 0.0, scale_geometry.frame_center_radius * glyph_scale)
    elif name == "tool-world":
        # Sparse stroked symbols need a larger authored envelope than solid
        # tools to carry comparable visual weight in the same 24-unit slot.
        source_width = p.stroke_width * p.stroke_compensation
        # Centerline endpoints account for both strokes. The round caps meet
        # the globe's inner edge. Paint the outer ring last so antialiasing at
        # the junction belongs to the frame rather than the internal strokes.
        inner_reach = 8.75 - source_width
        p.line((-inner_reach, 0.0), (inner_reach, 0.0))
        ellipse = tuple(
            (
                3.53 * math.cos(index * math.tau / 32),
                inner_reach * math.sin(index * math.tau / 32),
            )
            for index in range(32)
        )
        p.polyline(ellipse, closed=True)
        p.circle(0.0, 0.0, 8.75)
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
        source_width = p.stroke_width * p.stroke_compensation
        inset = source_width * 1.12
        for endpoint in (left, right, bottom):
            length = math.hypot(endpoint[0], endpoint[1])
            end = (
                endpoint[0] * (length - inset) / length,
                endpoint[1] * (length - inset) / length,
            )
            p.line(junction, end)
        p.circle_filled(*junction, p.stroke_width * p.stroke_compensation * 0.5)
        p.smooth_outline(
            (top, right, lower_right, bottom, lower_left, left),
            0.5,
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
        p.polyline(path)
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
        triangle = _triangle_path(direction, center_x=-direction * 1.27, scale=0.72)
        p.polygon(triangle)
        x = direction * 5.13
        triangle_min_y = min(point[1] for point in triangle)
        triangle_max_y = max(point[1] for point in triangle)
        _g3_rect(p, x - 0.75, triangle_min_y, x + 0.75, triangle_max_y, 0.58)
    elif kind == "reset":
        _arc_arrow(p, 6.8, -52.0, 255.0, center=(0.0, 0.47))
    elif kind == "record":
        p.circle_filled(0.0, 0.0, 4.8 * 0.98)
    elif kind == "stop":
        _g3_rect(p, -4.8, -4.8, 4.8, 4.8, 1.05)
    elif kind == "more":
        _rounded_polyline(
            p,
            _more_centerline(1.18),
            radius=0.68,
        )
    else:
        raise ValueError(f"unknown transport icon: {name!r}")


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
        p.line((-6.6, 0.0), (6.6, 0.0))
        p.line((0.0, -6.6), (0.0, 6.6))
    elif kind == "clear":
        p.line((-5.8, -5.8), (5.8, 5.8))
        p.line((5.8, -5.8), (-5.8, 5.8))
    elif kind in {"previous", "next"}:
        direction = -1.0 if kind == "previous" else 1.0
        _chevron(p, direction, direction * 4.4, scale=0.68)
        _diamond(p, center=(-direction * 3.2, 0.0), radius=3.5, filled=True)
    elif kind == "fit":
        arm_length = p.tuning.key_fit_arm_length
        inner = 6.5 - arm_length
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            _rounded_polyline(
                p,
                (
                    (sx * inner, sy * 6.5),
                    (sx * 6.5, sy * 6.5),
                    (sx * 6.5, sy * inner),
                ),
            )
    elif kind == "follow":
        p.arrow(
            (-7.38, 0.0),
            (4.42, 0.0),
            head_length=3.2,
            head_width=5.0,
            round_tail=True,
        )
        p.line((6.62, -5.8), (6.62, 5.8))
    else:
        p.rect(-7.5, -5.1, 7.5, 5.1, rounding=1.8)
        p.circle_filled(0.0, 0.0, 1.9)


def _eye_points() -> tuple[tuple[float, float], ...]:
    top = tuple((-8.2 + index * 2.05, -4.8 * math.sin(math.pi * index / 8)) for index in range(9))
    bottom = tuple((8.2 - index * 2.05, 4.8 * math.sin(math.pi * index / 8)) for index in range(9))
    return (*top, *bottom[1:-1])


@lru_cache(maxsize=64)
def _lashed_lid_outline(width: float) -> tuple[tuple[float, float], ...]:
    """Return the closed-eye lid and lashes as one non-overdrawn silhouette."""

    radius_x = 8.0
    lid_height = 3.2
    offset_y = -2.85
    lash_offsets = (-0.52, 0.0, 0.52)
    lash_xs = tuple(radius_x * offset for offset in lash_offsets)
    xs = tuple(
        sorted({-radius_x + radius_x * 2.0 * index / 16.0 for index in range(17)} | set(lash_xs))
    )
    lid = tuple(
        (x, offset_y + math.sin(math.pi * (x + radius_x) / (2.0 * radius_x)) * lid_height)
        for x in xs
    )
    lower, upper, _outline = polyline_ribbon(lid, float(width))
    lower, upper = list(lower), list(upper)

    def normalized(a, b) -> tuple[float, float]:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    end_direction = normalized(lid[-2], lid[-1])
    end_cap = smooth_line_cap(
        lid[-1],
        end_direction,
        width,
        max_inset=0.45 * math.dist(lid[-2], lid[-1]),
        smoothing=CORNER_SMOOTHING,
    )
    lower[-1], upper[-1] = tuple(end_cap[0]), tuple(end_cap[-1])
    start_direction = normalized(lid[1], lid[0])
    start_cap = smooth_line_cap(
        lid[0],
        start_direction,
        width,
        max_inset=0.45 * math.dist(lid[0], lid[1]),
        smoothing=CORNER_SMOOTHING,
    )
    upper[0], lower[0] = tuple(start_cap[0]), tuple(start_cap[-1])

    half_width = float(width) * 0.5

    def lower_at_x(x: float) -> tuple[float, float]:
        for start, end in pairwise(lower):
            if start[0] <= x <= end[0]:
                amount = (x - start[0]) / (end[0] - start[0])
                return x, start[1] + amount * (end[1] - start[1])
        raise ValueError("lash root lies outside the lid")

    lashes = []
    for lash_x, offset in zip(lash_xs, lash_offsets, strict=True):
        direction = normalized((0.0, 0.0), (offset * 1.95, 2.65))
        normal = (-direction[1], direction[0])
        root_half_width = abs(normal[0]) * half_width
        root_lo, root_hi = lash_x - root_half_width, lash_x + root_half_width
        start, finish = lower_at_x(root_lo), lower_at_x(root_hi)
        lid_y = offset_y + math.sin(math.pi * (lash_x + radius_x) / (2.0 * radius_x)) * lid_height
        tip = (lash_x + offset * 1.95, lid_y + 2.65)
        cap = tuple(
            (
                tip[0]
                + normal[0] * half_width * math.cos(angle)
                + direction[0] * half_width * math.sin(angle),
                tip[1]
                + normal[1] * half_width * math.cos(angle)
                + direction[1] * half_width * math.sin(angle),
            )
            for angle in (math.pi * step / 8.0 for step in range(9))
        )
        lashes.append((root_lo, root_hi, (start, *cap, finish)))

    integrated_lower = []
    lower_index = 0
    for root_lo, root_hi, lash in lashes:
        while lower_index < len(lower) and lower[lower_index][0] < root_lo:
            integrated_lower.append(lower[lower_index])
            lower_index += 1
        integrated_lower.extend(lash)
        while lower_index < len(lower) and lower[lower_index][0] <= root_hi:
            lower_index += 1
    integrated_lower.extend(lower[lower_index:])

    return (
        *integrated_lower,
        *map(tuple, end_cap[1:-1]),
        *reversed(upper),
        *map(tuple, start_cap[1:-1]),
    )


@lru_cache(maxsize=32)
def _search_icon_mesh(stroke: float = ICON_STROKE):
    """Return one hollow lens and handle joined by a G3 smooth union."""

    outer_radius = 5.55
    inner_radius = outer_radius - stroke
    handle_half_width = stroke * 0.5
    handle_start = 3.65
    handle_end = 10.775
    handle_tip = handle_end + handle_half_width
    blend = stroke * 0.7

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
        vertices, indices, outline, hole = _search_icon_mesh(p.stroke_width * p.stroke_compensation)
        p.indexed_fill(vertices, indices, outline=outline, hole=hole)
    elif kind == "sort":
        for y, end in ((-5.09, 2.24), (0.11, 0.04), (5.31, -2.16)):
            p.line((-6.76, y), (end, y))
        # The round tail aligns with the top bar centerline; the head tip aligns
        # with the visible lower edge of the bottom bar.
        p.arrow(
            (5.54, -5.09),
            (5.54, 5.31 + p.stroke_width * p.stroke_compensation * 0.5),
            head_length=3.0,
            head_width=4.6,
            round_tail=True,
        )
    elif kind == "clear":
        p.line((-5.8, -5.8), (5.8, 5.8))
        p.line((5.8, -5.8), (-5.8, 5.8))
    elif kind == "visible":
        p.polyline(_eye_points(), closed=True)
        p.circle_filled(0.0, 0.0, 2.05)
    elif kind == "hidden":
        # The lid and lashes intersect, so submit their combined silhouette
        # once. Separate translucent strokes accumulate alpha at every root.
        width = p.stroke_width * p.stroke_compensation
        p.polygon(_lashed_lid_outline(width))
    elif kind in {"perspective", "orthographic"}:
        near = 3.0 if kind == "perspective" else 6.1
        p.smooth_outline(
            ((-6.8, -near), (6.8, -6.1), (6.8, 6.1), (-6.8, near)),
            0.52,
        )
    elif kind == "right":
        _triangle(p, 1.0)
    else:
        p.polygon(tuple((-y, x) for x, y in _triangle_path(1.0)))


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
    # Scene helpers use a translucent neutral color. Keep the base separator
    # clear of the shell so their independent strokes never accumulate alpha.
    source_stroke = p.stroke_width * p.stroke_compensation
    base_half_width = max(0.35, 2.5 - source_stroke - 0.15)
    p.line((-base_half_width, 5.65), (base_half_width, 5.65))
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
    width = float(mouse_width)
    half_width = width * 0.5
    if not 5.0 <= mouse_width <= 24.0:
        raise ValueError("status mouse width must be between 5 and 24 grid units")
    height = OVERLAY_GEOMETRY.hint_control_height
    top = -height * 0.5
    outline_width = OVERLAY_GEOMETRY.hint_mouse_stroke
    button = kind.removeprefix("mouse-")
    button_geometry = mouse_button_geometry(
        -half_width,
        top,
        width,
        height,
        button,
        outline_width=outline_width,
        geometry=OVERLAY_GEOMETRY,
        smoothing=CAPSULE_SMOOTHING,
    )
    if button_geometry is None:
        radius = min(width * 0.22, height * 0.18)
        p.rect(
            -half_width,
            top,
            half_width,
            -top,
            rounding=radius,
            width=outline_width,
            smoothing=CAPSULE_SMOOTHING,
        )
    else:
        p.polyline(button_geometry.visible_shell, width=outline_width, cap="butt")
        accent.convex_polygon(button_geometry.fill)
    if button == "wheel":
        wheel = mouse_wheel_geometry(
            -half_width,
            top,
            width,
            height,
            outline_width=outline_width,
            pixel_size=1.0,
            geometry=OVERLAY_GEOMETRY,
        )
        accent.rect_filled(
            wheel.lo[0],
            wheel.lo[1],
            wheel.hi[0],
            wheel.hi[1],
            rounding=wheel.rounding,
        )


def _draw_concept_icon_raw(
    draw,
    center,
    size: float,
    name: str,
    color,
    *,
    accent_color=None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    stroke_compensation: float = 1.0,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
) -> None:
    """Draw authored geometry before shared placement is applied."""

    painter = _Painter(
        draw,
        center,
        size,
        color,
        stroke_width=stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
    )
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
    _draw_concept_icon_raw(
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
    # Rotate's outer screen-ring centerline is itself the slot frame. Its stroke
    # straddles the orange guide exactly as the production Tool Column painter
    # does, so the visible edge extends by half of the canonical stroke.
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
        stroke_width=stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
    )


@cache
def production_icon_style(name: str) -> IconStyle:
    """Resolve one immutable runtime style without per-frame dictionary work."""

    group = icon_component_group(name)
    return IconStyle(
        padding=ICON_GLYPH_PADDING_DEFAULTS.get(name, ICON_GROUP_LAYOUT_DEFAULTS[group]),
        stroke_width=ICON_GLYPH_STROKE_DEFAULTS.get(name, ICON_GROUP_STROKE_DEFAULTS[group]),
        alignment=ICON_GLYPH_ALIGNMENT_DEFAULTS.get(name),
    )


@cache
def _production_icon_layout(
    name: str,
) -> tuple[IconStyle, tuple[float, tuple[float, float], float]]:
    """Cache all measurement and centering work for the frozen runtime style."""

    style = production_icon_style(name)
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


def draw_icon(draw, center, size: float, name: str, color, *, accent_color=None) -> None:
    """Draw a production icon using its reviewed geometry and cached placement."""

    style, (layout_scale, offset, stroke_compensation) = _production_icon_layout(name)
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
        mouse_width=style.mouse_width,
        stroke_width=style.stroke_width,
        stroke_compensation=stroke_compensation,
        rotate_ring_gap_ratio=style.rotate_ring_gap_ratio,
        rotate_ring_cap=style.rotate_ring_cap,
        tuning=style.tuning,
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

    draw = _MetricsDraw()
    draw_concept_icon(
        draw,
        (0.0, 0.0),
        ICON_GRID,
        name,
        (1.0, 1.0, 1.0, 1.0),
        padding=padding,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
        alignment=alignment,
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
    """Return cached visible metrics for one frozen production icon."""

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
