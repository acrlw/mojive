"""Scale authored icon primitives through one drawing boundary."""

from __future__ import annotations

import math

from mojive.geometry2d.curves import (
    CORNER_SMOOTHING,
    box_handle_points,
    polyline_ribbon,
    smooth_polygon_corners,
    smooth_rect_points,
)
from mojive.geometry2d.polygons import signed_polygon_area as _signed_polygon_area
from mojive.geometry2d.polygons import simple_polygon_indices as _simple_polygon_indices

from .model import (
    ICON_GRID,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_STROKE,
    ICON_TUNING_DEFAULTS,
    MORE_ARM_RATIO,
    IconTuning,
)


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
    head_scale = p.tuning.reset_head_scale
    head_length *= head_scale
    head_half_width *= head_scale
    base_center = (
        end[0] - tangent[0] * 0.8 * head_scale,
        end[1] - tangent[1] * 0.8 * head_scale,
    )
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
        0.42 * head_scale,
        corners=(head_start, head_start + 1, head_start + 2),
        convex_only=False,
    )
    start_angle = angles[0]
    p.circle_filled(
        center[0] + radius * math.cos(start_angle),
        center[1] + radius * math.sin(start_angle),
        half_width,
    )
