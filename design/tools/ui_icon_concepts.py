"""Concept-only Mojive icon families for the UI feasibility workbench.

These painters deliberately do not feed production UI.  They provide a shared
24-unit comparison grid so candidate families can be reviewed together before
any production glyph is replaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from mojive.curves2d import arrow_mesh

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

    @property
    def center_offset(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5

    @property
    def radial_clearance(self) -> float:
        return ICON_BOUND_DIAMETER * 0.5 - self.radial_extent


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
    p.polygon(
        (
            (center_x - direction * 6.4 * scale, -6.8 * scale),
            (center_x + direction * 6.4 * scale, 0.0),
            (center_x - direction * 6.4 * scale, 6.8 * scale),
        )
    )


def _chevron(p: _Painter, direction: float, x: float, *, scale: float = 1.0) -> None:
    p.polyline(
        (
            (x - direction * 2.6 * scale, -4.4 * scale),
            (x + direction * 2.2 * scale, 0.0),
            (x - direction * 2.6 * scale, 4.4 * scale),
        )
    )


def _arc_arrow(
    p: _Painter,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    *,
    center=(0.0, 0.0),
) -> None:
    """Draw a circular arrow as one joined ribbon instead of two shapes."""

    direction = 1.0 if end_degrees >= start_degrees else -1.0
    count = max(24, round(abs(end_degrees - start_degrees) / 8.0))
    angles = tuple(
        math.radians(start_degrees + (end_degrees - start_degrees) * index / count)
        for index in range(count + 1)
    )
    half_width = ICON_STROKE * 0.5
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
    base_outer = (base_center[0] + normal[0] * 2.4, base_center[1] + normal[1] * 2.4)
    base_inner = (base_center[0] - normal[0] * 2.4, base_center[1] - normal[1] * 2.4)
    tip = (end[0] + tangent[0] * 3.0, end[1] + tangent[1] * 3.0)
    p.polygon((*outer, base_outer, tip, base_inner, *inner))
    start_angle = angles[0]
    p.circle_filled(
        center[0] + radius * math.cos(start_angle),
        center[1] + radius * math.sin(start_angle),
        half_width,
    )


def _draw_tool(p: _Painter, name: str) -> None:
    if name == "tool-move":
        # One connected outline avoids the visible seams produced by four
        # stroked shafts with separately filled arrowheads.
        tip, base, wing, shaft = 8.7, 5.6, 2.65, 0.82
        p.polygon(
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
            )
        )
    elif name == "tool-rotate":
        _arc_arrow(p, 6.9, -42.0, 265.0, center=(0.0, 0.8))
    elif name == "tool-scale":
        center = (0.0, 1.0)
        for end in ((0.0, -5.9), (-6.0, 5.9), (6.0, 5.9)):
            p.line(center, end, width=1.45)
            p.rect_filled(
                end[0] - 1.35,
                end[1] - 1.35,
                end[0] + 1.35,
                end[1] + 1.35,
                rounding=0.48,
            )
        p.circle_filled(*center, 1.15)
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
        p.polyline((top, right, lower_right, bottom, lower_left, left), closed=True)
        # A cube has three visible faces: the center joins the two rear side
        # corners and the lower vertex. The former top-to-center edge invented
        # a fourth face.
        p.line(left, (0.0, 0.0), width=1.45)
        p.line(right, (0.0, 0.0), width=1.45)
        p.line((0.0, 0.0), bottom, width=1.45)
    else:
        p.line((-5.4, -6.03), (-5.4, 1.17))
        p.arc(5.4, 180.0, 0.0, center=(0.0, 1.17))
        p.line((5.4, 1.17), (5.4, -6.03))
        p.rect_filled(-6.8, -7.33, -4.0, -5.53, rounding=0.45)
        p.rect_filled(4.0, -7.33, 6.8, -5.53, rounding=0.45)


def _draw_transport(p: _Painter, name: str) -> None:
    kind = name.removeprefix("transport-")
    if kind == "play":
        _triangle(p, 1.0)
    elif kind == "pause":
        p.rect_filled(-5.0, -6.8, -1.5, 6.8, rounding=0.75)
        p.rect_filled(1.5, -6.8, 5.0, 6.8, rounding=0.75)
    elif kind in {"previous", "next"}:
        direction = -1.0 if kind == "previous" else 1.0
        _chevron(p, direction, direction * 0.24, scale=1.18)
    elif kind in {"first", "last"}:
        direction = -1.0 if kind == "first" else 1.0
        # A skip-to-end symbol pairs one filled transport triangle with a bar.
        # The small separation is intentional and stays symmetrical when mirrored.
        _triangle(p, direction, center_x=-direction * 1.27, scale=0.72)
        x = direction * 5.13
        p.line((x, -5.8), (x, 5.8))
    elif kind == "reset":
        _arc_arrow(p, 6.8, -52.0, 255.0, center=(0.0, 0.6))
    elif kind == "record":
        p.circle_filled(0.0, 0.0, 4.8)
    elif kind == "stop":
        p.rect_filled(-4.8, -4.8, 4.8, 4.8, rounding=0.9)
    else:
        p.polyline(((-5.6, -2.8), (0.0, 2.8), (5.6, -2.8)))


def _diamond(p: _Painter, center=(0.0, 0.0), radius: float = 5.4, *, filled: bool) -> None:
    points = (
        (center[0], center[1] - radius),
        (center[0] + radius, center[1]),
        (center[0], center[1] + radius),
        (center[0] - radius, center[1]),
    )
    if filled:
        p.polygon(points)
    else:
        p.polyline(points, closed=True)


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


def _draw_panel(p: _Painter, name: str) -> None:
    kind = name.removeprefix("panel-")
    if kind == "search":
        p.circle(-1.7, -1.7, 4.8)
        p.line((1.7, 1.7), (6.5, 6.5), width=1.65)
    elif kind == "sort":
        for y, end in ((-5.09, 2.24), (0.11, 0.04), (5.31, -2.16)):
            p.line((-6.76, y), (end, y), width=1.45)
        p.arrow((5.54, -6.49), (5.54, 6.71), width=1.4, head_length=3.0, head_width=4.6)
    elif kind == "clear":
        p.line((-5.8, -5.8), (5.8, 5.8), width=1.65)
        p.line((5.8, -5.8), (-5.8, 5.8), width=1.65)
    elif kind in {"visible", "hidden"}:
        p.polyline(_eye_points(), closed=True, width=1.45)
        p.circle_filled(0.0, 0.0, 2.05)
        if kind == "hidden":
            p.line((-6.6, -6.6), (6.6, 6.6), width=1.65)
    elif kind in {"perspective", "orthographic"}:
        near = 3.0 if kind == "perspective" else 6.1
        p.polyline(((-6.8, -near), (6.8, -6.1), (6.8, 6.1), (-6.8, near)), closed=True)
    elif kind == "right":
        p.polygon(((-3.8, -6.2), (3.8, 0.0), (-3.8, 6.2)))
    else:
        p.polygon(((-6.2, -3.8), (0.0, 3.8), (6.2, -3.8)))


def _draw_camera(p: _Painter) -> None:
    """Draw one camera outline with an integrated top and centered lens."""

    # The body and viewfinder form one contour, so no interior stroke crosses
    # the shell. The lens uses the geometric center of the rectangular body.
    p.polyline(
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
        closed=True,
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
    p.polyline(
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
        closed=True,
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


def draw_concept_icon(draw, center, size: float, name: str, color) -> None:
    """Draw one candidate icon using a single proportional 24-unit master."""

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
    """Measure authored ink from the same primitives used by the review renderer."""

    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.radial_extent = 0.0

    def _add(self, points, pad: float = 0.0) -> None:
        for raw_x, raw_y in points:
            x, y = float(raw_x), float(raw_y)
            self.bounds[0] = min(self.bounds[0], x - pad)
            self.bounds[1] = min(self.bounds[1], y - pad)
            self.bounds[2] = max(self.bounds[2], x + pad)
            self.bounds[3] = max(self.bounds[3], y + pad)
            self.radial_extent = max(self.radial_extent, math.hypot(x, y) + pad)

    def line(self, a, b, _color, width, **_kwargs) -> None:
        self._add((a, b), float(width) * 0.5)

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
        )
        self._add(tuple((a[0] + x * ux - y * uy, a[1] + x * uy + y * ux) for x, y in outline))

    def polyline(self, points, _color, width, **_kwargs) -> None:
        self._add(tuple(points), float(width) * 0.5)

    def fringed_concave_fill(self, points, _color, **_kwargs) -> None:
        self._add(tuple(points))

    def circle(self, center, radius, _color, width, **_kwargs) -> None:
        self._add((center,), float(radius) + float(width) * 0.5)

    def circle_filled(self, center, radius, _color, **_kwargs) -> None:
        self._add((center,), float(radius))

    def rect(self, lo, hi, _color, width, **_kwargs) -> None:
        self._add((lo, hi), float(width) * 0.5)

    def rect_filled(self, lo, hi, _color, **_kwargs) -> None:
        self._add((lo, hi))


@lru_cache(maxsize=64)
def icon_metrics(name: str) -> IconMetrics:
    """Return 24-unit authored bounds and radial clearance for one candidate."""

    draw = _MetricsDraw()
    draw_concept_icon(draw, (0.0, 0.0), ICON_GRID, name, (1.0, 1.0, 1.0, 1.0))
    return IconMetrics(tuple(draw.bounds), draw.radial_extent)


def icon_family(label: str):
    """Return one family by its user-facing label."""

    return next(icons for family, icons in ICON_FAMILIES if family == label)
