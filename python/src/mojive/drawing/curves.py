"""Arc-length curvature profiles for G3 screen-space corners.

Each turn joins a cubic smoothstep curvature ramp, an optional circular arc,
and a reflected ramp. The analytic curve has zero curvature and zero
arc-length curvature derivative at both straight-edge joins. Returned paths
approximate that curve within a coordinate-space chord error.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise

import numpy as np

from mojive.drawing.polygons import remove_interior_loops, simple_polygon_indices

CORNER_SMOOTHING = 0.618
CURVE_TOLERANCE = 0.025
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(8)
_RAMP_QUADRATURE = tuple(
    (float(u**3 - 0.5 * u**4), float(w * 0.5))
    for u, w in zip((1.0 + _GAUSS_X) * 0.5, _GAUSS_W, strict=True)
)


def turn_curvature(distance, angle: float, smoothing: float = CORNER_SMOOTHING):
    """Return tangent angle, curvature and its arc-length derivative at unit peak curvature."""
    phi = abs(float(angle))
    sign = math.copysign(1.0, angle)
    ramp = float(smoothing) * phi
    arc = (1.0 - float(smoothing)) * phi
    total = 2.0 * ramp + arc
    s = np.clip(np.asarray(distance, np.float64), 0.0, total)
    if ramp == 0.0:
        return sign * s, np.full_like(s, sign), np.zeros_like(s)
    down = s > ramp + arc
    u = np.clip(np.where(down, total - s, s) / ramp, 0.0, 1.0)
    theta = ramp * (u**3 - 0.5 * u**4)
    theta = np.where(down, phi - theta, np.where(s > ramp, s - ramp * 0.5, theta))
    curvature = u * u * (3.0 - 2.0 * u)
    slope = 6.0 * u * (1.0 - u) / ramp * np.where(down, -1.0, 1.0)
    return sign * theta, sign * curvature, sign * slope


@lru_cache(maxsize=512)
def _turn_end(angle: float, smoothing: float) -> tuple[float, float]:
    ramp = smoothing * angle
    arc = (1.0 - smoothing) * angle
    nodes = (1.0 + _GAUSS_X) * ramp * 0.5
    theta, _, _ = turn_curvature(nodes, angle, smoothing)
    up = ramp * 0.5 * np.array((np.cos(theta) @ _GAUSS_W, np.sin(theta) @ _GAUSS_W))
    middle = np.array(
        (
            math.sin(angle - ramp * 0.5) - math.sin(ramp * 0.5),
            math.cos(ramp * 0.5) - math.cos(angle - ramp * 0.5),
        )
    )
    if arc == 0.0:
        middle[:] = 0.0
    reflection = np.array(((math.cos(angle), math.sin(angle)), (math.sin(angle), -math.cos(angle))))
    result = up + middle + reflection @ up
    return float(result[0]), float(result[1])


@lru_cache(maxsize=1024)
def smooth_turn_points(
    angle: float,
    scale: float = 1.0,
    smoothing: float = CORNER_SMOOTHING,
    tolerance: float = CURVE_TOLERANCE,
) -> tuple[tuple[float, float], ...]:
    """Sample a signed turn from the origin with initial tangent along +x.

    Scale is the minimum curvature radius, not the corner's edge footprint.
    The curvature bound gives chord error <= segment_length**2 / (8 * scale).
    Splitting at profile joins and at the bisector preserves symmetry.
    """
    phi = abs(float(angle))
    if not all(math.isfinite(v) for v in (angle, scale, smoothing, tolerance)):
        raise ValueError("curve parameters must be finite")
    if not (0.0 < phi <= math.pi and scale > 0.0 and 0.0 <= smoothing <= 1.0 and tolerance > 0.0):
        raise ValueError("invalid turn angle, scale, smoothing or tolerance")
    ramp = smoothing * phi
    arc = (1.0 - smoothing) * phi
    total = 2.0 * ramp + arc
    maximum_step = math.sqrt(8.0 * tolerance / scale)
    breaks = sorted({0.0, ramp, total * 0.5, ramp + arc, total})
    distances = [0.0]
    for lo, hi in pairwise(breaks):
        count = max(1, math.ceil((hi - lo) / maximum_step))
        distances.extend(np.linspace(lo, hi, count + 1)[1:])
    s = np.asarray(distances)
    half = np.diff(s) * 0.5
    samples = (s[:-1] + s[1:])[:, None] * 0.5 + half[:, None] * _GAUSS_X
    theta, _, _ = turn_curvature(samples, angle, smoothing)
    increments = half[:, None] * np.column_stack(
        (np.cos(theta) @ _GAUSS_W, np.sin(theta) @ _GAUSS_W)
    )
    points = np.vstack((np.zeros(2), np.cumsum(increments, axis=0))) * scale
    end = _turn_end(phi, smoothing)
    points[-1] = (end[0] * scale, math.copysign(1.0, angle) * end[1] * scale)
    return tuple(map(tuple, points.tolist()))


def smooth_polygon_corners(
    points,
    radius: float,
    corners: tuple[int, ...],
    *,
    smoothing: float = CORNER_SMOOTHING,
    tolerance: float = CURVE_TOLERANCE,
    convex_only: bool = True,
    corner_radii: dict[int, float] | None = None,
) -> np.ndarray:
    """Round selected corners, with optional per-corner radius overrides."""
    polygon = np.asarray(points, np.float64).reshape(-1, 2)
    if len(polygon) < 3 or (radius <= 0.0 and not corner_radii):
        return polygon.copy()
    area = np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    selected = {int(i) % len(polygon) for i in corners}
    fitted_runs = None
    if corner_radii:
        # Unequal corner radii share each edge's actual budget. Giving every
        # corner at most 45% would needlessly pinch a large concave shoulder
        # beside a deliberately tiny outer head corner.
        edges = np.roll(polygon, -1, axis=0) - polygon
        lengths = np.linalg.norm(edges, axis=1)
        directions = edges / np.maximum(lengths[:, None], 1e-30)
        incoming = np.roll(directions, 1, axis=0)
        turns = np.arctan2(
            incoming[:, 0] * directions[:, 1] - incoming[:, 1] * directions[:, 0],
            np.sum(incoming * directions, axis=1),
        )
        radii = np.array(
            [corner_radii.get(i, radius) if i in selected else 0.0 for i in range(len(polygon))]
        )
        valid = (np.abs(turns) < math.pi - 1e-4) & (np.minimum(lengths, np.roll(lengths, 1)) > 1e-9)
        if convex_only:
            valid &= turns * area > 0.0
        runs = np.where(valid, np.maximum(radii, 0.0) * np.tan(np.abs(turns) * 0.5), 0.0)
        factors = np.minimum(1.0, lengths * 0.99 / np.maximum(runs + np.roll(runs, -1), 1e-30))
        fitted_runs = runs * np.minimum(factors, np.roll(factors, 1))
    result = []
    for i, vertex in enumerate(polygon):
        corner_radius = radius if corner_radii is None else corner_radii.get(i, radius)
        incoming = vertex - polygon[i - 1]
        outgoing = polygon[(i + 1) % len(polygon)] - vertex
        lengths = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if i not in selected or corner_radius <= 0.0 or min(lengths) < 1e-9:
            result.append(vertex)
            continue
        incoming /= lengths[0]
        outgoing /= lengths[1]
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        angle = math.atan2(cross, float(incoming @ outgoing))
        if abs(angle) < 1e-6 or abs(angle) > math.pi - 1e-4 or (convex_only and cross * area <= 0):
            result.append(vertex)
            continue
        run = (
            float(fitted_runs[i])
            if fitted_runs is not None
            else min(corner_radius * math.tan(abs(angle) * 0.5), 0.45 * min(lengths))
        )
        end = _turn_end(abs(angle), smoothing)
        bisector = np.array((1.0 + math.cos(angle), math.sin(abs(angle))))
        unit_run = float(np.asarray(end) @ bisector / (bisector @ bisector))
        curve = np.asarray(smooth_turn_points(angle, run / unit_run, smoothing, tolerance))
        rotation = np.array(((incoming[0], -incoming[1]), (incoming[1], incoming[0])))
        curve = curve @ rotation.T + vertex - incoming * run
        curve[-1] = vertex + outgoing * run
        result.extend(curve)
    return np.asarray(result, np.float64)


@lru_cache(maxsize=128)
def _validate_arrow_style(width, head_length, head_width, radius, smoothing, join_radius=0.0):
    if not all(
        math.isfinite(v) for v in (width, head_length, head_width, radius, smoothing, join_radius)
    ):
        raise ValueError("arrow dimensions and smoothing must be finite")
    if (
        min(width, head_length, head_width, radius, join_radius) < 0.0
        or not 0.0 <= smoothing <= 1.0
    ):
        raise ValueError("arrow dimensions must be nonnegative and smoothing must be in [0, 1]")


@lru_cache(maxsize=128)
def _arrow_head(length, shaft, wing, radius, smoothing, support, join_radius):
    path = smooth_polygon_corners(
        (
            (-support, -shaft),
            (0.0, -shaft),
            (0.0, -wing),
            (length, 0.0),
            (0.0, wing),
            (0.0, shaft),
            (-support, shaft),
        ),
        radius,
        (1, 2, 3, 4, 5),
        smoothing=smoothing,
        convex_only=False,
        corner_radii={1: join_radius, 5: join_radius},
    )
    path.setflags(write=False)
    return path


def arrow_points(
    start,
    end,
    width: float = 2.0,
    *,
    head_length: float = 7.0,
    head_width: float = 8.0,
    corner_radius: float = 1.0,
    join_radius: float | None = None,
    smoothing: float = CORNER_SMOOTHING,
    round_tail: bool = False,
) -> np.ndarray:
    """Build one arrow silhouette with shared head, shoulder, and shaft transitions.

    Zero radius produces a sharp arrow. Zero smoothing with positive radius uses
    circular fillets. Positive smoothing matches G3 at every rounded straight join.
    Join radius defaults to half the head corner radius, keeping the concave
    shoulders subtler than the head's own corners. An explicit
    zero keeps the shoulders sharp. Dimensions must be finite and nonnegative;
    smoothing is in [0, 1]. Zero head
    length draws a plain shaft. Zero width or coincident endpoints return no path.
    """
    join_radius = corner_radius * 0.5 if join_radius is None else join_radius
    _validate_arrow_style(width, head_length, head_width, corner_radius, smoothing, join_radius)
    start, end = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if start.shape != (2,) or end.shape != (2,) or not all(map(math.isfinite, (*start, *end))):
        raise ValueError("arrow endpoints must each contain two finite coordinates")
    delta = end - start
    length = math.hypot(*delta)
    if not math.isfinite(length):
        raise ValueError("arrow endpoint separation must be finite")
    if length <= 1e-6 or width <= 0.0:
        return np.empty((0, 2), np.float64)
    direction = delta / length
    side = np.array((-direction[1], direction[0]))
    head = min(max(0.0, head_length), length * 0.42)
    shaft, wing = width * 0.5, max(width, head_width) * 0.5
    if head == 0.0:
        rear = (
            smooth_line_cap(
                (0.0, 0.0), (-1.0, 0.0), width, max_inset=length * 0.45, smoothing=smoothing
            )[::-1]
            if round_tail
            else np.array(((0.0, shaft), (0.0, -shaft)))
        )
        local = np.vstack(((length, -shaft), (length, shaft), rear))
        return start + local[:, :1] * direction + local[:, 1:] * side
    support = min(length - head, max(join_radius, 0.001) * 2.5)
    local = _arrow_head(head, shaft, wing, corner_radius, smoothing, support, join_radius)
    neck = end - direction * head
    path = neck + local[:, :1] * direction + local[:, 1:] * side
    if not round_tail:
        if support == length - head:
            return path
        return np.vstack((start - side * shaft, path, start + side * shaft))
    cap = smooth_line_cap(
        (0.0, 0.0),
        (-1.0, 0.0),
        width,
        max_inset=max(0.0, length - head - support) * 0.45,
        smoothing=smoothing,
    )[::-1]
    cap = start + cap[:, :1] * direction + cap[:, 1:] * side
    result = np.vstack((cap[-1], path, cap[:-1]))
    if support == length - head:
        # Foreshortened arrows can collapse the straight span onto the tail.
        result = result[np.linalg.norm(result - np.roll(result, 1, axis=0), axis=1) > 1e-8]
    return result


@lru_cache(maxsize=128)
def _box_handle_head(shaft, half, radius, join_radius, smoothing, support):
    path = smooth_polygon_corners(
        (
            (-support, -shaft),
            (0, -shaft),
            (0, -half),
            (2 * half, -half),
            (2 * half, half),
            (0, half),
            (0, shaft),
            (-support, shaft),
        ),
        radius,
        (1, 2, 3, 4, 5, 6),
        smoothing=smoothing,
        convex_only=False,
        corner_radii={1: join_radius, 6: join_radius},
    )
    path.setflags(write=False)
    return path


def box_handle_points(
    start,
    end,
    width: float,
    head_size: float,
    *,
    corner_radius: float = 0.5,
    join_radius: float | None = None,
    smoothing: float = CORNER_SMOOTHING,
) -> np.ndarray:
    """Join a shaft to an oriented square centered on end, using one G3 outline.

    Small outer corners retain the square footprint. The two concave joins default
    to half the outer corner radius and can be styled independently. Short shafts collapse to
    the square, avoiding self-intersections when projected along the viewing axis.
    """
    join_radius = corner_radius * 0.5 if join_radius is None else join_radius
    _validate_arrow_style(width, head_size, head_size, corner_radius, smoothing, join_radius)
    start, end = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if start.shape != (2,) or end.shape != (2,) or not all(map(math.isfinite, (*start, *end))):
        raise ValueError("box handle endpoints must each contain two finite coordinates")
    delta = end - start
    length = math.hypot(*delta)
    if not math.isfinite(length):
        raise ValueError("box handle endpoint separation must be finite")
    if length <= 1e-6 or width <= 0.0 or head_size <= 0.0:
        return np.empty((0, 2), np.float64)
    direction = delta / length
    side = np.array((-direction[1], direction[0]))
    shaft, half = min(width, head_size) * 0.5, head_size * 0.5
    if length <= half or shaft >= half:
        local = np.asarray(
            smooth_rect_points(
                min(-length, -half), -half, half, half, corner_radius, smoothing=smoothing
            )
        )
        return end + local[:, :1] * direction + local[:, 1:] * side
    support = min(length - half, max(join_radius, 0.001) * 2.5)
    local = _box_handle_head(shaft, half, corner_radius, join_radius, smoothing, support)
    neck = end - direction * half
    head = neck + local[:, :1] * direction + local[:, 1:] * side
    if support == length - half:
        return head
    return np.vstack((start - side * shaft, head, start + side * shaft))


@lru_cache(maxsize=256)
def _cap_geometry(width: float, smoothing: float, tolerance: float):
    height = _turn_end(math.pi, smoothing)[1]
    path = np.asarray(smooth_turn_points(math.pi, width / height, smoothing, tolerance))
    path[:, 1] -= width * 0.5
    path[0] = (0.0, -width * 0.5)
    path[-1] = (0.0, width * 0.5)
    path.setflags(write=False)
    return path, float(path[:, 0].max())


@lru_cache(maxsize=128)
def _affine_corner_template(radius: float, smoothing: float):
    points = np.asarray(smooth_turn_points(math.pi * 0.5, 2.0 * radius, smoothing))
    points /= points[-1].copy()
    points.setflags(write=False)
    return points


def smooth_affine_corners(points, radius: float, smoothing: float = CORNER_SMOOTHING):
    """Round convex projected handles by mapping one G3 template at each corner."""
    points = np.asarray(points, np.float64)
    if radius <= 0 or len(points) < 3:
        return points.copy()
    origin = points[0]
    local = tuple(map(tuple, (points - origin).tolist()))
    return _affine_polygon_shape(local, radius, smoothing) + origin


@lru_cache(maxsize=128)
def _affine_polygon_shape(local, radius: float, smoothing: float):
    points = np.asarray(local)
    # Round the sampling scale upward for bounded reuse during projection changes.
    # The geometric footprint itself is never quantized.
    template = _affine_corner_template(math.ceil(radius * 4.0) / 4.0, smoothing)
    result = []
    for index, point in enumerate(points):
        incoming, outgoing = point - points[index - 1], points[(index + 1) % len(points)] - point
        a, b = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(a, b) < 1e-9:
            result.append(point[None, :])
            continue
        run = min(radius, 0.4 * a, 0.4 * b)
        before, after = incoming * (run / a), outgoing * (run / b)
        result.append(point - before + template[:, :1] * before + template[:, 1:] * after)
    path = np.vstack(result)
    path.setflags(write=False)
    return path


@lru_cache(maxsize=256)
def _cap_depth_and_slope(smoothing: float) -> tuple[float, float]:
    """Evaluate cap depth and its analytic smoothing derivative in one quadrature."""
    ramp = smoothing * math.pi
    x = y = dx = dy = 0.0
    for phase, w in _RAMP_QUADRATURE:
        c, s = math.cos(ramp * phase), math.sin(ramp * phase)
        x += w * c
        y += w * s
        dx -= w * phase * s
        dy += w * phase * c
    c, s = math.cos(ramp * 0.5), math.sin(ramp * 0.5)
    numerator = ramp * x + 1.0 - s
    denominator = 2.0 * (ramp * y + c)
    d_numerator = x + ramp * dx - 0.5 * c
    d_denominator = 2.0 * (y + ramp * dy - 0.5 * s)
    return numerator / denominator, math.pi * (
        d_numerator * denominator - numerator * d_denominator
    ) / (denominator * denominator)


def _cap_depth_ratio(smoothing: float) -> float:
    return _cap_depth_and_slope(smoothing)[0]


@lru_cache(maxsize=256)
def _fit_cap_smoothing(depth_ratio: float, smoothing: float) -> float:
    maximum = _cap_depth_ratio(smoothing)
    if depth_ratio >= maximum:
        return smoothing
    if depth_ratio <= 0.5:
        return 0.0
    lo, hi = 0.0, smoothing
    q = smoothing * (depth_ratio - 0.5) / (maximum - 0.5)
    # Safeguarded Newton converges in a few evaluations for the monotone profile.
    # Retain the original bracketed solve as a bounded numerical fallback.
    for _ in range(8):
        value, slope = _cap_depth_and_slope(q)
        if abs(value - depth_ratio) <= 1e-13:
            return max(0.0, q - 1e-12)
        if value > depth_ratio:
            hi = q
        else:
            lo = q
        candidate = q - (value - depth_ratio) / slope if slope > 0.0 else lo
        q = candidate if lo < candidate < hi else (lo + hi) * 0.5
    for _ in range(30):
        mid = (lo + hi) * 0.5
        if _cap_depth_ratio(mid) > depth_ratio:
            hi = mid
        else:
            lo = mid
    return lo


@lru_cache(maxsize=512)
def smooth_capsule_points(
    x: float,
    y: float,
    width: float,
    height: float,
    smoothing: float = CORNER_SMOOTHING,
    tolerance: float = CURVE_TOLERANCE,
) -> tuple[tuple[float, float], ...]:
    """Return a horizontal or vertical G3 capsule with the requested outer bounds."""
    if width <= 0.0 or height <= 0.0:
        return ()
    local = _capsule_shape(width, height, smoothing, tolerance)
    return tuple((px + x, py + y) for px, py in local)


@lru_cache(maxsize=512)
def _capsule_shape(width: float, height: float, smoothing: float, tolerance: float):
    long, short = max(width, height), min(width, height)
    # Fit the scalar footprint first, then sample only the selected profile.
    # At square bounds the two caps form a complete circle.
    q = _fit_cap_smoothing(long / (2.0 * short), smoothing)
    cap, depth = _cap_geometry(short, q, tolerance)
    right = cap + np.array((long - depth, short * 0.5))
    left = -cap + (depth, short * 0.5)
    points = np.vstack((right, left))
    if height > width:
        points = np.column_stack((short - points[:, 1], points[:, 0]))
    # A fitted short capsule can leave a subpixel straight edge that collapses
    # to a duplicate ImVec2. Remove it before ImGui constructs AA normals.
    minimum_edge = min(tolerance * 0.01, max(1.0, short) * 1e-7)
    keep = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > minimum_edge
    return tuple(map(tuple, points[keep].tolist()))


@lru_cache(maxsize=1024)
def smooth_rect_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rounding: float,
    corners: tuple[bool, bool, bool, bool] = (True, True, True, True),
    tolerance: float = CURVE_TOLERANCE,
    *,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[float, float], ...]:
    """Return clockwise screen-space rectangle points; corner order is TL, TR, BR, BL."""
    width, height = x1 - x0, y1 - y0
    if width <= 0.0 or height <= 0.0:
        return ()
    radius = min(max(rounding, 0.0), min(width, height) * 0.5)
    local = _rect_shape(width, height, radius, corners, tolerance, smoothing)
    return tuple((x0 + px, y0 + py) for px, py in local)


@lru_cache(maxsize=1024)
def _rect_shape(
    width: float, height: float, radius: float, corners, tolerance: float, smoothing: float
):
    points = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    if radius == 0.0:
        return points
    # Rectangles may use the entire half-edge; unlike general polygons their
    # neighboring corner budgets are known and cannot overlap.
    quarter_scale = radius / _turn_end(math.pi * 0.5, smoothing)[0]
    curve = smooth_turn_points(math.pi * 0.5, quarter_scale, smoothing, tolerance)
    segments = (
        tuple((cy, radius - cx) for cx, cy in curve) if corners[0] else (points[0],),
        tuple((width - radius + cx, cy) for cx, cy in curve) if corners[1] else (points[1],),
        tuple((width - cy, height - radius + cx) for cx, cy in curve)
        if corners[2]
        else (points[2],),
        tuple((radius - cx, height - cy) for cx, cy in curve) if corners[3] else (points[3],),
    )
    result = []
    for segment in segments:
        for point in segment:
            if not result or math.dist(point, result[-1]) > 1e-9:
                result.append(point)
    if len(result) > 1 and math.dist(result[-1], result[0]) < 1e-9:
        result.pop()
    return tuple(result)


def smooth_line_cap(
    center,
    direction,
    width: float,
    *,
    tolerance: float = CURVE_TOLERANCE,
    max_inset: float = math.inf,
    smoothing: float = CORNER_SMOOTHING,
):
    """Return a cap from the left edge to the right edge, preserving the old tip extent."""
    q = smoothing
    if max_inset / width + 0.5 < _cap_depth_ratio(q):
        q = _fit_cap_smoothing(max_inset / width + 0.5, q)
    cap, depth = _cap_geometry(width, q, tolerance)
    tangent = np.asarray(direction, np.float64)
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array((-tangent[1], tangent[0]))
    # The cap starts slightly before the authored line endpoint so the tip
    # still extends by half the stroke width, as a conventional round cap does.
    return np.asarray(center) + (cap[:, :1] + width * 0.5 - depth) * tangent - cap[:, 1:] * normal


@lru_cache(maxsize=256)
def polyline_ribbon(
    path: tuple[tuple[float, float], ...],
    width: float,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
]:
    """Build cached left/right edges and the complete boundary of an open stroke."""

    if len(path) < 2 or width <= 0.0:
        return (), (), ()
    points = tuple((float(x), float(y)) for x, y in path)
    directions = []
    for start, end in pairwise(points):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return (), (), ()
        directions.append((dx / length, dy / length))
    normals = tuple((-dy, dx) for dx, dy in directions)
    half_width = 0.5 * float(width)
    offsets = []
    for index in range(len(points)):
        if index == 0:
            offsets.append((normals[0][0] * half_width, normals[0][1] * half_width))
            continue
        if index == len(points) - 1:
            offsets.append((normals[-1][0] * half_width, normals[-1][1] * half_width))
            continue
        mx = normals[index - 1][0] + normals[index][0]
        my = normals[index - 1][1] + normals[index][1]
        miter_length = math.hypot(mx, my)
        if miter_length <= 1e-9:
            offsets.append((normals[index][0] * half_width, normals[index][1] * half_width))
            continue
        mx, my = mx / miter_length, my / miter_length
        projection = max(mx * normals[index][0] + my * normals[index][1], 0.5)
        miter_scale = half_width / projection
        offsets.append((mx * miter_scale, my * miter_scale))
    left = tuple((p[0] + o[0], p[1] + o[1]) for p, o in zip(points, offsets, strict=True))
    right = tuple((p[0] - o[0], p[1] - o[1]) for p, o in zip(points, offsets, strict=True))
    return left, right, left + tuple(reversed(right))


def capped_polyline_points(
    points,
    width: float,
    *,
    round_start: bool,
    round_end: bool,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[float, float], ...]:
    """Build one non-overlapping silhouette with independent endpoint caps."""

    path = tuple((float(point[0]), float(point[1])) for point in points)
    return _capped_polyline_points(path, float(width), round_start, round_end, smoothing)


@lru_cache(maxsize=256)
def _capped_polyline_points(path, width, round_start, round_end, smoothing):
    """Retain the complete immutable stroke, including fitted endpoint caps."""
    left, right, _outline = polyline_ribbon(path, float(width))
    if not left:
        return ()

    start_direction = np.asarray(path[1], np.float64) - np.asarray(path[0], np.float64)
    end_direction = np.asarray(path[-1], np.float64) - np.asarray(path[-2], np.float64)
    start_direction /= np.linalg.norm(start_direction)
    end_direction /= np.linalg.norm(end_direction)
    left, right = list(left), list(right)
    if round_end:
        boundary = smooth_line_cap(
            path[-1],
            end_direction,
            width,
            max_inset=0.45 * math.dist(path[-2], path[-1]),
            smoothing=smoothing,
        )
        left[-1], right[-1] = tuple(boundary[0]), tuple(boundary[-1])
        end_boundary = tuple(map(tuple, boundary[1:-1]))
    else:
        end_boundary = ()
    if round_start:
        boundary = smooth_line_cap(
            path[0],
            -start_direction,
            width,
            max_inset=0.45 * math.dist(path[0], path[1]),
            smoothing=smoothing,
        )
        right[0], left[0] = tuple(boundary[0]), tuple(boundary[-1])
        start_boundary = tuple(map(tuple, boundary[1:-1]))
    else:
        start_boundary = ()
    outline = (*left, *end_boundary, *reversed(right), *start_boundary)
    # Match the clockwise screen-space winding of other shared shapes.
    return (*reversed(outline),)


def arc_ribbon_points(
    points,
    start_radial,
    end_radial,
    width: float,
    *,
    round_caps: bool = False,
    round_start: bool = False,
    round_end: bool = False,
    smoothing: float = CORNER_SMOOTHING,
) -> np.ndarray:
    """Build a sampled arc ribbon with optional radial cuts and G3 endpoint caps.

    Radial directions control endpoint cuts without changing caller arrays. Caps
    match short straight endpoint segments; the remaining centerline is sampled.
    Translation reuses the local ribbon, including its tangents and cap profiles.
    """
    return arc_ribbon_mesh(
        points,
        start_radial,
        end_radial,
        width,
        round_caps=round_caps,
        round_start=round_start,
        round_end=round_end,
        smoothing=smoothing,
    )[0]


def arc_ribbon_mesh(
    points,
    start_radial,
    end_radial,
    width: float,
    *,
    round_caps: bool = False,
    round_start: bool = False,
    round_end: bool = False,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return an arc stroke mesh, using strip indices unless its offset folds.

    The boundary is also the mesh vertex array. Apply AA only along that boundary.
    Zero smoothing uses circular caps without fitting or endpoint insets.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 2 or width <= 0.0:
        return np.empty((0, 2), np.float64), ()
    origin = points[0]
    # Translation can leave subtraction noise in an otherwise identical path.
    # Normalize far below pixel/chord precision so that it does not thrash the cache.
    local = tuple(map(tuple, np.round(points - origin, 9).tolist()))
    start = None if start_radial is None else tuple(map(float, start_radial))
    end = None if end_radial is None else tuple(map(float, end_radial))
    shape, indices = _arc_ribbon_shape(
        local,
        start,
        end,
        width,
        round_start=round_start or round_caps,
        round_end=round_end or round_caps,
        smoothing=smoothing,
    )
    return shape + origin, indices


@lru_cache(maxsize=128)
def _arc_ribbon_shape(
    points,
    start_radial,
    end_radial,
    width: float,
    *,
    round_start: bool = False,
    round_end: bool = False,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build one constant-width arc silhouette with independently rounded caps."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    width = float(width)
    if len(points) < 2 or width <= 0.0:
        return np.empty((0, 2), np.float64), ()

    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    for index in np.flatnonzero(lengths < 1e-6):
        candidates = []
        if index > 0:
            candidates.append(points[index] - points[index - 1])
        if index + 1 < len(points):
            candidates.append(points[index + 1] - points[index])
        if candidates:
            tangent = max(candidates, key=np.linalg.norm)
            tangents[index] = tangent
            lengths[index] = np.linalg.norm(tangent)
    if np.any(lengths < 1e-6):
        return np.empty((0, 2), np.float64), ()

    tangents /= lengths[:, None]
    offsets = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    for index, radial, rounded in ((0, start_radial, round_start), (-1, end_radial, round_end)):
        if radial is None or (rounded and smoothing == 0.0):
            continue
        radial = np.asarray(radial, np.float64).reshape(2)
        length = float(np.linalg.norm(radial))
        if length < 1e-6:
            continue
        radial /= length
        if np.dot(radial, offsets[index]) < 0.0:
            radial *= -1.0
        offsets[index] = radial

    half_width = 0.5 * width
    side_a = points - offsets * half_width
    side_b = points + offsets * half_width
    start_cap = end_cap = np.empty((0, 2), np.float64)
    lo, hi = 1, len(points) - 1
    cap_lo, cap_hi = lo, hi
    if smoothing == 0.0 and (round_start or round_end):
        # Sample only the interior of a semicircle; its endpoints are already
        # the strip vertices. No G3 fit, inward stub or cap trimming is needed.
        cosine, sine = _round_cap_profile(width)
        if round_end:
            end_cap = points[-1] + half_width * (-cosine * offsets[-1] + sine * tangents[-1])
        if round_start:
            start_cap = points[0] + half_width * (cosine * offsets[0] - sine * tangents[0])
    elif round_start or round_end:
        distance = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )
        run = min(width * 0.25, distance[-1] * 0.1)
        profile = smooth_line_cap(
            (0.0, 0.0),
            (1.0, 0.0),
            width,
            max_inset=max(0.0, distance[-1] * 0.45 - run),
            smoothing=smoothing,
        )[::-1]
        inset = max(0.0, -profile[0, 0]) + run

        def cap(index):
            direction = 1.0 if index == -1 else -1.0
            return points[index] + direction * (
                profile[:, :1] * tangents[index] + profile[:, 1:] * offsets[index]
            )

        if round_end:
            end_cap = cap(-1)
            side_a[-1] = end_cap[0] - tangents[-1] * run
            side_b[-1] = end_cap[-1] - tangents[-1] * run
        if round_start:
            start_cap = cap(0)
            side_b[0] = start_cap[0] + tangents[0] * run
            side_a[0] = start_cap[-1] + tangents[0] * run
        # Remove samples covered by a cap instead of refitting its curvature
        # whenever projection or centerline sampling changes one short segment.
        lo = max(1, int(np.searchsorted(distance, inset, side="right"))) if round_start else 1
        hi = (
            min(len(points) - 1, int(np.searchsorted(distance, distance[-1] - inset)))
            if round_end
            else len(points) - 1
        )
        cap_lo, cap_hi = lo, hi
        # The inner offset travels less than the centerline on a curved arc.
        # Keep both edges beyond the straight stub to prevent tiny folded quads.
        while (
            round_start
            and lo < hi
            and min((side_a[lo] - side_a[0]) @ tangents[0], (side_b[lo] - side_b[0]) @ tangents[0])
            <= 1e-9
        ):
            lo += 1
        while (
            round_end
            and hi > lo
            and max(
                (side_a[hi - 1] - side_a[-1]) @ tangents[-1],
                (side_b[hi - 1] - side_b[-1]) @ tangents[-1],
            )
            >= -1e-9
        ):
            hi -= 1
    result, indices = _arc_strip_mesh(side_a, side_b, end_cap, start_cap, lo, hi)
    triangles = result[np.asarray(indices).reshape(-1, 3)]
    ab = triangles[:, 1] - triangles[:, 0]
    ac = triangles[:, 2] - triangles[:, 0]
    if np.any(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0] < -1e-9):
        # Tight perspective projections can fold the inward offset, and a cap's
        # tangent half-plane can exclude the entire arc. Recover the exterior
        # of the untrimmed stroke instead of connecting its endpoints directly.
        if round_start or round_end:
            result, _indices = _arc_strip_mesh(side_a, side_b, end_cap, start_cap, cap_lo, cap_hi)
        outline = remove_interior_loops(tuple(map(tuple, result.tolist())))
        indices = simple_polygon_indices(outline)
        result = np.asarray(outline, np.float64)
    result.setflags(write=False)
    return result, indices


def _arc_strip_mesh(side_a, side_b, end_cap, start_cap, lo, hi):
    side_a = np.vstack((side_a[:1], side_a[lo:hi], side_a[-1:]))
    side_b = np.vstack((side_b[:1], side_b[lo:hi], side_b[-1:]))
    return (
        np.vstack((side_a, end_cap, side_b[::-1], start_cap)),
        _arc_ribbon_indices(len(side_a), len(end_cap), len(start_cap)),
    )


@lru_cache(maxsize=128)
def _round_cap_profile(width):
    count = max(8, math.ceil(math.pi * width * 0.5))
    angles = np.linspace(0.0, math.pi, count + 1)[1:-1, None]
    cosine, sine = np.cos(angles), np.sin(angles)
    cosine.setflags(write=False)
    sine.setflags(write=False)
    return cosine, sine


@lru_cache(maxsize=128)
def _arc_ribbon_indices(count: int, end_count: int, start_count: int) -> tuple[int, ...]:
    indices = []
    for a in range(count - 1):
        b = count * 2 + end_count - 1 - a
        indices.extend((a, a + 1, b - 1, a, b - 1, b))
    for i in range(count, count + end_count):
        indices.extend((count - 1, i, i + 1))
    start = count * 2 + end_count
    for i in range(start, start + start_count):
        indices.extend((start - 1, i, i + 1 if i + 1 < start + start_count else 0))
    return tuple(indices)


def offset_closed_path(points, distance: float) -> np.ndarray:
    """Offset a sampled closed boundary outward, preserving its winding."""
    path = np.asarray(points, np.float64).reshape(-1, 2)
    edges = np.roll(path, -1, axis=0) - path
    lengths = np.linalg.norm(edges, axis=1)
    if len(path) < 3 or np.any(lengths < 1e-9):
        return path.copy()
    area = np.sum(path[:, 0] * np.roll(path[:, 1], -1) - path[:, 1] * np.roll(path[:, 0], -1))
    normals = (
        math.copysign(1.0, area) * np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    )
    miters = (np.roll(normals, 1, axis=0) + normals) * 0.5
    factors = np.minimum(1.0 / np.maximum(np.sum(miters * miters, axis=1), 1e-4), 100.0)
    return path + distance * miters * factors[:, None]


def _sample_bezier(control, tolerance: float) -> np.ndarray:
    """Subdivide until the entire control hull lies within the chord tolerance."""
    result = [control[0]]

    def visit(p):
        chord = p[-1] - p[0]
        length2 = float(chord @ chord)
        fraction = np.clip((p - p[0]) @ chord / max(length2, 1e-30), 0.0, 1.0)
        if np.max(np.linalg.norm(p - p[0] - fraction[:, None] * chord, axis=1)) <= tolerance:
            result.append(p[-1])
            return
        levels = [p]
        for _ in range(len(p) - 1):
            levels.append((levels[-1][:-1] + levels[-1][1:]) * 0.5)
        visit(np.asarray([level[0] for level in levels]))
        visit(np.asarray([level[-1] for level in reversed(levels)]))

    visit(np.asarray(control))
    return np.asarray(result)


def _lollipop_bridge(radius: float, half: float, run: float, smoothing: float):
    """Match third-order jets from a straight shaft to the exact circular head."""
    angle = -math.pi + math.asin(half / radius) + 0.35 * smoothing
    radial = np.array((math.cos(angle), math.sin(angle)))
    tangent = np.array((-radial[1], radial[0]))
    start = np.array((-math.sqrt(radius * radius - half * half) - run, -half))
    end = radius * radial
    speed = float(np.linalg.norm(end - start))
    velocity = speed * tangent
    acceleration = -(speed * speed / radius) * radial
    jerk = -(speed**3 / radius**2) * tangent
    p = np.empty((8, 2))
    p[:4] = start + np.arange(4)[:, None] * np.array((speed / 7.0, 0.0))
    p[7] = end
    p[6] = end - velocity / 7.0
    p[5] = acceleration / 42.0 + 2.0 * p[6] - p[7]
    p[4] = 3.0 * p[5] - 3.0 * p[6] + p[7] - jerk / 210.0
    return p, angle


@lru_cache(maxsize=256)
def _lollipop_head(radius: float, half: float, run: float, smoothing: float, tolerance: float):
    control, start = _lollipop_bridge(radius, half, run, smoothing)
    bridge = _sample_bezier(control, tolerance) if smoothing > 0.0 else control[-1:]
    step = math.sqrt(8.0 * tolerance / radius)
    angles = np.linspace(start, -start, max(3, math.ceil(-2.0 * start / step) + 1))
    arc = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((bridge, arc[1:-1], bridge[::-1] * (1.0, -1.0)))


@lru_cache(maxsize=256)
def smooth_lollipop_points(
    distance: float,
    radius: float,
    width: float,
    smoothing: float = CORNER_SMOOTHING,
    tolerance: float = CURVE_TOLERANCE,
) -> tuple[tuple[float, float], ...]:
    """Join a round-capped shaft to a circle with G3 neck transitions when smoothing is positive."""
    # Foreshortening can hide the shaft inside its head. Fade the whole neck
    # into the circle limit instead of popping a full-width end cap away.
    visible = min(1.0, max(0.0, distance - radius) / width)
    half = min(width * 0.5, radius * 0.5) * visible
    smoothing *= visible
    run = min(radius * 0.6, max(0.0, distance - radius) * 0.4) * smoothing
    head = _lollipop_head(radius, half, run, smoothing, tolerance)
    cap = smooth_line_cap(
        (-distance, 0.0),
        (-1.0, 0.0),
        half * 2.0,
        smoothing=smoothing,
        max_inset=max(0.0, distance + head[0, 0]) * 0.4,
    )
    return tuple(map(tuple, np.vstack((cap[0], head, cap[:0:-1])).tolist()))


def clip_polygon_rect(points, bounds) -> tuple[tuple[float, float], ...]:
    """Clip a polygon against an axis-aligned rectangle without changing its winding."""
    result = [tuple(p) for p in points]
    for axis, value, sign in (
        (0, bounds[0], 1),
        (1, bounds[1], 1),
        (0, bounds[2], -1),
        (1, bounds[3], -1),
    ):
        if not result:
            break
        source, result = result, []
        previous = source[-1]
        for current in source:
            a = sign * (previous[axis] - value)
            b = sign * (current[axis] - value)
            if (a >= 0) != (b >= 0):
                u = a / (a - b)
                result.append(tuple(previous[j] + u * (current[j] - previous[j]) for j in (0, 1)))
            if b >= 0:
                result.append(current)
            previous = current
    return tuple(result)


def _ellipse_frame(a, b, parameter):
    t = np.asarray(parameter)[..., None]
    center = np.cos(t) * a + np.sin(t) * b
    velocity = -np.sin(t) * a + np.cos(t) * b
    speed = np.linalg.norm(velocity, axis=-1)
    tangent = velocity / speed[..., None]
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    return center, tangent, normal, speed


def _ellipse_advance(a, b, parameter, distance):
    """Invert short ellipse arc lengths with quadrature and Newton iteration."""
    distance = np.asarray(distance)
    speed = _ellipse_frame(a, b, parameter)[3]
    result = parameter + distance / speed
    for _ in range(5):
        half = (result - parameter) * 0.5
        nodes = parameter + half[..., None] * (1.0 + _GAUSS_X)
        length = half * (_ellipse_frame(a, b, nodes)[3] @ _GAUSS_W)
        result -= (length - distance) / _ellipse_frame(a, b, result)[3]
    return result


@lru_cache(maxsize=128)
def smooth_ellipse_stroke(
    cosine: tuple[float, float],
    sine: tuple[float, float],
    width: float,
    start: float = 0.0,
    end: float = math.pi,
    *,
    rounded: bool = True,
    tolerance: float = CURVE_TOLERANCE,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[float, float], ...]:
    """Offset an ellipse arc and join its two sides with G3 caps.

    A cap is mapped through F(s,n) = C(s) + n N(s). This smooth coordinate
    map preserves the matching third-order jets at the straight cap's joins,
    so the mapped cap matches each curved offset boundary through G3.
    Width must stay below the ellipse's minimum diameter of curvature.
    """
    a, b = np.asarray(cosine), np.asarray(sine)
    axes = np.linalg.svd(np.column_stack((a, b)), compute_uv=False)
    half_width = width * 0.5
    reach = axes[-1] ** 2 / axes[0]
    if not 0.0 < half_width < reach or end <= start:
        raise ValueError("ellipse stroke must have positive extent and regular offsets")
    start_cap = end_cap = np.empty((0, 2))
    if rounded:
        cap, depth = _cap_geometry(width, smoothing, tolerance * 0.125)
        cap = np.asarray(cap)
        inset = depth - half_width
        lo = float(_ellipse_advance(a, b, start, inset))
        hi = float(_ellipse_advance(a, b, end, -inset))
        if lo >= hi:
            raise ValueError("ellipse arc is too short for the requested caps")
        t = _ellipse_advance(a, b, hi, cap[:, 0])
        centers, _, normals, _ = _ellipse_frame(a, b, t)
        end_cap = centers - cap[:, 1:] * normals
        t = _ellipse_advance(a, b, lo, -cap[:, 0])
        centers, _, normals, _ = _ellipse_frame(a, b, t)
        start_cap = centers + cap[:, 1:] * normals
    else:
        lo, hi = start, end
    # The offset's second derivative is bounded using the ellipse's singular
    # values. The linear interpolation error is <= max|P''| * dt**2 / 8.
    normal_bound = (axes[0] / axes[-1]) ** 2 * 3.0 + 1.0
    acceleration_bound = axes[0] + half_width * normal_bound
    step = min(math.pi / 32.0, math.sqrt(8.0 * tolerance / acceleration_bound))
    parameters = np.linspace(lo, hi, max(2, math.ceil((hi - lo) / step) + 1))
    centers, _, normals, _ = _ellipse_frame(a, b, parameters)
    left = centers + half_width * normals
    right = centers - half_width * normals
    outline = np.vstack((left, end_cap[1:-1], right[::-1], start_cap[1:-1]))[::-1]
    return tuple(map(tuple, outline))


def polygon_fringe(points) -> np.ndarray:
    """Return a one-pixel outward miter ring for either polygon winding."""

    outline = np.asarray(points, np.float64).reshape(-1, 2)
    if len(outline) < 3:
        return np.empty((0, 2), np.float64)
    edges = np.roll(outline, -1, axis=0) - outline
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-9):
        return np.empty((0, 2), np.float64)
    signed_area = 0.5 * float(
        np.sum(outline[:, 0] * np.roll(outline[:, 1], -1))
        - np.sum(outline[:, 1] * np.roll(outline[:, 0], -1))
    )
    if abs(signed_area) < 1e-9:
        return np.empty((0, 2), np.float64)
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    if signed_area < 0.0:
        normals *= -1.0
    miters = (np.roll(normals, 1, axis=0) + normals) * 0.5
    scale = np.minimum(1.0 / np.maximum(np.sum(miters * miters, axis=1), 1e-4), 100.0)
    return outline + miters * scale[:, None]


@lru_cache(maxsize=128)
def arrow_mesh(
    length,
    width=2.0,
    *,
    head_length=7.0,
    head_width=8.0,
    corner_radius=1.0,
    smoothing=CORNER_SMOOTHING,
    round_tail=False,
    join_radius=None,
):
    """Return immutable local vertices, fan indices, and the external arrow contour.

    The neck is in the arrow's kernel, so construction is linear in boundary
    vertices. Placement and adapter-specific antialiasing do not affect the cache.
    """
    if not math.isfinite(length) or length < 0.0:
        raise ValueError("arrow length must be finite and nonnegative")
    outline = arrow_points(
        (0.0, 0.0),
        (length, 0.0),
        width,
        head_length=head_length,
        head_width=head_width,
        corner_radius=corner_radius,
        join_radius=join_radius,
        smoothing=smoothing,
        round_tail=round_tail,
    )
    if len(outline) < 3:
        return (), (), ()
    distinct = np.linalg.norm(outline - np.roll(outline, 1, axis=0), axis=1) > 1e-8
    outline = tuple(map(tuple, outline[distinct].tolist()))
    # The neck center is in the kernel of this star-shaped arrow, including its
    # concave shoulders. A triangle fan therefore needs no ear clipping.
    center = (length - min(head_length, length * 0.42), 0.0)
    n = len(outline)
    indices = tuple(index for i in range(n) for index in (0, i + 1, (i + 1) % n + 1))
    return (center, *outline), indices, outline


@lru_cache(maxsize=128)
def arrow_triangles(
    length,
    width,
    head_length,
    head_width,
    corner_radius,
    smoothing,
    round_tail,
    antialias,
    join_radius=None,
):
    """Pack the shared arrow mesh and external AA fringe for retained debug drawing."""
    vertices, indices, boundary = arrow_mesh(
        length,
        width,
        head_length=head_length,
        head_width=head_width,
        corner_radius=corner_radius,
        join_radius=join_radius,
        smoothing=smoothing,
        round_tail=round_tail,
    )
    if not indices:
        return np.empty((0, 3, 3), np.float64)
    outline = np.asarray(boundary)
    n = len(outline)
    fill = np.empty((n, 3, 3))
    fill[:, :, :2] = np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)]
    fill[:, :, 2] = 1.0
    if antialias:
        outside = polygon_fringe(outline)
        ring = np.empty((n, 6, 3))
        ring[:, :, :2] = np.stack(
            (
                outline,
                np.roll(outline, -1, axis=0),
                np.roll(outside, -1, axis=0),
                outline,
                np.roll(outside, -1, axis=0),
                outside,
            ),
            axis=1,
        )
        ring[:, :, 2] = (1, 1, 0, 1, 0, 0)
        fill = np.vstack((fill, ring.reshape(-1, 3, 3)))
    fill.setflags(write=False)
    return fill
