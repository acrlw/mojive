"""Canonical glyph shapes shared by production drawing and design studies."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from itertools import pairwise

from mojive.geometry2d.curves import (
    CORNER_SMOOTHING,
    box_handle_points,
    circular_stroke_mesh,
    polyline_ribbon,
    smooth_line_cap,
    smooth_polygon_corners,
)
from mojive.geometry2d.drag_link import smooth_union
from mojive.interaction.gizmo import DIMENSION_CORNER_RADIUS_RATIO

from ..overlay_geometry import (
    CAPSULE_SMOOTHING,
    OVERLAY_GEOMETRY,
    TOOL_GLYPH_SCALE,
    _rotate_visible_ring_polygons,
    _snap_glyph_shape,
    mouse_button_geometry,
    mouse_wheel_geometry,
)
from ..severity_icons import severity_icon
from .model import (
    ICON_GRID,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_STROKE,
    ICON_TUNING_DEFAULTS,
    ROTATE_FRINGE_GAP_FRACTION,
    ROTATE_FRINGE_MAX,
    STATUS_MOUSE_DEFAULT_WIDTH,
    IconTuning,
)
from .painter import (
    _arc_arrow,
    _chevron,
    _g3_rect,
    _more_centerline,
    _Painter,
    _rounded_polyline,
    _triangle,
    _triangle_path,
)


@lru_cache(maxsize=512)
def _rotate_ring_polygons(
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
        # Submit the outer frame and the three local half-rings as filled meshes
        # with the same fringe budget. ImGui's stroked-circle rasterizer has a
        # different subpixel coverage rule and made the frame read thinner at
        # the compact Tool Column size despite an identical nominal width.
        production_scale = 0.82
        glyph_scale = production_scale * TOOL_GLYPH_SCALE
        rendered_gap = p.stroke * p.rotate_ring_gap_ratio
        fringe_width = min(
            ROTATE_FRINGE_MAX,
            rendered_gap * ROTATE_FRINGE_GAP_FRACTION,
        )
        frame = circular_stroke_mesh(
            10.0 * glyph_scale,
            p.stroke_width * p.stroke_compensation,
        )
        p.indexed_fill(
            frame[0],
            frame[1],
            outline=frame[2],
            hole=frame[3],
            fringe_width=fringe_width,
        )
        for ring in _rotate_ring_polygons(
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
        endpoint_scale = p.tuning.snap_endpoint_scale
        endpoint_half_width = 1.35 * endpoint_scale
        endpoint_half_height = 0.9 * endpoint_scale
        for x, y in (path[0], path[-1]):
            p.smooth_polygon(
                (
                    (x - endpoint_half_width, y - endpoint_half_height),
                    (x + endpoint_half_width, y - endpoint_half_height),
                    (x + endpoint_half_width, y + endpoint_half_height),
                    (x - endpoint_half_width, y + endpoint_half_height),
                ),
                0.42 * endpoint_scale,
            )


@lru_cache(maxsize=512)
def _loop_arrow_path(width: float) -> tuple[tuple[float, float], ...]:
    """One bent arrow; its half-turn counterpart completes the repeat cycle."""
    centerline = smooth_polygon_corners(
        ((-7.0, 0.8), (-7.0, -4.8), (3.0, -4.8)),
        3.0,
        (1,),
        smoothing=CORNER_SMOOTHING,
        convex_only=False,
    )
    left, right, _ = polyline_ribbon(tuple(map(tuple, centerline)), width)
    cap = tuple(map(tuple, smooth_line_cap((-7.0, 0.8), (0.0, 1.0), width)))
    # Join shaft, head and tail into one contour: disabled alpha must not expose seams.
    outline = (*left[1:], (3.0, -2.0), (7.0, -4.8), (3.0, -7.6), *reversed(right[1:]), *cap)
    head = len(left) - 1
    rounded = smooth_polygon_corners(
        outline,
        0.42,
        tuple(range(head - 1, head + 4)),
        smoothing=CORNER_SMOOTHING,
        convex_only=False,
    )
    return tuple(map(tuple, rounded))


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
    elif kind == "loop":
        path = _loop_arrow_path(p.stroke_width * p.stroke_compensation)
        p.polygon(path)
        p.polygon(tuple((-x, -y) for x, y in path))
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
    elif kind in {"follow-off", "follow-page", "follow-locked"}:
        p.rect(-8.0, -6.0, 8.0, 6.0, rounding=1.6)
        if kind == "follow-off":
            p.line((-3.2, 0.0), (3.2, 0.0))
        elif kind == "follow-page":
            p.arrow(
                (-4.0, 0.0),
                (4.0, 0.0),
                head_length=2.8,
                head_width=4.6,
                round_tail=True,
            )
        else:
            # Locked holds the playhead's screen position, not edit permissions.
            # Join its triangular head and stem into one G3 silhouette.
            half = p.stroke_width * p.stroke_compensation * 0.5
            neck_y = 0.8 - half / 2.8 * 4.0
            p.smooth_polygon(
                (
                    (-2.8, -3.2),
                    (2.8, -3.2),
                    (half, neck_y),
                    (half, 3.3),
                    (-half, 3.3),
                    (-half, neck_y),
                ),
                0.35,
                convex_only=False,
            )
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
        severity_icon(p.draw, (p.cx, p.cy), p.scale * ICON_GRID, kind, p.color)
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


def _draw_glyph(
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
