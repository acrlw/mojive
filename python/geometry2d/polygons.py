"""Shared contour operations independent of UI and graphics backends."""

from functools import lru_cache


def _cross2(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def segment_intersection(start, end, other_start, other_end):
    direction = (end[0] - start[0], end[1] - start[1])
    other_direction = (
        other_end[0] - other_start[0],
        other_end[1] - other_start[1],
    )
    denominator = _cross2(direction, other_direction)
    if abs(denominator) <= 1e-9:
        return None
    offset = (other_start[0] - start[0], other_start[1] - start[1])
    amount = _cross2(offset, other_direction) / denominator
    other_amount = _cross2(offset, direction) / denominator
    if not (1e-9 < amount < 1.0 - 1e-9 and 1e-9 < other_amount < 1.0 - 1e-9):
        return None
    return (
        amount,
        other_amount,
        (
            start[0] + direction[0] * amount,
            start[1] + direction[1] * amount,
        ),
    )


def signed_polygon_area(points: tuple[tuple[float, float], ...]) -> float:
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
def simple_polygon_indices(points: tuple[tuple[float, float], ...]) -> tuple[int, ...]:
    """Triangulate one simple screen-clockwise contour without an ImGui context."""

    if len(points) < 3:
        return ()
    vertices = list(range(len(points)))
    if signed_polygon_area(points) < 0.0:
        vertices.reverse()
    scale = max(max(abs(value) for point in points for value in point), 1.0)
    epsilon = scale * scale * 1e-10
    reflex = {
        current
        for offset, current in enumerate(vertices)
        if _triangle_cross(
            points[vertices[offset - 1]],
            points[current],
            points[vertices[(offset + 1) % len(vertices)]],
        )
        <= epsilon
    }
    indices: list[int] = []
    while len(vertices) > 3:
        for offset, current in enumerate(vertices):
            if current in reflex:
                continue
            previous = vertices[offset - 1]
            following = vertices[(offset + 1) % len(vertices)]
            a, b, c = points[previous], points[current], points[following]
            cross = _triangle_cross(a, b, c)
            min_x, max_x = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
            min_y, max_y = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
            # The triangle test allows barycentric weights down to -epsilon /
            # cross. Expand the bounds to include that same tolerance region.
            pad_x = 2 * epsilon / cross * (max_x - min_x)
            pad_y = 2 * epsilon / cross * (max_y - min_y)
            min_x, max_x = min_x - pad_x, max_x + pad_x
            min_y, max_y = min_y - pad_y, max_y + pad_y
            if any(
                candidate not in (previous, current, following)
                and min_x <= points[candidate][0] <= max_x
                and min_y <= points[candidate][1] <= max_y
                and _inside_counterclockwise_triangle(points[candidate], a, b, c, epsilon)
                # Only reflex or collinear vertices can obstruct a convex ear.
                for candidate in reflex
            ):
                continue
            indices.extend((previous, current, following))
            del vertices[offset]
            for neighbor_offset in ((offset - 1) % len(vertices), offset % len(vertices)):
                neighbor = vertices[neighbor_offset]
                if (
                    _triangle_cross(
                        points[vertices[neighbor_offset - 1]],
                        points[neighbor],
                        points[vertices[(neighbor_offset + 1) % len(vertices)]],
                    )
                    <= epsilon
                ):
                    reflex.add(neighbor)
                else:
                    reflex.discard(neighbor)
            break
        else:
            raise RuntimeError("polygon contour could not be triangulated")
    indices.extend(vertices)
    return tuple(indices)


def remove_interior_loops(outline):
    """Remove interior loops from an offset stroke, retaining its exterior boundary.

    At tight turns, an inward offset or an inset endpoint cap can fold into
    the stroke. This operation assumes those loops are interior, not disjoint
    filled components or intentional holes.
    """
    # Sweep edge bounds before exact intersection tests. Near edge-on arcs have
    # hundreds of samples but only a few crossing offsets; testing every pair
    # on each camera frame made this repair dominate the UI frame time.
    bounds = [
        (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        for a, b in zip(outline, outline[1:] + outline[:1], strict=True)
    ]
    active = []
    crossing = None
    for index in sorted(range(len(outline)), key=lambda i: bounds[i][0]):
        left, bottom, _right, top = bounds[index]
        active = [other for other in active if bounds[other][2] >= left]
        for other in active:
            if bounds[other][3] < bottom or bounds[other][1] > top:
                continue
            first, second = sorted((index, other))
            if second < first + 2 or (crossing is not None and (first, second) >= crossing[:2]):
                continue
            intersection = segment_intersection(
                outline[first],
                outline[(first + 1) % len(outline)],
                outline[second],
                outline[(second + 1) % len(outline)],
            )
            if intersection is None:
                continue
            crossing = (first, second, intersection[2])
        active.append(index)
    if crossing is not None:
        # Preserve contour-order repair when multiple loops intersect.
        first, second, point = crossing
        inside = (point, *outline[first + 1 : second + 1])
        outside = (*outline[: first + 1], point, *outline[second + 1 :])
        boundary = max((inside, outside), key=lambda loop: abs(signed_polygon_area(loop)))
        return remove_interior_loops(boundary)
    return outline
