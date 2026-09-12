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
            raise RuntimeError("polygon contour could not be triangulated")
    indices.extend(vertices)
    return tuple(indices)


def remove_interior_loops(outline):
    """Remove interior loops from an offset stroke, retaining its exterior boundary.

    At tight turns, an inward offset or an inset endpoint cap can fold into
    the stroke. This operation assumes those loops are interior, not disjoint
    filled components or intentional holes.
    """
    for first in range(len(outline)):
        for second in range(first + 2, len(outline)):
            intersection = segment_intersection(
                outline[first],
                outline[(first + 1) % len(outline)],
                outline[second],
                outline[(second + 1) % len(outline)],
            )
            if intersection is None:
                continue
            point = intersection[2]
            inside = (point, *outline[first + 1 : second + 1])
            outside = (*outline[: first + 1], point, *outline[second + 1 :])
            boundary = max((inside, outside), key=lambda loop: abs(signed_polygon_area(loop)))
            return remove_interior_loops(boundary)
    return outline
