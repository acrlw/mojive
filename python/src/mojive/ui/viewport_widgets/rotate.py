"""Viewport widgets: rotate."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise

from mojive.drawing.curves import (
    CORNER_SMOOTHING,
    polyline_ribbon,
    smooth_ellipse_stroke,
)
from mojive.drawing.polygons import remove_interior_loops as _remove_ellipse_offset_folds
from mojive.drawing.polygons import segment_intersection as _segment_intersection
from mojive.drawing.polygons import signed_polygon_area as _polygon_area

from .model import (
    _ROTATE_HALF_RINGS,
    TOOL_GLYPH_SCALE,
)


def _counterclockwise(points):
    outline = tuple(points)
    return tuple(reversed(outline)) if _polygon_area(outline) < 0.0 else outline


@lru_cache(maxsize=32)
def _rotate_stroke_outline(
    path: tuple[tuple[float, float], ...],
    width: float,
    cap: str,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[float, float], ...]:
    """Return one inner ring's filled stroke silhouette in authored coordinates."""

    if cap not in {"butt", "round"}:
        raise ValueError(f"unknown rotate ring cap: {cap!r}")
    try:
        return smooth_ellipse_stroke(
            path[0], path[len(path) // 2], width, rounded=cap == "round", smoothing=smoothing
        )
    except ValueError:
        # Wide knockout masks can exceed the ellipse's normal-coordinate
        # reach. Use its miter envelope with folded interior loops removed.
        pass
    left, right, outline = polyline_ribbon(path, width)
    if not outline:
        return ()
    if cap == "butt":
        return _counterclockwise(_remove_ellipse_offset_folds(outline))
    if cap != "round":
        raise ValueError(f"unknown rotate ring cap: {cap!r}")

    cap_segments = 8
    start_direction = (
        path[1][0] - path[0][0],
        path[1][1] - path[0][1],
    )
    end_direction = (
        path[-1][0] - path[-2][0],
        path[-1][1] - path[-2][1],
    )
    start_length = math.hypot(*start_direction)
    end_length = math.hypot(*end_direction)
    start_direction = (start_direction[0] / start_length, start_direction[1] / start_length)
    end_direction = (end_direction[0] / end_length, end_direction[1] / end_length)
    start_normal = (-start_direction[1], start_direction[0])
    end_normal = (-end_direction[1], end_direction[0])
    radius = width * 0.5

    rounded = list(left)
    for index in range(1, cap_segments + 1):
        angle = math.pi * index / cap_segments
        rounded.append(
            (
                path[-1][0]
                + radius * (math.cos(angle) * end_normal[0] + math.sin(angle) * end_direction[0]),
                path[-1][1]
                + radius * (math.cos(angle) * end_normal[1] + math.sin(angle) * end_direction[1]),
            )
        )
    rounded.extend(reversed(right[:-1]))
    for index in range(1, cap_segments):
        angle = math.pi * index / cap_segments
        rounded.append(
            (
                path[0][0]
                + radius
                * (-math.cos(angle) * start_normal[0] - math.sin(angle) * start_direction[0]),
                path[0][1]
                + radius
                * (-math.cos(angle) * start_normal[1] - math.sin(angle) * start_direction[1]),
            )
        )
    return _counterclockwise(_remove_ellipse_offset_folds(rounded))


def _point_in_polygon(point, polygon) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (previous[0] - current[0]) * (y - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _polygon_difference(subject, clip):
    """Return simple boundaries for ``subject`` minus one crossing clip polygon."""

    subject_breaks = [[] for _ in subject]
    clip_breaks = [[] for _ in clip]
    for subject_index, start in enumerate(subject):
        end = subject[(subject_index + 1) % len(subject)]
        for clip_index, clip_start in enumerate(clip):
            clip_end = clip[(clip_index + 1) % len(clip)]
            intersection = _segment_intersection(start, end, clip_start, clip_end)
            if intersection is None:
                continue
            amount, clip_amount, point = intersection
            subject_breaks[subject_index].append((amount, point))
            clip_breaks[clip_index].append((clip_amount, point))

    if not any(subject_breaks):
        return () if _point_in_polygon(subject[0], clip) else (_counterclockwise(subject),)

    edges = []

    def append_boundary(polygon, breaks, other, *, keep_inside: bool, reverse: bool) -> None:
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            cuts = [(0.0, start), *breaks[index], (1.0, end)]
            cuts.sort(key=lambda item: item[0])
            for (_, first), (_, second) in pairwise(cuts):
                midpoint = (
                    (first[0] + second[0]) * 0.5,
                    (first[1] + second[1]) * 0.5,
                )
                if _point_in_polygon(midpoint, other) != keep_inside:
                    continue
                edges.append((second, first) if reverse else (first, second))

    # Keep subject edges outside the shell. Shell edges inside the subject are
    # traversed backwards so the remaining visible region stays on the left.
    append_boundary(subject, subject_breaks, clip, keep_inside=False, reverse=False)
    append_boundary(clip, clip_breaks, subject, keep_inside=True, reverse=True)

    def key(point) -> tuple[float, float]:
        return round(point[0], 8), round(point[1], 8)

    outgoing = {}
    for index, (start, _end) in enumerate(edges):
        outgoing.setdefault(key(start), []).append(index)

    unused = set(range(len(edges)))
    polygons = []
    while unused:
        edge_index = next(iter(unused))
        start_key = key(edges[edge_index][0])
        points = []
        while True:
            unused.remove(edge_index)
            start, end = edges[edge_index]
            if not points:
                points.append(start)
            points.append(end)
            end_key = key(end)
            if end_key == start_key:
                break
            candidates = [
                candidate for candidate in outgoing.get(end_key, ()) if candidate in unused
            ]
            if len(candidates) != 1:
                raise RuntimeError("rotate shell subtraction produced an open boundary")
            edge_index = candidates[0]
        points.pop()
        if len(points) >= 3:
            polygons.append(_counterclockwise(points))
    return tuple(polygons)


@lru_cache(maxsize=32)
def _rotate_visible_ring_polygons(
    tool_stroke: float,
    ring_gap_ratio: float,
    cap: str,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[tuple[tuple[float, float], ...], ...], ...]:
    """Subtract each front shell from the ring behind it for cyclic occlusion."""

    local_width = tool_stroke / TOOL_GLYPH_SCALE
    shell_width = tool_stroke * (1.0 + 2.0 * ring_gap_ratio) / TOOL_GLYPH_SCALE
    # Path order is Y, X, Z. These occluders produce Y > X, X > Z, Z > Y.
    occluder_by_ring = (2, 0, 1)
    result = []
    for index, path in enumerate(_ROTATE_HALF_RINGS):
        outline = _rotate_stroke_outline(path, local_width, cap, smoothing)
        shell = _rotate_stroke_outline(
            _ROTATE_HALF_RINGS[occluder_by_ring[index]],
            shell_width,
            cap,
            smoothing,
        )
        result.append(_polygon_difference(outline, shell))
    return tuple(result)
