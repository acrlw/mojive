"""Shared authored geometry for icon painters and viewport controls.

This module owns shapes and their dimensions without importing UI controls or icon submission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np

from mojive.geometry2d.curves import (
    CORNER_SMOOTHING,
    clip_polygon_rect,
    offset_closed_path,
    polyline_ribbon,
    smooth_ellipse_stroke,
    smooth_line_cap,
    smooth_rect_points,
)
from mojive.geometry2d.polygons import remove_interior_loops as _remove_ellipse_offset_folds
from mojive.geometry2d.polygons import segment_intersection as _segment_intersection
from mojive.geometry2d.polygons import signed_polygon_area as _polygon_area

CAPSULE_SMOOTHING = 0.382


@dataclass(frozen=True)
class OverlayGeometry:
    """Shared logical-pixel geometry for viewport chrome and its design probe."""

    icon_radius: float = 8.0
    radial_step: float = 6.0
    center_step: float = 34.0
    tool_center_step: float = 34.0
    # Measured by the capsule probe at CAPSULE_SMOOTHING; scales with the shell.
    end_padding_ratio: float = 1.0176593363285065
    tool_group_gap: float = 10.0
    divider_width: float = 20.0
    tool_stroke: float = 1.46
    rotate_ring_gap_ratio: float = 1.0
    rotate_ring_cap: str = "round"
    hint_control_height: float = 18.0
    hint_padding_x: float = 16.0
    hint_padding_y: float = 8.0
    hint_input_gap: float = 8.0
    hint_group_gap: float = 24.0
    hint_chord_gap: float = 10.0
    hint_key_padding_x: float = 8.0
    hint_mouse_width: float = 14.0
    hint_mouse_stroke: float = 1.0
    hint_mouse_button_width_ratio: float = 0.40
    hint_mouse_button_shell_ratio: float = 1.25
    hint_mouse_button_height_ratio: float = 0.40
    hint_mouse_wheel_width_ratio: float = 0.32
    hint_mouse_wheel_height_ratio: float = 0.40
    hint_mouse_wheel_gap_ratio: float = 1.0
    frame_center_radius: float = 1.45
    frame_center_gap_ratio: float = 1.4
    tooltip_padding_x: float = 7.0
    tooltip_padding_y: float = 4.0

    @property
    def state_radius(self) -> float:
        return self.icon_radius + self.radial_step

    @property
    def shell_radius(self) -> float:
        return self.state_radius + self.radial_step

    @property
    def end_padding(self) -> float:
        return self.shell_radius * self.end_padding_ratio

    @property
    def rotate_ring_gap(self) -> float:
        return self.tool_stroke * self.rotate_ring_gap_ratio


OVERLAY_GEOMETRY = OverlayGeometry()

TOOL_GLYPH_SCALE = 1.18

DEFAULT_RESET_HEAD_SCALE = 1.5

_ROTATE_HALF_RINGS = (
    (
        (3.177, 4.765),
        (3.488, 4.386),
        (3.740, 3.931),
        (3.928, 3.410),
        (4.048, 2.830),
        (4.099, 2.201),
        (4.080, 1.535),
        (3.992, 0.843),
        (3.835, 0.136),
        (3.612, -0.573),
        (3.328, -1.272),
        (2.986, -1.950),
        (2.594, -2.594),
        (2.157, -3.194),
        (1.683, -3.739),
        (1.181, -4.220),
        (0.658, -4.629),
        (0.124, -4.959),
        (-0.412, -5.204),
        (-0.941, -5.360),
        (-1.454, -5.424),
        (-1.942, -5.395),
        (-2.397, -5.274),
        (-2.811, -5.063),
        (-3.177, -4.765),
    ),
    (
        (-3.177, 4.765),
        (-3.488, 4.386),
        (-3.740, 3.931),
        (-3.928, 3.410),
        (-4.048, 2.830),
        (-4.099, 2.201),
        (-4.080, 1.535),
        (-3.992, 0.843),
        (-3.835, 0.136),
        (-3.612, -0.573),
        (-3.328, -1.272),
        (-2.986, -1.950),
        (-2.594, -2.594),
        (-2.157, -3.194),
        (-1.683, -3.739),
        (-1.181, -4.220),
        (-0.658, -4.629),
        (-0.124, -4.959),
        (0.412, -5.204),
        (0.941, -5.360),
        (1.454, -5.424),
        (1.942, -5.395),
        (2.397, -5.274),
        (2.811, -5.063),
        (3.177, -4.765),
    ),
    (
        (5.800, 0.000),
        (5.750, 0.379),
        (5.602, 0.751),
        (5.359, 1.110),
        (5.023, 1.450),
        (4.601, 1.765),
        (4.101, 2.051),
        (3.531, 2.301),
        (2.900, 2.511),
        (2.220, 2.679),
        (1.501, 2.801),
        (0.757, 2.875),
        (0.000, 2.900),
        (-0.757, 2.875),
        (-1.501, 2.801),
        (-2.220, 2.679),
        (-2.900, 2.511),
        (-3.531, 2.301),
        (-4.101, 2.051),
        (-4.601, 1.765),
        (-5.023, 1.450),
        (-5.359, 1.110),
        (-5.602, 0.751),
        (-5.750, 0.379),
        (-5.800, 0.000),
    ),
)


@dataclass(frozen=True)
class MouseButtonGeometry:
    """A highlighted button and the mouse-shell path visible around it."""

    visible_shell: tuple[tuple[float, float], ...]
    fill: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MouseWheelGeometry:
    """A highlighted wheel separated from the shell by a physical-pixel gap."""

    lo: tuple[float, float]
    hi: tuple[float, float]
    rounding: float
    gap: float


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


def _points_in_polygon(points, polygon):
    """Classify a batch of boundary midpoints with the same even/odd ray rule."""
    points, current = np.asarray(points), np.asarray(polygon)
    previous = np.roll(current, 1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    dy = previous[:, 1] - current[:, 1]
    crossing_x = (
        np.divide(
            (previous[:, 0] - current[:, 0]) * (y - current[:, 1]),
            dy,
            out=np.zeros((len(points), len(current))),
            where=dy != 0,
        )
        + current[:, 0]
    )
    crosses = ((current[:, 1] > y) != (previous[:, 1] > y)) & (x < crossing_x)
    return np.count_nonzero(crosses, axis=1) % 2 != 0


def _point_in_polygon(point, polygon) -> bool:
    return bool(_points_in_polygon((point,), polygon)[0])


def _polygon_difference(subject, clip):
    """Return simple boundaries for ``subject`` minus one crossing clip polygon."""

    subject_breaks = [[] for _ in subject]
    clip_breaks = [[] for _ in clip]
    subject_points, clip_points = np.asarray(subject), np.asarray(clip)
    subject_next, clip_next = np.roll(subject_points, -1, axis=0), np.roll(clip_points, -1, axis=0)
    subject_lo, subject_hi = (
        np.minimum(subject_points, subject_next),
        np.maximum(subject_points, subject_next),
    )
    clip_lo, clip_hi = np.minimum(clip_points, clip_next), np.maximum(clip_points, clip_next)
    candidates = np.all((subject_lo[:, None] <= clip_hi) & (clip_lo <= subject_hi[:, None]), axis=2)
    for subject_index, clip_index in zip(*np.nonzero(candidates), strict=True):
        intersection = _segment_intersection(
            subject[subject_index],
            subject[(subject_index + 1) % len(subject)],
            clip[clip_index],
            clip[(clip_index + 1) % len(clip)],
        )
        if intersection is None:
            continue
        amount, clip_amount, point = intersection
        subject_breaks[subject_index].append((amount, point))
        clip_breaks[clip_index].append((clip_amount, point))

    if not any(subject_breaks):
        return () if _points_in_polygon((subject[0],), clip)[0] else (_counterclockwise(subject),)

    edges = []

    def append_boundary(polygon, breaks, other, *, keep_inside: bool, reverse: bool) -> None:
        parts = []
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            cuts = [(0.0, start), *breaks[index], (1.0, end)]
            cuts.sort(key=lambda item: item[0])
            for (_, first), (_, second) in pairwise(cuts):
                parts.append((first, second))
        midpoints = np.asarray(parts).mean(axis=1)
        for (first, second), inside in zip(
            parts, _points_in_polygon(midpoints, other), strict=True
        ):
            if inside == keep_inside:
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


@lru_cache(maxsize=128)
def mouse_button_geometry(
    x: float,
    y: float,
    width: float,
    height: float,
    button: str,
    *,
    outline_width: float,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
    smoothing: float = CORNER_SMOOTHING,
) -> MouseButtonGeometry | None:
    """Return true-knockout shell and fill geometry for one mouse button."""

    if button not in {"left", "right"}:
        return None
    half_stroke = outline_width * 0.5
    shell_gap = outline_width * geometry.hint_mouse_button_shell_ratio
    button_bottom = y + height * geometry.hint_mouse_button_height_ratio
    shell_radius = min(width * 0.22, height * 0.18)
    outer_left = x - half_stroke
    button_width = (width + outline_width) * geometry.hint_mouse_button_width_ratio
    inner_edge = outer_left + button_width
    shell = smooth_rect_points(x, y, x + width, y + height, shell_radius, smoothing=smoothing)
    outer = offset_closed_path(shell, half_stroke)
    fill = clip_polygon_rect(outer, (outer_left, y - half_stroke, inner_edge, button_bottom))
    other_corners = smooth_rect_points(
        x, y, x + width, y + height, shell_radius, (False, True, True, True), smoothing=smoothing
    )
    visible_shell = (
        (inner_edge + shell_gap, y),
        *other_corners[1:],
        (x, button_bottom + shell_gap),
    )
    if button == "right":
        mirror_x = x * 2.0 + width
        visible_shell = tuple((mirror_x - point[0], point[1]) for point in visible_shell)
        fill = tuple((mirror_x - point[0], point[1]) for point in fill)
    fill = _counterclockwise(fill)
    return MouseButtonGeometry(visible_shell, fill)


@lru_cache(maxsize=128)
def mouse_wheel_geometry(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    outline_width: float,
    pixel_size: float,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
) -> MouseWheelGeometry:
    """Return wheel geometry with a scalable gap and one-pixel minimum."""

    gap = max(
        outline_width * geometry.hint_mouse_wheel_gap_ratio,
        max(float(pixel_size), 1e-6),
    )
    top = y + outline_width * 0.5 + gap
    wheel_width = width * geometry.hint_mouse_wheel_width_ratio
    wheel_height = height * geometry.hint_mouse_wheel_height_ratio
    center_x = x + width * 0.5
    return MouseWheelGeometry(
        (center_x - wheel_width * 0.5, top),
        (center_x + wheel_width * 0.5, top + wheel_height),
        wheel_width * 0.42,
        gap,
    )


@lru_cache(maxsize=64)
def _snap_glyph_shape(scale: float, smoothing: float = CORNER_SMOOTHING):
    radius = 6.4 * scale
    cap = smooth_line_cap((0.0, 0.0), (0.0, 1.0), 2.0 * radius, smoothing=smoothing)
    return ((-radius, -6.2 * scale), *map(tuple, cap.tolist()), (radius, -6.2 * scale))
