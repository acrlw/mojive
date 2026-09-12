"""Gizmo: joint labels."""

from __future__ import annotations

import numpy as np

from mojive.interaction.gizmo import RING_HIT_PT, GizmoHandle, screen_path_distance
from mojive.ui.draw2d import Draw2D
from mojive.ui.theme import THEME

from .state import (
    JOINT_LIMIT_HIT_PADDING_PT,
    JOINT_LIMIT_HIT_PT,
    JOINT_LIMIT_OUTER_HIT_FRACTION,
    JOINT_LIMIT_STROKE_HIT_PADDING_PT,
    JOINT_SLIDE_AXIS_HIT_PT,
    JointLimitHit,
    _HingeRangeProjection,
    _JointRangeState,
    _screen_segment_distance,
    _SlideRangeProjection,
)


def _draw_joint_value_label(
    overlay: Draw2D,
    anchor,
    semantic_color,
    label: str,
    style_scale: float,
    *,
    above: bool,
    align_right: bool,
    centered: bool = False,
) -> tuple[float, float, float, float]:
    """Draw a translucent semantic-dot label without coloring the value text."""

    measured = overlay.text_size(label)
    if measured is None:  # permissive for recording/test Draw2D adapters
        text_width, text_height = len(label) * 8.0 * style_scale, 14.0 * style_scale
    else:
        text_width, text_height = measured
    padding_x = 8.0 * style_scale
    padding_y = 5.0 * style_scale
    dot_radius = 3.0 * style_scale
    dot_gap = 6.0 * style_scale
    width = padding_x * 2.0 + dot_radius * 2.0 + dot_gap + text_width
    height = max(26.0 * style_scale, text_height + padding_y * 2.0)
    margin = 8.0 * style_scale
    if centered:
        x = float(anchor[0]) - width * 0.5
    else:
        x = float(anchor[0]) - width - margin if align_right else float(anchor[0]) + margin
    y = float(anchor[1]) - height - margin if above else float(anchor[1]) + margin
    overlay.rect_filled(
        (x, y),
        (x + width, y + height),
        (*THEME.bg_popup[:3], 0.92),
        rounding=3.0 * style_scale,
    )
    overlay.rect(
        (x, y),
        (x + width, y + height),
        THEME.border,
        1.0 * style_scale,
        rounding=3.0 * style_scale,
    )
    center_y = y + height * 0.5
    overlay.circle_filled(
        (x + padding_x + dot_radius, center_y),
        dot_radius,
        semantic_color,
        segments=16,
    )
    overlay.text(
        (
            x + padding_x + dot_radius * 2.0 + dot_gap,
            y + (height - text_height) * 0.5,
        ),
        THEME.text,
        label,
    )
    return (x, y, x + width, y + height)


def _point_in_rect(point, rect: tuple[float, float, float, float]) -> bool:
    x, y = (float(value) for value in point)
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _joint_range_handle(state: _JointRangeState | None) -> GizmoHandle:
    if state is not None and state.joint_type == "slide":
        return GizmoHandle.Z
    return GizmoHandle.ROTATE_Z


def _hinge_range_hit(
    cursor,
    projection: _HingeRangeProjection | None,
    style_scale: float,
) -> bool:
    return _hinge_range_path_hit(cursor, projection, style_scale) or _hinge_current_tick_hit(
        cursor, projection, style_scale
    )


def _hinge_range_path_hit(
    cursor,
    projection: _HingeRangeProjection | None,
    style_scale: float,
) -> bool:
    if projection is None:
        return False
    distance = float("inf")
    if projection.allowed is not None:
        distance = screen_path_distance(
            cursor,
            projection.allowed,
            closed=projection.full_range,
        )
    if projection.complement is not None:
        distance = min(distance, screen_path_distance(cursor, projection.complement, closed=False))
    return distance <= RING_HIT_PT * style_scale


def _hinge_current_tick_hit(
    cursor,
    projection: _HingeRangeProjection | None,
    style_scale: float,
) -> bool:
    return bool(
        projection is not None
        and projection.current_tick is not None
        and _screen_segment_distance(cursor, *projection.current_tick) <= RING_HIT_PT * style_scale
    )


def _slide_range_hit(
    cursor,
    projection: _SlideRangeProjection | None,
    style_scale: float,
) -> bool:
    return _slide_range_path_hit(cursor, projection, style_scale) or _slide_current_tick_hit(
        cursor, projection, style_scale
    )


def _slide_range_path_hit(
    cursor,
    projection: _SlideRangeProjection | None,
    style_scale: float,
) -> bool:
    if projection is None:
        return False
    distance = _screen_segment_distance(cursor, projection.lower, projection.upper)
    return distance <= JOINT_SLIDE_AXIS_HIT_PT * style_scale


def _slide_current_tick_hit(
    cursor,
    projection: _SlideRangeProjection | None,
    style_scale: float,
) -> bool:
    if projection is None:
        return False
    half_tick = 10.0 * style_scale
    distance = _screen_segment_distance(
        cursor,
        projection.current - projection.normal * half_tick,
        projection.current + projection.normal * half_tick,
    )
    return distance <= JOINT_SLIDE_AXIS_HIT_PT * style_scale


def _same_joint_limit(a: JointLimitHit, b: JointLimitHit) -> bool:
    return a.joint_id == b.joint_id and a.qpos_adr == b.qpos_adr and a.label[:3] == b.label[:3]


def _closest_joint_limit_hit(
    cursor,
    hits: tuple[JointLimitHit, ...],
    state: _JointRangeState,
    style_scale: float,
) -> JointLimitHit | None:
    """Pick the nearest visible outer tick stroke, never its bounding box."""

    point = np.asarray(cursor, np.float64)
    candidates: list[tuple[float, float, JointLimitHit]] = []
    for hit in hits:
        if hit.joint_id != state.joint_id or hit.qpos_adr != state.qpos_adr:
            continue
        start = np.asarray(hit.tick_start, np.float64)
        end = np.asarray(hit.tick_end, np.float64)
        edge = end - start
        denominator = float(np.dot(edge, edge))
        if denominator <= 1e-12:
            continue
        along = float(np.dot(point - start, edge) / denominator)
        if state.joint_type == "hinge" and along < JOINT_LIMIT_OUTER_HIT_FRACTION:
            continue
        if state.joint_type == "slide" and abs(along - 0.5) < (
            0.5 - JOINT_LIMIT_OUTER_HIT_FRACTION
        ):
            continue
        along = float(np.clip(along, 0.0, 1.0))
        closest = start + edge * along
        distance = float(np.linalg.norm(point - closest))
        threshold = hit.tick_width * 0.5 + JOINT_LIMIT_STROKE_HIT_PADDING_PT * style_scale
        if distance <= threshold:
            candidates.append((distance, float(np.linalg.norm(point - end)), hit))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2].label))[2]


def _joint_limit_tick_rect(
    tick,
    tick_width: float,
    style_scale: float,
) -> tuple[float, float, float, float]:
    """Return a scale-stable pointer target centered on one endpoint tick."""

    start, end = tick
    padding = JOINT_LIMIT_HIT_PADDING_PT * style_scale + tick_width * 0.5
    x0 = min(float(start[0]), float(end[0])) - padding
    y0 = min(float(start[1]), float(end[1])) - padding
    x1 = max(float(start[0]), float(end[0])) + padding
    y1 = max(float(start[1]), float(end[1])) + padding
    minimum = JOINT_LIMIT_HIT_PT * style_scale
    if x1 - x0 < minimum:
        center = (x0 + x1) * 0.5
        x0, x1 = center - minimum * 0.5, center + minimum * 0.5
    if y1 - y0 < minimum:
        center = (y0 + y1) * 0.5
        y0, y1 = center - minimum * 0.5, center + minimum * 0.5
    return x0, y0, x1, y1


def _joint_limit_label(prefix: str, value: float, joint_type: str) -> str:
    if joint_type == "hinge":
        return f"{prefix} {np.degrees(value):+.1f}°"
    return f"{prefix} {value:+.3f} m"
