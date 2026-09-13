"""Gizmo: projection."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from mojive import math3d
from mojive.commands import (
    SetLight,
    SetSceneCamera,
)
from mojive.interaction.gizmo import (
    RING_SEGMENTS,
    SIZE_PT,
    prepare_projection,
    project,
    rotation_dial,
    world_scale,
)
from mojive.scene.queries import camera_for_node
from mojive.types import CameraView
from mojive.ui.camera import ndc_from_viewport, unproject

if TYPE_CHECKING:
    from mojive.adapters.base import SceneFrame, SceneNode
    from mojive.session import Session


from .state import _FULL_TURN, _JOINT_RANGE_EPSILON, SNAP_TICK_FADE_STEPS, SNAP_TICK_FULL_STEPS


class _RotationDialProjector:
    """Project every active rotation-dial layer through one shared mapping."""

    def __init__(
        self,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        center,
        axis,
        start_direction,
        size_px: float,
    ) -> None:
        self._cam = cam
        self._rect = rect
        self._projection = prepare_projection(cam)
        self._center = np.asarray(center, np.float64)
        self._axis = np.asarray(axis, np.float64)
        self._start_direction = np.asarray(start_direction, np.float64)
        self._world_scale = world_scale(
            cam,
            center,
            rect[3],
            size_px,
            prepared=self._projection,
        )

    def points(self, radius, angles) -> np.ndarray:
        angles = np.atleast_1d(np.asarray(angles, np.float64))
        return project(
            self._cam,
            rotation_dial(
                self._center,
                self._axis,
                self._start_direction,
                self._world_scale,
                radius,
                angles,
            ),
            self._rect,
            prepared=self._projection,
        )

    def tick(
        self,
        radius: float,
        angle: float,
        length_px: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        points = self.points((radius, radius + 1.0 / SIZE_PT), (angle, angle))
        direction = points[1, :2] - points[0, :2]
        projected_length = float(np.linalg.norm(direction))
        if np.any(points[:, 2] <= 0.0) or projected_length < 1e-6:
            return None
        radial = direction / projected_length
        start = points[0, :2]
        return start, start + radial * float(length_px)


class _ScreenRotationDialProjector:
    """Keep the screen-rotation dial identical to its idle pixel-space ring."""

    def __init__(
        self,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        center,
        axis,
        start_direction,
        size_px: float,
    ) -> None:
        self._size_px = float(size_px)
        projection = prepare_projection(cam)
        projected_center = project(cam, (center,), rect, prepared=projection)[0]
        self._center = projected_center[:2]
        self._depth = float(projected_center[2])

        scale = world_scale(cam, center, rect[3], size_px, prepared=projection)
        start_world = rotation_dial(center, axis, start_direction, scale, 1.0, (0.0,))[0]
        quarter_world = rotation_dial(
            center,
            axis,
            start_direction,
            scale,
            1.0,
            (0.5 * np.pi,),
        )[0]
        projected = (
            project(cam, (start_world, quarter_world), rect, prepared=projection)[:, :2]
            - self._center
        )
        start = projected[0]
        length = float(np.linalg.norm(start))
        self._radial = start / length if length > 1e-9 else np.array((1.0, 0.0))
        left = np.array((-self._radial[1], self._radial[0]))
        self._tangent = left if np.dot(left, projected[1]) >= 0.0 else -left

    def points(self, radius, angles) -> np.ndarray:
        angles = np.atleast_1d(np.asarray(angles, np.float64))
        radii = np.broadcast_to(np.asarray(radius, np.float64), angles.shape)
        directions = (
            np.cos(angles)[:, None] * self._radial + np.sin(angles)[:, None] * self._tangent
        )
        screen = self._center + self._size_px * radii[:, None] * directions
        return np.column_stack((screen, np.full(len(screen), self._depth)))

    def tick(
        self,
        radius: float,
        angle: float,
        length_px: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self._depth <= 0.0:
            return None
        start = self.points(radius, (angle,))[0, :2]
        radial = start - self._center
        length = float(np.linalg.norm(radial))
        if length < 1e-9:
            return None
        return start, start + radial / length * float(length_px)


def _rotation_sweep(angle: float) -> float:
    shown = np.radians(round(float(np.degrees(angle)), 1))
    return float(np.copysign(np.fmod(abs(shown), 2.0 * np.pi), shown))


def _shortest_rotation_sweep(angle: float) -> float:
    """Return the equivalent dial sweep in the compact [-pi, pi] interval."""

    wrapped = float((angle + np.pi) % _FULL_TURN - np.pi)
    if abs(wrapped + np.pi) <= _JOINT_RANGE_EPSILON and angle > 0.0:
        return float(np.pi)
    return wrapped


def _rotation_fill_alpha(sweep: float) -> float:
    return 0.24 if abs(float(sweep)) > 1e-6 else 0.0


def _rotation_tick_length_pt(degrees: float) -> float:
    degrees = float(degrees) % 360.0
    rounded = round(degrees)
    if abs(degrees - rounded) < 1e-6 and rounded % 90 == 0:
        return 8.0
    if abs(degrees / 45.0 - round(degrees / 45.0)) < 1e-6:
        return 7.0
    if abs(degrees / 15.0 - round(degrees / 15.0)) < 1e-6:
        return 5.5
    return 4.0


def _clip_line_to_rect(origin, direction, rect) -> tuple[np.ndarray, np.ndarray] | None:
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    if float(np.linalg.norm(direction)) < 1e-6:
        return None
    x, y, w, h = rect
    limits = ((x, x + w), (y, y + h))
    lo, hi = -np.inf, np.inf
    for axis in range(2):
        if abs(direction[axis]) < 1e-9:
            if not limits[axis][0] <= origin[axis] <= limits[axis][1]:
                return None
            continue
        t0 = (limits[axis][0] - origin[axis]) / direction[axis]
        t1 = (limits[axis][1] - origin[axis]) / direction[axis]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if lo > hi:
        return None
    return origin + lo * direction, origin + hi * direction


def _dashed_line_segments(
    start,
    end,
    dash: float,
    gap: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Split one screen segment into center-anchored dash strokes."""

    start = np.asarray(start, np.float64)
    delta = np.asarray(end, np.float64) - start
    length = float(np.linalg.norm(delta))
    dash = max(float(dash), 0.0)
    gap = max(float(gap), 0.0)
    if length < 1e-9 or dash <= 0.0:
        return ()
    direction = delta / length
    stride = dash + gap
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0.0
    while offset < length:
        dash_end = min(length, offset + dash)
        segments.append((start + direction * offset, start + direction * dash_end))
        if stride <= 1e-9:
            break
        offset += stride
    return tuple(segments)


def _clip_segment_to_rect(start, end, rect) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a finite screen segment to a viewport rectangle."""

    start = np.asarray(start, np.float64)
    delta = np.asarray(end, np.float64) - start
    x, y, width, height = rect
    limits = ((float(x), float(x + width)), (float(y), float(y + height)))
    lo, hi = 0.0, 1.0
    for axis in range(2):
        if abs(float(delta[axis])) < 1e-9:
            if not limits[axis][0] <= start[axis] <= limits[axis][1]:
                return None
            continue
        t0 = (limits[axis][0] - start[axis]) / delta[axis]
        t1 = (limits[axis][1] - start[axis]) / delta[axis]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if lo > hi:
        return None
    return start + lo * delta, start + hi * delta


def _project_finite_axis_segment(
    cam: CameraView,
    origin,
    axis,
    extent: float,
    rect,
    *,
    inset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project a finite world axis, clipping its near end and viewport bounds."""

    origin = np.asarray(origin, np.float64).reshape(3)
    axis = np.asarray(axis, np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if length < 1e-9 or extent <= 0.0:
        return None
    axis /= length
    lo, hi = -float(extent), float(extent)
    if cam.projection_blend() < 1.0 - 1e-6:
        forward = np.asarray(cam.forward(), np.float64)
        origin_depth = float(np.dot(origin - np.asarray(cam.eye, np.float64), forward))
        axis_depth = float(np.dot(axis, forward))
        near_depth = max(float(cam.near) * 1.5, 1e-4)
        if origin_depth <= near_depth:
            return None
        if axis_depth > 1e-9:
            lo = max(lo, (near_depth - origin_depth) / axis_depth)
        elif axis_depth < -1e-9:
            hi = min(hi, (near_depth - origin_depth) / axis_depth)
        if lo >= hi:
            return None
    screen = project(cam, (origin + axis * lo, origin + axis * hi), rect)
    if np.any(screen[:, 2] <= 0.0):
        return None
    x, y, width, height = (float(value) for value in rect)
    inset = max(0.0, min(float(inset), 0.5 * min(width, height)))
    clipped_rect = (
        x + inset,
        y + inset,
        max(0.0, width - 2.0 * inset),
        max(0.0, height - 2.0 * inset),
    )
    return _clip_segment_to_rect(screen[0, :2], screen[1, :2], clipped_rect)


def _projected_line_parameters(cam, origin, axis, segment, rect) -> tuple[float, float] | None:
    mvp = np.asarray(cam.proj_matrix(), np.float64) @ np.asarray(cam.view_matrix(), np.float64)
    clip_origin = mvp @ np.append(np.asarray(origin, np.float64), 1.0)
    clip_axis = mvp @ np.append(np.asarray(axis, np.float64), 0.0)
    x, y, width, height = rect
    values = []
    for point in segment:
        ndc = np.array(
            (
                2.0 * (float(point[0]) - x) / width - 1.0,
                1.0 - 2.0 * (float(point[1]) - y) / height,
            )
        )
        denominator = clip_axis[:2] - ndc * clip_axis[3]
        component = int(np.argmax(np.abs(denominator)))
        if abs(denominator[component]) < 1e-10:
            return None
        numerator = ndc[component] * clip_origin[3] - clip_origin[component]
        values.append(float(numerator / denominator[component]))
    return values[0], values[1]


def _snap_tick_alpha(offset_steps: float) -> float:
    distance = abs(float(offset_steps))
    if distance <= SNAP_TICK_FULL_STEPS:
        return 1.0
    if distance >= SNAP_TICK_FADE_STEPS:
        return 0.0
    t = (distance - SNAP_TICK_FULL_STEPS) / (SNAP_TICK_FADE_STEPS - SNAP_TICK_FULL_STEPS)
    return float(1.0 - t * t * (3.0 - 2.0 * t))


def _split_segment_around_interval(
    start,
    end,
    origin,
    direction,
    lower: float,
    upper: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Remove one directed interval from a collinear screen-space segment."""

    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    direction_length = float(np.linalg.norm(direction))
    if direction_length < 1e-9 or lower >= upper:
        return ((start, end),)
    direction /= direction_length
    start_t = float(np.dot(start - origin, direction))
    end_t = float(np.dot(end - origin, direction))
    if start_t > end_t:
        start, end = end, start
        start_t, end_t = end_t, start_t
    segments = []
    if start_t < lower:
        stop_t = min(lower, end_t)
        segments.append((start, origin + direction * stop_t))
    if end_t > upper:
        begin_t = max(upper, start_t)
        segments.append((origin + direction * begin_t, end))
    return tuple(segments)


def _split_segment_around_point(
    start, end, center, radius: float
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    center = np.asarray(center, np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-9 or radius <= 0.0:
        return ((start, end),)
    direction = delta / length
    along = float(np.dot(center - start, direction))
    perpendicular = float(np.linalg.norm(center - (start + along * direction)))
    if perpendicular >= radius:
        return ((start, end),)
    half_gap = float(np.sqrt(radius * radius - perpendicular * perpendicular))
    gap_start = max(0.0, along - half_gap)
    gap_end = min(length, along + half_gap)
    if gap_start >= gap_end:
        return ((start, end),)
    segments = []
    if gap_start > 1e-6:
        segments.append((start, start + direction * gap_start))
    if gap_end < length - 1e-6:
        segments.append((start + direction * gap_end, end))
    return tuple(segments)


def _project_rotation_dial(
    cam: CameraView,
    rect: tuple[float, float, float, float],
    center,
    axis,
    start_direction,
    size_px: float,
    radius: float,
    angles,
) -> np.ndarray:
    """Project one dial through the same camera mapping as the idle gizmo."""
    return _RotationDialProjector(
        cam,
        rect,
        center,
        axis,
        start_direction,
        size_px,
    ).points(radius, angles)


def _project_rotation_tick(
    cam: CameraView,
    rect: tuple[float, float, float, float],
    center,
    axis,
    start_direction,
    size_px: float,
    radius: float,
    angle: float,
    length_px: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project one radial tick and keep its displayed length in pixels."""
    return _RotationDialProjector(
        cam,
        rect,
        center,
        axis,
        start_direction,
        size_px,
    ).tick(radius, angle, length_px)


def _rotation_dial_segments(
    cam: CameraView,
    center,
    axis,
) -> int:
    perspective = np.asarray(center, np.float64) - np.asarray(cam.eye, np.float64)
    perspective /= max(float(np.linalg.norm(perspective)), 1e-12)
    mix = cam.projection_blend()
    view = perspective * (1.0 - mix) + np.asarray(cam.forward(), np.float64) * mix
    view /= np.linalg.norm(view)
    facing = abs(float(np.dot(np.asarray(axis, np.float64), view)))
    multiplier = min(4.0, 1.0 / max(np.sqrt(facing), 0.25))
    return int(np.ceil(RING_SEGMENTS * multiplier))


def _cursor_plane(cam, rect, cursor, point, normal) -> np.ndarray | None:
    ndc = ndc_from_viewport(cursor[0], cursor[1], rect)
    origin, direction = unproject(cam, *ndc)
    den = float(np.dot(direction, normal))
    if abs(den) < 1e-8:
        return None
    t = float(np.dot(np.asarray(point) - origin, normal) / den)
    return np.asarray(origin, np.float64) + np.asarray(direction, np.float64) * t


def _joint_frame_available(frame: SceneFrame, joint_id: int) -> bool:
    """Check availability without constructing a pose for each tool's enabled state."""
    diagnostics = frame.diagnostics
    return (
        frame.qpos is not None
        and diagnostics is not None
        and 0 <= joint_id < len(diagnostics.joint_xpos)
        and joint_id < len(diagnostics.joint_xaxis)
    )


def _basis_from_z(axis) -> np.ndarray:
    # Scalar three-vector arithmetic avoids NumPy dispatch and must not normalize
    # a view into the adapter's published joint-axis buffer in place.
    zx, zy, zz = map(float, np.asarray(axis, np.float64).reshape(3))
    length = math.hypot(zx, zy, zz)
    if length < 1e-9:
        return np.eye(3, dtype=np.float64)
    zx, zy, zz = zx / length, zy / length, zz / length
    xx, xy, xz = (zz, 0.0, -zx) if abs(zz) > 0.9 else (-zy, zx, 0.0)
    length = math.hypot(xx, xy, xz)
    xx, xy, xz = xx / length, xy / length, xz / length
    return np.array(
        (
            (xx, zy * xz - zz * xy, zx),
            (xy, zz * xx - zx * xz, zy),
            (xz, zx * xy - zy * xx, zz),
        ),
        dtype=np.float64,
    )


def _source_light(session: Session, node: SceneNode):
    source = session.source
    if source is None or not 0 <= node.light_index < len(source.lights.lights):
        return None
    return source.lights.lights[node.light_index]


def _set_light_from_world(session: Session, node: SceneNode, position, rotation):
    light = _source_light(session, node)
    if light is None:
        return None
    position = np.asarray(position, np.float64).reshape(3)
    direction = -np.asarray(rotation, np.float64).reshape(3, 3)[:, 2]
    frame = session.frame
    body = int(node.body_index)
    if (
        frame.body_xpos is not None
        and frame.body_xmat is not None
        and 0 <= body < len(frame.body_xpos)
        and body < len(frame.body_xmat)
    ):
        body_position = np.asarray(frame.body_xpos[body], np.float64).reshape(3)
        body_rotation = np.asarray(frame.body_xmat[body], np.float64).reshape(3, 3)
        position = body_rotation.T @ (position - body_position)
        direction = body_rotation.T @ direction
    return SetLight(
        node.light_index,
        replace(
            light,
            position=np.asarray(position, np.float32),
            direction=math3d.normalize(direction),
        ),
    )


def _set_camera_from_world(session: Session, node: SceneNode, position, rotation):
    view = camera_for_node(session, node)
    if view is None or not 0 <= node.camera_index < len(session.cameras):
        return None
    rotation = np.asarray(rotation, np.float64).reshape(3, 3)
    eye = np.asarray(position, np.float32).reshape(3)
    distance = max(view.distance(), 1e-4)
    forward = -rotation[:, 2]
    target = eye + np.asarray(forward * distance, np.float32)
    up = math3d.normalize(rotation[:, 1])
    camera_id = session.cameras[node.camera_index].camera_id
    return SetSceneCamera(camera_id, replace(view, eye=eye, target=target, up=up))
