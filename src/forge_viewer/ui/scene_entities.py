"""Editor helpers for selecting and inspecting scene cameras and lights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import math3d
from ..adapters.base import NodeType
from ..gizmo import project, world_scale
from ..render.debugdraw import Occlusion
from ..types import CameraView, Light, LightType
from .camera import camera_basis

HELPER_LAYER = "ui.scene_entities"
CAMERA_COLOR = np.array((0.34, 0.72, 1.0, 0.95), np.float32)
LIGHT_COLOR = np.array((1.0, 0.76, 0.24, 0.95), np.float32)
SELECTED_COLOR = np.array((1.0, 0.58, 0.12, 1.0), np.float32)
ANCHOR_RADIUS_PT = 6.0
PICK_RADIUS_PT = 13.0
_CIRCLE_SEGMENTS = 48
_SPOT_HELPER_LENGTH_PT = 180.0
_SPOT_HELPER_RADIUS_PT = 120.0


@dataclass
class SceneEntityHelpers:
    """Publishes editor-only helpers and resolves their viewport hit targets."""

    visible: bool = True
    show_influence: bool = True

    def publish(
        self,
        backend,
        session,
        camera: CameraView,
        viewport_height: float,
        ui_scale: float,
        view_through_camera: bool = False,
        selected_camera_aspect: float | None = None,
    ) -> None:
        debug = getattr(backend, "debug", None)
        if debug is None:
            return
        layer = debug.layer(HELPER_LAYER, Occlusion.GHOST)
        layer.clear()
        if view_through_camera or not self.visible or session.source is None:
            return

        selected = session.selected
        camera_helpers: list[tuple[int, CameraView, bool]] = []
        for node in session.nodes:
            if node.type is NodeType.CAMERA and (view := _camera_view(session, node)) is not None:
                if selected == node.object_id and selected_camera_aspect is not None:
                    view = view.with_aspect(selected_camera_aspect)
                camera_helpers.append((node.object_id, view, selected == node.object_id))
        self._cameras(layer, camera_helpers, camera, viewport_height, ui_scale)

        frame = session.frame
        light_set = frame.lights if frame.lights is not None else session.source.lights
        for node in session.nodes:
            if node.type is NodeType.LIGHT and 0 <= node.light_index < len(light_set.lights):
                light = light_set.lights[node.light_index]
                if light.active and light.type is not LightType.IMAGE:
                    self._light(
                        layer,
                        node.object_id,
                        light,
                        selected == node.object_id,
                        camera,
                        viewport_height,
                        ui_scale,
                    )

    def pick(
        self,
        session,
        camera: CameraView,
        rect: tuple[float, float, float, float],
        cursor: tuple[float, float],
        style_scale: float,
        view_through_camera: bool = False,
    ) -> int:
        if view_through_camera or not self.visible or session.source is None:
            return 0
        anchors: list[tuple[int, np.ndarray]] = []
        lights = session.frame.lights or session.source.lights
        for node in session.nodes:
            if node.type is NodeType.CAMERA and (view := _camera_view(session, node)) is not None:
                anchors.append((node.object_id, np.asarray(view.eye, np.float64)))
            elif node.type is NodeType.LIGHT and 0 <= node.light_index < len(lights.lights):
                light = lights.lights[node.light_index]
                if light.active and light.type is not LightType.IMAGE:
                    anchors.append((node.object_id, np.asarray(light.position, np.float64)))
        if not anchors:
            return 0
        screen = project(camera, [anchor for _, anchor in anchors], rect)
        cursor_xy = np.asarray(cursor, np.float64)
        distance2 = np.sum((screen[:, :2] - cursor_xy) ** 2, axis=1)
        distance2[screen[:, 2] <= 0.0] = np.inf
        index = int(np.argmin(distance2))
        radius = PICK_RADIUS_PT * float(style_scale)
        return int(anchors[index][0]) if distance2[index] <= radius * radius else 0

    def _cameras(
        self,
        layer,
        helpers: list[tuple[int, CameraView, bool]],
        editor_camera: CameraView,
        viewport_height: float,
        ui_scale: float,
    ) -> None:
        if not helpers:
            return
        views = [view for _, view, _ in helpers]
        colors = np.asarray(
            [SELECTED_COLOR if selected else CAMERA_COLOR for _, _, selected in helpers],
            np.float32,
        )
        layer.points(
            "cameras:anchors",
            np.asarray([view.eye for view in views], np.float32),
            colors,
            ANCHOR_RADIUS_PT * ui_scale,
        )
        starts, ends = compact_camera_segments(views, editor_camera, viewport_height, 26.0)
        layer.lines(
            "cameras:icons",
            starts,
            ends,
            np.repeat(colors, 8, axis=0),
            1.6 * ui_scale,
        )
        if self.show_influence:
            for object_id, view, selected in helpers:
                if selected:
                    starts, ends = camera_frustum_segments(view)
                    layer.lines(
                        f"camera:{object_id}:frustum",
                        starts,
                        ends,
                        SELECTED_COLOR,
                        1.6 * ui_scale,
                    )

    def _light(
        self,
        layer,
        object_id: int,
        light: Light,
        selected: bool,
        editor_camera: CameraView,
        viewport_height: float,
        ui_scale: float,
    ) -> None:
        ident = f"light:{object_id}"
        color = SELECTED_COLOR if selected else LIGHT_COLOR
        position = np.asarray(light.position, np.float32)
        layer.point(f"{ident}:anchor", position, color, ANCHOR_RADIUS_PT * ui_scale)
        icon_size = world_scale(editor_camera, position, viewport_height, 30.0)
        direction = _direction(light.direction)
        if light.type not in (LightType.POINT, LightType.IMAGE):
            layer.arrow(
                f"{ident}:direction",
                position,
                position + direction * icon_size,
                color,
                1.8 * ui_scale,
                start_mask_px=ANCHOR_RADIUS_PT * ui_scale,
            )
        if not selected or not self.show_influence:
            return
        if light.type is LightType.POINT and light.range > 0.0:
            starts, ends = sphere_segments(position, light.range)
            layer.lines(f"{ident}:range", starts, ends, color, 1.2 * ui_scale)
        elif light.type is LightType.SPOT and light.range > 0.0:
            length = spot_helper_length(light, editor_camera, viewport_height)
            if length > 0.0:
                starts, ends = spot_cone_segments(light, length)
                layer.lines(f"{ident}:range", starts, ends, color, 1.4 * ui_scale)
        elif light.type is LightType.AREA and light.area_radius > 0.0:
            points = oriented_circle(position, direction, light.area_radius)
            layer.polyline(f"{ident}:area", points, color, 1.4 * ui_scale, closed=True)


def camera_rotation(view: CameraView) -> np.ndarray:
    right, up, forward = camera_basis(view)
    return np.column_stack((right, up, -forward)).astype(np.float32)


def direction_basis(direction) -> np.ndarray:
    forward = _direction(direction)
    reference = np.array((0.0, 0.0, 1.0))
    if abs(float(np.dot(forward, reference))) > 0.95:
        reference = np.array((0.0, 1.0, 0.0))
    right = math3d.normalize(np.cross(forward, reference))
    up = math3d.normalize(np.cross(right, forward))
    return np.column_stack((right, up, -forward)).astype(np.float32)


def camera_frustum_segments(view: CameraView) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(
        np.asarray(view.proj_matrix(), np.float64) @ np.asarray(view.view_matrix(), np.float64)
    )
    corners = []
    for z in (-1.0, 1.0):
        for x, y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            world = inverse @ np.array((x, y, z, 1.0), np.float64)
            corners.append(world[:3] / world[3])
    points = np.asarray(corners, np.float32)
    edges = np.asarray(
        (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ),
        np.intp,
    )
    return points[edges[:, 0]], points[edges[:, 1]]


def compact_camera_segments(
    views: list[CameraView],
    editor_camera: CameraView,
    viewport_height: float,
    pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build compact camera icons for one debug-draw batch."""
    if not views:
        empty = np.empty((0, 3), np.float32)
        return empty, empty

    eyes = np.asarray([view.eye for view in views], np.float64)
    targets = np.asarray([view.target for view in views], np.float64)
    ups = np.asarray([view.up for view in views], np.float64)
    forward = _normalize_rows(targets - eyes, (0.0, 0.0, -1.0))
    right = np.cross(forward, ups)
    right_norm = np.linalg.norm(right, axis=1)
    degenerate = right_norm <= 1e-9
    if np.any(degenerate):
        reference = np.zeros((int(np.count_nonzero(degenerate)), 3), np.float64)
        reference[:, 2] = 1.0
        vertical = np.abs(forward[degenerate, 2]) >= 0.95
        reference[vertical] = (0.0, 1.0, 0.0)
        right[degenerate] = np.cross(forward[degenerate], reference)
    right = _normalize_rows(right, (1.0, 0.0, 0.0))
    corrected_up = np.cross(right, forward)

    height = max(float(viewport_height), 1.0)
    projection = np.asarray(editor_camera.proj_matrix(), np.float64)
    mvp = projection @ np.asarray(editor_camera.view_matrix(), np.float64)
    clip_w = np.column_stack((eyes, np.ones(len(eyes), np.float64))) @ mvp[3]
    p11 = float(projection[1, 1])
    lengths = np.zeros(len(eyes), np.float64)
    if abs(p11) >= 1e-9:
        visible = clip_w > 0.0
        lengths[visible] = 2.0 * clip_w[visible] * float(pixels) / (p11 * height)

    centers = eyes + forward * lengths[:, None]
    half_height = lengths * 0.45
    half_width = half_height * np.clip(
        np.asarray([view.aspect for view in views], np.float64), 0.75, 1.8
    )
    horizontal = right * half_width[:, None]
    vertical = corrected_up * half_height[:, None]
    corners = np.stack(
        (
            centers - horizontal - vertical,
            centers + horizontal - vertical,
            centers + horizontal + vertical,
            centers - horizontal + vertical,
        ),
        axis=1,
    )
    starts = np.concatenate((np.repeat(eyes[:, None], 4, axis=1), corners), axis=1)
    ends = np.concatenate((corners, np.roll(corners, -1, axis=1)), axis=1)
    return starts.reshape(-1, 3).astype(np.float32), ends.reshape(-1, 3).astype(np.float32)


def _normalize_rows(values: np.ndarray, fallback) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    result = np.empty_like(values)
    valid = lengths > 1e-9
    result[valid] = values[valid] / lengths[valid, None]
    result[~valid] = fallback
    return result


def sphere_segments(center, radius: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, np.float32)
    angle = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_SEGMENTS, endpoint=False)
    rings = []
    for axes in ((0, 1), (1, 2), (2, 0)):
        points = np.repeat(center[None], _CIRCLE_SEGMENTS, axis=0)
        points[:, axes[0]] += np.cos(angle) * float(radius)
        points[:, axes[1]] += np.sin(angle) * float(radius)
        rings.append(points)
    points = np.concatenate(rings, axis=0)
    return points, np.concatenate([np.roll(ring, -1, axis=0) for ring in rings], axis=0)


def spot_helper_length(light: Light, camera: CameraView, viewport_height: float) -> float:
    """Bound the editor-only cone without changing the light's physical range."""
    axial_limit = world_scale(camera, light.position, viewport_height, _SPOT_HELPER_LENGTH_PT)
    if axial_limit <= 0.0:
        return 0.0
    radial_limit = world_scale(camera, light.position, viewport_height, _SPOT_HELPER_RADIUS_PT)
    tangent = np.tan(np.deg2rad(np.clip(float(light.cutoff), 0.1, 89.0)))
    radial_length = radial_limit / max(float(tangent), 1e-6)
    length = min(float(light.range), axial_limit, radial_length)

    # A light near the camera can move substantially in screen space along
    # its axis even when world_scale() bounds perpendicular motion. Refine the
    # candidate against its actual projected cone so an off-axis helper cannot
    # span the viewport or cross the near plane.
    height = max(float(viewport_height), 1.0)
    rect = (0.0, 0.0, max(float(camera.aspect), 1e-3) * height, height)
    anchor = project(camera, [light.position], rect)[0]
    budget = float(np.hypot(_SPOT_HELPER_LENGTH_PT, _SPOT_HELPER_RADIUS_PT))
    for _ in range(6):
        starts, ends = spot_cone_segments(light, length)
        screen = project(camera, np.concatenate((starts, ends)), rect)
        if np.all(screen[:, 2] > max(float(camera.near), 1e-6)):
            extent = float(np.max(np.linalg.norm(screen[:, :2] - anchor[:2], axis=1)))
            if np.isfinite(extent) and extent <= budget:
                return length
        else:
            extent = np.inf
        scale = 0.5 if not np.isfinite(extent) else min(0.75, 0.95 * budget / extent)
        length *= max(scale, 0.05)
    return 0.0


def spot_cone_segments(light: Light, length: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(light.position, np.float32)
    direction = _direction(light.direction)
    length = float(light.range if length is None else length)
    radius = length * np.tan(np.deg2rad(np.clip(float(light.cutoff), 0.1, 89.0)))
    circle = oriented_circle(position + direction * length, direction, radius)
    rim = np.roll(circle, -1, axis=0)
    ribs = circle[:: _CIRCLE_SEGMENTS // 8]
    return (
        np.concatenate((circle, np.repeat(position[None], len(ribs), axis=0))),
        np.concatenate((rim, ribs)),
    )


def oriented_circle(center, normal, radius: float) -> np.ndarray:
    basis = direction_basis(normal)
    angle = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_SEGMENTS, endpoint=False)
    return (
        np.asarray(center, np.float32)
        + float(radius)
        * (np.cos(angle)[:, None] * basis[:, 0] + np.sin(angle)[:, None] * basis[:, 1])
    ).astype(np.float32)


def _direction(value) -> np.ndarray:
    direction = math3d.normalize(np.asarray(value, np.float64))
    return direction if np.any(direction) else np.array((0.0, 0.0, -1.0), np.float32)


def _camera_view(session, node):
    index = int(node.camera_index)
    cameras = session.frame.cameras or session.source.cameras
    if 0 <= index < len(cameras):
        return cameras[index]
    if 0 <= index < len(session.cameras):
        view = session.camera_view(session.cameras[index].camera_id)
        if view is not None:
            return view
    return None
