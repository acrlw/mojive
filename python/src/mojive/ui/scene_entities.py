"""Editor helpers for selecting and inspecting scene cameras and lights."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import math3d
from ..adapters.base import NodeType, SceneNode
from ..gizmo import (
    camera_icon_paths,
    project,
    screen_constant_world_sizes,
    world_scale,
)
from ..math3d import camera_rotation as camera_rotation
from ..math3d import direction_basis
from ..render.debugdraw import Occlusion
from ..types import CameraView, Light, LightType
from .theme import THEME

HELPER_LAYER = "ui.scene_entities"
HELPER_ICON_LAYER = "ui.scene_entity_icons"
HELPER_COLOR = np.array((*THEME.text_disabled[:3], 0.82), np.float32)
SELECTED_COLOR = np.array(THEME.primary_bright, np.float32)
ANCHOR_RADIUS_PT = 6.0
PICK_RADIUS_PT = 13.0
CAMERA_HELPER_SIZE_PT = 24.0
LIGHT_HELPER_SCALE_PT = 1.2
_CIRCLE_SEGMENTS = 48
_SPOT_HELPER_LENGTH_PT = 180.0
_SPOT_HELPER_RADIUS_PT = 120.0


@dataclass
class SceneEntityHelpers:
    """Publishes editor-only helpers and resolves their viewport hit targets."""

    visible: bool = True
    show_influence: bool = True
    _cache_session: object | None = field(default=None, init=False, repr=False)
    _cache_generation: int = field(default=-1, init=False, repr=False)
    _camera_nodes: tuple[SceneNode, ...] = field(default=(), init=False, repr=False)
    _light_nodes: tuple[SceneNode, ...] = field(default=(), init=False, repr=False)

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
        icon_layer = debug.layer(HELPER_ICON_LAYER, Occlusion.ALWAYS)
        layer.clear()
        icon_layer.clear()
        if view_through_camera or not self.visible or session.source is None:
            return
        self._refresh_nodes(session)

        selected = session.selected
        camera_helpers: list[tuple[int, CameraView, bool]] = []
        for node in self._camera_nodes:
            if (view := _camera_view(session, node)) is not None:
                if selected == node.object_id and selected_camera_aspect is not None:
                    view = view.with_aspect(selected_camera_aspect)
                camera_helpers.append((node.object_id, view, selected == node.object_id))
        self._cameras(icon_layer, layer, camera_helpers, camera, viewport_height, ui_scale)

        frame = session.frame
        light_set = frame.lights if frame.lights is not None else session.source.lights
        light_helpers: list[tuple[int, Light, bool]] = []
        for node in self._light_nodes:
            if 0 <= node.light_index < len(light_set.lights):
                light = light_set.lights[node.light_index]
                if light.active and light.type is not LightType.IMAGE:
                    light_helpers.append((node.object_id, light, selected == node.object_id))
        self._lights(icon_layer, layer, light_helpers, camera, viewport_height, ui_scale)

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
        self._refresh_nodes(session)
        anchors: list[tuple[int, np.ndarray]] = []
        lights = session.frame.lights or session.source.lights
        for node in self._camera_nodes:
            if (view := _camera_view(session, node)) is not None:
                anchors.append((node.object_id, np.asarray(view.eye, np.float64)))
        for node in self._light_nodes:
            if 0 <= node.light_index < len(lights.lights):
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

    def _refresh_nodes(self, session) -> None:
        generation = int(session.structure_generation)
        if session is self._cache_session and generation == self._cache_generation:
            return
        self._cache_session = session
        self._cache_generation = generation
        nodes = session.nodes
        self._camera_nodes = tuple(node for node in nodes if node.type is NodeType.CAMERA)
        self._light_nodes = tuple(node for node in nodes if node.type is NodeType.LIGHT)

    def _cameras(
        self,
        icon_layer,
        layer,
        helpers: list[tuple[int, CameraView, bool]],
        editor_camera: CameraView,
        viewport_height: float,
        ui_scale: float,
    ) -> None:
        if not helpers:
            return
        views = [view for _, view, _ in helpers]
        outlines, lenses = camera_icon_paths(
            views,
            editor_camera,
            viewport_height,
            CAMERA_HELPER_SIZE_PT * ui_scale,
            visible_only=True,
        )
        for index, (object_id, _view, selected) in enumerate(helpers):
            color = SELECTED_COLOR if selected else HELPER_COLOR
            icon_layer.polyline(
                f"camera:{object_id}:outline",
                outlines[index],
                color,
                1.6 * ui_scale,
                closed=True,
            )
            icon_layer.polyline(
                f"camera:{object_id}:lens",
                lenses[index],
                color,
                1.6 * ui_scale,
                closed=True,
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

    def _lights(
        self,
        icon_layer,
        layer,
        helpers: list[tuple[int, Light, bool]],
        editor_camera: CameraView,
        viewport_height: float,
        ui_scale: float,
    ) -> None:
        if not helpers:
            return
        positions = np.asarray([light.position for _, light, _ in helpers], np.float32)
        colors = np.asarray(
            [SELECTED_COLOR if selected else HELPER_COLOR for _, _, selected in helpers],
            np.float32,
        )
        rings, detail_starts, detail_ends = _light_icon_geometry(
            positions,
            editor_camera,
            viewport_height,
            ui_scale,
        )
        for index, (object_id, _light, _selected) in enumerate(helpers):
            icon_layer.polyline(
                f"light:{object_id}:ring",
                rings[index],
                colors[index],
                1.4 * ui_scale,
                closed=True,
            )
            icon_layer.lines(
                f"light:{object_id}:details",
                detail_starts[index],
                detail_ends[index],
                colors[index],
                1.4 * ui_scale,
            )

        directed = np.asarray(
            [selected and light.type is not LightType.POINT for _, light, selected in helpers],
            bool,
        )
        if np.any(directed):
            starts = positions[directed]
            directions = np.asarray(
                [_direction(light.direction) for _, light, _ in helpers], np.float32
            )[directed]
            lengths = screen_constant_world_sizes(
                editor_camera, starts, viewport_height, 30.0, visible_only=True
            )
            layer.arrows(
                "lights:directions",
                starts,
                starts + directions * lengths[:, None],
                colors[directed],
                1.8 * ui_scale,
                start_mask_px=ANCHOR_RADIUS_PT * ui_scale,
            )

        if not self.show_influence:
            return
        for object_id, light, selected in helpers:
            if selected:
                self._light_influence(
                    layer,
                    object_id,
                    light,
                    editor_camera,
                    viewport_height,
                    ui_scale,
                )

    @staticmethod
    def _light_influence(
        layer,
        object_id: int,
        light: Light,
        editor_camera: CameraView,
        viewport_height: float,
        ui_scale: float,
    ) -> None:
        ident = f"light:{object_id}"
        color = SELECTED_COLOR
        position = np.asarray(light.position, np.float32)
        direction = _direction(light.direction)
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


_LIGHT_RING_ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_LIGHT_RAY_ANGLES = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
_LIGHT_RING_UNIT = np.column_stack((np.cos(_LIGHT_RING_ANGLES), np.sin(_LIGHT_RING_ANGLES)))
_LIGHT_RAY_UNIT = np.column_stack((np.cos(_LIGHT_RAY_ANGLES), np.sin(_LIGHT_RAY_ANGLES)))
_LIGHT_DETAIL_STARTS_2D = np.concatenate(
    (
        np.asarray(((-2.6, -5.2), (-1.9, -6.9)), np.float64),
        _LIGHT_RAY_UNIT * 5.9,
    )
)
_LIGHT_DETAIL_ENDS_2D = np.concatenate(
    (
        np.asarray(((2.6, -5.2), (1.9, -6.9)), np.float64),
        _LIGHT_RAY_UNIT * 7.8,
    )
)


def _light_icon_geometry(
    positions: np.ndarray,
    editor_camera: CameraView,
    viewport_height: float,
    ui_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, np.float64).reshape(-1, 3)
    if not len(positions):
        return (
            np.empty((0, len(_LIGHT_RING_UNIT), 3), np.float32),
            np.empty((0, len(_LIGHT_DETAIL_STARTS_2D), 3), np.float32),
            np.empty((0, len(_LIGHT_DETAIL_ENDS_2D), 3), np.float32),
        )
    units = screen_constant_world_sizes(
        editor_camera,
        positions,
        viewport_height,
        LIGHT_HELPER_SCALE_PT * ui_scale,
        visible_only=True,
    )
    view_rotation = np.asarray(editor_camera.view_matrix(), np.float64)[:3, :3]
    right = view_rotation[0]
    up = view_rotation[1]

    def transform(points: np.ndarray) -> np.ndarray:
        offsets = (
            points[None, :, 0, None] * right[None, None, :]
            + points[None, :, 1, None] * up[None, None, :]
        )
        return (positions[:, None, :] + units[:, None, None] * offsets).astype(np.float32)

    return (
        transform(_LIGHT_RING_UNIT * 4.3),
        transform(_LIGHT_DETAIL_STARTS_2D),
        transform(_LIGHT_DETAIL_ENDS_2D),
    )


def light_icon_segments(
    positions: np.ndarray,
    editor_camera: CameraView,
    viewport_height: float,
    ui_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return screen-facing bulb icons scaled with the rest of the UI."""

    rings, detail_starts, detail_ends = _light_icon_geometry(
        positions,
        editor_camera,
        viewport_height,
        ui_scale,
    )
    if not len(rings):
        empty = np.empty((0, 3), np.float32)
        return empty, empty
    starts = np.concatenate((rings, detail_starts), axis=1)
    ends = np.concatenate((np.roll(rings, -1, axis=1), detail_ends), axis=1)
    return starts.reshape(-1, 3), ends.reshape(-1, 3)


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
