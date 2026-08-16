from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from .types import CameraView

SIZE_PT = 88.0
AXIS_START = 0.0
AXIS_END = 1.0
AXIS_SHAFT_HALF_PT = 2.2
AXIS_HEAD_HALF_PT = 7.0
AXIS_HEAD_LENGTH_PT = 12.0
PLANE_INNER = 0.22
PLANE_OUTER = 0.40
PLANE_ALPHA = 0.42
PLANE_ACTIVE_ALPHA = 0.68
CENTER_RADIUS = 0.075
CENTER_SHELL_RADIUS = 0.11
RING_RADIUS = 0.78
SCREEN_RING_RADIUS = 0.96
RING_WIDTH_PT = 3.5
SCREEN_RING_WIDTH_PT = 2.5
CONTRAST_EDGE_PT = 0.75

RING_TUBE = RING_WIDTH_PT / (2.0 * SIZE_PT)
SCREEN_RING_TUBE = SCREEN_RING_WIDTH_PT / (2.0 * SIZE_PT)
SCREEN_RING_EDGE_TUBE = (SCREEN_RING_WIDTH_PT + 2.0 * CONTRAST_EDGE_PT) / (2.0 * SIZE_PT)

AXIS_HIT_PT = 7.0
RING_HIT_PT = 7.0
CENTER_HIT_PT = 9.0
RING_SEGMENTS = 64

AXIS_COLORS = np.array(
    (
        (239 / 255, 110 / 255, 106 / 255, 1.0),
        (84 / 255, 168 / 255, 83 / 255, 1.0),
        (105 / 255, 147 / 255, 246 / 255, 1.0),
    ),
    np.float32,
)
HOVER_COLOR = np.array((1.0, 0.72, 0.12, 1.0), np.float32)
CENTER_COLOR = np.array((0.92, 0.92, 0.92, 1.0), np.float32)
CONTRAST_EDGE_COLOR = np.array((0.68, 0.71, 0.76, 1.0), np.float32)
GUIDE_CORE_COLOR = np.array((0.98, 0.98, 0.99, 1.0), np.float32)


class GizmoMode(enum.StrEnum):
    TRANSLATE = "translate"
    ROTATE = "rotate"


class GizmoStyle(enum.StrEnum):
    FLAT = "2d"
    SOLID = "3d"


class GizmoSpace(enum.StrEnum):
    BODY = "body"
    WORLD = "world"


class GizmoHandle(enum.IntEnum):
    NONE = 0
    X = 1
    Y = 2
    Z = 3
    YZ = 4
    ZX = 5
    XY = 6
    SCREEN = 7
    ROTATE_X = 8
    ROTATE_Y = 9
    ROTATE_Z = 10
    ROTATE_SCREEN = 11


AXIS_HANDLES = (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z)
PLANE_HANDLES = (GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY)
ROTATE_AXIS_HANDLES = (GizmoHandle.ROTATE_X, GizmoHandle.ROTATE_Y, GizmoHandle.ROTATE_Z)
ROTATE_HANDLES = (*ROTATE_AXIS_HANDLES, GizmoHandle.ROTATE_SCREEN)


@dataclass
class GizmoFrame:
    mode: GizmoMode = GizmoMode.TRANSLATE
    style: GizmoStyle = GizmoStyle.FLAT
    space: GizmoSpace = GizmoSpace.BODY
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    size_px: float = SIZE_PT
    hovered: GizmoHandle = GizmoHandle.NONE
    active: GizmoHandle = GizmoHandle.NONE
    axis_mask: int = 0b111
    plane_mask: int = 0b111


def display_handles(frame: GizmoFrame) -> tuple[GizmoHandle, ...]:

    if frame.active is not GizmoHandle.NONE:
        return (frame.active,)
    if frame.mode is GizmoMode.ROTATE:
        return ROTATE_HANDLES
    axes = tuple(h for i, h in enumerate(AXIS_HANDLES) if frame.axis_mask & (1 << i))
    planes = tuple(h for i, h in enumerate(PLANE_HANDLES) if frame.plane_mask & (1 << i))
    return (*planes, *axes, GizmoHandle.SCREEN)


def world_scale(cam: CameraView, origin, viewport_height: float, size_pt: float = SIZE_PT) -> float:

    h = max(float(viewport_height), 1.0)
    if cam.orthographic:
        return float(cam.ortho_height) * float(size_pt) / h
    depth = float(np.dot(np.asarray(origin, np.float64) - cam.eye, cam.forward()))
    if depth <= 0.0:
        return 0.0
    return 2.0 * depth * float(np.tan(cam.fov_y * 0.5)) * float(size_pt) / h


def project(cam: CameraView, points, rect: tuple[float, float, float, float]) -> np.ndarray:

    p = np.asarray(points, np.float64).reshape(-1, 3)
    mvp = np.asarray(cam.proj_matrix(), np.float64) @ np.asarray(cam.view_matrix(), np.float64)
    clip = np.concatenate((p, np.ones((len(p), 1))), axis=1) @ mvp.T
    w = clip[:, 3]
    safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :2] / safe[:, None]
    x, y, width, height = rect
    return np.column_stack(
        (x + (ndc[:, 0] * 0.5 + 0.5) * width, y + (0.5 - ndc[:, 1] * 0.5) * height, w)
    )


def axis_rotation(rotation, axis: int) -> np.ndarray:

    r = np.asarray(rotation, np.float64).reshape(3, 3)
    order = ((1, 2, 0), (2, 0, 1), (0, 1, 2))[int(axis)]
    return r[:, order]


def screen_rotation_basis(cam: CameraView) -> np.ndarray:

    return np.asarray(cam.view_matrix(), np.float64)[:3, :3].T


def rotation_half_basis(cam: CameraView, origin, rotation, axis: int) -> np.ndarray:

    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    normal = r[:, axis]
    to_eye = -cam.forward() if cam.orthographic else np.asarray(cam.eye, np.float64) - o
    front = to_eye - normal * np.dot(to_eye, normal)
    length = float(np.linalg.norm(front))
    if length < 1e-9:
        front = r[:, ((axis + 1) % 3)]
    else:
        front /= length
    tangent = np.cross(front, normal)
    return np.column_stack((tangent, front, normal))


def rotation_ring_alpha(cam: CameraView, origin, normal) -> float:

    facing = _view_facing(cam, origin, normal)
    return float(np.clip((facing - 0.08) / 0.20, 0.0, 1.0))


def axis_handle_alpha(cam: CameraView, origin, axis) -> float:

    facing = _view_facing(cam, origin, axis)
    projected = np.sqrt(max(0.0, 1.0 - facing * facing))
    return float(np.clip((projected - 0.08) / 0.20, 0.0, 1.0))


def plane_handle_alpha(cam: CameraView, origin, normal) -> float:

    return rotation_ring_alpha(cam, origin, normal)


def _view_facing(cam: CameraView, origin, direction) -> float:
    to_eye = -cam.forward() if cam.orthographic else np.asarray(cam.eye) - np.asarray(origin)
    to_eye /= max(float(np.linalg.norm(to_eye)), 1e-12)
    return abs(float(np.dot(np.asarray(direction), to_eye)))


def rotation_ring(cam, origin, rotation, scale: float, axis: int, *, full: bool) -> np.ndarray:

    basis = (
        axis_rotation(rotation, axis) if full else rotation_half_basis(cam, origin, rotation, axis)
    )
    segments = RING_SEGMENTS if full else RING_SEGMENTS // 2
    angles = (
        np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        if full
        else np.linspace(0.0, np.pi, segments + 1)
    )
    radial = np.cos(angles)[:, None] * basis[:, 0] + np.sin(angles)[:, None] * basis[:, 1]
    return np.asarray(origin, np.float64) + scale * RING_RADIUS * radial


def plane_corners(origin, rotation, scale: float, axis: int) -> np.ndarray:

    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    u, v = ((1, 2), (2, 0), (0, 1))[int(axis)]
    return np.array(
        [
            o + scale * (r[:, u] * a + r[:, v] * b)
            for a, b in (
                (PLANE_INNER, PLANE_INNER),
                (PLANE_OUTER, PLANE_INNER),
                (PLANE_OUTER, PLANE_OUTER),
                (PLANE_INNER, PLANE_OUTER),
            )
        ]
    )


def visibility(
    cam: CameraView,
    origin,
    rotation,
    rect: tuple[float, float, float, float],
    scale: float,
) -> tuple[int, int]:

    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    center = project(cam, [o], rect)[0]
    if center[2] <= 0.0 or scale <= 0.0:
        return 0, 0
    axis_mask = 0
    plane_mask = 0
    for axis in range(3):
        end = project(cam, [o + r[:, axis] * scale * AXIS_END], rect)[0]
        if end[2] > 0.0 and axis_handle_alpha(cam, o, r[:, axis]) > 0.0:
            axis_mask |= 1 << axis
        poly = project(cam, plane_corners(o, r, scale, axis), rect)
        if np.all(poly[:, 2] > 0.0) and plane_handle_alpha(cam, o, r[:, axis]) > 0.0:
            plane_mask |= 1 << axis
    return axis_mask, plane_mask


def hit_test(
    cam: CameraView,
    origin,
    rotation,
    rect: tuple[float, float, float, float],
    cursor: tuple[float, float],
    mode: GizmoMode,
) -> tuple[GizmoHandle, int, int]:

    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    scale = world_scale(cam, o, rect[3])
    axis_mask, plane_mask = visibility(cam, o, r, rect, scale)
    if not axis_mask and not plane_mask:
        return GizmoHandle.NONE, axis_mask, plane_mask
    p = np.asarray(cursor, np.float64)
    center = project(cam, [o], rect)[0, :2]

    if mode is GizmoMode.TRANSLATE:
        if np.linalg.norm(p - center) <= CENTER_HIT_PT:
            return GizmoHandle.SCREEN, axis_mask, plane_mask
        for axis, handle in enumerate(PLANE_HANDLES):
            if plane_mask & (1 << axis) and plane_handle_alpha(cam, o, r[:, axis]) > 0.2:
                poly = project(cam, plane_corners(o, r, scale, axis), rect)[:, :2]
                if _inside_convex(p, poly):
                    return handle, axis_mask, plane_mask
        best = (AXIS_HIT_PT, GizmoHandle.NONE)
        for axis, handle in enumerate(AXIS_HANDLES):
            if not axis_mask & (1 << axis):
                continue
            if axis_handle_alpha(cam, o, r[:, axis]) <= 0.2:
                continue
            a, b = project(
                cam,
                [o + r[:, axis] * scale * AXIS_START, o + r[:, axis] * scale * AXIS_END],
                rect,
            )[:, :2]
            d = _segment_distance(p, a, b)
            if d <= best[0]:
                best = (d, handle)
        return best[1], axis_mask, plane_mask

    screen_radius = SCREEN_RING_RADIUS * SIZE_PT
    if abs(float(np.linalg.norm(p - center)) - screen_radius) <= RING_HIT_PT:
        return GizmoHandle.ROTATE_SCREEN, axis_mask, plane_mask

    best = (RING_HIT_PT, GizmoHandle.NONE)
    for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
        if rotation_ring_alpha(cam, o, r[:, axis]) <= 0.2:
            continue
        ring = rotation_ring(cam, o, r, scale, axis, full=False)
        screen = project(cam, ring, rect)
        if np.any(screen[:, 2] <= 0.0):
            continue
        d = min(
            _segment_distance(p, screen[i, :2], screen[i + 1, :2]) for i in range(len(screen) - 1)
        )
        if d <= best[0]:
            best = (d, handle)
    return best[1], axis_mask, plane_mask


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(np.dot(ab, ab))
    t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    return float(np.linalg.norm(p - (a + ab * t)))


def _inside_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    edge = np.roll(poly, -1, axis=0) - poly
    rel = p - poly
    cross = edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0]
    return bool(np.all(cross >= -1e-6) or np.all(cross <= 1e-6))
