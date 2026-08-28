"""Backend-neutral gizmo geometry, projection, and hit testing."""

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

AXIS_HIT_PADDING_PT = 4.0
PLANE_HIT_PADDING_PT = 2.0
RING_HIT_PT = 5.5
CENTER_HIT_PT = 9.0
HANDLE_HIT_ALPHA = 0.05
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
# Keep active and hover as separate semantic slots even while the palette matches.
ACTIVE_COLOR = np.array((1.0, 0.72, 0.12, 1.0), np.float32)
JOINT_HANDLE_COLOR = np.array((0.72, 0.48, 0.95, 1.0), np.float32)
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
ALL_HANDLE_MASK = sum(1 << int(handle) for handle in GizmoHandle if handle is not GizmoHandle.NONE)


def handle_mask(*handles: GizmoHandle) -> int:
    return sum(1 << int(handle) for handle in handles)


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
    active_rotation_overlay: bool = False
    axis_mask: int = 0b111
    plane_mask: int = 0b111
    handle_mask: int = ALL_HANDLE_MASK
    handle_color: np.ndarray | None = None
    active_projection_fade: bool = False


def display_handles(frame: GizmoFrame) -> tuple[GizmoHandle, ...]:
    if frame.active is not GizmoHandle.NONE:
        return (frame.active,)
    if frame.mode is GizmoMode.ROTATE:
        return tuple(handle for handle in ROTATE_HANDLES if frame.handle_mask & (1 << int(handle)))
    axes = tuple(h for i, h in enumerate(AXIS_HANDLES) if frame.axis_mask & (1 << i))
    planes = tuple(h for i, h in enumerate(PLANE_HANDLES) if frame.plane_mask & (1 << i))
    handles = (*planes, *axes, GizmoHandle.SCREEN)
    return tuple(handle for handle in handles if frame.handle_mask & (1 << int(handle)))


def rotation_ring_is_full(frame: GizmoFrame, handle: GizmoHandle) -> bool:
    """Return whether an axis ring should show its full circumference."""

    if frame.active is handle:
        return True
    if frame.active is not GizmoHandle.NONE:
        return False
    visible_axes = sum(
        bool(frame.handle_mask & (1 << int(candidate))) for candidate in ROTATE_AXIS_HANDLES
    )
    return visible_axes == 1 and bool(frame.handle_mask & (1 << int(handle)))


def world_scale(cam: CameraView, origin, viewport_height: float, size_pt: float = SIZE_PT) -> float:
    h = max(float(viewport_height), 1.0)
    view = np.asarray(cam.view_matrix(), np.float64)
    projection = np.asarray(cam.proj_matrix(), np.float64)
    point = np.append(np.asarray(origin, np.float64), 1.0)
    clip_w = float((projection @ (view @ point))[3])
    p11 = float(projection[1, 1])
    if clip_w <= 0.0 or abs(p11) < 1e-9:
        return 0.0
    return 2.0 * clip_w * float(size_pt) / (p11 * h)


def screen_constant_world_sizes(
    camera: CameraView,
    positions,
    viewport_height: float,
    pixels: float,
    *,
    visible_only: bool = False,
) -> np.ndarray:
    """Return world lengths that occupy a fixed number of viewport pixels."""
    positions = np.asarray(positions, np.float64).reshape(-1, 3)
    if not len(positions):
        return np.zeros(0, np.float64)
    if camera.orthographic:
        value = float(camera.ortho_height) * float(pixels) / max(float(viewport_height), 1.0)
        return np.full(len(positions), value, np.float64)
    forward = np.asarray(camera.forward(), np.float64)
    depths = (positions - np.asarray(camera.eye, np.float64)) @ forward
    visible = depths > 0.0
    if not visible_only:
        np.abs(depths, out=depths)
        visible[:] = True
    result = np.zeros(len(positions), np.float64)
    np.maximum(depths, max(float(camera.near), 1e-4), out=depths)
    result[visible] = (
        2.0
        * depths[visible]
        * np.tan(float(camera.fov_y) * 0.5)
        * float(pixels)
        / max(float(viewport_height), 1.0)
    )
    return result


def camera_icon_segments(
    views: tuple[CameraView, ...] | list[CameraView],
    editor_camera: CameraView,
    viewport_height: float,
    pixels: float,
    *,
    visible_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build compact screen-constant camera icons for one debug-draw batch."""
    if not views:
        empty = np.empty((0, 3), np.float32)
        return empty, empty
    eyes = np.asarray([view.eye for view in views], np.float64)
    targets = np.asarray([view.target for view in views], np.float64)
    ups = np.asarray([view.up for view in views], np.float64)
    forward = _normalize_rows(targets - eyes, (0.0, 0.0, -1.0))
    right = np.cross(forward, ups)
    degenerate = np.linalg.norm(right, axis=1) <= 1e-9
    if np.any(degenerate):
        reference = np.zeros((int(np.count_nonzero(degenerate)), 3), np.float64)
        reference[:, 2] = 1.0
        reference[np.abs(forward[degenerate, 2]) >= 0.95] = (0.0, 1.0, 0.0)
        right[degenerate] = np.cross(forward[degenerate], reference)
    right = _normalize_rows(right, (1.0, 0.0, 0.0))
    up = np.cross(right, forward)
    lengths = screen_constant_world_sizes(
        editor_camera, eyes, viewport_height, pixels, visible_only=visible_only
    )
    centers = eyes + forward * lengths[:, None]
    half_height = lengths * 0.45
    half_width = half_height * np.clip(
        np.asarray([view.aspect for view in views], np.float64), 0.75, 1.8
    )
    horizontal = right * half_width[:, None]
    vertical = up * half_height[:, None]
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
    values = np.asarray(values, np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(values, axis=1)
    result = np.empty_like(values)
    valid = lengths > 1e-9
    result[valid] = values[valid] / lengths[valid, None]
    result[~valid] = fallback
    return result


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


def handle_projection_alpha(
    frame: GizmoFrame,
    handle: GizmoHandle,
    cam: CameraView,
    origin,
    direction,
) -> float:
    """Return one projection-degeneracy alpha for every gizmo renderer."""

    if frame.active is handle and not frame.active_projection_fade:
        return 1.0
    if handle in AXIS_HANDLES:
        return axis_handle_alpha(cam, origin, direction)
    if handle in PLANE_HANDLES or handle in ROTATE_AXIS_HANDLES:
        return rotation_ring_alpha(cam, origin, direction)
    return 1.0


_PLANE_SPAN_AXES = ((1, 2), (2, 0), (0, 1))


def plane_direction(rotation, axis: int) -> np.ndarray:
    """World direction from the origin toward the plane handle's center."""
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    u, v = _PLANE_SPAN_AXES[int(axis)]
    return r[:, u] + r[:, v]


def paint_order(cam: CameraView, origin, directions) -> tuple[int, ...]:
    """Painter's-algorithm handle order: far-to-near along the view ray.

    Overlapping handles cannot rely on per-pixel depth (the 2D overlay draws
    into an imgui draw list; the 3D depth pin squashes handle depth against
    the near plane), so each handle group is drawn far-to-near by the depth
    of a point one unit along its direction (arrow axis or plane diagonal).
    Returns indices into ``directions``.
    """
    view = _view_direction(cam, origin)
    keys = [float(np.dot(np.asarray(d, np.float64), view)) for d in directions]
    return tuple(sorted(range(len(directions)), key=keys.__getitem__, reverse=True))


def _view_direction(cam: CameraView, origin) -> np.ndarray:
    """Unit vector pointing away from the camera through ``origin``."""
    d = (
        np.asarray(cam.forward(), np.float64)
        if cam.orthographic
        else np.asarray(origin, np.float64) - np.asarray(cam.eye, np.float64)
    )
    return d / max(float(np.linalg.norm(d)), 1e-12)


def _view_facing(cam: CameraView, origin, direction) -> float:
    return abs(float(np.dot(np.asarray(direction), _view_direction(cam, origin))))


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
    return rotation_dial(
        origin,
        basis[:, 2],
        basis[:, 0],
        scale,
        RING_RADIUS,
        angles,
    )


def rotation_dial(origin, axis, start_direction, scale: float, radius, angles) -> np.ndarray:
    """Build dial points in one world-space rotation plane."""
    normal = np.asarray(axis, np.float64)
    normal /= np.linalg.norm(normal)
    radial = np.asarray(start_direction, np.float64)
    radial -= normal * np.dot(radial, normal)
    radial /= np.linalg.norm(radial)
    tangent = np.cross(normal, radial)
    angles = np.atleast_1d(np.asarray(angles, np.float64))
    radii = np.broadcast_to(np.asarray(radius, np.float64), angles.shape)
    directions = np.cos(angles)[:, None] * radial + np.sin(angles)[:, None] * tangent
    return np.asarray(origin, np.float64) + float(scale) * radii[:, None] * directions


def plane_corners(origin, rotation, scale: float, axis: int) -> np.ndarray:
    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    u, v = _PLANE_SPAN_AXES[int(axis)]
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


def masked_axis_start(origin, end, radius: float) -> np.ndarray:
    """Clip a projected axis against the position handle's invisible shell."""
    origin = np.asarray(origin, np.float64)
    end = np.asarray(end, np.float64)
    direction = end - origin
    length = float(np.linalg.norm(direction))
    if length <= radius or length < 1e-9:
        return end
    return origin + direction * (float(radius) / length)


def axis_arrow_polygon(start, end, style_scale: float = 1.0) -> np.ndarray:
    """Return the exact screen-space silhouette used by a flat axis handle."""
    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.empty((0, 2), np.float64)
    direction /= length
    side = np.array((-direction[1], direction[0]))
    shaft = AXIS_SHAFT_HALF_PT * float(style_scale)
    head = min(AXIS_HEAD_LENGTH_PT * float(style_scale), length * 0.42)
    wing = AXIS_HEAD_HALF_PT * float(style_scale)
    neck = end - direction * head
    return np.asarray(
        (
            start - side * shaft,
            neck - side * shaft,
            neck - side * wing,
            end,
            neck + side * wing,
            neck + side * shaft,
            start + side * shaft,
        ),
        np.float64,
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
    style_scale: float = 1.0,
    allowed_handles: int = ALL_HANDLE_MASK,
) -> tuple[GizmoHandle, int, int]:
    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    style_scale = float(style_scale)
    scale = world_scale(cam, o, rect[3], SIZE_PT * style_scale)
    axis_mask, plane_mask = visibility(cam, o, r, rect, scale)
    if not axis_mask and not plane_mask:
        return GizmoHandle.NONE, axis_mask, plane_mask
    p = np.asarray(cursor, np.float64)
    center = project(cam, [o], rect)[0, :2]

    def allowed(handle: GizmoHandle) -> bool:
        return bool(allowed_handles & (1 << int(handle)))

    if mode is GizmoMode.TRANSLATE:
        if (
            allowed(GizmoHandle.SCREEN)
            and np.linalg.norm(p - center) <= CENTER_HIT_PT * style_scale
        ):
            return GizmoHandle.SCREEN, axis_mask, plane_mask
        axes = [
            axis for axis in range(3) if axis_mask & (1 << axis) and allowed(AXIS_HANDLES[axis])
        ]
        order = paint_order(cam, o, [r[:, axis] for axis in axes])
        for index in reversed(order):
            axis = axes[index]
            handle = AXIS_HANDLES[axis]
            if axis_handle_alpha(cam, o, r[:, axis]) <= HANDLE_HIT_ALPHA:
                continue
            if np.linalg.norm(p - center) < CENTER_SHELL_RADIUS * SIZE_PT * style_scale:
                continue
            screen = project(
                cam,
                (o + r[:, axis] * scale * AXIS_START, o + r[:, axis] * scale * AXIS_END),
                rect,
            )
            if np.any(screen[:, 2] <= 0.0):
                continue
            start = masked_axis_start(
                screen[0, :2],
                screen[1, :2],
                CENTER_SHELL_RADIUS * SIZE_PT * style_scale,
            )
            polygon = axis_arrow_polygon(start, screen[1, :2], style_scale)
            if _polygon_distance(p, polygon) <= AXIS_HIT_PADDING_PT * style_scale:
                return handle, axis_mask, plane_mask

        planes = [
            axis for axis in range(3) if plane_mask & (1 << axis) and allowed(PLANE_HANDLES[axis])
        ]
        order = paint_order(cam, o, [plane_direction(r, axis) for axis in planes])
        for index in reversed(order):
            axis = planes[index]
            if plane_handle_alpha(cam, o, r[:, axis]) <= HANDLE_HIT_ALPHA:
                continue
            polygon = project(cam, plane_corners(o, r, scale, axis), rect)[:, :2]
            if _polygon_distance(p, polygon) <= PLANE_HIT_PADDING_PT * style_scale:
                return PLANE_HANDLES[axis], axis_mask, plane_mask
        return GizmoHandle.NONE, axis_mask, plane_mask

    screen_radius = SCREEN_RING_RADIUS * SIZE_PT * style_scale
    if allowed(GizmoHandle.ROTATE_SCREEN) and (
        abs(float(np.linalg.norm(p - center)) - screen_radius) <= RING_HIT_PT * style_scale
    ):
        return GizmoHandle.ROTATE_SCREEN, axis_mask, plane_mask

    best_distance = RING_HIT_PT * style_scale
    best_handle = GizmoHandle.NONE
    full_axis_ring = sum(allowed(handle) for handle in ROTATE_AXIS_HANDLES) == 1
    for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
        if not allowed(handle):
            continue
        if rotation_ring_alpha(cam, o, r[:, axis]) <= HANDLE_HIT_ALPHA:
            continue
        ring = rotation_ring(cam, o, r, scale, axis, full=full_axis_ring)
        screen = project(cam, ring, rect)
        if np.any(screen[:, 2] <= 0.0):
            continue
        distance = min(
            _segment_distance(p, screen[i, :2], screen[(i + 1) % len(screen), :2])
            for i in range(len(screen) if full_axis_ring else len(screen) - 1)
        )
        if distance < best_distance - 1e-6 or (
            abs(distance - best_distance) <= 1e-6 and handle > best_handle
        ):
            best_distance = distance
            best_handle = handle
    return best_handle, axis_mask, plane_mask


def _polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return float("inf")
    if _inside_polygon(point, polygon):
        return 0.0
    return min(
        _segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(np.dot(ab, ab))
    t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    return float(np.linalg.norm(p - (a + ab * t)))


def _inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return whether a point lies inside a simple screen-space polygon."""
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > y) != (previous[1] > y):
            edge_x = previous[0] - current[0]
            edge_y = previous[1] - current[1]
            if x < current[0] + edge_x * (y - current[1]) / edge_y:
                inside = not inside
        previous = current
    return inside
