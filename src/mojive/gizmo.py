"""Backend-neutral gizmo geometry, projection, and hit testing."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .types import CameraView

SIZE_PT = 88.0
AXIS_START = 0.0
AXIS_END = 1.0
AXIS_SHAFT_HALF_PT = 2.2
AXIS_HEAD_HALF_PT = 7.0
AXIS_HEAD_LENGTH_PT = 12.0
ARROW_CORNER_RADIUS_PT = 0.5
PLANE_INNER = 0.22
PLANE_OUTER = 0.40
PLANE_ALPHA = 0.42
PLANE_ACTIVE_ALPHA = 0.68
CENTER_RADIUS = 0.075
CENTER_SHELL_RADIUS = 0.11
RING_RADIUS = 0.78
SCREEN_RING_RADIUS = 0.96
TRACKBALL_RADIUS = RING_RADIUS
RING_WIDTH_PT = 3.5
SCREEN_RING_WIDTH_PT = 2.5
CONTRAST_EDGE_PT = 0.75
JOINT_OUTLINE_PT = 0.6
ROTATE_RING_ALPHA = 0.68
ROTATE_RING_HOVER_ALPHA = 0.88
ROTATE_RING_ACTIVE_ALPHA = 0.96
TRACKBALL_ALPHA = 0.035
TRACKBALL_HOVER_ALPHA = 0.16
TRACKBALL_ACTIVE_ALPHA = 0.22
DIMENSION_HANDLE_HALF_PT = 5.5
DIMENSION_SHAFT_WIDTH_PT = 2.5

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
        (220 / 255, 119 / 255, 115 / 255, 1.0),
        (82 / 255, 170 / 255, 92 / 255, 1.0),
        (111 / 255, 148 / 255, 229 / 255, 1.0),
    ),
    np.float32,
)
HOVER_COLOR = np.array((184 / 255, 210 / 255, 172 / 255, 1.0), np.float32)
# Scalar-joint handles use the Primary palette so they remain legible in
# visually dense scenes and share the application's interaction language.
ACTIVE_HANDLE_COLOR = np.array((184 / 255, 210 / 255, 172 / 255, 1.0), np.float32)
ACTIVE_COLOR = np.array((103 / 255, 135 / 255, 90 / 255, 1.0), np.float32)
JOINT_HANDLE_COLOR = np.array((156 / 255, 191 / 255, 141 / 255, 1.0), np.float32)
JOINT_OUTLINE_COLOR = np.array((220 / 255, 223 / 255, 227 / 255, 0.94), np.float32)
CENTER_COLOR = np.array((0.92, 0.92, 0.92, 1.0), np.float32)
TRACKBALL_COLOR = np.array((0.70, 0.70, 0.70, 1.0), np.float32)
CONTRAST_EDGE_COLOR = np.array((0.68, 0.71, 0.76, 1.0), np.float32)
GUIDE_CORE_COLOR = np.array((0.98, 0.98, 0.99, 1.0), np.float32)


@dataclass(frozen=True)
class GizmoProjection:
    """Camera matrices prepared once for a related gizmo operation."""

    projection: np.ndarray
    view_projection: np.ndarray


def prepare_projection(cam: CameraView) -> GizmoProjection:
    """Prepare immutable per-operation camera matrices for gizmo projection."""

    projection = np.asarray(cam.proj_matrix(), np.float64)
    view = np.asarray(cam.view_matrix(), np.float64)
    return GizmoProjection(projection, projection @ view)


def axis_hover_color(color) -> tuple[float, float, float, float]:
    """Return a brighter interaction color that preserves the source axis hue."""

    rgba = tuple(float(value) for value in color)
    return (*(value + (1.0 - value) * 0.20 for value in rgba[:3]), rgba[3])


def axis_active_color(color) -> tuple[float, float, float, float]:
    """Return the pressed color for one axis without replacing it with primary."""

    rgba = tuple(float(value) for value in color)
    return (*(value + (1.0 - value) * 0.36 for value in rgba[:3]), rgba[3])


def axis_dark_color(color) -> tuple[float, float, float, float]:
    """Return the dark companion used by active rotation arcs and back rings."""

    rgba = tuple(float(value) for value in color)
    return (*(value * 0.58 for value in rgba[:3]), rgba[3])


class GizmoMode(enum.StrEnum):
    TRANSLATE = "translate"
    ROTATE = "rotate"
    DIMENSIONS = "dimensions"


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
    ROTATE_TRACKBALL = 12


AXIS_HANDLES = (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z)
PLANE_HANDLES = (GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY)
ROTATE_AXIS_HANDLES = (GizmoHandle.ROTATE_X, GizmoHandle.ROTATE_Y, GizmoHandle.ROTATE_Z)
ROTATE_HANDLES = (
    *ROTATE_AXIS_HANDLES,
    GizmoHandle.ROTATE_SCREEN,
    GizmoHandle.ROTATE_TRACKBALL,
)
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
    outline_color: np.ndarray | None = None
    active_projection_fade: bool = False


def rotation_handle_color(
    frame: GizmoFrame,
    handle: GizmoHandle,
    axis: int,
    projection_alpha: float = 1.0,
) -> np.ndarray:
    """Resolve one axis ring without replacing its semantic hue with primary."""

    base = AXIS_COLORS[axis] if frame.handle_color is None else frame.handle_color
    if frame.active is handle:
        color = ACTIVE_HANDLE_COLOR if frame.handle_color is not None else axis_active_color(base)
        opacity = ROTATE_RING_ACTIVE_ALPHA
    elif frame.hovered is handle:
        color = axis_hover_color(base)
        opacity = ROTATE_RING_HOVER_ALPHA
    else:
        color = base
        opacity = ROTATE_RING_ALPHA
    result = np.asarray(color, np.float32).copy()
    result[3] *= float(projection_alpha) * opacity
    return result


def trackball_color(frame: GizmoFrame) -> np.ndarray:
    """Resolve Blender-style low-opacity trackball background feedback."""

    color = TRACKBALL_COLOR.copy()
    if frame.active is GizmoHandle.ROTATE_TRACKBALL:
        color[3] = TRACKBALL_ACTIVE_ALPHA
    elif frame.hovered is GizmoHandle.ROTATE_TRACKBALL:
        color[3] = TRACKBALL_HOVER_ALPHA
    else:
        color[3] = TRACKBALL_ALPHA
    return color


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


def world_scale(
    cam: CameraView,
    origin,
    viewport_height: float,
    size_pt: float = SIZE_PT,
    *,
    prepared: GizmoProjection | None = None,
) -> float:
    h = max(float(viewport_height), 1.0)
    prepared = prepared or prepare_projection(cam)
    point = np.append(np.asarray(origin, np.float64), 1.0)
    clip_w = float((prepared.view_projection @ point)[3])
    p11 = float(prepared.projection[1, 1])
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
    projection = np.asarray(camera.proj_matrix(), np.float64)
    view_projection = projection @ np.asarray(camera.view_matrix(), np.float64)
    homogeneous = np.column_stack((positions, np.ones(len(positions), np.float64)))
    depths = homogeneous @ view_projection[3]
    visible = depths > 0.0
    if not visible_only:
        np.abs(depths, out=depths)
        np.maximum(depths, 1e-9, out=depths)
        visible[:] = True
    result = np.zeros(len(positions), np.float64)
    p11 = abs(float(projection[1, 1]))
    if p11 < 1e-9:
        return result
    result[visible] = (
        2.0 * depths[visible] * float(pixels) / (p11 * max(float(viewport_height), 1.0))
    )
    return result


_CAMERA_ICON_OUTLINE_2D = np.asarray(
    (
        (-0.50, -0.25),
        (-0.43, -0.34),
        (0.43, -0.34),
        (0.50, -0.25),
        (0.50, 0.25),
        (0.43, 0.34),
        (0.21, 0.34),
        (0.12, 0.50),
        (-0.17, 0.50),
        (-0.26, 0.34),
        (-0.43, 0.34),
        (-0.50, 0.25),
    ),
    np.float64,
)
_CAMERA_ICON_ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_CAMERA_ICON_LENS_2D = np.column_stack(
    (0.16 * np.cos(_CAMERA_ICON_ANGLES), 0.16 * np.sin(_CAMERA_ICON_ANGLES))
)


def camera_icon_paths(
    views: tuple[CameraView, ...] | list[CameraView],
    editor_camera: CameraView,
    viewport_height: float,
    pixels: float,
    *,
    visible_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build joined screen-facing camera outline and lens paths."""

    if not views:
        return (
            np.empty((0, len(_CAMERA_ICON_OUTLINE_2D), 3), np.float32),
            np.empty((0, len(_CAMERA_ICON_LENS_2D), 3), np.float32),
        )
    eyes = np.asarray([view.eye for view in views], np.float64)
    lengths = screen_constant_world_sizes(
        editor_camera, eyes, viewport_height, pixels, visible_only=visible_only
    )
    view_rotation = np.asarray(editor_camera.view_matrix(), np.float64)[:3, :3]
    right = view_rotation[0]
    up = view_rotation[1]

    def transform(points: np.ndarray) -> np.ndarray:
        values = eyes[:, None, :] + lengths[:, None, None] * (
            points[None, :, 0, None] * right[None, None, :]
            + points[None, :, 1, None] * up[None, None, :]
        )
        return values.astype(np.float32)

    return transform(_CAMERA_ICON_OUTLINE_2D), transform(_CAMERA_ICON_LENS_2D)


def camera_icon_segments(
    views: tuple[CameraView, ...] | list[CameraView],
    editor_camera: CameraView,
    viewport_height: float,
    pixels: float,
    *,
    visible_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build screen-facing, screen-constant camera glyphs for one draw batch.

    The helper is an editor icon, not a second projection frustum.  Keeping the
    body, viewfinder bump, and lens circle in the editor-camera plane makes it
    read as a camera from every view direction; the selected camera's actual
    orientation remains available through its separate frustum helper.
    """
    paths = camera_icon_paths(
        views,
        editor_camera,
        viewport_height,
        pixels,
        visible_only=visible_only,
    )
    if not views:
        empty = np.empty((0, 3), np.float32)
        return empty, empty
    starts = np.concatenate(paths, axis=1)
    ends = np.concatenate(tuple(np.roll(path, -1, axis=1) for path in paths), axis=1)
    return starts.reshape(-1, 3).astype(np.float32), ends.reshape(-1, 3).astype(np.float32)


def _normalize_rows(values: np.ndarray, fallback) -> np.ndarray:
    values = np.asarray(values, np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(values, axis=1)
    result = np.empty_like(values)
    valid = lengths > 1e-9
    result[valid] = values[valid] / lengths[valid, None]
    result[~valid] = fallback
    return result


def project(
    cam: CameraView,
    points,
    rect: tuple[float, float, float, float],
    *,
    prepared: GizmoProjection | None = None,
) -> np.ndarray:
    p = np.asarray(points, np.float64).reshape(-1, 3)
    prepared = prepared or prepare_projection(cam)
    clip = np.concatenate((p, np.ones((len(p), 1))), axis=1) @ prepared.view_projection.T
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
    to_eye = -_view_direction(cam, o)
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
    perspective = np.asarray(origin, np.float64) - np.asarray(cam.eye, np.float64)
    perspective /= max(float(np.linalg.norm(perspective)), 1e-12)
    orthographic = np.asarray(cam.forward(), np.float64)
    mix = cam.projection_blend()
    d = perspective * (1.0 - mix) + orthographic * mix
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


def dimension_axis_geometry(start, end, style_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a scale-style shaft endpoint and screen-aligned square handle."""

    start = np.asarray(start, np.float64).reshape(2)
    end = np.asarray(end, np.float64).reshape(2)
    direction = end - start
    length = float(np.linalg.norm(direction))
    half = DIMENSION_HANDLE_HALF_PT * float(style_scale)
    shaft_end = end if length <= half else end - direction * (half / length)
    square = end + half * np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)), np.float64
    )
    return shaft_end, square


def _rounded_polygon_corners(
    points,
    radius: float,
    corners: tuple[int, ...],
    *,
    segments: int = 6,
) -> np.ndarray:
    """Replace selected convex polygon corners with tangent circular fillets.

    ``radius`` is the desired screen-space radius. Edge trimming is derived from
    the corner angle and clamped to the available edge lengths. Sampling includes
    the arc midpoint, so symmetric arrow tips have one rounded apex rather than a
    short flat segment.
    """

    polygon = np.asarray(points, np.float64).reshape(-1, 2)
    count = len(polygon)
    if count < 3 or radius <= 0.0 or segments < 1:
        return polygon.copy()
    selected = {int(index) % count for index in corners}
    signed_area = 0.5 * float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )
    winding = 1.0 if signed_area >= 0.0 else -1.0
    rounded: list[np.ndarray] = []
    for index, point in enumerate(polygon):
        if index not in selected:
            rounded.append(point)
            continue
        incoming_edge = point - polygon[index - 1]
        outgoing_edge = polygon[(index + 1) % count] - point
        turn = incoming_edge[0] * outgoing_edge[1] - incoming_edge[1] * outgoing_edge[0]
        if turn * winding <= 1e-9:
            rounded.append(point)
            continue
        incoming = polygon[index - 1] - point
        outgoing = polygon[(index + 1) % count] - point
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length < 1e-9 or outgoing_length < 1e-9:
            rounded.append(point)
            continue
        incoming /= incoming_length
        outgoing /= outgoing_length
        angle = float(np.arccos(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
        half_angle = angle * 0.5
        tangent = float(radius) / max(float(np.tan(half_angle)), 1e-6)
        tangent = min(tangent, incoming_length * 0.45, outgoing_length * 0.45)
        effective_radius = tangent * float(np.tan(half_angle))
        bisector = incoming + outgoing
        bisector_length = float(np.linalg.norm(bisector))
        if bisector_length < 1e-9 or effective_radius < 1e-9:
            rounded.append(point)
            continue
        center = point + bisector / bisector_length * (
            effective_radius / max(float(np.sin(half_angle)), 1e-6)
        )
        start = point + incoming * tangent
        end = point + outgoing * tangent
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        if winding > 0.0:
            sweep = (end_angle - start_angle) % (2.0 * math.pi)
        else:
            sweep = -((start_angle - end_angle) % (2.0 * math.pi))
        for step in range(segments + 1):
            amount = step / segments
            rounded.append(
                center
                + effective_radius
                * np.asarray(
                    (
                        math.cos(start_angle + sweep * amount),
                        math.sin(start_angle + sweep * amount),
                    )
                )
            )
    return np.asarray(rounded, np.float64)


@lru_cache(maxsize=32)
def _rounded_arrow_head(
    head: float,
    shaft: float,
    wing: float,
    radius: float,
) -> np.ndarray:
    """Cache the invariant local head while the projected shaft length changes."""

    path = _rounded_polygon_corners(
        (
            (0.0, -shaft),
            (0.0, -wing),
            (head, 0.0),
            (0.0, wing),
            (0.0, shaft),
        ),
        radius,
        (1, 2, 3),
    )
    path.flags.writeable = False
    return path


@lru_cache(maxsize=32)
def _rounded_arrow_tail(shaft: float) -> np.ndarray:
    """Return the semicircle between the two shaft edges, excluding its ends."""

    angles = np.linspace(0.0, np.pi, 9)[1:-1]
    tail = shaft * np.column_stack((-np.sin(angles), np.cos(angles)))
    tail.flags.writeable = False
    return tail


def axis_arrow_polygon(
    start, end, style_scale: float = 1.0, *, round_tail: bool = False
) -> np.ndarray:
    """Return the rounded screen-space silhouette used by a flat axis handle."""
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
    local_head = _rounded_arrow_head(
        head,
        shaft,
        wing,
        ARROW_CORNER_RADIUS_PT * float(style_scale),
    )
    screen_head = neck + local_head[:, :1] * direction[None, :] + local_head[:, 1:] * side[None, :]
    local_tail = _rounded_arrow_tail(shaft) if round_tail else np.empty((0, 2), np.float64)
    screen_tail = start + local_tail[:, :1] * direction[None, :] + local_tail[:, 1:] * side[None, :]
    return np.concatenate(
        (
            (start - side * shaft)[None, :],
            screen_head,
            (start + side * shaft)[None, :],
            screen_tail,
        )
    )


def visibility(
    cam: CameraView,
    origin,
    rotation,
    rect: tuple[float, float, float, float],
    scale: float,
    *,
    prepared: GizmoProjection | None = None,
) -> tuple[int, int]:
    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    if scale <= 0.0:
        return 0, 0
    prepared = prepared or prepare_projection(cam)
    axis_ends = np.asarray(
        [o + r[:, axis] * scale * AXIS_END for axis in range(3)],
        np.float64,
    )
    plane_points = np.concatenate(
        [plane_corners(o, r, scale, axis) for axis in range(3)],
        axis=0,
    )
    screen = project(
        cam,
        np.concatenate((o[None, :], axis_ends, plane_points), axis=0),
        rect,
        prepared=prepared,
    )
    center = screen[0]
    if center[2] <= 0.0:
        return 0, 0
    axis_mask = 0
    plane_mask = 0
    for axis in range(3):
        end = screen[1 + axis]
        if end[2] > 0.0 and axis_handle_alpha(cam, o, r[:, axis]) > 0.0:
            axis_mask |= 1 << axis
        start = 4 + axis * 4
        poly = screen[start : start + 4]
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
    preferred_handle: GizmoHandle = GizmoHandle.NONE,
) -> tuple[GizmoHandle, int, int]:
    o = np.asarray(origin, np.float64)
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    style_scale = float(style_scale)
    prepared = prepare_projection(cam)
    scale = world_scale(cam, o, rect[3], SIZE_PT * style_scale, prepared=prepared)
    axis_mask, plane_mask = visibility(cam, o, r, rect, scale, prepared=prepared)
    if not axis_mask and not plane_mask:
        return GizmoHandle.NONE, axis_mask, plane_mask
    p = np.asarray(cursor, np.float64)
    center = project(cam, [o], rect, prepared=prepared)[0, :2]

    def allowed(handle: GizmoHandle) -> bool:
        return bool(allowed_handles & (1 << int(handle)))

    if mode in (GizmoMode.TRANSLATE, GizmoMode.DIMENSIONS):
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
                prepared=prepared,
            )
            if np.any(screen[:, 2] <= 0.0):
                continue
            start = masked_axis_start(
                screen[0, :2],
                screen[1, :2],
                CENTER_SHELL_RADIUS * SIZE_PT * style_scale,
            )
            if mode is GizmoMode.DIMENSIONS:
                shaft_end, polygon = dimension_axis_geometry(start, screen[1, :2], style_scale)
                distance = min(
                    _polygon_distance(p, polygon),
                    _segment_distance(p, start, shaft_end)
                    - DIMENSION_SHAFT_WIDTH_PT * 0.5 * style_scale,
                )
            else:
                polygon = axis_arrow_polygon(start, screen[1, :2], style_scale)
                distance = _polygon_distance(p, polygon)
            if distance <= AXIS_HIT_PADDING_PT * style_scale:
                return handle, axis_mask, plane_mask

        if mode is GizmoMode.DIMENSIONS:
            return GizmoHandle.NONE, axis_mask, plane_mask

        planes = [
            axis for axis in range(3) if plane_mask & (1 << axis) and allowed(PLANE_HANDLES[axis])
        ]
        order = paint_order(cam, o, [plane_direction(r, axis) for axis in planes])
        for index in reversed(order):
            axis = planes[index]
            if plane_handle_alpha(cam, o, r[:, axis]) <= HANDLE_HIT_ALPHA:
                continue
            polygon = project(
                cam,
                plane_corners(o, r, scale, axis),
                rect,
                prepared=prepared,
            )[:, :2]
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
    preferred_distance = float("inf")
    full_axis_ring = sum(allowed(handle) for handle in ROTATE_AXIS_HANDLES) == 1
    for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
        if not allowed(handle):
            continue
        if rotation_ring_alpha(cam, o, r[:, axis]) <= HANDLE_HIT_ALPHA:
            continue
        ring = rotation_ring(cam, o, r, scale, axis, full=full_axis_ring)
        screen = project(cam, ring, rect, prepared=prepared)
        if np.any(screen[:, 2] <= 0.0):
            continue
        distance = min(
            _segment_distance(p, screen[i, :2], screen[(i + 1) % len(screen), :2])
            for i in range(len(screen) if full_axis_ring else len(screen) - 1)
        )
        if handle is preferred_handle:
            preferred_distance = distance
        if distance < best_distance - 1e-6 or (
            abs(distance - best_distance) <= 1e-6 and handle > best_handle
        ):
            best_distance = distance
            best_handle = handle
    # Axis rings necessarily overlap at their projected crossings. Keep the
    # current ring while the pointer remains inside its hit tube so sub-pixel
    # cursor jitter cannot alternate the selected axis from frame to frame.
    if preferred_distance <= RING_HIT_PT * style_scale:
        return preferred_handle, axis_mask, plane_mask
    if best_handle is not GizmoHandle.NONE:
        return best_handle, axis_mask, plane_mask
    if allowed(GizmoHandle.ROTATE_TRACKBALL) and (
        np.linalg.norm(p - center) <= TRACKBALL_RADIUS * SIZE_PT * style_scale
    ):
        return GizmoHandle.ROTATE_TRACKBALL, axis_mask, plane_mask
    return GizmoHandle.NONE, axis_mask, plane_mask


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
