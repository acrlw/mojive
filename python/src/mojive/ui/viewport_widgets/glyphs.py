"""Viewport widgets: glyphs."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from mojive.drawing.curves import (
    CORNER_SMOOTHING,
    box_handle_points,
    smooth_line_cap,
    smooth_polygon_corners,
)
from mojive.interaction.gizmo import (
    ARROW_CORNER_RADIUS_PT,
    DIMENSION_CORNER_RADIUS_RATIO,
    _rounded_polygon_corners,
)
from mojive.ui.draw2d import Draw2D

from .model import (
    _FRAME_ARROW_CORNER_RADIUS_PT,
    _FRAME_AXES,
    _MOVE_ARROW_BASE,
    _MOVE_ARROW_TIP,
    _MOVE_ARROW_WING,
    _MOVE_SHAFT_VISUAL_RATIO,
    _PROJECTION_HALF_WIDTH_PT,
    _PROJECTION_STROKE_PT,
    CAPSULE_SMOOTHING,
    FRAME_LABEL_MAX_WIDTH,
    OVERLAY_GEOMETRY,
    PLAYBACK_HALF_HEIGHT_PT,
    PLAYBACK_RESET_SCALE,
    PLAYBACK_STEP_SCALE,
    RECORDING_OPTIONS_ENVELOPE_SCALE,
    RECORDING_OPTIONS_GLYPH_SCALE,
    RECORDING_OPTIONS_STROKE_SCALE,
    RESET_GLYPH_SCALE,
    TOOL_GLYPH_SCALE,
    OverlayGeometry,
    _transform_path,
)
from .rotate import (
    _rotate_visible_ring_polygons,
)


@lru_cache(maxsize=64)
def reset_glyph_path(stroke: float, smoothing: float = CORNER_SMOOTHING):
    """Construct a counterclockwise arrow within Geometry's normalized icon circle."""
    radius = OVERLAY_GEOMETRY.icon_radius - stroke / 2
    angles = np.radians(np.linspace(140, -140, 65))
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = np.array((radial[-1, 1], -radial[-1, 0]))
    outer, inner = (radius + stroke / 2) * radial, (radius - stroke / 2) * radial
    head = np.array(
        (
            (radius + 2 * stroke) * radial[-1],
            radius * radial[-1] + 4 * stroke * tangent,
            (radius - 2 * stroke) * radial[-1],
        )
    )
    cap = smooth_line_cap(
        radius * radial[0], (-radial[0, 1], radial[0, 0]), stroke, smoothing=smoothing
    )
    outline = np.vstack((outer, head, inner[::-1], cap))
    # The head and arc share one silhouette, rounded with the existing G3 primitive.
    outline = smooth_polygon_corners(
        outline, stroke * 0.28, tuple(range(len(outer), len(outer) + 3)), smoothing=smoothing
    )
    outline *= (
        OVERLAY_GEOMETRY.icon_radius * RESET_GLYPH_SCALE / np.linalg.norm(outline, axis=1).max()
    )
    return tuple(map(tuple, outline.tolist()))


@lru_cache(maxsize=128)
def _scaled_reset_glyph(scale, stroke, smoothing):
    return tuple((x * scale, y * scale) for x, y in reset_glyph_path(stroke, smoothing))


def draw_reset_glyph(
    draw, center, color, scale, stroke=OVERLAY_GEOMETRY.tool_stroke, *, smoothing=None
):
    path = _scaled_reset_glyph(
        scale,
        stroke,
        getattr(draw, "corner_smoothing", CORNER_SMOOTHING) if smoothing is None else smoothing,
    )
    draw.fringed_concave_fill(path, color, origin=center)


def draw_playback_glyph(
    draw: Draw2D,
    center,
    color,
    scale: float,
    kind: str,
    *,
    smoothing: float = CAPSULE_SMOOTHING,
) -> None:
    """Draw one normalized playback glyph for runtime and design probes."""

    x, y = center
    if kind in ("previous", "step"):
        scale *= PLAYBACK_STEP_SCALE
    elif kind == "stop":
        scale *= PLAYBACK_RESET_SCALE
    half_height = PLAYBACK_HALF_HEIGHT_PT * scale
    if kind in ("play", "reverse", "previous", "step"):
        triangle, barrier_x = _rounded_playback_triangle(kind, scale, smoothing)
        draw.fringed_concave_fill(tuple((x + px, y + py) for px, py in triangle), color)
        if kind in ("previous", "step"):
            draw.rect_filled(
                (x + barrier_x - 0.7 * scale, y - half_height),
                (x + barrier_x + 0.7 * scale, y + half_height),
                color,
                rounding=0.7 * scale,
                smoothing=smoothing,
            )
    elif kind == "pause":
        draw.rect_filled(
            (x - 5.4 * scale, y - half_height),
            (x - 1.0 * scale, y + half_height),
            color,
            rounding=0.9 * scale,
            smoothing=smoothing,
        )
        draw.rect_filled(
            (x + 1.0 * scale, y - half_height),
            (x + 5.4 * scale, y + half_height),
            color,
            rounding=0.9 * scale,
            smoothing=smoothing,
        )
    elif kind == "reset":
        draw_reset_glyph(draw, center, color, scale, smoothing=smoothing)
    elif kind == "stop":
        draw.rect_filled(
            (x - half_height, y - half_height),
            (x + half_height, y + half_height),
            color,
            rounding=1.0 * scale,
            smoothing=smoothing,
        )


@lru_cache(maxsize=128)
def _rounded_playback_triangle(
    kind: str,
    scale: float,
    smoothing: float = CAPSULE_SMOOTHING,
) -> tuple[tuple[tuple[float, float], ...], float]:
    points = _rounded_polygon_corners(
        np.array(((-4.6, -8.0), (9.2, 0.0), (-4.6, 8.0))) * scale,
        0.8 * scale,
        (0, 1, 2),
        smoothing=smoothing,
    )
    # Align the visible curved extrema, not the discarded sharp triangle tips.
    points *= PLAYBACK_HALF_HEIGHT_PT * scale / np.max(np.abs(points[:, 1]))
    barrier_x = 0.0
    if kind in ("previous", "step"):
        lo, hi = points[:, 0].min(), points[:, 0].max()
        barrier_x = hi + 2.0 * scale
        offset = (lo + barrier_x + 0.7 * scale) * 0.5
        points[:, 0] -= offset
        barrier_x -= offset
        if kind == "previous":
            points[:, 0] *= -1.0
            barrier_x *= -1.0
    elif kind == "reverse":
        points[:, 0] *= -1.0
    return tuple(map(tuple, points.tolist())), float(barrier_x)


def _play_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    from mojive.ui.icons import draw_icon

    draw_icon(draw, center, 2.0 * OVERLAY_GEOMETRY.icon_radius * scale, "playback-play", color)


def _pause_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    from mojive.ui.icons import draw_icon

    draw_icon(draw, center, 2.0 * OVERLAY_GEOMETRY.icon_radius * scale, "playback-pause", color)


def _step_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    from mojive.ui.icons import draw_icon

    draw_icon(draw, center, 2.0 * OVERLAY_GEOMETRY.icon_radius * scale, "playback-next", color)


def _previous_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    from mojive.ui.icons import draw_icon

    draw_icon(draw, center, 2.0 * OVERLAY_GEOMETRY.icon_radius * scale, "playback-previous", color)


def _reset_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    from mojive.ui.icons import draw_icon

    draw_icon(draw, center, 2.0 * OVERLAY_GEOMETRY.icon_radius * scale, "playback-reset", color)


def draw_projection_glyph(
    draw: Draw2D,
    center,
    color,
    scale: float,
    kind: str,
) -> None:
    """Draw a compact perspective frustum or orthographic volume glyph."""

    x, y = (float(value) for value in center)
    s = float(scale)
    half_near = (2.7 if kind == "persp" else 5.4) * s
    half_far = 5.4 * s
    near_x = x - _PROJECTION_HALF_WIDTH_PT * s
    far_x = x + _PROJECTION_HALF_WIDTH_PT * s
    width = max(1.0, _PROJECTION_STROKE_PT * s)
    draw.polyline(
        (
            (near_x, y - half_near),
            (far_x, y - half_far),
            (far_x, y + half_far),
            (near_x, y + half_near),
        ),
        color,
        width,
        closed=True,
    )


def draw_projection_label(draw: Draw2D, lo, hi, color, scale: float, kind: str, label: str) -> None:
    """Center the visible pair horizontally and its shared body line vertically."""

    from mojive.ui.icons import draw_icon_label

    icon_name = "panel-perspective" if kind == "persp" else "panel-orthographic"
    draw_icon_label(draw, lo, hi, color, scale, icon_name, label)


def _tool_icon(draw: Draw2D, center, color, scale: float, packed) -> None:
    _surface, payload = packed
    kind, space = payload
    from mojive.ui.icons import draw_icon

    name = {
        "move": "tool-move",
        "rotate": "tool-rotate",
        "dimensions": "tool-scale",
        "frame": "tool-world" if space == "world" else "tool-body",
        "snap": "tool-snap",
    }[kind]
    draw_icon(
        draw,
        center,
        2.0 * OVERLAY_GEOMETRY.icon_radius * TOOL_GLYPH_SCALE * scale,
        name,
        color,
    )


@lru_cache(maxsize=64)
def _move_glyph_path(
    x: float,
    y: float,
    scale: float,
    base: float,
    tip: float,
    wing: float,
    shaft_half: float,
    smoothing: float = CORNER_SMOOTHING,
) -> tuple[tuple[float, float], ...]:
    """Build the four-way move icon as one connected antialiased outline."""
    local = _move_glyph_shape(scale, base, tip, wing, shaft_half, smoothing)
    return _transform_path(local, x, y, 1.0)


@lru_cache(maxsize=64)
def _move_glyph_shape(
    scale: float, base: float, tip: float, wing: float, shaft_half: float, smoothing: float
):
    local = (
        (0.0, -tip),
        (wing, -base),
        (shaft_half, -base),
        (shaft_half, -shaft_half),
        (base, -shaft_half),
        (base, -wing),
        (tip, 0.0),
        (base, wing),
        (base, shaft_half),
        (shaft_half, shaft_half),
        (shaft_half, base),
        (wing, base),
        (0.0, tip),
        (-wing, base),
        (-shaft_half, base),
        (-shaft_half, shaft_half),
        (-base, shaft_half),
        (-base, wing),
        (-tip, 0.0),
        (-base, -wing),
        (-base, -shaft_half),
        (-shaft_half, -shaft_half),
        (-shaft_half, -base),
        (-wing, -base),
    )
    polygon = tuple((px * scale, py * scale) for px, py in local)
    radius = min(ARROW_CORNER_RADIUS_PT, (tip - base) * 0.18) * scale
    return tuple(
        map(
            tuple,
            smooth_polygon_corners(
                polygon,
                radius,
                tuple(range(len(polygon))),
                smoothing=smoothing,
                convex_only=False,
                corner_radii=dict.fromkeys((2, 4, 8, 10, 14, 16, 20, 22), radius * 0.5),
            ).tolist(),
        )
    )


def _draw_axis_arrow_glyph(
    draw: Draw2D,
    center,
    direction,
    color,
    geometry_scale: float,
    stroke_width: float,
    *,
    clear_radius: float,
    base: float,
    tip: float,
    wing: float,
    corner_radius: float,
    smoothing: float = CORNER_SMOOTHING,
) -> None:
    """Draw one continuous coordinate-axis silhouette with a transparent origin shell."""
    ux, uy = (float(value) for value in direction)
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    x, y = (float(value) for value in center)
    # A filled silhouette's fringe lies outside the core. Preserve the apparent
    # stroke weight of the former native line while joining it to the head.
    core_width = max(stroke_width * 0.45, stroke_width - 1.0)
    draw.arrow(
        (x + ux * clear_radius, y + uy * clear_radius),
        (x + ux * tip * geometry_scale, y + uy * tip * geometry_scale),
        color,
        core_width,
        head_length=(tip - base) * geometry_scale,
        head_width=2.0 * wing * geometry_scale,
        corner_radius=corner_radius,
        smoothing=smoothing,
    )


@lru_cache(maxsize=64)
def _snap_glyph_shape(scale: float, smoothing: float = CORNER_SMOOTHING):
    radius = 6.4 * scale
    cap = smooth_line_cap((0.0, 0.0), (0.0, 1.0), 2.0 * radius, smoothing=smoothing)
    return ((-radius, -6.2 * scale), *map(tuple, cap.tolist()), (radius, -6.2 * scale))


def _dimensions_glyph_geometry(
    center,
    scale: float,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
    *,
    smoothing: float = CAPSULE_SMOOTHING,
):
    """Return the original three Scale handles and center-dot radius."""

    x, y = float(center[0]), float(center[1])
    glyph_scale = float(scale) * TOOL_GLYPH_SCALE
    stroke = geometry.tool_stroke * float(scale)
    half = 1.5 * float(scale)
    reach = math.sqrt((9.0 * glyph_scale) ** 2 - half**2) - half
    clear_radius = (
        geometry.frame_center_radius * glyph_scale
        + geometry.tool_stroke * geometry.frame_center_gap_ratio * float(scale)
    )
    paths = tuple(
        box_handle_points(
            (x + ux * clear_radius, y + uy * clear_radius),
            (x + ux * reach, y + uy * reach),
            max(stroke * 0.45, stroke - 1.0),
            2.0 * half,
            corner_radius=2.0 * half * DIMENSION_CORNER_RADIUS_RATIO,
            smoothing=smoothing,
        )
        for ux, uy in _FRAME_AXES
    )
    return paths, geometry.frame_center_radius * glyph_scale


def draw_tool_glyph(
    draw: Draw2D,
    center,
    color,
    scale: float,
    kind: str,
    space: str,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
    *,
    smoothing: float = CAPSULE_SMOOTHING,
) -> None:
    """Draw one normalized Tool Column glyph for runtime and design probes."""

    x, y = center
    glyph_scale = scale * TOOL_GLYPH_SCALE
    stroke = geometry.tool_stroke * scale
    if kind == "move":
        draw.fringed_concave_fill(
            _move_glyph_path(
                float(x),
                float(y),
                float(glyph_scale),
                _MOVE_ARROW_BASE,
                _MOVE_ARROW_TIP,
                _MOVE_ARROW_WING,
                geometry.tool_stroke * _MOVE_SHAFT_VISUAL_RATIO * 0.5 / TOOL_GLYPH_SCALE,
                smoothing,
            ),
            color,
        )
    elif kind == "rotate":
        ring_stroke = stroke
        # The screen-rotation path itself is the Tool glyph envelope. Keep its
        # centerline on the construction bound; subtracting half the stroke
        # makes the ring read smaller even when its outer edge is technically
        # inside the same diameter.
        radius = 10.0 * glyph_scale
        draw.circle(center, radius, color, ring_stroke, segments=48)
        for ring in _rotate_visible_ring_polygons(
            geometry.tool_stroke,
            geometry.rotate_ring_gap_ratio,
            geometry.rotate_ring_cap,
            smoothing,
        ):
            for local in ring:
                path = tuple((px * glyph_scale, py * glyph_scale) for px, py in local)
                # Triangulate around the local origin, then translate the
                # complete mesh. ImGui's ear clipping loses precision on the
                # narrow knockout contours after large screen translations.
                draw.fringed_concave_fill(path, color, origin=(x, y))
    elif kind == "dimensions":
        paths, dot_radius = _dimensions_glyph_geometry(
            center,
            scale,
            geometry,
            smoothing=smoothing,
        )
        for path in paths:
            draw.fringed_concave_fill(path, color)
        draw.circle_filled(center, dot_radius, color, segments=16)
    elif kind == "frame":
        clear_radius = (
            geometry.frame_center_radius * glyph_scale
            + geometry.tool_stroke * geometry.frame_center_gap_ratio * scale
        )
        for direction in _FRAME_AXES:
            _draw_axis_arrow_glyph(
                draw,
                center,
                direction,
                color,
                glyph_scale,
                geometry.tool_stroke * scale,
                clear_radius=clear_radius,
                base=6.6,
                tip=9.0,
                wing=1.8,
                corner_radius=_FRAME_ARROW_CORNER_RADIUS_PT * scale,
                smoothing=smoothing,
            )
        # Shaft endpoints sit on the outside of the dot's relative transparent
        # shell; drawing the dot last supplies the visible white origin.
        draw.circle_filled(
            center,
            geometry.frame_center_radius * glyph_scale,
            color,
            segments=16,
        )
        draw.centered_label(
            "W" if space == "world" else "B",
            (x + 4.6 * glyph_scale, y - 4.5 * glyph_scale),
            color,
            FRAME_LABEL_MAX_WIDTH * scale,
        )
    else:
        path = _transform_path(_snap_glyph_shape(glyph_scale, smoothing), x, y, 1.0)
        draw.polyline(path, color, stroke)


@lru_cache(maxsize=64)
def expand_glyph_path(
    stroke: float,
    smoothing: float = CORNER_SMOOTHING,
    scale: float = 1.0,
    envelope: float = 1.0,
):
    """Join two rectangular arms, rounding every exposed corner with the shared G3 profile.

    ``scale`` magnifies the whole glyph, including the arm stroke, so an inline toolbar can
    match the optical weight of a filled symbol without redrawing the path. ``envelope``
    lengthens the arms at a constant stroke weight, which is how a glyph grows to fill a
    larger state circle without turning into a heavier mark.
    """
    stroke *= scale
    offset = stroke / math.sqrt(2)
    arm = 4.0 * envelope
    rise = 2.0 * envelope
    outline = (
        (-arm - offset, -rise),
        (0, rise + offset),
        (arm + offset, -rise),
        (arm, -rise - offset),
        (0, rise - offset),
        (-arm, -rise - offset),
    )
    corners = tuple(range(len(outline)))
    path = smooth_polygon_corners(
        outline,
        stroke / 2,
        corners,
        smoothing=smoothing,
        convex_only=False,
        corner_radii=dict.fromkeys(corners, stroke / 2),
    )
    return tuple(map(tuple, path.tolist()))


def draw_expand_glyph(
    draw,
    center,
    color,
    scale,
    stroke=OVERLAY_GEOMETRY.tool_stroke,
    glyph_scale: float = 1.0,
    envelope: float = 1.0,
):
    draw.fringed_concave_fill(
        tuple(
            (center[0] + x * scale, center[1] + y * scale)
            for x, y in expand_glyph_path(stroke, draw.corner_smoothing, glyph_scale, envelope)
        ),
        color,
    )


def draw_recording_glyph(draw, center, color, scale, *, recording=False):
    if recording:
        draw_playback_glyph(draw, center, color, scale, "stop", smoothing=draw.corner_smoothing)
    else:
        radius = 2 * PLAYBACK_HALF_HEIGHT_PT * PLAYBACK_RESET_SCALE / math.sqrt(math.pi)
        draw.circle_filled(center, radius * scale, color, segments=48)


def _record_icon(draw, center, color, scale, packed):
    _surface, (recording, accent, action) = packed
    from mojive.ui.icons import draw_icon

    size = 2.0 * OVERLAY_GEOMETRY.icon_radius * scale
    if recording and action in ("pause", "resume"):
        name = "playback-pause" if action == "pause" else "playback-play"
    else:
        name = "playback-stop" if recording else "playback-record"
    draw_icon(draw, center, size, name, accent)


def draw_recording_options_glyph(
    draw, center, color, scale, *, stroke=OVERLAY_GEOMETRY.tool_stroke
):
    """Draw the shared recording-menu chevron with rounded, joined arms."""
    draw_expand_glyph(
        draw,
        center,
        color,
        scale,
        stroke * RECORDING_OPTIONS_STROKE_SCALE,
        glyph_scale=RECORDING_OPTIONS_GLYPH_SCALE,
        envelope=RECORDING_OPTIONS_ENVELOPE_SCALE,
    )


def _recording_options_icon(draw, center, color, scale, _packed):
    from mojive.ui.icons import draw_icon

    draw_icon(draw, center, 2.0 * OVERLAY_GEOMETRY.icon_radius * scale, "playback-more", color)
