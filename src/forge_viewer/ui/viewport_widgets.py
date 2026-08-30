"""Production viewport chrome shared by the interactive viewer.

The glyphs are fixed screen-space geometry.  Curves used by the rotate, snap,
mouse, and capsule shapes are sampled once at import; a frame only scales and
translates those points before submitting them to ``Draw2D``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import lru_cache

from imgui_bundle import imgui

from .draw2d import Draw2D
from .input_bindings import DEFAULT_INPUT_BINDINGS, InputAction, InputBindings
from .theme import Theme


@dataclass(frozen=True)
class OverlayGeometry:
    """Shared logical-pixel geometry for viewport chrome and its design probe."""

    icon_radius: float = 10.0
    radial_step: float = 6.0
    center_step: float = 36.0
    tool_center_step: float = 40.0
    tool_group_gap: float = 10.0
    divider_width: float = 20.0
    tool_stroke: float = 1.42
    rotate_ring_gap: float = 0.32
    hint_control_height: float = 18.0
    hint_padding_x: float = 16.0
    hint_padding_y: float = 8.0
    hint_input_gap: float = 8.0
    hint_group_gap: float = 24.0
    hint_chord_gap: float = 10.0
    hint_key_padding_x: float = 8.0
    hint_mouse_width: float = 14.0

    @property
    def state_radius(self) -> float:
        return self.icon_radius + self.radial_step

    @property
    def shell_radius(self) -> float:
        return self.state_radius + self.radial_step


@dataclass(frozen=True)
class ViewportLabels:
    """Localized viewport chrome copy, resolved only when language changes."""

    play: str = "Play"
    pause: str = "Pause"
    step: str = "Step"
    pause_to_step: str = "Pause to step"
    stop: str = "Stop"
    move: str = "Move"
    rotate: str = "Rotate"
    world_body: str = "World / Body"
    snap: str = "Snap"
    orbit: str = "Orbit"
    pan: str = "Pan"
    zoom: str = "Zoom"
    frame: str = "Frame"
    type_value: str = "Type value"
    drag: str = "Drag"
    push: str = "Push"
    twist: str = "Twist"
    running: str = "Running"
    paused: str = "Paused"
    static: str = "Static"
    time: str = "Time"
    steps: str = "Steps"
    no_selection: str = "No selection"
    show_steps: str = "Click to show steps"
    show_time: str = "Click to show time"
    copy_exact: str = "Right-click to copy exact value"


DEFAULT_VIEWPORT_LABELS = ViewportLabels()


def localized_viewport_labels(translate: Callable[[str], str]) -> ViewportLabels:
    """Resolve the compact chrome catalog through the application's localizer."""

    return ViewportLabels(
        **{
            item.name: translate(getattr(DEFAULT_VIEWPORT_LABELS, item.name))
            for item in fields(DEFAULT_VIEWPORT_LABELS)
        }
    )


@dataclass(frozen=True)
class StatusLayout:
    """Interactive geometry emitted while drawing the application status bar."""

    metric_rect: tuple[float, float, float, float] | None = None
    metric_exact: str = ""


@dataclass(frozen=True)
class _StatusPerformanceLayout:
    """Stable right-edge columns for backend and frame telemetry."""

    backend_text: str
    delta_text: str
    fps_text: str
    backend_x: float
    delta_x: float
    fps_x: float
    delta_divider_x: float
    fps_divider_x: float
    left: float


OVERLAY_GEOMETRY = OverlayGeometry()
DEFAULT_VIEWPORT_OVERLAY_SCALE = 1.25
MIN_VIEWPORT_OVERLAY_SCALE = 0.85
MAX_VIEWPORT_OVERLAY_SCALE = 2.0
MAX_VIEWPORT_CHROME_STYLE_SCALE = 1.15
PLAYBACK_CHROME_SCALE = 0.82
TOOL_CHROME_SCALE = PLAYBACK_CHROME_SCALE
HINT_CHROME_SCALE = 0.68
# ImGui clips a window draw list at the host window boundary.  A capsule's
# antialiased outline extends beyond its mathematical path, and fractional
# framebuffer scaling can add another pixel.  Keep a logical-pixel guard band
# around every host window so the rounded end is never flattened by that clip.
OVERLAY_CLIP_PADDING = 5.0
# Tool glyphs need a little more optical weight than the playback symbols.
# This scale changes only their authored paths; hit regions, state circles, and
# capsule spacing continue to use the shared overlay geometry.
TOOL_GLYPH_SCALE = 1.18
FRAME_LABEL_MAX_WIDTH = 4.4
CAPSULE_SURFACE_ALPHA = 0.92

ICON_RADIUS = OVERLAY_GEOMETRY.icon_radius
STATE_RADIUS = OVERLAY_GEOMETRY.state_radius
SHELL_RADIUS = OVERLAY_GEOMETRY.shell_radius
CENTER_STEP = OVERLAY_GEOMETRY.center_step
TOOL_GROUP_GAP = OVERLAY_GEOMETRY.tool_group_gap
DIVIDER_WIDTH = OVERLAY_GEOMETRY.divider_width

_CAPSULE_RIGHT = tuple(
    (math.sin(math.pi * index / 48.0), -math.cos(math.pi * index / 48.0)) for index in range(49)
)
_CAPSULE_LEFT = tuple((-x, -y) for x, y in _CAPSULE_RIGHT)
_CAPSULE_TOP = tuple(
    (
        math.cos(math.pi + math.pi * index / 48.0),
        math.sin(math.pi + math.pi * index / 48.0),
    )
    for index in range(49)
)
_CAPSULE_BOTTOM = tuple(
    (math.cos(math.pi * index / 48.0), math.sin(math.pi * index / 48.0)) for index in range(49)
)
_SNAP_ARC = tuple(
    (math.cos(math.pi - math.pi * index / 48.0), math.sin(math.pi - math.pi * index / 48.0))
    for index in range(49)
)
_QUARTER_ARC = tuple(
    (
        math.cos(-math.pi * 0.5 - math.pi * 0.5 * index / 24.0),
        math.sin(-math.pi * 0.5 - math.pi * 0.5 * index / 24.0),
    )
    for index in range(1, 25)
)

# Fixed ISO-orthographic front half-rings. Draw order is Y, X, Z.
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

_FRAME_AXES = (
    (0.0, -1.0, 1.0, 0.0),
    (0.866025, 0.5, -0.5, 0.866025),
    (-0.866025, 0.5, -0.5, -0.866025),
)


@dataclass(frozen=True)
class _OverlayControl:
    """One declarative capsule action; groups provide separator placement."""

    name: str


PLAYBACK_CONTROLS = tuple(_OverlayControl(name) for name in ("toggle", "step", "stop"))
TOOL_GROUPS = (
    tuple(_OverlayControl(name) for name in ("move", "rotate", "frame")),
    (_OverlayControl("snap"),),
)


def _tool_control_centers() -> tuple[float, ...]:
    centers: list[float] = []
    cursor = SHELL_RADIUS
    for group_index, group in enumerate(TOOL_GROUPS):
        if group_index:
            cursor += TOOL_GROUP_GAP
        for _control in group:
            centers.append(cursor)
            cursor += OVERLAY_GEOMETRY.tool_center_step
    return tuple(centers)


TOOL_CONTROL_CENTERS = _tool_control_centers()
TOOL_CONTROLS = tuple(control for group in TOOL_GROUPS for control in group)


def viewport_chrome_scale(
    style_scale: float,
    overlay_scale: float,
    component_scale: float,
) -> float:
    """Scale transient viewport chrome without mirroring oversized panel UI."""

    return (
        min(float(style_scale), MAX_VIEWPORT_CHROME_STYLE_SCALE)
        * float(overlay_scale)
        * float(component_scale)
    )


@lru_cache(maxsize=64)
def capsule_points(x: float, y: float, width: float, height: float):
    if width >= height:
        radius = height * 0.5
        cy = y + radius
        right = x + width - radius
        left = x + radius
        return tuple((right + ux * radius, cy + uy * radius) for ux, uy in _CAPSULE_RIGHT) + tuple(
            (left + ux * radius, cy + uy * radius) for ux, uy in _CAPSULE_LEFT
        )
    radius = width * 0.5
    cx = x + radius
    top = y + radius
    bottom = y + height - radius
    return tuple((cx + ux * radius, top + uy * radius) for ux, uy in _CAPSULE_TOP) + tuple(
        (cx + ux * radius, bottom + uy * radius) for ux, uy in _CAPSULE_BOTTOM
    )


@lru_cache(maxsize=256)
def _transform_path(
    points: tuple[tuple[float, float], ...],
    x: float,
    y: float,
    scale: float,
) -> tuple[tuple[float, float], ...]:
    return tuple((x + px * scale, y + py * scale) for px, py in points)


def draw_capsule(
    draw: Draw2D,
    origin,
    width: float,
    height: float,
    theme: Theme,
    scale: float,
) -> None:
    points = capsule_points(float(origin[0]), float(origin[1]), width, height)
    draw.convex_fill(points, (*theme.bg_child[:3], CAPSULE_SURFACE_ALPHA))
    draw.polyline(points, theme.primary, 1.4 * scale, closed=True)


def playback_size(scale: float) -> tuple[float, float]:
    return (
        (SHELL_RADIUS * 2.0 + CENTER_STEP * (len(PLAYBACK_CONTROLS) - 1)) * scale,
        SHELL_RADIUS * 2.0 * scale,
    )


def tool_column_size(scale: float) -> tuple[float, float]:
    last_center = TOOL_CONTROL_CENTERS[-1]
    return (SHELL_RADIUS * 2.0 * scale, (last_center + SHELL_RADIUS) * scale)


def _circle_button(
    draw: Draw2D,
    item_id: str,
    center,
    theme: Theme,
    scale: float,
    icon: Callable[
        [Draw2D, tuple[float, float], tuple[float, float, float, float], float, object], None
    ],
    *,
    selected: bool = False,
    enabled: bool = True,
    payload: object = None,
) -> bool:
    hit = CENTER_STEP * scale
    lo = (center[0] - hit * 0.5, center[1] - hit * 0.5)
    imgui.set_cursor_screen_pos(imgui.ImVec2(*lo))
    imgui.begin_disabled(not enabled)
    clicked = imgui.invisible_button(item_id, imgui.ImVec2(hit, hit))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    imgui.end_disabled()
    if enabled and (selected or hovered or active):
        draw.circle_filled(center, STATE_RADIUS * scale, theme.bg_frame_active)
    foreground = (
        theme.primary_bright
        if enabled and (selected or hovered or active)
        else theme.text
        if enabled
        else theme.text_disabled
    )
    surface = theme.bg_frame_active if selected or hovered or active else theme.bg_popup
    icon(draw, center, foreground, scale, (surface, payload))
    return bool(clicked and enabled)


def draw_playback_glyph(
    draw: Draw2D,
    center,
    color,
    scale: float,
    kind: str,
) -> None:
    """Draw one normalized playback glyph for runtime and design probes."""

    x, y = center
    if kind == "play":
        draw.convex_fill(
            (
                (x - 4.8 * scale, y - 7.2 * scale),
                (x + 7.0 * scale, y),
                (x - 4.8 * scale, y + 7.2 * scale),
            ),
            color,
        )
    elif kind == "pause":
        draw.rect_filled(
            (x - 5.0 * scale, y - 7.5 * scale),
            (x - 1.0 * scale, y + 7.5 * scale),
            color,
        )
        draw.rect_filled(
            (x + 1.0 * scale, y - 7.5 * scale),
            (x + 5.0 * scale, y + 7.5 * scale),
            color,
        )
    elif kind == "step":
        draw.convex_fill(
            (
                (x - 6.3 * scale, y - 6.3 * scale),
                (x + 2.0 * scale, y),
                (x - 6.3 * scale, y + 6.3 * scale),
            ),
            color,
        )
        draw.rect_filled(
            (x + 4.8 * scale, y - 6.3 * scale),
            (x + 6.3 * scale, y + 6.3 * scale),
            color,
        )
    elif kind == "stop":
        draw.rect_filled(
            (x - 6.35 * scale, y - 6.35 * scale),
            (x + 6.35 * scale, y + 6.35 * scale),
            color,
            rounding=1.0 * scale,
        )


def _play_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "play")


def _pause_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "pause")


def _step_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "step")


def _stop_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "stop")


def draw_playback(
    draw: Draw2D,
    origin,
    theme: Theme,
    scale: float,
    *,
    playing: bool,
    step_enabled: bool,
    enabled: bool = True,
    bindings: InputBindings = DEFAULT_INPUT_BINDINGS,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> str:
    width, height = playback_size(scale)
    draw_capsule(draw, origin, width, height, theme, scale)
    x, y = origin
    result = ""
    controls = {
        "toggle": (_pause_icon if playing else _play_icon, playing, True),
        "step": (_step_icon, False, step_enabled),
        "stop": (_stop_icon, False, True),
    }
    for index, control in enumerate(PLAYBACK_CONTROLS):
        name = control.name
        icon, selected, action_enabled = controls[name]
        center = (
            x + (SHELL_RADIUS + CENTER_STEP * index) * scale,
            y + SHELL_RADIUS * scale,
        )
        if _circle_button(
            draw,
            f"##viewport-playback-{name}",
            center,
            theme,
            scale,
            icon,
            selected=selected,
            enabled=enabled and action_enabled,
        ):
            result = name
        imgui.set_item_tooltip(
            f"{labels.pause} ({bindings.label(InputAction.TOGGLE_PAUSE)})"
            if name == "toggle" and playing
            else f"{labels.play} ({bindings.label(InputAction.TOGGLE_PAUSE)})"
            if name == "toggle"
            else labels.step
            if name == "step" and action_enabled
            else labels.pause_to_step
            if name == "step"
            else labels.stop
        )
    return result


def _tool_icon(draw: Draw2D, center, color, scale: float, packed) -> None:
    surface, payload = packed
    kind, space = payload
    draw_tool_glyph(draw, center, color, scale, kind, surface, space)


@lru_cache(maxsize=128)
def _arrow_silhouette_path(
    x: float,
    y: float,
    ux: float,
    uy: float,
    scale: float,
    base: float,
    tip: float,
    wing: float,
    shaft_half: float,
) -> tuple[tuple[float, float], ...]:
    """Build one continuous narrow-shaft arrow silhouette."""

    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    nx, ny = -uy, ux

    def point(along: float, across: float) -> tuple[float, float]:
        return (
            x + (ux * along + nx * across) * scale,
            y + (uy * along + ny * across) * scale,
        )

    return (
        point(0.0, shaft_half),
        point(base, shaft_half),
        point(base, wing),
        point(tip, 0.0),
        point(base, -wing),
        point(base, -shaft_half),
        point(0.0, -shaft_half),
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
) -> tuple[tuple[float, float], ...]:
    """Build the four-way move icon as one connected antialiased outline."""

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
    return tuple((x + px * scale, y + py * scale) for px, py in local)


def _draw_arrow_glyph(
    draw: Draw2D,
    center,
    direction,
    color,
    scale: float,
    *,
    base: float,
    tip: float,
    wing: float,
    shaft_half: float,
) -> None:
    """Submit a shaft and triangular head as one explicitly AA-filled path."""

    draw.fringed_concave_fill(
        _arrow_silhouette_path(
            float(center[0]),
            float(center[1]),
            float(direction[0]),
            float(direction[1]),
            float(scale),
            float(base),
            float(tip),
            float(wing),
            float(shaft_half),
        ),
        color,
    )


def draw_tool_glyph(
    draw: Draw2D,
    center,
    color,
    scale: float,
    kind: str,
    surface,
    space: str,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
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
                5.0,
                9.0,
                2.35,
                geometry.tool_stroke * 0.5 / TOOL_GLYPH_SCALE,
            ),
            color,
        )
    elif kind == "rotate":
        ring_stroke = geometry.tool_stroke * 0.8 * scale
        halo = ring_stroke + geometry.rotate_ring_gap * 2.0 * scale
        # The screen-rotation path itself is the Tool glyph envelope. Keep its
        # centerline on the construction bound; subtracting half the stroke
        # makes the ring read smaller even when its outer edge is technically
        # inside the same diameter.
        radius = 10.0 * glyph_scale
        draw.circle(center, radius, surface, halo, segments=48)
        draw.circle(center, radius, color, ring_stroke, segments=48)
        for local in _ROTATE_HALF_RINGS:
            path = _transform_path(local, x, y, glyph_scale)
            draw.polyline(path, surface, halo)
            draw.polyline(path, color, ring_stroke)
    elif kind == "frame":
        for ux, uy, _nx, _ny in _FRAME_AXES:
            _draw_arrow_glyph(
                draw,
                center,
                (ux, uy),
                color,
                glyph_scale,
                base=7.6,
                tip=10.0,
                wing=1.8,
                shaft_half=geometry.tool_stroke * 0.44 / TOOL_GLYPH_SCALE,
            )
        draw.circle_filled(center, 2.2 * glyph_scale, surface, segments=20)
        draw.circle_filled(center, 1.05 * glyph_scale, color, segments=16)
        draw.centered_label(
            "W" if space == "world" else "B",
            (x + 5.25 * glyph_scale, y - 5.1 * glyph_scale),
            color,
            FRAME_LABEL_MAX_WIDTH * scale,
        )
    else:
        radius = 6.4 * glyph_scale
        local = (
            (-radius, -6.2 * glyph_scale),
            *((ux * radius, uy * radius) for ux, uy in _SNAP_ARC),
            (radius, -6.2 * glyph_scale),
        )
        path = _transform_path(local, x, y, 1.0)
        draw.polyline(path, color, stroke)


def draw_tool_column(
    draw: Draw2D,
    origin,
    theme: Theme,
    scale: float,
    *,
    mode: str,
    space: str,
    snap: bool,
    enabled: bool = True,
    disabled_reason: str = "",
    bindings: InputBindings = DEFAULT_INPUT_BINDINGS,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> str:
    width, height = tool_column_size(scale)
    draw_capsule(draw, origin, width, height, theme, scale)
    x, y = origin
    group_cursor = 0
    for previous in TOOL_GROUPS[:-1]:
        group_cursor += len(previous)
        divider_y = (
            y
            + (TOOL_CONTROL_CENTERS[group_cursor - 1] + TOOL_CONTROL_CENTERS[group_cursor])
            * 0.5
            * scale
        )
        draw.line(
            (x + (SHELL_RADIUS - DIVIDER_WIDTH * 0.5) * scale, divider_y),
            (x + (SHELL_RADIUS + DIVIDER_WIDTH * 0.5) * scale, divider_y),
            (*theme.border[:3], 0.72),
            1.0 * scale,
        )
    result = ""
    for index, control in enumerate(TOOL_CONTROLS):
        kind = control.name
        center = (x + SHELL_RADIUS * scale, y + TOOL_CONTROL_CENTERS[index] * scale)
        selected = (
            (kind == "move" and mode == "translate")
            or (kind == "rotate" and mode == "rotate")
            or (kind == "snap" and snap)
        )
        if _circle_button(
            draw,
            f"##viewport-tool-{kind}",
            center,
            theme,
            scale,
            _tool_icon,
            selected=selected,
            enabled=enabled,
            payload=(kind, space),
        ):
            result = kind
        imgui.set_item_tooltip(
            disabled_reason
            if not enabled and disabled_reason
            else f"{labels.move} ({bindings.label(InputAction.GIZMO_TRANSLATE)})"
            if kind == "move"
            else f"{labels.rotate} ({bindings.label(InputAction.GIZMO_ROTATE)})"
            if kind == "rotate"
            else f"{labels.world_body} ({bindings.label(InputAction.GIZMO_SPACE)})"
            if kind == "frame"
            else f"{labels.snap} ({bindings.label(InputAction.SNAP)})"
        )
    return result


def _inline_text(draw: Draw2D, x: float, center_y: float, value: str, color) -> float:
    width, height = draw.text_size(value)
    draw.text((x, center_y - height * 0.5), color, value)
    return width


def _key_width(draw: Draw2D, label: str, scale: float, text_scale: float = 1.0) -> float:
    return draw.text_size(label)[0] * text_scale + OVERLAY_GEOMETRY.hint_key_padding_x * 2.0 * scale


def _keycap(
    draw: Draw2D, x: float, center_y: float, label: str, theme: Theme, scale: float
) -> float:
    text_width, text_height = draw.text_size(label)
    width = text_width + OVERLAY_GEOMETRY.hint_key_padding_x * 2.0 * scale
    height = OVERLAY_GEOMETRY.hint_control_height * scale
    y = center_y - height * 0.5
    draw.rect_filled((x, y), (x + width, y + height), theme.bg_frame, rounding=3.0 * scale)
    draw.rect((x, y), (x + width, y + height), theme.border, 1.0 * scale, rounding=3.0 * scale)
    draw.text((x + (width - text_width) * 0.5, y + (height - text_height) * 0.5), theme.text, label)
    return width


def _mouse_width(
    draw: Draw2D,
    scale: float,
    suffix: str = "",
    text_scale: float = 1.0,
) -> float:
    width = OVERLAY_GEOMETRY.hint_mouse_width * scale
    return width if not suffix else width + 5.0 * scale + draw.text_size(suffix)[0] * text_scale


@lru_cache(maxsize=128)
def mouse_button_fill_points(
    x: float,
    y: float,
    width: float,
    height: float,
    button: str,
    *,
    outline_width: float,
    divider_width: float,
    safety_inset: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Return a button fill inset from the shell and tucked beneath its dividers."""

    if button not in {"left", "right"}:
        return ()
    split = y + height * 0.44
    outer_radius = min(width * 0.5, height * 0.32)
    shell_inset = outline_width * 0.5 + safety_inset
    left = x + shell_inset
    right = x + width - shell_inset
    top = y + shell_inset
    # The divider is painted after the fill.  Extending underneath half of that
    # stroke avoids the one-pixel dark seam produced by anti-aliasing while the
    # visible fill still ends exactly at the divider.
    bottom = split + divider_width * 0.5
    middle = x + width * 0.5
    radius = max(0.0, outer_radius - shell_inset)
    if bottom <= top or right <= left or radius <= 0.0:
        return ()
    if button == "left":
        inner_middle = middle
        return (
            (inner_middle, top),
            (left + radius, top),
            *((left + radius + ux * radius, top + radius + uy * radius) for ux, uy in _QUARTER_ARC),
            (left, bottom),
            (inner_middle, bottom),
        )
    inner_middle = middle
    return (
        (inner_middle, top),
        (right - radius, top),
        *((right - radius - ux * radius, top + radius + uy * radius) for ux, uy in _QUARTER_ARC),
        (right, bottom),
        (inner_middle, bottom),
    )


def _mouse(
    draw: Draw2D,
    x: float,
    center_y: float,
    button: str,
    suffix: str,
    theme: Theme,
    scale: float,
) -> float:
    width = OVERLAY_GEOMETRY.hint_mouse_width * scale
    height = OVERLAY_GEOMETRY.hint_control_height * scale
    y = center_y - height * 0.5
    split = y + height * 0.44
    radius = min(width * 0.5, height * 0.32)
    outline_width = 1.35 * scale
    divider_width = 1.0 * scale
    points = mouse_button_fill_points(
        x,
        y,
        width,
        height,
        button,
        outline_width=outline_width,
        divider_width=divider_width,
        safety_inset=0.18 * scale,
    )
    if points:
        draw.convex_fill(points, theme.primary_dim)
    draw.rect((x, y), (x + width, y + height), theme.text, outline_width, rounding=radius)
    draw.line(
        (x + width * 0.5, y + scale),
        (x + width * 0.5, split),
        theme.text,
        divider_width,
    )
    draw.line((x + scale, split), (x + width - scale, split), theme.text, divider_width)
    if button == "wheel":
        draw.rect_filled(
            (x + width * 0.40, y + 2.0 * scale),
            (x + width * 0.60, y + 7.0 * scale),
            theme.primary_bright,
            rounding=1.4 * scale,
        )
    if not suffix:
        return width
    label_x = x + width + 5.0 * scale
    return width + 5.0 * scale + _inline_text(draw, label_x, center_y, suffix, theme.primary_bright)


@dataclass(frozen=True)
class _HintGroup:
    kind: str
    control: str = ""
    label: str = ""
    suffix: str = ""


@lru_cache(maxsize=64)
def _hint_groups(
    variant: str,
    bindings: InputBindings,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> tuple[_HintGroup, ...]:
    snap = _HintGroup(
        "key",
        control=bindings.label(InputAction.SNAP),
        label=labels.snap,
    )
    perturb = _HintGroup(
        "perturb",
        control=bindings.label(InputAction.PERTURB),
    )
    if variant == "camera":
        return (
            _HintGroup("mouse", "left", labels.orbit),
            _HintGroup("mouse", "right", labels.pan),
            _HintGroup("mouse", "wheel", labels.zoom),
            _HintGroup("key", bindings.label(InputAction.FRAME_SCENE), labels.frame),
        )
    if variant == "dragging":
        return (snap,)
    if variant == "perturb":
        return (perturb,)
    if variant == "ready_minimal":
        return (_HintGroup("mouse", "left", labels.type_value, "×2"),)
    return (
        snap,
        _HintGroup("key", bindings.label(InputAction.GIZMO_SPACE), labels.world_body),
        _HintGroup("mouse", "left", labels.type_value, "×2"),
        perturb,
    )


def hint_size(
    draw: Draw2D,
    scale: float,
    variant: str,
    *,
    text_scale: float = 1.0,
    space: str = "world",
    bindings: InputBindings = DEFAULT_INPUT_BINDINGS,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> tuple[float, float]:
    gap = OVERLAY_GEOMETRY.hint_group_gap * scale
    chord = OVERLAY_GEOMETRY.hint_chord_gap * scale
    input_gap = OVERLAY_GEOMETRY.hint_input_gap * scale

    def text_width(value: str) -> float:
        return draw.text_size(value)[0] * text_scale

    mouse = _mouse_width(draw, scale, text_scale=text_scale)

    def group_width(group: _HintGroup) -> float:
        if group.kind == "text":
            return text_width(group.label)
        if group.kind == "key":
            return (
                _key_width(draw, group.control, scale, text_scale)
                + input_gap
                + text_width(group.label)
            )
        if group.kind == "mouse":
            return (
                _mouse_width(draw, scale, group.suffix, text_scale)
                + input_gap
                + text_width(group.label)
            )
        return (
            _key_width(draw, group.control, scale, text_scale)
            + input_gap
            + text_width("+")
            + chord
            + text_width(labels.drag)
            + chord
            + mouse
            + input_gap
            + text_width(labels.push)
            + chord
            + mouse
            + input_gap
            + text_width(labels.twist)
        )

    groups = tuple(group_width(group) for group in _hint_groups(variant, bindings, labels))
    return (
        sum(groups) + gap * (len(groups) - 1) + OVERLAY_GEOMETRY.hint_padding_x * 2.0 * scale,
        (OVERLAY_GEOMETRY.hint_control_height + OVERLAY_GEOMETRY.hint_padding_y * 2.0) * scale,
    )


def draw_hint(
    draw: Draw2D,
    origin,
    theme: Theme,
    scale: float,
    variant: str,
    *,
    space: str = "world",
    bindings: InputBindings = DEFAULT_INPUT_BINDINGS,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    size: tuple[float, float] | None = None,
) -> None:
    width, height = size or hint_size(
        draw,
        scale,
        variant,
        space=space,
        bindings=bindings,
        labels=labels,
    )
    draw_capsule(draw, origin, width, height, theme, scale)
    x, y = origin
    cursor = x + OVERLAY_GEOMETRY.hint_padding_x * scale
    center_y = y + height * 0.5
    input_gap = OVERLAY_GEOMETRY.hint_input_gap * scale
    group_gap = OVERLAY_GEOMETRY.hint_group_gap * scale
    chord_gap = OVERLAY_GEOMETRY.hint_chord_gap * scale

    def text(value: str, color=None) -> None:
        nonlocal cursor
        cursor += _inline_text(draw, cursor, center_y, value, color or theme.text)

    def key(label: str, meaning: str) -> None:
        nonlocal cursor
        cursor += _keycap(draw, cursor, center_y, label, theme, scale) + input_gap
        text(meaning)

    def mouse(button: str, meaning: str, *, suffix: str = "", after: float = 0.0) -> None:
        nonlocal cursor
        cursor += _mouse(draw, cursor, center_y, button, suffix, theme, scale) + input_gap
        text(meaning)
        cursor += after

    def perturb(modifier: str) -> None:
        nonlocal cursor
        cursor += _keycap(draw, cursor, center_y, modifier, theme, scale) + input_gap
        text("+", theme.text_disabled)
        cursor += chord_gap
        text(labels.drag, theme.primary_bright)
        cursor += chord_gap
        mouse("left", labels.push, after=chord_gap)
        mouse("right", labels.twist)

    groups = _hint_groups(variant, bindings, labels)
    for index, group in enumerate(groups):
        if group.kind == "text":
            text(group.label)
        elif group.kind == "key":
            key(group.control, group.label)
        elif group.kind == "mouse":
            mouse(group.control, group.label, suffix=group.suffix)
        else:
            perturb(group.control)
        if index + 1 < len(groups):
            divider_x = cursor + group_gap * 0.5
            draw.line(
                (divider_x, center_y - 5.0 * scale),
                (divider_x, center_y + 5.0 * scale),
                (*theme.border[:3], min(0.62, theme.border[3])),
                1.0 * scale,
            )
            cursor += group_gap


def format_simulation_time(value: float) -> str:
    """Format long simulation durations without allowing status width to grow."""

    seconds = max(0.0, float(value))
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    whole = int(seconds)
    milliseconds = round((seconds - whole) * 1000.0)
    if milliseconds == 1000:
        whole += 1
        milliseconds = 0
    minutes, second = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}:{second:02d}.{milliseconds:03d}"
    hours, minute = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}:{minute:02d}:{second:02d}"
    days, hour = divmod(hours, 24)
    return f"{days}d {hour:02d}:{minute:02d}:{second:02d}"


def format_simulation_steps(value: int) -> str:
    """Format large step counts with a stable compact suffix."""

    count = max(0, int(value))
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if count >= threshold:
            compact = f"{count / threshold:.1f}".rstrip("0").rstrip(".")
            return f"{compact}{suffix}"
    return str(count)


def format_simulation_metric(
    mode: str,
    sim_time: float,
    step: int,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> tuple[str, str]:
    """Return compact display copy and an exact clipboard representation."""

    if mode == "steps":
        return f"{labels.steps} {format_simulation_steps(step)}", str(max(0, int(step)))
    value = max(0.0, float(sim_time))
    return f"{labels.time} {format_simulation_time(value)}", f"{value:.17g} s"


def _fit_status_text(draw: Draw2D, value: str, max_width: float) -> str:
    text = " ".join(str(value).split())
    if draw.text_size(text)[0] <= max_width:
        return text
    ellipsis = "…"
    while text and draw.text_size(f"{text}{ellipsis}")[0] > max_width:
        text = text[:-1]
    return f"{text.rstrip()}{ellipsis}" if text else ""


def _status_performance_layout(
    draw: Draw2D,
    right: float,
    scale: float,
    backend: str,
    dt: float,
    fps: float,
) -> _StatusPerformanceLayout:
    """Lay out stable backend, delta-time, and FPS columns without jitter."""

    backend_text = str(backend)
    delta_text = f"Δt {max(0.0, float(dt)) * 1000.0:.1f} ms"
    fps_text = f"{max(0.0, float(fps)):.1f} fps"
    fps_width, _ = draw.text_size(fps_text)
    fps_reference_width, _ = draw.text_size("000.0 fps")
    fps_column_width = max(fps_width, fps_reference_width)
    fps_x = right - fps_width
    fps_divider_x = right - fps_column_width - 11.0 * scale
    delta_right = fps_divider_x - 11.0 * scale
    delta_width, _ = draw.text_size(delta_text)
    delta_reference_width, _ = draw.text_size("Δt 0000.0 ms")
    delta_column_width = max(delta_width, delta_reference_width)
    delta_x = delta_right - delta_width
    delta_divider_x = delta_right - delta_column_width - 11.0 * scale
    backend_width, _ = draw.text_size(backend_text)
    backend_x = delta_divider_x - 11.0 * scale - backend_width
    return _StatusPerformanceLayout(
        backend_text,
        delta_text,
        fps_text,
        backend_x,
        delta_x,
        fps_x,
        delta_divider_x,
        fps_divider_x,
        backend_x,
    )


def draw_status(
    draw: Draw2D,
    origin,
    width: float,
    height: float,
    theme: Theme,
    scale: float,
    *,
    selected: str,
    has_selection: bool,
    state: str,
    sim_time: float,
    step: int,
    metric_mode: str,
    backend: str,
    dt: float,
    fps: float,
    status: str = "",
    status_level: str = "info",
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> StatusLayout:
    x, y = origin
    running = state == "running"
    draw.rect_filled((x, y), (x + width, y + height), (*theme.bg_child[:3], 1.0))
    draw.line(
        (x, y),
        (x + width, y),
        theme.primary_dim if running else theme.border,
        1.0 * scale,
    )
    cy = y + height * 0.5
    cursor = x + 12.0 * scale
    cursor += _inline_text(
        draw,
        cursor,
        cy,
        selected,
        theme.text if has_selection else theme.text_disabled,
    )

    def separator() -> None:
        nonlocal cursor
        cursor += 12.0 * scale
        draw.line(
            (cursor, cy - 6.0 * scale),
            (cursor, cy + 6.0 * scale),
            theme.border,
            1.0 * scale,
        )
        cursor += 12.0 * scale

    separator()
    state_color = theme.primary if running else theme.text_disabled
    draw.circle_filled(
        (cursor + 3.5 * scale, cy),
        3.5 * scale,
        state_color,
        segments=20,
    )
    cursor += 12.0 * scale
    state_text = (
        labels.running if running else labels.static if state == "static" else labels.paused
    )
    cursor += _inline_text(draw, cursor, cy, state_text, state_color)

    metric_rect = None
    metric_exact = ""
    if state != "static":
        separator()
        metric_text, metric_exact = format_simulation_metric(
            metric_mode,
            sim_time,
            step,
            labels,
        )
        text_width, _ = draw.text_size(metric_text)
        pad = 7.0 * scale
        metric_height = min(height - 6.0 * scale, 20.0 * scale)
        metric_rect = (
            cursor,
            cy - metric_height * 0.5,
            cursor + text_width + pad * 2.0,
            cy + metric_height * 0.5,
        )
        draw.rect_filled(
            metric_rect[:2],
            metric_rect[2:],
            theme.bg_frame,
            rounding=3.0 * scale,
        )
        draw.rect(
            metric_rect[:2],
            metric_rect[2:],
            theme.border,
            1.0 * scale,
            rounding=3.0 * scale,
        )
        _inline_text(draw, cursor + pad, cy, metric_text, theme.text)
        cursor = metric_rect[2]

    performance = _status_performance_layout(
        draw,
        x + width - 12.0 * scale,
        scale,
        backend,
        dt,
        fps,
    )
    _inline_text(draw, performance.backend_x, cy, performance.backend_text, theme.text)
    _inline_text(draw, performance.delta_x, cy, performance.delta_text, theme.text)
    _inline_text(draw, performance.fps_x, cy, performance.fps_text, theme.text)
    draw.line(
        (performance.delta_divider_x, cy - 6.0 * scale),
        (performance.delta_divider_x, cy + 6.0 * scale),
        theme.border,
        1.0 * scale,
    )
    draw.line(
        (performance.fps_divider_x, cy - 6.0 * scale),
        (performance.fps_divider_x, cy + 6.0 * scale),
        theme.border,
        1.0 * scale,
    )

    compact_status = " ".join(str(status).split())
    available = performance.left - cursor - 34.0 * scale
    if compact_status and available > 48.0 * scale:
        shown = _fit_status_text(draw, compact_status, available)
        shown_width, _ = draw.text_size(shown)
        status_x = performance.left - 22.0 * scale - shown_width
        status_colors = {
            "error": theme.danger,
            "warning": theme.warning,
            "success": theme.primary_bright,
        }
        _inline_text(
            draw,
            status_x,
            cy,
            shown,
            status_colors.get(status_level, theme.primary_bright),
        )
        divider_x = performance.left - 11.0 * scale
        draw.line(
            (divider_x, cy - 6.0 * scale),
            (divider_x, cy + 6.0 * scale),
            theme.border,
            1.0 * scale,
        )

    return StatusLayout(metric_rect, metric_exact)
