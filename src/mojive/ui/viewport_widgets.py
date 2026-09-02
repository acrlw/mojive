"""Production viewport chrome shared by the interactive viewer.

The glyphs are fixed screen-space geometry.  Curves used by the rotate, snap,
mouse, and capsule shapes are sampled once at import; a frame only scales and
translates those points before submitting them to ``Draw2D``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import pairwise

from imgui_bundle import imgui

from ..gizmo import ARROW_CORNER_RADIUS_PT, _rounded_polygon_corners
from .draw2d import Draw2D, _open_polyline_ribbon
from .input_bindings import DEFAULT_INPUT_BINDINGS, InputAction, InputBindings
from .theme import Theme


@dataclass(frozen=True)
class OverlayGeometry:
    """Shared logical-pixel geometry for viewport chrome and its design probe."""

    icon_radius: float = 10.0
    radial_step: float = 8.0
    center_step: float = 42.0
    tool_center_step: float = 42.0
    tool_group_gap: float = 10.0
    divider_width: float = 20.0
    tool_stroke: float = 1.46
    rotate_ring_gap_ratio: float = 0.5
    rotate_ring_cap: str = "round"
    hint_control_height: float = 18.0
    hint_padding_x: float = 16.0
    hint_padding_y: float = 8.0
    hint_input_gap: float = 8.0
    hint_group_gap: float = 24.0
    hint_chord_gap: float = 10.0
    hint_key_padding_x: float = 8.0
    hint_mouse_width: float = 14.0
    hint_mouse_stroke: float = 1.25
    hint_mouse_button_width_ratio: float = 0.44
    hint_mouse_button_shell_ratio: float = 0.75
    hint_mouse_button_height_ratio: float = 0.40
    hint_mouse_wheel_width_ratio: float = 0.28
    hint_mouse_wheel_height_ratio: float = 7.0 / 18.0
    hint_mouse_wheel_gap_ratio: float = 0.25
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
    def rotate_ring_gap(self) -> float:
        return self.tool_stroke * self.rotate_ring_gap_ratio


@dataclass(frozen=True)
class ViewportLabels:
    """Localized viewport chrome copy, resolved only when language changes."""

    play: str = "Play"
    pause: str = "Pause"
    previous: str = "Previous frame"
    step: str = "Step"
    pause_to_step: str = "Pause to step"
    reset: str = "Reset"
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
    clear_selection: str = "Clear selection"
    show_steps: str = "Click to show steps"
    show_time: str = "Click to show time"
    copy_exact: str = "Right-click to copy exact value"


DEFAULT_VIEWPORT_LABELS = ViewportLabels()


@dataclass(frozen=True)
class ToolHint:
    """One input/meaning pair that can be rendered on any chrome surface."""

    kind: str
    control: str = ""
    label: str = ""
    suffix: str = ""
    hint_id: str = ""


class ToolHintRegistry:
    """User-extensible tool hints for the status bar and viewport scene.

    Defaults are supplied by the viewer each frame because they depend on the
    current interaction mode.  Callers can add custom hints or suppress a
    default by its stable ``hint_id`` without coupling to either renderer.
    """

    _SURFACES = frozenset(("status", "scene"))

    def __init__(self) -> None:
        self._custom: dict[str, dict[str, ToolHint]] = {surface: {} for surface in self._SURFACES}
        self._hidden_defaults: dict[str, set[str]] = {surface: set() for surface in self._SURFACES}

    def add(self, hint_id: str, hint: ToolHint, *, surface: str = "status") -> None:
        """Add or replace one custom hint on ``surface``."""

        target = self._surface(surface)
        key = str(hint_id).strip()
        if not key:
            raise ValueError("tool hint id must not be empty")
        self._custom[target][key] = ToolHint(
            hint.kind,
            hint.control,
            hint.label,
            hint.suffix,
            hint.hint_id or key,
        )
        self._hidden_defaults[target].discard(key)

    def remove(self, hint_id: str, *, surface: str = "status") -> None:
        """Remove a custom hint and suppress a default with the same id."""

        target = self._surface(surface)
        key = str(hint_id).strip()
        self._custom[target].pop(key, None)
        if key:
            self._hidden_defaults[target].add(key)

    def restore(self, hint_id: str, *, surface: str = "status") -> None:
        """Allow a previously suppressed default hint to render again."""

        self._hidden_defaults[self._surface(surface)].discard(str(hint_id).strip())

    def resolve(
        self,
        defaults: Sequence[ToolHint] = (),
        *,
        surface: str = "status",
    ) -> tuple[ToolHint, ...]:
        """Compose visible defaults followed by caller-defined hints."""

        target = self._surface(surface)
        hidden = self._hidden_defaults[target]
        visible = tuple(hint for hint in defaults if not hint.hint_id or hint.hint_id not in hidden)
        return visible + tuple(self._custom[target].values())

    @classmethod
    def _surface(cls, value: str) -> str:
        surface = str(value).strip().lower()
        if surface not in cls._SURFACES:
            raise ValueError(f"unknown tool hint surface: {value!r}")
        return surface


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
    """Stable, progressively collapsible right-edge telemetry columns."""

    backend_text: str
    metric_text: str
    delta_text: str
    fps_text: str
    backend_x: float
    metric_x: float
    metric_width: float
    delta_x: float
    fps_x: float
    dividers: tuple[float, ...]
    left: float


OVERLAY_GEOMETRY = OverlayGeometry()
DEFAULT_VIEWPORT_OVERLAY_SCALE = 1.25
MIN_VIEWPORT_OVERLAY_SCALE = 0.85
MAX_VIEWPORT_OVERLAY_SCALE = 2.0
PLAYBACK_CHROME_SCALE = 0.82
TOOL_CHROME_SCALE = PLAYBACK_CHROME_SCALE
HINT_CHROME_SCALE = PLAYBACK_CHROME_SCALE
# ImGui clips a window draw list at the host window boundary.  A capsule's
# antialiased outline extends beyond its mathematical path, and fractional
# framebuffer scaling can add another pixel.  Keep a logical-pixel guard band
# around every host window so the rounded end is never flattened by that clip.
OVERLAY_CLIP_PADDING = 5.0
# Tool glyphs need a little more optical weight than the playback symbols.
# This scale changes only their authored paths; hit regions, state circles, and
# capsule spacing continue to use the shared overlay geometry.
TOOL_GLYPH_SCALE = 1.18
_MOVE_ARROW_BASE = 5.0
_MOVE_ARROW_TIP = 9.0
_MOVE_ARROW_WING = (_MOVE_ARROW_TIP - _MOVE_ARROW_BASE) / math.sqrt(3.0)
_FRAME_ARROW_CORNER_RADIUS_PT = 0.25
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
# Fixed ISO-orthographic front half-rings. Path order is Y, X, Z; crossings
# cycle Y over X, X over Z, and Z over Y so no complete axis owns one layer.
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

_FRAME_AXES = ((0.0, -1.0), (0.866025, 0.5), (-0.866025, 0.5))


def _cross2(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _segment_intersection(start, end, other_start, other_end):
    direction = (end[0] - start[0], end[1] - start[1])
    other_direction = (
        other_end[0] - other_start[0],
        other_end[1] - other_start[1],
    )
    denominator = _cross2(direction, other_direction)
    if abs(denominator) <= 1e-9:
        return None
    offset = (other_start[0] - start[0], other_start[1] - start[1])
    amount = _cross2(offset, other_direction) / denominator
    other_amount = _cross2(offset, direction) / denominator
    if not (1e-9 < amount < 1.0 - 1e-9 and 1e-9 < other_amount < 1.0 - 1e-9):
        return None
    return (
        amount,
        other_amount,
        (
            start[0] + direction[0] * amount,
            start[1] + direction[1] * amount,
        ),
    )


def _polygon_area(points) -> float:
    return 0.5 * sum(
        start[0] * end[1] - start[1] * end[0]
        for start, end in zip(points, (*points[1:], points[0]), strict=True)
    )


def _counterclockwise(points):
    outline = tuple(points)
    return tuple(reversed(outline)) if _polygon_area(outline) < 0.0 else outline


@lru_cache(maxsize=32)
def _rotate_stroke_outline(
    path: tuple[tuple[float, float], ...],
    width: float,
    cap: str,
) -> tuple[tuple[float, float], ...]:
    """Return one inner ring's filled stroke silhouette in authored coordinates."""

    left, right, outline = _open_polyline_ribbon(path, width)
    if not outline:
        return ()
    if cap == "butt":
        return _counterclockwise(outline)
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
    return _counterclockwise(rounded)


def _point_in_polygon(point, polygon) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (previous[0] - current[0]) * (y - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _polygon_difference(subject, clip):
    """Return simple boundaries for ``subject`` minus one crossing clip polygon."""

    subject_breaks = [[] for _ in subject]
    clip_breaks = [[] for _ in clip]
    for subject_index, start in enumerate(subject):
        end = subject[(subject_index + 1) % len(subject)]
        for clip_index, clip_start in enumerate(clip):
            clip_end = clip[(clip_index + 1) % len(clip)]
            intersection = _segment_intersection(start, end, clip_start, clip_end)
            if intersection is None:
                continue
            amount, clip_amount, point = intersection
            subject_breaks[subject_index].append((amount, point))
            clip_breaks[clip_index].append((clip_amount, point))

    if not any(subject_breaks):
        return () if _point_in_polygon(subject[0], clip) else (_counterclockwise(subject),)

    edges = []

    def append_boundary(polygon, breaks, other, *, keep_inside: bool, reverse: bool) -> None:
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            cuts = [(0.0, start), *breaks[index], (1.0, end)]
            cuts.sort(key=lambda item: item[0])
            for (_, first), (_, second) in pairwise(cuts):
                midpoint = (
                    (first[0] + second[0]) * 0.5,
                    (first[1] + second[1]) * 0.5,
                )
                if _point_in_polygon(midpoint, other) != keep_inside:
                    continue
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
) -> tuple[tuple[tuple[tuple[float, float], ...], ...], ...]:
    """Subtract each front shell from the ring behind it for cyclic occlusion."""

    local_width = tool_stroke / TOOL_GLYPH_SCALE
    shell_width = tool_stroke * (1.0 + 2.0 * ring_gap_ratio) / TOOL_GLYPH_SCALE
    # Path order is Y, X, Z. These occluders produce Y > X, X > Z, Z > Y.
    occluder_by_ring = (2, 0, 1)
    result = []
    for index, path in enumerate(_ROTATE_HALF_RINGS):
        outline = _rotate_stroke_outline(path, local_width, cap)
        shell = _rotate_stroke_outline(
            _ROTATE_HALF_RINGS[occluder_by_ring[index]],
            shell_width,
            cap,
        )
        result.append(_polygon_difference(outline, shell))
    return tuple(result)


@dataclass(frozen=True)
class ViewportControl:
    """One declarative playback/tool action and its optional custom glyph."""

    name: str
    icon: Callable | None = None
    tooltip: str = ""


PLAYBACK_CONTROLS = tuple(ViewportControl(name) for name in ("previous", "toggle", "step", "reset"))
TOOL_GROUPS = (
    tuple(ViewportControl(name) for name in ("move", "rotate", "frame")),
    (ViewportControl("snap"),),
)


def tool_control_centers(
    groups: Sequence[Sequence[ViewportControl]] = TOOL_GROUPS,
) -> tuple[float, ...]:
    centers: list[float] = []
    cursor = SHELL_RADIUS
    for group_index, group in enumerate(groups):
        if group_index:
            cursor += TOOL_GROUP_GAP
        for _control in group:
            centers.append(cursor)
            cursor += OVERLAY_GEOMETRY.tool_center_step
    return tuple(centers)


TOOL_CONTROL_CENTERS = tool_control_centers()
TOOL_CONTROLS = tuple(control for group in TOOL_GROUPS for control in group)


class ViewportChromeRegistry:
    """Mutable extension seam for viewport actions and reusable tool hints."""

    def __init__(self) -> None:
        self.playback_controls: list[ViewportControl] = list(PLAYBACK_CONTROLS)
        self.tool_groups: list[list[ViewportControl]] = [list(group) for group in TOOL_GROUPS]
        self.tool_hints = ToolHintRegistry()
        self._handlers: dict[tuple[str, str], Callable[[], None]] = {}

    def add_playback(
        self,
        control: ViewportControl,
        handler: Callable[[], None],
        *,
        index: int | None = None,
    ) -> None:
        """Register a playback action without changing the viewer draw loop."""

        self._require_custom_icon(control)
        self.playback_controls[:] = [
            item for item in self.playback_controls if item.name != control.name
        ]
        target = len(self.playback_controls) if index is None else int(index)
        self.playback_controls.insert(max(0, min(target, len(self.playback_controls))), control)
        self._handlers[("playback", control.name)] = handler

    def add_tool(
        self,
        control: ViewportControl,
        handler: Callable[[], None],
        *,
        group: int | None = None,
    ) -> None:
        """Register a Tool Column action in an existing or new group."""

        self._require_custom_icon(control)
        self.tool_groups[:] = [
            [item for item in items if item.name != control.name] for items in self.tool_groups
        ]
        self.tool_groups[:] = [items for items in self.tool_groups if items]
        if group is None or group >= len(self.tool_groups):
            self.tool_groups.append([control])
        else:
            self.tool_groups[max(0, int(group))].append(control)
        self._handlers[("tool", control.name)] = handler

    def remove(self, surface: str, name: str) -> None:
        """Remove a registered action while preserving built-in dispatch code."""

        area = str(surface).strip().lower()
        key = str(name)
        if area == "playback":
            self.playback_controls[:] = [
                item for item in self.playback_controls if item.name != key
            ]
        elif area == "tool":
            self.tool_groups[:] = [
                [item for item in group if item.name != key] for group in self.tool_groups
            ]
            self.tool_groups[:] = [group for group in self.tool_groups if group]
        else:
            raise ValueError(f"unknown viewport chrome surface: {surface!r}")
        self._handlers.pop((area, key), None)

    def dispatch(self, surface: str, name: str) -> bool:
        handler = self._handlers.get((str(surface).strip().lower(), str(name)))
        if handler is None:
            return False
        handler()
        return True

    @staticmethod
    def _require_custom_icon(control: ViewportControl) -> None:
        if not control.name:
            raise ValueError("viewport control name must not be empty")
        if control.icon is None:
            raise ValueError("custom viewport controls require an icon callback")


def viewport_chrome_scale(
    style_scale: float,
    overlay_scale: float,
    component_scale: float,
) -> float:
    """Scale transient viewport chrome in the same logical space as panel UI."""

    return float(style_scale) * float(overlay_scale) * float(component_scale)


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


def playback_size(
    scale: float,
    controls: Sequence[ViewportControl] = PLAYBACK_CONTROLS,
) -> tuple[float, float]:
    if not controls:
        return (0.0, 0.0)
    return (
        (SHELL_RADIUS * 2.0 + CENTER_STEP * max(0, len(controls) - 1)) * scale,
        SHELL_RADIUS * 2.0 * scale,
    )


def tool_column_size(
    scale: float,
    groups: Sequence[Sequence[ViewportControl]] = TOOL_GROUPS,
) -> tuple[float, float]:
    centers = tool_control_centers(groups)
    if not centers:
        return (0.0, 0.0)
    last_center = centers[-1]
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
    triangle_radius = 0.8 * scale

    def rounded_triangle(points, geometry_scale: float = 1.0) -> None:
        path = _rounded_polygon_corners(
            points,
            triangle_radius * geometry_scale,
            (0, 1, 2),
            segments=5,
        )
        draw.fringed_concave_fill(tuple(map(tuple, path)), color)

    def play_triangle(cx: float, direction: float, geometry_scale: float):
        g = geometry_scale * scale
        return (
            (cx - direction * 4.6 * g, y - 8.0 * g),
            (cx + direction * 9.2 * g, y),
            (cx - direction * 4.6 * g, y + 8.0 * g),
        )

    if kind == "play":
        rounded_triangle(play_triangle(x, 1.0, 1.0))
    elif kind == "pause":
        draw.rect_filled(
            (x - 5.4 * scale, y - 8.4 * scale),
            (x - 1.0 * scale, y + 8.4 * scale),
            color,
            rounding=0.9 * scale,
        )
        draw.rect_filled(
            (x + 1.0 * scale, y - 8.4 * scale),
            (x + 5.4 * scale, y + 8.4 * scale),
            color,
            rounding=0.9 * scale,
        )
    elif kind in ("previous", "step"):
        direction = -1.0 if kind == "previous" else 1.0
        geometry_scale = 0.78
        # Reuse the play triangle exactly, only mirrored and uniformly scaled.
        # The barrier adds weight on the travel side, so offset the pair back
        # toward the button center as one optical unit.
        icon_x = x - direction * 2.7 * scale
        rounded_triangle(
            play_triangle(icon_x, direction, geometry_scale),
            geometry_scale,
        )
        barrier_x = icon_x + direction * 8.3 * scale
        draw.rect_filled(
            (barrier_x - 0.7 * scale, y - 5.4 * scale),
            (barrier_x + 0.7 * scale, y + 5.4 * scale),
            color,
            rounding=0.7 * scale,
        )
    elif kind in ("reset", "stop"):
        draw.rect_filled(
            (x - 6.8 * scale, y - 6.8 * scale),
            (x + 6.8 * scale, y + 6.8 * scale),
            color,
            rounding=1.0 * scale,
        )


def _play_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "play")


def _pause_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "pause")


def _step_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "step")


def _previous_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "previous")


def _reset_icon(draw: Draw2D, center, color, scale: float, _payload) -> None:
    draw_playback_glyph(draw, center, color, scale, "reset")


def _viewport_tooltip_padding(scale: float) -> tuple[float, float]:
    return (
        OVERLAY_GEOMETRY.tooltip_padding_x * scale,
        OVERLAY_GEOMETRY.tooltip_padding_y * scale,
    )


def _set_viewport_tooltip(text: str, scale: float) -> None:
    padding_x, padding_y = _viewport_tooltip_padding(scale)
    imgui.push_style_var(
        imgui.StyleVar_.window_padding,
        imgui.ImVec2(padding_x, padding_y),
    )
    try:
        imgui.set_item_tooltip(text)
    finally:
        imgui.pop_style_var()


def draw_playback(
    draw: Draw2D,
    origin,
    theme: Theme,
    scale: float,
    *,
    playing: bool,
    step_enabled: bool,
    previous_enabled: bool = False,
    enabled: bool = True,
    bindings: InputBindings = DEFAULT_INPUT_BINDINGS,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    control_specs: Sequence[ViewportControl] = PLAYBACK_CONTROLS,
) -> str:
    width, height = playback_size(scale, control_specs)
    draw_capsule(draw, origin, width, height, theme, scale)
    x, y = origin
    result = ""
    states = {
        "toggle": (_pause_icon if playing else _play_icon, playing, True),
        "previous": (_previous_icon, False, previous_enabled),
        "step": (_step_icon, False, step_enabled),
        "reset": (_reset_icon, False, True),
        # Preserve custom registries created against the earlier control name.
        "stop": (_reset_icon, False, True),
    }
    for index, control in enumerate(control_specs):
        name = control.name
        icon, selected, action_enabled = states.get(name, (control.icon, False, True))
        if icon is None:
            continue
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
        tooltip = control.tooltip
        if not tooltip:
            tooltip = (
                f"{labels.pause} ({bindings.label(InputAction.TOGGLE_PAUSE)})"
                if name == "toggle" and playing
                else f"{labels.play} ({bindings.label(InputAction.TOGGLE_PAUSE)})"
                if name == "toggle"
                else labels.previous
                if name == "previous"
                else labels.step
                if name == "step" and action_enabled
                else labels.pause_to_step
                if name == "step"
                else labels.reset
                if name in ("reset", "stop")
                else name
            )
        _set_viewport_tooltip(tooltip, scale)
    return result


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
    near_x = x - 5.2 * s
    far_x = x + 5.2 * s
    width = max(1.0, 1.25 * s)
    draw.line((near_x, y - half_near), (near_x, y + half_near), color, width, cap="round")
    draw.line((far_x, y - half_far), (far_x, y + half_far), color, width, cap="round")
    draw.line((near_x, y - half_near), (far_x, y - half_far), color, width, cap="round")
    draw.line((near_x, y + half_near), (far_x, y + half_far), color, width, cap="round")


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

    polygon = (
        point(0.0, shaft_half),
        point(base, shaft_half),
        point(base, wing),
        point(tip, 0.0),
        point(base, -wing),
        point(base, -shaft_half),
        point(0.0, -shaft_half),
    )
    return tuple(
        map(
            tuple,
            _rounded_polygon_corners(
                polygon,
                ARROW_CORNER_RADIUS_PT * scale,
                (2, 3, 4),
            ),
        )
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
    polygon = tuple((x + px * scale, y + py * scale) for px, py in local)
    return tuple(
        map(
            tuple,
            _rounded_polygon_corners(
                polygon,
                ARROW_CORNER_RADIUS_PT * scale,
                (23, 0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19),
            ),
        )
    )


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
) -> None:
    """Draw a coordinate axis with a native-AA shaft and rounded arrowhead."""

    ux, uy = (float(value) for value in direction)
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    nx, ny = -uy, ux
    x, y = (float(value) for value in center)

    # A filled ribbon receives Draw2D's one-pixel fringe outside its authored
    # width. That fixed fringe dominates each independent narrow shaft at small
    # UI scales; Move is one connected silhouette and Rotate's comparable ring
    # uses stroke semantics. Submit this shaft as an actual stroke so
    # ``stroke_width`` includes its AA coverage and remains proportional.
    draw.line(
        (x + ux * clear_radius, y + uy * clear_radius),
        (
            x + ux * (base + 0.25) * geometry_scale,
            y + uy * (base + 0.25) * geometry_scale,
        ),
        color,
        stroke_width,
    )
    draw.fringed_concave_fill(
        tuple(
            (
                x + ux * along + nx * across,
                y + uy * along + ny * across,
            )
            for along, across in _rounded_axis_head_path(
                base,
                tip,
                wing,
                geometry_scale,
                corner_radius,
            )
        ),
        color,
    )


@lru_cache(maxsize=64)
def _rounded_axis_head_path(
    base: float,
    tip: float,
    wing: float,
    geometry_scale: float,
    corner_radius: float,
) -> tuple[tuple[float, float], ...]:
    """Cache one local Tool Column arrowhead at its effective UI scale."""

    return tuple(
        map(
            tuple,
            _rounded_polygon_corners(
                (
                    (tip * geometry_scale, 0.0),
                    (base * geometry_scale, wing * geometry_scale),
                    (base * geometry_scale, -wing * geometry_scale),
                ),
                corner_radius,
                (0, 1, 2),
            ),
        )
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
                _MOVE_ARROW_BASE,
                _MOVE_ARROW_TIP,
                _MOVE_ARROW_WING,
                geometry.tool_stroke * 0.5 / TOOL_GLYPH_SCALE,
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
        ):
            for local in ring:
                path = _transform_path(local, x, y, glyph_scale)
                draw.concave_fill(path, color)
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
                base=7.6,
                tip=10.0,
                wing=1.8,
                corner_radius=_FRAME_ARROW_CORNER_RADIUS_PT * scale,
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
    groups: Sequence[Sequence[ViewportControl]] = TOOL_GROUPS,
) -> str:
    width, height = tool_column_size(scale, groups)
    draw_capsule(draw, origin, width, height, theme, scale)
    x, y = origin
    centers = tool_control_centers(groups)
    controls = tuple(control for group in groups for control in group)
    group_cursor = 0
    for previous in groups[:-1]:
        group_cursor += len(previous)
        divider_y = y + (centers[group_cursor - 1] + centers[group_cursor]) * 0.5 * scale
        draw.line(
            (x + (SHELL_RADIUS - DIVIDER_WIDTH * 0.5) * scale, divider_y),
            (x + (SHELL_RADIUS + DIVIDER_WIDTH * 0.5) * scale, divider_y),
            (*theme.border[:3], 0.72),
            1.0 * scale,
        )
    result = ""
    for index, control in enumerate(controls):
        kind = control.name
        center = (x + SHELL_RADIUS * scale, y + centers[index] * scale)
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
            control.icon or _tool_icon,
            selected=selected,
            enabled=enabled,
            payload=(kind, space),
        ):
            result = kind
        _set_viewport_tooltip(
            disabled_reason
            if not enabled and disabled_reason
            else control.tooltip
            if control.tooltip
            else f"{labels.move} ({bindings.label(InputAction.GIZMO_TRANSLATE)})"
            if kind == "move"
            else f"{labels.rotate} ({bindings.label(InputAction.GIZMO_ROTATE)})"
            if kind == "rotate"
            else f"{labels.world_body} ({bindings.label(InputAction.GIZMO_SPACE)})"
            if kind == "frame"
            else f"{labels.snap} ({bindings.label(InputAction.SNAP)})"
            if kind == "snap"
            else kind,
            scale,
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
) -> MouseButtonGeometry | None:
    """Return true-knockout shell and fill geometry for one mouse button."""

    if button not in {"left", "right"}:
        return None
    half_stroke = outline_width * 0.5
    shell_gap = outline_width * geometry.hint_mouse_button_shell_ratio
    button_bottom = y + height * geometry.hint_mouse_button_height_ratio
    shell_radius = min(width * 0.22, height * 0.18)
    outer_radius = shell_radius + half_stroke
    outer_left = x - half_stroke
    outer_top = y - half_stroke
    button_width = (width + outline_width) * geometry.hint_mouse_button_width_ratio
    inner_edge = outer_left + button_width
    arc_steps = 12

    def arc(
        center_x: float,
        center_y: float,
        radius: float,
        start: float,
        end: float,
    ) -> tuple[tuple[float, float], ...]:
        return tuple(
            (
                center_x + math.cos(start + (end - start) * step / arc_steps) * radius,
                center_y + math.sin(start + (end - start) * step / arc_steps) * radius,
            )
            for step in range(arc_steps + 1)
        )

    fill = (
        (inner_edge, outer_top),
        *arc(
            x + shell_radius,
            y + shell_radius,
            outer_radius,
            -math.pi * 0.5,
            -math.pi,
        ),
        (outer_left, button_bottom),
        (inner_edge, button_bottom),
    )
    visible_shell = (
        (inner_edge + shell_gap, y),
        *arc(
            x + width - shell_radius,
            y + shell_radius,
            shell_radius,
            -math.pi * 0.5,
            0.0,
        ),
        *arc(
            x + width - shell_radius,
            y + height - shell_radius,
            shell_radius,
            0.0,
            math.pi * 0.5,
        ),
        *arc(
            x + shell_radius,
            y + height - shell_radius,
            shell_radius,
            math.pi * 0.5,
            math.pi,
        ),
        (x, button_bottom + shell_gap),
    )
    if button == "right":
        mirror_x = x * 2.0 + width
        visible_shell = tuple((mirror_x - point[0], point[1]) for point in visible_shell)
        # The mirrored outline has ImGui's expected clockwise screen winding.
        fill = tuple((mirror_x - point[0], point[1]) for point in fill)
    else:
        # Match the right button's clockwise screen winding so both convex
        # fills receive the same outward antialias fringe.
        fill = tuple(reversed(fill))
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


def draw_mouse_hint_glyph(
    draw: Draw2D,
    x: float,
    center_y: float,
    button: str,
    suffix: str,
    theme: Theme,
    scale: float,
    *,
    size: tuple[float, float] | None = None,
    pixel_size: float = 1.0,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
) -> float:
    """Draw a Blender-style mouse shell with one semantic control highlighted."""

    logical_width, logical_height = size or (
        geometry.hint_mouse_width,
        geometry.hint_control_height,
    )
    width = logical_width * scale
    height = logical_height * scale
    y = center_y - height * 0.5
    radius = min(width * 0.22, height * 0.18)
    outline_width = geometry.hint_mouse_stroke * scale
    button_geometry = mouse_button_geometry(
        x,
        y,
        width,
        height,
        button,
        outline_width=outline_width,
        geometry=geometry,
    )
    if button_geometry is None:
        draw.rect((x, y), (x + width, y + height), theme.text, outline_width, rounding=radius)
    else:
        # Omit the shell beneath the button's transparent outer contour instead
        # of repainting it with a guessed background color. This remains a
        # genuine knockout over translucent chrome and arbitrary viewports.
        draw.polyline(button_geometry.visible_shell, theme.text, outline_width)
        draw.convex_fill(button_geometry.fill, theme.primary)
    if button == "wheel":
        wheel = mouse_wheel_geometry(
            x,
            y,
            width,
            height,
            outline_width=outline_width,
            pixel_size=pixel_size,
            geometry=geometry,
        )
        draw.rect_filled(
            wheel.lo,
            wheel.hi,
            theme.primary,
            rounding=wheel.rounding,
        )
    if not suffix:
        return width
    label_x = x + width + 5.0 * scale
    return width + 5.0 * scale + _inline_text(draw, label_x, center_y, suffix, theme.primary_bright)


@lru_cache(maxsize=64)
def default_tool_hints(
    variant: str,
    bindings: InputBindings,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> tuple[ToolHint, ...]:
    """Return the context defaults independently of their render surface."""

    snap = ToolHint(
        "key",
        control=bindings.label(InputAction.SNAP),
        label=labels.snap,
        hint_id="snap",
    )
    perturb = ToolHint(
        "perturb",
        control=bindings.label(InputAction.PERTURB),
        hint_id="perturb",
    )
    if variant == "camera":
        return (
            ToolHint("mouse", "left", labels.orbit, hint_id="camera.orbit"),
            ToolHint("mouse", "right", labels.pan, hint_id="camera.pan"),
            ToolHint("mouse", "wheel", labels.zoom, hint_id="camera.zoom"),
            ToolHint(
                "key",
                bindings.label(InputAction.FRAME_SCENE),
                labels.frame,
                hint_id="camera.frame",
            ),
        )
    if variant == "dragging":
        return (snap,)
    if variant == "perturb":
        return (perturb,)
    if variant == "ready_minimal":
        return (
            ToolHint(
                "mouse",
                "left",
                labels.type_value,
                "×2",
                hint_id="gizmo.type_value",
            ),
        )
    return (
        snap,
        ToolHint(
            "key",
            bindings.label(InputAction.GIZMO_SPACE),
            labels.world_body,
            hint_id="gizmo.space",
        ),
        ToolHint(
            "mouse",
            "left",
            labels.type_value,
            "×2",
            hint_id="gizmo.type_value",
        ),
        perturb,
    )


# Compatibility for downstream code that used the earlier private name. New
# integrations should consume the public data model and renderer above.
_HintGroup = ToolHint
_hint_groups = default_tool_hints


def _tool_hint_width(
    draw: Draw2D,
    scale: float,
    group: ToolHint,
    *,
    text_scale: float = 1.0,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> float:
    chord = OVERLAY_GEOMETRY.hint_chord_gap * scale
    input_gap = OVERLAY_GEOMETRY.hint_input_gap * scale

    def text_width(value: str) -> float:
        return draw.text_size(value)[0] * text_scale

    mouse = _mouse_width(draw, scale, text_scale=text_scale)
    if group.kind == "text":
        return text_width(group.label)
    if group.kind == "key":
        return (
            _key_width(draw, group.control, scale, text_scale) + input_gap + text_width(group.label)
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


def tool_hints_size(
    draw: Draw2D,
    scale: float,
    hints: Sequence[ToolHint],
    *,
    text_scale: float = 1.0,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    padding: bool = False,
) -> tuple[float, float]:
    """Measure reusable hints without assuming a capsule or status surface."""

    groups = tuple(hints)
    gap = OVERLAY_GEOMETRY.hint_group_gap * scale
    padding_x = OVERLAY_GEOMETRY.hint_padding_x * 2.0 * scale if padding else 0.0
    padding_y = OVERLAY_GEOMETRY.hint_padding_y * 2.0 * scale if padding else 0.0
    return (
        sum(
            _tool_hint_width(draw, scale, group, text_scale=text_scale, labels=labels)
            for group in groups
        )
        + gap * max(0, len(groups) - 1)
        + padding_x,
        OVERLAY_GEOMETRY.hint_control_height * scale + padding_y,
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
    del space
    return tool_hints_size(
        draw,
        scale,
        default_tool_hints(variant, bindings, labels),
        text_scale=text_scale,
        labels=labels,
        padding=True,
    )


def fitting_tool_hints(
    draw: Draw2D,
    scale: float,
    hints: Sequence[ToolHint],
    max_width: float,
    *,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> tuple[ToolHint, ...]:
    """Return the largest whole-hint prefix that fits ``max_width``."""

    fitted: list[ToolHint] = []
    used = 0.0
    gap = OVERLAY_GEOMETRY.hint_group_gap * scale
    for hint in hints:
        width = _tool_hint_width(draw, scale, hint, labels=labels)
        candidate = used + (gap if fitted else 0.0) + width
        if candidate > max(0.0, float(max_width)):
            break
        fitted.append(hint)
        used = candidate
    return tuple(fitted)


def draw_tool_hints(
    draw: Draw2D,
    origin,
    theme: Theme,
    scale: float,
    hints: Sequence[ToolHint],
    *,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    pixel_size: float = 1.0,
) -> float:
    """Draw tool hints inline and return their consumed width."""

    x, center_y = float(origin[0]), float(origin[1])
    cursor = x
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
        cursor += (
            draw_mouse_hint_glyph(
                draw,
                cursor,
                center_y,
                button,
                suffix,
                theme,
                scale,
                pixel_size=pixel_size,
            )
            + input_gap
        )
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

    groups = tuple(hints)
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
    return cursor - x


def draw_scene_tool_hints(
    draw: Draw2D,
    origin,
    theme: Theme,
    scale: float,
    hints: Sequence[ToolHint],
    *,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    size: tuple[float, float] | None = None,
    pixel_size: float = 1.0,
) -> None:
    """Render arbitrary hints in the original viewport capsule surface."""

    width, height = size or tool_hints_size(draw, scale, hints, labels=labels, padding=True)
    draw_capsule(draw, origin, width, height, theme, scale)
    draw_tool_hints(
        draw,
        (
            float(origin[0]) + OVERLAY_GEOMETRY.hint_padding_x * scale,
            float(origin[1]) + height * 0.5,
        ),
        theme,
        scale,
        hints,
        labels=labels,
        pixel_size=pixel_size,
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
    pixel_size: float = 1.0,
) -> None:
    del space
    draw_scene_tool_hints(
        draw,
        origin,
        theme,
        scale,
        default_tool_hints(variant, bindings, labels),
        labels=labels,
        size=size,
        pixel_size=pixel_size,
    )


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
    *,
    metric_text: str = "",
    max_width: float | None = None,
) -> _StatusPerformanceLayout:
    """Lay out telemetry without jitter and collapse it before it can overlap.

    The visual order is backend, simulation metric, delta time, and FPS.  When
    space is constrained, backend and FPS yield first, followed by the metric;
    delta time remains until even its compact form no longer fits.
    """

    backend_text = str(backend)
    delta_text = f"Δt {max(0.0, float(dt)):.6g} s"
    fps_text = f"{max(0.0, float(fps)):.1f} fps"
    metric_text = str(metric_text)

    actual = {
        "backend": draw.text_size(backend_text)[0],
        "metric": (draw.text_size(metric_text)[0] + 14.0 * scale if metric_text else 0.0),
        "delta": draw.text_size(delta_text)[0],
        "fps": draw.text_size(fps_text)[0],
    }
    reserved = {
        "backend": actual["backend"],
        "metric": actual["metric"],
        "delta": max(actual["delta"], draw.text_size("Δt 0.000000 s")[0]),
        "fps": max(actual["fps"], draw.text_size("000.0 fps")[0]),
    }
    gap = 22.0 * scale

    def required(names: set[str]) -> float:
        count = sum(1 for name in ("backend", "metric", "delta", "fps") if name in names)
        return sum(reserved[name] for name in names) + max(0, count - 1) * gap

    limit = float("inf") if max_width is None else max(0.0, float(max_width))
    visible: set[str] = set()
    if reserved["delta"] <= limit:
        visible.add("delta")
    else:
        compact = f"Δt {max(0.0, float(dt)):.4g}s"
        compact_width = draw.text_size(compact)[0]
        if compact_width <= limit:
            delta_text = compact
            actual["delta"] = compact_width
            reserved["delta"] = compact_width
            visible.add("delta")

    # Preserve the simulation metric beside delta time before spending scarce
    # width on FPS or the backend label.
    if "delta" in visible:
        for name in ("metric", "fps", "backend"):
            if reserved[name] <= 0.0:
                continue
            candidate = {*visible, name}
            if required(candidate) <= limit:
                visible = candidate

    order = [name for name in ("backend", "metric", "delta", "fps") if name in visible]
    positions = {name: float(right) for name in ("backend", "metric", "delta", "fps")}
    dividers: list[float] = []
    cursor = float(right)
    for reverse_index, name in enumerate(reversed(order)):
        positions[name] = cursor - actual[name]
        cursor -= reserved[name]
        if reverse_index < len(order) - 1:
            dividers.append(cursor - gap * 0.5)
            cursor -= gap

    return _StatusPerformanceLayout(
        backend_text if "backend" in visible else "",
        metric_text if "metric" in visible else "",
        delta_text if "delta" in visible else "",
        fps_text if "fps" in visible else "",
        positions["backend"],
        positions["metric"],
        actual["metric"] if "metric" in visible else 0.0,
        positions["delta"],
        positions["fps"],
        tuple(dividers),
        cursor,
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
    tool_hints: Sequence[ToolHint] = (),
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    pixel_size: float = 1.0,
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
    state_width = draw.text_size(state_text)[0]
    if cursor + state_width <= x + width - 12.0 * scale:
        cursor += _inline_text(draw, cursor, cy, state_text, state_color)

    metric_text = ""
    metric_exact = ""
    if state != "static":
        metric_text, metric_exact = format_simulation_metric(
            metric_mode,
            sim_time,
            step,
            labels,
        )
    selected_width = draw.text_size(selected)[0]
    # Reserve a useful selection fragment on ordinary windows. At genuinely
    # narrow sizes selection yields to the stable simulation telemetry instead
    # of colliding with it.
    selection_reserve = (
        min(selected_width, 88.0 * scale) + 24.0 * scale if width >= 340.0 * scale else 0.0
    )
    telemetry_budget = max(
        0.0,
        x + width - 12.0 * scale - cursor - 18.0 * scale - selection_reserve,
    )
    performance = _status_performance_layout(
        draw,
        x + width - 12.0 * scale,
        scale,
        backend,
        dt,
        fps,
        metric_text=metric_text,
        max_width=telemetry_budget,
    )
    telemetry_fields = (
        performance.backend_text,
        performance.metric_text,
        performance.delta_text,
        performance.fps_text,
    )
    left_limit = performance.left - (18.0 * scale if any(telemetry_fields) else 0.0)

    selection_available = left_limit - cursor - 24.0 * scale
    if selection_available >= draw.text_size("…")[0]:
        selected_shown = _fit_status_text(draw, selected, selection_available)
        if selected_shown:
            separator()
            cursor += _inline_text(
                draw,
                cursor,
                cy,
                selected_shown,
                theme.text if has_selection else theme.text_disabled,
            )
    left_neighbor_end = cursor

    metric_rect = None
    compact_status = " ".join(str(status).split())
    available = performance.left - cursor - 34.0 * scale
    shown = ""
    shown_width = 0.0
    status_x = performance.left - 22.0 * scale
    if compact_status and available > 48.0 * scale:
        # A transient report remains readable without evicting every context
        # hint from a wide status bar.
        shown = _fit_status_text(draw, compact_status, min(available, width * 0.28))
        shown_width, _ = draw.text_size(shown)
        status_x = performance.left - 22.0 * scale - shown_width

    hint_right = status_x - (14.0 * scale if shown else 0.0)
    hint_available = hint_right - cursor - 24.0 * scale
    fitted_hints = fitting_tool_hints(
        draw,
        scale,
        tool_hints,
        hint_available,
        labels=labels,
    )
    if fitted_hints:
        separator()
        hint_width = draw_tool_hints(
            draw,
            (cursor, cy),
            theme,
            scale,
            fitted_hints,
            labels=labels,
            pixel_size=pixel_size,
        )
        left_neighbor_end = cursor + hint_width

    if shown:
        draw.line(
            (status_x - 7.0 * scale, cy - 6.0 * scale),
            (status_x - 7.0 * scale, cy + 6.0 * scale),
            theme.border,
            1.0 * scale,
        )
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
        left_neighbor_end = status_x + shown_width
    telemetry_present = any(telemetry_fields)
    leading_gap = performance.left - left_neighbor_end
    # A separator belongs between neighboring groups, never at the edge of a
    # right-aligned telemetry island. This also adapts when future fields are
    # added or current fields collapse on narrow windows.
    if telemetry_present and 0.0 < leading_gap <= 30.0 * scale:
        divider_x = left_neighbor_end + leading_gap * 0.5
        draw.line(
            (divider_x, cy - 6.0 * scale),
            (divider_x, cy + 6.0 * scale),
            theme.border,
            1.0 * scale,
        )
    if performance.backend_text:
        _inline_text(draw, performance.backend_x, cy, performance.backend_text, theme.text)
    if performance.metric_text:
        metric_height = min(height - 6.0 * scale, 20.0 * scale)
        metric_rect = (
            performance.metric_x,
            cy - metric_height * 0.5,
            performance.metric_x + performance.metric_width,
            cy + metric_height * 0.5,
        )
        draw.rect_filled(metric_rect[:2], metric_rect[2:], theme.bg_frame, rounding=3.0 * scale)
        draw.rect(
            metric_rect[:2],
            metric_rect[2:],
            theme.border,
            1.0 * scale,
            rounding=3.0 * scale,
        )
        _inline_text(draw, performance.metric_x + 7.0 * scale, cy, metric_text, theme.text)
    if performance.delta_text:
        _inline_text(draw, performance.delta_x, cy, performance.delta_text, theme.text)
    if performance.fps_text:
        _inline_text(draw, performance.fps_x, cy, performance.fps_text, theme.text)
    for divider_x in performance.dividers:
        draw.line(
            (divider_x, cy - 6.0 * scale),
            (divider_x, cy + 6.0 * scale),
            theme.border,
            1.0 * scale,
        )

    return StatusLayout(metric_rect, metric_exact if metric_rect is not None else "")
