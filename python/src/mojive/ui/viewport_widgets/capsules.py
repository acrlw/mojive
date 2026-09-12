"""Viewport widgets: capsules."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import fields
from functools import lru_cache

from imgui_bundle import imgui

from mojive.drawing.curves import (
    smooth_capsule_points,
)
from mojive.ui.draw2d import Draw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction, InputBindings
from mojive.ui.theme import Theme

from .glyphs import (
    _pause_icon,
    _play_icon,
    _previous_icon,
    _record_icon,
    _recording_options_icon,
    _reset_icon,
    _step_icon,
    _tool_icon,
)
from .model import (
    CAPSULE_SMOOTHING,
    CENTER_STEP,
    DEFAULT_VIEWPORT_LABELS,
    DIVIDER_WIDTH,
    OVERLAY_GEOMETRY,
    PLAYBACK_CONTROLS,
    PLAYBACK_HALF_HEIGHT_PT,
    SHELL_RADIUS,
    STATE_RADIUS,
    TOOL_GLYPH_SCALE,
    TOOL_GROUP_GAP,
    TOOL_GROUPS,
    ViewportControl,
    ViewportLabels,
    tool_control_centers,
)


def localized_viewport_labels(translate: Callable[[str], str]) -> ViewportLabels:
    """Resolve the compact chrome catalog through the application's localizer."""

    return ViewportLabels(
        **{
            item.name: translate(getattr(DEFAULT_VIEWPORT_LABELS, item.name))
            for item in fields(DEFAULT_VIEWPORT_LABELS)
        }
    )


def positioned_overlay_rect(
    viewport: tuple[float, float, float, float],
    size: tuple[float, float],
    position: tuple[float, float] | None,
    default_center: tuple[float, float],
    *,
    margin: float = 4.0,
) -> tuple[float, float, float, float]:
    """Place an overlay by normalized center and clamp it inside the viewport."""

    x, y, width, height = (float(value) for value in viewport)
    item_width, item_height = (max(0.0, float(value)) for value in size)
    if position is None:
        center_x, center_y = default_center
    else:
        center_x = x + width * min(1.0, max(0.0, float(position[0])))
        center_y = y + height * min(1.0, max(0.0, float(position[1])))
    half_width, half_height = item_width * 0.5, item_height * 0.5
    guard = max(0.0, float(margin))
    min_x, max_x = x + half_width + guard, x + width - half_width - guard
    min_y, max_y = y + half_height + guard, y + height - half_height - guard
    center_x = x + width * 0.5 if min_x > max_x else min(max(center_x, min_x), max_x)
    center_y = y + height * 0.5 if min_y > max_y else min(max(center_y, min_y), max_y)
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def normalized_overlay_position(
    viewport: tuple[float, float, float, float],
    rect: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return one overlay center in viewport-relative coordinates."""

    x, y, width, height = (float(value) for value in viewport)
    center_x = (float(rect[0]) + float(rect[2])) * 0.5
    center_y = (float(rect[1]) + float(rect[3])) * 0.5
    return (
        min(1.0, max(0.0, (center_x - x) / max(width, 1e-6))),
        min(1.0, max(0.0, (center_y - y) / max(height, 1e-6))),
    )


def overlay_border_hit(
    point: tuple[float, float],
    rect: tuple[float, float, float, float],
    thickness: float,
) -> bool:
    """Return whether a point lies on the draggable capsule border band."""

    px, py = float(point[0]), float(point[1])
    x0, y0, x1, y1 = (float(value) for value in rect)
    if not (x0 <= px <= x1 and y0 <= py <= y1):
        return False
    inset = max(0.0, float(thickness))
    return not (x0 + inset < px < x1 - inset and y0 + inset < py < y1 - inset)


TOOL_CONTROL_CENTERS = tool_control_centers()


def viewport_chrome_scale(
    style_scale: float,
    overlay_scale: float,
    component_scale: float,
) -> float:
    """Scale transient viewport chrome in the same logical space as panel UI."""

    return float(style_scale) * float(overlay_scale) * float(component_scale)


@lru_cache(maxsize=64)
def capsule_points(
    x: float, y: float, width: float, height: float, smoothing: float = CAPSULE_SMOOTHING
):
    return smooth_capsule_points(x, y, width, height, smoothing)


def draw_capsule(
    draw: Draw2D,
    origin,
    width: float,
    height: float,
    theme: Theme,
    scale: float,
) -> None:
    points = capsule_points(float(origin[0]), float(origin[1]), width, height)
    draw.convex_fill(points, theme.viewport.surface)
    draw.polyline(points, theme.viewport.outline, 1.4 * scale, closed=True)


def overlay_divider_length(width: float = DIVIDER_WIDTH, *, playback: bool) -> float:
    """Keep the same divider-to-glyph proportion for filled playback and line tools.

    Width uses the tool glyph's reference diameter. Playback's smaller filled
    symbols use their visible height, independently of hit targets and shell size.
    """
    if playback:
        return width * PLAYBACK_HALF_HEIGHT_PT / (OVERLAY_GEOMETRY.icon_radius * TOOL_GLYPH_SCALE)
    return width


def draw_overlay_divider(
    draw: Draw2D,
    center,
    theme: Theme,
    scale: float,
    *,
    playback: bool,
    width: float = DIVIDER_WIDTH,
) -> None:
    """Draw the shared separator for horizontal playback and vertical tool capsules."""
    half = overlay_divider_length(width, playback=playback) * scale * 0.5
    dx, dy = (0.0, half) if playback else (half, 0.0)
    x, y = center
    draw.line((x - dx, y - dy), (x + dx, y + dy), theme.viewport.divider, scale)


def playback_control_centers(controls: Sequence[ViewportControl] = PLAYBACK_CONTROLS):
    """Keep group boundaries declarative when callers extend the toolbar."""
    cursor = SHELL_RADIUS
    centers = []
    for index, control in enumerate(controls):
        if index and control.name in ("reset", "record"):
            cursor += TOOL_GROUP_GAP
        centers.append(cursor)
        cursor += CENTER_STEP
    return tuple(centers)


def playback_size(scale: float, controls: Sequence[ViewportControl] = PLAYBACK_CONTROLS):
    centers = playback_control_centers(controls)
    return (
        ((centers[-1] + SHELL_RADIUS) * scale, SHELL_RADIUS * 2 * scale) if centers else (0.0, 0.0)
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
    background, foreground = _viewport_control_colors(
        theme, selected=selected, hovered=hovered, active=active, enabled=enabled
    )
    if background[3] > 0.0:
        draw.circle_filled(center, STATE_RADIUS * scale, background)
    surface = background if background[3] > 0.0 else theme.viewport.surface
    icon(draw, center, foreground, scale, (surface, payload))
    return bool(clicked and enabled)


def _viewport_control_colors(
    theme: Theme, *, selected: bool, hovered: bool, active: bool, enabled: bool
):
    """Resolve one capsule control state with press taking visual precedence."""

    colors = theme.viewport
    if not enabled:
        return colors.disabled_background, colors.disabled_foreground
    if active:
        return colors.press_background, colors.press_foreground
    if selected:
        return colors.on_background, colors.on_foreground
    if hovered:
        return colors.hover_background, colors.hover_foreground
    return colors.off_background, colors.off_foreground


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
    recording: bool = False,
    record_enabled: bool = False,
    record_action: str = "stop",
    record_tooltip: str = "",
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
        "record": (_record_icon, recording, record_enabled),
        "recording-options": (_recording_options_icon, False, True),
        # Preserve custom registries created against the earlier control name.
        "stop": (_reset_icon, False, True),
    }
    centers = playback_control_centers(control_specs)
    for index, control in enumerate(control_specs):
        if index and control.name in ("reset", "record"):
            divider_x = x + (centers[index - 1] + centers[index]) * 0.5 * scale
            draw_overlay_divider(
                draw,
                (divider_x, y + SHELL_RADIUS * scale),
                theme,
                scale,
                playback=True,
            )
        name = control.name
        icon, selected, action_enabled = states.get(name, (control.icon, False, True))
        if icon is None:
            continue
        center = (
            x + centers[index] * scale,
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
            payload=(recording, theme.viewport.record, record_action),
        ):
            result = name
        tooltip = control.tooltip
        if not tooltip:
            tooltip = (
                f"{labels.pause} ({bindings.label(InputAction.TOGGLE_PAUSE)})"
                if name == "toggle" and playing
                else f"{labels.play} ({bindings.label(InputAction.TOGGLE_PAUSE)})"
                if name == "toggle"
                else f"{labels.previous} ({bindings.label(InputAction.STEP_BACK)})"
                if name == "previous"
                else labels.step
                if name == "step" and action_enabled
                else labels.pause_to_step
                if name == "step"
                else labels.stop_recording
                if name == "record" and recording
                else labels.record
                if name == "record"
                else labels.recording_options
                if name == "recording-options"
                else labels.reset
                if name in ("reset", "stop")
                else name
            )
        _set_viewport_tooltip(
            record_tooltip if name == "record" and record_tooltip else tooltip, scale
        )
    return result


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
    enabled_controls: Collection[str] | None = None,
    disabled_reasons: Mapping[str, str] | None = None,
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
        draw_overlay_divider(
            draw,
            (x + SHELL_RADIUS * scale, divider_y),
            theme,
            scale,
            playback=False,
        )
    result = ""
    for index, control in enumerate(controls):
        kind = control.name
        center = (x + SHELL_RADIUS * scale, y + centers[index] * scale)
        selected = (
            (kind == "move" and mode == "translate")
            or (kind == "rotate" and mode == "rotate")
            or (kind == "dimensions" and mode == "dimensions")
            or (kind == "snap" and snap)
        )
        control_enabled = enabled and (enabled_controls is None or kind in enabled_controls)
        if _circle_button(
            draw,
            f"##viewport-tool-{kind}",
            center,
            theme,
            scale,
            control.icon or _tool_icon,
            selected=selected,
            enabled=control_enabled,
            payload=(kind, space),
        ):
            result = kind
        _set_viewport_tooltip(
            (disabled_reasons or {}).get(kind, disabled_reason)
            if not control_enabled and ((disabled_reasons or {}).get(kind) or disabled_reason)
            else control.tooltip
            if control.tooltip
            else f"{labels.move} ({bindings.label(InputAction.GIZMO_TRANSLATE)})"
            if kind == "move"
            else f"{labels.rotate} ({bindings.label(InputAction.GIZMO_ROTATE)})"
            if kind == "rotate"
            else f"{labels.dimensions} ({bindings.label(InputAction.GIZMO_DIMENSIONS)})"
            if kind == "dimensions"
            else f"{labels.world_body} ({bindings.label(InputAction.GIZMO_SPACE)})"
            if kind == "frame"
            else f"{labels.snap} ({bindings.label(InputAction.SNAP)})"
            if kind == "snap"
            else kind,
            scale,
        )
    return result
