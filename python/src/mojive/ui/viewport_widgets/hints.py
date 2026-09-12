"""Viewport widgets: hints."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from mojive.drawing.curves import (
    CORNER_SMOOTHING,
    clip_polygon_rect,
    offset_closed_path,
    smooth_rect_points,
)
from mojive.ui.draw2d import Draw2D, text_line_y
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction, InputBindings
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.theme import RGBA, Theme

from .capsules import (
    draw_capsule,
)
from .model import (
    DEFAULT_VIEWPORT_LABELS,
    OVERLAY_GEOMETRY,
    MouseButtonGeometry,
    MouseWheelGeometry,
    OverlayGeometry,
    ToolHint,
    ViewportLabels,
)
from .rotate import (
    _counterclockwise,
)


def _inline_text(draw: Draw2D, x: float, center_y: float, value: str, color) -> float:
    width = draw.text_size(value)[0]
    draw.text((x, text_line_y(draw, center_y)), color, value)
    return width


def _key_width(draw: Draw2D, label: str, scale: float, text_scale: float = 1.0) -> float:
    return draw.text_size(label)[0] * text_scale + OVERLAY_GEOMETRY.hint_key_padding_x * 2.0 * scale


def keycap_rounding(width: float, height: float) -> float:
    """Use the same keycap proportions in status bars and geometry previews."""
    return min(width * 0.5, height * 0.35)


def _keycap(
    draw: Draw2D,
    x: float,
    center_y: float,
    label: str,
    theme: Theme,
    scale: float,
    *,
    muted: bool = False,
) -> float:
    text_width = draw.text_size(label)[0]
    width = text_width + OVERLAY_GEOMETRY.hint_key_padding_x * 2.0 * scale
    height = OVERLAY_GEOMETRY.hint_control_height * scale
    y = center_y - height * 0.5
    rounding = keycap_rounding(width, height)
    draw.rect_filled((x, y), (x + width, y + height), theme.bg_frame, rounding=rounding)
    draw.rect((x, y), (x + width, y + height), theme.border, 1.0 * scale, rounding=rounding)
    draw.text(
        (x + (width - text_width) * 0.5, text_line_y(draw, center_y)),
        theme.text_disabled if muted else theme.text,
        label,
    )
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
def mouse_button_geometry(
    x: float,
    y: float,
    width: float,
    height: float,
    button: str,
    *,
    outline_width: float,
    geometry: OverlayGeometry = OVERLAY_GEOMETRY,
    smoothing: float = CORNER_SMOOTHING,
) -> MouseButtonGeometry | None:
    """Return true-knockout shell and fill geometry for one mouse button."""

    if button not in {"left", "right"}:
        return None
    half_stroke = outline_width * 0.5
    shell_gap = outline_width * geometry.hint_mouse_button_shell_ratio
    button_bottom = y + height * geometry.hint_mouse_button_height_ratio
    shell_radius = min(width * 0.22, height * 0.18)
    outer_left = x - half_stroke
    button_width = (width + outline_width) * geometry.hint_mouse_button_width_ratio
    inner_edge = outer_left + button_width
    shell = smooth_rect_points(x, y, x + width, y + height, shell_radius, smoothing=smoothing)
    outer = offset_closed_path(shell, half_stroke)
    fill = clip_polygon_rect(outer, (outer_left, y - half_stroke, inner_edge, button_bottom))
    other_corners = smooth_rect_points(
        x, y, x + width, y + height, shell_radius, (False, True, True, True), smoothing=smoothing
    )
    visible_shell = (
        (inner_edge + shell_gap, y),
        *other_corners[1:],
        (x, button_bottom + shell_gap),
    )
    if button == "right":
        mirror_x = x * 2.0 + width
        visible_shell = tuple((mirror_x - point[0], point[1]) for point in visible_shell)
        fill = tuple((mirror_x - point[0], point[1]) for point in fill)
    fill = _counterclockwise(fill)
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


def mouse_hint_colors(theme: Theme, *, muted: bool = False) -> tuple[RGBA, RGBA, RGBA]:
    """Resolve the shared shell, control, and suffix colors for mouse hints."""

    return (
        theme.text_disabled if muted else theme.text,
        theme.bg_frame_active if muted else theme.primary,
        theme.text_disabled if muted else theme.primary_bright,
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
    smoothing: float = CORNER_SMOOTHING,
    muted: bool = False,
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
        smoothing=smoothing,
    )
    shell_color, fill_color, suffix_color = mouse_hint_colors(theme, muted=muted)
    if button_geometry is None:
        draw.rect(
            (x, y),
            (x + width, y + height),
            shell_color,
            outline_width,
            rounding=radius,
            smoothing=smoothing,
        )
    else:
        # Omit the shell beneath the button's transparent outer contour instead
        # of repainting it with a guessed background color. This remains a
        # genuine knockout over translucent chrome and arbitrary viewports.
        draw.polyline(button_geometry.visible_shell, shell_color, outline_width)
        draw.convex_fill(button_geometry.fill, fill_color)
    if button in ("wheel", "middle"):
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
            fill_color,
            rounding=wheel.rounding,
            smoothing=smoothing,
        )
    if not suffix:
        return width
    label_x = x + width + 5.0 * scale
    return width + 5.0 * scale + _inline_text(draw, label_x, center_y, suffix, suffix_color)


def pointer_tool_hint(
    action: PointerAction, bindings: InputBindings, label: str, *, hint_id: str
) -> ToolHint | None:
    """Draw hints from the same resolved chord used by interaction routing."""
    chords = bindings.pointer_chords(action)
    chord = next(
        (resolved for value in chords if (resolved := bindings.resolved_chord(value)) is not None),
        None,
    )
    if chord is None:
        return None
    modifier = " + ".join(key.capitalize() for key in chord.modifiers)
    if len(chord.buttons) <= 1:
        control = "wheel" if chord.wheel else ("left", "right", "middle")[chord.buttons[0]]
        return ToolHint(
            "mouse",
            control,
            label,
            "×2" if chord.clicks == 2 else "",
            modifier=modifier,
            hint_id=hint_id,
        )
    buttons = " + ".join(("LMB", "RMB", "MMB")[button] for button in chord.buttons)
    control = f"{modifier} + {buttons}" if modifier else buttons
    return ToolHint("key", control, label, "×2" if chord.clicks == 2 else "", hint_id=hint_id)


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
    perturb = tuple(
        hint
        for hint in (
            pointer_tool_hint(
                PointerAction.PERTURB_TRANSLATE, bindings, labels.push, hint_id="perturb.translate"
            ),
            pointer_tool_hint(
                PointerAction.PERTURB_ROTATE, bindings, labels.twist, hint_id="perturb.rotate"
            ),
        )
        if hint is not None
    )
    numeric = pointer_tool_hint(
        PointerAction.GIZMO_VALUE, bindings, labels.type_value, hint_id="gizmo.type_value"
    )
    if variant == "camera":
        return tuple(
            hint
            for hint in (
                pointer_tool_hint(
                    PointerAction.ORBIT, bindings, labels.orbit, hint_id="camera.orbit"
                ),
                pointer_tool_hint(PointerAction.PAN, bindings, labels.pan, hint_id="camera.pan"),
                pointer_tool_hint(
                    PointerAction.DOLLY, bindings, labels.zoom, hint_id="camera.zoom"
                ),
                ToolHint(
                    "key",
                    bindings.label(InputAction.FRAME_SCENE),
                    labels.frame,
                    hint_id="camera.frame",
                ),
            )
            if hint is not None
        )
    if variant == "dragging":
        return (snap,)
    if variant == "perturb":
        return perturb
    if variant == "ready_minimal":
        return (numeric,) if numeric is not None else ()
    return tuple(
        hint
        for hint in (
            snap,
            ToolHint(
                "key",
                bindings.label(InputAction.GIZMO_SPACE),
                labels.world_body,
                hint_id="gizmo.space",
            ),
            numeric,
            *perturb,
        )
        if hint is not None
    )


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
        modifier_width = (
            _key_width(draw, group.modifier, scale, text_scale)
            + input_gap
            + text_width("+")
            + chord
            if group.modifier
            else 0.0
        )
        return (
            modifier_width
            + _mouse_width(draw, scale, group.suffix, text_scale)
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
    muted: bool = False,
) -> float:
    """Draw tool hints inline and return their consumed width."""

    x, center_y = float(origin[0]), float(origin[1])
    cursor = x
    input_gap = OVERLAY_GEOMETRY.hint_input_gap * scale
    group_gap = OVERLAY_GEOMETRY.hint_group_gap * scale
    chord_gap = OVERLAY_GEOMETRY.hint_chord_gap * scale
    text_color = theme.text_disabled if muted else theme.text
    accent_color = theme.text_disabled if muted else theme.primary_bright

    def text(value: str, color=None) -> None:
        nonlocal cursor
        cursor += _inline_text(draw, cursor, center_y, value, color or text_color)

    def key(label: str, meaning: str) -> None:
        nonlocal cursor
        cursor += _keycap(draw, cursor, center_y, label, theme, scale, muted=muted) + input_gap
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
                muted=muted,
            )
            + input_gap
        )
        text(meaning)
        cursor += after

    def perturb(modifier: str) -> None:
        nonlocal cursor
        cursor += _keycap(draw, cursor, center_y, modifier, theme, scale, muted=muted) + input_gap
        text("+", theme.text_disabled)
        cursor += chord_gap
        text(labels.drag, accent_color)
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
            if group.modifier:
                cursor += (
                    _keycap(draw, cursor, center_y, group.modifier, theme, scale, muted=muted)
                    + input_gap
                )
                text("+", theme.text_disabled)
                cursor += chord_gap
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
