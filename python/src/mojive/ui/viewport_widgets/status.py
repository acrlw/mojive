"""Viewport widgets: status."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mojive.ui.draw2d import Draw2D, fit_text
from mojive.ui.theme import Theme

from .hints import (
    _inline_text,
    _key_width,
    _keycap,
    draw_tool_hints,
    fitting_tool_hints,
)
from .model import (
    DEFAULT_VIEWPORT_LABELS,
    OVERLAY_GEOMETRY,
    StatusLayout,
    ToolHint,
    ViewportLabels,
    _StatusPerformanceLayout,
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
    physics_hz: float | None = None,
    show_physics: bool = False,
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
) -> _StatusPerformanceLayout:
    """Pack telemetry by its displayed width and collapse it before it can overlap.

    The visual order is backend, simulation metric, delta time, and rates.
    Preserve the physics/render pair when it fits. Otherwise prioritize the
    simulation metric and delta time, then use remaining space for render FPS.
    """

    backend_text = str(backend)
    delta_text = f"Δt {max(0.0, float(dt)):.6g} s"
    fps_text = f"{labels.render} {max(0.0, float(fps)):.1f} FPS"
    if show_physics:
        rate = "—" if physics_hz is None else f"{max(0.0, physics_hz):.0f}"
        fps_text = f"{labels.physics} {rate} Hz · {fps_text}"
    metric_text = str(metric_text)

    widths = {
        "backend": draw.text_size(backend_text)[0],
        "metric": (_key_width(draw, metric_text, scale) if metric_text else 0.0),
        "delta": draw.text_size(delta_text)[0],
        "fps": draw.text_size(fps_text)[0],
    }
    gap = 22.0 * scale

    def required(names: set[str]) -> float:
        count = sum(1 for name in ("backend", "metric", "delta", "fps") if name in names)
        return sum(widths[name] for name in names) + max(0, count - 1) * gap

    limit = float("inf") if max_width is None else max(0.0, float(max_width))
    visible: set[str] = set()
    if show_physics and widths["fps"] <= limit:
        visible.add("fps")
        for name in ("metric", "delta", "backend"):
            candidate = {*visible, name}
            if widths[name] > 0.0 and required(candidate) <= limit:
                visible = candidate
    elif widths["delta"] <= limit:
        visible.add("delta")
    else:
        compact = f"Δt {max(0.0, float(dt)):.4g}s"
        compact_width = draw.text_size(compact)[0]
        if compact_width <= limit:
            delta_text = compact
            widths["delta"] = compact_width
            visible.add("delta")

    # Preserve the simulation metric beside delta time before spending scarce
    # width on FPS or the backend label.
    if "delta" in visible:
        for name in ("metric", "fps", "backend"):
            if widths[name] <= 0.0:
                continue
            candidate = {*visible, name}
            if required(candidate) <= limit:
                visible = candidate

    order = [name for name in ("backend", "metric", "delta", "fps") if name in visible]
    positions = {name: float(right) for name in ("backend", "metric", "delta", "fps")}
    dividers: list[float] = []
    cursor = float(right)
    for reverse_index, name in enumerate(reversed(order)):
        positions[name] = cursor - widths[name]
        cursor -= widths[name]
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
        widths["metric"] if "metric" in visible else 0.0,
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
    state: str,
    sim_time: float,
    step: int,
    metric_mode: str,
    backend: str,
    dt: float,
    fps: float,
    physics_hz: float | None = None,
    show_physics: bool = False,
    status: str = "",
    status_level: str = "info",
    status_path: bool = False,
    recording_phase: str = "idle",
    recording_duration: float = 0.0,
    countdown_remaining: float = 0.0,
    recording_surface: str = "scene",
    tool_hints: Sequence[ToolHint] = (),
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    pixel_size: float = 1.0,
) -> StatusLayout:
    x, y = origin
    running = state == "running"
    recording = recording_phase in {"countdown", "recording", "paused"}
    draw.rect_filled((x, y), (x + width, y + height), (*theme.bg_child[:3], 1.0))
    top_divider_width = 1.0 * scale
    top_divider_y = y + top_divider_width * 0.5
    draw.line(
        (x, top_divider_y),
        (x + width, top_divider_y),
        theme.warning if recording else theme.primary_dim if running else theme.border,
        top_divider_width,
    )
    cy = y + (height + top_divider_width) * 0.5
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

    recording_pause_rect = None
    recording_stop_rect = None
    if recording and width >= 360.0 * scale:
        separator()
        accent = theme.warning if recording_phase in {"countdown", "paused"} else theme.danger
        draw.circle_filled((cursor + 3.5 * scale, cy), 3.5 * scale, accent, segments=20)
        cursor += 11.0 * scale
        seconds = max(0, int(recording_duration))
        surface_label = {"scene": "SCENE", "viewport": "VIEW", "window": "WINDOW"}.get(
            recording_surface, "REC"
        )
        record_text = (
            f"{surface_label} {max(0, math.ceil(countdown_remaining))} s"
            if recording_phase == "countdown"
            else f"{surface_label} {seconds // 60:02d}:{seconds % 60:02d}"
        )
        cursor += _inline_text(draw, cursor, cy, record_text, accent)
        cursor += 7.0 * scale
        button_size = min(height - 5.0 * scale, 19.0 * scale)
        button_y = cy - button_size * 0.5
        if recording_phase != "countdown":
            recording_pause_rect = (cursor, button_y, cursor + button_size, button_y + button_size)
            draw.rect_filled(
                recording_pause_rect[:2],
                recording_pause_rect[2:],
                theme.bg_frame,
                rounding=3.0 * scale,
            )
            icon = accent
            center_x = cursor + button_size * 0.5
            if recording_phase == "paused":
                draw.line(
                    (center_x - 2.0 * scale, cy - 4.0 * scale),
                    (center_x + 3.5 * scale, cy),
                    icon,
                    1.6 * scale,
                )
                draw.line(
                    (center_x + 3.5 * scale, cy),
                    (center_x - 2.0 * scale, cy + 4.0 * scale),
                    icon,
                    1.6 * scale,
                )
            else:
                for offset in (-2.0, 2.0):
                    draw.line(
                        (center_x + offset * scale, cy - 4.0 * scale),
                        (center_x + offset * scale, cy + 4.0 * scale),
                        icon,
                        1.8 * scale,
                    )
            cursor += button_size + 4.0 * scale
        recording_stop_rect = (cursor, button_y, cursor + button_size, button_y + button_size)
        draw.rect_filled(
            recording_stop_rect[:2],
            recording_stop_rect[2:],
            theme.bg_frame,
            rounding=3.0 * scale,
        )
        inset = 5.5 * scale
        draw.rect_filled(
            (cursor + inset, button_y + inset),
            (cursor + button_size - inset, button_y + button_size - inset),
            accent,
            rounding=1.0 * scale,
        )
        cursor += button_size

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
        physics_hz=physics_hz,
        show_physics=show_physics,
        labels=labels,
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
        selected_shown = fit_text(draw, selected, selection_available)
        if selected_shown:
            separator()
            cursor += _inline_text(
                draw,
                cursor,
                cy,
                selected_shown,
                theme.text_disabled,
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
        shown = fit_text(draw, compact_status, min(available, width * 0.28), middle=status_path)
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
            muted=True,
        )
        left_neighbor_end = cursor + hint_width

    if shown:
        status_colors = {
            "error": theme.danger,
            "warning": theme.warning,
            "success": theme.text_disabled,
        }
        _inline_text(
            draw,
            status_x,
            cy,
            shown,
            status_colors.get(status_level, theme.text_disabled),
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
        _inline_text(draw, performance.backend_x, cy, performance.backend_text, theme.text_disabled)
    if performance.metric_text:
        metric_height = OVERLAY_GEOMETRY.hint_control_height * scale
        metric_rect = (
            performance.metric_x,
            cy - metric_height * 0.5,
            performance.metric_x + performance.metric_width,
            cy + metric_height * 0.5,
        )
        _keycap(draw, performance.metric_x, cy, metric_text, theme, scale, muted=True)
    if performance.delta_text:
        _inline_text(draw, performance.delta_x, cy, performance.delta_text, theme.text_disabled)
    if performance.fps_text:
        _inline_text(draw, performance.fps_x, cy, performance.fps_text, theme.text_disabled)
    for divider_x in performance.dividers:
        draw.line(
            (divider_x, cy - 6.0 * scale),
            (divider_x, cy + 6.0 * scale),
            theme.border,
            1.0 * scale,
        )

    return StatusLayout(
        metric_rect=metric_rect,
        metric_exact=metric_exact if metric_rect is not None else "",
        recording_pause_rect=recording_pause_rect,
        recording_stop_rect=recording_stop_rect,
        message_rect=(status_x, y, status_x + shown_width, y + height) if shown else None,
    )
