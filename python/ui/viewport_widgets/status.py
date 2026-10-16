"""Viewport widgets: status."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from mojive.ui.paint_protocol import Draw2D
from mojive.ui.text_layout import fit_text
from mojive.ui.theme import Theme
from mojive.ui.viewport_widgets.hints import (
    _inline_text,
    _key_width,
    _keycap,
    draw_tool_hints,
    fitting_tool_hints,
    tool_hints_height,
)
from mojive.ui.viewport_widgets.model import (
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
    group_gap: float | None = None,
    keycap=None,
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
        "metric": (
            _key_width(draw, metric_text, scale, presentation=keycap) if metric_text else 0.0
        ),
        "delta": draw.text_size(delta_text)[0],
        "fps": draw.text_size(fps_text)[0],
    }
    gap = 22.0 * scale if group_gap is None else max(0, group_gap)

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


@dataclass(frozen=True)
class _RecordingStatusLayout:
    text: str = ""
    width: float = 0.0
    button_size: float = 0.0
    pause: bool = False
    stop: bool = False


def _recording_status_layout(
    draw: Draw2D,
    available: float,
    height: float,
    scale: float,
    phase: str,
    duration: float,
    countdown: float,
    surface: str,
    labels: ViewportLabels,
) -> _RecordingStatusLayout:
    """Reserve recording actions before spending the remaining width on their label."""
    if phase not in {"countdown", "recording", "paused", "finalizing"}:
        return _RecordingStatusLayout()
    button_size = max(0.0, min(height - 5.0 * scale, 19.0 * scale))
    stop = phase != "finalizing" and 0.0 < button_size <= available
    pause = stop and phase in {"recording", "paused"} and 2 * button_size + 4.0 * scale <= available
    controls_width = (button_size if stop else 0.0) + (button_size + 4.0 * scale if pause else 0.0)
    seconds = max(0, int(duration))
    surface_label = {
        "scene": labels.recording_scene,
        "viewport": labels.recording_viewport,
        "window": labels.recording_window,
    }.get(surface, labels.recording)
    text = (
        labels.recording_finalizing
        if phase == "finalizing"
        else f"{surface_label} {max(0, math.ceil(countdown))} s"
        if phase == "countdown"
        else f"{surface_label} {seconds // 60:02d}:{seconds % 60:02d}"
    )
    text = fit_text(draw, text, max(0.0, available - controls_width - 18.0 * scale))
    text_width = draw.text_size(text)[0] + 18.0 * scale if text else 0.0
    return _RecordingStatusLayout(text, text_width + controls_width, button_size, pause, stop)


def _draw_recording_status(
    draw: Draw2D,
    cursor: float,
    cy: float,
    theme: Theme,
    scale: float,
    phase: str,
    layout: _RecordingStatusLayout,
) -> tuple[float, tuple | None, tuple | None]:
    """Paint the fitted recording group and return its exact interactive rectangles."""
    accent = theme.warning if phase in {"countdown", "paused"} else theme.danger
    if layout.text:
        draw.circle_filled((cursor + 3.5 * scale, cy), 3.5 * scale, accent, segments=20)
        cursor += 11.0 * scale
        cursor += _inline_text(draw, cursor, cy, layout.text, accent) + 7.0 * scale
    button_size = layout.button_size
    button_y = cy - button_size * 0.5
    pause_rect = None
    stop_rect = None
    if layout.pause:
        pause_rect = (cursor, button_y, cursor + button_size, button_y + button_size)
        draw.rect_filled(pause_rect[:2], pause_rect[2:], theme.bg_frame, rounding=3.0 * scale)
        center_x = cursor + button_size * 0.5
        if phase == "paused":
            draw.line(
                (center_x - 2.0 * scale, cy - 4.0 * scale),
                (center_x + 3.5 * scale, cy),
                accent,
                1.6 * scale,
            )
            draw.line(
                (center_x + 3.5 * scale, cy),
                (center_x - 2.0 * scale, cy + 4.0 * scale),
                accent,
                1.6 * scale,
            )
        else:
            for offset in (-2.0, 2.0):
                draw.line(
                    (center_x + offset * scale, cy - 4.0 * scale),
                    (center_x + offset * scale, cy + 4.0 * scale),
                    accent,
                    1.8 * scale,
                )
        cursor += button_size + 4.0 * scale
    if layout.stop:
        stop_rect = (cursor, button_y, cursor + button_size, button_y + button_size)
        draw.rect_filled(stop_rect[:2], stop_rect[2:], theme.bg_frame, rounding=3.0 * scale)
        inset = 5.5 * scale
        draw.rect_filled(
            (cursor + inset, button_y + inset),
            (cursor + button_size - inset, button_y + button_size - inset),
            accent,
            rounding=1.0 * scale,
        )
        cursor += button_size
    return cursor, pause_rect, stop_rect


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
    show_metrics: bool = True,
    status: str = "",
    passive: bool = False,
    activity: str = "",
    status_level: str = "info",
    status_path: bool = False,
    recording_phase: str = "idle",
    recording_duration: float = 0.0,
    countdown_remaining: float = 0.0,
    recording_surface: str = "scene",
    tool_hints: Sequence[ToolHint] = (),
    labels: ViewportLabels = DEFAULT_VIEWPORT_LABELS,
    pixel_size: float = 1.0,
    group_gap: float | None = None,
    hints_gap: float | None = None,
    icon_size: float | None = None,
    line_height: float | None = None,
    align_items: str = "center",
    keycap=None,
    separator_width=None,
    separator_margin=None,
    record_group=None,
    presentation=None,
) -> StatusLayout:
    x, y = origin
    running = state in {"running", "replaying"}
    recording = recording_phase in {"countdown", "recording", "paused", "finalizing"}
    draw.rect_filled((x, y), (x + width, y + height), (*theme.bg_child[:3], 1.0))
    top_divider_width = 1.0 * scale
    top_divider_y = y + top_divider_width * 0.5
    draw.line(
        (x, top_divider_y),
        (x + width, top_divider_y),
        theme.warning if recording else theme.primary_dim if running else theme.border,
        top_divider_width,
    )
    hint_height = tool_hints_height(draw, scale, tool_hints, keycap=keycap)
    line = min(
        height,
        max(
            height if line_height is None else max(0, line_height),
            draw.text_size("M")[1],
            hint_height,
            icon_size or 7 * scale,
            keycap.size("Esc")[1] if keycap is not None else 0,
        ),
    )
    cy = y + (
        line / 2
        if align_items in ("start", "flex-start")
        else height - line / 2
        if align_items in ("end", "flex-end")
        else height / 2
    )
    edge = 12.0 * scale if presentation is None else 0.0
    context_edges = presentation.edges("status_context") if presentation else (0, 0, 0, 0)
    context_x = x + edge + context_edges[0]
    cursor = context_x
    hints_edges = presentation.edges("status_hints") if presentation else (0, 0, 0, 0)
    metrics_edges = presentation.edges("status_metrics") if presentation else (0, 0, 0, 0)
    context_cy = cy + (context_edges[1] - context_edges[3]) / 2
    hint_cy = context_cy + (hints_edges[1] - hints_edges[3]) / 2
    metric_cy = cy + (metrics_edges[1] - metrics_edges[3]) / 2
    group_fields = []
    context_fields = []

    def record(part, rect, fields=()):
        if part in ("state", "recording", "selection", "status_hints"):
            context_fields.append(rect)
        if presentation is not None:
            presentation.record(part, rect, fields)
        elif part == "status_hints" and record_group is not None:
            record_group(part, rect)

    def divider(at_x, center=cy):
        if presentation is not None:
            presentation.divider(draw, at_x, center, height, theme)
            return
        half = 6 * scale if separator_margin is None else max(0, height / 2 - separator_margin)
        draw.line(
            (at_x, center - half),
            (at_x, center + half),
            theme.border,
            scale if separator_width is None else separator_width,
        )

    gap = 24.0 * scale if group_gap is None else max(0, group_gap)
    context_gap = presentation.gap("status_context", gap / scale) if presentation else gap
    boundary_gap = gap if presentation else 18 * scale
    status_gap = gap if presentation else 22 * scale
    dot_size = 7.0 * scale if icon_size is None else max(0, icon_size)

    def separator() -> None:
        nonlocal cursor
        cursor += context_gap / 2
        divider(cursor, context_cy)
        if presentation is not None:
            presentation.record_gap((cursor - context_gap / 2, y, context_gap, height))
        cursor += context_gap / 2

    state_color = theme.primary if running else theme.text_disabled
    state_text = activity or (
        ""
        if state == "external"
        else labels.replaying
        if state == "replaying"
        else labels.running
        if running
        else labels.static
        if state == "static"
        else labels.paused
    )
    if passive:
        state_text = f"{labels.passive} · {state_text}" if state_text else labels.passive
    available = max(0.0, width - 2 * edge - context_edges[0] - context_edges[2])
    recording_layout = _recording_status_layout(
        draw,
        available,
        height,
        scale,
        recording_phase,
        recording_duration,
        countdown_remaining,
        recording_surface,
        labels,
    )
    recording_reserve = recording_layout.width + context_gap if recording_layout.width else 0.0
    state_text = fit_text(draw, state_text, max(0.0, available - recording_reserve - 12.0 * scale))
    passive_rect = None
    if state_text:
        draw.circle_filled(
            (cursor + dot_size / 2, context_cy),
            dot_size / 2,
            state_color,
            segments=20,
        )
        cursor += dot_size + 5.0 * scale
        if passive:
            passive_width = min(draw.text_size(labels.passive)[0], draw.text_size(state_text)[0])
            passive_rect = (cursor, y, cursor + passive_width, y + height)
        cursor += _inline_text(draw, cursor, context_cy, state_text, state_color)

    if state_text:
        record("state", (context_x, context_cy - line / 2, cursor - context_x, line))

    recording_pause_rect = None
    recording_stop_rect = None
    if recording_layout.width:
        if state_text:
            separator()
        recording_x = cursor
        cursor, recording_pause_rect, recording_stop_rect = _draw_recording_status(
            draw, cursor, context_cy, theme, scale, recording_phase, recording_layout
        )
        record("recording", (recording_x, context_cy - line / 2, cursor - recording_x, line))

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
        x
        + width
        - edge
        - cursor
        - context_edges[2]
        - boundary_gap
        - selection_reserve
        - metrics_edges[0]
        - metrics_edges[2],
    )
    performance = _status_performance_layout(
        draw,
        x + width - edge - metrics_edges[2],
        scale,
        backend,
        dt,
        fps,
        metric_text=metric_text,
        max_width=telemetry_budget if show_metrics else 0,
        physics_hz=physics_hz,
        show_physics=show_physics,
        labels=labels,
        group_gap=presentation.gap("status_metrics") if presentation else gap,
        keycap=keycap,
    )
    telemetry_fields = (
        performance.backend_text,
        performance.metric_text,
        performance.delta_text,
        performance.fps_text,
    )
    metrics_left = performance.left - metrics_edges[0]
    left_limit = metrics_left - (boundary_gap if any(telemetry_fields) else 0.0)

    selection_available = left_limit - cursor - context_edges[2] - context_gap
    if selection_available >= draw.text_size("…")[0]:
        selected_shown = fit_text(draw, selected, selection_available)
        if selected_shown:
            separator()
            selection_x = cursor
            cursor += _inline_text(
                draw,
                cursor,
                context_cy,
                selected_shown,
                theme.text_disabled,
            )
            record("selection", (selection_x, context_cy - line / 2, cursor - selection_x, line))
    left_neighbor_end = cursor + context_edges[2]

    metric_rect = None
    compact_status = " ".join(str(status).split())
    available = metrics_left - cursor - context_edges[2] - (gap if presentation else 34 * scale)
    shown = ""
    shown_width = 0.0
    status_x = metrics_left - status_gap
    if compact_status and available > 48.0 * scale:
        # A transient report remains readable without evicting every context
        # hint from a wide status bar.
        shown = fit_text(draw, compact_status, min(available, width * 0.28), middle=status_path)
        shown_width, _ = draw.text_size(shown)
        status_x = metrics_left - status_gap - shown_width

    hint_right = status_x - (14.0 * scale if shown else 0.0)
    hint_available = (
        hint_right - cursor - context_edges[2] - context_gap - hints_edges[0] - hints_edges[2]
    )
    fitted_hints = fitting_tool_hints(
        draw,
        scale,
        tool_hints,
        hint_available,
        labels=labels,
        group_gap=hints_gap,
        keycap=keycap,
    )
    if fitted_hints:
        separator()
        hints_x = cursor
        hint_fields = []
        hint_width = draw_tool_hints(
            draw,
            (cursor + hints_edges[0], hint_cy),
            theme,
            scale,
            fitted_hints,
            labels=labels,
            pixel_size=pixel_size,
            muted=True,
            group_gap=hints_gap,
            keycap=keycap,
            separator_width=separator_width,
            separator_margin=separator_margin,
            record_item=hint_fields.append if presentation else None,
            separator_draw=(lambda at_x: divider(at_x, hint_cy)) if presentation else None,
        )
        hint_width += hints_edges[0] + hints_edges[2]
        left_neighbor_end = hints_x + hint_width + context_edges[2]
        painted_height = tool_hints_height(draw, scale, fitted_hints, keycap=keycap)
        outer_height = painted_height + hints_edges[1] + hints_edges[3]
        record(
            "status_hints",
            (hints_x, context_cy - outer_height / 2, hint_width, outer_height),
            hint_fields,
        )

    if context_fields:
        top = min(rect[1] for rect in context_fields) - context_edges[1]
        bottom = max(rect[1] + rect[3] for rect in context_fields) + context_edges[3]
        record(
            "status_context",
            (x + edge, top, left_neighbor_end - x - edge, bottom - top),
            context_fields,
        )

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
    leading_gap = metrics_left - left_neighbor_end
    # A separator belongs between neighboring groups, never at the edge of a
    # right-aligned telemetry island. This also adapts when future fields are
    # added or current fields collapse on narrow windows.
    if telemetry_present and 0.0 < leading_gap <= 30.0 * scale:
        divider_x = left_neighbor_end + leading_gap * 0.5
        divider(divider_x)
    if performance.backend_text:
        _inline_text(
            draw, performance.backend_x, metric_cy, performance.backend_text, theme.text_disabled
        )
        group_fields.append(
            (
                performance.backend_x,
                metric_cy - line / 2,
                draw.text_size(performance.backend_text)[0],
                line,
            )
        )
    if performance.metric_text:
        metric_height = (
            OVERLAY_GEOMETRY.hint_control_height * scale
            if keycap is None
            else keycap.size(metric_text)[1]
        )
        metric_rect = (
            performance.metric_x,
            metric_cy - metric_height * 0.5,
            performance.metric_x + performance.metric_width,
            metric_cy + metric_height * 0.5,
        )
        _keycap(
            draw,
            performance.metric_x,
            metric_cy,
            metric_text,
            theme,
            scale,
            muted=True,
            presentation=keycap,
        )
    if metric_rect is not None:
        mx, my, mr, mb = metric_rect
        group_fields.append((mx, my, mr - mx, mb - my))
    if performance.delta_text:
        _inline_text(
            draw, performance.delta_x, metric_cy, performance.delta_text, theme.text_disabled
        )
        group_fields.append(
            (
                performance.delta_x,
                metric_cy - line / 2,
                draw.text_size(performance.delta_text)[0],
                line,
            )
        )
    if performance.fps_text:
        _inline_text(draw, performance.fps_x, metric_cy, performance.fps_text, theme.text_disabled)
        group_fields.append(
            (performance.fps_x, metric_cy - line / 2, draw.text_size(performance.fps_text)[0], line)
        )
    for divider_x in performance.dividers:
        divider(divider_x, metric_cy)
    if group_fields:
        group_height = line + metrics_edges[1] + metrics_edges[3]
        record(
            "status_metrics",
            (metrics_left, cy - group_height / 2, x + width - edge - metrics_left, group_height),
            group_fields,
        )

    return StatusLayout(
        metric_rect=metric_rect,
        metric_exact=metric_exact if metric_rect is not None else "",
        recording_pause_rect=recording_pause_rect,
        recording_stop_rect=recording_stop_rect,
        message_rect=(status_x, y, status_x + shown_width, y + height) if shown else None,
        passive_rect=passive_rect,
    )
