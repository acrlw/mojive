"""Compact telemetry columns with tabular numbers in the selected UI font."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from mojive.ui.text_layout import text_line_y
from mojive.ui.viewport_widgets.hints import _inline_text
from mojive.ui.viewport_widgets.model import DEFAULT_VIEWPORT_LABELS

METRIC_TEXT_INSET = 2
METRIC_LABEL_GAP = 4
METRIC_UNIT_GAP = 2
METRIC_GROUP_GAP = 6

_TIMESTEP_UNITS = ((1e-9, "ns"), (1e-6, "μs"), (1e-3, "ms"), (1, "s"))
_PHYSICS_UNITS = tuple(
    (1000**index, prefix + "Hz") for index, prefix in enumerate(("", "k", "M", "G", "T"))
)
_RENDER_UNITS = tuple((factor, unit.replace("Hz", "FPS")) for factor, unit in _PHYSICS_UNITS)
_STEP_UNITS = tuple(
    (1000**index, suffix) for index, suffix in enumerate(("", "k", "M", "B", "T", "P", "E"))
)
_TIME_SAMPLES = {"seconds": "999.999", "clock": "99:59:59"}


def _time_readout(seconds, display_format, width, measure):
    """Fit an elapsed time without deriving its column width from the live value."""
    seconds = max(0.0, float(seconds))
    if not isfinite(seconds):
        return "∞" if measure("∞") <= width else "…", "s", None
    if display_format == "clock" and seconds < 100 * 3600:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, second = divmod(remainder, 60)
        clock = f"{hours:02d}:{minutes:02d}:{second:02d}"
        if measure(clock) <= width:
            return clock, "", None
    fixed = f"{seconds:.3f}"
    if (seconds == 0 or float(fixed) > 0) and measure(fixed) <= width:
        return fixed, "s", 3
    # Scientific seconds retain a complete magnitude when a duration exceeds
    # its static budget. Never show a clipped tail of a time or exponent.
    for digits in (3, 2, 1, 0):
        mantissa, exponent = f"{seconds:.{digits}e}".split("e")
        compact = f"{mantissa}e{int(exponent)}"
        if measure(compact) <= width:
            return compact, "s", None
    return "0" if seconds == 0 and measure("0") <= width else "…", "s", None


@dataclass(frozen=True)
class StatusReadoutStyle:
    draw: Any
    measure: Callable[[str], float]
    value_width: float
    label_gap: float
    unit_gap: float
    padding_left: float = 0
    padding_right: float = 0
    right_gap: float = 0
    left_gap: float = 0


@dataclass(frozen=True)
class _StatusReadout:
    key: str
    label: str
    value: str
    unit: str
    sample: str
    unit_samples: tuple[str, ...]
    decimals: int | None = None
    exact: str = ""
    x: float = 0
    width: float = 0
    value_x: float = 0
    value_width: float = 0
    unit_x: float = 0
    unit_width: float = 0
    unit_text_width: float = 0
    content_x: float = 0
    right_gap: float = 0
    left_gap: float = 0
    compact: bool = False
    value_text_width: float = 0
    unit_budget_width: float = 0
    label_width: float = 0
    label_gap: float = 0
    unit_gap: float = 0
    format_width: float = 0


@dataclass(frozen=True)
class _StatusPerformanceLayout:
    fields: tuple[_StatusReadout, ...]
    dividers: tuple[float, ...]
    left: float


def _rate(value, unit, decimals):
    if value is None:
        return "—", unit
    value = max(0.0, float(value))
    for prefix in ("", "k", "M", "G", "T"):
        shown = f"{value:.{decimals}f}"
        if (value < 1000 and float(shown) < 1000) or prefix == "T":
            return shown, prefix + unit
        value /= 1000
    raise AssertionError("Unreachable rate unit")


def _timestep(value):
    value = max(0.0, float(value))
    for factor, unit in ((1, "s"), (1e3, "ms"), (1e6, "μs"), (1e9, "ns")):
        scaled = value * factor
        if scaled >= 1 or unit == "ns" or value == 0:
            return f"{scaled:.3g}", unit
    raise AssertionError("Unreachable timestep unit")


def _fit_readout(
    value, units, canonical_unit, width, precision, measure, *, promoted_precision=None
):
    """Fit complete numbers, promoting units before resorting to an overflow mark."""
    canonical = next(index for index, (_, unit) in enumerate(units) if unit == canonical_unit)
    if value is None:
        return ("—" if measure("—") <= width else "…"), canonical_unit, None
    value = max(0.0, float(value))
    digit, dot = measure("0"), measure(".")
    compact_precision = (
        max(1, min(3, int((width - dot) / digit)))
        if promoted_precision is None
        else promoted_precision
    )
    for index in range(canonical, len(units)):
        factor, unit = units[index]
        scaled = value / factor
        decimals = (
            compact_precision
            if not precision and (index > canonical or 0 < scaled < 1)
            else precision
        )
        for count in (decimals, 0) if decimals else (0,):
            shown = f"{scaled:.{count}f}"
            rounded = float(shown)
            if (value > 0 and rounded == 0) or (rounded >= 1000 and index < len(units) - 1):
                continue
            if measure(shown) <= width:
                return shown, unit, count
            if shown.startswith("0.") and measure(shown[1:]) <= width:
                return shown[1:], unit, count
    # A partial number could be mistaken for the actual value. Exact values
    # remain available on hover and copy even in an intentionally tiny slot.
    return "…", canonical_unit, None


def _status_performance_layout(
    draw,
    right,
    scale,
    backend,
    dt,
    fps,
    *,
    metric_text="",
    metric_mode="time",
    max_width=None,
    physics_hz=None,
    show_physics=False,
    labels=DEFAULT_VIEWPORT_LABELS,
    group_gap=None,
    separator_width=None,
    text_budget=None,
    label_gap=None,
    unit_gap=None,
    delta_value_width=0,
    physics_value_width=0,
    steps_value_width=0,
    time_value_width=0,
    render_value_width=0,
    step=None,
    metric_styles=None,
    sim_time=None,
    time_format="seconds",
):
    """Pack measured metric phrases; authored numeric columns remain fixed."""
    if text_budget is None:
        widest_digit = max("0123456789", key=lambda char: draw.text_size(char)[0])

        def text_budget(sample):
            return sum(
                draw.text_size(widest_digit if char.isdigit() else char)[0] for char in sample
            )

    label_gap = METRIC_LABEL_GAP * scale if label_gap is None else max(0, label_gap)
    unit_gap = METRIC_UNIT_GAP * scale if unit_gap is None else max(0, unit_gap)
    styles = metric_styles or {}
    separator_width = scale if separator_width is None else max(0, separator_width)

    def measure(key, value):
        return styles[key].measure(value) if key in styles else text_budget(value)

    def requested_width(key, fallback):
        return styles[key].value_width if key in styles else fallback

    def number_width(key):
        return lambda value: measure(key, value)

    def value_budget(key, requested, sample):
        minimum = max(measure(key, "0"), measure(key, "…"))
        return max(minimum, requested_width(key, requested) or measure(key, sample))

    delta_width = value_budget("delta", delta_value_width, "99.9")
    physics_width = value_budget("physics", physics_value_width, "999")
    render_width = value_budget("render", render_value_width, "999.9")

    _, delta_unit = _timestep(dt)
    delta_precision = max(
        1, min(3, int((delta_width - measure("delta", ".")) / measure("delta", "0")) - 2)
    )
    delta, delta_unit, delta_decimals = _fit_readout(
        dt,
        _TIMESTEP_UNITS,
        delta_unit,
        delta_width,
        delta_precision,
        number_width("delta"),
    )
    render, render_unit = _rate(fps, "FPS", 1)
    render_decimals = 1
    render, render_unit, render_decimals = _fit_readout(
        fps, _RENDER_UNITS, render_unit, render_width, 1, number_width("render")
    )
    readouts = [
        _StatusReadout("backend", "", str(backend), "", "", (), exact=str(backend)),
        _StatusReadout(
            "delta",
            "Δt",
            delta,
            delta_unit,
            "99.9",
            ("s", "ms", "μs", "ns"),
            delta_decimals,
            f"{max(0.0, float(dt))} s",
        ),
        _StatusReadout(
            "render",
            labels.render,
            render,
            render_unit,
            "999.9",
            ("FPS", "kFPS", "MFPS", "GFPS", "TFPS"),
            render_decimals,
            f"{max(0.0, float(fps))} FPS",
        ),
    ]
    if metric_text:
        label = labels.steps if metric_mode == "steps" else labels.time
        value = metric_text.removeprefix(label + " ")
        if metric_mode == "steps":
            unit = value[-1] if value[-1].isalpha() else ""
            value = value[:-1] if unit else value
            metric = _StatusReadout(
                "metric", label, value, unit, "999.9", ("k", "M", "B", "T", "P", "E"), 1
            )
            if requested_width("metric", steps_value_width):
                factor = next(factor for factor, suffix in _STEP_UNITS if suffix == unit)
                count = step if step is not None else float(value) * factor
                value, unit, decimals = _fit_readout(
                    count,
                    _STEP_UNITS,
                    unit,
                    value_budget("metric", steps_value_width, "999.9"),
                    1 if unit else 0,
                    number_width("metric"),
                    promoted_precision=1,
                )
                metric = replace(metric, value=value, unit=unit, decimals=decimals)
        else:
            value, unit = (value[:-2], "s") if value.endswith(" s") else (value, "")
            sample = _TIME_SAMPLES[time_format]
            decimals = None
            if sim_time is not None:
                value, unit, decimals = _time_readout(
                    sim_time,
                    time_format,
                    value_budget("metric", time_value_width, sample),
                    number_width("metric"),
                )
            metric = _StatusReadout("metric", label, value, unit, sample, ("s",), decimals)
        readouts.insert(1, metric)
    if show_physics:
        _, physics_unit = _rate(physics_hz, "Hz", 0)
        physics, physics_unit, physics_decimals = _fit_readout(
            physics_hz,
            _PHYSICS_UNITS,
            physics_unit,
            physics_width,
            0,
            number_width("physics"),
        )
        readouts.insert(
            -1,
            _StatusReadout(
                "physics",
                labels.physics,
                physics,
                physics_unit,
                "999",
                ("Hz", "kHz", "MHz", "GHz", "THz"),
                physics_decimals,
                "— Hz" if physics_hz is None else f"{max(0.0, float(physics_hz))} Hz",
            ),
        )
    value_widths = {"delta": delta_width, "physics": physics_width, "render": render_width}
    measured = {}
    for field in readouts:
        item = styles.get(field.key)
        field_draw = item.draw if item else draw
        left_pad, right_pad = (
            (item.padding_left, item.padding_right)
            if item
            else (METRIC_TEXT_INSET * scale, METRIC_TEXT_INSET * scale)
        )
        field_label_gap, field_unit_gap = (
            (item.label_gap, item.unit_gap) if item else (label_gap, unit_gap)
        )
        default_gap = METRIC_GROUP_GAP * scale if group_gap is None else max(0, group_gap)
        before_gap = item.left_gap if item else (0 if field.key == "backend" else default_gap)
        after_gap = item.right_gap if item else (0 if field.key == "render" else default_gap)
        if field.key == "backend":
            from mojive.ui.text_layout import fit_text

            value_width = requested_width("backend", 0) or field_draw.text_size(field.value)[0]
            value = fit_text(field_draw, field.value, value_width)
            measured[field.key] = replace(
                field,
                content_x=left_pad,
                value_x=left_pad,
                value_width=value_width,
                width=left_pad + value_width + right_pad,
                right_gap=after_gap,
                left_gap=before_gap,
                compact=True,
                value_text_width=field_draw.text_size(value)[0],
                format_width=value_width,
            )
            continue
        label_width = field_draw.text_size(field.label)[0]
        prefix = left_pad + label_width + field_label_gap
        if field.key == "metric":
            requested = steps_value_width if metric_mode == "steps" else time_value_width
            value_width = value_budget("metric", requested, field.sample)
        else:
            requested = {
                "delta": delta_value_width,
                "physics": physics_value_width,
                "render": render_value_width,
            }[field.key]
            value_width = value_widths[field.key]
        if (
            field.key == "metric"
            and metric_mode != "steps"
            and measure("metric", field.value) > value_width
        ):
            field = replace(field, value="…")
        compact = not requested_width(field.key, requested)
        suffix_gap = (
            0
            if (field.key == "metric" and metric_mode == "steps") or (compact and not field.unit)
            else field_unit_gap
        )
        value_text_width = measure(field.key, field.value)
        unit_width = field_draw.text_size(field.unit)[0] if field.unit else 0
        unit_budget = max(field_draw.text_size(unit)[0] for unit in field.unit_samples)
        # A formatting cap is not empty layout space. Automatic phrases use
        # their displayed tabular advances; only digit-count or unit changes
        # resize them. Explicit columns retain their authored anchor budget.
        displayed_width = value_text_width if compact else value_width
        measured[field.key] = replace(
            field,
            value_x=prefix,
            value_width=displayed_width,
            unit_x=prefix + displayed_width + suffix_gap,
            unit_width=unit_width,
            unit_text_width=unit_width,
            width=prefix
            + displayed_width
            + suffix_gap
            + (unit_width if compact else unit_budget)
            + right_pad,
            content_x=left_pad,
            right_gap=after_gap,
            left_gap=before_gap,
            compact=compact,
            value_text_width=value_text_width,
            unit_budget_width=unit_budget,
            label_width=label_width,
            label_gap=field_label_gap,
            unit_gap=suffix_gap,
            format_width=value_width,
        )
    limit = float("inf") if max_width is None else max(0, max_width)

    def fits(names):
        ordered = [measured[field.key] for field in readouts if field.key in names]
        return (
            sum(field.left_gap + field.width + field.right_gap for field in ordered)
            + separator_width * max(0, len(ordered) - 1)
            <= limit
        )

    visible = {"physics", "render"} if show_physics and fits(("physics", "render")) else set()
    for name in ("delta", "metric", "render", "physics", "backend"):
        if name in measured and fits(visible | {name}):
            visible.add(name)
    fields = [measured[field.key] for field in readouts if field.key in visible]
    total = sum(field.left_gap + field.width + field.right_gap for field in fields) + (
        separator_width * max(0, len(fields) - 1)
    )
    cursor = right - total
    left, placed, dividers = cursor, [], []
    for index, field in enumerate(fields):
        if index:
            dividers.append(cursor + separator_width / 2)
            cursor += separator_width
        cursor += field.left_gap
        placed.append(
            replace(
                field,
                x=cursor,
                content_x=cursor + field.content_x,
                value_x=cursor + field.value_x,
                unit_x=cursor + field.unit_x,
            )
        )
        cursor += field.width + field.right_gap
    return _StatusPerformanceLayout(tuple(placed), tuple(dividers), left)


class _StatusNumbers:
    """Lay out digit cells without replacing the component's chosen font family."""

    def __init__(self, draw, widths=None):
        self.draw = draw
        self.widths = (
            widths
            if widths is not None
            else {char: draw.text_size(char)[0] for char in "0123456789."}
        )
        self.digit = max(self.widths[char] for char in "0123456789")

    def width(self, text):
        return sum(self.digit if char.isdigit() else self.advance(char) for char in text)

    def advance(self, char):
        if char not in self.widths:
            self.widths[char] = self.draw.text_size(char)[0]
        return self.widths[char]

    def paint(self, field, center_y, color):
        if field.key == "backend":
            from mojive.ui.text_layout import fit_text

            _inline_text(
                self.draw,
                field.content_x,
                center_y,
                fit_text(self.draw, field.value, field.value_width),
                color,
            )
            return
        _inline_text(self.draw, field.content_x, center_y, field.label, color)
        right = field.value_x + field.value_width
        value = field.value
        if self.width(value) > field.value_width:
            while value and self.width("…" + value) > field.value_width:
                value = value[1:]
            value = "…" + value
        integer, dot, fraction = value.partition(".")
        if field.compact:
            cursor = field.value_x
        elif (
            field.decimals is not None
            and (integer.isdigit() or not integer)
            and (not dot or fraction.isdigit())
        ):
            anchor = right - (
                self.advance(".") + field.decimals * self.digit if field.decimals else 0
            )
            cursor = anchor - self.width(integer)
        else:
            cursor = right - self.width(value)
        line_y = text_line_y(self.draw, center_y)
        for char in value:
            advance = self.digit if char.isdigit() else self.advance(char)
            self.draw.text(
                (cursor + (advance - self.advance(char)) / 2, line_y),
                color,
                char,
                pixel_snap=False,
            )
            cursor += advance
        if field.unit:
            _inline_text(
                self.draw,
                field.unit_x,
                center_y,
                field.unit,
                color,
            )
