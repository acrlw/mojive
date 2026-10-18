"""Measured status badges share the native component typography and boxes."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from math import ceil

from imgui_bundle import imgui

from mojive.design.native.geometry import Box, Rect, part_scope
from mojive.ui.viewport_widgets.model import OVERLAY_GEOMETRY
from mojive.ui.viewport_widgets.status_metrics import (
    METRIC_GROUP_GAP,
    METRIC_LABEL_GAP,
    METRIC_TEXT_INSET,
    METRIC_UNIT_GAP,
    StatusReadoutStyle,
    _StatusNumbers,
)
from mojive.ui.workspace.widgets import DIM, TEXT


@dataclass
class _StatusFontMetrics:
    advances: dict[str, float]
    digit: float
    samples: dict[str, float]


@dataclass(frozen=True, slots=True)
class StatusBarLayout:
    height: float
    frame_height: float
    padding: tuple[float, float, float, float]
    margin: tuple[float, float, float, float]
    gap: float
    minimum_content_height: float

    @property
    def content_height(self):
        return self.frame_height - self.padding[1] - self.padding[3]

    def box(self, x, y, width):
        outer = Rect(x, y + self.gap, width, self.height - self.gap)
        frame = Rect(
            x + self.margin[0],
            outer.y + self.margin[1],
            max(0, width - self.margin[0] - self.margin[2]),
            self.frame_height,
        )
        return Box(outer, frame, frame.inset(*self.padding), self.padding, self.margin)


@part_scope("statusbar")
def resolve_status_bar_layout(ui, *, badge=None, groups=None, fallback_height=28):
    """Reserve the same measured row in the application and its Guide preview."""
    style, scale = ui.component_style("statusbar"), ui.s
    badge = badge if badge is not None else StatusBadge(ui, keycap=True)
    groups = groups if groups is not None else StatusGroups(ui)
    requested = (
        style.get("height", fallback_height)
        if style.explicit("height")
        else ui.metric("application", "statusHeight", fallback_height)
    )
    padding = style.edges("padding", scale, (10, 2, 10, 2))
    margin = style.edges("margin", scale)
    font = style.get("fontSize", 12)
    minimum = groups.minimum_height(
        max(
            24 * scale,
            font * scale,
            style.line_height(font * 1.4) * scale,
            badge.size("Esc")[1],
            style.get("iconSize", 7) * scale,
        )
    )
    frame_height = max(requested * scale, minimum + padding[1] + padding[3])
    gap = ui.metric("application", "rowGap", 0) * scale
    # The native sidebar consumes integer window coordinates. Quantize its
    # complete reservation outward once, then give the remainder to the frame.
    height = ceil(frame_height + margin[1] + margin[3] + gap)
    return StatusBarLayout(
        height,
        height - margin[1] - margin[3] - gap,
        padding,
        margin,
        gap,
        minimum,
    )


def metric_tooltip_options(ui):
    style, scale = ui.component_style("status_metrics"), ui.s
    return (
        (style.get("tooltipPaddingX", 10) * scale, style.get("tooltipPaddingY", 8) * scale),
        style.get("tooltipRowGap", 6) * scale,
    )


class StatusBadge:
    def __init__(self, ui, *, keycap=False):
        self.ui = ui
        self.keycap = keycap
        self._instances = {}

    def size(self, label):
        with part_scope("status_hints") if self.keycap else nullcontext():
            return self._size(label)

    def _size(self, label):
        style, scale = self.ui.component_style("badge"), self.ui.s
        font = style.get("fontSize", 14)
        padding = style.edges("padding", scale, (6, 2, 6, 2))
        margins = style.edges("margin", scale)
        width = (
            self.ui.measure(
                label, font, translate=False, font_resolved=True, mono=self.keycap, style=style
            )
            + padding[0]
            + padding[2]
        )
        height = max(font, style.line_height(font * 1.4)) * scale + padding[1] + padding[3]
        return width + margins[0] + margins[2], height + margins[1] + margins[3]

    def draw(self, x, center_y, label, theme, *, muted=False, width=None, height=None):
        with part_scope("status_hints") if self.keycap else nullcontext():
            return self._draw(x, center_y, label, theme, muted=muted, width=width, height=height)

    def _draw(self, x, center_y, label, theme, *, muted=False, width=None, height=None):
        ui, scale = self.ui, self.ui.s
        style = ui.component_style("badge")
        measured_width, measured_height = self._size(label)
        width = measured_width if width is None else width
        height = measured_height if height is None else height
        ordinal = self._instances.get(label, 0)
        self._instances[label] = ordinal + 1
        box = ui.component_box(
            "badge",
            x,
            center_y - height / 2,
            width,
            height,
            key=f"status-badge-{label}-{ordinal}",
            padding=(6 * scale, 2 * scale, 6 * scale, 2 * scale),
        )
        ui.rect(
            *box.frame.tuple(),
            tuple(round(v * 255) for v in theme.bg_frame),
            style.get("radius", 3),
        )
        ui.component_text(
            "badge",
            box.content,
            label,
            DIM if muted else TEXT,
            translate=False,
            mono=self.keycap,
            align="center" if self.keycap else None,
        )
        return width


class StatusGroups:
    """Present status groups and dividers with independent, observable geometry."""

    def __init__(self, ui):
        self.ui = ui
        self.fields, self.gaps = [], []
        self._divider_count = 0
        # Statusbar icon dimensions describe activity dots. Pointer diagrams
        # retain their own canvas instead of inheriting that unrelated diameter.
        with part_scope("status_hints", inherit=False):
            self._hint_icon_style = ui.component_style("status_hints")

    def edges(self, part):
        style = self.ui.component_style(part)
        padding, margin = style.edges("padding", self.ui.s), style.edges("margin", self.ui.s)
        return tuple(p + m for p, m in zip(padding, margin, strict=True))

    def gap(self, part, fallback=None):
        if fallback is None:
            fallback = METRIC_GROUP_GAP if part == "status_metrics" else 24
        style = self.ui.component_style(part)
        return style.get("columnGap", style.get("gap", fallback)) * self.ui.s

    def metric_options(self):
        style = self.ui.component_style("status_metrics")
        time_style = self.ui.component_style("status_time")
        separator = self.ui.component_style("status_separator")
        return {
            "separator_width": separator.get("thickness", 1) * self.ui.s
            if separator.get("visibility", "visible") == "visible"
            else 0,
            "label_gap": style.get("labelGap", METRIC_LABEL_GAP) * self.ui.s,
            "unit_gap": style.get("unitGap", METRIC_UNIT_GAP) * self.ui.s,
            "delta_value_width": style.get("deltaValueWidth", 0) * self.ui.s,
            "physics_value_width": style.get("physicsValueWidth", 0) * self.ui.s,
            "steps_value_width": style.get("stepsValueWidth", 0) * self.ui.s,
            "time_value_width": style.get("timeValueWidth", 0) * self.ui.s,
            "time_format": time_style.get("displayFormat", "seconds"),
            "render_value_width": style.get("renderValueWidth", 0) * self.ui.s,
        }

    def metric_styles(self, draw, mode):
        """Keep the common path shared; prepare local columns only when customized."""
        runtime = self.ui.design.runtime if self.ui.design is not None else None
        names = ("backend", "steps" if mode == "steps" else "time", "delta", "physics", "render")
        if runtime is None or not any(
            runtime.spec.get("parts", {}).get("status_" + name) for name in names
        ):
            return {}
        result = {}
        parent = self.ui.component_style("status_metrics")
        for name in names:
            part = "status_" + name
            parts = ("status_metrics", part)
            with part_scope("status_metrics"):
                style = self.ui.component_style(part)
                text_draw = self.ui.text_draw(draw, part)
            width_field = {
                "steps": "stepsValueWidth",
                "time": "timeValueWidth",
                "delta": "deltaValueWidth",
                "physics": "physicsValueWidth",
                "render": "renderValueWidth",
            }.get(name)
            width = parent.get(width_field, 0) if width_field else 0
            result["metric" if name in ("steps", "time") else name] = StatusReadoutStyle(
                text_draw,
                partial(self.text_budget, part=parts),
                style.get("valueWidth", width) * self.ui.s,
                style.get("labelGap", parent.get("labelGap", METRIC_LABEL_GAP)) * self.ui.s,
                style.get("unitGap", parent.get("unitGap", METRIC_UNIT_GAP)) * self.ui.s,
                style.get("paddingLeft", METRIC_TEXT_INSET) * self.ui.s,
                style.get("paddingRight", METRIC_TEXT_INSET) * self.ui.s,
                style.get(
                    "rightGap", 0 if name == "render" else self.gap("status_metrics") / self.ui.s
                )
                * self.ui.s,
                style.get(
                    "leftGap", 0 if name == "backend" else self.gap("status_metrics") / self.ui.s
                )
                * self.ui.s,
            )
        return result

    def record_readout(self, field, rect, mode):
        name = mode if field.key == "metric" else field.key
        part = "status_" + name
        if self.ui.design is None or not self.ui.design.runtime.observes((part,)):
            return
        outer = Rect(*rect)
        with part_scope("status_metrics"):
            style = self.ui.component_style(part)
        padding = (
            style.get("paddingLeft", METRIC_TEXT_INSET) * self.ui.s,
            0,
            style.get("paddingRight", METRIC_TEXT_INSET) * self.ui.s,
            0,
        )
        box = Box(outer, outer, outer.inset(*padding), padding, (0, 0, 0, 0))
        unit_right = (
            field.value_x + field.value_text_width
            if field.key == "backend"
            else field.unit_x + field.unit_width
        )
        numeric_right = (
            field.value_x + field.value_text_width
            if field.compact
            else field.value_x + field.value_width
        )
        self.ui.record_component_box(
            part,
            box,
            key="application-" + part,
            valueRect=[field.value_x, outer.y, field.value_width, outer.height],
            valueTextRect=[
                field.value_x if field.compact else numeric_right - field.value_text_width,
                outer.y,
                field.value_text_width,
                outer.height,
            ],
            valueBudgetRect=[field.value_x, outer.y, field.value_width, outer.height],
            formatBudgetWidth=field.format_width,
            labelRect=[field.content_x, outer.y, field.label_width, outer.height],
            labelGap=field.label_gap,
            unitGap=field.unit_gap,
            compact=field.compact,
            trailingReserve=max(0, box.content.right - unit_right),
            unitX=unit_right - field.unit_text_width,
            unitRight=unit_right,
            unitRect=[
                unit_right - field.unit_text_width,
                outer.y,
                field.unit_text_width,
                outer.height,
            ],
            rightGap=field.right_gap,
            leftGap=field.left_gap,
            fieldRegions={
                "valueWidth": [field.value_x, outer.y, field.value_width, outer.height],
                "leftGap": [outer.x - field.left_gap, outer.y, field.left_gap, outer.height],
                "rightGap": [outer.right, outer.y, field.right_gap, outer.height],
            },
        )

    def text_draw(self, draw, part):
        return self.ui.text_draw(draw, part)

    def theme_for(self, part, theme):
        if self.ui.design is None or not self.ui.design.runtime.has_module_palette:
            return theme
        style = self._hint_icon_style if part == "status_hints" else self.ui.component_style(part)
        return self.ui.design.runtime.theme_for(theme, style.family)

    def icon_size(self, part, fallback):
        style = self._hint_icon_style if part == "status_hints" else self.ui.component_style(part)
        return style.get("iconSize", fallback / self.ui.s) * self.ui.s

    def icon_color(self, part, fallback=None):
        style = self._hint_icon_style if part == "status_hints" else self.ui.component_style(part)
        if style.get("iconColor", "inherit") == "inherit":
            return fallback
        tint = self.ui.style_color(
            style, "iconColor", tuple(round(v * 255) for v in (fallback or (1, 1, 1, 1)))
        )
        return tuple(v / 255 for v in tint)

    def _font_metrics(self, font, size):
        key = (font.font_id, size, imgui.get_io().display_framebuffer_scale.x)
        caches = getattr(self.ui, "_status_measurements", None)
        if caches is None:
            caches = self.ui._status_measurements = {}
        cache = caches.get(key)
        if cache is None:
            imgui.push_font(font, size)
            try:
                advances = {char: imgui.calc_text_size(char).x for char in "0123456789.…"}
            finally:
                imgui.pop_font()
            cache = _StatusFontMetrics(advances, max(advances[char] for char in "0123456789"), {})
            if len(caches) >= 16:
                caches.clear()
            caches[key] = cache
        return cache

    def text_budget(self, sample, *, part="status_metrics"):
        """Share font advances between stable number slots and actual digit painting."""
        style = self.ui.component_style(part)
        font, size = self.ui.resolve_font(style), style.get("fontSize", 14) * self.ui.s
        cache = self._font_metrics(font, size)
        if sample not in cache.samples:
            missing = {char for char in sample if char not in cache.advances}
            if missing:
                imgui.push_font(font, size)
                try:
                    cache.advances.update((char, imgui.calc_text_size(char).x) for char in missing)
                finally:
                    imgui.pop_font()
            if len(cache.samples) >= 64:
                cache.samples.clear()
            cache.samples[sample] = sum(
                cache.digit if char.isdigit() else cache.advances[char] for char in sample
            )
        return cache.samples[sample]

    def number_painter(self, draw):
        cache = self._font_metrics(draw.font, draw.size)
        return _StatusNumbers(draw, cache.advances)

    def minimum_height(self, height):
        def type_height(part):
            style = self.ui.component_style(part)
            font = style.get("fontSize", 14)
            icon = self.icon_size(
                part,
                (OVERLAY_GEOMETRY.hint_control_height if part == "status_hints" else 7) * self.ui.s,
            )
            return max(height, icon, max(font, style.line_height(font * 1.4)) * self.ui.s)

        context = self.edges("status_context")
        hints = self.edges("status_hints")
        metrics = self.edges("status_metrics")
        metric_height = type_height("status_metrics")
        if self.ui.design is not None:
            for name in ("backend", "steps", "time", "delta", "physics", "render"):
                part = "status_" + name
                if self.ui.design.runtime.spec.get("parts", {}).get(part):
                    metric_height = max(metric_height, type_height(("status_metrics", part)))
        return max(
            max(type_height("status_context"), type_height("status_hints"))
            + context[1]
            + context[3]
            + hints[1]
            + hints[3],
            metric_height + metrics[1] + metrics[3],
        )

    def record(self, part, rect, fields=()):
        outer = Rect(*rect)
        self.fields.append(outer.tuple())
        if part not in ("status_context", "status_hints", "status_metrics"):
            return
        style = self.ui.component_style(part)
        padding, margin = style.edges("padding", self.ui.s), style.edges("margin", self.ui.s)
        frame = outer.inset(*margin)
        box = Box(outer, frame, frame.inset(*padding), padding, margin)
        self.ui.record_component_box(part, box, key="application-" + part, fields=list(fields))

    def record_gap(self, rect):
        self.gaps.append(rect)

    def divider(self, draw, x, center_y, available_height, theme):
        ui, scale = self.ui, self.ui.s
        style = ui.component_style("status_separator")
        thickness = style.get("thickness", 1) * scale
        top = style.get("marginTop", 5) * scale
        bottom = style.get("marginBottom", 5) * scale
        height = min(
            max(0, available_height - top - bottom),
            style.get("height", 18) * scale
            if style.get("heightMode", "auto") == "fixed"
            else available_height,
        )
        occupied = height + top + bottom
        outer = Rect(x - thickness / 2, center_y - occupied / 2, thickness, occupied)
        frame = Rect(outer.x, outer.y + top, thickness, height)
        shown = style.get("visibility", "visible") == "visible"
        if shown and thickness > 0 and height > 0:
            draw.line((x, frame.y), (x, frame.bottom), theme.border, thickness)
        box = Box(outer, frame, frame, (0, 0, 0, 0), (0, top, 0, bottom))
        self.ui.record_component_box(
            "status_separator",
            box,
            key=f"status-divider-{self._divider_count}",
            visible=shown,
            fieldRegions={
                "thickness": frame.tuple(),
                "height": frame.tuple(),
                "heightMode": frame.tuple(),
                "visibility": outer.tuple(),
            },
        )
        self._divider_count += 1
