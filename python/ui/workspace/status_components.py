"""Measured status badges share the native component typography and boxes."""

from mojive.design.native.geometry import Box, Rect
from mojive.ui.workspace.widgets import DIM, TEXT


class StatusBadge:
    def __init__(self, ui):
        self.ui = ui
        self._instances = {}

    def size(self, label):
        style, scale = self.ui.component_style("badge"), self.ui.s
        font = style.get("fontSize", 14)
        padding = style.edges("padding", scale, (6, 2, 6, 2))
        margins = style.edges("margin", scale)
        width = (
            self.ui.measure(label, font, translate=False, font_resolved=True)
            + padding[0]
            + padding[2]
        )
        height = max(font, style.line_height(font * 1.4)) * scale + padding[1] + padding[3]
        return width + margins[0] + margins[2], height + margins[1] + margins[3]

    def draw(self, x, center_y, label, theme, *, muted=False):
        ui, scale = self.ui, self.ui.s
        style = ui.component_style("badge")
        width, height = self.size(label)
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
        ui.component_text("badge", box.content, label, DIM if muted else TEXT, translate=False)
        return width


class StatusGroups:
    """Present status groups and dividers with independent, observable geometry."""

    def __init__(self, ui):
        self.ui = ui
        self.fields, self.gaps = [], []
        self._divider_count = 0

    def edges(self, part):
        style = self.ui.component_style(part)
        padding, margin = style.edges("padding", self.ui.s), style.edges("margin", self.ui.s)
        return tuple(p + m for p, m in zip(padding, margin, strict=True))

    def gap(self, part, fallback=24):
        style = self.ui.component_style(part)
        return style.get("columnGap", style.get("gap", fallback)) * self.ui.s

    def minimum_height(self, height):
        context = self.edges("status_context")
        hints = self.edges("status_hints")
        metrics = self.edges("status_metrics")
        return max(
            height + context[1] + context[3] + hints[1] + hints[3],
            height + metrics[1] + metrics[3],
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
