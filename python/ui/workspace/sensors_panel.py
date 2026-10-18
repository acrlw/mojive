"""Compact sensor selection, metadata and stable live readout grids."""

from dataclasses import replace
from math import ceil, floor

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.numeric_text import fit_numeric_text
from mojive.ui.panels import copy_state_vector
from mojive.ui.panels.sensors import SensorsPanel, sensor_component_label
from mojive.ui.workspace.component_layout import component_flow, control_frame_size, control_size
from mojive.ui.workspace.design import DESIGN, current_design, use_design
from mojive.ui.workspace.property_style import NativePropertyPresentation, advance
from mojive.ui.workspace.widgets import INPUT, MUTED, TEXT


def sensor_value_columns(width, dimension, minimum_cell_width, gap):
    """Choose a stable grid from metadata and typography, independent of live values."""
    return max(1, min(dimension, 3, floor((width + gap) / (minimum_cell_width + gap))))


class WorkspaceSensors(SensorsPanel):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.presentation = NativePropertyPresentation(workspace.ui, prefix="sensor")
        self._content_width = None

    def _width(self):
        available = imgui.get_content_region_avail().x
        return available if self._content_width is None else min(available, self._content_width)

    def _metadata(self, ctx, sensor):
        ui, scale = self.workspace.ui, self.workspace.s
        style = ui.component_style("sensor")
        pos = imgui.get_cursor_screen_pos()
        gap = style.get("gap", 8) * scale
        selector = ui.hits.get("##sensor")
        y = selector[1] + selector[3] + gap if selector else pos.y
        detail = style.get("detailSize", 12)
        height = ceil(style.line_height(detail * 1.4) * scale)
        rect = Rect(pos.x, y, self._width(), height)
        kind = sensor.type.removeprefix("mjSENS_").lower()
        ui.component_text(
            "sensor",
            rect,
            f"{kind} · {ctx.tr('dimension')} {sensor.dim}",
            MUTED,
            size=detail,
            font_field="detailSize",
            translate=False,
        )
        ui.hits["sensor-metadata"] = rect.tuple()
        advance(ui, pos.x, rect.bottom + gap + 4 * scale)

    @part_scope("button_row")
    def _actions(self, ctx, values):
        ui, scale = self.workspace.ui, self.workspace.s
        p = imgui.get_cursor_screen_pos()
        width = self._width()
        style, row_style = ui.component_style("sensor"), ui.component_style("button_row")
        gap = style.get("gap", 8) * scale
        icon = style.get("iconSize", 16)
        actions = (("copy", "Copy readings", "copy"), ("plot", "Open in Plot", "waveform"))
        sizes = [
            control_size(ui, "sensor-" + key, 28, 28, icon=glyph, icon_size=icon)
            for key, _, glyph in actions
        ]
        row_pads, row_margins = row_style.edges("padding", scale), row_style.edges("margin", scale)
        action_gap = row_style.get("columnGap", row_style.get("gap", 8)) * scale
        action_width = min(
            width,
            sum(size[0] for size in sizes)
            + action_gap
            + row_pads[0]
            + row_pads[2]
            + row_margins[0]
            + row_margins[2],
        )
        box, flow = component_flow(
            ui,
            "button_row",
            p.x + width - action_width,
            p.y,
            action_width,
            sizes,
            key="sensor-readout-actions",
            column_gap=8,
            row_gap=8,
        )
        heading = Rect(p.x, p.y, max(0, width - action_width - gap), box.outer.height)
        ui.component_text(
            "sensor",
            heading,
            "Readings",
            MUTED,
            size=style.get("detailSize", 12),
            font_field="detailSize",
        )
        ui.hits["sensor-readout-heading"] = heading.tuple()
        for (key, label, glyph), rect in zip(actions, flow.rectangles, strict=True):
            fw, fh = control_frame_size(ui, "sensor-" + key, rect.width, rect.height, icon=glyph)
            if ui.button(
                "sensor-" + key,
                rect.x,
                rect.y,
                fw,
                fh,
                icon=glyph,
                icon_size=icon,
                tooltip=label,
                width_limit=fw * scale,
                enabled=values is not None if key == "copy" else ctx.panels is not None,
            ):
                if key == "copy":
                    copy_state_vector(values)
                else:
                    self._open_plot(ctx)
        advance(ui, p.x, box.outer.bottom + gap)

    def _values(self, ctx, sensor, values):
        ui, scale = self.workspace.ui, self.workspace.s
        p = imgui.get_cursor_screen_pos()
        style = ui.component_style("sensor")
        margin = style.edges("margin", scale)
        width = max(0, self._width() - margin[0] - margin[2])
        x, y = p.x + margin[0], p.y + margin[1]
        size, detail = style.get("valueSize", 18), style.get("detailSize", 12)
        pads = style.edges("padding", scale, (8, 8, 8, 8))
        gap = style.get("columnGap", style.get("gap", 8)) * scale
        row_gap = style.get("rowGap", style.get("gap", 8)) * scale

        def measure(value):
            return ui.measure(
                value, size, mono=True, translate=False, style=style, font_resolved=True
            ), size * scale

        cartesian = sensor.dim == 3 and sensor_component_label(sensor.type, 3, 0) == "X"
        index_width = (
            0
            if sensor.dim == 1 or cartesian
            else ui.measure(
                str(max(sensor.dim - 1, 0)).rjust(2, "0"),
                detail,
                mono=True,
                translate=False,
                style=style,
                font_resolved=True,
            )
        )
        index_gap = gap if index_width else 0
        minimum = (
            measure("-0.00" if cartesian else "-0.0e+00")[0]
            + index_width
            + index_gap
            + pads[0]
            + pads[2]
        )
        columns = sensor_value_columns(width, sensor.dim, minimum, gap)
        if cartesian and columns < 3:
            columns = 1
        cell_width = max(0, (width - (columns - 1) * gap) / columns)
        value_line = max(size, style.line_height(size * 1.4)) * scale
        label_line = max(detail, style.line_height(detail * 1.4)) * scale if cartesian else 0
        label_gap = 4 * scale if label_line else 0
        height = max(
            style.get("height", 40) * scale,
            value_line + label_line + label_gap + pads[1] + pads[3],
            64 * scale if sensor.dim == 1 else 0,
        )
        rows = ceil(len(values) / columns)
        window = imgui.internal.get_current_window()
        first_row = max(0, floor((window.clip_rect.min.y - y) / (height + row_gap)))
        last_row = min(rows, ceil((window.clip_rect.max.y - y) / (height + row_gap)))
        for row in range(first_row, last_row):
            for column in range(columns):
                index = row * columns + column
                if index >= len(values):
                    break
                frame = Rect(
                    x + column * (cell_width + gap),
                    y + row * (height + row_gap),
                    cell_width,
                    height,
                )
                content = frame.inset(*pads)
                ui.rect(*frame.tuple(), INPUT, style.get("radius", DESIGN.control_radius))
                if sensor.dim > 1:
                    ui.component_text(
                        "sensor",
                        Rect(
                            content.x,
                            content.y,
                            content.width if cartesian else min(index_width, content.width),
                            label_line if cartesian else content.height,
                        ),
                        sensor_component_label(sensor.type, sensor.dim, index),
                        MUTED,
                        size=detail,
                        font_field="detailSize",
                        mono=True,
                        translate=False,
                    )
                value_box = Rect(
                    content.x + index_width + index_gap,
                    content.y + label_line + label_gap,
                    max(0, content.width - index_width - index_gap),
                    max(0, content.height - label_line - label_gap),
                )
                value = float(values[index])
                shown = fit_numeric_text(value, "%.6g", value_box.width, measure)
                ui.component_text(
                    "sensor",
                    value_box,
                    shown,
                    TEXT,
                    size=size,
                    align="center" if sensor.dim == 1 else "start",
                    font_field="valueSize",
                    mono=True,
                    translate=False,
                )
                key = f"sensor-value-{sensor.sensor_id}-{index}"
                imgui.set_cursor_screen_pos((frame.x, frame.y))
                imgui.invisible_button(key, (frame.width, frame.height))
                ui.hits[key] = frame.tuple()
                if ui.design is not None:
                    ui.design.record(
                        key,
                        ("sensor",),
                        frame.tuple(),
                        "sensor",
                        content=list(content.tuple()),
                        valueRect=list(value_box.tuple()),
                        fieldRegions={
                            "gap": [frame.right, frame.y, gap, frame.height],
                            "columnGap": [frame.right, frame.y, gap, frame.height],
                            "rowGap": [frame.x, frame.bottom, frame.width, row_gap],
                        },
                    )
                if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                    ui.tooltip(repr(value), translate=False)
        advance(ui, p.x, y + max(0, rows * (height + row_gap) - row_gap) + margin[3])

    def _missing(self, ctx):
        self._notice(ctx.tr("not produced this frame"))

    def _empty(self, ctx):
        self._notice(ctx.tr("This scene source provides no sensors."))

    def _notice(self, text):
        ui = self.workspace.ui
        style = ui.component_style("sensor")
        size = style.get("detailSize", 12)
        imgui.push_font(ui.resolve_font(style), size * ui.s)
        imgui.push_style_color(imgui.Col_.text, tuple(channel / 255 for channel in MUTED))
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + self._width())
        try:
            imgui.text_unformatted(text)
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
            imgui.pop_font()

    @part_scope("sensor")
    def draw(self, ctx):
        ui, scale = self.workspace.ui, self.workspace.s
        p, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
        style = ui.component_style("sensor")
        # A wide design canvas uses the same readable content measure as a detached panel.
        width = min(available.x, max(640, style.get("valueSize", 18) * 32) * scale)
        self._content_width = width
        self.presentation._control_width = self._content_width
        dimensions = replace(current_design(), body_size=style.get("fontSize", DESIGN.body_size))
        # Native items reset X to their enclosing group after submission. Keep
        # the panel-body anchor for selectors, metadata, actions and readouts.
        imgui.begin_group()
        try:
            with use_design(lambda _: dimensions), ui.component_scope("sensor"):
                super().draw(ctx)
        finally:
            imgui.end_group()
            self.presentation._control_width = None
            self._content_width = None
        advance(ui, p.x, imgui.get_cursor_screen_pos().y)
