"""Present selected sensor values with native component geometry."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.panels.sensors import SensorsPanel
from mojive.ui.workspace.design import DESIGN, current_design, use_design
from mojive.ui.workspace.property_style import NativePropertyPresentation, advance
from mojive.ui.workspace.widgets import INPUT, TEXT


class SensorPresentation(NativePropertyPresentation):
    def labeled(self, label, value):
        ui = self.ui
        style = ui.component_style("sensor")
        imgui.table_next_row()
        imgui.table_next_column()
        p = imgui.get_cursor_screen_pos()
        x, y, width = self.groups.row(p.x, p.y, imgui.get_content_region_avail().x)
        size = style.get("valueSize", DESIGN.mono_size)
        detail = style.get("detailSize", DESIGN.body_size)
        height = max(detail, size, style.line_height(max(detail, size) * 1.4)) * ui.s
        layout = ui.form_layout(width, 1, minimum_height=height)
        label_rect = Rect(
            x + layout.label_x, y + layout.label_y, layout.label_width, layout.label_height
        )
        ui.component_text(
            "sensor", label_rect, label, TEXT, font_field="detailSize", translate=False
        )
        dx, dy, field_width, field_height = layout.fields[0]
        ui.component_text(
            "sensor",
            Rect(x + dx, y + dy, field_width, field_height),
            str(value),
            TEXT,
            size=size,
            font_field="valueSize",
            mono=True,
            translate=False,
        )
        advance(ui, x, y + layout.height)


class WorkspaceSensors(SensorsPanel):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.presentation = SensorPresentation(workspace.ui, prefix="sensor")

    def draw(self, ctx):
        ui, scale = self.workspace.ui, self.workspace.s
        p, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
        style = ui.component_style("sensor")
        pads = style.edges("padding", scale)
        margin = style.edges("margin", scale)
        icon = style.get("iconSize", 16)
        line = (
            max(
                style.get("height", DESIGN.control_height),
                style.line_height(DESIGN.body_size * 1.4),
                icon,
            )
            * scale
        )
        gap = style.get("columnGap", style.get("gap", 8)) * scale
        width = max(scale, available.x - pads[0] - pads[2] - margin[0] - margin[2])
        x, y = p.x + pads[0] + margin[0], p.y + pads[1] + margin[1]
        draw = ui.draw
        splitter = imgui.ImDrawListSplitter()
        splitter.split(draw, 2)
        splitter.set_current_channel(draw, 1)
        title_rect = Rect(x, y, width, line)
        glyph = style.align_rect(title_rect, icon * scale, icon * scale, horizontal="start")
        ui.icon("chart", glyph.x, glyph.y, icon)
        ui.component_text(
            "sensor",
            Rect(glyph.right + gap, y, max(0, width - icon * scale - gap), line),
            "Sensors",
            TEXT,
            bold=True,
        )
        y += line + gap
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        # Scroll and selection remain owned by the registered SensorsPanel.
        child_height = max(scale, available.y - (y - p.y) - pads[3] - margin[3])
        imgui.begin_child("Sensor values", (width, child_height))
        imgui.pop_style_var()
        dimensions = replace(current_design(), body_size=style.get("detailSize", DESIGN.body_size))
        try:
            with use_design(lambda _: dimensions), ui.component_scope("sensor"):
                super().draw(ctx)
        finally:
            imgui.end_child()
        bottom = y + child_height + pads[3] + margin[3]
        splitter.set_current_channel(draw, 0)
        frame = Rect(
            p.x + margin[0],
            p.y + margin[1],
            available.x - margin[0] - margin[2],
            max(line, bottom - p.y - margin[1] - margin[3]),
        )
        ui.rect(*frame.tuple(), INPUT, style.get("radius", DESIGN.control_radius))
        splitter.merge(draw)
        if ui.design is not None:
            ui.design.record(
                "sensor-card",
                ("sensor",),
                frame.tuple(),
                "sensor",
                content=[x, p.y + pads[1] + margin[1], width, line + gap + child_height],
                fieldRegions={
                    name: [glyph.right, title_rect.y, gap, title_rect.height]
                    for name in ("gap", "columnGap")
                },
            )
        advance(ui, p.x, min(p.y + available.y, bottom))
