"""Live MuJoCo sensor values sourced from Session."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import (
    Panel,
    PanelContext,
    begin_kv_table,
    button_row_layout,
    button_width,
    labeled,
)


class SensorsPanel(Panel):
    name = "Sensors"
    default_open = False
    shortcut = "F11"

    def __init__(self) -> None:
        super().__init__()
        self.sensor_index = 0

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, sensors=True)

    def draw(self, ctx: PanelContext) -> None:
        infos = ctx.session.sensor_infos
        if not infos:
            imgui.text_disabled(ctx.tr("no sensors"))
            return

        self.sensor_index = min(self.sensor_index, len(infos) - 1)
        names = [sensor.name for sensor in infos]
        imgui.set_next_item_width(-1)
        _changed, self.sensor_index = imgui.combo("##sensor", self.sensor_index, names)
        sensor = infos[self.sensor_index]
        values = ctx.session.frame.sensors
        value = (
            None
            if values is None
            else np.asarray(values[sensor.data_adr : sensor.data_adr + sensor.dim])
        )

        if begin_kv_table("sensor_kv"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            labeled(ctx.tr("type"), sensor.type.removeprefix("mjSENS_").lower())
            labeled(ctx.tr("dimension"), str(sensor.dim))
            if value is None:
                labeled(ctx.tr("value"), ctx.tr("not produced this frame"))
            elif sensor.dim <= 6:
                labeled(ctx.tr("value"), np.array2string(value, precision=6, separator=", "))
            imgui.end_table()
        if value is not None and sensor.dim > 6:
            imgui.separator()
            imgui.text_disabled(ctx.tr("value"))
            formatted = np.array2string(
                value,
                precision=6,
                separator=", ",
                max_line_width=72,
            )
            labels = (ctx.tr("Copy"), ctx.tr("Open in Plot"))
            inline = button_row_layout(
                tuple(button_width(label) for label in labels),
                imgui.get_content_region_avail().x,
                imgui.get_style().item_spacing.x,
            )
            if imgui.small_button(labels[0]):
                imgui.set_clipboard_text(formatted)
            if inline[1]:
                imgui.same_line()
            if imgui.small_button(labels[1]) and ctx.panels is not None:
                panel = ctx.panels.get("Plot")
                focus = getattr(panel, "focus_sensor", None)
                if callable(focus):
                    focus(self.sensor_index)
                ctx.panels.open_panel("Plot")
            height = min(150.0, 38.0 + 18.0 * max(1, formatted.count("\n") + 1))
            if imgui.begin_child("sensor_value_block", imgui.ImVec2(0.0, height), 1):
                imgui.text_wrapped(formatted)
            imgui.end_child()
