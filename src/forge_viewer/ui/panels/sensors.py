"""Live MuJoCo sensor values sourced from Session."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import Panel, PanelContext, begin_kv_table, labeled


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
            imgui.text_disabled("no sensors")
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
            labeled("type", sensor.type.removeprefix("mjSENS_").lower())
            labeled("dimension", str(sensor.dim))
            if value is None:
                labeled("value", "not produced this frame")
            elif sensor.dim <= 6:
                labeled("value", np.array2string(value, precision=6, separator=", "))
            imgui.end_table()
        if value is not None and sensor.dim > 6:
            imgui.separator()
            imgui.text_disabled("value")
            formatted = np.array2string(
                value,
                precision=6,
                separator=", ",
                max_line_width=72,
            )
            if imgui.small_button("Copy"):
                imgui.set_clipboard_text(formatted)
            imgui.same_line()
            if imgui.small_button("Open in Plot") and ctx.panels is not None:
                panel = ctx.panels.get("Plot")
                focus = getattr(panel, "focus_sensor", None)
                if callable(focus):
                    focus(self.sensor_index)
                ctx.panels.open_panel("Plot")
            height = min(150.0, 38.0 + 18.0 * max(1, formatted.count("\n") + 1))
            if imgui.begin_child("sensor_value_block", imgui.ImVec2(0.0, height), 1):
                imgui.text_wrapped(formatted)
            imgui.end_child()
