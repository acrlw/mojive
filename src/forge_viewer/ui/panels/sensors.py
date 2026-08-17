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

        if begin_kv_table("sensor_kv"):
            labeled("type", sensor.kind.removeprefix("mjSENS_").lower())
            labeled("dimension", str(sensor.dim))
            if values is None:
                labeled("value", "not produced this frame")
            else:
                value = np.asarray(values[sensor.data_adr : sensor.data_adr + sensor.dim])
                labeled("value", np.array2string(value, precision=6, separator=", "))
            imgui.end_table()
