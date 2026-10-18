"""Live sensor metadata and values supplied by the active scene source."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive.adapters.base import FrameNeeds
from mojive.ui.panels import (
    Panel,
    PanelContext,
    button_row_layout,
    button_width,
    copy_state_vector,
)
from mojive.ui.panels.presentation import PropertyPresentation

_CARTESIAN_SENSOR_TYPES = frozenset(
    (
        "accelerometer",
        "velocimeter",
        "gyro",
        "force",
        "torque",
        "magnetometer",
        "framepos",
        "framexaxis",
        "frameyaxis",
        "framezaxis",
        "framelinvel",
        "frameangvel",
        "framelinacc",
        "frameangacc",
        "subtreecom",
        "subtreelinvel",
        "subtreeangmom",
    )
)


def sensor_component_label(sensor_type: str, dimension: int, index: int) -> str:
    """Name Cartesian components only when the sensor metadata defines those axes."""
    kind = sensor_type.removeprefix("mjSENS_").lower()
    if dimension == 3 and kind in _CARTESIAN_SENSOR_TYPES:
        return "XYZ"[index]
    return str(index)


def sensor_value_preview(values, *, max_items: int = 24) -> str:
    """Format a bounded head/tail preview without walking the complete sensor array."""

    vector = np.asarray(values).reshape(-1)
    limit = max(2, int(max_items))
    if len(vector) <= limit:
        return np.array2string(vector, precision=6, separator=", ", max_line_width=72)
    head_count = limit // 2
    tail_count = limit - head_count
    head = np.array2string(vector[:head_count], precision=6, separator=", ", max_line_width=72)[
        1:-1
    ]
    tail = np.array2string(vector[-tail_count:], precision=6, separator=", ", max_line_width=72)[
        1:-1
    ]
    return f"[{head},\n ...,\n {tail}]"


class SensorsPanel(Panel):
    id = "sensors"
    name = "Sensors"
    default_open = False
    shortcut = "F11"

    def __init__(self) -> None:
        super().__init__()
        self.sensor_index = 0
        self.presentation: PropertyPresentation | None = None

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, sensors=True)

    def draw(self, ctx: PanelContext) -> None:
        infos = ctx.session.sensor_infos
        if not infos:
            self._empty(ctx)
            return

        self.sensor_index = max(0, min(self.sensor_index, len(infos) - 1))
        names = [sensor.name for sensor in infos]
        imgui.set_next_item_width(-1)
        _changed, self.sensor_index = (
            self.presentation.combo("##sensor", self.sensor_index, names)
            if self.presentation
            else imgui.combo("##sensor", self.sensor_index, names)
        )
        sensor = infos[self.sensor_index]
        values = ctx.session.frame.sensors
        value = (
            None
            if values is None or sensor.data_adr < 0 or sensor.data_adr + sensor.dim > len(values)
            else np.asarray(values[sensor.data_adr : sensor.data_adr + sensor.dim])
        )

        self._metadata(ctx, sensor)
        self._actions(ctx, value)
        if value is None:
            self._missing(ctx)
        else:
            self._values(ctx, sensor, value)

    def _metadata(self, ctx, sensor):
        kind = sensor.type.removeprefix("mjSENS_").lower()
        imgui.text_disabled(f"{kind} · {ctx.tr('dimension')} {sensor.dim}")

    def _missing(self, ctx):
        imgui.text_wrapped(ctx.tr("not produced this frame"))

    def _empty(self, ctx):
        imgui.text_wrapped(ctx.tr("This scene source provides no sensors."))

    def _open_plot(self, ctx):
        if ctx.panels is not None:
            panel = ctx.panels.load("Plot")
            focus = getattr(panel, "focus_sensor", None)
            if callable(focus):
                focus(self.sensor_index)
            ctx.panels.open_panel("Plot")

    def _actions(self, ctx, values):
        labels = (ctx.tr("Copy"), ctx.tr("Open in Plot"))
        inline = button_row_layout(
            tuple(button_width(label) for label in labels),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        button = self.presentation.button if self.presentation is not None else imgui.button
        imgui.begin_disabled(values is None)
        if button(labels[0]):
            copy_state_vector(values)
        imgui.end_disabled()
        if inline[1]:
            imgui.same_line()
        imgui.begin_disabled(ctx.panels is None)
        if button(labels[1]):
            self._open_plot(ctx)
        imgui.end_disabled()

    def _values(self, ctx, sensor, values):
        if sensor.dim == 1:
            imgui.text_unformatted(f"{float(values[0]):.6g}")
            imgui.set_item_tooltip(repr(float(values[0])))
            return
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.row_bg
        if not imgui.begin_table("sensor_components", 2, flags):
            return
        try:
            imgui.table_setup_column(ctx.tr("Component"), imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column(ctx.tr("value"), imgui.TableColumnFlags_.width_stretch)
            imgui.table_headers_row()
            clipper = imgui.ImGuiListClipper()
            clipper.begin(len(values))
            while clipper.step():
                for index in range(clipper.display_start, clipper.display_end):
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_unformatted(sensor_component_label(sensor.type, sensor.dim, index))
                    imgui.table_next_column()
                    imgui.text_unformatted(f"{float(values[index]):.6g}")
                    imgui.set_item_tooltip(repr(float(values[index])))
            clipper.end()
        finally:
            imgui.end_table()
