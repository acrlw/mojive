"""Scrolling telemetry plot panel."""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
from imgui_bundle import imgui

from mojive.adapters.base import FrameNeeds
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.numeric_text import fit_numeric_text
from mojive.ui.panels import Panel, PanelContext, segmented_control
from mojive.ui.panels.presentation import PropertyPresentation
from mojive.ui.panels.sensors import sensor_component_label
from mojive.ui.text_layout import fit_text, text_line_y
from mojive.ui.theme import pack_u32, with_alpha

CAPACITY = 512


class Ring:
    __slots__ = ("_span", "_token", "data", "filled", "index", "times")

    def __init__(self, capacity: int = CAPACITY) -> None:
        self.data = np.zeros(capacity, np.float32)
        self.times = np.zeros(capacity, np.float64)
        self.index = 0
        self.filled = 0
        self._token = None
        self._span = None

    def push(self, value: float, *, time: float | None = None, step: int = 0) -> None:
        if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
            return
        if time is not None:
            if not math.isfinite(time):
                return
            token = (time, step)
            if self._token is not None:
                if time < self._token[0] or step < self._token[1]:
                    self.clear()
                elif token == self._token:
                    # Editing a paused frame updates its value without inventing elapsed time.
                    self.data[(self.index - 1) % len(self.data)] = value
                    return
            self._token = token
        self.data[self.index] = value
        self.times[self.index] = 0.0 if time is None else time
        self.index = (self.index + 1) % len(self.data)
        self.filled = min(self.filled + 1, len(self.data))

    def clear(self) -> None:
        self.data[:] = 0.0
        self.index = 0
        self.filled = 0
        self._token = None
        self._span = None

    @property
    def duration(self) -> float:
        if self.filled < 2:
            return 0.0
        first = self.index if self.filled == len(self.data) else 0
        return max(0.0, float(self.times[(self.index - 1) % len(self.data)] - self.times[first]))

    @property
    def values(self) -> np.ndarray:
        """Return initialized samples without copying their backing storage."""
        return self.data if self.filled == len(self.data) else self.data[: self.filled]

    @property
    def offset(self) -> int:
        return self.index if self.filled == len(self.data) else 0

    def span(self) -> tuple[float, float]:
        if self.filled == 0:
            return -1.0, 1.0
        view = self.values
        lo, hi = float(view.min()), float(view.max())
        mid = (lo + hi) * 0.5
        width = max((hi - lo) * 1.16, 0.001, max(abs(lo), abs(hi)) * 0.02)
        bounds = (mid - width * 0.5, mid + width * 0.5)
        if self._span is None:
            self._span = bounds
        else:
            # Keep a constant channel's scale stable when older extrema leave the window.
            self._span = min(self._span[0], bounds[0]), max(self._span[1], bounds[1])
        return self._span


def joint_channels(kind: str, *, velocity=False):
    """Describe state coordinates using the shared joint contract, without adapter lookups."""
    if kind == "hinge":
        return (("Angular velocity", "", "rad/s"),) if velocity else (("Angle", "", "rad"),)
    if kind == "slide":
        return (("Linear velocity", "", "m/s"),) if velocity else (("Displacement", "", "m"),)
    if kind in ("ball", "free"):
        if velocity:
            rotation = tuple(("Angular velocity", axis, "rad/s") for axis in "XYZ")
            position = tuple(("Linear velocity", axis, "m/s") for axis in "XYZ")
        else:
            rotation = tuple(("Quaternion", axis, "") for axis in "WXYZ")
            position = tuple(("Position", axis, "m") for axis in "XYZ")
        return (*position, *rotation) if kind == "free" else rotation
    return (("Velocity" if velocity else "Position", "0", ""),)


def channel_label(ctx, channel):
    return " ".join(part for part in (ctx.tr(channel[0]), channel[1]) if part)


class PlotPanel(Panel):
    id = "plot"
    name = "Plot"
    default_open = False
    shortcut = "F7"

    def __init__(self) -> None:
        super().__init__()
        self.source = "joint"
        self.presentation: PropertyPresentation | None = None
        self.show_angle = True
        self.show_velocity = True
        self.show_contact = False

        self.joint_index = 0
        self.position_component = 0
        self.velocity_component = 0
        self.sensor_index = 0
        self.sensor_component = 0
        self._angle = Ring()
        self._velocity = Ring()
        self._contact = Ring()
        self._sensor = Ring()
        self._tracked = -1
        self._tracked_sensor = (-1, -1)
        self._structure_generation = -1
        self._current = {}
        self._points = {}

    def focus_sensor(self, sensor_index: int, component: int = 0) -> None:
        """Open the plot on one sensor component from a diagnostic deep link."""

        self.source = "sensor"
        self.sensor_index = max(0, int(sensor_index))
        self.sensor_component = max(0, int(component))
        self._tracked_sensor = (-1, -1)

    def frame_needs(self) -> FrameNeeds:
        if self.source == "sensor":
            return FrameNeeds(poses=False, sensors=True)
        return FrameNeeds(
            poses=False,
            qpos=self.show_angle,
            qvel=self.show_velocity,
            contacts=self.show_contact,
        )

    def draw(self, ctx: PanelContext) -> None:
        if self._structure_generation != ctx.session.structure_generation:
            self._structure_generation = ctx.session.structure_generation
            for ring in (self._angle, self._velocity, self._contact, self._sensor):
                ring.clear()
            self._current.clear()
        self._toolbar(ctx)
        if self.source == "sensor":
            self._draw_sensor(ctx)
        else:
            self._draw_joint(ctx)

    def _toolbar(self, ctx):
        start = imgui.get_cursor_screen_pos()
        width, scale = imgui.get_content_region_avail().x, ctx.style_scale
        gap = imgui.get_style().item_spacing.x
        inline = width >= 560 * scale
        source_width = min(208 * scale, width) if inline else width
        channels_width = min(112 * scale, (width - gap) * 0.44)
        source_labels = (ctx.tr("Joints"), ctx.tr("Sensors"))
        imgui.set_next_item_width(source_width)
        selected = int(self.source != "joint")
        source = (
            self.presentation.segmented("plot-source", source_labels, selected)
            if self.presentation
            else segmented_control(
                "plot-source", source_labels, selected, width=source_width, theme=ctx.theme
            )
        )
        self.source = "joint" if source == 0 else "sensor"
        control_height = (
            self.presentation.choice_height() if self.presentation else imgui.get_frame_height()
        )
        row_bottom = max(imgui.get_item_rect_max().y, start.y + control_height)
        if inline:
            object_x, object_y = start.x + source_width + gap, start.y
            object_width = width - source_width - gap
            if self.source == "joint":
                object_width -= channels_width + gap
        else:
            object_x, object_y = start.x, row_bottom + imgui.get_style().item_spacing.y
            object_width = width
            if self.source == "joint":
                object_width -= channels_width + gap
        imgui.set_cursor_screen_pos((object_x, object_y))
        imgui.set_next_item_width(object_width)
        if self.source == "joint":
            objects = ctx.session.joints
            self.joint_index = max(0, min(self.joint_index, len(objects) - 1))
            names = tuple(j.name or f"joint{j.joint_id}" for j in objects)
            imgui.begin_disabled(not names)
            changed, self.joint_index = self._combo(
                "##joint", self.joint_index, names or (ctx.tr("No data"),)
            )
            imgui.end_disabled()
            if changed or self._tracked != self.joint_index:
                self._angle.clear()
                self._velocity.clear()
                self._tracked = self.joint_index
                self.position_component = self.velocity_component = 0
        else:
            objects = ctx.session.sensor_infos
            self.sensor_index = max(0, min(self.sensor_index, len(objects) - 1))
            imgui.begin_disabled(not objects)
            _, self.sensor_index = self._combo(
                "##plot_sensor",
                self.sensor_index,
                tuple(sensor.name for sensor in objects) or (ctx.tr("No data"),),
            )
            imgui.end_disabled()
        bottom = max(row_bottom, imgui.get_item_rect_max().y, object_y + control_height)
        if self.source == "joint" and objects:
            joint = objects[self.joint_index]
            labels = (
                ctx.tr(joint_channels(joint.type)[0][0]),
                ctx.tr(joint_channels(joint.type, velocity=True)[0][0]),
                ctx.tr("Total contact force"),
            )
            imgui.set_cursor_screen_pos((start.x + width - channels_width, object_y))
            imgui.set_next_item_width(channels_width)
            values = (self.show_angle, self.show_velocity, self.show_contact)
            if self.presentation:
                values = self.presentation.multi_select("plot-channels", labels, values)
            elif imgui.begin_combo("##plot-channels", ctx.tr("Channels")):
                edited = list(values)
                for index, label in enumerate(labels):
                    _, edited[index] = imgui.checkbox(label, edited[index])
                values = tuple(edited)
                imgui.end_combo()
            self.show_angle, self.show_velocity, self.show_contact = values
            bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.set_cursor_screen_pos((start.x, bottom + imgui.get_style().item_spacing.y))

    def _draw_joint(self, ctx: PanelContext) -> None:
        joints = ctx.session.joints
        if not joints:
            self._line(ctx.tr("no joints to plot"), muted=True)
            return
        joint = joints[self.joint_index]
        positions = joint_channels(joint.type)
        velocities = joint_channels(joint.type, velocity=True)
        self.position_component = min(self.position_component, len(positions) - 1)
        self.velocity_component = min(self.velocity_component, len(velocities) - 1)
        self._sample(ctx)
        cards = []
        for enabled, key, ring, channels, attribute in (
            (self.show_angle, "qpos", self._angle, positions, "position_component"),
            (self.show_velocity, "qvel", self._velocity, velocities, "velocity_component"),
        ):
            if enabled:
                cards.append((key, ring, channels, attribute))
        if self.show_contact:
            cards.append(("contact", self._contact, (("Total contact force", "", "N"),), None))
        self._draw_cards(ctx, cards)

    def _draw_sensor(self, ctx: PanelContext) -> None:
        infos = ctx.session.sensor_infos
        if not infos:
            self._line(ctx.tr("no sensors to plot"), muted=True)
            return
        sensor = infos[self.sensor_index]
        self.sensor_component = max(0, min(self.sensor_component, sensor.dim - 1))
        tracked = (self.sensor_index, self.sensor_component)
        if tracked != self._tracked_sensor:
            self._sensor.clear()
            self._tracked_sensor = tracked
        self._sample_sensor(ctx)
        # SensorInfo defines no unit contract; unknown units must remain unspecified.
        channels = tuple(
            ("Component", sensor_component_label(sensor.type, sensor.dim, index), "")
            for index in range(sensor.dim)
        ) or (("value", "", ""),)
        self._draw_cards(ctx, [("sensor", self._sensor, channels, "sensor_component")])

    def _sample_sensor(self, ctx):
        sensor = ctx.session.sensor_infos[self.sensor_index]
        self._current.clear()
        self._record(
            "sensor",
            self._sensor,
            ctx.session.frame,
            ctx.session.frame.sensors,
            sensor.data_adr + self.sensor_component if sensor.dim > 0 else -1,
        )

    def _draw_cards(self, ctx, cards):
        rings = tuple(card[1] for card in cards)
        window = sample_window(rings)
        self._status(ctx, rings, window)
        if not cards:
            self._line(ctx.tr("Select channels to display."), muted=True)
            return
        start = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail()
        scale = ctx.style_scale
        gap = max(imgui.get_style().item_spacing.x, imgui.get_style().item_spacing.y)
        columns = max(1, min(len(cards), int((available.x + gap) / (240 * scale + gap))))
        rows = math.ceil(len(cards) / columns)
        width = (available.x - gap * (columns - 1)) / columns
        line = imgui.get_text_line_height()
        header = max(
            self.presentation.choice_height() if self.presentation else imgui.get_frame_height(),
            line + 12 * scale,
        )
        overhead = header + line + 32 * scale
        height = max(
            80 * scale, min(220 * scale, (available.y - gap * (rows - 1)) / rows - overhead)
        )
        card_height = height + overhead
        for index, (key, ring, channels, attribute) in enumerate(cards):
            x = start.x + index % columns * (width + gap)
            y = start.y + index // columns * (card_height + gap)
            imgui.set_cursor_screen_pos((x, y))
            self._curve(
                key,
                ring,
                height,
                "",
                ctx,
                width=width,
                window=window,
                channels=channels,
                attribute=attribute,
                header_height=header,
            )
        imgui.set_cursor_screen_pos((start.x, start.y + rows * (card_height + gap) - gap))
        imgui.dummy((available.x, 0))

    def _combo(self, key, index, labels):
        return (
            self.presentation.combo(key, index, labels)
            if self.presentation
            else imgui.combo(key, index, labels)
        )

    def _sample(self, ctx: PanelContext) -> None:
        frame = ctx.session.frame
        joints = ctx.session.joints
        j = joints[self.joint_index]

        self._current.clear()
        if self.show_angle:
            self._record(
                "qpos",
                self._angle,
                frame,
                frame.qpos,
                j.qpos_adr + self.position_component if j.qpos_adr >= 0 else -1,
            )
        if self.show_velocity:
            self._record(
                "qvel",
                self._velocity,
                frame,
                frame.qvel,
                j.qvel_adr + self.velocity_component if j.qvel_adr >= 0 else -1,
            )
        if self.show_contact:
            c = frame.contacts
            values = None if c is None else (float(np.abs(c[:, 6]).sum()) if len(c) else 0.0,)
            self._record("contact", self._contact, frame, values, 0)

    def _record(self, name, ring, frame, values, address):
        value = (
            float(values[address]) if values is not None and 0 <= address < len(values) else None
        )
        self._current[name] = value
        if value is not None:
            ring.push(value, time=frame.time, step=frame.step)

    def _status(self, ctx, rings, window):
        state = ctx.tr("Paused") if ctx.session.paused else ctx.tr("Live")
        if not self._current or all(
            value is None or not math.isfinite(value) for value in self._current.values()
        ):
            state = ctx.tr("No data")
        counts = [ring.filled for ring in rings]
        low, high = min(counts, default=0), max(counts, default=0)
        count = str(high) if low == high else f"{low}–{high}"
        samples = ctx.tr("sample" if low == high == 1 else "samples")
        self._line(f"{state} · {window[1] - window[0]:.3g} s · {count} {samples}", muted=True)

    def _line(self, text, *, muted=False, tooltip=None):
        """Allocate one line regardless of live numeric magnitude or title length."""
        shown = fit_text(ImguiDraw2D(), text, imgui.get_content_region_avail().x)
        if muted:
            imgui.text_disabled(shown)
        else:
            imgui.text_unformatted(shown)
        if (shown != text or tooltip) and imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
            self._tooltip(tooltip or text, hovered=True)

    def _numeric_style(self, size_scale=18 / 14):
        # Current values use an 18 pt reference; auxiliary axes use 12 pt with
        # the same face. Both scale with the owning group's 14 pt body style.
        return (
            self.presentation.numeric_text_style(size_scale) if self.presentation else nullcontext()
        )

    def _curve(
        self,
        label,
        ring,
        height,
        overlay,
        ctx=None,
        unit="",
        *,
        width=None,
        window=None,
        channels=None,
        attribute=None,
        header_height=None,
    ):
        if ctx is None:
            lo, hi = ring.span()
            imgui.plot_lines(
                f"##{label}",
                ring.values,
                values_offset=ring.offset,
                overlay_text=f"{label}  {overlay}",
                scale_min=lo,
                scale_max=hi,
                graph_size=imgui.ImVec2(-1, height),
            )
            return
        scale = ctx.style_scale
        draw, origin = ImguiDraw2D(), imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x if width is None else width
        pad, gap, line = 12 * scale, 8 * scale, imgui.get_text_line_height()
        header_height = header_height or max(imgui.get_frame_height(), line + 12 * scale)
        card_height = header_height + line + 32 * scale + height
        draw.rect_filled(
            (origin.x, origin.y),
            (origin.x + width, origin.y + card_height),
            ctx.theme.bg_child,
            rounding=imgui.get_style().child_rounding,
            smoothing=0,
        )
        channels = channels or ((overlay, "", unit),)
        component = getattr(self, attribute) if attribute else 0
        channel = channels[component]
        unit = channel[2]
        with self._numeric_style():
            # The budget depends on font/format, never the current sample's magnitude.
            desired = draw.text_size("-8.88888e-308")[0]
        inner_width = max(1, width - 2 * pad)
        value_width = min(desired, inner_width * 0.56)
        title_width = max(1, inner_width - value_width - gap)
        top = origin.y + pad
        title = channel_label(ctx, channel)
        imgui.set_cursor_screen_pos((origin.x + pad, top))
        if len(channels) > 1:
            imgui.set_next_item_width(title_width)
            changed, component = self._combo(
                "##plot-" + label + "-component",
                component,
                tuple(channel_label(ctx, c) for c in channels),
            )
            if changed:
                setattr(self, attribute, component)
                ring.clear()
                if label == "sensor":
                    self._tracked_sensor = (self.sensor_index, component)
                    self._sample_sensor(ctx)
                else:
                    self._sample(ctx)
                channel, unit = channels[component], channels[component][2]
                title = channel_label(ctx, channel)
            header_height = max(header_height, imgui.get_item_rect_max().y - top)
        else:
            imgui.set_cursor_screen_pos((origin.x + pad, top + (header_height - line) / 2))
            shown = fit_text(draw, title, title_width)
            imgui.text_unformatted(shown)
            if shown != title:
                self._tooltip(title)
        value = self._current.get(label)
        suffix = f" {unit}" if unit else ""
        imgui.set_cursor_screen_pos(
            (origin.x + width - pad - value_width, top + (header_height - line) / 2)
        )
        numeric = value is not None and math.isfinite(value)
        with self._numeric_style() if numeric else nullcontext():
            current = (
                fit_numeric_text(value, "%.6g", value_width, draw.text_size)
                if numeric
                else ctx.tr("Unavailable" if value is None else "Invalid value")
            )
            shown = fit_text(draw, current, value_width)
            imgui.set_cursor_screen_pos(
                (
                    origin.x + width - pad - draw.text_size(shown)[0],
                    text_line_y(draw, top + header_height / 2),
                )
            )
            imgui.text_unformatted(shown)
            if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                self._tooltip(
                    title + ": " + (repr(value) + suffix if numeric else current), hovered=True
                )
        graph_y = top + header_height + 6 * scale
        self._graph(
            label,
            ring,
            ctx,
            origin.x + pad,
            graph_y,
            inner_width,
            height,
            window or sample_window((ring,)),
            unit,
        )
        imgui.set_cursor_screen_pos((origin.x, origin.y + card_height))
        imgui.dummy((width, 0))

    def _graph(self, key, ring, ctx, x, y, width, height, window, unit=""):
        scale, draw = ctx.style_scale, ImguiDraw2D()
        lo, hi = ring.span()
        # Keep points clear of their caps and the short in-plot range labels.
        inset, label_height = 4 * scale, imgui.get_text_line_height()
        left, right = x + inset, x + width - inset
        top, bottom = y + label_height + 4 * scale, y + height - label_height - 4 * scale
        theme = ctx.theme
        grid = with_alpha(theme.border, 0.48)
        for fraction in (0.0, 0.5, 1.0):
            py = top + (bottom - top) * fraction
            draw.line((left, py), (right, py), grid, scale)
        with self._numeric_style(12 / 14):
            unit_width = draw.text_size(unit)[0]
            budget = max(1, width - unit_width - 16 * scale)
            top_text_y = text_line_y(draw, y + label_height / 2)
            bottom_text_y = text_line_y(draw, bottom + 4 * scale + label_height / 2)
            draw.text(
                (left, top_text_y),
                theme.text_disabled,
                fit_numeric_text(hi, "%.3g", budget, draw.text_size),
            )
            draw.text(
                (left, bottom_text_y),
                theme.text_disabled,
                fit_numeric_text(lo, "%.3g", budget, draw.text_size),
            )
            draw.text((right - unit_width, top_text_y), theme.text_disabled, unit)
        points = self._points.setdefault(key, [])
        while len(points) < ring.filled:
            points.append(imgui.ImVec2())
        del points[ring.filled :]
        start, end = window
        for index, point in enumerate(points):
            offset = (ring.offset + index) % len(ring.data)
            fraction = (ring.times[offset] - start) / (end - start) if end > start else 1.0
            point.x = left + (right - left) * max(0.0, min(1.0, fraction))
            point.y = bottom - (bottom - top) * (float(ring.data[offset]) - lo) / (hi - lo)
        dl = imgui.get_window_draw_list()
        if len(points) > 1:
            dl.add_polyline(points, pack_u32(theme.primary), 1.5 * scale, 0)
        if points:
            point = points[-1]
            draw.circle_filled((point.x, point.y), 2.5 * scale, theme.primary)
        else:
            message = ctx.tr("No data")
            tw, th = draw.text_size(message)
            draw.text(
                (x + (width - tw) / 2, top + (bottom - top - th) / 2), theme.text_disabled, message
            )
        with self._numeric_style(12 / 14):
            end_text = f"{end:.4g} s"
            end_width = draw.text_size(end_text)[0]
            axis_y = text_line_y(draw, y + height + 4 * scale + label_height / 2)
            draw.text((right - end_width, axis_y), theme.text_disabled, end_text)
            if end > start:
                start_text = f"{start:.4g} s"
                if draw.text_size(start_text)[0] + end_width + 16 * scale <= width:
                    draw.text((left, axis_y), theme.text_disabled, start_text)
        imgui.set_cursor_screen_pos((x, y))
        imgui.invisible_button("##plot-graph-" + key, (width, height))
        if points and imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
            mouse = imgui.get_io().mouse_pos.x
            nearest = min(
                reversed(range(len(points))), key=lambda index: abs(points[index].x - mouse)
            )
            offset = (ring.offset + nearest) % len(ring.data)
            self._tooltip(
                ctx.tr("{value} {unit} · {time} s").format(
                    value=f"{float(ring.data[offset]):.9g}",
                    unit=unit,
                    time=f"{ring.times[offset]:.6g}",
                ),
                hovered=True,
            )


def sample_window(rings):
    """Use one actual time domain for every visible channel, including partial histories."""
    populated = [ring for ring in rings if ring.filled]
    if not populated:
        return 0.0, 0.0
    return (
        min(float(ring.times[ring.offset]) for ring in populated),
        max(float(ring.times[(ring.index - 1) % len(ring.data)]) for ring in populated),
    )
