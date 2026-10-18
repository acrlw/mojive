"""Native presentation of the registered editor's real timeline and transport."""

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import Box, Rect, part_scope
from mojive.ui.icons import draw_icon_label
from mojive.ui.keyframe_editor.controls import FOLLOW_MODE_TOOLTIPS
from mojive.ui.keyframe_editor.toolbar import TimelineToolbar
from mojive.ui.keyframe_editor.track import draw_dope_sheet
from mojive.ui.timeline import nearest_take_frame, timeline_range_format, valid_timeline_range
from mojive.ui.workspace.component_layout import (
    control_frame_size,
    control_size,
)
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.form_popup import form_popup, measure_form_popup
from mojive.ui.workspace.property_style import NativePropertyPresentation, toggle_row_height
from mojive.ui.workspace.timeline_body import body_corner_scope, body_geometry
from mojive.ui.workspace.timeline_details import NativeTimelineDetails
from mojive.ui.workspace.timeline_header import toolbar_group_layout
from mojive.ui.workspace.timeline_track import NativeTimelineTrack, measure_layout
from mojive.ui.workspace.widgets import MUTED
from mojive.ui.workspace.window_chrome import native_child_extent

_COMMAND_ICONS = {
    "first": "transport-first",
    "last": "transport-last",
    "previous": "stepb",
    "next": "stepf",
    "play": "play",
    "pause": "pause",
    "stop": "stop",
    "loop": "loop",
    "options": "settings",
    "record": "record",
    "add": "plus",
    "clear": "delete",
    "key-previous": "previous",
    "key-next": "next",
    "fit": "fit",
    "follow": "follow",
    "view": "video",
    "key": "snapshot",
    "key-keyframe": "diamond",
}
_FOLLOW_ICONS = {
    "key-follow-off": "follow-off",
    "key-follow-page": "follow-page",
    "key-follow-locked": "follow-locked",
}
_FOLLOW_MODES = ("off", "page", "locked")
_NATIVE_FOLLOW_ICONS = tuple(_FOLLOW_ICONS.values())
_BUTTON_HITS = {
    "##take-first": "timeline-first",
    "##take-previous": "timeline-previous",
    "##take-play-pause": "timeline-play-pause",
    "##take-next": "timeline-next",
    "##take-last": "timeline-last",
    "##take-record": "timeline-record",
    "##take-video": "timeline-video",
    "##capture-snapshot": "timeline-snapshot",
    "##key-view": "timeline-fit",
    "##timeline-loop": "timeline-loop",
    "##add-model-keyframe": "timeline-add-key",
}


@dataclass(frozen=True)
class _RangeEdit:
    """Range state before a numeric drag, scoped to its original take."""

    key: str
    take_id: int
    selected: tuple[int, int] | None
    loop_enabled: bool
    view: tuple[float, float, bool]
    manual_view_range: tuple[float, float] | None


class _ToolbarRows:
    """Keep measured control groups together while allowing narrow rows to wrap."""

    def __init__(self, origin, width, scale, *, column_gap=None, row_gap=None, placements=()):
        self.left, self.top = origin.x, origin.y
        self.right = origin.x + width
        self.scale = scale
        self.x, self.y, self.row_height = self.left, self.top, 0.0
        self.column_gap = DESIGN.timeline_column_gap if column_gap is None else column_gap
        self.row_gap = DESIGN.timeline_row_gap if row_gap is None else row_gap
        self.placements = iter(placements)
        self.planned = bool(placements)
        self.planned_bottom = max((r.bottom for r in placements), default=self.top)

    def _next_row(self):
        self.x = self.left
        self.y += self.row_height + self.row_gap * self.scale
        self.row_height = 0.0

    def group(self, width):
        if self.planned or self.x == self.left:
            return
        if self.x + 2 * self.column_gap * self.scale + width > self.right:
            self._next_row()
        else:
            self.x += self.column_gap * self.scale

    def place(self, width, height=28):
        if self.planned:
            rect = next(self.placements)
            self.y = rect.y
            return rect.x, rect.y, rect.width / self.scale
        width = min(width * self.scale, self.right - self.left)
        if self.x > self.left and self.x + width > self.right + 0.5:
            self._next_row()
        position = self.x, self.y
        self.x += width + self.column_gap * self.scale
        self.row_height = max(self.row_height, height * self.scale)
        return *position, width / self.scale

    @property
    def bottom(self):
        return self.planned_bottom if self.planned else self.y + self.row_height


class NativeTimelineToolbar(TimelineToolbar):
    """Keep shared commands and responsive groups with the native icon family."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.settings_fields = NativePropertyPresentation(workspace.ui, prefix="timeline-settings")
        self._recording_edit: tuple[str, float] | None = None
        self._range_edit: _RangeEdit | None = None
        super().__init__(self.draw_follow_icon, command_icon_drawer=self.draw_command_icon)

    def _icon(self, name, center, color, scale):
        size = 28 * scale
        self.workspace.ui.role_icon(
            name,
            center[0] - size / 2,
            center[1] - size / 2,
            size,
            size,
            role="toolbar",
            c=tuple(round(value * 255) for value in color),
        )

    def draw_command_icon(self, draw, center, kind, color, scale, *, smoothing=0.6):
        self._icon(_COMMAND_ICONS[kind], center, color, scale)
        item_id = imgui.get_item_id()
        for label, hit in _BUTTON_HITS.items():
            if item_id == imgui.get_id(label):
                lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                self.workspace.ui.hits[hit] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
                break

    def draw_follow_icon(self, draw, lo, hi, color, scale, name, label):
        if label:
            draw_icon_label(draw, lo, hi, color, scale, name, label)
        else:
            self._icon(
                _FOLLOW_ICONS[name], ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), color, scale
            )

    def _collapse(self, ctx, *, width_limit=None):
        pos = imgui.get_cursor_screen_pos()
        ui = self.workspace.ui
        measured = control_size(ui, "collapse-timeline", 28, 28, icon="down")
        frame = control_frame_size(
            ui,
            "collapse-timeline",
            min(measured[0], width_limit) if width_limit is not None else measured[0],
            measured[1],
            icon="down",
        )
        if self.workspace.ui.button(
            "collapse-timeline",
            pos.x,
            pos.y,
            *frame,
            icon="down",
            tooltip=ctx.tr("Collapse Timeline"),
            width_limit=frame[0] * ui.s,
        ):
            self.workspace.collapse_bottom()

    @part_scope("timeline_toolbar")
    def draw_compact_toolbar(self, editor, ctx, take_times):
        """Keep transport, capture, range and view controls in complete wrapping groups."""
        ui, scale = self.workspace.ui, ctx.style_scale
        origin = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail().x
        width = available / scale
        style = ui.component_style("timeline_toolbar")
        pads = style.edges("padding", scale)
        margins = style.edges("margin", scale)
        content_width = max(scale, available - pads[0] - pads[2] - margins[0] - margins[2])
        content_width = min(
            content_width,
            max(0, style.dimension("width", available / scale) * scale - pads[0] - pads[2]),
        )
        width = content_width / scale
        labels = self._range_labels(ctx, take_times)
        label_width = max(ui.measure(title) for title in labels) / scale
        number_width, number_height = self._numeric_size(
            timeline_range_format(*self._range_endpoints(editor, ctx, take_times), suffix=" s")
        )
        from mojive.ui.workspace.labeled_input import measure_labeled_input

        loop_size = control_size(ui, "timeline-loop", 28, 28, icon="loop")
        column_gap = style.get("columnGap", style.get("gap", DESIGN.timeline_column_gap)) * scale
        range_budget = max(0, (content_width - loop_size[0] - 2 * column_gap) / 2)
        range_layout = measure_labeled_input(
            ui,
            range_budget,
            labels[0],
            (number_width * scale, number_height * scale),
            label_width=label_width * scale,
        )
        range_size = (range_layout.width, range_layout.height)
        capture_labels = width >= 600
        captures = tuple(
            (
                kind,
                caption if capture_labels else "",
                max(minimum, ui.measure(caption) / scale + 37) if capture_labels else 28,
            )
            for kind, caption, minimum in (
                ("snapshot", "Snapshot", 100),
                ("video", "Stop" if ctx.take_video_active else "Video", 76),
            )
        )

        def button_size(name, baseline, *, icon, label=""):
            return control_size(ui, "timeline-" + name, baseline, 28, icon=icon, label=label)

        playing, recording = ctx.session.state_take_playing, ctx.session.state_take_recording
        compact_play = width < 360
        compact_record = width < 360
        sizes = [
            button_size(
                name,
                (28 if compact_play else 72) if name == "play-pause" else 24,
                icon=("pause" if playing else "play")
                if name == "play-pause"
                else _COMMAND_ICONS[name],
                label=("Pause" if playing else "Play")
                if name == "play-pause" and not compact_play
                else "",
            )
            for name in ("first", "previous", "play-pause", "next", "last")
        ]
        sizes += [
            (80 * scale, 28 * scale),
            button_size(
                "record",
                28 if compact_record else 88,
                icon="stop" if recording else "record",
                label="" if compact_record else "Stop" if recording else "Record simulation state",
            ),
            (120 * scale, 28 * scale),
        ]
        sizes += [
            button_size(kind, w, icon="snapshot" if kind == "snapshot" else "video", label=caption)
            for kind, caption, w in captures
        ]
        sizes += [
            range_size,
            range_size,
            loop_size,
            (90 * scale, 28 * scale),
            button_size("fit", 28, icon="fit"),
            button_size("settings", 28, icon="settings"),
            control_size(ui, "collapse-timeline", 28, 28, icon="down"),
        ]
        box, layout, groups = toolbar_group_layout(
            ui,
            origin.x,
            origin.y,
            available,
            (sizes[:6], sizes[6:10], sizes[10:13], sizes[13:]),
            column_gap=DESIGN.timeline_column_gap,
            row_gap=DESIGN.timeline_row_gap,
        )
        for name, rect in zip(("transport", "capture", "range", "view"), groups, strict=True):
            ui.hits["timeline-group-" + name] = rect.tuple()
        available = box.content.width
        rows = _ToolbarRows(
            imgui.ImVec2(box.content.x, box.content.y),
            available,
            scale,
            placements=layout.rectangles,
        )
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (DESIGN.frame_padding_x * scale, DESIGN.control_padding_y(BODY_SIZE) * scale),
        )
        try:
            for name in ("first", "previous", "play-pause", "next", "last"):
                button_width = 72 if name == "play-pause" else 24
                x, y, allocated = rows.place(button_width)
                self._draw_transport(
                    editor,
                    ctx,
                    take_times,
                    x,
                    y,
                    4,
                    names=(name,),
                    compact_play=compact_play,
                    width_limit=allocated * scale,
                )
            x, y, time_width = rows.place(80)
            self._draw_time(editor, ctx, x, y, width=time_width)
            tools_top = rows.y
            record_width = 28 if compact_record else 88
            x, y, allocated = rows.place(record_width)
            self._draw_record(
                editor, ctx, x, y, compact=record_width == 28, width_limit=allocated * scale
            )
            x, y, take_width = rows.place(120)
            self._draw_take_selector(editor, ctx, x, y, take_width)
            for kind, caption, capture_width in captures:
                x, y, capture_width = rows.place(capture_width)
                self._draw_capture(editor, ctx, kind, x, y, capture_width, caption)
            self._draw_range(editor, ctx, take_times, rows, field_width=range_budget)
            x, y, _ = rows.place(90)
            self._draw_follow(editor, ctx, x, y)
            x, y, allocated = rows.place(28)
            self._draw_fit(editor, ctx, x, y, width_limit=allocated * scale)
            x, y, allocated = rows.place(28)
            imgui.set_cursor_screen_pos((x, y))
            self._draw_settings(ctx, width_limit=allocated * scale)
            x, y, allocated = rows.place(28)
            imgui.set_cursor_screen_pos((x, y))
            self._collapse(ctx, width_limit=allocated * scale)
            ui.hits["timeline-tools"] = (origin.x, tools_top, available, rows.bottom - tools_top)
            imgui.set_cursor_screen_pos((origin.x, rows.bottom + box.padding[3] + box.margin[3]))
            # Manual control placement must publish its final extent to ImGui.
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
            imgui.dummy(
                (available, style.get("rowGap", style.get("gap", DESIGN.timeline_row_gap)) * scale)
            )
            imgui.pop_style_var()
        finally:
            imgui.pop_style_var()

    def _draw_settings(self, ctx, *, width_limit=None):
        ui, scale = self.workspace.ui, ctx.style_scale
        pos = imgui.get_cursor_screen_pos()
        if self._button(
            ctx,
            "settings",
            pos.x,
            pos.y,
            width=28,
            icon="settings",
            tooltip=ctx.tr("Timeline settings"),
            width_limit=width_limit,
        ):
            imgui.open_popup("Timeline settings")
        with part_scope("form_popup", inherit=False):
            labels = [ctx.tr("Pause at last frame")]
            if ctx.recording_config is not None:
                labels.extend(ctx.tr(label) for label in ("Start delay (s)", "End hold (s)"))
            number_width, number_height = self._numeric_size("%.1f")
            layout = measure_form_popup(
                ui,
                labels,
                field_width=max(96 * scale, number_width * scale, ui.numeric_minimum()),
                row_height=max(number_height * scale, toggle_row_height(ui)),
                group_starts=(1,),
            )
            with form_popup(ui, "Timeline settings", layout) as opened:
                if opened:
                    origin = imgui.get_window_pos()
                    self._draw_pause_at_end(ctx, layout.row(origin, 0))
                    if ctx.recording_config is not None:
                        self._draw_recording_fields(ctx, layout, origin)
                    self._record_settings_layout(layout, origin, labels)

    def _record_settings_layout(self, layout, origin, labels):
        ui = self.workspace.ui
        window = imgui.internal.get_current_window()
        rows = [layout.row(origin, index).tuple() for index in range(len(labels))]
        ui.hits["timeline-settings-popup"] = (origin.x, origin.y, window.size.x, window.size.y)
        if ui.design is not None:
            ui.design.record(
                "timeline-settings-popup",
                ("form_popup",),
                ui.hits["timeline-settings-popup"],
                "form_popup",
                padding=list(layout.padding),
                radius=window.window_rounding,
                innerRadius=layout.inner_radius,
                rowGap=layout.row_gap,
                groupGap=layout.group_gap,
                rows=rows,
                labels=labels,
                labelWidth=layout.label_width,
                fields=[layout.field(origin, index).tuple() for index in range(1, len(labels))],
            )

    def _button(self, ctx, name, x, y, *, width=24, icon=None, label="", **kwargs):
        ui = self.workspace.ui
        measured = control_size(ui, "timeline-" + name, width, 28, icon=icon, label=label)
        width_limit = kwargs.pop("width_limit", None)
        frame = control_frame_size(
            ui,
            "timeline-" + name,
            min(measured[0], width_limit) if width_limit is not None else measured[0],
            measured[1],
            icon=icon,
            label=label,
        )
        return self.workspace.ui.button(
            "timeline-" + name,
            x,
            y,
            *frame,
            icon=icon,
            label=label,
            align="center",
            width_limit=frame[0] * ui.s,
            **kwargs,
        )

    def _draw_transport(
        self,
        editor,
        ctx,
        take_times,
        x,
        y,
        steps,
        *,
        include_play=True,
        compact_play=False,
        names=None,
        width_limit=None,
    ):
        playing = ctx.session.state_take_playing
        first, last = ctx.session.state_take_range or (0, max(0, len(take_times) - 1))
        cursor = ctx.session.state_take_cursor
        enabled = bool(take_times) and not (
            ctx.session.state_take_recording or ctx.take_video_active
        )
        actions = (
            ("first", "First frame", cmd.SeekStateTake(first)),
            ("previous", "Previous frame", cmd.SeekStateTake(min(last, max(first, cursor - 1)))),
            (
                "play-pause",
                "Pause" if playing else "Play",
                cmd.PauseStateTake() if playing else cmd.PlayStateTake(),
            ),
            ("next", "Next frame", cmd.SeekStateTake(min(last, max(first, cursor + 1)))),
            ("last", "Last frame", cmd.SeekStateTake(last)),
        )
        names = (
            set(names)
            if names is not None
            else (
                {"first", "previous", "play-pause", "next", "last"}
                if steps == 4
                else {"previous", "play-pause", "next"}
                if steps == 2
                else {"play-pause"}
            )
        )
        if not include_play:
            names.discard("play-pause")
        offset = 0
        for name, label, command in actions:
            if name not in names:
                continue
            main = name == "play-pause"
            width = 72 if main and not compact_play else 28 if main else 24
            action_enabled = enabled or (
                name == "first"
                and not ctx.session.state_take_recording
                and not ctx.take_video_active
            )
            if self._button(
                ctx,
                name,
                x + offset * ctx.style_scale,
                y,
                width=width,
                icon=("pause" if playing else "play") if main else _COMMAND_ICONS[name],
                label=label if main and not compact_play else "",
                active=main,
                solid=main,
                enabled=action_enabled,
                tooltip=ctx.tr(label),
                width_limit=width_limit,
            ):
                if name == "first" and not take_times:
                    editor.playhead, editor.last_followed_playhead = 0, None
                    if not editor.view_start <= 0 <= editor.view_end:
                        editor.view_start, editor.view_end = 0, editor.view_end - editor.view_start
                else:
                    result = ctx.submit(command)
                    editor.error = "" if result.ok else result.message
            offset += width + 2
        return offset - 2

    def _draw_time(self, editor, ctx, x, y, *, width=80):
        from mojive.ui.workspace.time_readout import draw_time_readout

        draw_time_readout(self.workspace.ui, "timeline-time", x, y, editor.playhead, width=width)
        if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
            text = ctx.tr("Timeline time") + f": {editor.playhead:.6f} s"
            times, index = ctx.session.state_take_times, ctx.session.state_take_cursor
            if 0 <= index < len(times):
                text += (
                    f"\n{ctx.tr('Recorded frame')} {index + 1}/{len(times)}: {times[index]:.6f} s"
                )
            self.workspace.ui.tooltip(text, translate=False)

    def _draw_record(self, editor, ctx, x, y, *, compact=False, width_limit=None):
        recording = ctx.session.state_take_recording
        caps = ctx.session.adapter.caps
        if self._button(
            ctx,
            "record",
            x,
            y,
            width=28 if compact else 88,
            label="" if compact else "Stop" if recording else "Record simulation state",
            icon="stop" if recording else "record",
            icon_color=(222, 123, 117, 255),
            filled=True,
            width_limit=width_limit,
            enabled=caps.simulation
            and caps.state_snapshots
            and caps.clock_control
            and not ctx.take_video_active,
            tooltip=ctx.tr("Keep frames through the playhead and overwrite later frames")
            if ctx.session.state_take_times and not recording
            else ctx.tr("Stop Recording" if recording else "Record Take"),
        ):
            editor.toggle_recording(ctx)

    def _draw_capture(self, editor, ctx, kind, x, y, width, label):
        video = kind == "video"
        stopping = video and ctx.take_video_active
        enabled = (
            ctx.start_take_video is not None
            and (
                ctx.take_video_active
                or (bool(ctx.session.state_take_times) and not ctx.session.state_take_recording)
            )
            if video
            else ctx.session.adapter.caps.state_snapshots and not ctx.take_video_active
        )
        tooltip = (
            "Stop Video"
            if video and ctx.take_video_active
            else "Export Video"
            if video
            else "Save the scene state at this time; restore it from the Snapshots track."
        )
        if stopping and label:
            label = "Stop" if label == "Video" else "Stop Video"
        if self._button(
            ctx,
            kind,
            x,
            y,
            width=width,
            icon="stop" if stopping else "video" if video else "snapshot",
            label=label,
            enabled=enabled,
            tooltip=ctx.tr(tooltip),
            width_limit=width * ctx.style_scale,
        ):
            if video:
                try:
                    ctx.stop_recording() if ctx.take_video_active else ctx.start_take_video()
                    editor.error = ""
                except (RuntimeError, ValueError) as exc:
                    editor.error = str(exc)
            else:
                editor.capture_snapshot(ctx)

    def _draw_fit(self, editor, ctx, x, y, *, width_limit=None):
        if self._button(
            ctx, "fit", x, y, icon="fit", tooltip=ctx.tr("View all"), width_limit=width_limit
        ):
            editor.view_needs_fit = True

    def _draw_follow(self, editor, ctx, x, y):
        selected = _FOLLOW_MODES.index(editor.follow_mode)
        mode = self.workspace.ui.segmented(
            "timeline-follow",
            x,
            y,
            90,
            FOLLOW_MODE_TOOLTIPS,
            selected,
            icons=_NATIVE_FOLLOW_ICONS,
            layout="content",
            show_labels=False,
        )
        if mode != selected:
            editor.set_follow_mode(_FOLLOW_MODES[mode])

    def _draw_take_selector(self, editor, ctx, x, y, width):
        ui, menus, scale = self.workspace.ui, self.workspace.menus, ctx.style_scale
        from mojive.ui.workspace.popup_style import menu_choice_width

        labels = (
            *[self.take_name(ctx, take) for take in ctx.session.state_takes],
            ctx.tr("New Take"),
            ctx.tr("Clear take"),
        )
        popup_width = menu_choice_width(ui, labels, -1)
        opened = ui.begin_combo(
            "timeline-take",
            x,
            y,
            width * scale,
            self.take_label(editor, ctx),
            enabled=not (ctx.session.state_take_recording or ctx.take_video_active),
            popup_width=popup_width,
        )
        if opened:
            with menus.local_content("timeline-take-choices", choices=True):
                ui.text(
                    imgui.get_cursor_screen_pos().x,
                    imgui.get_cursor_screen_pos().y,
                    self.take_status(ctx, ctx.session.state_take_times),
                )
                imgui.dummy((0, imgui.get_text_line_height_with_spacing()))
                for take in ctx.session.state_takes:
                    selected, removed = self.draw_take_choice(editor, ctx, take)
                    if removed or selected:
                        result = ctx.submit(
                            cmd.RemoveStateTake(take.take_id)
                            if removed
                            else cmd.SelectStateTake(take.take_id)
                        )
                        editor.error = "" if result.ok else result.message
                menus.local_separator()
                if menus.local_item("New Take", key="timeline-new-take", icon="plus")[0]:
                    result = ctx.submit(cmd.CreateStateTake())
                    editor.error = "" if result.ok else result.message
                if (
                    ctx.session.state_take_times
                    and menus.local_item("Clear take", key="timeline-clear-take", icon="delete")[0]
                ):
                    result = ctx.submit(cmd.ClearStateTake())
                    editor.error = "" if result.ok else result.message
            imgui.end_combo()

    def draw_take_choice(self, editor, ctx, take):
        ui, menus = self.workspace.ui, self.workspace.menus
        origin = imgui.get_cursor_screen_pos()
        width = max(0, menus._content_right - origin.x)
        remove_key = f"timeline-remove-take-{take.take_id}"
        remove_size = control_size(ui, remove_key, 26, 26, icon="close", icon_role="inline")
        item_style = ui.component_style("select_option")
        padding, margin = (
            item_style.edges("padding", ui.s, (12, 0, 12, 0)),
            item_style.edges("margin", ui.s),
        )
        gap = ui.metric("select_option", "gap", 12) * ui.s
        remove_x = origin.x + max(0, width - remove_size[0] - padding[2] - margin[2])
        frame = control_frame_size(ui, remove_key, *remove_size, icon="close")
        removed = ui.button(
            remove_key,
            remove_x,
            origin.y + margin[1] + padding[1],
            *frame,
            icon="close",
            icon_role="inline",
            tooltip="Delete take",
            width_limit=frame[0] * ui.s,
        )
        imgui.set_cursor_screen_pos(origin)
        selected, _ = menus.local_item(
            self.take_name(ctx, take),
            checked=take.take_id == ctx.session.active_state_take_id,
            checkable=True,
            key=f"timeline-take-choice-{take.take_id}",
            width=max(0, width - remove_size[0] - gap),
            minimum_height=remove_size[1] + padding[1] + padding[3],
            icon="record",
        )
        return selected, removed

    def _numeric_size(self, format):
        ui, scale = self.workspace.ui, self.workspace.s
        style = ui.component_style("input_number")
        margin = style.edges("margin")
        width = max(82, ui.numeric_minimum(axis=False, format=format) / scale)
        if style.get("widthMode", "auto") == "fixed":
            width = style.get("width", width)
        else:
            width = math.ceil(width / 2) * 2
        height = ui.design.primitives.field_height("input_number") if ui.design else 28
        return width + margin[0] + margin[2], height + margin[1] + margin[3]

    @staticmethod
    def _range_labels(ctx, take_times):
        titles = ("Loop start", "Loop end") if len(take_times) > 1 else ("View start", "View end")
        return tuple(ctx.tr(title) for title in titles)

    @staticmethod
    def _range_endpoints(editor, ctx, take_times):
        if len(take_times) < 2:
            return editor.view_start, editor.view_end
        limits = ctx.session.state_take_range
        return (
            (take_times[limits[0]], take_times[limits[1]])
            if limits
            else (take_times[0], take_times[-1])
        )

    def _numeric_field(
        self, rows, key, label, value, *, format, speed=0.01, label_width=None, available=None
    ):
        from mojive.ui.workspace.labeled_input import draw_labeled_input, measure_labeled_input

        ui, scale = self.workspace.ui, self.workspace.s
        number_width, number_height = self._numeric_size(format)
        layout = measure_labeled_input(
            ui,
            rows.right - rows.left if available is None else available,
            label,
            (number_width * scale, number_height * scale),
            label_width=None if label_width is None else label_width * scale,
        )
        x, y, _ = rows.place(layout.width / scale, layout.height / scale)
        return draw_labeled_input(ui, key, x, y, layout, label, value, format=format, speed=speed)

    def _draw_range(self, editor, ctx, take_times, rows, *, field_width=None):
        labels = self._range_labels(ctx, take_times)
        label_width = max(self.workspace.ui.measure(title) for title in labels) / ctx.style_scale
        limits = ctx.session.state_take_range
        endpoints = self._range_endpoints(editor, ctx, take_times)
        range_format = timeline_range_format(*endpoints, suffix=" s")
        changed, values, canceled = False, [], False
        range_key = None
        imgui.begin_disabled(ctx.session.state_take_recording or ctx.take_video_active)
        for name, title, value in zip(("start", "end"), labels, endpoints, strict=True):
            key = "##timeline-range-" + name
            edited, value = self._numeric_field(
                rows,
                key,
                title,
                value,
                format=range_format,
                label_width=label_width,
                available=field_width,
            )
            if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
                hint = ctx.tr("Playback range" if len(take_times) > 1 else "Visible time range")
                self.workspace.ui.tooltip(
                    f"{hint}\n{endpoints[0]:.6f}–{endpoints[1]:.6f} s", translate=False
                )
            state = self.workspace.ui.numeric_edits[key]
            if state.activated:
                self._range_edit = _RangeEdit(
                    key,
                    ctx.session.active_state_take_id,
                    limits,
                    ctx.session.state_take_loop_enabled,
                    (editor.view_start, editor.view_end, editor.view_needs_fit),
                    editor.manual_view_range,
                )
            if self._range_edit is not None and self._range_edit.key == key:
                range_key = key
                canceled |= state.canceled
            changed |= edited
            values.append(value)
        x, y, allocated = rows.place(28)
        if self._button(
            ctx,
            "loop",
            x,
            y,
            width=28,
            icon="loop",
            tooltip=ctx.tr("Loop"),
            active=ctx.session.state_take_loop_enabled,
            width_limit=allocated * ctx.style_scale,
            enabled=len(take_times) > 1,
        ):
            ctx.submit(cmd.SetStateTakeLoopEnabled(not ctx.session.state_take_loop_enabled))
        start, end = values
        if canceled:
            original = self._range_edit
            if original.take_id == ctx.session.active_state_take_id:
                if len(take_times) > 1:
                    result = ctx.submit(cmd.SetStateTakeRange(*(original.selected or (None, None))))
                    if result.ok:
                        result = ctx.submit(cmd.SetStateTakeLoopEnabled(original.loop_enabled))
                    editor.error = "" if result.ok else result.message
                editor.view_start, editor.view_end, editor.view_needs_fit = original.view
                editor.manual_view_range = original.manual_view_range
            imgui.internal.clear_active_id()
            self._range_edit = None
        elif changed and valid_timeline_range(start, end):
            if len(take_times) > 1:
                first, last = (
                    nearest_take_frame(take_times, start),
                    nearest_take_frame(take_times, end),
                )
                if first < last:
                    ctx.submit(cmd.SetStateTakeLoop(first, last))
            else:
                editor.set_view_range(start, end)
        if range_key is not None and self.workspace.ui.numeric_edits[range_key].deactivated:
            self._range_edit = None
        ui = self.workspace.ui
        if ui.design is not None:
            ui.design.record(
                "timeline-range-context",
                ("range_bounds",),
                ui.hits["timeline-group-range"],
                "timeline_range",
                rangeKind="loop" if len(take_times) > 1 else "view",
                timeRange=list(endpoints),
                playhead=editor.playhead,
                viewRange=[editor.view_start, editor.view_end],
                loopEnabled=ctx.session.state_take_loop_enabled,
            )
        imgui.end_disabled()

    def _draw_pause_at_end(self, ctx, row):
        ui = self.workspace.ui
        on = ctx.session.state_take_pause_at_end
        label = ctx.tr("Pause at last frame")
        imgui.begin_disabled(ctx.take_video_active)
        if ui.design is not None:
            painter = ui.design.primitives
            painter.family = ui.design.family()
            box = Box(row, row, row, (0, 0, 0, 0), (0, 0, 0, 0))
            value = painter.switch(
                "timeline-pause-at-end",
                row,
                on,
                label=label,
                label_style=ui.component_style("form_label"),
                layout_box=box,
                control_x=row.right
                - ui.metric("switch", "switchWidth", DESIGN.switch_track_width) * ui.s,
            ).value
            ui.hits["##timeline-pause-label"] = (row.x, row.y, row.width / 2, row.height)
        else:
            ui.component_text("form_label", row, label, MUTED, align="start", translate=False)
            imgui.set_cursor_screen_pos((row.x, row.y))
            clicked = imgui.invisible_button("##timeline-pause-label", (row.width / 2, row.height))
            value = ui.switch(
                "timeline-pause-at-end", row.right - 48 * ui.s, row.y, not on if clicked else on
            )
        imgui.end_disabled()
        if value != on:
            if ctx.set_take_pause_at_end is not None:
                ctx.apply_setting(ctx.set_take_pause_at_end, value)
            else:
                ctx.submit(cmd.SetStateTakePauseAtEnd(value))

    def _draw_recording_fields(self, ctx, layout, origin):
        ui = self.workspace.ui
        config = ctx.recording_config
        labels = tuple(ctx.tr(title) for title in ("Start delay (s)", "End hold (s)"))
        for index, (field, label) in enumerate(
            zip(("countdown", "end_hold"), labels, strict=True), 1
        ):
            key = "##timeline-" + field
            row, target = layout.row(origin, index), layout.field(origin, index)
            ui.component_text(
                "form_label",
                Rect(row.x, row.y, layout.label_width, row.height),
                label,
                MUTED,
                align="start",
                translate=False,
            )
            if ui.design is not None:
                result = ui.design.primitives.input_number(
                    key,
                    target,
                    getattr(config, field),
                    speed=0.01,
                    format="%.1f",
                    minimum_height=target.height / ui.s,
                )
                changed, value = result.changed, result.value
            else:
                changed, value = ui.input(
                    key,
                    target.x,
                    target.y,
                    target.width,
                    getattr(config, field),
                    speed=0.01,
                    format="%.1f",
                )
            state = self.workspace.ui.numeric_edits[key]
            if state.activated:
                self._recording_edit = key, getattr(config, field)
            if state.canceled and self._recording_edit is not None:
                edit_key, original = self._recording_edit
                if edit_key == key:
                    config = replace(config, **{field: original})
                    ctx.apply_setting(ctx.set_recording_config, config, persist=False)
                    imgui.internal.clear_active_id()
                    self._recording_edit = None
                    continue
            if changed:
                config = replace(config, **{field: value})
                ctx.apply_setting(ctx.set_recording_config, config, persist=False)
            if state.committed:
                ctx.apply_setting(ctx.set_recording_config, config)
            if state.deactivated:
                self._recording_edit = None


class WorkspaceTimeline:
    """Adapt presentation without replacing the registered editor or its state."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.panel = workspace.app.panels.get("keyframes")
        self.panel.toolbar = NativeTimelineToolbar(workspace)
        self.track = NativeTimelineTrack(workspace)
        self.details = NativeTimelineDetails(workspace)
        self.last_layout = None

    @property
    def editor(self):
        return self.panel.editor

    def draw(self, ctx, x, y, w, h):
        del w
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_font(self.workspace.ui.body, BODY_SIZE * self.workspace.s)
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing, (6 * self.workspace.s, 6 * self.workspace.s)
        )
        try:
            if ctx.session.replay_info is not None:
                self._draw_replay(ctx, y, h)
                return
            editor = self.editor
            models = tuple(ctx.session.scene_models)
            model_ids = tuple(model.model_id for model in models)
            selected = ctx.session.selected_node
            if editor.model_id not in model_ids:
                preferred = selected.model_id if selected is not None else -1
                editor.set_model(
                    preferred if preferred in model_ids else model_ids[0] if model_ids else -1
                )
            _, keyframe_by_id = editor.keyframes(ctx)
            take_times = ctx.session.state_take_times
            editable = bool(
                ctx.session.paused
                and not ctx.take_video_active
                and ctx.session.adapter.caps.supports("model.keyframe_edit")
            )
            editor.sync_selection(ctx, keyframe_by_id, take_times)
            imgui.begin_group()
            self.panel.toolbar.draw_compact_toolbar(editor, ctx, take_times)
            imgui.end_group()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.workspace.ui.hits["timeline-toolbar"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            take_times = ctx.session.state_take_times
            editor.sync_selection(ctx, keyframe_by_id, take_times)
            with self._body_scope(y, h):
                available = imgui.get_content_region_avail()
                layout = measure_layout(
                    editor, ctx, imgui.get_cursor_screen_pos(), available, ui=self.workspace.ui
                )
                self.last_layout = layout
                self.workspace.ui.hits["timeline-sheet"] = (*layout.lo, layout.width, layout.height)
                draw_dope_sheet(
                    editor,
                    ctx,
                    models,
                    take_times,
                    editable,
                    icon_drawer=self.panel.toolbar.command_icon_drawer,
                    layout=layout,
                    presentation=self.track,
                    selection_menu=self.track.draw_selection_menu,
                )
                detail_y = layout.lo[1] + max(
                    layout.height + DESIGN.timeline_sheet_gap * self.workspace.s,
                    available.y - self.details.height(editor, ctx, available.x),
                )
                imgui.set_cursor_screen_pos((layout.lo[0], detail_y))
                self.details.draw(editor, ctx, editable)
        finally:
            imgui.pop_style_var()
            imgui.pop_font()

    def _begin_body(self, y, h):
        ui, scale = self.workspace.ui, self.workspace.s
        geometry = body_geometry(ui)
        pos = imgui.get_cursor_screen_pos()
        height = native_child_extent(pos.y, y + h - pos.y)
        imgui.push_style_var(imgui.StyleVar_.window_padding, geometry.padding)
        imgui.push_style_var(imgui.StyleVar_.child_rounding, geometry.radius)
        try:
            imgui.begin_child(
                "Timeline body",
                (0, max(scale, height)),
                imgui.ChildFlags_.always_use_window_padding,
                imgui.WindowFlags_.always_vertical_scrollbar,
            )
        finally:
            imgui.pop_style_var(2)
        window = imgui.internal.get_current_window()
        origin, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
        frame = Rect(*window.pos, *window.size)
        content = Rect(*origin, *available)
        native_padding = (window.window_padding.x, window.window_padding.y) * 2
        ui.hits["timeline-body"] = frame.tuple()
        ui.hits["timeline-body-content"] = content.tuple()
        ui.record_component_box(
            "timeline_body",
            Box(frame, frame, content, native_padding, (0, 0, 0, 0)),
            key="timeline-body",
            radius=window.window_rounding,
            innerRadius=geometry.inner_radius,
            childRadii={"selector": geometry.selector_radius, "lane": geometry.lane_radius},
            requestedPadding=list(geometry.requested_padding),
            scrollbarGutter=list(window.scrollbar_sizes),
            scroll=list(window.scroll),
        )
        return geometry

    @contextmanager
    def _body_scope(self, y, h):
        geometry = self._begin_body(y, h)
        try:
            with body_corner_scope(self.workspace.ui, geometry):
                yield
        finally:
            imgui.end_child()

    def _draw_replay(self, ctx, y, h):
        self.panel.toolbar._collapse(ctx)
        with self._body_scope(y, h):
            self.panel._replay.draw(
                ctx,
                combo=self.panel.toolbar.settings_fields.combo,
                tooltip=lambda text: self.workspace.ui.tooltip(text, translate=False),
            )
