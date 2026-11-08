"""Native presentation of the registered editor's real timeline and transport."""

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import Box, Rect, part_scope
from mojive.ui.compound_fields import end_content_child
from mojive.ui.icons import draw_icon_label
from mojive.ui.keyframe_editor.controls import FOLLOW_MODE_TOOLTIPS
from mojive.ui.keyframe_editor.toolbar import TimelineToolbar
from mojive.ui.keyframe_editor.track import draw_dope_sheet
from mojive.ui.timeline import timeline_range_format, valid_timeline_range
from mojive.ui.workspace.component_layout import (
    control_frame_size,
    control_size,
)
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.form_popup import form_popup, measure_form_popup
from mojive.ui.workspace.property_style import NativePropertyPresentation, toggle_row_height
from mojive.ui.workspace.timeline_body import body_corner_scope, body_geometry
from mojive.ui.workspace.timeline_details import NativeTimelineDetails
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
        """Keep complete semantic groups visible, wrapping only when necessary."""
        from mojive.ui.workspace.labeled_input import measure_labeled_input
        from mojive.ui.workspace.timeline_header import toolbar_group_layout

        ui, scale = self.workspace.ui, ctx.style_scale
        origin = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail().x
        style = ui.component_style("timeline_toolbar")
        padding, margin = style.edges("padding", scale), style.edges("margin", scale)
        content_width = max(scale, available - padding[0] - padding[2] - margin[0] - margin[2])
        range_budget = content_width
        with part_scope("range_bounds"):
            number_width, number_height = self._numeric_size("%.3f s")
        range_sizes = tuple(
            measure_labeled_input(
                ui, range_budget, label, (number_width * scale, number_height * scale)
            )
            for label in self._range_labels(ctx, take_times)
        )
        control = (28 * scale, 28 * scale)
        number_height = (
            ui.design.primitives.field_height("input_number") * scale if ui.design else 28 * scale
        )
        groups = (
            ("transport", (control,) * 6 + ((90 * scale, number_height),)),
            ("capture", (control,) * 3),
            ("keys", (control,) * 3),
            ("view", ((68 * scale, number_height), (76 * scale, number_height), control)),
            ("follow", ((84 * scale, 28 * scale),)),
            ("range", tuple((item.width, item.height) for item in range_sizes)),
            ("utilities", (control,) * 2),
        )
        box, layout, bounds = toolbar_group_layout(
            ui,
            origin.x,
            origin.y,
            available,
            tuple(sizes for _, sizes in groups),
            column_gap=4,
            row_gap=4,
        )
        for (name, _), rect in zip(groups, bounds, strict=True):
            ui.hits["timeline-group-" + name] = rect.tuple()
        rows = _ToolbarRows(origin, box.content.width, scale, placements=layout.rectangles)
        for name in ("first", "previous", "play-pause", "next", "last"):
            x, y, width = rows.place(28)
            self._draw_transport(
                editor,
                ctx,
                take_times,
                x,
                y,
                4,
                names=(name,),
                compact_play=True,
                width_limit=width * scale,
            )
        x, y, width = rows.place(28)
        self._draw_loop(ctx, take_times, x, y, width)
        x, y, width = rows.place(90)
        self._draw_time(editor, ctx, x, y, width=width)
        x, y, width = rows.place(28)
        self._draw_record(editor, ctx, x, y, compact=True, width_limit=width * scale)
        for name in ("video", "snapshot"):
            x, y, width = rows.place(28)
            self._draw_capture(editor, ctx, name, x, y, width, "")
        for name in ("add", "previous", "next"):
            x, y, width = rows.place(28)
            self._draw_key_action(editor, ctx, name, x, y, width)
        x, y, width = rows.place(68)
        self._draw_speed(ctx, x, y, width)
        x, y, width = rows.place(76)
        self._draw_zoom(editor, ctx, take_times, x, y, width)
        x, y, width = rows.place(28)
        self._draw_fit(editor, ctx, x, y, width_limit=width * scale)
        x, y, _ = rows.place(84)
        self._draw_follow(editor, ctx, x, y)
        self._draw_range(editor, ctx, take_times, rows, field_width=range_budget)
        x, y, width = rows.place(28)
        imgui.set_cursor_screen_pos((x, y))
        self._draw_settings(ctx, width_limit=width * scale)
        x, y, width = rows.place(28)
        imgui.set_cursor_screen_pos((x, y))
        self._collapse(ctx, width_limit=width * scale)
        ui.hits["timeline-tools"] = box.content.tuple()
        imgui.set_cursor_screen_pos((origin.x, math.ceil(box.outer.bottom)))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((box.outer.width, 0))
        imgui.pop_style_var()

    def _draw_speed(self, ctx, x, y, width):
        ui = self.workspace.ui
        imgui.begin_disabled(
            ctx.take_video_active
            or ctx.session.state_take_recording
            or len(ctx.session.state_take_times) < 2
        )
        changed, value = ui.input(
            "##timeline-speed",
            x,
            y,
            width * ctx.style_scale,
            ctx.session.state_take_speed,
            speed=0.05,
            lo=0.05,
            hi=4,
            format="%.2f ×",
            reset_value=1.0,
        )
        imgui.end_disabled()
        if changed:
            ctx.submit(cmd.SetStateTakeSpeed(value))
        if imgui.is_item_hovered():
            ui.tooltip("Recorded playback speed")

    def _draw_settings(self, ctx, *, width_limit=None):
        ui = self.workspace.ui
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
            labels, layout = self.settings_layout(ctx)
            with form_popup(ui, "Timeline settings", layout) as opened:
                if opened:
                    origin = imgui.get_window_pos()
                    self._draw_pause_at_end(ctx, layout.row(origin, 0))
                    if ctx.recording_config is not None:
                        self._draw_recording_fields(ctx, layout, origin)
                    self._record_settings_layout(layout, origin, labels)

    def settings_layout(self, ctx):
        """Share the production form's label column and field budgets with its specimen."""
        ui, scale = self.workspace.ui, ctx.style_scale
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
            label_inset=ui.component_style("section_header").edges("padding", scale, (8, 0, 8, 0))[
                0
            ],
        )
        return labels, layout

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
            width = 72 if main and not compact_play else 28
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
                icon_color=self.workspace.ui.palette("primary") if main else None,
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
            icon_color=self.workspace.ui.palette("danger"),
            filled=recording,
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

    def _draw_loop(self, ctx, take_times, x, y, width):
        if self._button(
            ctx,
            "loop",
            x,
            y,
            width=28,
            icon="loop",
            tooltip=ctx.tr("Loop"),
            active=ctx.session.state_take_loop_enabled,
            width_limit=width * ctx.style_scale,
            enabled=len(take_times) > 1 and not ctx.session.state_take_recording,
        ):
            ctx.submit(cmd.SetStateTakeLoopEnabled(not ctx.session.state_take_loop_enabled))

    def _draw_key_action(self, editor, ctx, action, x, y, width):
        from mojive.ui.keyframe_editor.toolbar import add_model_keyframe

        keys, _ = editor.keyframes(ctx)
        candidates = [
            key
            for key in keys
            if (
                key.time < editor.playhead - 1e-9
                if action == "previous"
                else key.time > editor.playhead + 1e-9
            )
        ]
        enabled = (
            (ctx.session.paused and ctx.session.adapter.caps.supports("model.keyframe_edit"))
            if action == "add"
            else bool(candidates)
        )
        if self._button(
            ctx,
            "key-" + action,
            x,
            y,
            width=28,
            icon="diamond" if action == "add" else action,
            tooltip=ctx.tr(
                {"add": "Add Keyframe", "previous": "Previous keyframe", "next": "Next keyframe"}[
                    action
                ]
            ),
            enabled=enabled and not ctx.take_video_active,
            width_limit=width * ctx.style_scale,
        ):
            if action == "add":
                add_model_keyframe(editor, ctx)
            else:
                key = (max if action == "previous" else min)(candidates, key=lambda key: key.time)
                editor.load_keyframe(ctx, key)

    def _draw_zoom(self, editor, ctx, times, x, y, width):
        ui = self.workspace.ui
        base = max(0.5, times[-1] - times[0]) if len(times) > 1 else 0.5
        span = max(1e-9, editor.view_end - editor.view_start)
        changed, value = ui.input(
            "##timeline-zoom",
            x,
            y,
            width * ui.s,
            100 * base / span,
            speed=1,
            lo=1,
            hi=10000,
            format="%.0f %%",
        )
        state = ui.numeric_edits["##timeline-zoom"]
        if state.activated:
            self._zoom_edit = (
                editor.view_start,
                editor.view_end,
                editor.view_needs_fit,
                editor.manual_view_range,
            )
        if state.canceled and getattr(self, "_zoom_edit", None) is not None:
            editor.view_start, editor.view_end, editor.view_needs_fit, editor.manual_view_range = (
                self._zoom_edit
            )
        elif changed:
            center = (editor.view_start + editor.view_end) / 2
            half = base * 50 / value
            editor.set_view_range(center - half, center + half)
        if state.deactivated:
            self._zoom_edit = None

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
            84,
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

        takes = ctx.session.state_takes
        labels = tuple(self.take_name(ctx, take) for take in takes)
        selected = next(
            (i for i, take in enumerate(takes) if take.take_id == ctx.session.active_state_take_id),
            -1,
        )
        popup_width = menu_choice_width(ui, labels, selected) + 28 * scale
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

            def remove_take(take_id):
                result = ctx.submit(cmd.RemoveStateTake(take_id))
                editor.error = "" if result.ok else result.message
                if not ctx.session.state_takes:
                    imgui.close_current_popup()

            with menus.local_content("timeline-take-choices", choices=True):
                for take, label in zip(tuple(takes), labels, strict=True):
                    if menus.local_item(
                        label,
                        checked=take.take_id == ctx.session.active_state_take_id,
                        checkable=True,
                        key=f"timeline-take-choice-{take.take_id}",
                        on_remove=lambda take_id=take.take_id: remove_take(take_id),
                    )[0]:
                        result = ctx.submit(cmd.SelectStateTake(take.take_id))
                        editor.error = "" if result.ok else result.message
            imgui.end_combo()

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
        titles = ("View", "–")
        return tuple(ctx.tr(title) for title in titles)

    @staticmethod
    def _range_endpoints(editor, ctx, take_times):
        return editor.view_start, editor.view_end

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
                available=field_width,
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
        start, end = values
        if canceled:
            original = self._range_edit
            if original.take_id == ctx.session.active_state_take_id:
                editor.view_start, editor.view_end, editor.view_needs_fit = original.view
                editor.manual_view_range = original.manual_view_range
            imgui.internal.clear_active_id()
            self._range_edit = None
        elif changed and valid_timeline_range(start, end):
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
                rangeKind="view",
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
            from .property_style import popup_row_box

            painter = ui.design.primitives
            painter.family = ui.design.family()
            box = popup_row_box(ui, row)
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
            ui.hits["##timeline-pause-label"] = (
                box.content.x,
                box.content.y,
                box.content.width / 2,
                box.content.height,
            )
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
            row, target = layout.content_row(origin, index), layout.field(origin, index)
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
        self.track = NativeTimelineTrack(workspace, toolbar=self.panel.toolbar)
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
            imgui.set_cursor_screen_pos((lo.x, hi.y + 4 * self.workspace.s))
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
                imgui.WindowFlags_.none,
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
            end_content_child()

    def _draw_replay(self, ctx, y, h):
        self.panel.toolbar._collapse(ctx)
        with self._body_scope(y, h):
            self.panel._replay.draw(
                ctx,
                combo=self.panel.toolbar.settings_fields.combo,
                tooltip=lambda text: self.workspace.ui.tooltip(text, translate=False),
            )
