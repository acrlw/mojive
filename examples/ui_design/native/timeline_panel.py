"""Native presentation of the registered editor's real timeline and transport."""

import math
from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.icons import draw_icon_label
from mojive.ui.keyframe_editor.controls import FOLLOW_MODE_TOOLTIPS, _draw_command_icon
from mojive.ui.keyframe_editor.toolbar import TimelineToolbar
from mojive.ui.keyframe_editor.track import draw_dope_sheet
from mojive.ui.timeline import nearest_take_frame

from .timeline_details import NativeTimelineDetails
from .timeline_track import NativeTimelineTrack, measure_layout
from .widgets import BODY_SIZE, SWITCH_WIDTH

_COMMAND_ICONS = {
    "first": "previous",
    "last": "next",
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
    "key": "link",
    "key-keyframe": "diamond",
}
_FOLLOW_ICONS = {
    "key-follow-off": "pin",
    "key-follow-page": "follow",
    "key-follow-locked": "lock",
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


class _ToolbarRows:
    """Keep measured control groups together while allowing narrow rows to wrap."""

    def __init__(self, origin, width, scale):
        self.left, self.top = origin.x, origin.y
        self.right = origin.x + width
        self.scale = scale
        self.x, self.y, self.row_height = self.left, self.top, 0.0

    def _next_row(self):
        self.x = self.left
        self.y += self.row_height + 8 * self.scale
        self.row_height = 0.0

    def group(self, width):
        if self.x == self.left:
            return
        if self.x + 12 * self.scale + width > self.right:
            self._next_row()
        else:
            self.x += 6 * self.scale

    def place(self, width, height=28):
        width = min(width * self.scale, self.right - self.left)
        if self.x > self.left and self.x + width > self.right + 0.5:
            self._next_row()
        position = self.x, self.y
        self.x += width + 6 * self.scale
        self.row_height = max(self.row_height, height * self.scale)
        return *position, width / self.scale

    @property
    def bottom(self):
        return self.y + self.row_height


class NativeTimelineToolbar(TimelineToolbar):
    """Keep shared commands and responsive groups with the native icon family."""

    def __init__(self, study):
        self.study = study
        super().__init__(self.draw_follow_icon, command_icon_drawer=self.draw_command_icon)

    def _icon(self, name, center, color, scale):
        size = 28 * scale
        self.study.ui.role_icon(
            name,
            center[0] - size / 2,
            center[1] - size / 2,
            size,
            size,
            role="toolbar",
            c=tuple(round(value * 255) for value in color),
        )

    def draw_command_icon(self, draw, center, kind, color, scale, *, smoothing=0.6):
        if kind in ("first", "last"):
            _draw_command_icon(draw, center, kind, color, scale, smoothing=smoothing)
        else:
            self._icon(_COMMAND_ICONS[kind], center, color, scale)
        item_id = imgui.get_item_id()
        for label, hit in _BUTTON_HITS.items():
            if item_id == imgui.get_id(label):
                lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                self.study.ui.hits[hit] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
                break

    def draw_follow_icon(self, draw, lo, hi, color, scale, name, label):
        if label:
            draw_icon_label(draw, lo, hi, color, scale, name, label)
        else:
            self._icon(
                _FOLLOW_ICONS[name], ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), color, scale
            )

    def _collapse(self, ctx):
        pos = imgui.get_cursor_screen_pos()
        if self.study.ui.button(
            "collapse-timeline",
            pos.x,
            pos.y,
            28,
            28,
            icon="down",
            tooltip=ctx.tr("Collapse Timeline"),
        ):
            self.study.collapse_bottom()

    def draw_compact_toolbar(self, editor, ctx, take_times):
        scale = ctx.style_scale
        origin = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x / scale
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (8 * scale, max(0, (28 * scale - imgui.get_font_size()) / 2)),
        )
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (5 * scale, 6 * scale))
        try:
            steps = 4 if width >= 330 else 2 if width >= 278 else 0
            self._visible_transport_steps = steps
            transport_width = self._draw_transport(
                editor, ctx, take_times, origin.x, origin.y, steps, compact_play=width < 160
            )
            time_width = min(80, max(1, width - transport_width - 74))
            prefix = transport_width + time_width + 12
            label_width = max(
                ui_width
                for ui_width in (
                    self.study.ui.measure("Start") / scale,
                    self.study.ui.measure("End") / scale,
                )
            )
            icons_width = 88 + 120 + 56 + 2 * (label_width + 6 + 82) + 28 + 90 + 28 + 66
            labels_width = (
                icons_width
                - 56
                + max(100, self.study.ui.measure("Snapshot") / scale + 37)
                + max(76, self.study.ui.measure("Video") / scale + 37)
            )
            self._inline = width >= prefix + icons_width + 68
            self._inline_capture_labels = width >= prefix + labels_width + 68
            self._draw_time(
                editor,
                ctx,
                origin.x
                + (transport_width + 6 if self._inline else width - time_width - 68) * scale,
                origin.y,
                width=time_width,
            )
            if self._inline:
                imgui.set_cursor_screen_pos((origin.x + prefix * scale, origin.y))
                self.draw_editor_controls(
                    editor, ctx, take_times, inline=True, width=width - prefix - 68
                )
            imgui.set_cursor_screen_pos((origin.x + (width - 62) * scale, origin.y))
            self._draw_settings(ctx)
            imgui.set_cursor_screen_pos((origin.x + (width - 28) * scale, origin.y))
            self._collapse(ctx)
            imgui.set_cursor_screen_pos((origin.x, origin.y + 28 * scale))
            imgui.dummy((0, 0))
        finally:
            imgui.pop_style_var(2)

    def draw_editor_controls(self, editor, ctx, take_times, *, inline=False, width=None):
        """Keep common actions together; move only infrequent configuration to Settings."""
        if self._inline and not inline:
            return
        ui, scale = self.study.ui, ctx.style_scale
        origin = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x / scale if width is None else width
        rows = _ToolbarRows(origin, width * scale, scale)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (8 * scale, max(0, (28 * scale - imgui.get_font_size()) / 2)),
        )
        imgui.begin_group()
        try:
            steps = self._visible_transport_steps
            missing = ("first", "last") if steps == 2 else ("first", "previous", "next", "last")
            if steps < 4:
                x, y, _ = rows.place(len(missing) * 26 - 2)
                self._draw_transport(
                    editor, ctx, take_times, x, y, 4, include_play=False, names=missing
                )
            capture_labels = self._inline_capture_labels if inline else width >= 600
            record_width = 88 if width >= 180 else 28
            take_width = min(120, max(84, width - record_width - 12))
            capture_widths = (
                (
                    max(100, ui.measure("Snapshot") / scale + 37),
                    max(76, ui.measure("Stop" if ctx.take_video_active else "Video") / scale + 37),
                )
                if capture_labels
                else (28, 28)
            )
            rows.group((record_width + take_width + sum(capture_widths) + 18) * scale)
            x, y, _ = rows.place(record_width)
            self._draw_record(editor, ctx, x, y, compact=record_width == 28)
            x, y, w = rows.place(take_width)
            self._draw_take_selector(editor, ctx, x, y, w)
            for kind, label, capture_width in zip(
                ("snapshot", "video"), ("Snapshot", "Video"), capture_widths, strict=True
            ):
                x, y, w = rows.place(capture_width)
                self._draw_capture(editor, ctx, kind, x, y, w, label if capture_labels else "")
            self._draw_range(editor, ctx, take_times, rows)
            rows.group((90 + 28 + 12) * scale)
            x, y, _ = rows.place(90)
            self._draw_follow(editor, ctx, x, y)
            x, y, _ = rows.place(28)
            self._draw_fit(editor, ctx, x, y)
            imgui.set_cursor_screen_pos((origin.x, rows.bottom))
            imgui.dummy((0, 0))
        finally:
            imgui.end_group()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            ui.hits["timeline-tools"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            imgui.pop_style_var()
        imgui.set_cursor_screen_pos((origin.x, rows.bottom + 8 * scale))

    def _draw_settings(self, ctx):
        ui, scale = self.study.ui, ctx.style_scale
        pos = imgui.get_cursor_screen_pos()
        if ui.button(
            "timeline-settings",
            pos.x,
            pos.y,
            28,
            28,
            icon="settings",
            tooltip=ctx.tr("Timeline settings"),
        ):
            imgui.open_popup("Timeline settings")
        labels = ("Start delay (s)", "End hold (s)")
        fields_width = max(ui.measure(ctx.tr(label)) for label in labels) + (6 + 82) * scale
        toggle_width = ui.measure(ctx.tr("Pause at last frame")) + (12 + SWITCH_WIDTH) * scale
        width = min(
            max(fields_width, toggle_width) + 20 * scale,
            imgui.get_io().display_size.x - 20 * scale,
        )
        imgui.set_next_window_size((width, 0))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (10 * scale, 10 * scale))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
        try:
            if imgui.begin_popup("Timeline settings"):
                origin = imgui.get_cursor_screen_pos()
                available = imgui.get_content_region_avail().x
                self._draw_pause_at_end(ctx, origin.x, origin.y, available / scale)
                if ctx.recording_config is not None:
                    imgui.set_cursor_screen_pos((origin.x, origin.y + 36 * scale))
                    rows = _ToolbarRows(imgui.get_cursor_screen_pos(), available, scale)
                    self._draw_recording_fields(ctx, rows)
                    imgui.set_cursor_screen_pos((origin.x, rows.bottom))
                    imgui.dummy((0, 0))
                imgui.end_popup()
        finally:
            imgui.pop_style_var(2)

    def _button(self, ctx, name, x, y, *, width=24, icon=None, label="", **kwargs):
        return self.study.ui.button(
            "timeline-" + name, x, y, width, 28, icon=icon, label=label, align="center", **kwargs
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
                icon=("pause" if playing else "play")
                if main
                else None
                if name in ("first", "last")
                else _COMMAND_ICONS[name],
                label=label if main and not compact_play else "",
                active=main,
                solid=main,
                enabled=action_enabled,
                tooltip=ctx.tr(label),
            ):
                if name == "first" and not take_times:
                    editor.playhead, editor.last_followed_playhead = 0, None
                    if not editor.view_start <= 0 <= editor.view_end:
                        editor.view_start, editor.view_end = 0, editor.view_end - editor.view_start
                else:
                    result = ctx.submit(command)
                    editor.error = "" if result.ok else result.message
            if name in ("first", "last"):
                # Boundary glyphs differ from the browser's single-chevron step masters.
                lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                _draw_command_icon(
                    ctx.painter(),
                    ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2),
                    name,
                    ctx.theme.text if action_enabled else ctx.theme.text_disabled,
                    ctx.style_scale,
                )
            offset += width + 2
        return offset - 2

    def _draw_time(self, editor, ctx, x, y, *, width=80):
        scale = ctx.style_scale
        imgui.set_cursor_screen_pos((x, y))
        imgui.dummy((width * scale, 28 * scale))
        self.study.ui.hits["timeline-time"] = (x, y, width * scale, 28 * scale)
        self.study.ui.rect(x, y, width * scale, 28 * scale, (24, 27, 30, 255))
        imgui.push_font(self.study.ui.mono, 14 * scale)
        try:
            caption = f"{editor.playhead:.0e}s"
            for digits in (3, 2, 1, 0):
                candidate = f"{editor.playhead:.{digits}f} s"
                if imgui.calc_text_size(candidate).x <= max(0, width - 16) * scale:
                    caption = candidate
                    break
        finally:
            imgui.pop_font()
        self.study.ui.text(
            x + 8 * scale,
            y + 7 * scale,
            caption,
            size=14,
            mono=True,
            max_width=max(0, width - 16) * scale,
        )
        imgui.set_item_tooltip(ctx.tr("Timeline time") + f": {editor.playhead:.6f} s")

    def _draw_record(self, editor, ctx, x, y, *, compact=False):
        recording = ctx.session.state_take_recording
        caps = ctx.session.adapter.caps
        if self._button(
            ctx,
            "record",
            x,
            y,
            width=28 if compact else 88,
            label="" if compact else "Stop" if recording else "Record",
            icon="stop" if recording else "record",
            icon_color=(222, 123, 117, 255),
            filled=True,
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
            else "Capture Snapshot"
        )
        if stopping and label:
            label = "Stop" if label == "Video" else "Stop Video"
        if self._button(
            ctx,
            kind,
            x,
            y,
            width=width,
            icon="stop" if stopping else "video" if video else "link",
            label=label,
            enabled=enabled,
            tooltip=ctx.tr(tooltip),
        ):
            if video:
                try:
                    ctx.stop_recording() if ctx.take_video_active else ctx.start_take_video()
                    editor.error = ""
                except (RuntimeError, ValueError) as exc:
                    editor.error = str(exc)
            else:
                editor.capture_snapshot(ctx)

    def _draw_fit(self, editor, ctx, x, y):
        if self._button(ctx, "fit", x, y, icon="fit", tooltip=ctx.tr("View all")):
            editor.view_needs_fit = True

    def _draw_follow(self, editor, ctx, x, y):
        selected = _FOLLOW_MODES.index(editor.follow_mode)
        mode = self.study.ui.segmented(
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
        scale = ctx.style_scale
        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(width * scale)
        imgui.begin_disabled(ctx.session.state_take_recording or ctx.take_video_active)
        label = self.take_label(editor, ctx)
        opened = imgui.begin_combo("##timeline-take", label)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.study.ui.hits["timeline-take"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        imgui.set_item_tooltip(label)
        if opened:
            imgui.text_disabled(self.take_status(ctx, ctx.session.state_take_times))
            for take in ctx.session.state_takes:
                selected, removed = self.draw_take_choice(editor, ctx, take)
                if removed or selected:
                    result = ctx.submit(
                        cmd.RemoveStateTake(take.take_id)
                        if removed
                        else cmd.SelectStateTake(take.take_id)
                    )
                    editor.error = "" if result.ok else result.message
            imgui.separator()
            if imgui.selectable(ctx.tr("New Take") + "##new-take", False)[0]:
                result = ctx.submit(cmd.CreateStateTake())
                editor.error = "" if result.ok else result.message
            if ctx.session.state_take_times and imgui.selectable(ctx.tr("Clear take"), False)[0]:
                result = ctx.submit(cmd.ClearStateTake())
                editor.error = "" if result.ok else result.message
            imgui.end_combo()
        imgui.end_disabled()

    def _numeric_field(self, rows, key, label, value, *, format, speed=0.01, label_width=None):
        ui, scale = self.study.ui, self.study.s
        label_width = ui.measure(label) / scale if label_width is None else label_width
        available = (rows.right - rows.left) / scale
        above = label_width + 6 + 82 > available
        width, height = (available, 50) if above else (label_width + 6 + 82, 28)
        x, y, _ = rows.place(width, height)
        ui.text(x, y + (0 if above else 7.5) * scale, label)
        field_x, field_y = (x, y + 22 * scale) if above else (x + (label_width + 6) * scale, y)
        return ui.input(
            key,
            field_x,
            field_y,
            (width if above else 82) * scale,
            value,
            speed=speed,
            format=format,
        )

    def _draw_range(self, editor, ctx, take_times, rows):
        labels = tuple(ctx.tr(title) for title in ("Start", "End"))
        label_width = max(self.study.ui.measure(title) for title in labels) / ctx.style_scale
        rows.group((2 * (label_width + 6 + 82) + 40) * ctx.style_scale)
        limits = ctx.session.state_take_range
        endpoints = (
            (take_times[limits[0]], take_times[limits[1]])
            if limits
            else (take_times[0], take_times[-1])
            if take_times
            else (editor.view_start, editor.view_end)
        )
        changed, values = False, []
        imgui.begin_disabled(ctx.session.state_take_recording or ctx.take_video_active)
        for name, title, value in zip(("start", "end"), labels, endpoints, strict=True):
            edited, value = self._numeric_field(
                rows,
                "##timeline-range-" + name,
                title,
                value,
                format="%.2f s",
                label_width=label_width,
            )
            changed |= edited
            values.append(value)
        x, y, _ = rows.place(28)
        if self._button(
            ctx,
            "loop",
            x,
            y,
            width=28,
            icon="loop",
            tooltip=ctx.tr("Loop"),
            active=ctx.session.state_take_loop_enabled,
            enabled=len(take_times) > 1,
        ):
            ctx.submit(cmd.SetStateTakeLoopEnabled(not ctx.session.state_take_loop_enabled))
        start, end = values
        if changed and math.isfinite(start) and math.isfinite(end) and end > start:
            if len(take_times) > 1:
                first, last = (
                    nearest_take_frame(take_times, start),
                    nearest_take_frame(take_times, end),
                )
                if first < last:
                    ctx.submit(cmd.SetStateTakeLoop(first, last))
            else:
                editor.view_start, editor.view_end, editor.view_needs_fit = start, end, False
        imgui.end_disabled()

    def _draw_pause_at_end(self, ctx, x, y, width):
        ui, scale = self.study.ui, ctx.style_scale
        on = ctx.session.state_take_pause_at_end
        label_width = max(1, width - SWITCH_WIDTH - 12) * scale
        ui.text(
            x,
            y + (28 - BODY_SIZE) * scale / 2,
            ctx.tr("Pause at last frame"),
            max_width=label_width,
        )
        imgui.begin_disabled(ctx.take_video_active)
        imgui.set_cursor_screen_pos((x, y))
        pressed = imgui.invisible_button("##timeline-pause-label", (label_width, 28 * scale))
        label_hovered = imgui.is_item_hovered()
        value = ui.switch("timeline-pause-at-end", x + (width - SWITCH_WIDTH) * scale, y, on)
        if label_hovered or imgui.is_item_hovered():
            ui.tooltip("When off, the playhead continues while the last pose holds")
        imgui.end_disabled()
        if pressed:
            value = not on
        if value != on:
            if ctx.set_take_pause_at_end is not None:
                ctx.apply_setting(ctx.set_take_pause_at_end, value)
            else:
                ctx.submit(cmd.SetStateTakePauseAtEnd(value))

    def _draw_recording_fields(self, ctx, rows):
        config = ctx.recording_config
        labels = tuple(ctx.tr(title) for title in ("Start delay (s)", "End hold (s)"))
        label_width = max(self.study.ui.measure(title) for title in labels) / ctx.style_scale
        rows.group((2 * (label_width + 6 + 82) + 6) * ctx.style_scale)
        for field, label in zip(("countdown", "end_hold"), labels, strict=True):
            key = "##timeline-" + field
            changed, value = self._numeric_field(
                rows, key, label, getattr(config, field), format="%.1f", label_width=label_width
            )
            if changed:
                config = replace(config, **{field: value})
                ctx.apply_setting(ctx.set_recording_config, config, persist=False)
            if self.study.ui.numeric_edits[key].committed:
                ctx.apply_setting(ctx.set_recording_config, config)


class ReferenceTimeline:
    """Adapt presentation without replacing the registered editor or its state."""

    def __init__(self, study):
        self.study = study
        self.panel = study.preview.panels.get("keyframes")
        self.panel.toolbar = NativeTimelineToolbar(study)
        self.track = NativeTimelineTrack(study)
        self.details = NativeTimelineDetails(study)
        self.last_layout = None

    @property
    def editor(self):
        return self.panel.editor

    def draw(self, ctx, x, y, w, h):
        del w
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_font(self.study.ui.body, BODY_SIZE * self.study.s)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (6 * self.study.s, 6 * self.study.s))
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
            self.study.ui.hits["timeline-toolbar"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            take_times = ctx.session.state_take_times
            editor.sync_selection(ctx, keyframe_by_id, take_times)
            self._begin_body(y, h)
            try:
                tools_origin = imgui.get_cursor_screen_pos()
                self.panel.toolbar.draw_editor_controls(editor, ctx, take_times)
                track_origin = imgui.get_cursor_screen_pos()
                available = imgui.ImVec2(
                    imgui.get_content_region_avail().x,
                    max(
                        1,
                        imgui.get_window_height()
                        - 2 * imgui.get_style().window_padding.y
                        - (track_origin.y - tools_origin.y),
                    ),
                )
                layout = measure_layout(editor, ctx, imgui.get_cursor_screen_pos(), available)
                self.last_layout = layout
                self.study.ui.hits["timeline-sheet"] = (*layout.lo, layout.width, layout.height)
                draw_dope_sheet(
                    editor,
                    ctx,
                    models,
                    take_times,
                    editable,
                    icon_drawer=self.panel.toolbar.command_icon_drawer,
                    layout=layout,
                    presentation=self.track,
                )
                detail_y = layout.lo[1] + max(
                    layout.height + 6 * self.study.s,
                    available.y - self.details.height(editor, ctx, available.x),
                )
                imgui.set_cursor_screen_pos((layout.lo[0], detail_y))
                self.details.draw(editor, ctx, editable)
            finally:
                imgui.end_child()
        finally:
            imgui.pop_style_var()
            imgui.pop_font()

    def _begin_body(self, y, h):
        pos = imgui.get_cursor_screen_pos()
        imgui.begin_child(
            "Timeline body",
            (0, max(self.study.s, y + h - pos.y)),
            imgui.ChildFlags_.none,
            imgui.WindowFlags_.always_vertical_scrollbar,
        )

    def _draw_replay(self, ctx, y, h):
        self.panel.toolbar._collapse(ctx)
        self._begin_body(y, h)
        try:
            self.panel._replay.draw(ctx)
        finally:
            imgui.end_child()
