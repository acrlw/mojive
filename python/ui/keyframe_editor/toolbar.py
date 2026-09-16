"""Timeline toolbar layout and transport controls."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels import PanelContext, button_row_layout, segmented_control
from mojive.ui.text_layout import fit_text

from ..controls import IconLabelDrawer, clear_button, padded_selectable, segmented_control_width
from ..icons import draw_icon_label
from ..timeline import (
    nearest_take_frame,
)
from .controller import TimelineEditor
from .controls import (
    _COMMAND_HEIGHT_PT,
    FOLLOW_MODE_TOOLTIPS,
    _command_button,
    _command_button_width,
    _toolbar_status,
    unique_keyframe_name,
)


class TimelineToolbar:
    """Toolbar layout cache and presentation; editing state lives in TimelineEditor."""

    def __init__(self, icon_drawer: IconLabelDrawer = draw_icon_label) -> None:
        self.follow_mode_icon_drawer = icon_drawer
        self.command_layouts: dict = {}

    def draw_compact_toolbar(self, editor: TimelineEditor, ctx, take_times):
        scale = ctx.style_scale
        gap = 5 * scale
        width = max(1.0, imgui.get_content_region_avail().x)
        height = _COMMAND_HEIGHT_PT * scale
        # Measure with the same padding and glyph scale used by the actual controls.
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (8 * scale, max(0, (height - imgui.get_font_size()) * 0.5)),
        )
        labels = (
            ctx.tr(
                "Stop Recording"
                if ctx.session.state_take_recording
                else "Record from Playhead"
                if take_times
                else "Record Take"
            ),
            ctx.tr("Stop Video" if ctx.take_video_active else "Export Video"),
            ctx.tr("Capture Snapshot"),
        )
        icon_width = _command_button_width("", scale)
        first, last = (
            (take_times[0], take_times[-1]) if take_times else (editor.view_start, editor.view_end)
        )
        time_reference = f"{max(99, abs(first), abs(last)):.3f} s"
        time_width = imgui.calc_text_size(("-" if first < 0 else "") + time_reference).x
        transport_width = 5 * icon_width + time_width + 5 * gap
        field_width = max(
            56 * scale, imgui.calc_text_size(f"{max(abs(first), abs(last)):.2f}").x + 16 * scale
        )
        take_width = max(
            84 * scale,
            imgui.calc_text_size(self.take_label(editor, ctx)).x + 44 * scale,
        )
        range_widths = (take_width, field_width, field_width, icon_width)
        range_width = sum(range_widths) + 3 * gap
        follow_labels = tuple(ctx.tr(text) for text in FOLLOW_MODE_TOOLTIPS)
        follow_icons = ("key-follow-off", "key-follow-page", "key-follow-locked")
        follow_width = segmented_control_width(follow_labels, icons=follow_icons, show_labels=False)
        view_width = follow_width + 2 * (icon_width + gap)
        separator_width = 13 * scale
        fixed_width = transport_width + range_width + view_width + 3 * separator_width
        command_minimum = 3 * icon_width + 2 * gap
        if fixed_width + command_minimum <= width:
            caption_budget = width - fixed_width
        elif command_minimum + transport_width + separator_width <= width:
            caption_budget = width - transport_width - separator_width
        else:
            caption_budget = width
        shown = list(labels)
        for index in (1, 2, 0):
            total = sum(_command_button_width(label, scale) for label in shown) + 2 * gap
            if total <= caption_budget:
                break
            shown[index] = ""
        widths = [_command_button_width(label, scale) for label in shown]
        right = imgui.get_cursor_screen_pos().x + width
        first_group = True

        def group(group_width):
            nonlocal first_group
            if (
                not first_group
                and imgui.get_item_rect_max().x + separator_width + group_width <= right
            ):
                imgui.same_line(0, 6 * scale)
                pos = imgui.get_cursor_screen_pos()
                imgui.dummy((scale, height))
                ImguiDraw2D().line(
                    (pos.x, pos.y + 5 * scale),
                    (pos.x, pos.y + height - 5 * scale),
                    (*ctx.theme.text_disabled[:3], 0.25),
                    scale,
                )
                imgui.same_line(0, 6 * scale)
            first_group = False
            imgui.begin_group()

        imgui.push_style_var(imgui.StyleVar_.item_spacing, (gap, 6 * scale))
        group(sum(widths) + 2 * gap)
        recording = ctx.session.state_take_recording
        supported = (
            ctx.session.adapter.caps.simulation
            and ctx.session.adapter.caps.state_snapshots
            and ctx.session.adapter.caps.clock_control
        )
        if _command_button(
            "##take-record",
            "stop" if recording else "record",
            ctx.tr("Keep frames through the playhead and overwrite later frames")
            if take_times and not recording
            else labels[0],
            ctx.theme,
            scale,
            label=shown[0],
            width=widths[0],
            layouts=self.command_layouts,
            enabled=supported and not ctx.take_video_active,
            selected=recording,
            draw=ctx.painter(),
        ):
            editor.toggle_recording(ctx)
        imgui.same_line()
        if _command_button(
            "##take-video",
            "view",
            labels[1],
            ctx.theme,
            scale,
            label=shown[1],
            draw=ctx.painter(),
            width=widths[1],
            layouts=self.command_layouts,
            enabled=ctx.start_take_video is not None
            and (ctx.take_video_active or (bool(take_times) and not recording)),
        ):
            try:
                ctx.stop_recording() if ctx.take_video_active else ctx.start_take_video()
                editor.error = ""
            except (RuntimeError, ValueError) as exc:
                editor.error = str(exc)
        imgui.same_line()
        if _command_button(
            "##capture-snapshot",
            "key",
            ctx.tr("Capture complete scene state"),
            ctx.theme,
            scale,
            label=shown[2],
            draw=ctx.painter(),
            width=widths[2],
            layouts=self.command_layouts,
            enabled=ctx.session.adapter.caps.state_snapshots and not ctx.take_video_active,
        ):
            result = ctx.submit(cmd.CaptureSceneSnapshot())
            if result.ok:
                editor.selected_snapshot = result.entity_id
                editor.selected_id = -1
                editor.selected_keyframes.clear()
                editor.edit_lane = "snapshots"
                editor.view_needs_fit = True
            editor.error = "" if result.ok else result.message
        imgui.end_group()
        group(transport_width)
        self.draw_transport_header(editor, ctx, take_times, time_width)
        imgui.end_group()
        group(range_width)
        self.draw_take_range(editor, ctx, take_times, range_widths)
        imgui.end_group()
        group(view_width)
        follow_inline = button_row_layout(
            (follow_width, icon_width, icon_width),
            imgui.get_content_region_avail().x,
            gap,
        )
        mode = segmented_control(
            "timeline-follow",
            follow_labels,
            ("off", "page", "locked").index(editor.follow_mode),
            width=min(follow_width, imgui.get_content_region_avail().x),
            theme=ctx.theme,
            icons=follow_icons,
            icon_label_drawer=self.follow_mode_icon_drawer,
            draw=ctx.painter(),
            show_labels=False,
        )
        if ("off", "page", "locked")[mode] != editor.follow_mode:
            editor.set_follow_mode(("off", "page", "locked")[mode])
        if follow_inline[1]:
            imgui.same_line()
        self.draw_view_controls(editor, ctx)
        if follow_inline[2]:
            imgui.same_line(0, gap)
        if _command_button(
            "##timeline-options",
            "options",
            ctx.tr("Timeline settings"),
            ctx.theme,
            scale,
            draw=ctx.painter(),
        ):
            imgui.open_popup("timeline-options")
        imgui.push_style_var(imgui.StyleVar_.window_padding, (10 * scale, 8 * scale))
        if imgui.begin_popup("timeline-options"):
            imgui.begin_disabled(ctx.take_video_active)
            changed, value = imgui.checkbox(
                ctx.tr("Pause at last frame"), ctx.session.state_take_pause_at_end
            )
            imgui.set_item_tooltip(
                ctx.tr("When off, the playhead continues while the last pose holds")
            )
            if changed:
                if ctx.set_take_pause_at_end is not None:
                    ctx.set_take_pause_at_end(value)
                else:
                    ctx.submit(cmd.SetStateTakePauseAtEnd(value))
            imgui.end_disabled()
            if ctx.recording_config is not None:
                imgui.separator()
                titles = (ctx.tr("Start delay (s)"), ctx.tr("End hold (s)"))
                label_width = max(imgui.calc_text_size(title).x for title in titles) + 8 * scale
                flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
                if imgui.begin_table(
                    "timeline_recording_settings", 2, flags, (label_width + 140 * scale, 0)
                ):
                    imgui.table_setup_column(
                        "label", imgui.TableColumnFlags_.width_fixed, label_width
                    )
                    imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
                    for field, title in zip(("countdown", "end_hold"), titles, strict=True):
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.align_text_to_frame_padding()
                        imgui.text_disabled(title)
                        imgui.table_next_column()
                        imgui.set_next_item_width(-1)
                        changed, value = imgui.input_float(
                            "##timeline-" + field,
                            getattr(ctx.recording_config, field),
                            0.5,
                            5.0,
                            "%.1f",
                        )
                        if changed:
                            ctx.set_recording_config(
                                replace(ctx.recording_config, **{field: value})
                            )
                    imgui.end_table()
            imgui.end_popup()
        imgui.end_group()
        imgui.pop_style_var(3)

    def draw_take_range(self, editor: TimelineEditor, ctx, take_times, widths):
        scale = ctx.style_scale
        gap = 5 * scale
        recording = ctx.session.state_take_recording
        play_range = ctx.session.state_take_range
        endpoints = (
            [take_times[play_range[0]], take_times[play_range[1]]]
            if play_range
            else [take_times[0], take_times[-1]]
            if take_times
            else [editor.view_start, editor.view_end]
        )
        inline = button_row_layout(widths, imgui.get_content_region_avail().x, gap)
        imgui.set_next_item_width(widths[0])
        imgui.set_next_window_size_constraints(
            (160 * scale + 2 * imgui.get_style().frame_padding.x, 0), (10000, 10000)
        )
        imgui.begin_disabled(recording or ctx.take_video_active)
        if imgui.begin_combo("##timeline-take", self.take_label(editor, ctx)):
            imgui.text_disabled(self.take_status(ctx, take_times))
            for take in ctx.session.state_takes:
                selected, removed = self.draw_take_choice(editor, ctx, take)
                if removed:
                    result = ctx.submit(cmd.RemoveStateTake(take.take_id))
                    editor.error = "" if result.ok else result.message
                elif selected:
                    result = ctx.submit(cmd.SelectStateTake(take.take_id))
                    editor.error = "" if result.ok else result.message
            imgui.separator()
            if imgui.selectable(ctx.tr("New Take") + "##new-take", False)[0]:
                result = ctx.submit(cmd.CreateStateTake())
                editor.error = "" if result.ok else result.message
            if take_times and imgui.selectable(ctx.tr("Clear take"), False)[0]:
                result = ctx.submit(cmd.ClearStateTake())
                editor.error = "" if result.ok else result.message
            imgui.end_combo()
        imgui.end_disabled()
        imgui.begin_disabled(recording or ctx.take_video_active)
        values, changed = [], False
        for i, (name, title, value) in enumerate(
            zip(("start", "end"), ("Range start (s)", "Range end (s)"), endpoints, strict=True),
            start=1,
        ):
            if inline[i]:
                imgui.same_line()
            imgui.set_next_item_width(widths[i])
            edited, value = imgui.drag_float("##timeline-range-" + name, value, 0.01, format="%.2f")
            imgui.set_item_tooltip(ctx.tr(title))
            changed |= edited
            values.append(value)
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
        if inline[3]:
            imgui.same_line()
        if _command_button(
            "##timeline-loop",
            "loop",
            ctx.tr("Loop"),
            ctx.theme,
            scale,
            selected=ctx.session.state_take_loop_enabled,
            draw=ctx.painter(),
            enabled=len(take_times) > 1 and not recording,
        ):
            ctx.submit(cmd.SetStateTakeLoopEnabled(not ctx.session.state_take_loop_enabled))
        imgui.end_disabled()

    def draw_take_choice(self, editor: TimelineEditor, ctx, take):
        width = max(160 * ctx.style_scale, imgui.get_content_region_avail().x)
        height = imgui.get_frame_height()
        padding = imgui.get_style().frame_padding.x
        origin = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        splitter = imgui.ImDrawListSplitter()
        splitter.split(draw_list, 2)
        # The inset cross owns input first and paints over the full-row selection.
        splitter.set_current_channel(draw_list, 1)
        imgui.set_cursor_screen_pos((origin.x + width - height, origin.y))
        removed = clear_button(
            f"##remove-take-{take.take_id}", (height, height), ctx.tr("Delete take")
        )
        imgui.set_cursor_screen_pos(origin)
        splitter.set_current_channel(draw_list, 0)
        name = fit_text(ctx.painter(), self.take_name(ctx, take), width - height - 2 * padding)
        selected = padded_selectable(
            name + f"##take-{take.take_id}",
            take.take_id == ctx.session.active_state_take_id,
            size=imgui.ImVec2(width, height),
        )[0]
        splitter.merge(draw_list)
        return selected, removed

    @staticmethod
    def take_name(ctx, take):
        if take.name.startswith("Take ") and take.name[5:].isdecimal():
            return f"{ctx.tr('Take')} {take.name[5:]}"
        return take.name

    def take_label(self, editor: TimelineEditor, ctx):
        take = next(
            (
                take
                for take in ctx.session.state_takes
                if take.take_id == ctx.session.active_state_take_id
            ),
            None,
        )
        return ctx.tr("No take") if take is None else self.take_name(ctx, take)

    def draw_transport_header(self, editor: TimelineEditor, ctx, take_times, time_width):
        scale = ctx.style_scale
        width = imgui.get_content_region_avail().x
        button_width = _command_button_width("", scale)
        enabled = (
            bool(take_times) and not ctx.session.state_take_recording and not ctx.take_video_active
        )
        playing = ctx.session.state_take_playing
        play_range = ctx.session.state_take_range
        first, last = play_range or (0, max(0, len(take_times) - 1))
        cursor = ctx.session.state_take_cursor
        actions = (
            (
                "first",
                "first",
                cmd.SeekStateTake(first),
                "Range first frame" if play_range else "First frame",
            ),
            (
                "previous",
                "previous",
                cmd.SeekStateTake(min(last, max(first, cursor - 1))),
                "Previous frame",
            ),
            (
                "play-pause",
                "pause" if playing else "play",
                cmd.PauseStateTake() if playing else cmd.PlayStateTake(),
                "Pause" if playing else "Replay",
            ),
            ("next", "next", cmd.SeekStateTake(min(last, max(first, cursor + 1))), "Next frame"),
            (
                "last",
                "last",
                cmd.SeekStateTake(last),
                "Range last frame" if play_range else "Last frame",
            ),
        )
        widths = (button_width,) * len(actions) + (time_width,)
        inline = button_row_layout(widths, width, imgui.get_style().item_spacing.x)
        for i, (name, icon, command, label) in enumerate(actions):
            if inline[i]:
                imgui.same_line()
            if _command_button(
                f"##take-{name}",
                icon,
                ctx.tr(label),
                ctx.theme,
                scale,
                enabled=enabled
                or (
                    name == "first"
                    and not ctx.session.state_take_recording
                    and not ctx.take_video_active
                ),
                draw=ctx.painter(),
                selected=name == "play-pause" and playing,
            ):
                if name == "first" and not take_times:
                    editor.playhead = 0.0
                    editor.last_followed_playhead = None
                    if not editor.view_start <= 0 <= editor.view_end:
                        editor.view_start, editor.view_end = (
                            0.0,
                            editor.view_end - editor.view_start,
                        )
                else:
                    result = ctx.submit(command)
                    editor.error = "" if result.ok else result.message
        if inline[len(actions)]:
            imgui.same_line()
        _toolbar_status(
            f"{editor.playhead:.3f} s", ctx.theme.text_disabled, scale, width=time_width
        )

    def draw_view_controls(self, editor: TimelineEditor, ctx):
        if _command_button(
            "##key-view",
            "fit",
            ctx.tr("View all"),
            ctx.theme,
            ctx.style_scale,
            draw=ctx.painter(),
        ):
            editor.view_needs_fit = True

    @staticmethod
    def take_status(ctx: PanelContext, take_times: Sequence[float]) -> str:
        session = ctx.session
        cursor = session.state_take_cursor
        if session.state_take_recording:
            return f"{ctx.tr('REC')} · {ctx.tr('Recorded frames')}: {len(take_times)}"
        if session.state_take_playing:
            return f"{ctx.tr('PLAY')} · {ctx.tr('frame')}: {cursor + 1}/{len(take_times)}"
        if take_times:
            if cursor >= 0:
                return f"{ctx.tr('frame')}: {cursor + 1}/{len(take_times)}"
            return f"{ctx.tr('Recorded frames')}: {len(take_times)}"
        supported = bool(session.adapter.caps.simulation and session.adapter.caps.state_snapshots)
        return ctx.tr("no recorded take" if supported else "state recording unavailable")


def draw_model_header(editor: TimelineEditor, ctx, models, keyframes, editable, width):
    scale = ctx.style_scale
    gap = 5 * scale
    icon_width = _command_button_width("", scale)
    imgui.set_next_item_width(max(1, width - icon_width - gap))
    model_ids = tuple(model.model_id for model in models)
    slot = model_ids.index(editor.model_id) if editor.model_id in model_ids else 0
    imgui.begin_disabled(not models)
    changed, slot = imgui.combo(
        "##keyframe-model", slot, tuple(model.name for model in models) or (ctx.tr("Scene"),)
    )
    imgui.end_disabled()
    imgui.set_item_tooltip(models[slot].name if models else ctx.tr("Scene"))
    if changed:
        editor.set_model(model_ids[slot])
    imgui.same_line()
    if _command_button(
        "##add-model-keyframe",
        "add",
        ctx.tr("Add Keyframe"),
        ctx.theme,
        scale,
        enabled=editable and editor.model_id >= 0,
        draw=ctx.painter(),
    ):
        existing = (
            ctx.model_keyframe_names(editor.model_id)
            if ctx.model_keyframe_names is not None
            else {key.name for key in keyframes}
        )
        name = unique_keyframe_name(existing)
        editor.edit_lane = "model"
        ctx.submit_model_edit(cmd.AddModelKeyframe(editor.model_id, name), editor.snapshot_created)
