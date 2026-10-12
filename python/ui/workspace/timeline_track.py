"""Native timeline lanes sharing the editor's projection and gesture geometry."""

from __future__ import annotations

import bisect
import math

from imgui_bundle import imgui

from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.keyframe_editor.controls import _format_tick
from mojive.ui.keyframe_editor.toolbar import draw_model_header
from mojive.ui.keyframe_editor.track import TimelineLayout
from mojive.ui.text_layout import fit_text, text_line_y
from mojive.ui.theme import with_alpha
from mojive.ui.timeline import nice_timeline_step, take_frame_bounds, timeline_time_to_x

from .design import CONTROL_RADIUS, DESIGN
from .widgets import BODY_SIZE, TEXT


def measure_layout(editor, ctx, origin, available):
    """Reserve complete markers at both edges and stable room for all three lanes."""
    scale = ctx.style_scale
    width = max(1.0, available.x)
    model = next(
        (item for item in ctx.session.scene_models if item.model_id == editor.model_id), None
    )
    caption = imgui.calc_text_size(model.name if model is not None else ctx.tr("Scene")).x
    channel = min(width * 0.5, max(128 * scale, min(280 * scale, caption + 85 * scale)))
    ruler = 30 * scale
    height = ruler + 3 * DESIGN.timeline_lane_height * scale
    return TimelineLayout(tuple(origin), width, height, ruler, channel, 6 * scale, 10 * scale)


class NativeTimelineTrack:
    """Paint quiet lanes, named keys and sample coverage without replacing editing."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.snapshot_source = None
        self.snapshot_by_id = {}

    def draw_header(self, editor, ctx, layout, models, editable, icon_drawer):
        scale = ctx.style_scale
        imgui.set_cursor_screen_pos((layout.lo[0] + 2 * scale, layout.lo[1] + scale))
        imgui.push_style_color(imgui.Col_.frame_bg, (0, 0, 0, 0))
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding, (8 * scale, (28 - BODY_SIZE) * scale / 2)
        )
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (5 * scale, 6 * scale))
        try:
            draw_model_header(
                editor,
                ctx,
                models,
                editable,
                layout.channel_width - 4 * scale,
                icon_drawer=icon_drawer,
                selector=self._model_selector,
            )
        finally:
            imgui.pop_style_var(2)
            imgui.pop_style_color()

    def _model_selector(self, editor, ctx, models, width):
        ui, scale = self.workspace.ui, ctx.style_scale
        model_ids = tuple(model.model_id for model in models)
        slot = model_ids.index(editor.model_id) if editor.model_id in model_ids else 0
        labels = tuple(model.name for model in models) or (ctx.tr("Scene"),)
        draw_list = imgui.get_window_draw_list()
        imgui.set_next_item_width(width)
        opened = imgui.begin_combo("##keyframe-model", "")
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits["timeline-model"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        left = lo.x + 8 * scale
        right = max(left, hi.x - imgui.get_frame_height() - 8 * scale)
        ui.hits["timeline-model-caption"] = (left, lo.y, right - left, hi.y - lo.y)
        draw_list.push_clip_rect((left, lo.y), (right, hi.y), True)
        ui.text(
            left,
            (lo.y + hi.y - BODY_SIZE * scale) / 2,
            labels[slot],
            TEXT,
            draw=draw_list,
            max_width=right - left,
            translate=False,
        )
        draw_list.pop_clip_rect()
        changed = False
        if opened:
            for index, label in enumerate(labels):
                chosen, _ = imgui.selectable(label + f"##keyframe-model-{index}", slot == index)
                if chosen:
                    changed, slot = True, index
                if slot == index:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        return changed, slot

    def paint(self, editor, ctx, layout, projection, snapshots, hit, keyframes, take_times):
        imgui.push_font(self.workspace.ui.body, DESIGN.timeline_lane_size * ctx.style_scale)
        try:
            self._paint(editor, ctx, layout, projection, snapshots, hit, keyframes, take_times)
        finally:
            imgui.pop_font()

    def _paint(self, editor, ctx, layout, projection, snapshots, hit, keyframes, take_times):
        draw, scale, theme = ImguiDraw2D(), ctx.style_scale, ctx.theme
        track_left = layout.lo[0] + layout.channel_width
        draw.rect_filled(layout.lo, layout.hi, theme.bg_window)
        draw.rect_filled((track_left, layout.lo[1]), layout.hi, theme.bg_child, rounding=4 * scale)
        lane_colors = (theme.primary, theme.primary, theme.info)
        rows = (
            ("Keyframes", "model", "diamond"),
            ("Take", "take", "record"),
            ("Snapshots", "snapshots", "link"),
        )
        for index, (label, lane, icon) in enumerate(rows):
            center = layout.lane_y(index)
            top, bottom = center - layout.row_height / 2, center + layout.row_height / 2
            color = lane_colors[index]
            self.workspace.ui.hits["timeline-lane-label-" + str(index)] = (
                layout.lo[0],
                top,
                layout.channel_width,
                layout.row_height,
            )
            if editor.edit_lane == lane:
                draw.rect_filled(
                    (layout.lo[0], top),
                    (track_left, bottom),
                    with_alpha(color, 0.09),
                    rounding=CONTROL_RADIUS * scale,
                )
            self.workspace.ui.role_icon(
                icon,
                layout.lo[0] + 8 * scale,
                center - 14 * scale,
                14 * scale,
                28 * scale,
                role="inline",
                c=tuple(round(value * 255) for value in color),
            )
            caption_x = 8 + 14 + DESIGN.timeline_label_gap
            caption = fit_text(draw, ctx.tr(label), layout.channel_width - (caption_x + 6) * scale)
            draw.text(
                (layout.lo[0] + caption_x * scale, text_line_y(draw, center)), theme.text, caption
            )
            if index:
                draw.line(
                    (track_left, top),
                    (layout.hi[0], top),
                    with_alpha(theme.text, 0.045),
                    0.5 * scale,
                )
        draw.line(
            (track_left, layout.ruler_bottom),
            (layout.hi[0], layout.ruler_bottom),
            with_alpha(theme.text, 0.08),
            0.5 * scale,
        )
        imgui.push_clip_rect((track_left, layout.lo[1]), layout.hi, True)
        try:
            self._ruler(draw, editor, ctx, layout)
            self._range(draw, editor, ctx, layout, take_times)
            self._take(draw, editor, ctx, layout, take_times)
            self._keys(draw, editor, ctx, layout, projection, hit, keyframes)
            self._snapshots(draw, editor, ctx, layout, snapshots, hit)
            self._playhead(draw, editor, ctx, layout)
        finally:
            imgui.pop_clip_rect()
        if hit.snapshot_id >= 0:
            item = self.snapshot_by_id[hit.snapshot_id]
            imgui.set_tooltip(f"{item.name}  {item.time:g} s\n{ctx.tr('Double-click to load')}")
        elif hit.keyframe_id >= 0:
            key = keyframes[hit.keyframe_id]
            imgui.set_tooltip(
                f"{key.name or ctx.tr('keyframe')}  {key.time:g} s\n"
                f"{ctx.tr('Double-click to load')}"
            )

    @staticmethod
    def _x(editor, layout, time):
        return timeline_time_to_x(time, editor.view_start, editor.view_end, *layout.time_bounds)

    def _ruler(self, draw, editor, ctx, layout):
        imgui.push_font(self.workspace.ui.body, DESIGN.timeline_ruler_size * ctx.style_scale)
        try:
            self._draw_ruler(draw, editor, ctx, layout)
        finally:
            imgui.pop_font()

    def _draw_ruler(self, draw, editor, ctx, layout):
        scale = ctx.style_scale
        left, right = layout.time_bounds
        step = nice_timeline_step(editor.view_end - editor.view_start, right - left, 80 * scale)
        tick = math.ceil(editor.view_start / (step / 5)) * (step / 5)
        for _ in range(1000):
            if tick > editor.view_end + step * 1e-7:
                break
            x = self._x(editor, layout, tick)
            major = abs(tick / step - round(tick / step)) < 1e-6
            draw.line(
                (x, layout.ruler_bottom - (5 if major else 3) * scale),
                (x, layout.ruler_bottom),
                with_alpha(ctx.theme.text, 0.24 if major else 0.1),
                0.7 * scale,
            )
            if major:
                draw.line(
                    (x, layout.ruler_bottom),
                    (x, layout.hi[1]),
                    with_alpha(ctx.theme.text, 0.065),
                    0.5 * scale,
                )
                caption = _format_tick(tick, step)
                if x + 4 * scale + draw.text_size(caption)[0] <= layout.hi[0] - 3 * scale:
                    draw.text(
                        (x + 4 * scale, text_line_y(draw, layout.lo[1] + 13 * scale)),
                        ctx.theme.text_disabled,
                        caption,
                    )
            tick += step / 5

    def _range(self, draw, editor, ctx, layout, take_times):
        selection = editor.range_preview or ctx.session.state_take_range
        if selection is None or not take_times:
            return
        scale, color = ctx.style_scale, ctx.theme.warning
        first, last = self._sample_bounds(
            editor, layout, take_times, selection, ctx.session.state_take_cells
        )
        draw.rect_filled(
            (first, layout.ruler_bottom), (last, layout.hi[1]), with_alpha(color, 0.045)
        )
        draw.rect_filled(
            (first, layout.ruler_bottom - 3 * scale),
            (last, layout.ruler_bottom),
            with_alpha(color, 0.7),
            rounding=scale,
        )
        for x in (first, last):
            draw.line(
                (x, layout.ruler_bottom - 7 * scale),
                (x, layout.ruler_bottom + 4 * scale),
                color,
                1.5 * scale,
            )

    def _take(self, draw, editor, ctx, layout, take_times):
        if not take_times:
            return
        session = ctx.session
        cells, segments = session.state_take_cells, session.state_take_segments
        scale, center, color = ctx.style_scale, layout.lane_y(1), ctx.theme.primary
        left, right = layout.time_bounds
        take = next(
            (item for item in session.state_takes if item.take_id == session.active_state_take_id),
            None,
        )
        label = take.name if take is not None else ctx.tr("Take")
        if label.startswith("Take ") and label[5:].isdecimal():
            label = f"{ctx.tr('Take')} {label[5:]}"
        named = False
        start = max(0, bisect.bisect_left(take_times, editor.view_start) - 1)
        end = min(len(take_times), bisect.bisect_right(take_times, editor.view_end) + 1)
        show_cells = end - start <= (right - left) / (10 * scale)
        first_segment = max(0, bisect.bisect_right(segments, (start, len(take_times))) - 1)
        for slot in range(first_segment, len(segments)):
            a, b = segments[slot]
            if a >= end:
                break
            first, last = self._sample_bounds(editor, layout, take_times, (a, b), cells)
            first, last = max(left, first), min(right, last)
            if first >= last:
                continue
            rect = (first, center - 11 * scale, last - first, 22 * scale)
            self.workspace.ui.hits[f"timeline-take-segment-{slot}"] = rect
            if "timeline-take-strip" not in self.workspace.ui.hits:
                self.workspace.ui.hits["timeline-take-strip"] = rect
            draw.rect_filled(
                (first, center - 11 * scale),
                (last, center + 11 * scale),
                with_alpha(color, 0.18),
                rounding=3 * scale,
            )
            draw.rect_filled(
                (first, center - 11 * scale),
                (last, center - 9 * scale),
                with_alpha(color, 0.65),
            )
            # Only recording adjacency joins cells. Zoom changes tick density;
            # deleting frames leaves their original coverage empty.
            if show_cells:
                for index in range(max(a + 1, start), min(b + 1, end)):
                    x = self._x(editor, layout, cells[index][0])
                    if left <= x <= right:
                        draw.line(
                            (x, center - 8 * scale),
                            (x, center + 8 * scale),
                            with_alpha(color, 0.2),
                            0.7 * scale,
                        )
            if not named and last - first > 88 * scale:
                caption = fit_text(draw, label, last - first - 16 * scale)
                draw.text((first + 8 * scale, text_line_y(draw, center)), color, caption)
                named = True
            if editor.take_selection is None:
                continue
            selected_first = max(a, editor.take_selection[0])
            selected_last = min(b, editor.take_selection[1])
            if selected_first > selected_last:
                continue
            first, last = self._sample_bounds(
                editor, layout, take_times, (selected_first, selected_last), cells
            )
            first, last = max(left, first), min(right, last)
            if first >= last:
                continue
            rect = (first, center - 11 * scale, last - first, 22 * scale)
            self.workspace.ui.hits[f"timeline-take-selection-{slot}"] = rect
            self.workspace.ui.hits["timeline-take-selection"] = rect
            draw.rect_filled(
                (first, center - 11 * scale),
                (last, center + 11 * scale),
                with_alpha(ctx.theme.warning, 0.22),
                rounding=3 * scale,
            )
            draw.rect(
                (first, center - 11 * scale),
                (last, center + 11 * scale),
                with_alpha(ctx.theme.warning, 0.7),
                scale,
                rounding=3 * scale,
            )

    def _sample_bounds(self, editor, layout, times, selection, cells):
        first, last = (
            self._x(editor, layout, time)
            for time in take_frame_bounds(times, *selection, cells=cells)
        )
        if first == last:
            return first - layout.marker_radius, last + layout.marker_radius
        return first, last

    def _keys(self, draw, editor, ctx, layout, projection, hit, keyframes):
        radius, center, scale = layout.marker_radius, layout.lane_y(0), ctx.style_scale
        ids = projection.draw_ids(
            *layout.time_bounds,
            radius * 2.5,
            (editor.selected_id, ctx.session.active_keyframe, hit.keyframe_id, editor.drag_id),
        )
        ordered = sorted(ids, key=lambda key_id: projection.positions[key_id])
        for index, key_id in enumerate(ordered):
            key, x = keyframes[key_id], projection.positions[key_id]
            selected = key_id == editor.selected_id or key_id in editor.selected_keyframes
            color = ctx.theme.warning if selected else ctx.theme.primary
            points = (
                (x, center - radius),
                (x + radius, center),
                (x, center + radius),
                (x - radius, center),
            )
            if key_id == hit.keyframe_id or selected:
                draw.circle_filled((x, center), radius + 5 * scale, with_alpha(color, 0.1))
            draw.convex_fill(points, color)
            next_x = (
                projection.positions[ordered[index + 1]]
                if index + 1 < len(ordered)
                else layout.hi[0]
            )
            room = next_x - x - radius * 2 - 10 * scale
            if room >= 36 * scale:
                caption = fit_text(draw, key.name or ctx.tr("keyframe"), room)
                draw.text(
                    (x + radius + 6 * scale, text_line_y(draw, center)),
                    color if selected else ctx.theme.text,
                    caption,
                )

    def _snapshots(self, draw, editor, ctx, layout, projection, hit):
        source = ctx.session.scene_snapshots
        if source is not self.snapshot_source:
            self.snapshot_source = source
            self.snapshot_by_id = {item.snapshot_id: item for item in source}
        radius, center, scale = layout.marker_radius, layout.lane_y(2), ctx.style_scale
        ids = projection.draw_ids(
            *layout.time_bounds, radius * 2.5, (editor.selected_snapshot, hit.snapshot_id)
        )
        ordered = sorted(ids, key=lambda item_id: projection.positions[item_id])
        for index, item_id in enumerate(ordered):
            item, x = self.snapshot_by_id[item_id], projection.positions[item_id]
            selected = item_id == editor.selected_snapshot
            color = ctx.theme.info
            draw.rect_filled(
                (x - radius, center - radius),
                (x + radius, center + radius),
                with_alpha(color, 0.22 if selected else 0.06),
                rounding=2 * scale,
            )
            draw.rect(
                (x - radius, center - radius),
                (x + radius, center + radius),
                color,
                (1.5 if item_id == hit.snapshot_id or selected else 1) * scale,
                rounding=2 * scale,
            )
            draw.circle_filled((x, center), 1.5 * scale, color)
            next_x = (
                projection.positions[ordered[index + 1]]
                if index + 1 < len(ordered)
                else layout.hi[0]
            )
            room = next_x - x - radius * 2 - 10 * scale
            if room >= 36 * scale:
                caption = fit_text(draw, item.name, room)
                draw.text(
                    (x + radius + 6 * scale, text_line_y(draw, center)),
                    color if selected else ctx.theme.text_disabled,
                    caption,
                )

    def _playhead(self, draw, editor, ctx, layout):
        scale, x = ctx.style_scale, self._x(editor, layout, editor.playhead)
        if not layout.time_bounds[0] <= x <= layout.time_bounds[1]:
            return
        color = ctx.theme.danger
        draw.line((x, layout.lo[1] + 6 * scale), (x, layout.hi[1]), with_alpha(color, 0.85), scale)
        draw.convex_fill(
            (
                (x - 4 * scale, layout.lo[1]),
                (x + 4 * scale, layout.lo[1]),
                (x + 4 * scale, layout.lo[1] + 5 * scale),
                (x, layout.lo[1] + 9 * scale),
                (x - 4 * scale, layout.lo[1] + 5 * scale),
            ),
            color,
        )
