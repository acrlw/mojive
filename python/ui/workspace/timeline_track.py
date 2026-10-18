"""Native timeline lanes sharing the editor's projection and gesture geometry."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.keyframe_editor.controls import _format_tick
from mojive.ui.keyframe_editor.toolbar import add_model_keyframe
from mojive.ui.keyframe_editor.track import TimelineLayout, draw_selection_menu
from mojive.ui.theme import with_alpha
from mojive.ui.timeline import (
    nearest_take_frame,
    nice_timeline_step,
    take_frame_bounds,
    timeline_time_to_x,
    timeline_x_to_time,
)
from mojive.ui.workspace.component_layout import control_frame_size
from mojive.ui.workspace.design import CONTROL_RADIUS, DESIGN
from mojive.ui.workspace.timeline_header import model_header_layout
from mojive.ui.workspace.widgets import BODY_SIZE


@dataclass(frozen=True)
class NativeTimelineLayout(TimelineLayout):
    """One measured geometry shared by marker projection, gestures and painting."""

    lane_height: float = 0.0
    lane_gap: float = 0.0
    lane_margin_top: float = 0.0
    lane_margin_bottom: float = 0.0
    right_inset: float = 0.0
    marker_offset: float = 0.0
    left_inset: float = 0.0
    header_gap: float = 0.0

    @property
    def header_height(self):
        return self.ruler_height - self.header_gap

    @property
    def row_height(self):
        return self.lane_height

    def lane_y(self, index):
        return (
            self.ruler_bottom
            + index * (self.lane_height + self.lane_gap)
            + self.lane_margin_top
            + (self.lane_height - self.lane_margin_top - self.lane_margin_bottom) / 2
            + (self.marker_offset if index != 1 else 0)
        )

    def lane_at(self, y):
        index = min(
            2, max(0, int((y - self.ruler_bottom) / max(1, self.lane_height + self.lane_gap)))
        )
        return ("model", "take", "snapshots")[index]

    @property
    def time_bounds(self):
        left = min(self.hi[0], self.track_left + self.time_padding + self.left_inset)
        return left, max(left, self.hi[0] - self.right_inset)


@part_scope("timeline_sheet")
def measure_layout(editor, ctx, origin, available, *, ui=None):
    """Resolve the same track geometry for drawing and native timeline input."""
    scale = ctx.style_scale
    width = max(1.0, available.x)
    model = next(
        (item for item in ctx.session.scene_models if item.model_id == editor.model_id), None
    )
    caption = imgui.calc_text_size(model.name if model is not None else ctx.tr("Scene")).x
    channel = min(width * 0.5, max(128 * scale, min(280 * scale, caption + 85 * scale)))
    ruler, lane, gap, marker, right = (
        30 * scale,
        DESIGN.timeline_lane_height * scale,
        0.0,
        6 * scale,
        10 * scale,
    )
    top_margin = bottom_margin = marker_offset = left_inset = 0.0
    header_gap = 0.0
    column_gap = 12 * scale
    lo = tuple(origin)
    if ui is not None:
        sheet = ui.component_style("timeline_sheet")
        ruler_style = ui.component_style("timeline_ruler")
        lane_style = ui.component_style("timeline_lane")
        ruler_margin = ruler_style.edges("margin", scale)
        lane_margin = lane_style.edges("margin", scale)
        ruler_font = ruler_style.get("fontSize", DESIGN.timeline_ruler_size)
        ruler_line = max(ruler_font, ruler_style.line_height(ruler_font * 1.4))
        ruler = (
            max(
                ruler_style.get("height", ruler / scale),
                2 * abs(ruler_style.get("labelTop", 0)) + ruler_line,
            )
            * scale
            + ruler_margin[1]
            + ruler_margin[3]
        )
        lane_font = lane_style.get("fontSize", BODY_SIZE)
        lane_line = max(lane_font, lane_style.line_height(lane_font * 1.4))
        lane = (
            max(lane_style.get("height", lane / scale), lane_line) * scale
            + lane_margin[1]
            + lane_margin[3]
        )
        top_margin, bottom_margin = lane_margin[1], lane_margin[3]
        gap = sheet.get("rowGap", 0) * scale
        column_gap = sheet.get("columnGap", 12) * scale
        channel = min(max(0, width - 2 * scale), sheet.get("labelWidth", channel / scale) * scale)
        header, _ = model_header_layout(ui, 0, 0, channel)
        header_gap = sheet.get("headerGap", 8) * scale
        ruler = max(ruler, header.outer.height) + header_gap
        key_style = ui.component_style("timeline_key")
        marker = key_style.get("size", 12) * scale / 2
        key_margin = key_style.edges("margin", scale)
        lane = max(lane, 2 * marker + key_margin[1] + key_margin[3])
        left_inset = max(lane_margin[0], ruler_margin[0])
        if key_style.explicit("top"):
            marker_offset = (
                key_style.get("top", 4) * scale + marker - (lane - top_margin - bottom_margin) / 2
            )
        right = sheet.get("rightInset", 10) * scale + max(lane_margin[2], ruler_margin[2])
        sheet_margin = sheet.edges("margin", scale)
        natural_height = ruler + 3 * lane + 2 * gap
        requested_height = sheet.get("height", natural_height / scale) * scale
        track_height = max(natural_height, requested_height)
        lane += max(0, track_height - natural_height) / 3
        lo = (origin[0] + sheet_margin[0], origin[1] + sheet_margin[1])
        width = max(scale, width - sheet_margin[0] - sheet_margin[2])
    minimum_track = marker + 4 * scale + left_inset + right + scale
    channel = min(channel, max(0, width - column_gap - minimum_track))
    column_gap = min(column_gap, max(0, width - channel - minimum_track))
    height = ruler + 3 * lane + 2 * gap
    if ui is not None:
        ui.hits["timeline-gutter"] = (lo[0] + channel, lo[1], column_gap, height)
        if ui.design is not None:
            ui.design.record(
                "timeline-sheet",
                ("timeline_sheet",),
                (*lo, width, height),
                "container",
                content=[*lo, width, height],
                outer=[
                    origin[0],
                    origin[1],
                    width + sheet_margin[0] + sheet_margin[2],
                    height + sheet_margin[1] + sheet_margin[3],
                ],
                margin=list(sheet_margin),
                columnGap=column_gap / scale,
                gutterRect=[lo[0] + channel, lo[1], column_gap, height],
                fieldRegions={
                    "columnGap": [lo[0] + channel, lo[1], column_gap, height],
                    "headerGap": [lo[0], lo[1] + ruler - header_gap, width, header_gap],
                    "rowGap": [
                        [lo[0], lo[1] + ruler + (i + 1) * lane + i * gap, width, gap]
                        for i in range(2)
                    ],
                },
            )
    return NativeTimelineLayout(
        lo,
        width,
        height,
        ruler,
        channel,
        marker,
        marker + 4 * scale,
        lane,
        gap,
        top_margin,
        bottom_margin,
        right,
        marker_offset,
        left_inset,
        header_gap,
        column_gap=column_gap,
    )


class NativeTimelineTrack:
    """Paint quiet lanes, named keys and sample coverage without replacing editing."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.snapshot_source = None
        self.snapshot_by_id = {}

    def draw_selection_menu(self, editor, ctx, editable):
        draw_selection_menu(editor, ctx, editable, menus=self.workspace.menus)

    @part_scope("timeline_model_group")
    def draw_header(self, editor, ctx, layout, models, editable, icon_drawer):
        ui, scale = self.workspace.ui, ctx.style_scale
        box, group = model_header_layout(ui, *layout.lo, layout.channel_width, observe=True)
        selector, add = group.rectangles
        ui.hits["timeline-model-group"] = box.outer.tuple()
        imgui.set_cursor_screen_pos((selector.x, selector.y))
        imgui.begin_disabled(not models)
        changed, slot = self._model_selector(editor, ctx, models, selector.width)
        imgui.end_disabled()
        model_ids = tuple(model.model_id for model in models)
        if changed:
            editor.set_model(model_ids[slot])
        frame = control_frame_size(ui, "timeline-add-key", add.width, add.height, icon="plus")
        if ui.button(
            "timeline-add-key",
            add.x,
            add.y,
            *frame,
            icon="plus",
            enabled=editable and editor.model_id >= 0,
            tooltip="Add Keyframe",
            width_limit=frame[0] * scale,
        ):
            add_model_keyframe(editor, ctx)

    def _model_selector(self, editor, ctx, models, width):
        ui, menus = self.workspace.ui, self.workspace.menus
        model_ids = tuple(model.model_id for model in models)
        slot = model_ids.index(editor.model_id) if editor.model_id in model_ids else 0
        labels = tuple(model.name for model in models) or (ctx.tr("Scene"),)
        pos = imgui.get_cursor_screen_pos()
        changed = False
        from mojive.ui.workspace.popup_style import menu_choice_width

        opened = ui.begin_combo(
            "timeline-model",
            pos.x,
            pos.y,
            width,
            labels[slot],
            popup_width=menu_choice_width(ui, labels, slot, icon="file"),
        )
        frame = ui.hits["timeline-model"]
        padding = ui.component_style("select").edges("padding", ui.s, (8, 0, 8, 0))
        left = frame[0] + padding[0]
        right = max(left, frame[0] + frame[2] - padding[2] - frame[3])
        ui.hits["timeline-model-caption"] = (left, frame[1], right - left, frame[3])
        if opened:
            with menus.local_content("timeline-model-choices", choices=True):
                for index, label in enumerate(labels):
                    chosen, _ = menus.local_item(
                        label,
                        checked=slot == index,
                        checkable=True,
                        key=f"timeline-model-choice-{index}",
                        icon="file",
                    )
                    if chosen:
                        changed, slot = True, index
                    if slot == index:
                        imgui.set_item_default_focus()
            imgui.end_combo()
        return changed, slot

    @part_scope("timeline_sheet")
    def paint(self, editor, ctx, layout, projection, snapshots, hit, keyframes, take_times):
        imgui.push_font(
            self.workspace.ui.resolve_font(self.workspace.ui.component_style("timeline_lane")),
            self.workspace.ui.metric("timeline_lane", "fontSize", DESIGN.timeline_lane_size)
            * ctx.style_scale,
        )
        try:
            self._paint(editor, ctx, layout, projection, snapshots, hit, keyframes, take_times)
        finally:
            imgui.pop_font()

    def _paint(self, editor, ctx, layout, projection, snapshots, hit, keyframes, take_times):
        draw, scale, theme = ImguiDraw2D(), ctx.style_scale, ctx.theme
        track_left = layout.track_left
        ui = self.workspace.ui
        radius = ui.component_style("timeline_sheet").get("radius", 4) * scale
        draw.rect_filled(layout.lo, layout.hi, theme.bg_window, rounding=radius)
        draw.rect_filled((track_left, layout.lo[1]), layout.hi, theme.bg_child, rounding=radius)
        if ui.design is not None:
            ui.design.record(
                "timeline-sheet-background",
                ("timeline_sheet",),
                (*layout.lo, layout.width, layout.height),
                "timeline_track",
                radius=radius,
            )
            ui.design.record(
                "timeline-track-background",
                ("timeline_sheet",),
                (track_left, layout.lo[1], layout.hi[0] - track_left, layout.height),
                "timeline_track",
                radius=radius,
            )
        lane_colors = (theme.primary, theme.primary, theme.info)
        rows = (
            ("Keyframes", "model", "diamond"),
            ("Take", "take", "record"),
            ("Snapshots", "snapshots", "snapshot"),
        )
        lane_boxes = []
        self._lane_text_rects = [[], [], []]
        for index, (label, lane, icon) in enumerate(rows):
            top = layout.ruler_bottom + index * (layout.row_height + layout.lane_gap)
            color = lane_colors[index]
            self.workspace.ui.hits["timeline-lane-label-" + str(index)] = (
                layout.lo[0],
                top,
                layout.channel_width,
                layout.row_height,
            )
            ui = self.workspace.ui
            lane_box = ui.component_box(
                "timeline_lane",
                layout.lo[0],
                top,
                layout.width,
                layout.row_height,
                key=f"timeline-lane-{index}",
                observe=False,
            )
            lane_boxes.append(lane_box)
            with ui.component_scope("timeline_lane"):
                label_style = ui.component_style("lane_label")
                label_box = ui.component_box(
                    "lane_label",
                    lane_box.content.x,
                    lane_box.content.y,
                    max(0, layout.channel_right - lane_box.content.x),
                    lane_box.content.height,
                    key=f"timeline-lane-label-{index}",
                    padding=(8 * scale, 0, 6 * scale, 0),
                    background=editor.edit_lane == lane,
                    observe=False,
                )
                if editor.edit_lane == lane:
                    ui.rect(
                        *label_box.frame.tuple(),
                        tuple(round(v * 255) for v in with_alpha(color, 0.09)),
                        label_style.get("radius", CONTROL_RADIUS),
                    )
                extent = label_style.get("iconSize", 14) * scale
                gap = (
                    label_style.get("columnGap", label_style.get("gap", DESIGN.timeline_label_gap))
                    * scale
                )
                icon_rect = label_style.align_rect(
                    label_box.content, extent, extent, horizontal="start"
                )
                text_rect = Rect(
                    label_box.content.x + extent + gap,
                    label_box.content.y,
                    max(0, label_box.content.width - extent - gap),
                    label_box.content.height,
                )
                ui.record_component_box(
                    "lane_label",
                    label_box,
                    key=f"timeline-lane-label-{index}",
                    background=editor.edit_lane == lane,
                    radius=label_style.get("radius", CONTROL_RADIUS) * scale,
                    iconRect=list(icon_rect.tuple()),
                    textRect=list(text_rect.tuple()),
                )
                ui.icon(
                    icon,
                    icon_rect.x,
                    icon_rect.y,
                    extent / scale,
                    tuple(round(value * 255) for value in color),
                    style=label_style,
                )
                ui.component_text(
                    "lane_label",
                    text_rect,
                    label,
                    tuple(round(v * 255) for v in theme.text),
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
        for index, box in enumerate(lane_boxes):
            self.workspace.ui.record_component_box(
                "timeline_lane",
                box,
                key=f"timeline-lane-{index}",
                hasLaneText=bool(self._lane_text_rects[index]),
                captionRects=self._lane_text_rects[index],
            )
        if hit.snapshot_id >= 0:
            item = self.snapshot_by_id[hit.snapshot_id]
            self.workspace.ui.tooltip(
                f"{item.name}  {item.time:g} s\n{ctx.tr('Double-click to load')}", translate=False
            )
        elif hit.keyframe_id >= 0:
            key = keyframes[hit.keyframe_id]
            self.workspace.ui.tooltip(
                f"{key.name or ctx.tr('keyframe')}  {key.time:g} s\n"
                f"{ctx.tr('Double-click to load')}",
                translate=False,
            )
        elif take_times:
            self._take_tooltip(editor, ctx, layout, hit, take_times)

    @staticmethod
    def _x(editor, layout, time):
        return timeline_time_to_x(time, editor.view_start, editor.view_end, *layout.time_bounds)

    def _ruler(self, draw, editor, ctx, layout):
        imgui.push_font(
            self.workspace.ui.resolve_font(self.workspace.ui.component_style("timeline_ruler")),
            self.workspace.ui.metric("timeline_ruler", "fontSize", DESIGN.timeline_ruler_size)
            * ctx.style_scale,
        )
        try:
            self._draw_ruler(draw, editor, ctx, layout)
        finally:
            imgui.pop_font()

    @part_scope("timeline_ruler")
    def _draw_ruler(self, draw, editor, ctx, layout):
        scale = ctx.style_scale
        left, right = layout.time_bounds
        ui = self.workspace.ui
        ui.component_box(
            "timeline_ruler",
            layout.lo[0],
            layout.lo[1],
            layout.width,
            layout.ruler_height,
            key="timeline-ruler",
            minimum_height=layout.ruler_height
            - sum(ui.component_style("timeline_ruler").edges("margin", scale)[n] for n in (1, 3)),
        )
        step = nice_timeline_step(editor.view_end - editor.view_start, right - left, 80 * scale)
        first_tick = math.ceil(editor.view_start / (step / 5))
        label_right = left
        for index in range(1000):
            tick_index = first_tick + index
            tick = tick_index * (step / 5)
            if tick > editor.view_end + step * 1e-7:
                break
            x = self._x(editor, layout, tick)
            major = tick_index % 5 == 0
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
                ruler = ui.component_style("timeline_ruler")
                caption_width = ui.measure(
                    caption,
                    ruler.get("fontSize", DESIGN.timeline_ruler_size),
                    translate=False,
                    font_resolved=True,
                    style=ruler,
                )
                caption_left = max(left, min(x - caption_width / 2, right - caption_width))
                caption_right = caption_left + caption_width
                if caption_left >= label_right and caption_right <= right:
                    margin = ruler.edges("margin", scale)
                    label_top = ruler.get("labelTop", 0) * scale
                    rect = Rect(
                        caption_left,
                        layout.lo[1] + margin[1] + label_top,
                        caption_width,
                        max(0, layout.header_height - margin[1] - margin[3]),
                    )
                    ui.component_text(
                        "timeline_ruler",
                        rect,
                        caption,
                        tuple(round(v * 255) for v in ctx.theme.text_disabled),
                        align="start",
                        translate=False,
                    )
                    if ui.design is not None:
                        ui.design.record(
                            f"timeline-ruler-tick-{tick_index}",
                            ("timeline_ruler",),
                            rect.tuple(),
                            "timeline_tick",
                            time=tick,
                            tickX=x,
                            label=caption,
                            step=step,
                        )
                    label_right = caption_right + 8 * scale

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
        endpoint_times = (take_times[selection[0]], take_times[selection[1]])
        handles = tuple(self._x(editor, layout, time) for time in endpoint_times)
        draw.rect_filled(
            (handles[0], layout.ruler_bottom - 3 * scale),
            (handles[1], layout.ruler_bottom),
            with_alpha(color, 0.7),
            rounding=scale,
        )
        for x in handles:
            draw.line(
                (x, layout.ruler_bottom - 7 * scale),
                (x, layout.ruler_bottom + 4 * scale),
                color,
                1.5 * scale,
            )
        ui = self.workspace.ui
        if ui.design is not None:
            ui.design.record(
                "timeline-loop-range",
                ("timeline_sheet",),
                (first, layout.ruler_bottom, last - first, layout.hi[1] - layout.ruler_bottom),
                "timeline_loop_range",
                endpointTimes=list(endpoint_times),
                handleXs=list(handles),
                coverageXs=[first, last],
            )

    def _take(self, draw, editor, ctx, layout, take_times):
        if not take_times:
            return
        session = ctx.session
        cells, segments = session.state_take_cells, session.state_take_segments
        scale, center, color = ctx.style_scale, layout.lane_y(1), ctx.theme.primary
        lane_style = self.workspace.ui.component_style("timeline_lane")
        font = lane_style.get("fontSize", BODY_SIZE)
        strip_height = min(
            max(0, layout.row_height - layout.lane_margin_top - layout.lane_margin_bottom),
            max(22, font, lane_style.line_height(font * 1.4)) * scale,
        )
        half = strip_height / 2
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
        show_samples = end - start <= (right - left) / (10 * scale)
        first_segment = max(0, bisect.bisect_right(segments, (start, len(take_times))) - 1)
        for slot in range(first_segment, len(segments)):
            a, b = segments[slot]
            if a >= end:
                break
            first, last = self._sample_bounds(editor, layout, take_times, (a, b), cells)
            first, last = max(left, first), min(right, last)
            if first >= last:
                continue
            rect = (first, center - half, last - first, strip_height)
            self.workspace.ui.hits[f"timeline-take-segment-{slot}"] = rect
            if "timeline-take-strip" not in self.workspace.ui.hits:
                self.workspace.ui.hits["timeline-take-strip"] = rect
            draw.rect_filled(
                (first, center - half),
                (last, center + half),
                with_alpha(color, 0.18),
                rounding=3 * scale,
            )
            draw.rect_filled(
                (first, center - half),
                (last, center - half + 2 * scale),
                with_alpha(color, 0.65),
            )
            # Only recording adjacency joins cells. Zoom changes tick density;
            # deleting frames leaves their original coverage empty.
            if show_samples:
                for index in range(max(a, start), min(b + 1, end)):
                    # Cell edges delimit sample coverage; marks identify the
                    # actual recorded timestamp used by seek and the playhead.
                    x = self._x(editor, layout, take_times[index])
                    if left <= x <= right:
                        active = index == session.state_take_cursor
                        mark_color = ctx.theme.danger if active else color
                        bottom = center + half
                        draw.line(
                            (x, bottom - (10 if active else 6) * scale),
                            (x, bottom),
                            with_alpha(mark_color, 0.9 if active else 0.55),
                            scale,
                        )
                        ui = self.workspace.ui
                        rect = (x - 3 * scale, center - half, 6 * scale, strip_height)
                        ui.hits[f"timeline-take-frame-{index}"] = rect
                        if ui.design is not None:
                            ui.design.record(
                                f"timeline-take-frame-{index}",
                                ("timeline_lane",),
                                rect,
                                "take_sample",
                                frameIndex=index,
                                time=take_times[index],
                                sampleX=x,
                                cellBounds=list(cells[index]),
                                active=active,
                            )
            if not named and last - first > 88 * scale:
                self._lane_caption(
                    1,
                    (first + 8 * scale, center - half, last - first - 16 * scale, strip_height),
                    label,
                    color,
                )
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
            rect = (first, center - half, last - first, strip_height)
            self.workspace.ui.hits[f"timeline-take-selection-{slot}"] = rect
            self.workspace.ui.hits["timeline-take-selection"] = rect
            draw.rect_filled(
                (first, center - half),
                (last, center + half),
                with_alpha(ctx.theme.warning, 0.22),
                rounding=3 * scale,
            )
            draw.rect(
                (first, center - half),
                (last, center + half),
                with_alpha(ctx.theme.warning, 0.7),
                scale,
                rounding=3 * scale,
            )

    def _take_tooltip(self, editor, ctx, layout, hit, times):
        if not imgui.is_window_hovered() or not hit.on_track or hit.lane != "take":
            return
        left, right = layout.time_bounds
        x, y = hit.position
        top = layout.ruler_bottom + layout.row_height + layout.lane_gap
        if not (left <= x <= right and top <= y <= top + layout.row_height):
            return
        time = timeline_x_to_time(x, editor.view_start, editor.view_end, left, right)
        index = nearest_take_frame(times, time)
        first, last = self._sample_bounds(
            editor, layout, times, (index, index), ctx.session.state_take_cells
        )
        if first <= x <= last:
            self.workspace.ui.tooltip(
                f"{ctx.tr('Recorded frame')} {index + 1}/{len(times)}\n{times[index]:.6f} s",
                translate=False,
            )

    def _sample_bounds(self, editor, layout, times, selection, cells):
        first, last = (
            self._x(editor, layout, time)
            for time in take_frame_bounds(times, *selection, cells=cells)
        )
        if first == last:
            return first - layout.marker_radius, last + layout.marker_radius
        return first, last

    @part_scope("timeline_key")
    def _marker(self, key, x, center, radius, icon, tint, selected):
        ui, scale = self.workspace.ui, self.workspace.s
        style = ui.component_style("timeline_key")
        margins = style.edges("margin", scale)
        box = ui.component_box(
            "timeline_key",
            x - radius - margins[0],
            center - radius - margins[1],
            2 * radius,
            2 * radius,
            key=key,
            padding=(2 * scale, 2 * scale, 2 * scale, 2 * scale),
            background=selected or style.explicit("radius"),
        )
        if selected or style.explicit("radius"):
            ui.rect(*box.frame.tuple(), (*tint[:3], 25 if selected else 12), style.get("radius", 3))
        extent = min(
            box.content.width,
            box.content.height,
            style.get("iconSize", radius * 2 / scale - 4) * scale,
        )
        icon_rect = style.align_rect(box.content, extent, extent)
        ui.icon(icon, icon_rect.x, icon_rect.y, extent / scale, tint)
        ui.hits[key] = box.frame.tuple()

    def _keys(self, draw, editor, ctx, layout, projection, hit, keyframes):
        radius, center, scale = layout.marker_radius, layout.lane_y(0), ctx.style_scale
        margins = self.workspace.ui.component_style("timeline_key").edges("margin", scale)
        ids = projection.draw_ids(
            *layout.time_bounds,
            radius * 2.5 + margins[0] + margins[2],
            (editor.selected_id, ctx.session.active_keyframe, hit.keyframe_id, editor.drag_id),
        )
        ordered = sorted(ids, key=lambda key_id: projection.positions[key_id])
        for index, key_id in enumerate(ordered):
            key, x = keyframes[key_id], projection.positions[key_id]
            selected = key_id == editor.selected_id or key_id in editor.selected_keyframes
            color = ctx.theme.warning if selected else ctx.theme.primary
            if key_id == hit.keyframe_id or selected:
                draw.circle_filled((x, center), radius + 5 * scale, with_alpha(color, 0.1))
            self._marker(
                f"timeline-key-{key_id}",
                x,
                center,
                radius,
                "diamond",
                tuple(round(value * 255) for value in color),
                selected,
            )
            next_x = (
                projection.positions[ordered[index + 1]]
                if index + 1 < len(ordered)
                else layout.hi[0]
            )
            room = next_x - x - radius * 2 - 10 * scale - margins[0] - margins[2]
            if room >= 36 * scale:
                self._lane_caption(
                    0,
                    (
                        x + radius + 6 * scale,
                        center
                        - (layout.row_height - layout.lane_margin_top - layout.lane_margin_bottom)
                        / 2,
                        room,
                        layout.row_height - layout.lane_margin_top - layout.lane_margin_bottom,
                    ),
                    key.name or ctx.tr("keyframe"),
                    color if selected else ctx.theme.text,
                )

    def _snapshots(self, draw, editor, ctx, layout, projection, hit):
        source = ctx.session.scene_snapshots
        if source is not self.snapshot_source:
            self.snapshot_source = source
            self.snapshot_by_id = {item.snapshot_id: item for item in source}
        radius, center, scale = layout.marker_radius, layout.lane_y(2), ctx.style_scale
        margins = self.workspace.ui.component_style("timeline_key").edges("margin", scale)
        ids = projection.draw_ids(
            *layout.time_bounds,
            radius * 2.5 + margins[0] + margins[2],
            (editor.selected_snapshot, hit.snapshot_id),
        )
        ordered = sorted(ids, key=lambda item_id: projection.positions[item_id])
        for index, item_id in enumerate(ordered):
            item, x = self.snapshot_by_id[item_id], projection.positions[item_id]
            selected = item_id == editor.selected_snapshot
            color = ctx.theme.info
            self._marker(
                f"timeline-snapshot-{item_id}",
                x,
                center,
                radius,
                "snapshot",
                tuple(round(value * 255) for value in color),
                selected,
            )
            next_x = (
                projection.positions[ordered[index + 1]]
                if index + 1 < len(ordered)
                else layout.hi[0]
            )
            room = next_x - x - radius * 2 - 10 * scale - margins[0] - margins[2]
            if room >= 36 * scale:
                self._lane_caption(
                    2,
                    (
                        x + radius + 6 * scale,
                        center
                        - (layout.row_height - layout.lane_margin_top - layout.lane_margin_bottom)
                        / 2,
                        room,
                        layout.row_height - layout.lane_margin_top - layout.lane_margin_bottom,
                    ),
                    item.name,
                    color if selected else ctx.theme.text_disabled,
                )

    def _lane_caption(self, index, rect, caption, color):
        ui = self.workspace.ui
        ui.component_text(
            "timeline_lane",
            rect,
            caption,
            tuple(round(v * 255) for v in color),
            size=BODY_SIZE,
            translate=False,
        )
        self._lane_text_rects[index].append(list(rect))

    def _playhead(self, draw, editor, ctx, layout):
        scale, x = ctx.style_scale, self._x(editor, layout, editor.playhead)
        if not layout.time_bounds[0] <= x <= layout.time_bounds[1]:
            return
        ui = self.workspace.ui
        rect = (x - 4 * scale, layout.lo[1], 8 * scale, layout.height)
        ui.hits["timeline-playhead"] = rect
        if ui.design is not None:
            ui.design.record(
                "timeline-playhead",
                ("timeline_sheet",),
                rect,
                "timeline_playhead",
                time=editor.playhead,
                playheadX=x,
                viewRange=[editor.view_start, editor.view_end],
                timeBounds=list(layout.time_bounds),
            )
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
