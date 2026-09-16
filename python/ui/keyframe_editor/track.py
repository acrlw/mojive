"""Timeline track layout, input sampling, and painting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from imgui_bundle import imgui

from mojive.adapters.base import KeyframeInfo
from mojive.interaction.input import InputClaim, physical_ctrl_super
from mojive.interaction.timeline import MarkerProjection, TimelineMarkerIndex
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels import PanelContext
from mojive.ui.text_layout import text_line_y

from ..input_bindings import DEFAULT_INPUT_BINDINGS
from ..pointer_bindings import PointerAction, PointerChord
from ..theme import with_alpha
from ..timeline import (
    fitted_timeline_range,
    follow_timeline_range,
    nearest_take_frame,
    nice_timeline_step,
    recorded_take_spans,
    timeline_channel_width,
    timeline_time_to_x,
    timeline_x_to_time,
    zoom_timeline_range,
)
from . import controls
from .controller import TimelineEditor, TimelineHit
from .controls import (
    _COMMAND_HEIGHT_PT,
    _LOOP_COLOR,
    _MARKER_SPACING_FACTOR,
    _format_tick,
    _rounded_command_icon_path,
    timeline_status_hints,
)
from .toolbar import draw_model_header


def draw_dope_sheet(
    editor: TimelineEditor,
    ctx: PanelContext,
    models,
    keyframes: tuple[KeyframeInfo, ...],
    keyframe_by_id: dict[int, KeyframeInfo],
    take_times: Sequence[float],
    editable: bool,
) -> None:
    scale = ctx.style_scale
    available = max(1.0, float(imgui.get_content_region_avail().x))
    ruler_height = (_COMMAND_HEIGHT_PT + 4) * scale
    tracks = 3
    height = max(
        ruler_height + tracks * 20 * scale,
        min(ruler_height + tracks * 30 * scale, imgui.get_content_region_avail().y - 8 * scale),
    )
    channel_width = timeline_channel_width(available, scale)
    lo_vec = imgui.get_cursor_screen_pos()
    lo = (float(lo_vec.x), float(lo_vec.y))
    hi = (lo[0] + available, lo[1] + height)
    time_lo = lo[0] + channel_width
    time_hi = hi[0]
    time_width = max(1.0, time_hi - time_lo)

    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(draw_list, 2)
    splitter.set_current_channel(draw_list, 1)
    imgui.set_cursor_screen_pos((lo[0] + 2 * scale, lo[1] + 2 * scale))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (5 * scale, 0))
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding,
        (8 * scale, max(0, (_COMMAND_HEIGHT_PT * scale - imgui.get_font_size()) * 0.5)),
    )
    draw_model_header(editor, ctx, models, keyframes, editable, channel_width - 4 * scale)
    imgui.pop_style_var(2)
    imgui.set_cursor_screen_pos(lo)
    splitter.set_current_channel(draw_list, 0)

    flags = (
        imgui.ButtonFlags_.mouse_button_left.value
        | imgui.ButtonFlags_.mouse_button_right.value
        | imgui.ButtonFlags_.mouse_button_middle.value
    )
    imgui.invisible_button("##keyframe-dope-sheet", imgui.ImVec2(available, height), flags)
    timeline_id = imgui.get_item_id()
    hovered = imgui.is_item_hovered()
    mouse = imgui.get_mouse_pos()
    mouse_xy = (float(mouse.x), float(mouse.y))
    over_timeline = hovered and time_lo <= mouse_xy[0] <= time_hi
    if over_timeline:
        # The dope sheet uses the wheel for zoom. Owning the wheel here
        # prevents the docked Keyframes window from scrolling as well.
        imgui.set_item_key_owner(imgui.Key.mouse_wheel_y)

    claim = ctx.input_claim or InputClaim()
    owns_escape = (
        imgui.is_window_focused()
        and not claim.claims_key("escape")
        and not ctx.popup_owned_frame
        and (
            editor.pointer_mode in ("range", "select")
            or ctx.session.state_take_range is not None
            or editor.take_selection is not None
            or editor.selected_keyframes
        )
    )
    if owns_escape:
        imgui.internal.set_key_owner(
            imgui.Key.escape, timeline_id, imgui.internal.InputFlagsPrivate_.lock_until_release
        )

    fitted = editor.view_needs_fit or editor.view_model_id != editor.model_id
    if fitted:
        editor.view_start, editor.view_end = fitted_timeline_range(
            tuple(key.time for key in keyframes)
            + tuple(take_times)
            + tuple(item.time for item in ctx.session.scene_snapshots)
            + (editor.playhead,),
            editor.playhead,
        )
        editor.view_model_id = editor.model_id
        editor.view_needs_fit = False

    io = imgui.get_io()
    bindings = ctx.input_bindings or DEFAULT_INPUT_BINDINGS
    pointer = bindings.pointer_frame(claim)
    pan_press = bindings.pointer_match(PointerAction.TIMELINE_PAN, pointer, press=True)
    range_press = bindings.pointer_match(PointerAction.TIMELINE_RANGE, pointer, press=True)
    clear_range_press = bindings.pointer_match(
        PointerAction.TIMELINE_CLEAR_RANGE, pointer, press=True
    )
    select_press = bindings.pointer_match(PointerAction.TIMELINE_SCRUB, pointer, press=True)
    additive = bool(pointer.keys & {"ctrl", "super"})
    additive_modifier = "ctrl" if "ctrl" in pointer.keys else "super"
    if additive and pointer.matches(PointerChord((0,), (additive_modifier,)), press=True):
        select_press = PointerChord((0,), (additive_modifier,))
    load_press = bindings.pointer_match(PointerAction.TIMELINE_LOAD, pointer, press=True)
    if over_timeline and (pan_press or range_press) and not editor.pointer_mode:
        editor.drag_start_x = mouse_xy[0]
        editor.pan_moved = False
        editor.pointer_chord = range_press or pan_press
        if range_press:
            if len(take_times) > 1 and not ctx.session.state_take_recording:
                editor.pointer_mode = "range"
                time = timeline_x_to_time(
                    mouse_xy[0], editor.view_start, editor.view_end, time_lo, time_hi
                )
                editor.range_anchor = nearest_take_frame(take_times, time)
                editor.range_preview = (editor.range_anchor, editor.range_anchor)
        else:
            editor.pointer_mode = "pan"

    zooming = bool(
        over_timeline
        and bindings.pointer_match(PointerAction.TIMELINE_ZOOM, pointer)
        and not editor.pointer_mode
    )
    if zooming:
        anchor = timeline_x_to_time(
            mouse_xy[0], editor.view_start, editor.view_end, time_lo, time_hi
        )
        if editor.follow_mode == "locked":
            anchor = editor.playhead
        editor.view_start, editor.view_end = zoom_timeline_range(
            editor.view_start, editor.view_end, anchor, float(io.mouse_wheel)
        )
    if (
        editor.pointer_mode == "pan"
        and pointer.held(editor.pointer_chord)
        and abs(mouse_xy[0] - editor.drag_start_x) >= float(io.mouse_drag_threshold)
    ):
        editor.pan_moved = True
        editor.follow_mode = "off"
        shift = -float(io.mouse_delta.x) * (editor.view_end - editor.view_start) / time_width
        editor.view_start += shift
        editor.view_end += shift

    if (
        not editor.pointer_mode
        and not zooming
        and not fitted
        and (
            ctx.session.state_take_playing
            or ctx.session.state_take_recording
            or editor.playhead != editor.last_followed_playhead
        )
    ):
        editor.view_start, editor.view_end = follow_timeline_range(
            editor.view_start,
            editor.view_end,
            editor.playhead,
            editor.follow_mode,
            editor.locked_fraction,
        )
    editor.last_followed_playhead = editor.playhead

    row_height = (height - ruler_height) / tracks
    marker_y = lo[1] + ruler_height + row_height * 0.5
    take_y = marker_y + row_height
    snapshot_y = take_y + row_height
    lane = (
        "model"
        if mouse_xy[1] < marker_y + row_height * 0.5
        else "take"
        if mouse_xy[1] < take_y + row_height * 0.5
        else "snapshots"
    )
    if hovered and mouse_xy[1] >= lo[1] + ruler_height and (select_press or load_press):
        editor.edit_lane = lane
    marker_radius = 7.0 * scale
    projection = editor.marker_index.project(
        editor.view_start,
        editor.view_end,
        time_lo,
        time_hi,
        marker_radius + 4.0,
        moved=(editor.drag_id, editor.drag_preview_time) if editor.drag_moved else None,
    )
    marker_positions = projection.positions
    snapshots = ctx.session.scene_snapshots
    if snapshots is not editor.snapshot_cache:
        editor.snapshot_index = TimelineMarkerIndex(
            tuple((item.snapshot_id, item.time) for item in snapshots)
        )
        editor.snapshot_cache = snapshots
    snapshot_projection = editor.snapshot_index.project(
        editor.view_start, editor.view_end, time_lo, time_hi, marker_radius + 4.0
    )
    snapshot_positions = snapshot_projection.positions
    snapshot_hit = (
        snapshot_projection.hit(mouse_xy[0], marker_radius + 4.0)
        if hovered and abs(snapshot_y - mouse_xy[1]) <= marker_radius + 4.0
        else -1
    )
    hit_id = (
        projection.hit(mouse_xy[0], marker_radius + 4.0)
        if hovered and abs(marker_y - mouse_xy[1]) <= marker_radius + 4.0
        else -1
    )

    hit = TimelineHit(
        mouse_xy,
        (time_lo, time_hi),
        lane,
        mouse_xy[1] >= lo[1] + ruler_height,
        hit_id,
        snapshot_hit,
        marker_positions,
    )
    if (
        over_timeline
        and (select_press or load_press)
        and not editor.pointer_mode
        and not ctx.take_video_active
    ):
        editor.pointer_chord = load_press or select_press
        editor.begin_timeline_edit(
            ctx,
            keyframe_by_id,
            take_times,
            hit,
            editable=editable,
            load=bool(load_press),
            additive=bool(additive),
        )

    escape = (
        owns_escape
        and not io.want_text_input
        and imgui.internal.is_key_pressed(imgui.Key.escape, 0, timeline_id)
    )
    released = editor.update_timeline_drag(
        ctx,
        keyframes,
        take_times,
        hit,
        pointer,
        editable=editable,
        clear_range=bool(over_timeline and clear_range_press),
        escape=bool(escape),
    )
    if released and editor.finish_timeline_range(ctx, hit, hovered=hovered):
        imgui.open_popup("timeline-selection-menu")

    handle_editor_keys(editor, ctx, keyframes, take_times, editable, timeline_id)
    ctx.status_hints = timeline_status_hints(
        ctx.tr,
        has_range=ctx.session.state_take_range is not None,
        edit_lane=(editor.edit_lane if mouse_xy[1] < lo[1] + ruler_height else lane)
        if hovered
        else "",
        over_ruler=mouse_xy[1] < lo[1] + ruler_height,
        has_selection=bool(
            editor.take_selection
            if editor.edit_lane == "take"
            else editor.selected_keyframes
            if editor.edit_lane == "model"
            else False
        ),
        bindings=bindings,
    )
    paint_dope_sheet(
        editor,
        ctx,
        lo,
        hi,
        time_lo,
        time_hi,
        ruler_height,
        marker_y,
        take_y,
        marker_radius,
        marker_positions,
        keyframe_by_id,
        take_times,
        hit_id,
        projection,
    )
    overlay = ctx.painter()
    imgui.push_clip_rect((time_lo, lo[1] + ruler_height), (time_hi, hi[1]), True)
    for snapshot_id in snapshot_projection.draw_ids(
        time_lo,
        time_hi,
        marker_radius * 1.5,
        (editor.selected_snapshot, snapshot_hit),
    ):
        x = snapshot_positions[snapshot_id]
        radius = marker_radius * 0.7
        controls._draw_command_icon(
            overlay,
            (x, snapshot_y),
            "key-keyframe",
            ctx.theme.info if snapshot_id != editor.selected_snapshot else ctx.theme.primary,
            radius / 6,
        )
    imgui.pop_clip_rect()
    if snapshot_hit >= 0 and hovered:
        snapshot = next(
            item for item in ctx.session.scene_snapshots if item.snapshot_id == snapshot_hit
        )
        imgui.set_tooltip(f"{snapshot.name}  {snapshot.time:g} s\n{ctx.tr('Double-click to load')}")
    elif hit_id >= 0 and hovered:
        key = keyframe_by_id[hit_id]
        imgui.set_tooltip(
            f"{key.name or ctx.tr('keyframe')}  ·  {key.time:g} s\n{ctx.tr('Double-click to load')}"
        )
    splitter.merge(draw_list)
    draw_selection_menu(editor, ctx, editable)


def handle_editor_keys(editor: TimelineEditor, ctx, keyframes, take_times, editable, timeline_id):
    io = imgui.get_io()
    claim = ctx.input_claim or InputClaim()
    ctrl, super_key = physical_ctrl_super(io)
    if (
        not imgui.is_window_focused()
        or io.want_text_input
        or ctx.popup_owned_frame
        or imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
        or claim.keyboard
        or (ctrl and claim.claims_key("ctrl"))
        or (super_key and claim.claims_key("super"))
        or (io.key_shift and claim.claims_key("shift"))
    ):
        return
    for key in ("a", "delete", "backspace"):
        if not claim.claims_key(key):
            imgui.internal.set_key_owner(
                getattr(imgui.Key, key),
                timeline_id,
                imgui.internal.InputFlagsPrivate_.lock_until_release,
            )

    def pressed(key):
        return not claim.claims_key(key) and imgui.internal.is_key_pressed(
            getattr(imgui.Key, key), 0, timeline_id
        )

    if (ctrl or super_key) and pressed("a"):
        if editor.edit_lane == "take" and take_times:
            editor.take_selection = (0, len(take_times) - 1)
        elif editor.edit_lane == "model":
            editor.selected_keyframes = {key.keyframe_id for key in keyframes}
            editor.selected_id = (
                next(iter(editor.selected_keyframes)) if len(editor.selected_keyframes) == 1 else -1
            )
    if pressed("delete") or pressed("backspace"):
        editor.delete_selection(ctx, editable)


def draw_selection_menu(editor: TimelineEditor, ctx, editable):
    if imgui.begin_popup("timeline-selection-menu"):
        enabled = (
            bool(editor.selection_count())
            and not (ctx.session.state_take_recording or ctx.take_video_active)
            and (editor.edit_lane == "take" or editable)
        )
        if imgui.menu_item(ctx.tr("Delete selection"), "Delete", False, enabled)[0]:
            editor.delete_selection(ctx, editable)
        imgui.end_popup()


def paint_dope_sheet(
    editor: TimelineEditor,
    ctx: PanelContext,
    lo: tuple[float, float],
    hi: tuple[float, float],
    time_lo: float,
    time_hi: float,
    ruler_height: float,
    marker_y: float,
    take_y: float,
    marker_radius: float,
    marker_positions: Mapping[int, float],
    keyframe_by_id: dict[int, KeyframeInfo],
    take_times: Sequence[float],
    hit_id: int,
    projection: MarkerProjection,
) -> None:
    overlay = ImguiDraw2D()
    theme = ctx.theme
    ruler_bottom = lo[1] + ruler_height
    overlay.rect_filled(lo, hi, theme.bg_child, rounding=3.0 * ctx.style_scale)
    overlay.rect_filled(lo, (time_lo, hi[1]), theme.bg_header)
    overlay.rect_filled((time_lo, lo[1]), (time_hi, ruler_bottom), theme.bg_frame)
    overlay.line((time_lo, lo[1]), (time_lo, hi[1]), theme.border, 1.0)
    overlay.line((lo[0], ruler_bottom), (hi[0], ruler_bottom), theme.border, 1.0)
    row_divider = (marker_y + take_y) * 0.5
    overlay.line((lo[0], row_divider), (hi[0], row_divider), theme.border, 1.0)

    loop = editor.range_preview or ctx.session.state_take_range
    if loop is not None:
        start_x, end_x = (
            timeline_time_to_x(
                take_times[index], editor.view_start, editor.view_end, time_lo, time_hi
            )
            for index in loop
        )
        left, right = max(time_lo, start_x), min(time_hi, end_x)
        if left <= right:
            overlay.rect_filled((left, ruler_bottom), (right, hi[1]), with_alpha(_LOOP_COLOR, 0.12))
            overlay.rect_filled((left, lo[1]), (right, ruler_bottom), with_alpha(_LOOP_COLOR, 0.3))
            for x in (start_x, end_x):
                if time_lo <= x <= time_hi:
                    overlay.line((x, lo[1]), (x, hi[1]), _LOOP_COLOR, 1.5 * ctx.style_scale)

    step = nice_timeline_step(
        editor.view_end - editor.view_start, time_hi - time_lo, 90 * ctx.style_scale
    )
    first = math.ceil(editor.view_start / step) * step
    tick = first
    iterations = 0
    while tick <= editor.view_end + step * 1e-7 and iterations < 1000:
        x = timeline_time_to_x(tick, editor.view_start, editor.view_end, time_lo, time_hi)
        overlay.line((x, ruler_bottom), (x, hi[1]), with_alpha(theme.border, 0.65), 1.0)
        label = _format_tick(tick, step)
        label_width, _ = overlay.text_size(label)
        label_x = min(
            time_hi - label_width - 3 * ctx.style_scale,
            max(time_lo + 3 * ctx.style_scale, x + 4 * ctx.style_scale),
        )
        overlay.text(
            (label_x, text_line_y(overlay, lo[1] + ruler_height * 0.5)),
            theme.text_disabled,
            label,
        )
        tick += step
        iterations += 1

    inset = 10.0 * ctx.style_scale
    label_width = max(1.0, time_lo - lo[0] - 2.0 * inset)
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect((lo[0], ruler_bottom), (time_lo, hi[1]), True)
    rows = [
        (marker_y, "Keyframes", "model"),
        (take_y, "Take", "take"),
        (take_y + (take_y - marker_y), "Snapshots", "snapshots"),
    ]
    row_height = take_y - marker_y
    for center_y, label, lane in rows:
        if editor.edit_lane == lane:
            overlay.rect_filled(
                (lo[0], center_y - row_height * 0.5),
                (time_lo, center_y + row_height * 0.5),
                with_alpha(theme.primary, 0.22),
            )
        text = ctx.tr(label)
        size = imgui.calc_text_size(text, wrap_width=label_width)
        draw_list.add_text(
            imgui.get_font(),
            imgui.get_font_size(),
            (lo[0] + inset, center_y - size.y * 0.5),
            imgui.get_color_u32(imgui.Col_.text),
            text,
            wrap_width=label_width,
        )
    draw_list.pop_clip_rect()
    draw_list.push_clip_rect((time_lo, lo[1]), (time_hi, hi[1]), True)

    playhead_x = timeline_time_to_x(
        editor.playhead, editor.view_start, editor.view_end, time_lo, time_hi
    )
    if time_lo <= playhead_x <= time_hi:
        overlay.line((playhead_x, lo[1]), (playhead_x, hi[1]), theme.danger, 1.5 * ctx.style_scale)
        overlay.convex_fill(
            tuple(
                (playhead_x + px * ctx.style_scale, lo[1] + py * ctx.style_scale)
                for px, py in _rounded_command_icon_path(
                    ((-5.0, 0.0), (5.0, 0.0), (0.0, 7.0)),
                    0.6,
                )
            ),
            theme.danger,
        )

    for keyframe_id in projection.draw_ids(
        time_lo,
        time_hi,
        marker_radius * _MARKER_SPACING_FACTOR,
        (editor.selected_id, ctx.session.active_keyframe, hit_id, editor.drag_id),
    ):
        key = keyframe_by_id[keyframe_id]
        x = marker_positions[keyframe_id]
        selected = (
            key.keyframe_id in editor.selected_keyframes or key.keyframe_id == editor.selected_id
        )
        hovered = key.keyframe_id == hit_id
        fill = (0.98, 0.67, 0.24, 1.0) if hovered else theme.warning if selected else theme.primary
        points = tuple(
            (x + px, marker_y + py)
            for px, py in _rounded_command_icon_path(
                (
                    (0.0, -marker_radius),
                    (marker_radius, 0.0),
                    (0.0, marker_radius),
                    (-marker_radius, 0.0),
                ),
                0.6 * ctx.style_scale,
            )
        )
        overlay.convex_fill(points, fill)
        overlay.polyline(points, theme.bg_window, 1.0, closed=True)
        if key.keyframe_id == ctx.session.active_keyframe:
            overlay.circle_filled((x, marker_y), 2.25 * ctx.style_scale, theme.primary_bright)

    cursor = ctx.session.state_take_cursor
    length = 8.0 * ctx.style_scale
    color = with_alpha(theme.primary, 0.72)
    for left, right in recorded_take_spans(
        take_times, editor.view_start, editor.view_end, time_lo, time_hi
    ):
        if right - left <= 2.0:
            x = (left + right) * 0.5
            overlay.line((x, take_y - length), (x, take_y + length), color, 2.0)
        else:
            overlay.rect_filled((left, take_y - length), (right, take_y + length), color)
    if editor.take_selection is not None:
        first, last = editor.take_selection
        start_x, end_x = (
            timeline_time_to_x(
                take_times[index], editor.view_start, editor.view_end, time_lo, time_hi
            )
            for index in (first, last)
        )
        overlay.rect_filled(
            (start_x - 2 * ctx.style_scale, take_y - row_height * 0.5),
            (end_x + 2 * ctx.style_scale, take_y + row_height * 0.5),
            with_alpha(theme.warning, 0.3),
        )
    if 0 <= cursor < len(take_times):
        x = timeline_time_to_x(
            take_times[cursor], editor.view_start, editor.view_end, time_lo, time_hi
        )
        if time_lo <= x <= time_hi:
            length = 13.0 * ctx.style_scale
            overlay.line((x, take_y - length), (x, take_y + length), theme.danger, 2.5)

    if editor.selected_id in marker_positions:
        key = keyframe_by_id[editor.selected_id]
        x = marker_positions[key.keyframe_id]
        if time_lo <= x <= time_hi:
            value = (
                editor.drag_preview_time
                if editor.drag_id == key.keyframe_id and editor.drag_moved
                else key.time
            )
            label = f"{key.name or ctx.tr('keyframe')}  {value:g} s"
            text_width, _ = overlay.text_size(label)
            label_x = min(time_hi - text_width - 6.0, max(time_lo + 6.0, x + 10.0))
            overlay.text((label_x, marker_y + 13.0 * ctx.style_scale), theme.warning, label)

    draw_list.pop_clip_rect()
    overlay.rect(lo, hi, theme.border, 1.0, rounding=3.0 * ctx.style_scale)


def hit_marker(
    positions: dict[int, float],
    marker_y: float,
    radius: float,
    mouse: tuple[float, float],
) -> int:
    limit = radius + 4.0
    hits = (
        (abs(x - mouse[0]) + abs(marker_y - mouse[1]), key_id)
        for key_id, x in positions.items()
        if abs(x - mouse[0]) <= limit and abs(marker_y - mouse[1]) <= limit
    )
    return min(hits, default=(math.inf, -1))[1]
