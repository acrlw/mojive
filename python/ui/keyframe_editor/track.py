"""Timeline track layout, input sampling, and painting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.adapters.base import KeyframeInfo, SceneModelInfo
from mojive.interaction.input import InputClaim, physical_ctrl_super
from mojive.interaction.timeline import MarkerProjection, TimelineMarkerIndex
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels import PanelContext
from mojive.ui.text_layout import text_line_y

from ..input_bindings import DEFAULT_INPUT_BINDINGS, InputBindings
from ..pointer_bindings import PointerAction, PointerChord, PointerFrame
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
from .controller import TimelineEditor, TimelineHit
from .controls import (
    _COMMAND_HEIGHT_PT,
    _LOOP_COLOR,
    _MARKER_SPACING_FACTOR,
    CommandIconDrawer,
    _draw_command_icon,
    _format_tick,
    _rounded_command_icon_path,
    timeline_status_hints,
)
from .toolbar import draw_model_header


@dataclass(frozen=True)
class TimelineLayout:
    """The geometry shared by track input, marker projection and painting."""

    lo: tuple[float, float]
    width: float
    height: float
    ruler_height: float
    channel_width: float
    marker_radius: float

    @classmethod
    def measure(
        cls, origin: Sequence[float], available: Sequence[float], scale: float
    ) -> TimelineLayout:
        ruler = (_COMMAND_HEIGHT_PT + 4) * scale
        width = max(1.0, float(available[0]))
        height = max(ruler + 60 * scale, min(ruler + 90 * scale, available[1] - 8 * scale))
        return cls(
            tuple(origin), width, height, ruler, timeline_channel_width(width, scale), 7 * scale
        )

    @property
    def hi(self) -> tuple[float, float]:
        return self.lo[0] + self.width, self.lo[1] + self.height

    @property
    def time_bounds(self) -> tuple[float, float]:
        return self.lo[0] + self.channel_width, self.hi[0]

    @property
    def ruler_bottom(self) -> float:
        return self.lo[1] + self.ruler_height

    @property
    def row_height(self) -> float:
        return (self.height - self.ruler_height) / 3

    def lane_y(self, index: int) -> float:
        return self.ruler_bottom + self.row_height * (index + 0.5)

    def lane_at(self, y: float) -> str:
        return (
            "model"
            if y < self.lane_y(0) + self.row_height * 0.5
            else ("take" if y < self.lane_y(1) + self.row_height * 0.5 else "snapshots")
        )


@dataclass(frozen=True)
class TimelineInput:
    """One sampled ImGui frame; gesture updates below do not query input again."""

    position: tuple[float, float]
    hovered: bool
    over_timeline: bool
    timeline_id: int
    pointer: PointerFrame
    bindings: InputBindings
    pan: PointerChord | None
    range: PointerChord | None
    select: PointerChord | None
    load: PointerChord | None
    clear_range: bool
    escape: bool
    wheel: float
    delta_x: float
    drag_threshold: float


def _sample_timeline_input(
    editor: TimelineEditor, ctx: PanelContext, layout: TimelineLayout
) -> TimelineInput:
    flags = (
        imgui.ButtonFlags_.mouse_button_left.value
        | imgui.ButtonFlags_.mouse_button_right.value
        | imgui.ButtonFlags_.mouse_button_middle.value
    )
    imgui.invisible_button(
        "##keyframe-dope-sheet", imgui.ImVec2(layout.width, layout.height), flags
    )
    timeline_id = imgui.get_item_id()
    hovered = imgui.is_item_hovered()
    mouse = imgui.get_mouse_pos()
    position = (float(mouse.x), float(mouse.y))
    over_timeline = hovered and layout.time_bounds[0] <= mouse.x <= layout.time_bounds[1]
    if over_timeline:
        # Track zoom owns the wheel instead of scrolling its docked parent.
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
    io = imgui.get_io()
    bindings = ctx.input_bindings or DEFAULT_INPUT_BINDINGS
    pointer = bindings.pointer_frame(claim)

    def press(action):
        return bindings.pointer_match(action, pointer, press=True)

    select = press(PointerAction.TIMELINE_SCRUB)
    modifier = "ctrl" if "ctrl" in pointer.keys else "super"
    additive = PointerChord((0,), (modifier,))
    if pointer.keys & {"ctrl", "super"} and pointer.matches(additive, press=True):
        select = additive
    return TimelineInput(
        position,
        hovered,
        over_timeline,
        timeline_id,
        pointer,
        bindings,
        press(PointerAction.TIMELINE_PAN),
        press(PointerAction.TIMELINE_RANGE),
        select,
        press(PointerAction.TIMELINE_LOAD),
        bool(press(PointerAction.TIMELINE_CLEAR_RANGE)),
        bool(
            owns_escape
            and not io.want_text_input
            and imgui.internal.is_key_pressed(imgui.Key.escape, 0, timeline_id)
        ),
        float(io.mouse_wheel),
        float(io.mouse_delta.x),
        float(io.mouse_drag_threshold),
    )


def _update_timeline_view(
    editor: TimelineEditor,
    ctx: PanelContext,
    layout: TimelineLayout,
    inputs: TimelineInput,
    keyframes: tuple[KeyframeInfo, ...],
    take_times: Sequence[float],
) -> None:
    time_lo, time_hi = layout.time_bounds
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
    if inputs.over_timeline and (inputs.pan or inputs.range) and not editor.pointer_mode:
        editor.drag_start_x = inputs.position[0]
        editor.pan_moved = False
        editor.pointer_chord = inputs.range or inputs.pan
        if inputs.range:
            if len(take_times) > 1 and not ctx.session.state_take_recording:
                editor.pointer_mode = "range"
                time = timeline_x_to_time(
                    inputs.position[0], editor.view_start, editor.view_end, time_lo, time_hi
                )
                editor.range_anchor = nearest_take_frame(take_times, time)
                editor.range_preview = (editor.range_anchor, editor.range_anchor)
        else:
            editor.pointer_mode = "pan"
    zooming = bool(
        inputs.over_timeline
        and not editor.pointer_mode
        and inputs.bindings.pointer_match(PointerAction.TIMELINE_ZOOM, inputs.pointer)
    )
    if zooming:
        anchor = timeline_x_to_time(
            inputs.position[0], editor.view_start, editor.view_end, time_lo, time_hi
        )
        if editor.follow_mode == "locked":
            anchor = editor.playhead
        editor.view_start, editor.view_end = zoom_timeline_range(
            editor.view_start, editor.view_end, anchor, inputs.wheel
        )
    if (
        editor.pointer_mode == "pan"
        and inputs.pointer.held(editor.pointer_chord)
        and abs(inputs.position[0] - editor.drag_start_x) >= inputs.drag_threshold
    ):
        editor.pan_moved = True
        editor.follow_mode = "off"
        shift = -inputs.delta_x * (editor.view_end - editor.view_start) / max(1, time_hi - time_lo)
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


def _project_timeline(
    editor: TimelineEditor, ctx: PanelContext, layout: TimelineLayout, inputs: TimelineInput
) -> tuple[MarkerProjection, MarkerProjection, TimelineHit]:
    time_lo, time_hi = layout.time_bounds
    radius = layout.marker_radius + 4
    projection = editor.marker_index.project(
        editor.view_start,
        editor.view_end,
        time_lo,
        time_hi,
        radius,
        moved=(editor.drag_id, editor.drag_preview_time) if editor.drag_moved else None,
    )
    snapshots = ctx.session.scene_snapshots
    if snapshots is not editor.snapshot_cache:
        editor.snapshot_index = TimelineMarkerIndex(
            tuple((item.snapshot_id, item.time) for item in snapshots)
        )
        editor.snapshot_cache = snapshots
    snapshot_projection = editor.snapshot_index.project(
        editor.view_start, editor.view_end, time_lo, time_hi, radius
    )
    x, y = inputs.position
    hit = TimelineHit(
        inputs.position,
        layout.time_bounds,
        layout.lane_at(y),
        y >= layout.ruler_bottom,
        projection.hit(x, radius) if inputs.hovered and abs(layout.lane_y(0) - y) <= radius else -1,
        snapshot_projection.hit(x, radius)
        if inputs.hovered and abs(layout.lane_y(2) - y) <= radius
        else -1,
        projection.positions,
    )
    return projection, snapshot_projection, hit


def _edit_timeline(
    editor: TimelineEditor,
    ctx: PanelContext,
    inputs: TimelineInput,
    hit: TimelineHit,
    keyframes: tuple[KeyframeInfo, ...],
    keyframe_by_id: dict[int, KeyframeInfo],
    take_times: Sequence[float],
    editable: bool,
) -> None:
    if inputs.hovered and hit.on_track and (inputs.select or inputs.load):
        editor.edit_lane = hit.lane
    if (
        inputs.over_timeline
        and (inputs.select or inputs.load)
        and not editor.pointer_mode
        and not ctx.take_video_active
    ):
        editor.pointer_chord = inputs.load or inputs.select
        editor.begin_timeline_edit(
            ctx,
            keyframe_by_id,
            take_times,
            hit,
            editable=editable,
            load=bool(inputs.load),
            additive=bool(inputs.pointer.keys & {"ctrl", "super"}),
        )
    released = editor.update_timeline_drag(
        ctx,
        keyframes,
        take_times,
        hit,
        inputs.pointer,
        editable=editable,
        clear_range=inputs.over_timeline and inputs.clear_range,
        escape=inputs.escape,
    )
    if released and editor.finish_timeline_range(ctx, hit, hovered=inputs.hovered):
        imgui.open_popup("timeline-selection-menu")
    handle_editor_keys(editor, ctx, keyframes, take_times, editable, inputs.timeline_id)
    ctx.status_hints = timeline_status_hints(
        ctx.tr,
        has_range=ctx.session.state_take_range is not None,
        edit_lane=(hit.lane if hit.on_track else editor.edit_lane) if inputs.hovered else "",
        over_ruler=not hit.on_track,
        has_selection=bool(
            editor.take_selection
            if editor.edit_lane == "take"
            else editor.selected_keyframes
            if editor.edit_lane == "model"
            else False
        ),
        bindings=inputs.bindings,
    )


def _paint_snapshot_track(
    editor: TimelineEditor,
    ctx: PanelContext,
    layout: TimelineLayout,
    projection: MarkerProjection,
    hit: TimelineHit,
    keyframes: dict[int, KeyframeInfo],
    icon_drawer: CommandIconDrawer,
) -> None:
    time_lo, time_hi = layout.time_bounds
    radius = layout.marker_radius
    overlay = ctx.painter()
    imgui.push_clip_rect((time_lo, layout.ruler_bottom), (time_hi, layout.hi[1]), True)
    for snapshot_id in projection.draw_ids(
        time_lo, time_hi, radius * 1.5, (editor.selected_snapshot, hit.snapshot_id)
    ):
        icon_drawer(
            overlay,
            (projection.positions[snapshot_id], layout.lane_y(2)),
            "key-keyframe",
            ctx.theme.info if snapshot_id != editor.selected_snapshot else ctx.theme.primary,
            radius * 0.7 / 6,
        )
    imgui.pop_clip_rect()
    if hit.snapshot_id >= 0:
        snapshot = next(
            item for item in ctx.session.scene_snapshots if item.snapshot_id == hit.snapshot_id
        )
        imgui.set_tooltip(f"{snapshot.name}  {snapshot.time:g} s\n{ctx.tr('Double-click to load')}")
    elif hit.keyframe_id >= 0:
        key = keyframes[hit.keyframe_id]
        imgui.set_tooltip(
            f"{key.name or ctx.tr('keyframe')}  ·  {key.time:g} s\n{ctx.tr('Double-click to load')}"
        )


def draw_dope_sheet(
    editor: TimelineEditor,
    ctx: PanelContext,
    models: tuple[SceneModelInfo, ...],
    take_times: Sequence[float],
    editable: bool,
    *,
    icon_drawer: CommandIconDrawer = _draw_command_icon,
) -> None:
    layout = TimelineLayout.measure(
        imgui.get_cursor_screen_pos(), imgui.get_content_region_avail(), ctx.style_scale
    )
    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(draw_list, 2)
    splitter.set_current_channel(draw_list, 1)
    imgui.set_cursor_screen_pos(
        (layout.lo[0] + 2 * ctx.style_scale, layout.lo[1] + 2 * ctx.style_scale)
    )
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (5 * ctx.style_scale, 0))
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding,
        (
            8 * ctx.style_scale,
            max(0, (_COMMAND_HEIGHT_PT * ctx.style_scale - imgui.get_font_size()) * 0.5),
        ),
    )
    draw_model_header(
        editor,
        ctx,
        models,
        editable,
        layout.channel_width - 4 * ctx.style_scale,
        icon_drawer=icon_drawer,
    )
    imgui.pop_style_var(2)
    # The header can change the model. Query its keys only after that choice, so
    # fitting, hit testing and paint all use the same model in this frame.
    keyframes, keyframe_by_id = editor.keyframes(ctx)
    imgui.set_cursor_screen_pos(layout.lo)
    splitter.set_current_channel(draw_list, 0)
    inputs = _sample_timeline_input(editor, ctx, layout)
    _update_timeline_view(editor, ctx, layout, inputs, keyframes, take_times)
    projection, snapshots, hit = _project_timeline(editor, ctx, layout, inputs)
    _edit_timeline(editor, ctx, inputs, hit, keyframes, keyframe_by_id, take_times, editable)
    paint_dope_sheet(
        editor,
        ctx,
        layout.lo,
        layout.hi,
        *layout.time_bounds,
        layout.ruler_height,
        layout.lane_y(0),
        layout.lane_y(1),
        layout.marker_radius,
        projection.positions,
        keyframe_by_id,
        take_times,
        hit.keyframe_id,
        projection,
    )
    _paint_snapshot_track(editor, ctx, layout, snapshots, hit, keyframe_by_id, icon_drawer)
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
