"""The native reference timeline edits and replays actual Session data at 2×."""

from dataclasses import asdict
from math import ceil, floor

import numpy as np
import pytest
from imgui_bundle import imgui
from tests.gpu.test_ui_design_runtime import runtime as runtime
from tests.gpu.test_ui_design_runtime import wait_loading

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.ui.timeline import take_frame_bounds, timeline_time_to_x

pytestmark = pytest.mark.gpu


def _click(study, point):
    study.event("add_mouse_pos_event", *point)
    study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_button_event", 0, False)


def _point(study, time, lane="ruler"):
    _reveal(study, lambda: _project(study, time, lane))
    return _project(study, time, lane)


def _project(study, time, lane):
    layout = study.timeline_panel.last_layout
    assert layout is not None
    editor = study.timeline_panel.editor
    left, right = layout.time_bounds
    xx = timeline_time_to_x(time, editor.view_start, editor.view_end, left, right)
    yy = (
        layout.lo[1] + layout.ruler_height / 2
        if lane == "ruler"
        else layout.lane_y({"model": 0, "take": 1, "snapshots": 2}[lane])
    )
    return min(floor(right - 1), max(ceil(left + 1), round(xx))), round(yy)


def _body(study):
    parent = imgui.internal.find_window_by_name("Timeline")
    return next(
        window
        for window in imgui.get_current_context().windows
        if window.parent_window is not None
        and window.parent_window.id_ == parent.id_
        and "Timeline body" in window.name
    )


def _reveal(study, point):
    for _ in range(24):
        child, target = _body(study), point()
        clip = child.inner_clip_rect
        if clip.min.y + study.s <= target[1] <= clip.max.y - study.s:
            return
        study.event(
            "add_mouse_pos_event",
            child.pos.x + child.size.x - imgui.get_style().scrollbar_size / 2,
            (clip.min.y + clip.max.y) / 2,
        )
        study.event("add_mouse_wheel_event", 0, -1 if target[1] > clip.max.y else 1)
        study.frame(2)
    raise AssertionError("Timeline control could not be scrolled into view")


def _action(study, key):
    assert key in study.ui.hits, key
    primary = {"collapse-timeline", "timeline-settings", "timeline-play-pause", "timeline-time"}
    if study.timeline_panel.panel.toolbar._inline:
        primary.update(study.ui.hits)
    steps = study.timeline_panel.panel.toolbar._visible_transport_steps
    if steps >= 2:
        primary.update(("timeline-previous", "timeline-next"))
    if steps == 4:
        primary.update(("timeline-first", "timeline-last"))

    def center():
        x, yy, w, hh = study.ui.hits[key]
        return x + w / 2, yy + hh / 2

    if key not in primary and not imgui.get_current_context().open_popup_stack:
        _reveal(study, center)
    study.click_hit(key)
    study.frame(3)


def _assert_timeline_layout(study):
    _reveal(study, lambda: _project(study, study.timeline_panel.editor.playhead, "take"))
    layout = study.timeline_panel.last_layout
    assert layout is not None
    np.testing.assert_allclose(
        study.ui.hits["timeline-sheet"], (*layout.lo, layout.width, layout.height)
    )
    window = imgui.internal.find_window_by_name("Timeline")
    assert window is not None
    clip = window.inner_clip_rect
    assert clip.min.x - 1 <= layout.lo[0] < layout.hi[0] <= clip.max.x + 1
    assert _body(study).scroll_max.x == 0
    left, right = layout.time_bounds
    assert left - (layout.lo[0] + layout.channel_width) >= layout.marker_radius
    assert layout.hi[0] - right >= layout.marker_radius
    centers = [layout.lane_y(index) for index in range(3)]
    assert layout.ruler_bottom < centers[0] < centers[1] < centers[2] < layout.hi[1]
    x, y, w, h = study.ui.hits["timeline-toolbar"]
    assert clip.min.x - 1 <= x < x + w <= clip.max.x + 1
    assert clip.min.y - 1 <= y < y + h <= clip.max.y + 1
    assert _body(study).inner_clip_rect.min.y >= y + h - 1
    actions = (
        "timeline-first",
        "timeline-previous",
        "timeline-play-pause",
        "timeline-next",
        "timeline-last",
        "timeline-time",
        "timeline-record",
        "timeline-take",
        "timeline-video",
        "timeline-snapshot",
        "timeline-fit",
        "timeline-loop",
        "timeline-settings",
        "timeline-follow-0",
        "timeline-follow-1",
        "timeline-follow-2",
        "##timeline-range-start",
        "##timeline-range-end",
        "collapse-timeline",
    )
    tools = study.ui.hits["timeline-tools"]
    steps = study.timeline_panel.panel.toolbar._visible_transport_steps
    primary = {"timeline-play-pause", "timeline-time", "timeline-settings", "collapse-timeline"}
    if study.timeline_panel.panel.toolbar._inline:
        primary.update(actions)
    if steps >= 2:
        primary.update(("timeline-previous", "timeline-next"))
    if steps == 4:
        primary.update(("timeline-first", "timeline-last"))
    visible = []
    for key in actions:
        if key not in study.ui.hits:
            continue
        ax, ay, aw, ah = study.ui.hits[key]
        assert aw > 0 and ah > 0, key
        assert x - 1 <= ax < ax + aw <= x + w + 1, key
        area = (x, y, w, h) if key in primary else tools
        assert area[1] - 1 <= ay < ay + ah <= area[1] + area[3] + 1, key
        item_clip = clip if key in primary else _body(study).inner_clip_rect
        if not (item_clip.min.y <= ay and ay + ah <= item_clip.max.y):
            continue
        for other, (bx, by, bw, bh) in visible:
            assert min(ax + aw, bx + bw) <= max(ax, bx) or min(ay + ah, by + bh) <= max(ay, by), (
                key,
                other,
            )
        visible.append((key, (ax, ay, aw, ah)))
    for key in (
        "timeline-play-pause",
        "timeline-time",
        "timeline-record",
        "timeline-take",
        "timeline-loop",
        "timeline-fit",
        "timeline-settings",
        "collapse-timeline",
    ):
        assert key in study.ui.hits, key


def _drag(study, start, end, *, button=0, shift=False, cancel=False):
    study.event("add_mouse_pos_event", *start)
    if shift:
        study.event("add_key_event", imgui.Key.mod_shift, True)
    study.event("add_mouse_button_event", button, True)
    for fraction in (0.25, 0.5, 0.75, 1):
        study.event(
            "add_mouse_pos_event",
            start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction,
        )
    if cancel:
        study.event("add_key_event", imgui.Key.escape, True)
        study.event("add_key_event", imgui.Key.escape, False)
    study.event("add_mouse_button_event", button, False)
    if shift:
        study.event("add_key_event", imgui.Key.mod_shift, False)
    study.frame(3)


def _key(session, model_id, name, time):
    result = session.submit(cmd.AddModelKeyframe(model_id, name))
    assert result.ok, result.message
    properties = session.keyframe_properties(result.entity_id)
    result = session.submit(cmd.SetModelKeyframe(**{**asdict(properties), "time": time}))
    assert result.ok, result.message
    return next(key for key in session.keyframes if key.name == name).keyframe_id


@pytest.fixture
def timeline(runtime):
    study, session = runtime, runtime.preview.session
    model_id = session.scene_models[0].model_id
    hinge = next(joint for joint in session.joints if joint.name == "hinge_limited")
    assert session.submit(cmd.SetQpos(hinge.qpos_adr, 0.0)).ok
    home = _key(session, model_id, "Home", 0.0)
    assert session.submit(cmd.SetQpos(hinge.qpos_adr, 0.4)).ok
    reach = _key(session, model_id, "Reach", 2.0)
    # Generate deterministic real take frames before restoring the interactive worker.
    session.set_threaded_physics(False)
    assert session.submit(cmd.StartStateTakeRecording()).ok
    for _ in range(90):
        session.tick(FrameNeeds.none(), wall_dt=1 / 30)
    assert session.submit(cmd.StopStateTakeRecording()).ok
    session.set_threaded_physics(True)
    assert session.submit(cmd.SeekStateTake(0)).ok
    result = session.submit(cmd.CaptureSceneSnapshot())
    assert result.ok, result.message
    study.activate("Timeline")
    study.frame(6)
    _action(study, "timeline-fit")
    study.frame(3)
    editor = study.timeline_panel.editor
    editor.set_follow_mode("off")
    assert editor is study.preview.panels.get("keyframes").editor
    study.timeline_ids = (home, reach, result.entity_id)
    return study


def test_real_lanes_selection_loading_and_playback_guard(timeline):
    study, session = timeline, timeline.preview.session
    home, reach, snapshot = study.timeline_ids
    assert len(session.keyframes) == 2 and len(session.state_take_times) > 80
    assert len(session.scene_snapshots) == 1
    _assert_timeline_layout(study)
    sheet = study.ui.hits["timeline-sheet"]
    window = imgui.internal.find_window_by_name("Timeline")
    assert sheet[2] > 200 * study.s
    assert sheet[1] + sheet[3] <= window.pos.y + window.size.y - 4 * study.s
    study.capture("timeline-real-three-lanes")
    point = _point(study, 0, "model")
    _click(study, point)
    assert study.timeline_panel.editor.selected_id == home
    study.frame(24)
    study.capture("timeline-selected-key")
    point = _point(study, 2, "model")
    _click(study, point)
    _click(study, point)
    assert session.active_keyframe == reach
    hinge = next(joint for joint in session.joints if joint.name == "hinge_limited")
    assert session.frame.qpos[hinge.qpos_adr] == pytest.approx(0.4)
    study.frame(24)
    assert session.submit(cmd.Play()).ok
    point = _point(study, 0, "model")
    _click(study, point)
    _click(study, point)
    assert session.active_keyframe == reach
    assert session.submit(cmd.Pause()).ok
    study.frame(24)
    point = _point(study, 0, "model")
    _click(study, point)
    _click(study, point)
    assert session.active_keyframe == home
    assert session.frame.qpos[hinge.qpos_adr] == pytest.approx(0)
    study.frame(24)
    snapshot_time = session.scene_snapshots[0].time
    _click(study, _point(study, snapshot_time, "snapshots"))
    assert study.timeline_panel.editor.selected_snapshot == snapshot
    study.capture("timeline-selected-snapshot")


def test_key_drag_cancel_undo_and_take_scrub_range_zoom_pan(timeline):
    study, session = timeline, timeline.preview.session
    editor = study.timeline_panel.editor
    _, reach, _ = study.timeline_ids
    _drag(study, _point(study, 2, "model"), _point(study, 1.5, "model"))
    wait_loading(study)
    assert session.keyframe_properties(reach).time == pytest.approx(1.5, abs=0.035)
    moved_time = session.keyframe_properties(reach).time
    study.frame(24)
    _drag(study, _point(study, moved_time, "model"), _point(study, 0.8, "model"), cancel=True)
    assert session.keyframe_properties(reach).time == moved_time
    assert session.submit(cmd.Undo()).ok
    study.frame(3)
    assert session.keyframe_properties(reach).time == pytest.approx(2)
    _drag(study, _point(study, 0.5), _point(study, 1.2))
    assert session.state_take_times[session.state_take_cursor] == pytest.approx(1.2, abs=0.05)
    _drag(study, _point(study, 0.4, "take"), _point(study, 1.6, "take"), shift=True)
    assert session.state_take_range is not None
    first, last = session.state_take_range
    assert session.state_take_times[first] == pytest.approx(0.4, abs=0.06)
    assert session.state_take_times[last] == pytest.approx(1.6, abs=0.06)
    loop_before = session.state_take_loop_enabled
    _action(study, "timeline-loop")
    assert session.state_take_loop_enabled is not loop_before
    before = editor.view_end - editor.view_start
    study.event("add_mouse_pos_event", *_point(study, 1))
    study.event("add_mouse_wheel_event", 0, 2)
    assert editor.view_end - editor.view_start < before
    before = editor.view_start
    start = _point(study, 1)
    _drag(study, start, (start[0] + 32 * study.s, start[1]), button=1)
    assert not np.isclose(editor.view_start, before)
    _action(study, "timeline-fit")
    assert editor.view_start <= 0 and editor.view_end >= 2
    study.capture("timeline-edited-loop-range")


def test_toolbar_capture_record_transport_and_main_controls(timeline):
    study, session = timeline, timeline.preview.session
    snapshots = len(session.scene_snapshots)
    _action(study, "timeline-snapshot")
    assert len(session.scene_snapshots) == snapshots + 1
    assert study.timeline_panel.editor.selected_snapshot == session.scene_snapshots[-1].snapshot_id
    _action(study, "timeline-next")
    assert session.state_take_cursor == 1
    _action(study, "timeline-last")
    assert session.state_take_cursor == len(session.state_take_times) - 1
    _action(study, "timeline-first")
    assert session.state_take_cursor == 0
    _action(study, "timeline-play-pause")
    assert session.state_take_playing
    _action(study, "timeline-play-pause")
    assert not session.state_take_playing
    _action(study, "timeline-record")
    assert session.state_take_recording and not session.paused
    study.frame(8)
    _action(study, "timeline-record")
    assert not session.state_take_recording and session.paused
    assert study.timeline_panel.editor.take_selection is not None
    _action(study, "timeline-follow-2")
    assert study.timeline_panel.editor.follow_mode == "locked"
    before = session.state_take_pause_at_end
    _action(study, "timeline-settings")
    _action(study, "timeline-pause-at-end")
    assert session.state_take_pause_at_end is not before
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    assert not imgui.get_current_context().open_popup_stack
    study.capture("timeline-main-take-range-follow")


@pytest.mark.parametrize("runtime", ((2880, 1400),), indirect=True)
def test_recorded_clip_stays_continuous_and_single_frame_selection_matches_its_cell(
    timeline, monkeypatch
):
    study, session = timeline, timeline.preview.session
    panel = study.panel

    def floating_panel(name, ctx):
        if name == "Timeline":
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            imgui.set_next_window_pos((48 * study.s, 320 * study.s), imgui.Cond_.always)
            imgui.set_next_window_size((1280 * study.s, 260 * study.s), imgui.Cond_.always)
        return panel(name, ctx)

    monkeypatch.setattr(study, "panel", floating_panel)
    study.frame(6)
    editor = study.timeline_panel.editor
    editor.take_selection = None
    times = session.state_take_times
    for span, name in ((0.08, "frames"), (0.8, "medium"), (6.0, "dense")):
        editor.view_start, editor.view_end, editor.view_needs_fit = 0, span, False
        study.frame(3)
        strip = study.ui.hits["timeline-take-strip"]
        assert strip[2] > 0
        assert study.ui.hits["timeline-record"][1] == study.ui.hits["timeline-play-pause"][1]
        study.capture("timeline-continuous-" + name)
    index = len(times) // 2
    dt = times[index + 1] - times[index]
    editor.view_start, editor.view_end = times[index] - dt * 4, times[index] + dt * 4
    study.frame(3)
    _drag(
        study,
        _point(study, times[index] - dt * 0.2, "take"),
        _point(study, times[index] + dt * 0.2, "take"),
    )
    assert editor.take_selection == (index, index)
    layout = study.timeline_panel.last_layout
    left, right = (
        timeline_time_to_x(time, editor.view_start, editor.view_end, *layout.time_bounds)
        for time in take_frame_bounds(times, index, index)
    )
    selection = study.ui.hits["timeline-take-selection"]
    assert selection[0] == pytest.approx(left)
    assert selection[2] == pytest.approx(right - left)
    study.capture("timeline-single-frame-cell")
    original_cells = tuple(session.state_take_cells)
    original_times = tuple(times)
    study.event("add_key_event", imgui.Key.delete, True)
    study.event("add_key_event", imgui.Key.delete, False)
    study.frame(3)
    assert tuple(times) == original_times[:index] + original_times[index + 1 :]
    assert editor.take_selection is None
    assert len(session.state_take_segments) == 2
    layout = study.timeline_panel.last_layout

    def project(time):
        return timeline_time_to_x(time, editor.view_start, editor.view_end, *layout.time_bounds)

    gap = (project(original_cells[index][0]), project(original_cells[index][1]))
    strips = [study.ui.hits[f"timeline-take-segment-{slot}"] for slot in range(2)]
    assert strips[0][0] + strips[0][2] == pytest.approx(gap[0])
    assert strips[1][0] == pytest.approx(gap[1])
    pixels = np.asarray(study.draw(readback=True))[::-1].copy()
    yy = round(layout.lane_y(1) + 5 * study.s)
    empty = pixels[yy, round(sum(gap) / 2), :3].astype(float)
    filled = pixels[yy, round(gap[0] - (gap[1] - gap[0]) / 2), :3].astype(float)
    assert np.linalg.norm(filled - empty) > 8
    study.capture("timeline-deleted-single-frame")
    study.event("add_key_event", imgui.Key.mod_ctrl, True)
    study.event("add_key_event", imgui.Key.a, True)
    study.event("add_key_event", imgui.Key.a, False)
    study.event("add_key_event", imgui.Key.mod_ctrl, False)
    assert editor.take_selection == (0, len(times) - 1)
    assert "timeline-take-selection-0" in study.ui.hits
    assert "timeline-take-selection-1" in study.ui.hits
    pixels = np.asarray(study.draw(readback=True))[::-1].copy()
    np.testing.assert_allclose(pixels[yy, round(sum(gap) / 2), :3], empty, atol=1)
    study.capture("timeline-selection-across-deletion")
    editor.take_selection = None
    for span, name in ((0.8, "medium"), (6.0, "dense")):
        editor.view_start, editor.view_end = 0, span
        study.frame(3)
        assert len(session.state_take_segments) == 2
        study.capture("timeline-deleted-" + name)


@pytest.mark.parametrize("width", (230, 360, 600))
def test_responsive_toolbar_and_track_keep_actions_visible_while_properties_scroll(
    timeline, monkeypatch, width
):
    study, session = timeline, timeline.preview.session
    panel = study.panel
    height = 300

    def floating_panel(name, ctx):
        if name == "Timeline":
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            imgui.set_next_window_pos((48 * study.s, 72 * study.s), imgui.Cond_.always)
            imgui.set_next_window_size((width * study.s, height * study.s), imgui.Cond_.always)
        return panel(name, ctx)

    monkeypatch.setattr(study, "panel", floating_panel)
    study.frame(6)
    _assert_timeline_layout(study)
    _click(study, _point(study, 2, "model"))
    study.frame(3)
    toolbar = study.ui.hits["timeline-toolbar"]
    parent = imgui.internal.find_window_by_name("Timeline")
    child = next(
        window
        for window in imgui.get_current_context().windows
        if window.parent_window is not None
        and window.parent_window.id_ == parent.id_
        and "Timeline body" in window.name
    )
    layout = study.timeline_panel.last_layout
    detail_height = study.ui.hits["timeline-detail-bar"][3]
    height -= (child.size.y - layout.height - detail_height / 2) / study.s
    study.frame(3)
    # Tracks become denser before they scroll; keep shrinking until the
    # selected details require scrolling without hiding any complete lane.
    for _ in range(4):
        if child.scroll_max.y > 0:
            break
        height -= 10
        study.frame(3)
    _assert_timeline_layout(study)
    assert child.scroll_max.y > 0
    clip = child.inner_clip_rect
    study.event("add_mouse_pos_event", clip.min.x + 5 * study.s, clip.min.y + 5 * study.s)
    study.event("add_mouse_wheel_event", 0, -5)
    study.frame(3)
    assert child.scroll.y > 0
    assert parent.scroll.y == pytest.approx(0)
    np.testing.assert_allclose(study.ui.hits["timeline-toolbar"], toolbar)
    snapshots = len(session.scene_snapshots)
    _action(study, "timeline-snapshot")
    assert len(session.scene_snapshots) == snapshots + 1
    study.capture(f"timeline-responsive-{width}")
    study.click_hit("collapse-timeline")
    assert not study.open["Timeline"]


def test_model_lane_add_key_uses_worker_and_single_undo(timeline):
    study, session = timeline, timeline.preview.session
    before = tuple((key.name, key.time) for key in session.keyframes)
    pose = session.frame.qpos.copy()
    _action(study, "timeline-add-key")
    wait_loading(study)
    assert len(session.keyframes) == len(before) + 1
    key = next(key for key in session.keyframes if (key.name, key.time) not in before)
    np.testing.assert_allclose(session.keyframe_properties(key.keyframe_id).qpos, pose)
    assert study.timeline_panel.editor.selected_id == key.keyframe_id
    assert session.submit(cmd.Undo()).ok
    study.frame(3)
    assert tuple((key.name, key.time) for key in session.keyframes) == before
