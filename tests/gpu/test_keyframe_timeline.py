"""Ruler and loop gestures restore real take poses and leave camera navigation alone."""

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import ViewerConfig, build
from mojive import commands as cmd
from mojive.assets import resolve
from mojive.tools.keyframe_timeline import (
    choose_follow,
    drag,
    populate_take,
    show_timeline,
    timeline_point,
)
from mojive.tools.ui_runtime import _click, _item_center
from mojive.ui.panels.keyframes import nearest_take_frame

pytestmark = pytest.mark.gpu


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    with build(
        resolve("joint_types"),
        paused=True,
        config=ViewerConfig(threaded_physics=False),
        vsync=False,
        width=1600,
        height=1000,
    ) as viewer:
        populate_take(viewer, 900)
        show_timeline(viewer)
        panel = viewer.panels.get("Keyframes")
        panel._view_start, panel._view_end = 0, 30
        panel._view_needs_fit = False
        panel._set_follow_mode("off")
        yield viewer


def test_ruler_click_and_drag_seek_recorded_pose_and_stay_at_the_chosen_frame(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    _click(viewer, timeline_point(viewer, 12))
    expected = nearest_take_frame(session.state_take_times, 12)
    for _ in range(5):
        viewer.sync()
    assert session.state_take_cursor == expected
    assert panel._playhead == session.state_take_times[expected]
    assert session.adapter.data.time == pytest.approx(session.state_take_times[expected])
    drag(viewer, timeline_point(viewer, 8), timeline_point(viewer, 17))
    expected = nearest_take_frame(session.state_take_times, 17)
    assert session.state_take_cursor == expected
    assert not session.state_take_playing
    assert session.submit(cmd.PlayStateTake())
    drag(viewer, timeline_point(viewer, 4), timeline_point(viewer, 7))
    assert session.state_take_playing
    assert 7 - 1e-9 <= session.adapter.data.time < 7.5


def test_shift_right_range_drag_cancel_and_clear_do_not_pan_or_change_camera(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    before = session.camera
    view_range = panel._view_start, panel._view_end
    drag(viewer, timeline_point(viewer, 18), timeline_point(viewer, 9), button=1, shift=True)
    expected = (
        nearest_take_frame(session.state_take_times, 9),
        nearest_take_frame(session.state_take_times, 18),
    )
    assert session.state_take_loop == expected
    assert (panel._view_start, panel._view_end) == view_range
    np.testing.assert_allclose(session.camera.eye, before.eye)
    drag(
        viewer,
        timeline_point(viewer, 3),
        timeline_point(viewer, 6),
        button=1,
        shift=True,
        cancel=True,
    )
    assert session.state_take_loop == expected
    _click(viewer, _item_center(viewer, "button", "Clear range##timeline-clear-range"))
    assert session.state_take_loop is None


def test_loop_survives_panel_close_and_space_resumes_replay(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    assert session.submit(cmd.SetStateTakeLoop(100, 103))
    assert session.submit(cmd.PlayStateTake())
    panel.open = False
    for _ in range(20):
        viewer.sync()
        assert 100 <= session.state_take_cursor <= 103
        assert session.state_take_playing
    viewer.app._toggle_playback()
    assert not session.state_take_playing
    viewer.app._toggle_playback()
    assert session.state_take_playing and session.paused


def test_playhead_pages_at_edge_and_stays_stationary_when_locked(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    assert session.submit(cmd.SeekStateTake(360))
    viewer.sync()
    panel._view_start, panel._view_end = 1, 11
    choose_follow(viewer, "page")
    assert panel._view_start < panel._playhead < panel._view_start + 1
    assert panel._view_end - panel._view_start == pytest.approx(10)
    choose_follow(viewer, "locked")
    fraction = (panel._playhead - panel._view_start) / 10
    for index in (400, 450, 200):
        session.submit(cmd.SeekStateTake(index))
        viewer.sync()
        assert (panel._playhead - panel._view_start) / 10 == pytest.approx(fraction)
    point = timeline_point(viewer, panel._playhead)
    drag(viewer, point, (point[0] + 50, point[1]), button=1)
    assert panel._follow_mode == "off"


def test_escape_clears_range_without_clearing_scene_selection(viewer):
    session = viewer.session
    node = next(n for n in session.nodes if n.posable)
    session.submit(cmd.SelectNode(node.node_id))
    session.submit(cmd.SetStateTakeLoop(10, 20))
    viewer.sync()
    imgui.internal.focus_window(imgui.internal.find_window_by_name("Keyframes"))
    _click(viewer, timeline_point(viewer, 5))
    hints = viewer.app._status_tool_hints(loading=False)
    assert [hint.hint_id for hint in hints if hint.control == "Esc"] == ["keyframes.clear_range"]
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()
    assert session.state_take_loop is None
    assert session.selected_node is not None
    assert session.selected_node.node_id == node.node_id


def test_transport_status_geometry_stays_stable_across_time_and_frame_digits(viewer, monkeypatch):
    from mojive.ui.panels import keyframes

    original = keyframes._toolbar_status
    rows = []

    def capture(*args, **kwargs):
        original(*args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rows.append((lo.x, lo.y, hi.x, hi.y))

    monkeypatch.setattr(keyframes, "_toolbar_status", capture)
    baseline = None
    for index in (9, 10, 99, 100, 299, 300, 899):
        assert viewer.session.submit(cmd.SeekStateTake(index))
        rows.clear()
        viewer.sync()
        assert len(rows) == 2
        if baseline is None:
            baseline = rows.copy()
        assert rows == baseline
