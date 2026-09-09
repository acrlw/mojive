"""Native take-video controls preserve both endpoint poses and finalize real videos."""

import math
import time

import numpy as np
import pytest
from imageio_ffmpeg import read_frames
from imgui_bundle import imgui

from mojive import CaptureSurface, RecordingConfig, RecordingPhase, ViewerConfig, build
from mojive import commands as cmd
from mojive.assets import resolve
from mojive.tools.keyframe_timeline import populate_take, show_timeline
from mojive.tools.ui_runtime import _activate_panel, _click, _item_center, _item_rect

pytestmark = pytest.mark.gpu


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    with build(
        resolve("joint_types"),
        paused=True,
        vsync=False,
        width=1280,
        height=800,
        config=ViewerConfig(recording=RecordingConfig(countdown=60, end_hold=0.2)),
    ) as viewer:
        populate_take(viewer, 12)
        show_timeline(viewer)
        yield viewer


def tick(viewer, dt=0.08):
    # A slow display must produce the complete video, including its first pose.
    viewer.app._last_time = time.perf_counter() - dt
    viewer.sync()


@pytest.mark.parametrize("surface", (CaptureSurface.SCENE, CaptureSurface.VIEWPORT))
def test_button_rewinds_countdown_pause_tail_and_copy_saved_path(
    viewer, tmp_path, monkeypatch, surface
):
    from mojive.recording import VideoRecorder

    session, app = viewer.session, viewer.app
    path = tmp_path / "full take with spaces.mp4"
    app._capture_output = lambda *_args: path
    viewer.configure_recording(RecordingConfig(countdown=60, end_hold=0.2, surface=surface))
    assert session.submit(cmd.SetStateTakeLoop(2, 4))
    assert session.submit(cmd.SeekStateTake(8))
    recorded = []
    append = VideoRecorder.append

    def capture(recorder, image):
        recorded.append((session.state_take_cursor, image.copy()))
        append(recorder, image)

    monkeypatch.setattr(VideoRecorder, "append", capture)
    _click(viewer, _item_center(viewer, "invisible_button", "##take-video"))
    assert viewer.recording.phase is RecordingPhase.COUNTDOWN
    assert session.state_take_cursor == 0 and not session.state_take_playing
    tick(viewer)
    assert not path.exists()
    assert session.frame.time == pytest.approx(session.state_take_times[0])
    app._recording_deadline = 0
    tick(viewer)
    assert [cursor for cursor, _ in recorded] == [0]
    for _ in range(3):
        tick(viewer)
    viewer.pause_recording()
    cursor, count = session.state_take_cursor, len(recorded)
    for _ in range(4):
        tick(viewer)
    assert session.state_take_cursor == cursor and len(recorded) == count
    assert not session.state_take_playing
    viewer.resume_recording()
    while app._take_video.tail_frames is None:
        tick(viewer)
        assert len(recorded) < 100
    viewer.pause_recording()
    count = len(recorded)
    for _ in range(3):
        tick(viewer)
    assert len(recorded) == count
    viewer.resume_recording()
    while viewer.recording.active:
        tick(viewer)
        assert len(recorded) < 100
    expected_motion = math.ceil(
        (session.state_take_times[-1] - session.state_take_times[0]) * 60 - 1e-9
    )
    assert len(recorded) == expected_motion + 1 + 12
    assert recorded[0][0] == 0 and recorded[-1][0] == len(session.state_take_times) - 1
    assert [cursor for cursor, _ in recorded] == sorted(cursor for cursor, _ in recorded)
    assert session.state_take_loop == (2, 4)
    assert not session.state_take_playing and session.paused
    reader = read_frames(str(path))
    try:
        metadata = next(reader)
        frames = list(reader)
    finally:
        reader.close()
    assert len(frames) == len(recorded) and metadata["fps"] == 60
    width, height = metadata["size"]
    for raw, (_, expected) in ((frames[0], recorded[0]), (frames[-1], recorded[-1])):
        actual = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        h, w = expected.shape[:2]
        assert np.mean(np.abs(actual[:h, :w].astype(float) - expected)) < 5
    viewer.sync()
    viewer.sync()
    status = app.output.active_status()
    assert status.level == "success" and str(path.resolve()) in status.text
    _activate_panel(viewer, "Output")
    point = _item_center(viewer, "invisible_button", f"##output-row-{status.sequence}")
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(1, True)
    viewer.sync()
    io.add_mouse_button_event(1, False)
    viewer.sync()
    _click(viewer, _item_center(viewer, "menu_item", "Copy path"))
    assert imgui.get_clipboard_text() == str(path.resolve())
    assert app.output.active_status(now=time.monotonic() + 9) is None
    assert status in app.output.entries()


def test_countdown_cancel_and_transport_changes_end_take_video(viewer, tmp_path):
    canceled = tmp_path / "canceled.mp4"
    viewer.start_take_video(canceled)
    assert viewer.stop_recording() is None
    assert not canceled.exists()
    assert not viewer.session.state_take_playing
    path = tmp_path / "interrupted.mp4"
    viewer.start_take_video(path, countdown=0)
    tick(viewer)
    assert viewer.recording.frames == 1
    viewer.session.submit(cmd.SeekStateTake(7))
    tick(viewer)
    assert not viewer.recording.active
    assert viewer.session.state_take_cursor == 7
    assert path.exists()


def test_closing_video_settings_with_escape_preserves_loop_and_allows_recording(viewer):
    session = viewer.session
    assert session.submit(cmd.SetStateTakeLoop(2, 4))
    _click(viewer, _item_center(viewer, "invisible_button", "##timeline-options"))
    for _ in range(3):
        viewer.sync()
    io = imgui.get_io()
    for field, value in (("countdown", "2.5"), ("end_hold", "0.5")):
        lo, hi = _item_rect(
            viewer, "input_float", "Start delay (s)" if field == "countdown" else "End hold (s)"
        )
        _click(viewer, (lo[0] + 20, (lo[1] + hi[1]) * 0.5))
        modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
        io.add_key_event(modifier, True)
        io.add_key_event(imgui.Key.a, True)
        viewer.sync()
        io.add_key_event(imgui.Key.a, False)
        io.add_key_event(modifier, False)
        io.add_input_characters_utf8(value)
        viewer.sync()
        io.add_key_event(imgui.Key.enter, True)
        viewer.sync()
        io.add_key_event(imgui.Key.enter, False)
        viewer.sync()
        assert getattr(viewer.app.recording_config, field) == float(value)
        assert viewer.app.localizer.preference("recording")[field] == float(value)
    # Leave the number editor before testing Escape's popup ownership.
    _click(viewer, _item_center(viewer, "input_float", "Start delay (s)"))
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()
    assert session.state_take_loop == (2, 4)
    flags = imgui.PopupFlags_.any_popup_id | imgui.PopupFlags_.any_popup_level
    assert not imgui.is_popup_open("", flags)
    _click(viewer, _item_center(viewer, "invisible_button", "##take-video"))
    assert viewer.recording.phase is RecordingPhase.COUNTDOWN


def test_single_frame_take_without_end_hold_saves_one_frame(viewer, tmp_path):
    session = viewer.session
    assert session.submit(cmd.StartStateTakeRecording())
    assert session.submit(cmd.StopStateTakeRecording())
    assert len(session.state_take_times) == 1
    path = tmp_path / "one-frame.mp4"
    viewer.start_take_video(path, countdown=0, end_hold=0)
    tick(viewer)
    assert not viewer.recording.active and session.state_take_cursor == 0
    reader = read_frames(str(path))
    try:
        next(reader)
        assert len(list(reader)) == 1
    finally:
        reader.close()
