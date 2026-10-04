"""Recording acceptance operates real controls in clipped docks and narrow status bars."""

import time

import pytest
from imgui_bundle import imgui

from mojive import CaptureSurface, RecordingPhase
from mojive.app.composition import build
from mojive.config import PanelConfig, RecordingConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.tools.recording_layers import _run
from mojive.tools.ui_runtime import _click, _item_rect

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.mark.parametrize("language,scale", [("en", 1.0), ("zh_CN", 2.25)])
def test_layers_acceptance_scrolls_controls_and_records_visible_changes(
    tmp_path, monkeypatch, language, scale
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_LANGUAGE", language)
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    config = ViewerConfig(
        panels={"layers": PanelConfig(open=True), "inspector": PanelConfig(open=False)},
        recording=RecordingConfig(copy_to_clipboard=False),
    )
    with build(
        resolve("joint_gizmo"),
        config=config,
        vsync=False,
        show_window=False,
        width=1600,
        height=1000,
    ) as viewer:
        _run(viewer, tmp_path)


@pytest.mark.parametrize("language,scale", [("en", 1.0), ("zh_CN", 2.25)])
def test_narrow_status_keeps_recording_actions_visible_and_clickable(
    tmp_path, monkeypatch, language, scale
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_LANGUAGE", language)
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    config = ViewerConfig(recording=RecordingConfig(copy_to_clipboard=False))
    with build(
        resolve("joint_types"),
        config=config,
        paused=True,
        vsync=False,
        show_window=False,
        width=round(360 * scale),
        height=round(480 * scale),
    ) as viewer:
        viewer.app.external_status = (
            "Connecting to remote publisher with a very long application name"
            if language == "en"
            else "正在连接名称很长的远程发布应用程序并等待场景同步"
        )
        for _ in range(4):
            viewer.sync()

        def click_action(label):
            lo, hi = _item_rect(viewer, "invisible_button", label)
            status = imgui.internal.find_window_by_name("Status###application_status")
            assert status.pos.x <= lo[0] < hi[0] <= status.pos.x + status.size.x
            assert status.pos.y <= lo[1] < hi[1] <= status.pos.y + status.size.y
            _click(viewer, ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5))

        canceled = tmp_path / "canceled.mp4"
        viewer.start_recording(canceled, surface=CaptureSurface.WINDOW, countdown=30)
        viewer.sync()
        assert viewer.recording.phase is RecordingPhase.COUNTDOWN
        viewer.capture(tmp_path / "status-countdown.png", surface=CaptureSurface.WINDOW)
        click_action("##status_recording_stop")
        assert viewer.recording.phase is RecordingPhase.IDLE
        assert not canceled.exists()

        path = tmp_path / "status-recording.mp4"
        viewer.start_recording(path, surface=CaptureSurface.WINDOW, countdown=0)
        viewer.sync()
        viewer.sync()
        assert viewer.recording.phase is RecordingPhase.RECORDING
        viewer.capture(tmp_path / "status-recording.png", surface=CaptureSurface.WINDOW)
        click_action("##status_recording_pause")
        assert viewer.recording.phase is RecordingPhase.PAUSED
        viewer.capture(tmp_path / "status-paused.png", surface=CaptureSurface.WINDOW)
        click_action("##status_recording_pause")
        assert viewer.recording.phase is RecordingPhase.RECORDING
        click_action("##status_recording_stop")
        deadline = time.monotonic() + 10
        while viewer.recording.phase is RecordingPhase.FINALIZING:
            assert time.monotonic() < deadline
            viewer.sync()
            time.sleep(0.005)
        assert viewer.recording.phase is RecordingPhase.IDLE
        assert path.stat().st_size > 0
