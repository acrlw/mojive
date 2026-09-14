"""Real input, recording and status with a caller-owned physics clock."""

from queue import Queue

import pytest
from imgui_bundle import imgui

from mojive import PassiveAction, RecordingConfig, ViewerConfig, build
from mojive.app.passive_input import PassiveInput
from mojive.tools.keyframe_timeline import show_settings
from mojive.tools.ui_runtime import _click, _item_center

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
mujoco = pytest.importorskip("mujoco")


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    with build(
        model=model,
        data=mujoco.MjData(model),
        width=1280,
        height=800,
        show_window=False,
        vsync=False,
        config=ViewerConfig(recording=RecordingConfig(countdown=0, run_simulation=True)),
    ) as viewer:
        viewer.app.passive_mode = True
        viewer.sync()
        yield viewer


def test_passive_toolbar_records_without_starting_or_stopping_physics(
    viewer, tmp_path, monkeypatch
):
    from mojive.capture import RecordingPhase

    output = tmp_path / "toolbar.mp4"
    monkeypatch.setattr(viewer.app, "_capture_output", lambda *args: output)
    before = viewer.session.frame.time

    def button(name):
        return _item_center(viewer, "invisible_button", f"##viewport-playback-{name}")

    _click(viewer, button("toggle"))
    assert viewer.session.frame.time == before
    _click(viewer, button("record"))
    viewer.sync()
    assert viewer.recording.phase is RecordingPhase.RECORDING
    assert not viewer.app._recording_run_simulation
    _click(viewer, button("record"))
    assert viewer.recording.phase is RecordingPhase.PAUSED
    viewer.stop_recording()
    assert output.stat().st_size > 0
    assert viewer.session.frame.time == before
    show_settings(viewer)
    _click(viewer, _item_center(viewer, "checkbox", "##recording_run_simulation"))
    assert viewer.app.recording_config.run_simulation  # Disabled, not silently persisted off.


def test_passive_transport_and_focused_keys_enqueue_without_mutating_physics(viewer):
    bridge = PassiveInput(viewer, Queue(maxsize=64))
    bridge.configure((PassiveAction("pause_policy", "8", "Pause policy", "toggle"),))
    viewer.set_input_handler(bridge)
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-toggle"))
    event = bridge.events.get_nowait()
    assert event.action == "pause_policy"
    assert viewer.session.frame.time == 0
    assert viewer.app.external_paused is None
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-toggle"))
    assert bridge.events.empty()

    bridge.acknowledge(event)
    x, y, width, height = viewer.app._viewport_rect
    _click(viewer, (x + width * 0.5, y + height * 0.6))
    io = imgui.get_io()
    io.add_key_event(imgui.Key._8, True)
    viewer.sync()
    io.add_key_event(imgui.Key._8, False)
    viewer.sync()
    event = bridge.events.get_nowait()
    assert event.action == "pause_policy"
    bridge.acknowledge(event)
    show_settings(viewer)
    io.add_key_event(imgui.Key._8, True)
    viewer.sync()
    io.add_key_event(imgui.Key._8, False)
    viewer.sync()
    assert bridge.events.empty()


@pytest.mark.parametrize("paused,state", ((None, "external"), (False, "running"), (True, "paused")))
def test_passive_status_uses_caller_state_without_changing_clock(
    viewer, monkeypatch, paused, state
):
    from mojive.ui.app import status

    observed = []
    draw_status = status.draw_status

    def observe(*args, **kwargs):
        observed.append(kwargs)
        return draw_status(*args, **kwargs)

    monkeypatch.setattr(status, "draw_status", observe)
    viewer.app.external_paused = paused
    before = (viewer.session.paused, viewer.session.frame.time)
    viewer.sync()
    assert observed[-1]["state"] == state
    assert observed[-1]["activity"] == ""
    assert observed[-1]["passive"]
    assert (viewer.session.paused, viewer.session.frame.time) == before
