"""Real input, recording and status with a caller-owned physics clock."""

from dataclasses import replace
from queue import Queue

import pytest
from imgui_bundle import imgui

from mojive import PassiveAction, RecordingConfig, ViewerConfig, build
from mojive.app.passive_input import PassiveInput
from mojive.tools.keyframe_timeline import show_settings
from mojive.tools.ui_runtime import _activate_panel, _click, _item_center
from mojive.ui import ToolHint

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("writable", (False, True))
def test_external_control_capability_gates_native_actuator_slider(viewer, monkeypatch, writable):
    from mojive import commands as cmd

    viewer.session.submit(cmd.Pause())
    adapter = viewer.session.adapter
    monkeypatch.setattr(
        adapter, "caps", replace(adapter.caps, external_clock=True, write_ctrl=writable)
    )
    observed = []
    slider = imgui.slider_float

    def observe(label, *args, **kwargs):
        result = slider(label, *args, **kwargs)
        if label == "##control-actuator-0":
            observed.append(bool(imgui.get_item_flags() & imgui.ItemFlags_.disabled))
        return result

    monkeypatch.setattr(imgui, "slider_float", observe)
    _activate_panel(viewer, "Control")
    viewer.sync()
    before = float(viewer.session.frame.ctrl[0])
    point = _item_center(viewer, "slider_float", "##control-actuator-0")
    _click(viewer, (point[0] + 20, point[1]))
    viewer.sync()
    assert observed and all(value is not writable for value in observed)
    after = float(viewer.session.frame.ctrl[0])
    assert (after != before) is writable


@pytest.fixture
def viewer(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    width, scale = getattr(request, "param", (1280, 1.0))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    with build(
        model=model,
        data=mujoco.MjData(model),
        width=width,
        height=800,
        show_window=False,
        vsync=False,
        config=ViewerConfig(recording=RecordingConfig(countdown=0, run_simulation=True)),
    ) as viewer:
        viewer.app.passive_mode = True
        # Dock tabs and clipped value rows become available after layout settles.
        for _ in range(8):
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


@pytest.mark.parametrize("viewer", ((1280, 1.5), (1280, 2.5), (3200, 2.5)), indirect=True)
@pytest.mark.parametrize("docked", (False, True))
def test_grouped_scene_hints_reflow_without_duplicating_status(viewer, monkeypatch, docked):
    from mojive.ui.app import viewport
    from mojive.ui.viewport_widgets import tool_hint_text, tool_hints_size

    if not docked:
        for panel in viewer.panels:
            panel.open = False
    hints = (
        ToolHint("keys", label="Horizontal speed", keys=(("D", "+"), ("A", "−"))),
        ToolHint("keys", label="Vertical speed", keys=(("W", "+"), ("S", "−"))),
        ToolHint("keys", label="Rotation speed", keys=(("E", "+"), ("Q", "−"))),
        ToolHint("key", "X", "Zero velocity"),
        ToolHint("key", "Space", "Pause / resume"),
    )
    bridge = PassiveInput(viewer, Queue(maxsize=64))
    bridge.configure(
        tuple(PassiveAction(key, key) for key in ("d", "a", "w", "s", "e", "q", "x", "space")),
        hints=hints,
        hint_surface="scene",
    )
    viewer.set_input_handler(bridge)
    for _ in range(3):
        viewer.sync()
    rows = []
    original = viewport.draw_scene_tool_hints

    def observe(draw, origin, theme, scale, row, **kwargs):
        rows.append((origin, row, kwargs["size"]))
        measured = tool_hints_size(draw, scale, row, padding=True)
        assert measured == pytest.approx(kwargs["size"], abs=0.01)
        return original(draw, origin, theme, scale, row, **kwargs)

    monkeypatch.setattr(viewport, "draw_scene_tool_hints", observe)
    viewer.sync()
    displayed = tuple(hint.label for _, row, _ in rows for hint in row)
    if displayed[-1] == "…":
        assert displayed[:-1] == tuple(hint.label for hint in hints[: len(displayed) - 1])
        details = []
        text_wrapped = imgui.text_wrapped

        def observe_details(text):
            details.append(text)
            text_wrapped(text)

        monkeypatch.setattr(imgui, "text_wrapped", observe_details)
        (left, top), _, (row_width, row_height) = rows[-1]
        imgui.get_io().add_mouse_pos_event(left + row_width * 0.5, top + row_height * 0.5)
        viewer.sync()
        assert all(tool_hint_text(hint) in details for hint in hints)
    else:
        assert displayed == tuple(hint.label for hint in hints)
    scene_ids = {hint.hint_id for hint in viewer.tool_hints.resolve(surface="scene")}
    assert scene_ids.isdisjoint(
        hint.hint_id for hint in viewer.app._status_tool_hints(loading=False)
    )
    x, y, width, height = viewer.app._viewport_rect
    for (left, top), _, (row_width, row_height) in rows:
        assert x <= left and left + row_width <= x + width
        assert y <= top and top + row_height <= y + height
    assert rows[-1][0][1] + rows[-1][2][1] - rows[0][0][1] <= height * 0.3
    if docked:
        return
    _click(viewer, (x + width * 0.5, y + height * 0.4))
    io = imgui.get_io()
    for key, action in ((imgui.Key.d, "d"), (imgui.Key.a, "a")):
        io.add_key_event(key, True)
        viewer.sync()
        io.add_key_event(key, False)
        viewer.sync()
        event = bridge.events.get_nowait()
        assert event.action == action
        bridge.acknowledge(event)
    assert viewer.session.frame.time == 0
