"""Production interaction contracts for the selective UI design adoption."""

from __future__ import annotations

import time

import pytest
from imgui_bundle import imgui

from mojive import build
from mojive import commands as cmd
from mojive.assets import resolve
from mojive.tools.ui_runtime import (
    _activate_panel,
    _click,
    _item_center,
    _item_rect,
    _right_click,
    _settle,
)
from mojive.ui.app import Keys
from mojive.ui.panels.filters import severity_group

pytestmark = pytest.mark.gpu


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    with build(
        resolve("joint_gizmo"), paused=True, vsync=False, show_window=False, width=1400, height=1000
    ) as viewer:
        _settle(viewer, 8)
        yield viewer


def _key(viewer, key):
    imgui.get_io().add_key_event(key, True)
    viewer.sync()
    imgui.get_io().add_key_event(key, False)
    viewer.sync()


def _name_edit(viewer, text):
    _activate_panel(viewer, "Inspector")
    point = _item_center(viewer, "invisible_button", "##entity_name_label")
    _click(viewer, point)
    _click(viewer, point)
    viewer.sync()
    assert viewer.panels.get("Inspector")._renaming
    imgui.get_io().add_input_characters_utf8(text)
    viewer.sync()


def _wait_edits(viewer):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        viewer.sync()
        if not viewer.app._model_load_future and not viewer.app._model_load_queue:
            _settle(viewer, 3)
            return
        time.sleep(0.005)
    raise AssertionError("model edit did not finish")


def test_selection_keeps_tools_off_and_tools_toggle_without_clearing_selection(viewer):
    gizmo = viewer.app.gizmo
    node = next(n for n in viewer.session.nodes if n.posable)
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _settle(viewer, 3)
    assert not gizmo.enabled and not gizmo.visible
    for mode, key in (("translate", "gizmo_translate"), ("rotate", "gizmo_rotate")):
        viewer.app.apply_keys(Keys(**{key: True}))
        _settle(viewer, 3)
        assert gizmo.enabled and gizmo.mode == mode and gizmo.visible
        viewer.app.apply_keys(Keys(**{key: True}))
        _settle(viewer, 3)
        assert not gizmo.enabled and not gizmo.visible
        assert viewer.session.selected_node is node
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-tool-move"))
    assert gizmo.enabled
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-tool-move"))
    assert not gizmo.enabled
    assert viewer.session.selected_node is node


@pytest.mark.parametrize("finish", ("enter", "blur", "selection", "hidden", "escape"))
def test_name_edit_commits_on_enter_or_blur_and_preserves_entity_identity(viewer, finish):
    node = next(n for n in viewer.session.nodes if n.name == "04_free")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _settle(viewer, 3)
    _name_edit(viewer, "renamed_entity")
    if finish == "selection":
        other = next(n for n in viewer.session.nodes if n.name == "01_revolute")
        viewer.session.submit(cmd.SelectNode(other.node_id))
        viewer.sync()
    elif finish == "hidden":
        viewer.panels.get("Inspector").open = False
        viewer.sync()
    elif finish == "blur":
        window = imgui.internal.find_window_by_name("Inspector")
        _click(viewer, (window.pos.x + 30, window.pos.y + window.size.y - 40))
    else:
        _key(viewer, imgui.Key.escape if finish == "escape" else imgui.Key.enter)
    _wait_edits(viewer)
    assert not viewer.panels.get("Inspector")._renaming
    names = {n.name for n in viewer.session.nodes}
    assert ("renamed_entity" in names) is (finish != "escape")
    if finish == "selection":
        assert viewer.session.selected_node.name == "01_revolute"


def test_output_toggles_and_copy_are_scoped_to_output_focus(viewer):
    output = viewer.app.output
    output.clear()
    for level, message in (("info", "one"), ("warning", "two"), ("critical", "three")):
        output.write(message, level=level)
    _activate_panel(viewer, "Output")
    panel = viewer.panels.get("Output")
    _click(viewer, _item_center(viewer, "button", "##output-level-warning"))
    assert {severity_group(e.level) for e in panel._filtered_entries} == {"info", "error"}
    entry = panel._filtered_entries[0]
    _click(viewer, _item_center(viewer, "invisible_button", f"##output-row-{entry.sequence}"))
    io = imgui.get_io()
    io.add_key_event(imgui.Key.mod_ctrl, True)
    _key(viewer, imgui.Key.c)
    io.add_key_event(imgui.Key.mod_ctrl, False)
    viewer.sync()
    assert imgui.get_clipboard_text() == "one"
    _activate_panel(viewer, "Hierarchy")
    window = imgui.internal.find_window_by_name("Hierarchy")
    imgui.internal.focus_window(window)
    imgui.set_clipboard_text("unrelated")
    io.add_key_event(imgui.Key.mod_ctrl, True)
    _key(viewer, imgui.Key.c)
    io.add_key_event(imgui.Key.mod_ctrl, False)
    viewer.sync()
    assert imgui.get_clipboard_text() == "unrelated"


@pytest.mark.parametrize("name", ("hinge_drive", "body_drive", "site_drive"))
def test_actuator_card_selects_then_double_clicks_to_focus_and_arm_a_tool(viewer, name):
    viewer.session.submit(cmd.LoadAsset(resolve("actuator_visuals")))
    _activate_panel(viewer, "Control")
    actuator = next(a for a in viewer.session.actuators if a.name == name)
    panel = viewer.panels.get("Control")
    point = _item_center(viewer, "invisible_button", f"##actuator-select-{actuator.ctrl_address}")
    _click(viewer, point)
    assert not viewer.app.gizmo.enabled
    assert panel._selected_address == actuator.ctrl_address
    assert viewer.session.selected_node.node_id == actuator.target_node_id
    _click(viewer, point)
    assert viewer.app.gizmo.enabled
    assert viewer.app.camera.animating
    assert viewer.session.selected_node.node_id == actuator.target_node_id


def test_hierarchy_filter_pills_wrap_within_the_panel(viewer):
    panel = viewer.panels.get("Hierarchy")
    _activate_panel(viewer, "Hierarchy")
    bounds = [
        _item_rect(viewer, "button", f"##hierarchy-type-{kind}")
        for kind, count in panel._type_counts.items()
        if count and kind not in {"world", "environment"}
    ]
    window = imgui.internal.find_window_by_name("Hierarchy")
    assert len({lo[1] for lo, _ in bounds}) > 1
    for lo, hi in bounds:
        assert lo[0] >= window.inner_clip_rect.min.x
        assert hi[0] <= window.inner_clip_rect.max.x


@pytest.mark.parametrize("limited", (True, False))
def test_control_rail_and_numeric_entry_preserve_real_limits_and_reset(viewer, tmp_path, limited):
    path = tmp_path / "control.xml"
    path.write_text(
        '<mujoco><worldbody><body><joint name="hinge"/>'
        '<geom type="sphere" size=".1"/></body></worldbody><actuator>'
        f'<motor name="drive" joint="hinge" ctrllimited="{str(limited).lower()}" '
        'ctrlrange="-2 2"/></actuator></mujoco>'
    )
    viewer.session.submit(cmd.LoadAsset(path))
    _activate_panel(viewer, "Control")
    _settle(viewer, 4)
    label = "##control-actuator-0"
    if limited:
        lo, hi = _item_rect(viewer, "slider_float", label)
        _click(viewer, (lo[0] + 0.8 * (hi[0] - lo[0]), (lo[1] + hi[1]) * 0.5))
        assert 0.8 < viewer.session.frame.ctrl[0] < 1.8
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    viewer.sync()
    _click(viewer, _item_center(viewer, "drag_float", f"{label}-value"))
    _key(viewer, imgui.Key.a)
    io.add_key_event(modifier, False)
    viewer.sync()
    assert io.want_text_input
    io.add_input_characters_utf8("123.456")
    viewer.sync()
    _key(viewer, imgui.Key.enter)
    _settle(viewer, 2)
    assert viewer.session.frame.ctrl[0] == pytest.approx(2.0 if limited else 123.456)
    _right_click(viewer, _item_center(viewer, "drag_float", f"{label}-value"))
    _settle(viewer, 2)
    assert viewer.session.frame.ctrl[0] == pytest.approx(0.0)


def test_output_filters_precede_the_search_on_one_wide_toolbar(viewer):
    _activate_panel(viewer, "Output")
    rects = [
        _item_rect(viewer, "button", f"##output-level-{level}")
        for level in ("info", "warning", "error")
    ]
    search = _item_rect(viewer, "input_text_with_hint", "##output-filter")
    assert all(abs(lo[1] - search[0][1]) < 1 for lo, _ in rects)
    assert rects[-1][1][0] < search[0][0]
    assert all(hi[0] - lo[0] < 65 for lo, hi in rects)


def test_output_collapse_expands_the_viewport_and_restores_the_log(viewer):
    _activate_panel(viewer, "Output")
    panel = viewer.panels.get("Output")
    viewer.app.output.write("Summary survives collapse", level="warning")
    _settle(viewer, 3)
    _click(viewer, _item_center(viewer, "button", "##output-collapse"))
    _settle(viewer, 4)
    assert panel.collapsed
    summary = imgui.internal.find_window_by_name("##output-summary")
    assert summary is not None and summary.active
    assert summary.size.y < 60 * viewer.window.style_scale
    _click(viewer, _item_center(viewer, "button", "##output-expand"))
    _settle(viewer, 4)
    assert not panel.collapsed
    assert any(e.text == "Summary survives collapse" for e in panel._filtered_entries)
    assert imgui.internal.find_window_by_name("Output").active


def test_viewport_recording_controls_drive_a_real_take(viewer):
    before = tuple(viewer.session.keyframes)
    _click(
        viewer, _item_center(viewer, "invisible_button", "##viewport-playback-recording-options")
    )
    _click(viewer, _item_center(viewer, "menu_item", "Record Take"))
    _settle(viewer, 4)
    assert viewer.session.state_take_recording
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-record"))
    assert not viewer.session.state_take_recording
    assert viewer.session.paused and viewer.session.state_take_times
    assert tuple(viewer.session.keyframes) == before
    _click(
        viewer, _item_center(viewer, "invisible_button", "##viewport-playback-recording-options")
    )
    assert imgui.is_popup_open("viewport-recording-options", imgui.PopupFlags_.any_popup_id)


def test_capture_snapshot_does_not_recompile_or_modify_model_keyframes(viewer):
    from mojive.tools.keyframe_timeline import show_timeline
    from mojive.ui.panels.keyframes import timeline_channel_width, timeline_time_to_x

    show_timeline(viewer)
    session = viewer.session
    generation = session.structure_generation
    before = tuple(session.keyframes)
    _click(viewer, _item_center(viewer, "invisible_button", "##capture-snapshot"))
    assert len(session.scene_snapshots) == 1
    snapshot = session.scene_snapshots[0]
    assert session.structure_generation == generation
    assert tuple(session.keyframes) == before
    assert not viewer.app._model_load_future
    assert session.submit(cmd.Step(8))
    _settle(viewer, 3)
    assert session.frame.time > snapshot.time
    panel = viewer.panels.get("Keyframes")
    lo, hi = _item_rect(viewer, "invisible_button", "##keyframe-dope-sheet")
    scale = viewer.window.style_scale
    left = lo[0] + timeline_channel_width(hi[0] - lo[0], scale)
    right = hi[0] - (64 * scale if hi[0] - lo[0] >= 600 * scale else 0)
    x = timeline_time_to_x(snapshot.time, panel._view_start, panel._view_end, left, right)
    y = lo[1] + 27 * scale + (hi[1] - lo[1] - 27 * scale) * 0.75
    _click(viewer, (x, y))
    _click(viewer, (x, y))
    _settle(viewer, 3)
    assert session.frame.time == pytest.approx(snapshot.time)
    assert session.structure_generation == generation


@pytest.mark.parametrize("panel_name", ("Joints", "Control"))
def test_angular_unit_toggle_converts_display_and_input_without_changing_physics(
    viewer, monkeypatch, panel_name
):
    import math

    if panel_name == "Control":
        viewer.session.submit(cmd.LoadAsset(resolve("actuator_visuals")))
        actuator = viewer.session.actuators[0]
        # Only adapter-declared control units opt into conversion.
        actuator.unit = "rad"
        index = actuator.ctrl_address
        label = f"##control-actuator-{index}"
        command_type, field_name = cmd.SetCtrl, "ctrl"
    else:
        joint = next(j for j in viewer.session.joints if j.type == "hinge")
        index = joint.qpos_adr
        label = f"##joint-qpos-{index}"
        command_type, field_name = cmd.SetQpos, "qpos"

    def submit_value(value):
        return viewer.session.submit(command_type(index, value))

    def read_value():
        return getattr(viewer.session.frame, field_name)[index]

    _activate_panel(viewer, panel_name)
    _settle(viewer, 3)
    initial = float(read_value())
    assert submit_value(0.25).ok
    observed = {}
    native = imgui.drag_float

    def capture(item_label, value, *args, **kwargs):
        observed[item_label] = value
        return native(item_label, value, *args, **kwargs)

    monkeypatch.setattr(imgui, "drag_float", capture)
    _settle(viewer, 3)
    _click(viewer, _item_center(viewer, "button", f"rad##{label}-unit"))
    _settle(viewer, 2)
    assert read_value() == pytest.approx(0.25)
    assert observed[f"{label}-value"] == pytest.approx(math.degrees(0.25))
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    viewer.sync()
    _click(viewer, _item_center(viewer, "drag_float", f"{label}-value"))
    _key(viewer, imgui.Key.a)
    io.add_key_event(modifier, False)
    viewer.sync()
    io.add_input_characters_utf8("15")
    viewer.sync()
    _key(viewer, imgui.Key.enter)
    _settle(viewer, 2)
    assert read_value() == pytest.approx(math.radians(15), abs=1e-6)
    _right_click(viewer, _item_center(viewer, "drag_float", f"{label}-value"))
    _settle(viewer, 2)
    assert read_value() == pytest.approx(initial)
    _click(viewer, _item_center(viewer, "button", f"deg##{label}-unit"))
    _settle(viewer, 2)
    assert observed[f"{label}-value"] == pytest.approx(initial)


def test_camera_units_share_aligned_fields_and_preserve_view_on_toggle(viewer, monkeypatch):
    import math

    _activate_panel(viewer, "Camera")
    camera = viewer.app.camera
    camera.yaw = 30.0
    _settle(viewer, 3)
    observed = {}
    native = imgui.drag_float

    def capture(label, value, *args, **kwargs):
        observed[label] = value
        return native(label, value, *args, **kwargs)

    monkeypatch.setattr(imgui, "drag_float", capture)
    angular = "##camera-yaw"
    distance = "##camera-distance"
    fields = [_item_rect(viewer, "drag_float", f"{name}-value") for name in (angular, distance)]
    badges = [
        _item_rect(viewer, "button", f"{unit}##{name}-unit")
        for unit, name in (("deg", angular), ("m", distance))
    ]
    assert fields[0][0][0] == pytest.approx(fields[1][0][0], abs=0.01)
    assert badges[0][1][0] - badges[0][0][0] == pytest.approx(badges[1][1][0] - badges[1][0][0])
    _click(viewer, _item_center(viewer, "button", f"deg##{angular}-unit"))
    _settle(viewer, 2)
    assert camera.yaw == pytest.approx(30.0)
    assert observed[f"{angular}-value"] == pytest.approx(math.radians(30.0))
    previous_distance = camera.distance
    _click(viewer, _item_center(viewer, "button", f"m##{distance}-unit"))
    assert camera.distance == previous_distance
    assert not viewer.panels.get("Camera")._angular_degrees
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    viewer.sync()
    _click(viewer, _item_center(viewer, "drag_float", f"{angular}-value"))
    _key(viewer, imgui.Key.a)
    io.add_key_event(modifier, False)
    viewer.sync()
    io.add_input_characters_utf8("0.25")
    viewer.sync()
    _key(viewer, imgui.Key.enter)
    _settle(viewer, 2)
    assert camera.yaw == pytest.approx(math.degrees(0.25), abs=1e-5)
    _right_click(viewer, _item_center(viewer, "drag_float", f"{angular}-value"))
    _settle(viewer, 2)
    from mojive.ui.camera import DEFAULT_YAW

    assert camera.yaw == pytest.approx(DEFAULT_YAW)


def test_viewport_video_button_shows_countdown_and_can_cancel(viewer):
    from mojive.capture import RecordingPhase
    from mojive.config import RecordingConfig

    viewer.configure_recording(RecordingConfig(countdown=60))
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-record"))
    _settle(viewer, 3)
    assert viewer.app.recording.phase is RecordingPhase.COUNTDOWN
    countdown = imgui.internal.find_window_by_name("##recording_countdown")
    assert countdown.active
    capsule = viewer.app._playback_widget_rect
    assert countdown.pos.y >= capsule[3]
    assert countdown.pos.y - capsule[3] <= 12
    assert countdown.size.x < 180
    assert not viewer.session.state_take_recording
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-record"))
    assert not viewer.app.recording.active


def test_inspector_camera_sync_and_bookmark_paste_preserve_the_editor_view(viewer, tmp_path):
    import json
    from dataclasses import replace

    import numpy as np

    from mojive.scene_state import camera_bookmark

    path = tmp_path / "camera.xml"
    path.write_text(
        '<mujoco><worldbody><geom type="sphere" size=".1"/>'
        '<camera name="shot" pos="0 -3 1"/></worldbody></mujoco>'
    )
    viewer.session.submit(cmd.LoadAsset(path))
    node = next(n for n in viewer.session.nodes if n.type.value == "camera")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _activate_panel(viewer, "Inspector")
    _settle(viewer, 3)
    info = viewer.session.cameras[node.camera_index]
    editor = viewer.app.camera.view()
    _click(viewer, _item_center(viewer, "button", "Sync view##camera-sync"))
    _settle(viewer, 3)
    np.testing.assert_allclose(viewer.session.camera_view(info.camera_id).eye, editor.eye)
    imported = replace(editor, eye=np.asarray(editor.eye) + np.array((1, 2, 3)), fov_y=0.7)
    imgui.set_clipboard_text(json.dumps(camera_bookmark(viewer.app.camera, imported)))
    _click(viewer, _item_center(viewer, "button", "Paste bookmark##camera-paste"))
    _settle(viewer, 3)
    restored = viewer.session.camera_view(info.camera_id)
    np.testing.assert_allclose(restored.eye, imported.eye)
    assert restored.fov_y == pytest.approx(0.7)
    np.testing.assert_allclose(viewer.app.camera.view().eye, editor.eye)
    imgui.set_clipboard_text('{"format":"unrelated"}')
    _click(viewer, _item_center(viewer, "button", "Paste bookmark##camera-paste"))
    np.testing.assert_allclose(viewer.session.camera_view(info.camera_id).eye, imported.eye)


def test_view_camera_click_eases_and_pointer_handoff_keeps_the_displayed_pose(viewer, tmp_path):
    import numpy as np

    path = tmp_path / "transition-camera.xml"
    path.write_text(
        '<mujoco><worldbody><geom type="sphere" size=".1"/>'
        '<camera name="shot" pos="0 -3 1"/></worldbody></mujoco>'
    )
    assert viewer.session.submit(cmd.LoadAsset(path))
    node = next(n for n in viewer.session.nodes if n.type.value == "camera")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _activate_panel(viewer, "Inspector")
    _settle(viewer, 4)
    window = imgui.internal.find_window_by_name("Inspector")
    imgui.internal.set_scroll_y(window, window.scroll_max.y)
    _settle(viewer, 3)
    app = viewer.app
    editor = app.camera.view()
    point = _item_center(viewer, "button", "View Camera")
    _click(viewer, point)
    assert app._camera_transition is not None, (point, window.inner_clip_rect, app._model_camera_id)
    app._dt = 0.07
    app._sync_model_camera()
    intermediate = app._camera_view()
    target = viewer.session.camera_view(viewer.session.cameras[node.camera_index].camera_id)
    assert not np.allclose(intermediate.eye, target.eye)
    app._select_model_camera_animated(-1)
    app._advance_camera(0.3)
    np.testing.assert_allclose(app._camera_view().eye, editor.eye)
    app._select_model_camera_animated(viewer.session.cameras[node.camera_index].camera_id)
    app._dt = 0.07
    app._sync_model_camera()
    intermediate = app._camera_view()
    app._leave_model_camera()
    assert app._camera_transition is None and app._model_camera_id == -1
    np.testing.assert_allclose(
        app.camera.view().view_matrix(), intermediate.view_matrix(), atol=1e-6
    )


def test_pending_model_apply_discard_and_realtime_mode(viewer, tmp_path):
    import numpy as np

    from mojive.model_edits import model_edit_scope

    app, session = viewer.app, viewer.session
    path = tmp_path / "pending.xml"
    path.write_text(
        '<mujoco><worldbody><body name="arm"><joint/><geom name="box" type="box" size=".1 .2 .3"/></body></worldbody></mujoco>'
    )
    assert session.submit(cmd.LoadAsset(path)).ok
    _settle(viewer, 4)
    node = next(
        n
        for n in session.nodes
        if n.geom_index >= 0
        and n.source_editable
        and np.all(session.source.geom_size[n.geom_index] > 0)
    )
    original = session.source.geom_size[node.geom_index].copy()
    session.submit(cmd.SelectNode(node.node_id))
    assert not app.live_model_updates
    with model_edit_scope(session, app._intercept_model_edit):
        assert session.submit(cmd.BeginEditTransaction("Resize")).ok
        assert session.submit(cmd.SetGeometrySize(node.node_id, original * 1.2)).ok
        assert session.submit(cmd.EndEditTransaction()).ok
    _settle(viewer, 4)
    np.testing.assert_allclose(session.source.geom_size[node.geom_index], original * 1.2)
    assert app.model_edits.active
    apply = _item_center(
        viewer, "button", viewer.app.localizer.text("Apply") + "##apply_model_edits"
    )
    discard = _item_center(
        viewer, "button", viewer.app.localizer.text("Discard") + "##discard_model_edits"
    )
    assert apply[0] < discard[0]
    _click(viewer, discard)
    _settle(viewer, 2)
    assert not app.model_edits.active
    np.testing.assert_allclose(session.source.geom_size[node.geom_index], original)
    with model_edit_scope(session, app._intercept_model_edit):
        assert session.submit(cmd.RenameModelElement(node.node_id, "pending_name")).ok
    _settle(viewer, 3)
    _click(
        viewer,
        _item_center(viewer, "button", viewer.app.localizer.text("Apply") + "##apply_model_edits"),
    )
    _wait_edits(viewer)
    assert not app.model_edits.active, (
        app.model_edits.error,
        session.last_message,
        app.model_edits._checkpoint,
        session.editing,
        app._apply_model_edits_requested,
        app._model_load_error,
    )
    assert session.node(node.node_id).name == "pending_name"
    if session.adapter.caps.edit_history:
        assert session.submit(cmd.Undo()).ok
    app.set_live_model_updates(True)
    with model_edit_scope(session, app._intercept_model_edit):
        assert session.submit(cmd.RenameModelElement(node.node_id, "live_name")).ok
    assert not app.model_edits.active
    assert session.node(node.node_id).name == "live_name"
