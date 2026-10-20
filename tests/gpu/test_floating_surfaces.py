"""Production floating forms share geometry and preserve native edit ownership."""

from pathlib import Path
from threading import Event
from time import monotonic

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.capture import RecordingPhase
from mojive.config import LayoutConfig, ViewerConfig
from mojive.interaction.gizmo import GizmoHandle
from mojive.scene.workspace import MissingResource

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/ui-floating"


@pytest.fixture(params=((1, "en"), (1.25, "zh_CN"), (2, "zh_CN")))
def floating(request, backend_name, monkeypatch, tmp_path):
    scale, language = request.param
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        width=round(1440 * scale),
        height=round(900 * scale),
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    viewer.app.set_language(language)
    events = []
    process = viewer.window._input.process_inputs

    def inputs():
        process()
        for method, args in events:
            getattr(imgui.get_io(), method)(*args)
        events.clear()
        imgui.get_io().delta_time = 1 / 60

    monkeypatch.setattr(viewer.window._input, "process_inputs", inputs)

    def frames(count=4):
        for _ in range(count):
            viewer.sync()

    def event(method, *args):
        events.append((method, args))
        frames(1)

    def key(key):
        event("add_key_event", key, True)
        event("add_key_event", key, False)
        frames()

    def click(rect):
        x, y, width, height = rect
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frames()

    def save(name):
        frames()
        folder = OUTPUT / backend_name / f"{scale:g}-{language}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(folder / (name + ".png"))

    viewer.frames, viewer.event, viewer.key, viewer.click_rect, viewer.save_ui = (
        frames,
        event,
        key,
        click,
        save,
    )
    try:
        frames(8)
        yield viewer
    finally:
        viewer.release()


def _window(fragment):
    return next(
        w
        for w in imgui.get_current_context().windows
        if w.active and not w.hidden and fragment in w.name
    )


def _assert_surface(window, scale):
    assert window.flags & imgui.WindowFlags_.no_title_bar
    inset = 12 if window.flags & imgui.WindowFlags_.modal else 8
    assert window.window_padding.x == window.window_padding.y == round(inset * scale)
    assert window.window_rounding - window.window_padding.x == pytest.approx(4 * scale)
    viewport = imgui.get_main_viewport()
    assert window.pos.x >= viewport.pos.x
    assert window.pos.y >= viewport.pos.y
    assert window.pos.x + window.size.x <= viewport.pos.x + viewport.size.x + 1
    assert window.pos.y + window.size.y <= viewport.pos.y + viewport.size.y + 1


def test_joint_picker_and_scalar_form_share_surface_and_keep_input(floating):
    v, app = floating, floating.app
    ui, scale = app.workspace.ui, v.window.style_scale
    app.tool = "move"
    node = next(n for n in v.session.nodes if n.name == "05_multi_joint")
    assert v.session.submit(cmd.SelectNode(node.node_id)).ok
    v.frames()
    picker = _window("viewport_joint_gizmo")
    _assert_surface(picker, scale)
    joint = next(j for j in v.session.joints if j.name == "05_multi_slide_y")
    v.click_rect(ui.hits[f"viewport-joint-{joint.joint_id}"])
    assert app.gizmo.selected_joint_id(node.body_index) == joint.joint_id
    v.save_ui("joint-picker")

    title = ui.hits["floating-heading/" + picker.name]
    start = title[0] + title[2] / 2, title[1] + title[3] / 2
    before = tuple(picker.pos)
    v.event("add_mouse_pos_event", *start)
    v.event("add_mouse_button_event", 0, True)
    v.event("add_mouse_pos_event", start[0] + 30 * scale, start[1] + 20 * scale)
    v.frames(2)
    v.event("add_mouse_button_event", 0, False)
    assert tuple(picker.pos) != before

    edit = None
    for handle in (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z):
        app.gizmo._hovered = handle
        edit = app.gizmo.precise_input(v.session)
        if edit is not None:
            break
    assert edit is not None
    # Place the form beside the persistent selector for visual comparison.
    v.event("add_mouse_pos_event", picker.pos.x + picker.size.x + 8 * scale, picker.pos.y)
    app._begin_precise_gizmo_input(edit)
    v.frames()
    precise = _window("##Popup_")
    _assert_surface(precise, scale)
    assert precise.window_rounding == picker.window_rounding
    field = ui.hits["precise-gizmo-value"]
    assert field[0] == pytest.approx(precise.pos.x + precise.window_padding.x)
    assert field[0] + field[2] == pytest.approx(
        precise.pos.x + precise.size.x - precise.window_padding.x, abs=scale
    )
    v.save_ui("joint-precise-unit")
    original = v.session.frame.qpos.copy()
    v.event("add_input_characters_utf8", "0.05")
    v.save_ui("joint-precise")
    v.key(imgui.Key.enter)
    assert app._precise_gizmo_edit is None
    assert v.session.frame.qpos[joint.qpos_adr] == pytest.approx(original[joint.qpos_adr] + 0.05)
    app._begin_precise_gizmo_input(edit)
    v.frames()
    applied = v.session.frame.qpos.copy()
    v.event("add_input_characters_utf8", "0.12")
    v.key(imgui.Key.escape)
    np.testing.assert_array_equal(v.session.frame.qpos, applied)
    assert app._precise_gizmo_edit is None


def test_joint_picker_closes_after_focus_and_preserves_gizmo(floating):
    v, app = floating, floating.app
    ui = app.workspace.ui
    node = next(n for n in v.session.nodes if n.name == "05_multi_joint")
    joint = next(j for j in v.session.joints if j.name == "05_multi_revolute_z")
    app.tool = "rotate"
    assert v.session.submit(cmd.SelectNode(node.node_id)).ok
    v.frames()
    picker = _window("viewport_joint_gizmo")
    close = ui.hits[f"floating-close/{picker.id_}"]
    heading = ui.hits["floating-heading/" + picker.name]
    assert heading[0] + heading[2] < close[0]
    assert close[0] + close[2] <= picker.pos.x + picker.size.x - picker.window_padding.x + 1
    row = ui.hits[f"viewport-joint-{joint.joint_id}"]
    v.click_rect(row)
    assert picker.active
    before = v.session.frame.qpos.copy()
    v.click_rect(row)
    v.frames(30)
    assert not picker.active
    assert v.session.selected_node.node_id == node.node_id
    assert app.gizmo.selected_joint_id(node.body_index) == joint.joint_id
    assert app.gizmo.last_verdict.ok and app.gizmo.visible
    assert app._pending_joint_focus_id is None
    np.testing.assert_array_equal(v.session.frame.qpos, before)
    v.save_ui("joint-focused")

    # A new selection reopens the chooser, while Close keeps the current target.
    assert v.session.submit(cmd.Select(0)).ok
    v.frames()
    assert v.session.submit(cmd.SelectNode(node.node_id)).ok
    v.frames()
    assert picker.active
    v.click_rect(ui.hits[f"floating-close/{picker.id_}"])
    v.frames(30)
    assert not picker.active
    assert app.gizmo.last_verdict.ok and app.gizmo.visible
    app.gizmo._hovered = GizmoHandle.ROTATE_Z
    edit = app.gizmo.precise_input(v.session)
    assert edit is not None
    app._begin_precise_gizmo_input(edit)
    v.frames()
    v.event("add_input_characters_utf8", "15")
    v.key(imgui.Key.enter)
    assert v.session.frame.qpos[joint.qpos_adr] == pytest.approx(
        before[joint.qpos_adr] + np.deg2rad(15)
    )
    assert not picker.active


def test_joint_angle_form_keeps_aligned_unit_switch_and_conversion(floating):
    v, app = floating, floating.app
    ui, scale = app.workspace.ui, v.window.style_scale
    node = next(n for n in v.session.nodes if n.name == "05_multi_joint")
    joint = next(j for j in v.session.joints if j.name == "05_multi_revolute_z")
    assert v.session.submit(cmd.SelectNode(node.node_id)).ok
    app.tool = "rotate"
    app.gizmo.select_joint(node.body_index, joint.joint_id)
    v.frames()
    picker = _window("viewport_joint_gizmo")
    v.event("add_mouse_pos_event", picker.pos.x + picker.size.x + 8 * scale, picker.pos.y)
    app.gizmo._hovered = GizmoHandle.ROTATE_Z
    edit = app.gizmo.precise_input(v.session)
    assert edit is not None and edit.unit == "°"
    app._begin_precise_gizmo_input(edit)
    app._precise_gizmo_value = 30
    v.frames()
    value = ui.hits["precise-gizmo-value"]
    degrees, radians = (ui.hits[f"precise-gizmo-angle-unit-{i}"] for i in (0, 1))
    assert value[1] == pytest.approx(degrees[1], abs=1)
    assert value[3] == pytest.approx(degrees[3], abs=1)
    assert degrees[1:] == pytest.approx(radians[1:], abs=1)
    assert degrees[0] > value[0] + value[2]
    v.save_ui("joint-angle-degrees")
    v.click_rect(radians)
    assert app._precise_gizmo_angle_unit == "radians"
    assert app._precise_gizmo_value == pytest.approx(np.pi / 6)
    v.click_rect(degrees)
    assert app._precise_gizmo_value == pytest.approx(30)
    v.key(imgui.Key.u)
    assert app._precise_gizmo_angle_unit == "radians"
    assert app._precise_gizmo_value == pytest.approx(np.pi / 6)
    v.save_ui("joint-angle-radians")
    original = v.session.frame.qpos[joint.qpos_adr]
    v.key(imgui.Key.enter)
    assert app._precise_gizmo_edit is None
    assert v.session.frame.qpos[joint.qpos_adr] == pytest.approx(original + np.pi / 6)


def test_camera_units_share_the_numeric_field_and_keep_a_space(floating):
    v, app = floating, floating.app
    ui = app.workspace.ui
    ui.design.runtime.observe_components = True
    app.workspace.activate("Cameras")
    camera = app.panels.get("camera")
    camera._angular_degrees = False
    v.frames()

    def caption():
        return next(
            row
            for row in ui.design.primitives.observed
            if "camera-yaw" in row["key"] and row["kind"] == "input_shell"
        )

    row = caption()
    text, unit, field = row["textRect"], row["unitRect"], row["numericRect"]
    assert row["unit"] == "rad"
    assert unit[0] > text[0] + text[2]
    assert unit[0] + unit[2] < field[0] + field[2]
    assert row["valueTextPosition"][1] == row["unitTextPosition"][1]
    v.save_ui("camera-radians")
    yaw = app.camera.yaw
    v.click_rect(ui.hits[row["key"].removesuffix("-shell") + "-unit"])
    assert camera._angular_degrees
    assert app.camera.yaw == yaw
    assert caption()["unit"] == "°"
    v.save_ui("camera-degrees")


def test_application_dialogs_and_rename_use_the_same_surface(floating, tmp_path):
    v, app = floating, floating.app
    scale = v.window.style_scale
    app.workspace.menus.dialog_pending = "Interaction guide"
    v.frames()
    guide = _window("##workspace-dialog")
    done = app.workspace.ui.hits["done-guide"]
    inset = guide.pos.x + guide.size.x - done[0] - done[2]
    assert inset == pytest.approx(12 * scale)
    assert guide.pos.y + guide.size.y - done[1] - done[3] == pytest.approx(inset, abs=1)
    assert guide.window_rounding - inset == pytest.approx(4 * scale)
    v.save_ui("interaction-guide")
    v.click_rect(app.workspace.ui.hits["done-guide"])
    node = next(n for n in v.session.nodes if n.name == "05_multi_joint")
    assert v.session.submit(cmd.SelectNode(node.node_id)).ok
    v.frames()
    app.request_model_rename(node.node_id)
    v.frames()
    _assert_surface(_window("##Popup_"), scale)
    v.save_ui("rename")
    v.key(imgui.Key.escape)
    assert v.session.node(node.node_id).name == "05_multi_joint"
    app.request_model_rename(node.node_id)
    v.frames()
    v.event("add_input_characters_utf8", "renamed_joint")
    v.key(imgui.Key.enter)
    assert v.session.node(node.node_id).name == "renamed_joint"

    app._report_model_error("The selected model references an unavailable mesh file.")
    v.frames()
    _assert_surface(_window("File operation failed"), scale)
    v.save_ui("file-error")
    v.key(imgui.Key.escape)

    app._pending_document_action = ("new_scene", None)
    v.frames()
    _assert_surface(_window("Unsaved changes"), scale)
    v.save_ui("unsaved-changes")
    v.key(imgui.Key.escape)
    assert app._pending_document_action is None
    assert v.session.node(node.node_id) is not None

    app._pending_pose_save = (tmp_path / "pose.xml", None)
    v.frames()
    _assert_surface(_window("Save current pose"), scale)
    v.save_ui("save-pose")
    v.key(imgui.Key.escape)
    assert app._pending_pose_save is None
    assert not (tmp_path / "pose.xml").exists()

    app._missing_resources = (
        MissingResource(
            0,
            1,
            "Robot",
            "meshes/" + "long_directory_name/" * 4 + "robot.xml",
            tmp_path / "robot.xml",
        ),
    )
    app._open_resource_repair_popup = True
    v.frames()
    _assert_surface(_window("Missing Resources"), scale)
    locate_key = app.localizer.text("Locate...") + "##missing-resource-0"
    assert app.workspace.ui.hits[locate_key][2] > 40 * scale
    v.save_ui("missing-resources")
    v.key(imgui.Key.escape)
    assert not app._missing_resources


def test_source_component_loading_and_countdown_surfaces(floating, monkeypatch, tmp_path):
    v, app = floating, floating.app
    scale, ui = v.window.style_scale, app.workspace.ui
    v.event("add_mouse_pos_event", -100, -100)
    v.save_ui("playback")
    x, y, width, height = ui.hits["go-start"]
    v.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    v.frames(45)
    v.save_ui("playback-reset")
    v.event("add_mouse_pos_event", -100, -100)
    assert v.session.submit(cmd.NewScene()).ok
    assert app.add_model(resolve("joint_gizmo")).ok
    model = next(node for node in v.session.nodes if node.type is NodeType.MODEL)
    assert v.session.submit(cmd.SelectNode(model.node_id)).ok
    v.frames()
    panel = app.panels.get("inspector")
    panel._source_model_id = model.model_id
    panel._source_text = v.session.adapter.scene_model_source(model.model_id)
    panel._open_source_popup = True
    v.frames()
    _assert_surface(_window("MJCF Source"), scale)
    v.save_ui("model-source")
    v.click_rect(ui.hits[app.localizer.text("Cancel")])
    assert not any(
        w.active and "MJCF Source" in w.name for w in imgui.get_current_context().windows
    )

    assert v.session.submit(cmd.AddModelComponent(model.model_id, "sensor", "jointpos", "angle")).ok
    v.frames()
    component = v.session.model_components(model.model_id, "sensor")[0]
    panel._begin_component_edit(component)
    v.frames()
    _assert_surface(_window("Model Component"), scale)
    fields = [("name", "##component-name")] + [
        (field[0], f"##component-field-{index}")
        for index, field in enumerate(panel._component_fields)
    ]
    first_x, first_width = ui.hits[fields[0][1]][0::2]
    for label, key in fields:
        value = ui.hits[key]
        caption = ui.hits["component_fields/property-label-" + label]
        assert value[0] == pytest.approx(first_x, abs=scale)
        assert value[2] == pytest.approx(first_width, abs=scale)
        assert caption[1] + caption[3] / 2 == pytest.approx(value[1] + value[3] / 2, abs=scale)
    cancel, apply = (ui.hits[app.localizer.text(label)] for label in ("Cancel", "Apply"))
    assert cancel[1:] == pytest.approx(apply[1:])
    v.save_ui("model-component")
    v.click_rect(ui.hits[app.localizer.text("Cancel")])
    assert panel._component_edit is None

    # Hold the real loader before it mutates Session, so its busy UI is observable.
    ready = Event()
    load = app._load_model

    def held_load(command):
        assert ready.wait(10)
        return load(command)

    monkeypatch.setattr(app, "_load_model", held_load)
    app._queue_scene_open(resolve("joint_gizmo"))
    try:
        v.frames()
        _assert_surface(_window("model_loading"), scale)
        v.save_ui("loading")
    finally:
        ready.set()
    deadline = monotonic() + 10
    while app._model_load_future is not None and monotonic() < deadline:
        v.frames(1)
    assert app._model_load_future is None
    v.frames()

    path = tmp_path / "canceled.mp4"
    app.start_recording(path, countdown=30)
    v.frames()
    countdown = _window("recording_countdown")
    _assert_surface(countdown, scale)
    assert countdown.pos.y >= app._playback_widget_rect[3] + 7 * scale
    cancel = ui.hits[app.localizer.text("Cancel")]
    style = ui.component_style("button")
    caption_width = ui.measure(
        app.localizer.text("Cancel"), style.get("fontSize", 14), translate=False, style=style
    )
    assert cancel[2] >= caption_width + 15 * scale
    v.save_ui("recording-countdown")
    v.click_rect(ui.hits[app.localizer.text("Cancel")])
    assert app.recording.phase is RecordingPhase.IDLE
    assert not path.exists()
