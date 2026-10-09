"""Reference styles with production scene interactions and settings actions."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from examples.ui_design.native.docking_study import Study
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.app.ui.window import create_window
from mojive.interaction.gizmo import (
    AXIS_END,
    SCREEN_RING_RADIUS,
    GizmoHandle,
    GizmoStyle,
    project,
    world_scale,
)
from mojive.scene.queries import node_world_pose
from mojive.session.model_edits import model_edit_scope
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/ui_design/native"


@pytest.fixture
def reference(backend_name, monkeypatch, tmp_path, request):
    scale = getattr(request, "param", 1.0)
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    window = create_window(
        WindowConfig(
            width=round(1440 * scale),
            height=round(900 * scale),
            ui_scale=scale,
            docking=True,
            ini_path="",
            show_on_start=False,
        ),
        backend_name,
    )
    study = Study(window, backend_name, output_directory=tmp_path)
    events = []
    process_inputs = window._input.process_inputs

    def inputs():
        process_inputs()
        io = imgui.get_io()
        for method, args in events:
            getattr(io, method)(*args)
        events.clear()
        io.delta_time = 1 / 60

    monkeypatch.setattr(window._input, "process_inputs", inputs)

    def frame(count=1):
        pixels = None
        for _ in range(count):
            pixels = study.draw(readback=True)
        return np.asarray(pixels)[::-1].copy()

    def event(method, *args):
        events.append((method, args))
        return frame()

    def click(point):
        event("add_mouse_pos_event", *point)
        event("add_mouse_button_event", 0, True)
        return event("add_mouse_button_event", 0, False)

    study.frame = frame
    study.event = event
    study.click = click
    study.backend_name = backend_name
    frame(8)
    try:
        yield study
    finally:
        study.close()
        window.close()


def save(study, name, pixels=None):
    path = OUTPUT / study.backend_name / "interactions" / (name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(study.frame() if pixels is None else pixels).save(path)


def wait_edits(study):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        study.frame()
        if not study.preview._model_load_future and not study.preview._model_load_queue:
            study.frame(3)
            return
        time.sleep(0.005)
    raise AssertionError("Pending source edits did not finish")


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_pose_and_actuator_value_rails_write_session_values(reference):
    study, app = reference, reference.preview
    study.activate("Inspector")
    study.frame(3)
    rect = study.ui.hits["tab-Joint"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    study.frame(2)

    def drag_rail(key):
        x, y, width, height = study.ui.hits[key + "-track"]
        entry = study.ui.hits[key + "-value"]
        assert height == pytest.approx(28 * study.s, abs=1)
        assert x + width < entry[0]
        start = (x + width / 2, y + height / 2)
        study.event("add_mouse_pos_event", *start)
        study.event("add_mouse_button_event", 0, True)
        study.event("add_mouse_pos_event", x + width * 0.8, start[1])
        study.event("add_mouse_button_event", 0, False)

    hinge = next(joint for joint in app.session.joints if joint.name == "hinge_limited")
    before = app.session.frame.qpos.copy()
    drag_rail("##pose-hinge")
    after = app.session.frame.qpos.copy()
    assert after[hinge.qpos_adr] > before[hinge.qpos_adr] + 0.1
    assert after[hinge.qpos_adr] < np.radians(60)
    np.testing.assert_allclose(np.delete(after, hinge.qpos_adr), np.delete(before, hinge.qpos_adr))
    assert app.document["pose"]["hinge"] == pytest.approx(np.degrees(after[hinge.qpos_adr]))
    app.set_pose_value("hinge", np.degrees(before[hinge.qpos_adr]))
    study.frame(2)
    np.testing.assert_allclose(app.session.frame.qpos, before, atol=1e-5)

    study.activate("Control")
    study.frame(3)
    actuator = app.session.actuators[0]
    before = app.session.frame.ctrl.copy()
    drag_rail("##ctrl-" + str(actuator.ctrl_address))
    after = app.session.frame.ctrl.copy()
    assert after[actuator.ctrl_address] > before[actuator.ctrl_address] + 0.1
    assert after[actuator.ctrl_address] < actuator.ctrl_range[1]
    np.testing.assert_allclose(
        np.delete(after, actuator.ctrl_address), np.delete(before, actuator.ctrl_address)
    )
    save(study, f"pose-and-control-rails-{study.s:g}")


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
@pytest.mark.parametrize(
    "tool,handle",
    (("move", GizmoHandle.X), ("rotate", GizmoHandle.ROTATE_SCREEN), ("scale", GizmoHandle.X)),
)
def test_flat_drag_feedback_and_undo(reference, tool, handle):
    study, app = reference, reference.preview
    app.selected = "free_box" if tool == "scale" else "free_body"
    app.tool = tool
    study.frame(3)
    node = app.session.selected_node
    before = (
        app.session.source.geom_size.copy()
        if tool == "scale"
        else np.concatenate([part.ravel() for part in node_world_pose(app.session, node)])
    )
    app.gizmos = False
    hidden = study.frame(2)
    app.gizmos = True
    visible = study.frame(2)
    assert np.count_nonzero(np.max(abs(visible.astype(int) - hidden), axis=2) > 10) > 100
    gizmo = app.gizmo
    frame, camera, rect = gizmo._frame, app._camera_view(), app._viewport_rect
    assert frame.style is GizmoStyle.FLAT and app.backend._gizmo is None
    length = world_scale(camera, frame.position, rect[3], frame.size_px)
    center = project(camera, (frame.position,), rect)[0, :2]
    if tool == "rotate":
        radius = length * SCREEN_RING_RADIUS
        direction = camera.eye - frame.position
        direction /= np.linalg.norm(direction)
        right = np.cross(direction, camera.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        start = project(camera, (frame.position + right * radius,), rect)[0, :2]
        end = project(
            camera, (frame.position + (right * np.cos(0.3) + up * np.sin(0.3)) * radius,), rect
        )[0, :2]
    else:
        start = project(
            camera, (frame.position + frame.rotation[:, 0] * length * AXIS_END * 0.85,), rect
        )[0, :2]
        direction = start - center
        direction /= np.linalg.norm(direction)
        end = start + direction * 30 * study.s
    study.event("add_mouse_pos_event", *start)
    study.frame(2)
    assert gizmo.hovered_handle is handle
    study.event("add_mouse_button_event", 0, True)
    assert gizmo.using
    feedback = study.event("add_mouse_pos_event", *end)
    assert gizmo.value_label
    study.event("add_mouse_button_event", 0, False)
    wait_edits(study)
    after = (
        app.session.source.geom_size.copy()
        if tool == "scale"
        else np.concatenate(
            [part.ravel() for part in node_world_pose(app.session, app.session.node(node.node_id))]
        )
    )
    assert not np.allclose(after, before)
    assert app.session.can_undo
    assert app.session.submit(cmd.Undo()).ok
    restored = (
        app.session.source.geom_size
        if tool == "scale"
        else np.concatenate(
            [part.ravel() for part in node_world_pose(app.session, app.session.node(node.node_id))]
        )
    )
    np.testing.assert_allclose(restored, before, atol=1e-5)
    assert app.session.submit(cmd.Redo()).ok
    save(study, f"{tool}-drag-{study.s:g}", feedback)


@pytest.mark.parametrize("name", ("hinge_body", "slide_body"))
def test_joint_ranges_endpoints_and_precise_input(reference, name):
    study, app = reference, reference.preview
    app.selected = name
    app.tool = "rotate" if name == "hinge_body" else "move"
    study.frame(3)
    gizmo = app.gizmo
    target, reason = gizmo._joint_target(app.session, app.session.selected_node)
    assert target is not None, reason
    assert len(gizmo.joint_limit_hits) == 2
    lower, _upper = gizmo.joint_limit_hits
    study.click(lower.tick_end)
    study.frame(2)
    assert app.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(lower.value), (
        gizmo.hovered_joint_limit,
        app._state,
        lower,
        app.router.claim,
    )
    app.set_pose_value("hinge" if name == "hinge_body" else "slide", 0)
    study.frame(2)
    # The same shared precision request is exposed through the native popup.
    app.gizmo._hovered = GizmoHandle.ROTATE_Z if name == "hinge_body" else GizmoHandle.Z
    edit = gizmo.precise_input(app.session)
    assert edit is not None
    app._begin_precise_gizmo_input(edit)
    study.frame(2)
    assert app._precise_gizmo_edit is not None
    save(study, name + "-precise")
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    save(study, name + "-range")


def test_picking_focus_navigation_helpers_and_chrome_ownership(reference):
    study, app = reference, reference.preview
    app.tool = "select"
    app.selected = None
    study.frame(3)
    node = next(n for n in app.session.nodes if n.name == "free_body")
    point = project(
        app._camera_view(), (node_world_pose(app.session, node)[0],), app._viewport_rect
    )[0, :2]
    study.click(point)
    assert app.session.selected_node.node_id == node.node_id
    study.click(point)
    assert app.camera.animating
    app.camera.advance(1.0, app.camera_out)
    study.frame(3)
    x, y, w, h = app._viewport_rect
    empty = (x + w * 0.55, y + h * 0.75)
    study.event("add_mouse_pos_event", *empty)
    before = app.camera.view().eye.copy()
    study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_pos_event", empty[0] + 40, empty[1] + 25)
    study.event("add_mouse_button_event", 0, False)
    assert not np.allclose(app.camera.view().eye, before)
    before = app.camera.distance
    study.event("add_mouse_wheel_event", 0, 1)
    assert app.camera.distance != before
    # Clicking the new tool capsule must not start orbit or selection.
    camera = app.camera.view().eye.copy()
    rect = study.ui.hits["tool-move"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    np.testing.assert_allclose(app.camera.view().eye, camera)
    assert app.tool == "move"
    app.tool = "select"
    app._frame_scene(animate=False)
    study.frame(3)
    camera_node = next(n for n in app.session.nodes if n.name == "inspection")
    point = project(
        app._camera_view(), (node_world_pose(app.session, camera_node)[0],), app._viewport_rect
    )[0, :2]
    study.click(point)
    assert app.session.selected_node.node_id == camera_node.node_id
    visible = study.frame(2)
    app.set_viewport_layers(replace(app.viewport_layers, helpers=False), persist=False)
    hidden = study.frame(2)
    assert np.count_nonzero(np.max(abs(visible.astype(int) - hidden), axis=2) > 10) > 100
    app.set_viewport_layers(replace(app.viewport_layers, helpers=True), persist=False)
    save(study, "camera-helper")
    light = next(n for n in app.session.nodes if n.name == "sun")
    position, _rotation = node_world_pose(app.session, light)
    app.camera.adopt(
        replace(app._camera_view(), eye=position + np.array([4, -6, 4]), target=position.copy())
    )
    app.selected = None
    study.frame(3)
    point = project(app._camera_view(), (position,), app._viewport_rect)[0, :2]
    study.click(point)
    assert app.session.selected_node.node_id == light.node_id
    visible = study.frame(2)
    app.set_viewport_layers(replace(app.viewport_layers, helpers=False), persist=False)
    hidden = study.frame(2)
    assert np.count_nonzero(np.max(abs(visible.astype(int) - hidden), axis=2) > 10) > 100
    app.set_viewport_layers(replace(app.viewport_layers, helpers=True), persist=False)
    save(study, "light-helper")
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    assert app.session.selected_node is None


def test_new_settings_entry_categories_language_and_panel_style(reference):
    study, app = reference, reference.preview
    rect = study.ui.hits["preferences"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    study.frame(3)
    assert study.open["Settings"]
    assert app.panels.get("settings") is study.settings
    assert "settings-General" in study.ui.hits
    assert "tab-Properties" in study.ui.hits
    save(study, "settings-general")
    for category in ("Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals"):
        rect = study.ui.hits["settings-" + category]
        study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
        assert study.settings._category == category
        study.frame()
    app._open_recording_settings()
    assert study.settings._category == "Recording"
    study.settings.show_category("General")
    app.set_language("zh_CN")
    study.frame(3)
    assert app.localizer.text("Settings") != "Settings"
    assert app.window.font_report.cjk_path
    save(study, "settings-chinese")
    app.set_language("en")
    app.reset_layout(persist=False)
    study.frame(3)
    assert study.workspace == "Edit" and not study.open["Settings"]


def test_inspector_dimensions_apply_discard_and_hidden_chrome(reference):
    study, app = reference, reference.preview
    study.menus.dispatch("create:Box")
    study.frame(3)
    assert app.entity["scalable"]
    before = app.session.source.geom_size.copy()
    with model_edit_scope(app.session, app._intercept_model_edit):
        app.entity["scale"] = [1.5, 1.0, 1.0]
        app.apply_document_edits()
    assert app.model_edits.active, (app.session.last_message, app.entity, app.model_edits.commands)
    study.frame(2)
    assert app.model_edits.active
    save(study, "pending-dimensions")
    rect = study.ui.hits["apply-model-edits"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    wait_edits(study)
    assert not app.model_edits.active
    after = app.session.source.geom_size.copy()
    assert not np.allclose(after, before)
    assert app.session.submit(cmd.Undo()).ok
    np.testing.assert_allclose(app.session.source.geom_size, before)
    study.frame(2)
    with model_edit_scope(app.session, app._intercept_model_edit):
        app.entity["scale"] = [2.0, 1.0, 1.0]
        app.apply_document_edits()
    app.set_viewport_layers(replace(app.viewport_layers, viewport_ui=False), persist=False)
    study.frame(2)
    assert app.model_edits.active
    assert "discard-model-edits" in study.ui.hits
    rect = study.ui.hits["discard-model-edits"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    assert not app.model_edits.active
    np.testing.assert_allclose(app.session.source.geom_size, before)


def test_empty_scene_authoring_materials_without_joints(reference):
    study, app = reference, reference.preview
    assert app.session.submit(cmd.NewScene()).ok
    study.frame(3)
    assert not app.session.joints and not app.session.actuators
    app._next_entity_color = lambda: (0.25, 0.5, 0.75, 1.0)
    study.menus.dispatch("create:Box")
    study.frame(3)
    assert app.entity["posable"] and app.entity["scalable"] and app.entity["colorable"]
    app.tool = "move"
    study.frame(2)
    assert app.gizmo._frame is not None
    assert app.gizmo._frame.style is GizmoStyle.FLAT
    study.activate("Assets")
    study.frame(3)
    before = app.session.source.geom_rgba.copy()
    rect = study.ui.hits["material-Sage ceramic"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    rect = study.ui.hits["apply-material"]
    study.click((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    assert not np.allclose(app.session.source.geom_rgba, before)
    np.testing.assert_allclose(app.session.source.geom_rgba[0, :3], np.array([156, 191, 141]) / 255)
    assert app.session.submit(cmd.Undo()).ok
    np.testing.assert_allclose(app.session.source.geom_rgba, before)
    study.activate("Control")
    study.frame(3)
    assert "reset-actuators" in study.ui.hits
