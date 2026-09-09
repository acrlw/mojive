"""Configured physical chords must drive real viewport ownership and write-back."""

from dataclasses import replace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import build
from mojive import commands as cmd
from mojive.assets import resolve
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    with build(
        resolve("joint_gizmo"), paused=True, vsync=False, width=1280, height=800, show_window=False
    ) as viewer:
        viewer.app.input_bindings = DEFAULT_INPUT_BINDINGS
        for _ in range(10):
            viewer.sync()
        yield viewer


def point(viewer):
    x, y, w, h = viewer.app._viewport_rect
    return x + w * 0.5, y + h * 0.3


def move(viewer, button, *, dx=50, dy=25):
    io = imgui.get_io()
    x, y = point(viewer)
    io.add_mouse_pos_event(x, y)
    viewer.sync()
    io.add_mouse_button_event(button, True)
    viewer.sync()
    for i in range(1, 7):
        io.add_mouse_pos_event(x + dx * i / 6, y + dy * i / 6)
        viewer.sync()
    io.add_mouse_button_event(button, False)
    viewer.sync()


def test_navigation_preset_changes_mouse_actions_and_hints(viewer):
    viewer.configure_navigation_preset("Blender")
    original = viewer.app.camera.direction().copy()
    pivot = viewer.app.camera.pivot.copy()
    move(viewer, 0)
    np.testing.assert_allclose(viewer.app.camera.direction(), original)
    np.testing.assert_allclose(viewer.app.camera.pivot, pivot)
    move(viewer, 2)
    assert np.linalg.norm(viewer.app.camera.direction() - original) > 0.01
    assert viewer.app.input_bindings.pointer_label("camera.orbit") == "middle"


def test_two_button_pan_assembles_across_frames_and_does_not_pick_on_partial_release(viewer):
    viewer.configure_pointer_binding("camera.pan", ("left+right",))
    io = imgui.get_io()
    x, y = point(viewer)
    direction = viewer.app.camera.direction().copy()
    pivot = viewer.app.camera.pivot.copy()
    io.add_mouse_pos_event(x, y)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(1, True)
    viewer.sync()
    for i in range(1, 7):
        io.add_mouse_pos_event(x + i * 5, y + i * 3)
        viewer.sync()
    np.testing.assert_allclose(viewer.app.camera.direction(), direction, atol=1e-6)
    assert np.linalg.norm(viewer.app.camera.pivot - pivot) > 0.001
    io.add_mouse_button_event(1, False)
    viewer.sync()
    released = viewer.app.camera.direction().copy()
    io.add_mouse_pos_event(x + 70, y + 80)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    np.testing.assert_allclose(viewer.app.camera.direction(), released)
    assert viewer.session.selected_node is None


def test_unsupported_perturbation_never_submits_or_moves_camera(viewer, monkeypatch):
    node = next(node for node in viewer.session.nodes if node.posable)
    viewer.session.submit(cmd.SelectNode(node.node_id))
    monkeypatch.setattr(
        viewer.session.adapter, "caps", replace(viewer.session.adapter.caps, perturb=False)
    )
    viewer.configure_pointer_binding("perturb.translate", ("alt+middle",))
    submitted = []
    original_submit = viewer.session.submit

    def submit(command):
        submitted.append(command)
        return original_submit(command)

    monkeypatch.setattr(viewer.session, "submit", submit)
    camera = viewer.app.camera.view()
    imgui.get_io().add_key_event(imgui.Key.left_alt, True)
    viewer.sync()
    move(viewer, 2)
    imgui.get_io().add_key_event(imgui.Key.left_alt, False)
    viewer.sync()
    assert not any(isinstance(command, cmd.Perturb) for command in submitted)
    np.testing.assert_allclose(viewer.app.camera.view().eye, camera.eye)
    np.testing.assert_allclose(viewer.app.camera.view().target, camera.target)
    assert all(
        not hint.hint_id.startswith("perturb")
        for hint in viewer.app._status_tool_hints(loading=False)
    )


def test_remapped_joint_slider_reset_uses_only_the_configured_button(viewer):
    from mojive.tools.ui_runtime import _activate_panel, _item_center

    viewer.panels.open_panel("Joints")
    for _ in range(3):
        viewer.sync()
    _activate_panel(viewer, "Joints")
    joint = next(joint for joint in viewer.session.joints if joint.type == "hinge")
    address = joint.qpos_adr
    initial = float(viewer.session.frame.qpos[address])
    assert viewer.session.submit(cmd.SetQpos(address, initial + 0.3))
    viewer.sync()
    viewer.configure_pointer_binding("value.reset", ("middle",))
    x, y = _item_center(viewer, "slider_float", f"##joint-qpos-{address}")
    io = imgui.get_io()
    io.add_mouse_pos_event(x, y)
    viewer.sync()
    for button in (1, 2):
        io.add_mouse_button_event(button, True)
        viewer.sync()
        io.add_mouse_button_event(button, False)
        viewer.sync()
        expected = initial + 0.3 if button == 1 else initial
        assert viewer.session.frame.qpos[address] == pytest.approx(expected)


def test_remapped_panel_focus_keeps_the_same_selection(viewer):
    from mojive.tools.ui_runtime import _activate_panel, _item_center

    viewer.panels.open_panel("Joints")
    for _ in range(3):
        viewer.sync()
    _activate_panel(viewer, "Joints")
    joint = viewer.session.joints[0]
    label = f"##joint-select-{joint.joint_id}"
    x, y = _item_center(viewer, "invisible_button", label)
    viewer.configure_pointer_binding("panel.focus", ("middle",))
    io = imgui.get_io()
    io.add_mouse_pos_event(x, y)
    viewer.sync()
    io.add_mouse_button_event(2, True)
    viewer.sync()
    io.add_mouse_button_event(2, False)
    for _ in range(30):
        viewer.sync()
    assert viewer.session.selected_node.joint_index == joint.joint_id
