"""Input ownership regressions exercised through the real viewer frame loop."""

from __future__ import annotations

import pytest
from imgui_bundle import imgui

from mojive import InputAction, InputClaim, Scene, build_scene
from mojive import commands as cmd

pytestmark = pytest.mark.gpu


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    box = scene.box(name="original")
    with build_scene(scene, vsync=False, show_window=False, width=1280, height=800) as viewer:
        for _ in range(12):
            viewer.sync()
        viewer.session.submit(cmd.Select(box.object_id))
        yield viewer, box


def press(viewer, key, *, ctrl=False, shift=False):
    io = imgui.get_io()
    io.add_key_event(imgui.Key.mod_ctrl, ctrl)
    io.add_key_event(imgui.Key.mod_shift, shift)
    io.add_key_event(key, True)
    viewer.sync()
    io.add_key_event(key, False)
    io.add_key_event(imgui.Key.mod_ctrl, False)
    io.add_key_event(imgui.Key.mod_shift, False)
    viewer.sync()


@pytest.mark.parametrize("claim", [InputClaim(keyboard=True), InputClaim(keys={"delete"})])
def test_claimed_delete_preserves_selected_object(viewer, claim):
    viewer, _ = viewer
    calls = []

    def handler(context):
        calls.append(context)
        return claim

    viewer.set_input_handler(handler)
    press(viewer, imgui.Key.delete)
    assert viewer.session.source.instance_count == 1
    assert len(calls) == 2
    viewer.set_input_handler(None)
    press(viewer, imgui.Key.delete)
    assert viewer.session.source.instance_count == 0


@pytest.mark.parametrize(
    "claim", [InputClaim(keyboard=True), InputClaim(keys={"z", "d"}), InputClaim(keys={"ctrl"})]
)
def test_claimed_editor_chords_preserve_history_and_scene(viewer, claim):
    viewer, box = viewer
    assert viewer.session.submit(cmd.RenameSceneEntity(box.object_id, "renamed")).ok
    viewer.set_input_handler(lambda context: claim)
    press(viewer, imgui.Key.z, ctrl=True)
    assert viewer.session.selected_node.name == "renamed"
    press(viewer, imgui.Key.d, ctrl=True)
    assert viewer.session.source.instance_count == 1
    viewer.set_input_handler(None)
    press(viewer, imgui.Key.z, ctrl=True)
    assert viewer.session.selected_node.name == "original"
    press(viewer, imgui.Key.d, ctrl=True)
    assert viewer.session.source.instance_count == 2


def test_keyboard_claim_blocks_quit_capture_and_document_shortcuts(viewer, monkeypatch):
    viewer, _ = viewer
    actions = []
    monkeypatch.setattr(viewer.app, "_request_document_action", lambda *args: actions.append(args))
    monkeypatch.setattr(viewer.app, "_open_scene_dialog", lambda *args: actions.append(args))
    monkeypatch.setattr(viewer.app, "request_capture", lambda **kwargs: actions.append(kwargs))
    viewer.set_input_handler(lambda context: InputClaim(keyboard=True))
    for key in (imgui.Key.q, imgui.Key.n, imgui.Key.o, imgui.Key.s):
        press(viewer, key, ctrl=True)
    press(viewer, imgui.Key.p, ctrl=True, shift=True)
    assert actions == []
    viewer.set_input_handler(None)
    press(viewer, imgui.Key.q, ctrl=True)
    assert actions == [("quit",)]


def test_control_can_be_remapped_to_a_tool_without_leaking_chord_letters(viewer):
    viewer, _ = viewer
    viewer.set_gizmo_mode("translate")
    viewer.configure_input_binding(InputAction.GIZMO_ROTATE, "ctrl")
    press(viewer, imgui.Key.left_ctrl, ctrl=True)
    assert viewer.gizmo_mode == "rotate"
    viewer.set_gizmo_mode("translate")
    viewer.set_input_handler(lambda context: InputClaim(keys={"ctrl"}))
    press(viewer, imgui.Key.left_ctrl, ctrl=True)
    assert viewer.gizmo_mode == "translate"
    viewer.set_input_handler(None)
    press(viewer, imgui.Key.r, ctrl=True)
    assert viewer.gizmo_mode == "translate"


@pytest.mark.parametrize("key", ["slash", "shift"])
def test_help_alias_honors_claimed_physical_keys(viewer, key):
    viewer, _ = viewer
    help_panel = viewer.panels.get("help")
    assert help_panel is not None
    help_panel.open = False
    viewer.set_input_handler(lambda context: InputClaim(keys={key}))
    press(viewer, imgui.Key.slash, shift=True)
    assert not help_panel.open
    viewer.set_input_handler(None)
    press(viewer, imgui.Key.slash, shift=True)
    assert help_panel.open
