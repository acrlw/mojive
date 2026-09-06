"""Input ownership regressions exercised through the real viewer frame loop."""

from __future__ import annotations

import pytest
from imgui_bundle import imgui

from mojive import InputAction, InputClaim, Scene, build_scene
from mojive import commands as cmd

pytestmark = pytest.mark.gpu


@pytest.fixture(params=[False, True], ids=["standard-keys", "macos-keys"])
def viewer(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    box = scene.box(name="original")
    with build_scene(scene, vsync=False, show_window=False, width=1280, height=800) as viewer:
        imgui.get_io().config_mac_osx_behaviors = request.param
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


def test_input_hook_observes_physical_control_independently_of_command(viewer):
    viewer, _ = viewer
    observed = []

    def handler(context):
        observed.append(
            (
                context.key_down("ctrl"),
                context.key_down("super"),
                context.key_pressed("ctrl"),
                context.key_released("ctrl"),
            )
        )
        return InputClaim(keys={"ctrl"})

    viewer.set_input_handler(handler)
    press(viewer, imgui.Key.left_ctrl, ctrl=True)
    assert observed == [(True, False, True, False), (False, False, False, True)]


@pytest.mark.parametrize("button", [0, 1], ids=["left", "right"])
def test_glfw_control_click_preserves_physical_button_and_native_key_mode(viewer, button):
    import glfw

    viewer, _ = viewer
    window = viewer.window._window
    keyboard = glfw.set_key_callback(window, None)
    mouse = glfw.set_mouse_button_callback(window, None)
    glfw.set_key_callback(window, keyboard)
    glfw.set_mouse_button_callback(window, mouse)
    assert keyboard is not None and mouse is not None
    io = imgui.get_io()
    macos = io.config_mac_osx_behaviors
    observed = []

    def handler(context):
        observed.append(
            (
                context.key_down("ctrl"),
                context.mouse_down(0),
                context.mouse_down(1),
                context.mouse_clicked(button),
                context.mouse_released(button),
            )
        )
        return InputClaim(pointer=True)

    viewer.set_input_handler(handler)
    # Both events may arrive between frames; modifier lookup must see queued keys.
    keyboard(window, glfw.KEY_LEFT_CONTROL, 0, glfw.PRESS, glfw.MOD_CONTROL)
    mouse(window, button, glfw.PRESS, glfw.MOD_CONTROL)
    viewer.sync()
    # Releasing Control first must not leave an aliased right button held.
    keyboard(window, glfw.KEY_LEFT_CONTROL, 0, glfw.RELEASE, 0)
    mouse(window, button, glfw.RELEASE, 0)
    viewer.sync()
    assert observed == [
        (True, button == 0, button == 1, True, False),
        (False, False, False, False, True),
    ]
    assert io.config_mac_osx_behaviors == macos
