"""Input ownership and key state without a native window or GPU context."""

from __future__ import annotations

import pytest
from imgui_bundle import imgui

from mojive.ui import window as wm


@pytest.fixture(params=[False, True], ids=["physical-modifiers", "mac-shortcut-modifiers"])
def adapter(monkeypatch, request):
    for name in ("glfw", "imgui"):
        monkeypatch.setattr(wm, name, getattr(wm, name))
    wm._load_window_deps()
    for name in (
        "key",
        "cursor_pos",
        "mouse_button",
        "char",
        "scroll",
        "window_focus",
        "cursor_enter",
    ):
        monkeypatch.setattr(wm.glfw, f"set_{name}_callback", lambda *_args: None)
    context = imgui.create_context()
    io = imgui.get_io()
    io.config_mac_osx_behaviors = request.param
    io.display_size = (800, 600)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        yield wm.GlfwInputAdapter(None), context
    finally:
        imgui.destroy_context(context)


def frame():
    imgui.new_frame()
    imgui.end_frame()


def test_unicode_text_reaches_the_owner_even_when_another_context_is_current(adapter):
    input, owner = adapter
    foreign = imgui.create_context()
    try:
        for codepoint in (ord("关"), 0x1F642, 0, 0xD800, 0x110000):
            input.char_callback(None, codepoint)
        assert not foreign.input_events_queue
        imgui.set_current_context(owner)
        imgui.new_frame()
        assert list(input.io.input_queue_characters) == [ord("关"), 0x1F642]
        imgui.end_frame()
    finally:
        imgui.destroy_context(foreign)
        imgui.set_current_context(owner)


@pytest.mark.parametrize("modifier", ["CONTROL", "SHIFT", "ALT", "SUPER"])
def test_modifier_stays_pressed_until_both_sides_are_released(adapter, modifier):
    input, _owner = adapter
    left = getattr(wm.glfw, f"KEY_LEFT_{modifier}")
    right = getattr(wm.glfw, f"KEY_RIGHT_{modifier}")
    name = "ctrl" if modifier == "CONTROL" else modifier.lower()
    if input.io.config_mac_osx_behaviors:
        name = {"ctrl": "super", "super": "ctrl"}.get(name, name)
    attribute = "key_" + name
    for key, state in ((left, wm.glfw.PRESS), (right, wm.glfw.PRESS), (left, wm.glfw.RELEASE)):
        input.keyboard_callback(None, key, 0, state, 0)
        frame()
    assert getattr(input.io, attribute)
    input.keyboard_callback(None, right, 0, wm.glfw.RELEASE, 0)
    frame()
    assert not getattr(input.io, attribute)


def test_focus_loss_clears_held_input_in_the_owner(adapter):
    input, _owner = adapter
    input.keyboard_callback(None, wm.glfw.KEY_LEFT_CONTROL, 0, wm.glfw.PRESS, 0)
    input.mouse_button_callback(None, 0, wm.glfw.PRESS, 0)
    frame()
    control = "key_super" if input.io.config_mac_osx_behaviors else "key_ctrl"
    assert getattr(input.io, control) and input.io.mouse_down[0]
    input.focus_callback(None, False)
    frame()
    assert not getattr(input.io, control) and not input.io.mouse_down[0]
