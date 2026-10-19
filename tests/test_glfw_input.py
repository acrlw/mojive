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


def test_queued_pointer_click_uses_event_position_in_the_owning_context(adapter, monkeypatch):
    input, owner = adapter
    # The current OS cursor may already have moved beyond the queued press.
    monkeypatch.setattr(wm.glfw, "get_cursor_pos", lambda _window: (373, -64))
    foreign = imgui.create_context()
    try:
        input.mouse_callback(None, 176, 420)
        input.mouse_button_callback(None, 0, wm.glfw.PRESS, 0)
        input.mouse_callback(None, 220, 460)
        input.mouse_button_callback(None, 0, wm.glfw.RELEASE, 0)
        assert not foreign.input_events_queue
        imgui.set_current_context(owner)
        frame()
        assert input.io.mouse_clicked[0]
        assert tuple(input.io.mouse_pos) == (176, 420)
        for _ in range(3):
            frame()
        assert tuple(input.io.mouse_pos) == (220, 460)
        assert not input.io.mouse_down[0]
    finally:
        imgui.destroy_context(foreign)
        imgui.set_current_context(owner)


def test_cursor_reentry_queries_position_when_no_motion_event_is_available(adapter, monkeypatch):
    input, _owner = adapter
    queries = []

    def position(window):
        queries.append(window)
        return (96, 118)

    monkeypatch.setattr(wm.glfw, "get_cursor_pos", position)
    input.cursor_enter_callback(None, True)
    frame()
    assert tuple(input.io.mouse_pos) == (96, 118)
    input.cursor_enter_callback(None, False)
    frame()
    assert input.io.mouse_pos.x < -1e30 and input.io.mouse_pos.y < -1e30
    assert queries == [input.window]
