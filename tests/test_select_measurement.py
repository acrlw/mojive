"""Closed native selects do not measure menus; opening uses current choice geometry."""

from math import ceil

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace import popup_style
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("activation", ("mouse", "keyboard"))
def test_choice_width_is_deferred_until_native_popup_opens(appearance, monkeypatch, activation):
    ui, _runtime, update, frame = appearance
    imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    choices = ("Short", "A much longer shortcut description")
    calls = []
    original = popup_style.menu_choice_width

    def measure(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(popup_style, "menu_choice_width", measure)
    activate = False

    def paint():
        if activate:
            imgui.get_current_context().nav_activate_id = imgui.get_id("##choices")
            imgui.get_current_context().nav_activate_down_id = imgui.get_id("##choices")
        ui.combo("choices", 40 * ui.s, 40 * ui.s, 160 * ui.s, 0, choices)

    for _ in range(3):
        frame(paint)
    assert not calls
    update({"select_option": {"fontSize": 20, "paddingX": 16}})
    frame(paint)
    assert not calls
    if activation == "keyboard":
        activate = True
        frame(paint)
        activate = False
    else:
        io = imgui.get_io()
        io.add_mouse_pos_event(80 * ui.s, 50 * ui.s)
        frame(paint)
        io.add_mouse_button_event(0, True)
        frame(paint)
        io.add_mouse_button_event(0, False)
        frame(paint)
    for _ in range(3):
        frame(paint)
    popup = next(
        w for w in imgui.get_current_context().windows if w.active and "##Combo_" in w.name
    )
    assert calls
    assert popup.size.x == ceil(calls[-1])
    assert popup.size.x > 160 * ui.s
    imgui.get_io().add_key_event(imgui.Key.escape, True)
    frame(paint)
    imgui.get_io().add_key_event(imgui.Key.escape, False)
    imgui.get_io().add_mouse_pos_event(-100, -100)
    frame(paint)
    calls.clear()
    for _ in range(3):
        frame(paint)
    assert not calls
