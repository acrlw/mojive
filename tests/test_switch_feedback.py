"""Switch feedback follows actual native interaction and reference state plates."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native import controls as native_controls
from mojive.design.native.geometry import Rect
from mojive.ui.interaction_paint import ItemPaint, reference_paint
from tests.test_component_appearance import appearance as appearance


@pytest.fixture
def switch(appearance, monkeypatch):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"switch": {"switchWidth": 44, "switchHeight": 24, "switchThumb": 18}})
    imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    state = SimpleNamespace(ui=ui, runtime=runtime, value=False, enabled=True, apply=False)
    samples, tracks = [], []
    original_rect, original_switch = ui.rect, native_controls.draw_switch

    def rect(x, y, width, height, tint, **options):
        first = len(ui.draw.vtx_buffer)
        result = original_rect(x, y, width, height, tint, **options)
        vertices = list(ui.draw.vtx_buffer)[first:]
        tracks.append(
            {
                "color": tint,
                "resolved": options.get("color_resolved", False),
                "alpha": max(vertex.col >> 24 for vertex in vertices),
                "end": len(ui.draw.vtx_buffer),
            }
        )
        return result

    def paint(ui, x, y, width, height, thumb, value, interaction):
        original_switch(ui, x, y, width, height, thumb, value, interaction)
        track = tracks[-1]
        vertices = list(ui.draw.vtx_buffer)[track["end"] :]
        left, right = min(v.pos.x for v in vertices), max(v.pos.x for v in vertices)
        top, bottom = min(v.pos.y for v in vertices), max(v.pos.y for v in vertices)
        samples.append(
            {
                **track,
                "paint": interaction,
                "value": value,
                "thumb": (left, top, right - left, bottom - top),
            }
        )

    monkeypatch.setattr(ui, "rect", rect)
    monkeypatch.setattr(native_controls, "draw_switch", paint)

    def draw():
        ui.design.begin_frame()
        state.result = ui.design.primitives.switch(
            "feedback",
            Rect(60 * ui.s, 60 * ui.s, 120 * ui.s, 40 * ui.s),
            state.value,
            enabled=state.enabled,
        )
        state.item_id = imgui.get_item_id()
        if state.apply:
            state.value = state.result.value
        row = ui.design.primitives.observed[-1]
        samples[-1]["hit"], samples[-1]["track"] = row["hit"], row["switchRect"]
        return samples[-1]

    def tick():
        return frame(draw)

    def event(method, *arguments):
        getattr(imgui.get_io(), method)(*arguments)
        return tick()

    state.frame, state.event = tick, event
    state.frame()
    yield state


def _feedback(state, value):
    state.value = value
    state.event("add_mouse_pos_event", -100, -100)
    idle = state.frame()
    x, y, width, height = idle["track"]
    hover = state.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    held = state.event("add_mouse_button_event", 0, True)
    assert idle["paint"] == ItemPaint()
    assert hover["paint"].hovered and not hover["paint"].active
    assert held["paint"].active
    assert state.result.value is value
    state.event("add_mouse_button_event", 0, False)
    assert state.result.changed and state.result.value is not value
    return idle, hover, held


@pytest.mark.parametrize("value", (False, True))
def test_live_mouse_states_change_track_and_press_thumb_without_moving_hit_bounds(switch, value):
    samples = _feedback(switch, value)
    assert len({sample["color"] for sample in samples}) == 3
    for sample in samples[1:]:
        assert (
            max(
                abs(a - b)
                for a, b in zip(samples[0]["color"][:3], sample["color"][:3], strict=True)
            )
            >= 10
        )
    assert all(sample["resolved"] for sample in samples)
    assert all(sample["hit"] == samples[0]["hit"] for sample in samples)
    assert all(sample["track"] == samples[0]["track"] for sample in samples)
    for index in (0, 1):
        left, top, width, height = samples[index]["thumb"]
        held_left, held_top, held_width, held_height = samples[2]["thumb"]
        assert held_width < width and held_height < height
        assert held_left + held_width / 2 == pytest.approx(left + width / 2)
        assert held_top + held_height / 2 == pytest.approx(top + height / 2)


@pytest.mark.parametrize("value", (False, True))
def test_disabled_switch_ignores_hover_and_press_and_has_native_disabled_opacity(switch, value):
    switch.value = value
    normal = switch.frame()
    switch.enabled = False
    first = switch.frame()
    x, y, width, height = first["track"]
    hover = switch.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    held = switch.event("add_mouse_button_event", 0, True)
    switch.event("add_mouse_button_event", 0, False)
    assert switch.result.value is value and not switch.result.changed
    assert first["paint"] == hover["paint"] == held["paint"] == ItemPaint(disabled=True)
    assert first["color"] == hover["color"] == held["color"]
    assert first["alpha"] < normal["alpha"]


@pytest.mark.parametrize("value", (False, True))
def test_reference_plates_use_the_same_feedback_as_live_native_items(switch, value):
    samples = _feedback(switch, value)
    switch.event("add_mouse_pos_event", -100, -100)
    for paint, expected in zip(
        (ItemPaint(), ItemPaint(hovered=True), ItemPaint(active=True)), samples, strict=True
    ):
        with reference_paint(paint):
            shown = switch.frame()
        assert shown["color"] == expected["color"]
        assert shown["thumb"] == pytest.approx(expected["thumb"])


def test_switch_feedback_uses_module_palette_without_changing_a_sibling(switch):
    spec = deepcopy(switch.runtime.spec)
    spec["components"]["settings"] = {"sage_color": "#386BA6", "input_color": "#26303B"}
    switch.runtime.update_spec(spec)
    switch.value = True
    with switch.ui.design.family_scope("inspector"):
        peer = switch.frame()
    with switch.ui.design.family_scope("settings"):
        local = switch.frame()
    assert local["color"] != peer["color"]
    assert local["color"][2] > local["color"][0]
    with switch.ui.design.family_scope("inspector"):
        assert switch.frame()["color"] == peer["color"]


def test_keyboard_activation_retains_focus_and_uses_held_feedback(switch):
    switch.apply = True
    first = switch.frame()
    x, y, width, height = first["track"]
    switch.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    switch.event("add_mouse_button_event", 0, True)
    switch.event("add_mouse_button_event", 0, False)
    assert switch.value
    switch.event("add_mouse_pos_event", -100, -100)
    switch.event("add_key_event", imgui.Key.tab, True)
    switch.event("add_key_event", imgui.Key.tab, False)
    switch.frame()
    focused_id = imgui.get_current_context().nav_id
    assert focused_id == switch.item_id
    held = switch.event("add_key_event", imgui.Key.space, True)
    assert held["paint"].active
    switch.event("add_key_event", imgui.Key.space, False)
    switch.frame()
    assert not switch.value
    assert imgui.get_current_context().nav_id == focused_id
