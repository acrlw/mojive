"""Production workspace controls exercised through ImGui without a GPU window."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace.design import DESIGN, MONO_SIZE
from mojive.ui.workspace.widgets import UI, NumericEditState


@pytest.fixture(params=(1.0, 2.0))
def native_ui(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 500 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    state = SimpleNamespace(ui=ui, io=io, value=1.0, disabled=False)

    def frame(count=1):
        for _ in range(count):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Production controls", flags=imgui.WindowFlags_.no_title_bar)
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.begin_disabled(state.disabled)
            state.changed, state.value = ui.input(
                "number",
                40 * ui.s,
                40 * ui.s,
                110 * ui.s,
                state.value,
                axis="X",
                speed=0.1,
                reset_value=0.25,
            )
            imgui.end_disabled()
            ui.button("icon", 180 * ui.s, 40 * ui.s, icon="camera")
            ui.segmented("mode", 40 * ui.s, 100 * ui.s, 180, ("First", "Second"), 0)
            ui.switch("switch", 240 * ui.s, 100 * ui.s, True)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def point(key="number"):
        x, y, width, height = ui.hits[key]
        return x + width / 2, y + height / 2

    state.frame, state.event, state.point = frame, event, point
    try:
        frame(3)
        yield state
    finally:
        imgui.destroy_context(context)


def test_production_controls_keep_one_hit_and_edit_state_owner(native_ui):
    state, ui = native_ui, native_ui.ui
    assert set(ui.hits) >= {"number", "number-reset", "icon", "mode", "switch"}
    for key in ("number", "icon", "mode", "switch"):
        assert ui.hits[key][3] == pytest.approx(DESIGN.control_height * ui.s, abs=1)
    x, y = state.point()
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    assert isinstance(ui.numeric_edits["number"], NumericEditState)
    assert ui.numeric_edits["number"].activated
    state.event("add_mouse_pos_event", x + 25 * ui.s, y)
    assert state.changed and state.value > 1.0
    state.event("add_mouse_button_event", 0, False)
    assert ui.numeric_edits["number"].committed
    state.event("add_mouse_pos_event", *state.point("number-reset"))
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    assert state.value == 0.25 and ui.numeric_edits["number"].reset


def test_production_text_entry_cancellation_and_disabled_input(native_ui):
    state, ui = native_ui, native_ui.ui
    state.event("add_key_event", imgui.Key.mod_ctrl, True)
    state.event("add_mouse_pos_event", *state.point())
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    state.event("add_key_event", imgui.Key.mod_ctrl, False)
    state.event("add_input_characters_utf8", "5.25")
    state.event("add_key_event", imgui.Key.escape, True)
    assert state.value == 1.0 and ui.numeric_edits["number"].canceled
    state.event("add_key_event", imgui.Key.escape, False)
    state.disabled = True
    x, y = state.point()
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + 25 * ui.s, y)
    state.event("add_mouse_button_event", 0, False)
    assert state.value == 1.0 and not state.changed


def test_numeric_measurement_reuses_alternating_formats(native_ui, monkeypatch):
    ui, calls = native_ui.ui, []
    measure = imgui.calc_text_size

    def count(text, *args, **kwargs):
        calls.append(text)
        return measure(text, *args, **kwargs)

    monkeypatch.setattr(imgui, "calc_text_size", count)
    for _ in range(10):
        short = ui.numeric_minimum(format="%.3f", values=(1.0,))
        precise = ui.numeric_minimum(format="%.7f", values=(1.0,))
    assert precise > short
    assert len(calls) == 8
    without_axis = ui.numeric_minimum(axis=False, format="%.3f", values=(1.0,))
    assert without_axis + 18 * ui.s == short
    assert len(calls) == 8


def test_numeric_measurement_tracks_value_font_scale_and_framebuffer(native_ui, monkeypatch):
    ui = native_ui.ui
    measure = imgui.calc_text_size
    calls = []

    def count(text, *args, **kwargs):
        calls.append(text)
        return measure(text, *args, **kwargs)

    monkeypatch.setattr(imgui, "calc_text_size", count)
    baseline = ui.numeric_minimum(format="%.3f", values=(1.0,))
    wider = ui.numeric_minimum(format="%.3f", values=(1234567890123.0,))
    assert wider > baseline and len(calls) == 8
    original_font = ui.mono
    ui.mono = ui.body
    changed_font = ui.numeric_minimum(format="%.3f", values=(1.0,))
    assert len(calls) == 12
    imgui.push_font(ui.body, MONO_SIZE * ui.s)
    expected = max(measure(value).x for value in ("0.123", "123.457", "-0.123", "1.000"))
    imgui.pop_font()
    assert changed_font == pytest.approx(expected + 34 * ui.s)
    ui.mono = original_font
    ui.s *= 1.5
    scaled = ui.numeric_minimum(format="%.3f", values=(1.0,))
    assert scaled > baseline and len(calls) == 16
    native_ui.io.display_framebuffer_scale = (2, 2)
    ui.numeric_minimum(format="%.3f", values=(1.0,))
    assert len(calls) == 20
    for index in range(100):
        ui.numeric_minimum(format="%.3f", values=(index,))
    assert len(ui._numeric_measurements) <= 64
