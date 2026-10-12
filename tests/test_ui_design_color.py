"""Color entry preserves channel ranges and remains usable in small native popups."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
from examples.ui_design.native.color_editor import format_color, parse_color
from examples.ui_design.native.inspector_style import NativeInspectorPresentation
from examples.ui_design.native.widgets import UI
from imgui_bundle import imgui

from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext


@pytest.fixture(params=(1.0, 2.25))
def color_popup(request, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    report = SimpleNamespace(cjk_path=None)
    language = request.node.callspec.params.get("language", "en")
    if language == "zh_CN":
        from mojive.text.sources import resolve_font_sources

        sources = resolve_font_sources(allow_download=False)
        if len(sources) < 2:
            imgui.destroy_context(context)
            pytest.skip("CJK font is unavailable")
        report.cjk_path, report.cjk_index = sources[1].path, sources[1].index
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=report))
    ui.translate = Localizer(Language(language)).text
    imgui.get_style().window_min_size = (ui.s, ui.s)
    session = Session(StaticSceneAdapter(Scene()))
    presentation = NativeInspectorPresentation(ui)
    presentation.node_id = 1
    presentation.ctx = PanelContext(session, NullBackend(), translate=ui.translate)
    state = SimpleNamespace(
        ui=ui,
        io=io,
        presentation=presentation,
        values=[0.2, 0.4, 0.6, 1.0],
        width=180,
        height=220,
        key="##test-color",
        popup="##test-color-color-editor-1",
    )

    def frame(count=1):
        result = None
        for _ in range(count):
            ui.hits.clear()
            ui.numeric_edits.clear()
            io.display_size = (state.width * ui.s, state.height * ui.s)
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Color Inspector", flags=imgui.WindowFlags_.no_title_bar)
            result = presentation.color(state.key, state.values)
            if result[0]:
                state.values = list(result[1])
            imgui.end()
            imgui.render()
        return result

    def click(key):
        x, y, w, h = ui.hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        return frame(2)

    state.frame, state.click = frame, click
    frame(2)
    try:
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "text,expected",
    (
        ("#abc", (170 / 255, 187 / 255, 204 / 255)),
        ("#abcd", (170 / 255, 187 / 255, 204 / 255, 221 / 255)),
        ("11223344", (17 / 255, 34 / 255, 51 / 255, 68 / 255)),
        ("rgba(128, 64, 255, 128)", (128 / 255, 64 / 255, 1, 128 / 255)),
        ("rgb(0.125, 0.5, 1)", (0.125, 0.5, 1)),
        ("0.125, 0.5, 1, 0.2", (0.125, 0.5, 1, 0.2)),
    ),
)
def test_color_clipboard_parses_explicit_channels(text, expected):
    assert parse_color(text) == pytest.approx(expected)


@pytest.mark.parametrize("mode", (0, 1, 2))
@pytest.mark.parametrize("values", ((0, 1 / 255, 1 / 255, 1 / 255), (0.2, 0.4, 0.6, 1)))
def test_color_copy_round_trip_retains_small_channels_and_alpha(mode, values):
    text = format_color(values, mode)
    parsed = parse_color(text, normalized=(mode == 2) if mode else None)
    if len(parsed) == 3:
        parsed.append(1)
    assert parsed == pytest.approx(values, abs=1 / 255 if mode == 0 else 1e-6)


@pytest.mark.parametrize(
    "text", ("#zzzzzz", "#12345", "nan, 0, 1", "inf, 0, 1", "-1,0,1", "256,0,0", "1,2", "1,2,3,4,5")
)
def test_invalid_color_clipboard_is_rejected(text):
    with pytest.raises(ValueError):
        parse_color(text)


def test_normalized_rgba_does_not_accept_byte_range():
    with pytest.raises(ValueError):
        parse_color("rgba1(128,0,0,1)")


@pytest.mark.parametrize("source", (0, 1, 2))
@pytest.mark.parametrize("destination", (0, 1, 2))
@pytest.mark.parametrize("values", ((0, 1 / 255, 1 / 255, 1 / 255), (0.2, 0.4, 0.6, 1)))
def test_color_clipboard_format_survives_switching_input_modes(source, destination, values):
    copied = format_color(values, source)
    pasted = parse_color(copied, normalized=(destination == 2) if destination else None)
    if len(pasted) == 3:
        pasted.append(1)
    assert pasted == pytest.approx(values, abs=1 / 255 if source == 0 else 1e-6)


@pytest.mark.parametrize("text", ("rgb(1,2,3,4)", "rgba(1,2,3)", "rgba(1,2,3", "1,2,3)"))
def test_malformed_color_functions_are_rejected(text):
    with pytest.raises(ValueError):
        parse_color(text)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_small_color_popup_keeps_stacked_formats_clear_of_entry_and_done(
    color_popup, language, mode
):
    state = color_popup
    ui, presentation = state.ui, state.presentation
    presentation._color_editors[(1, state.key)].mode = mode
    state.click(state.key + "-picker")
    stack = imgui.get_current_context().open_popup_stack
    assert len(stack) == 1
    window = stack[0].window
    assert 0 <= window.pos.x <= state.width * ui.s - window.size.x + 1
    assert 0 <= window.pos.y <= state.height * ui.s - window.size.y + 1
    _, y, _, height = ui.hits[state.popup + "-format"]
    assert height > 28 * ui.s
    fields = (
        [ui.hits[state.popup + "-hex"]]
        if mode == 0
        else [ui.hits[state.popup + axis + "-frame"] for axis in "RGBA"]
    )
    assert fields[0][1] - y - height >= 8 * ui.s - 1
    for x, _field_y, w, h in fields:
        assert window.pos.x <= x < x + w <= window.pos.x + window.size.x
        assert h == pytest.approx(28 * ui.s, abs=1)
    for first, second in pairwise(fields):
        assert second[1] - first[1] - first[3] >= 8 * ui.s - 1

    state.io.add_mouse_pos_event(window.pos.x + window.size.x / 2, window.pos.y + window.size.y / 2)
    state.frame(2)
    state.io.add_mouse_wheel_event(0, -100)
    state.frame(3)
    x, y, w, h = ui.hits[state.popup + "-done"]
    clip = window.inner_clip_rect
    assert clip.min.x <= x < x + w <= clip.max.x + 1
    assert clip.min.y <= y < y + h <= clip.max.y + 1
    state.click(state.popup + "-done")
    assert not imgui.get_current_context().open_popup_stack
    assert presentation.color_edit_action == "apply"


@pytest.mark.parametrize(
    "source,destination,values",
    ((1, 0, (0, 1 / 255, 1 / 255, 1 / 255)), (2, 1, (0.2, 0.4, 0.6, 0.5))),
)
def test_actual_color_copy_and_paste_preserve_source_range_across_modes(
    color_popup, source, destination, values
):
    state = color_popup
    state.width, state.height = 360, 420
    state.values = list(values)
    editor = state.presentation._color_editors[(1, state.key)]
    editor.mode = source
    state.frame(2)
    state.click(state.key + "-picker")
    state.click(state.popup + "-copy")
    assert imgui.get_clipboard_text() == format_color(values, source)
    state.click(state.popup + f"-format-{destination}")
    assert editor.mode == destination
    state.values = [0.8, 0.7, 0.9, 0.8]
    state.frame(2)
    state.click(state.popup + "-paste")
    assert state.values == pytest.approx(values, abs=1e-6)
    assert not editor.error
