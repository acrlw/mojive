"""Native output behavior using the shared diagnostic buffer and row renderer."""

from types import SimpleNamespace

import pytest
from examples.ui_design.native import output_panel as native_output
from imgui_bundle import imgui

from mojive.ui.messages import OutputBuffer
from mojive.ui.panels import output as shared_output
from mojive.ui.theme import THEME
from mojive.ui.workspace.widgets import UI


def _panel():
    return native_output.ReferenceOutput(SimpleNamespace(s=1))


def test_native_output_copy_uses_visible_selection_and_preserves_multiline_messages():
    panel, output = _panel(), OutputBuffer()
    output.write("cache detail", level="debug", timestamp="10:00:00")
    info = output.write("compile\nmodel detail", level="info", timestamp="10:00:01")
    error = output.write("compile failed", level="error", timestamp="10:00:02")
    panel._selected_sequences = {info.sequence, error.sequence}
    panel._levels = {"info"}
    panel._filter_text = "compile detail"

    assert panel._copy_text(output) == "compile\nmodel detail"
    assert panel._filter_entries(output.entries()) == (info,)
    panel._selected_sequences = {error.sequence}
    assert panel._copy_text(output) == "10:00:01 [INFO] compile\nmodel detail"


def test_hidden_search_cannot_leave_an_invisible_output_filter():
    panel, output = _panel(), OutputBuffer()
    first = output.write("first result", timestamp="10:00:00")
    second = output.write("second result", timestamp="10:00:01")
    panel._toggle_search()
    panel._filter_text = "first"
    assert panel._filter_entries(output.entries()) == (first,)

    panel._toggle_search()

    assert not panel.show_search
    assert panel._filter_entries(output.entries()) == (first, second)


def test_output_clear_and_capacity_eviction_remove_stale_selections():
    panel, output = _panel(), OutputBuffer(capacity=2)
    first = output.write("first", timestamp="10:00:00")
    panel._selected_sequences = {first.sequence}
    panel._selection_anchor = first.sequence
    output.write("second", timestamp="10:00:01")
    status = output.publish("third", duration=None)

    assert "first" not in panel._copy_text(output)
    assert not panel._selected_sequences
    panel._clear_output(output)
    assert panel._copy_text(output) == ""
    assert panel._selection_anchor == 0
    assert output.active_status() is status


@pytest.mark.parametrize("metadata", [False, True])
def test_native_output_file_actions_use_the_actual_message_path(monkeypatch, tmp_path, metadata):
    path = tmp_path / "saved scene.mojive.json"
    path.write_text("{}")
    output = OutputBuffer()
    entry = output.write(f"Saved scene to {path}", copy_text=str(path) if metadata else None)
    copied, revealed, labels = [], [], []

    def menu_item(label, *_args):
        labels.append(label)
        return True, False

    monkeypatch.setattr(native_output.imgui, "menu_item", menu_item)
    monkeypatch.setattr(native_output.imgui, "set_clipboard_text", copied.append)
    monkeypatch.setattr(native_output, "reveal_path", revealed.append)
    _panel()._draw_file_actions(SimpleNamespace(tr=str), entry)

    assert revealed == [path.resolve()]
    assert copied == ([] if metadata else [str(path.resolve())])
    assert labels == (["Reveal file"] if metadata else ["Copy path", "Reveal file"])


def test_multiline_rows_and_auto_scroll_use_the_shared_renderer(monkeypatch):
    panel, output = _panel(), OutputBuffer()
    output.write("first line\nsecond line", timestamp="10:00:00")
    scroll, rows, child_ends = [], [], []
    gui = SimpleNamespace(
        separator=lambda: None,
        begin_child=lambda *_args: True,
        end_child=lambda: child_ends.append(True),
        ImVec2=imgui.ImVec2,
        get_font_size=lambda: 14,
        get_font=lambda: SimpleNamespace(font_id=1),
        get_content_region_avail=lambda: imgui.ImVec2(400, 200),
        calc_text_size=lambda text, **_kwargs: imgui.ImVec2(100, 14 * (text.count("\n") + 1)),
        is_rect_visible=lambda _size: True,
        get_io=lambda: SimpleNamespace(want_text_input=False, key_ctrl=False, key_super=False),
        is_window_focused=lambda _flags: False,
        FocusedFlags_=SimpleNamespace(root_and_child_windows=1),
        set_scroll_here_y=scroll.append,
    )
    monkeypatch.setattr(shared_output, "imgui", gui)
    monkeypatch.setattr(shared_output, "ImguiDraw2D", lambda: None)
    monkeypatch.setattr(shared_output, "text_line_y", lambda _draw, center: center - 7)
    monkeypatch.setattr(panel, "_draw_entry", lambda *args: rows.append(args))
    ctx = SimpleNamespace(output=output, style_scale=1, language="en", tr=str)
    panel._toggle_auto_scroll()
    shared_output.OutputPanel._draw_entries(panel, ctx, output.entries(), False)
    assert rows[0][3] == "10:00:00  first line\nsecond line"
    assert rows[0][4] > rows[0][5]
    output.write("another result", timestamp="10:00:01")
    shared_output.OutputPanel._draw_entries(panel, ctx, output.entries(), False)
    assert scroll == []

    panel._toggle_auto_scroll()
    shared_output.OutputPanel._draw_entries(panel, ctx, output.entries(), False)
    shared_output.OutputPanel._draw_entries(panel, ctx, output.entries(), False)
    assert scroll == [1.0]
    assert len(child_ends) == 4


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("width", [96, 124, 160, 196, 206, 480, 960])
def test_native_output_toolbar_fits_and_collapses_the_bottom_dock(monkeypatch, scale, width):
    hits, collapsed = {}, []

    def button(key, x, y, w=28, h=28, **_kwargs):
        hits[key] = (x, y, w * scale, h * scale)
        return key == "collapse-output"

    study = SimpleNamespace(
        s=scale,
        ui=SimpleNamespace(button=button, measure=lambda text: len(text) * 7 * scale),
        collapse_bottom=lambda: collapsed.append(True),
    )
    panel, output = native_output.ReferenceOutput(study), OutputBuffer()
    output.write("result")
    gui = SimpleNamespace(
        get_cursor_screen_pos=lambda: imgui.ImVec2(20, 20),
        get_content_region_avail=lambda: imgui.ImVec2(width * scale, 100 * scale),
        begin_popup=lambda _id: False,
        set_cursor_screen_pos=lambda _pos: None,
    )
    monkeypatch.setattr(native_output, "imgui", gui)
    ctx = SimpleNamespace(output=output, tr=str, theme=THEME)
    entries, newest = panel._counts(ctx)
    panel._draw_toolbar(ctx, entries, newest)

    assert collapsed == [True]
    rectangles = tuple(hits.values())
    for index, (x, y, w, h) in enumerate(rectangles):
        assert x >= 20
        assert x + w <= 20 + width * scale
        for bx, by, bw, bh in rectangles[index + 1 :]:
            assert x + w <= bx or bx + bw <= x or y + h <= by or by + bh <= y


def test_output_publishes_resolved_pointer_hints_once():
    ctx = SimpleNamespace(input_bindings=None, status_hints=(), tr=str)
    panel = _panel()
    panel.finish_frame(ctx)
    panel.finish_frame(ctx)

    assert {hint.hint_id for hint in ctx.status_hints} == {
        "output.select",
        "output.range_selection",
        "output.context",
    }
    assert len(ctx.status_hints) == 3


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("width", [180, 960])
def test_native_output_draws_wrapped_rows_and_search_with_real_imgui(scale, width):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (width * scale, 500 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s = scale
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.translate = str
    ui.hits = {}
    study = SimpleNamespace(s=scale, ui=ui, collapse_bottom=lambda: None)
    panel, output = native_output.ReferenceOutput(study), OutputBuffer()
    output.write("first line\nsecond line", timestamp="10:00:00")
    output.write("A long diagnostic message. " * 50, timestamp="10:00:01", level="error")
    ctx = SimpleNamespace(
        output=output,
        tr=str,
        language="en",
        style_scale=scale,
        theme=THEME,
        input_bindings=None,
        input_claim=None,
    )
    try:
        for search_open in (False, True):
            panel.show_search = search_open
            for _ in range(3):
                imgui.new_frame()
                imgui.set_next_window_pos((0, 0))
                imgui.set_next_window_size(io.display_size)
                imgui.begin("Output", flags=imgui.WindowFlags_.no_title_bar)
                panel.draw(ctx)
                imgui.end()
                imgui.render()
            assert imgui.get_draw_data().total_vtx_count > 0
            assert len(panel._wrapped_rows) == 2
            assert panel._wrapped_rows[0][0] == "10:00:00  first line\nsecond line"
            assert panel._wrapped_rows[1][1] > panel._wrapped_rows[0][1]
    finally:
        imgui.destroy_context(context)
