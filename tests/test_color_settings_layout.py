"""Color groups and settings navigation retain their complete content budgets."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.color_editor import ColorEditor, color_popup
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.settings_layout import draw_navigation
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.25))
def layout(request, tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.display_size = (1000 * request.param, 1400 * request.param)
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    state = SimpleNamespace(
        ui=ui, runtime=runtime, width=320, editor=ColorEditor(), values=[0, 0.5, 1]
    )

    def frame(count=3):
        for _ in range(count):
            ui.hits.clear()
            runtime.observed.clear()
            ui.design_primitives.observed.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Layout host", flags=imgui.WindowFlags_.no_title_bar)
            if not imgui.is_popup_open("Color"):
                imgui.open_popup("Color")
            imgui.set_next_window_size_constraints(
                (state.width * ui.s, 0), (state.width * ui.s, 1300 * ui.s)
            )
            with color_popup(ui, "Color") as opened:
                if opened:
                    state.editor.draw(ui, "color", state.values, imgui.get_content_region_avail().x)
                    position = imgui.get_cursor_screen_pos()
                    gap = ui.metric("color_editor", "columnGap", 6) * ui.s
                    half = (state.editor.content_width - gap) / 2
                    ui.button("cancel", position.x, position.y, half / ui.s, label="Cancel")
                    ui.button(
                        "done", position.x + half + gap, position.y, half / ui.s, label="Done"
                    )
            imgui.end()
            imgui.render()

    state.frame = frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("count", (3, 4))
@pytest.mark.parametrize("width", (160, 320))
@pytest.mark.parametrize("mode", (1, 2))
@pytest.mark.parametrize("columns", (2, 4))
def test_color_channels_keep_one_complete_row_or_one_channel_per_row(
    layout, count, width, mode, columns
):
    state, ui = layout, layout.ui
    authored = deepcopy(state.runtime.spec)
    authored["parts"]["rgba"] = {"columns": columns}
    state.runtime.update_spec(authored)
    state.width, state.values, state.editor.mode = width, [0.125, 0.25, 0.5, 0.75][:count], mode
    state.frame()

    rows = [ui.hits["color" + axis + "-frame"] for axis in "RGBA"[:count]]
    assert ui.hits["color-copy"][1] == ui.hits["color-paste"][1]
    row_count = len({round(row[1], 2) for row in rows})
    assert row_count in (1, count)
    if width == 320 and columns == 4:
        assert row_count == 1
    elif columns < count:
        assert row_count == count
    window = imgui.get_current_context().open_popup_stack[0].window
    for x, _y, width, _height in rows:
        assert x >= window.pos.x + native_unit(8 * ui.s) - 1
        assert x + width <= window.pos.x + window.size.x - native_unit(8 * ui.s) + 1
    before = rows.copy()
    state.values = [0.999, 0.001, 0.333, 1][:count]
    state.frame()
    assert [ui.hits["color" + axis + "-frame"] for axis in "RGBA"[:count]] == before


def test_color_popup_resolves_uniform_native_insets_and_preserves_independent_gaps(layout):
    state, ui = layout, layout.ui
    authored = deepcopy(state.runtime.spec)
    authored["parts"]["color_editor"] = {
        "paddingLeft": 5,
        "paddingTop": 7,
        "paddingRight": 11,
        "paddingBottom": 13,
        "rowGap": 10,
        "columnGap": 9,
    }
    authored["parts"]["rgba"] = {"columns": 1, "rowGap": 12}
    state.runtime.update_spec(authored)
    state.editor.mode = 2
    state.frame()
    window = imgui.get_current_context().open_popup_stack[0].window
    copy, paste = ui.hits["color-copy"], ui.hits["color-paste"]
    cancel, done = ui.hits["cancel"], ui.hits["done"]
    header = ui.hits["color-header"]
    inset = native_unit(13 * ui.s)
    assert header[0] - window.pos.x == pytest.approx(inset, abs=1)
    assert header[1] - window.pos.y == pytest.approx(inset, abs=1)
    assert copy[0] >= header[0]
    assert paste[0] + paste[2] <= header[0] + header[2] + 1
    assert paste[0] - copy[0] - copy[2] == pytest.approx(9 * ui.s)
    assert done[0] - cancel[0] - cancel[2] == pytest.approx(9 * ui.s)
    assert window.pos.x + window.size.x - done[0] - done[2] == pytest.approx(inset, abs=1)
    assert window.pos.y + window.size.y - done[1] - done[3] == pytest.approx(inset, abs=1)
    first, second = (ui.hits["color" + axis + "-frame"] for axis in "RG")
    assert second[1] - first[1] - first[3] == pytest.approx(12 * ui.s)
    assert state.editor.content_origin == pytest.approx(header[:2])
    assert state.editor.content_width == pytest.approx(state.width * ui.s - inset * 2, abs=1)
    assert state.runtime.spec == authored


@pytest.mark.parametrize("narrow", (False, True))
def test_settings_navigation_rows_are_inset_from_the_actual_container(layout, narrow):
    ui, runtime = layout.ui, layout.runtime
    authored = deepcopy(runtime.spec)
    authored["parts"]["settings_nav"] = {
        "paddingLeft": 7,
        "paddingTop": 9,
        "paddingRight": 13,
        "paddingBottom": 11,
        "navWidth": 128,
    }
    runtime.update_spec(authored)
    panel = SimpleNamespace(workspace=SimpleNamespace(ui=ui, s=ui.s), _category="General")
    for _ in range(3):
        runtime.observed.clear()
        ui.design_primitives.observed.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.begin("Settings layout", flags=imgui.WindowFlags_.no_title_bar)
        draw_navigation(
            panel,
            SimpleNamespace(tr=str),
            Rect(20 * ui.s, 20 * ui.s, 700 * ui.s, 500 * ui.s),
            narrow,
        )
        imgui.end()
        imgui.render()
    records = {record["key"]: record for record in runtime.observed}
    container = records["settings-navigation"]
    x, y, width, height = container["outer"]
    cx, cy, cw, ch = container["content"]
    assert (cx - x, cy - y, x + width - cx - cw, y + height - cy - ch) == pytest.approx(
        (native_unit(13 * ui.s),) * 4
    )
    rows = [
        record
        for record in ui.design_primitives.observed
        if record["key"].startswith("settings-") and "clippedLabel" in record
    ]
    assert len(rows) == 6
    for row in rows:
        rx, ry, rw, rh = row["hit"]
        assert cx <= rx < rx + rw <= cx + cw + 1
        assert cy <= ry < ry + rh <= cy + ch + 1
        assert not row["clippedLabel"], row
