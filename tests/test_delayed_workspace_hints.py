"""Workspace hints share native hover timing without delaying control actions."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.render.backend import NullBackend
from mojive.ui.workspace.filter_chip import draw_filter_chip
from mojive.ui.workspace.geometry_view import geometry_selector
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import paint_menu_row
from mojive.ui.workspace.numeric_fields import NumericGroupLayout
from mojive.ui.workspace.settings_visuals import _label
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 2))
def hints(request, monkeypatch, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (600 * request.param, 400 * request.param), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    imgui.get_style().hover_delay_normal = 0.25
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "design.json")))
    state = SimpleNamespace(ui=ui, io=io, hints=[], clicked=False)
    monkeypatch.setattr(ui, "tooltip", lambda text, **_kwargs: state.hints.append(text))

    def frame(paint, count=1):
        for _ in range(count):
            state.hints.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Hint host", flags=imgui.WindowFlags_.no_title_bar)
            if ui.design is not None:
                ui.design.begin_frame()
            paint()
            imgui.end()
            imgui.render()

    state.frame = frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "surface", ("filter", "menu-disabled", "menu-truncated", "label", "number", "switch", "vector")
)
def test_short_and_clipped_hints_wait_for_the_shared_delay(hints, surface):
    state, ui = hints, hints.ui
    scale = ui.s
    rect = Rect(40 * scale, 50 * scale, 90 * scale, 32 * scale)
    expected = "Geometry visibility"

    def paint():
        nonlocal expected
        if surface == "filter":
            state.clicked = draw_filter_chip(
                ui, "filter", rect, 52, "geom", expected, (150, 190, 140, 255), False
            )
        elif surface.startswith("menu"):
            disabled = surface == "menu-disabled"
            expected = "Unavailable for this scene" if disabled else "A deliberately long command"
            imgui.set_cursor_screen_pos((rect.x, rect.y))
            state.clicked = paint_menu_row(
                ui,
                Item(
                    "Unavailable" if disabled else expected, enabled=not disabled, reason=expected
                ),
                "menu",
                content_right=rect.right,
            ).activated
        elif surface == "label":
            expected = "A deliberately long property name"
            _label(ui, "label", rect, expected)
        elif surface == "number":
            expected = "%.6f" % -1234567890123.456
            imgui.begin_disabled()
            ui.input("number", *rect.tuple()[:3], -1234567890123.456, format="%.6f")
            imgui.end_disabled()
        elif surface == "switch":
            state.clicked = ui.design.primitives.switch(
                "switch", rect, False, label="Visible", enabled=False, tooltip=expected
            ).changed
        else:
            expected = "A deliberately long vector label"
            layout = NumericGroupLayout(
                tuple((offset * scale, 0, 80 * scale, 28 * scale) for offset in (110, 200, 290)),
                3,
                28 * scale,
                label_width=90 * scale,
                label_height=28 * scale,
            )
            ui.vector("vector", rect.x, rect.y, 400 * scale, expected, [0, 0, 0], layout=layout)

    state.frame(paint, 3)
    state.io.add_mouse_pos_event(rect.x + rect.width / 2, rect.y + 14 * scale)
    state.frame(paint)
    assert state.hints == []
    state.frame(paint, 4)
    assert state.hints == []
    state.frame(paint, 16)
    assert state.hints == [expected]
    state.io.add_mouse_pos_event(500 * scale, 300 * scale)
    state.frame(paint)
    assert state.hints == []


def test_filter_click_does_not_wait_for_tooltip(hints):
    state, ui = hints, hints.ui
    rect = Rect(40 * ui.s, 50 * ui.s, 90 * ui.s, 32 * ui.s)

    def paint():
        state.clicked = draw_filter_chip(
            ui, "filter", rect, 52, "geom", "Geometries", (150, 190, 140, 255), False
        )

    state.frame(paint, 3)
    state.io.add_mouse_pos_event(rect.x + rect.width / 2, rect.y + rect.height / 2)
    state.frame(paint)
    state.io.add_mouse_button_event(0, True)
    state.frame(paint)
    state.io.add_mouse_button_event(0, False)
    state.frame(paint)
    assert state.clicked
    assert state.hints == []


@pytest.mark.parametrize("width", (100, 460))
def test_geometry_selector_has_one_descriptive_hint_with_or_without_captions(
    hints, monkeypatch, width
):
    state, ui = hints, hints.ui
    ctx = SimpleNamespace(tr=str, backend=NullBackend())
    native_caption_hints = []
    monkeypatch.setattr(imgui, "set_item_tooltip", native_caption_hints.append)

    def paint():
        geometry_selector(ui, ctx, 40 * ui.s, 50 * ui.s, width * ui.s, key="geometry-hints")

    state.frame(paint, 3)
    x, y, w, h = ui.hits["geometry-hints-collision"]
    state.io.add_mouse_pos_event(x + w / 2, y + h / 2)
    state.frame(paint)
    assert state.hints == []
    state.frame(paint, 20)
    assert state.hints == ["Show collision geometry"]
    assert native_caption_hints == []


@pytest.mark.parametrize("disabled", (False, True))
def test_numeric_reset_badge_keeps_disabled_hint_and_waits_without_changing_value(hints, disabled):
    state, ui = hints, hints.ui
    ui.design = None

    def paint():
        imgui.begin_disabled(disabled)
        changed, value = ui.input("axis", 40 * ui.s, 50 * ui.s, 100 * ui.s, 8.75, axis="X")
        imgui.end_disabled()
        assert not changed and value == 8.75

    state.frame(paint, 3)
    x, y, width, height = ui.hits["axis-reset"]
    state.io.add_mouse_pos_event(x + width / 2, y + height / 2)
    state.frame(paint)
    assert state.hints == []
    state.frame(paint, 20)
    assert state.hints == ["Read only" if disabled else "Reset: 0"]
