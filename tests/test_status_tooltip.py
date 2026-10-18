"""Status hints keep readable content geometry and restore their caller's styles."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace import status_tooltip as tooltip_module
from mojive.ui.workspace.design import CONTROL_RADIUS
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.0))
def tooltip_frame(request, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1200 * request.param, 700 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    rows = []
    original = imgui.text_unformatted

    def text(value):
        window = imgui.get_current_context().current_window
        rows.append(
            SimpleNamespace(
                value=value,
                cursor=tuple(imgui.get_cursor_pos()),
                padding=tuple(window.window_padding),
                spacing=tuple(imgui.get_style().item_spacing),
                minimum=tuple(imgui.get_style().window_min_size),
                rounding=window.window_rounding,
                font=imgui.get_font().font_id,
                size=imgui.get_font_size(),
                tint=tuple(imgui.get_style_color_vec4(imgui.Col_.text)),
                width=imgui.calc_text_size(value).x,
            )
        )
        original(value)

    def frame(paint):
        rows.clear()
        imgui.new_frame()
        imgui.begin("Status tooltip caller")
        try:
            return paint()
        finally:
            imgui.end()
            imgui.render()

    monkeypatch.setattr(imgui, "text_unformatted", text)
    try:
        yield request.param, rows, frame
    finally:
        imgui.destroy_context(context)


def _state():
    context, style = imgui.get_current_context(), imgui.get_style()
    return (
        len(context.style_var_stack),
        len(context.color_stack),
        len(context.current_window_stack),
        context.current_window.id_,
        tuple(style.window_padding),
        tuple(style.window_min_size),
        tuple(style.item_spacing),
        style.window_rounding,
        imgui.get_font().font_id,
        imgui.get_font_size(),
        tuple(style.color_(imgui.Col_.text)),
    )


def test_default_status_hint_has_scaled_padding_and_separate_action_rows(tooltip_frame):
    scale, rows, frame = tooltip_frame

    def paint():
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (300, 200))
        before = _state()
        tooltip_module.status_tooltip(
            None, "Steps 123456", "Click to show time", "Right-click to copy", scale=scale
        )
        assert _state() == before
        imgui.pop_style_var(3)

    for _ in range(3):
        frame(paint)
    assert [row.value for row in rows] == [
        "Steps 123456",
        "Click to show time",
        "Right-click to copy",
    ]
    assert all(row.padding == (10 * scale, 8 * scale) for row in rows)
    assert all(row.spacing == (0, 6 * scale) for row in rows)
    assert all(row.minimum == (0, 0) for row in rows)
    assert all(row.rounding == CONTROL_RADIUS * scale for row in rows)
    assert rows[0].cursor == (10 * scale, 8 * scale)
    assert rows[1].cursor[1] - rows[0].cursor[1] == rows[0].size + 6 * scale
    assert rows[2].cursor[1] - rows[1].cursor[1] == rows[1].size + 6 * scale


def test_authored_status_hint_uses_its_own_metrics_style(tooltip_frame, tmp_path):
    scale, rows, frame = tooltip_frame
    ui = UI(SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)))
    ui.prepare_fonts("system", ("portable",))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)
    spec = deepcopy(runtime.spec)
    spec["parts"].update(
        {
            "statusbar": {"fontSize": 32, "fontFamily": "system", "textColor": "#ff0000"},
            "status_metrics": {
                "fontSize": 19,
                "fontFamily": "portable",
                "textColor": "#2080e0",
                "tooltipPaddingX": 18,
                "tooltipPaddingY": 12,
                "tooltipRowGap": 10,
            },
        }
    )
    runtime.update_spec(spec)
    style = ui.component_style("status_metrics")
    selected_font = ui.resolve_font(style).font_id

    def paint():
        before = _state()
        with part_scope("statusbar"):
            tooltip_module.status_tooltip(ui, "Physics 100.125 Hz", "Right-click to copy")
        assert _state() == before

    for _ in range(3):
        frame(paint)
    assert [row.value for row in rows] == ["Physics 100.125 Hz", "Right-click to copy"]
    assert all(row.padding == (18 * scale, 12 * scale) for row in rows)
    assert all(row.spacing == (0, 10 * scale) for row in rows)
    assert all(row.font == selected_font and row.size == 19 * scale for row in rows)
    assert all(row.tint == pytest.approx((32 / 255, 128 / 255, 224 / 255, 1)) for row in rows)
    assert rows[1].cursor[1] - rows[0].cursor[1] == 29 * scale


@pytest.mark.parametrize("styled", (False, True))
def test_status_hint_restores_window_and_stacks_when_painting_fails(
    tooltip_frame, monkeypatch, styled
):
    scale, _rows, frame = tooltip_frame
    ui = (
        SimpleNamespace(
            s=scale,
            component_style=lambda _part: {"fontSize": 20, "textColor": "#2080e0"},
            resolve_font=lambda _style: imgui.get_font(),
            style_color=lambda _style, _field, _fallback: (32, 128, 224, 255),
        )
        if styled
        else None
    )

    def fail(_value):
        raise ValueError("tooltip paint failed")

    def paint():
        before = _state()
        with pytest.raises(ValueError, match="tooltip paint failed"):
            tooltip_module.status_tooltip(ui, "Exact value", "Copy", scale=scale)
        assert _state() == before

    monkeypatch.setattr(imgui, "text_unformatted", fail)
    frame(paint)
