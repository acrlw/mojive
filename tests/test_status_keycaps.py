"""Status keyboard hints keep their own face and padding beside pointer diagrams."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets import ToolHint, draw_status
from mojive.ui.workspace.status_components import StatusBadge, StatusGroups
from mojive.ui.workspace.typography import TextRole, Typography
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.0))
def status_frame(request, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1800 * request.param, 700 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)

    def frame(paint):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Status keycaps", flags=imgui.WindowFlags_.no_title_bar)
        try:
            return paint()
        finally:
            imgui.end()
            imgui.render()

    frame(lambda: None)
    try:
        yield ui, runtime, frame
    finally:
        imgui.destroy_context(context)


def test_keyboard_caps_keep_monospace_when_ui_and_numeric_fields_use_body(
    status_frame, monkeypatch
):
    ui, runtime, frame = status_frame
    ui.typography = Typography(numeric=TextRole("body"))
    spec = deepcopy(runtime.spec)
    spec["parts"].update(
        {
            "statusbar": {"fontSize": 14, "fontFamily": "portable"},
            "status_hints": {"fontSize": 16, "fontFamily": "portable"},
        }
    )
    runtime.update_spec(spec)
    painted = []
    original = ImguiDraw2D.text

    def capture(draw, position, color, text, **options):
        painted.append((text, imgui.get_font().font_id, tuple(position)))
        return original(draw, position, color, text, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", capture)

    def paint():
        with part_scope("statusbar"):
            keycap = StatusBadge(ui, keycap=True)
            assert keycap.size("iiii")[0] == pytest.approx(keycap.size("WWWW")[0])
            return draw_status(
                ImguiDraw2D(),
                (20 * ui.s, 40 * ui.s),
                1600 * ui.s,
                50 * ui.s,
                THEME,
                ui.s,
                selected="",
                state="paused",
                sim_time=0,
                step=0,
                metric_mode="steps",
                backend="OpenGL",
                dt=0.002,
                fps=59.5,
                show_metrics=False,
                tool_hints=(
                    ToolHint("key", "Esc", "Cancel"),
                    ToolHint("key", "Shift", "Snap"),
                    ToolHint("mouse", "left", "Push", modifier="Ctrl"),
                ),
                keycap=keycap,
                presentation=StatusGroups(ui),
            )

    frame(paint)
    keyboard = {text: font for text, font, _position in painted if text in ("Esc", "Shift", "Ctrl")}
    descriptions = {
        text: font for text, font, _position in painted if text in ("Cancel", "Snap", "Push")
    }
    assert keyboard == dict.fromkeys(("Esc", "Shift", "Ctrl"), ui.mono.font_id)
    assert descriptions == dict.fromkeys(("Cancel", "Snap", "Push"), ui.body.font_id)


def test_keycap_group_padding_does_not_resize_generic_badges_or_telemetry(status_frame):
    ui, runtime, frame = status_frame

    def measure():
        with part_scope("statusbar"):
            keycap, badge = StatusBadge(ui, keycap=True), StatusBadge(ui)
            metrics = draw_status(
                ImguiDraw2D(),
                (20 * ui.s, 40 * ui.s),
                1600 * ui.s,
                50 * ui.s,
                THEME,
                ui.s,
                selected="",
                state="paused",
                sim_time=0,
                step=0,
                metric_mode="steps",
                backend="OpenGL",
                dt=0.002,
                fps=59.5,
                tool_hints=(ToolHint("key", "Esc", "Cancel"),),
                keycap=keycap,
                presentation=StatusGroups(ui),
            )
            return keycap.size("Esc"), badge.size("Esc"), tuple(m.rect for m in metrics.metrics)

    before = frame(measure)
    spec = deepcopy(runtime.spec)
    spec["parts"].setdefault("status_hints_badge", {}).update(
        paddingLeft=13, paddingRight=17, paddingTop=5, paddingBottom=7
    )
    runtime.update_spec(spec)
    after = frame(measure)
    assert after[0][0] - before[0][0] == pytest.approx(18 * ui.s)
    assert after[0][1] - before[0][1] == pytest.approx(8 * ui.s)
    assert after[1:] == before[1:]
