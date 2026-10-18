"""Production panel text uses hierarchy without shrinking ordinary controls."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels.camera import CameraPanel
from mojive.ui.workspace.panel_host import panel_header
from mojive.ui.workspace.property_style import NativePropertyPresentation
from tests.test_component_appearance import appearance as appearance


def test_camera_source_keeps_body_size_beneath_weighted_panel_headings(appearance, monkeypatch):
    ui, runtime, update, frame = appearance
    panel = CameraPanel()
    panel.presentation = NativePropertyPresentation(ui, prefix="camera")
    ctx = SimpleNamespace(
        session=SimpleNamespace(cameras=(SimpleNamespace(camera_id=0, name="front"),)),
        model_camera_id=-1,
        select_model_camera=None,
        tr=ui.translate,
    )
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        header_actions={"Cameras": ()},
        menus=SimpleNamespace(local_popup=lambda _key: nullcontext(False)),
    )
    painted = {}
    original = ImguiDraw2D.text

    def observe(draw, position, color, value, **kwargs):
        painted[value] = (imgui.get_font_size() / ui.s, imgui.get_font().font_id)
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)

    def paint():
        with group_scope("Cameras"):
            panel_header(workspace, "Cameras", 40 * ui.s, 30 * ui.s, 500 * ui.s)
            ui.section("View direction", 40 * ui.s, 90 * ui.s, 500 * ui.s, divider=False)
            imgui.set_cursor_screen_pos((40 * ui.s, 150 * ui.s))
            imgui.set_next_item_width(500 * ui.s)
            panel._source(ctx)

    source = ui.translate("source") + ": " + ui.translate("Editor Camera")
    frame(paint)
    assert painted[ui.translate("Cameras")] == (16, ui.bold.font_id)
    assert painted[ui.translate("View direction")] == (14, ui.bold.font_id)
    assert painted[source] == (14, ui.body.font_id)
    update({"cameras_panel_header": {"fontSize": 20}, "cameras_section_header": {"fontSize": 15}})
    frame(paint)
    assert painted[ui.translate("Cameras")] == (20, ui.bold.font_id)
    assert painted[ui.translate("View direction")] == (15, ui.bold.font_id)
    assert painted[source] == (14, ui.body.font_id)
    spec = deepcopy(runtime.spec)
    spec["parts"]["cameras_panel_header"] = {}
    spec["parts"]["cameras_section_header"] = {}
    spec["tokens"]["title_font_size"] = 18
    runtime.update_spec(spec)
    frame(paint)
    assert painted[ui.translate("Cameras")] == (18, ui.bold.font_id)
    assert painted[ui.translate("View direction")] == (14, ui.bold.font_id)
    assert painted[source] == (14, ui.body.font_id)
