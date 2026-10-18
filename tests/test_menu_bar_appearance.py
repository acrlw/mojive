"""Menu-bar typography shares trigger measurement without styling popup content."""

from copy import deepcopy
from math import floor
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.widgets import UI


def test_menu_bar_font_reaches_triggers_and_stays_out_of_popups(tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900, 700)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=1.0, font_report=SimpleNamespace(cjk_path=None)))
    ui.prepare_fonts("system", ("portable",))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    spec = deepcopy(DEFAULT)
    spec["parts"]["menu_bar"] = {"fontFamily": "portable", "fontSize": 22, "textColor": "#ff0000"}
    runtime.update_spec(spec)
    ui.design = DesignPresentation(ui, runtime)
    menus = Menus(SimpleNamespace(ui=ui, s=1.0), ())
    menus.items = lambda name: [Item("Menu action")]
    menus.preview(("File",))
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append((value, imgui.get_font().font_id, imgui.get_font_size(), color))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    try:
        for _ in range(3):
            painted.clear()
            imgui.new_frame()
            menus.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.push_style_var(imgui.StyleVar_.frame_padding, (8, 16))
            imgui.begin("Application", flags=imgui.WindowFlags_.menu_bar)
            menus.bar(880)
            imgui.end()
            imgui.pop_style_var()
            imgui.render()
        trigger = next(item for item in painted if item[0] == "File")
        action = next(item for item in painted if item[0] == "Menu action")
        assert trigger[1:] == (ui._font_families["portable"][1][0].font_id, 22, (1, 0, 0, 1))
        assert action[1] == ui.body.font_id
        assert action[2] == 14
        assert action[3] != trigger[3]
        assert ui.hits["menu-File"][2] == menus.trigger_size("File")[0]
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("scale", (1, 1.2, 1.25, 2.25))
@pytest.mark.parametrize("local_trigger_size", (None, 18))
def test_production_menu_titles_use_body_size_and_keep_local_authoring(
    tmp_path, monkeypatch, scale, local_trigger_size
):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1200 * scale, 700 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)))
    ui.prepare_fonts("portable")
    runtime = DesignRuntime(SpecificationStore(tmp_path / "menu-design.json"))
    spec = deepcopy(DEFAULT)
    if local_trigger_size is not None:
        spec["parts"]["menu_trigger"] = {"fontSize": local_trigger_size}
    runtime.update_spec(spec)
    ui.design = DesignPresentation(ui, runtime)
    menus = Menus(SimpleNamespace(ui=ui, s=scale), ())
    menus.items = lambda name: [Item("Menu action")]
    menus.preview(("File",))
    painted, original = {}, ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted[value] = (imgui.get_font().font_id, imgui.get_font_size())
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    try:
        for _ in range(3):
            painted.clear()
            imgui.new_frame()
            menus.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.push_style_var(imgui.StyleVar_.frame_padding, (8 * scale, 16 * scale))
            imgui.begin("Application", flags=imgui.WindowFlags_.menu_bar)
            with ui.design.family_scope("chrome"), part_scope("titlebar"):
                menus.bar(1180 * scale)
            imgui.end()
            imgui.pop_style_var()
            imgui.render()
        assert ui.component_style("button").get("fontSize") == 12
        assert painted["File"][1] == floor((local_trigger_size or 14) * scale + 0.5)
        assert painted["Menu action"][1] == floor(14 * scale + 0.5)
        assert painted["File"][0] == painted["Menu action"][0]
        assert ui.hits["menu-File"][2] == pytest.approx(menus.trigger_size("File")[0] * scale)
    finally:
        imgui.destroy_context(context)
