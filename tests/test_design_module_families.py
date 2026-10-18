"""Module appearance follows the nearest semantic window through real dock hosts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import THEME
from mojive.ui.workspace import widgets
from mojive.ui.workspace.widgets import MUTED, TEXT, UI


@pytest.fixture
def module_ui(tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1000, 700)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.docking_enable
    ui = UI(SimpleNamespace(style_scale=1.0, font_report=SimpleNamespace(cjk_path=None)))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    spec = deepcopy(runtime.spec)
    spec["components"] = {
        "navigation": {"font_family": "portable", "body_font_size": 20},
        "inspector": {"font_family": "system", "body_font_size": 18},
        "chrome": {"font_family": "portable", "body_font_size": 16},
    }
    runtime.update_spec(spec)
    ui.design = DesignPresentation(ui, runtime)
    try:
        yield ui, io
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "name,family,size,font",
    (
        ("Scene", "navigation", 20, "portable"),
        ("场景###Scene", "navigation", 20, "portable"),
        ("Inspector", "inspector", 18, "system"),
    ),
)
def test_docked_panels_children_and_popups_keep_their_module_font(
    module_ui, name, family, size, font
):
    ui, _io = module_ui
    seen = []

    def check():
        current = imgui.get_current_context().current_window
        ancestors = []
        while current is not None:
            ancestors.append(current.name)
            current = current.parent_window
        assert "DockHost" in ancestors
        assert ui.design.family() == family
        style = ui.component_style("input_text")
        assert style.family == family
        assert style.get("fontSize", 14) == size
        assert ui.resolve_font(style).font_id == ui._font_families[font][1][0].font_id
        seen.append(ancestors)

    for index in range(3):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((1000, 700))
        imgui.begin("DockHost", flags=imgui.WindowFlags_.no_docking)
        dock_id = imgui.get_id("DockSpace")
        if index == 0:
            imgui.internal.dock_builder_add_node(dock_id, imgui.DockNodeFlags_.none)
            imgui.internal.dock_builder_set_node_size(dock_id, (1000, 700))
            imgui.internal.dock_builder_dock_window(name, dock_id)
            imgui.internal.dock_builder_finish(dock_id)
        imgui.dock_space(dock_id, (0, 0))
        imgui.end()
        imgui.begin(name)
        check()
        imgui.begin_child("Panel content", (300, 200))
        check()
        if index == 1:
            imgui.open_popup("Context")
        if imgui.begin_popup("Context"):
            check()
            imgui.text_unformatted("Menu")
            imgui.end_popup()
        imgui.end_child()
        imgui.end()
        imgui.render()
    assert len(seen) == 8
    assert any(chain[0].startswith("##Popup_") for chain in seen)


@pytest.mark.parametrize("name", ("Application", "Navigation", "Status###application_status"))
def test_native_chrome_windows_and_children_use_the_chrome_module(module_ui, name):
    ui, _io = module_ui
    imgui.new_frame()
    imgui.begin(name)
    try:
        assert ui.design.family() == "chrome"
        imgui.begin_child("Content", (200, 100))
        try:
            assert ui.design.family() == "chrome"
            style = ui.component_style("input_text")
            assert style.family == "chrome" and style.get("fontSize", 14) == 16
            assert ui.resolve_font(style).font_id == ui._font_families["portable"][1][0].font_id
        finally:
            imgui.end_child()
    finally:
        imgui.end()
        imgui.render()


def test_docked_module_inks_apply_between_class_and_global_colors(module_ui, monkeypatch):
    ui, _io = module_ui
    runtime = ui.design.runtime
    spec = deepcopy(runtime.spec)
    spec["tokens"].update(text_color="#2020ff", muted_color="#00ffff")
    spec["components"]["navigation"].update(text_color="#ff0000", muted_color="#ffff00")
    spec["parts"]["button"].update(textColor="#00ff00", iconColor="#ff00ff")
    runtime.update_spec(spec)
    ui.window._imgui_context = imgui.get_current_context()
    ui.window.apply_theme = lambda theme: None
    shell = SimpleNamespace(window=ui.window, app=SimpleNamespace(theme=THEME), s=1.0)
    runtime.apply_theme(shell)
    painted, icons = {}, []
    original = ImguiDraw2D.text

    def capture(draw, position, color, value, **options):
        painted[value] = color
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", capture)
    monkeypatch.setattr(
        widgets, "draw_icon", lambda draw, name, x, y, size, tint, scale: icons.append(tint)
    )
    imgui.new_frame()
    imgui.set_next_window_size((1000, 700))
    imgui.begin("DockHost", flags=imgui.WindowFlags_.no_docking)
    dock_id = imgui.get_id("DockSpace")
    imgui.internal.dock_builder_add_node(dock_id, imgui.DockNodeFlags_.none)
    imgui.internal.dock_builder_set_node_size(dock_id, (1000, 700))
    imgui.internal.dock_builder_dock_window("Scene", dock_id)
    imgui.internal.dock_builder_finish(dock_id)
    imgui.dock_space(dock_id, (0, 0))
    imgui.end()
    imgui.begin("Scene")
    try:
        scene_style = ui.component_style("input_text")
        assert scene_style.family == "navigation"
        ui.component_text("input_text", Rect(30, 40, 250, 32), "Module", TEXT)
        ui.text_draw(ImguiDraw2D(), "input_text").text(
            (30, 90), shell.app.theme.text, "Applied theme"
        )
        ui.icon("camera", 30, 140, c=MUTED)
        ui.button("class", 30, 180, 220, 32, icon="camera", label="Class")
        painter = ui.design.primitives
        box = scene_style.box(30, 230, 220, 32)
        with painter.native_field(scene_style, box):
            assert tuple(imgui.get_style().color_(imgui.Col_.text)) == (1, 0, 0, 1)
            imgui.input_text("##native", "Module")
    finally:
        imgui.end()
    imgui.begin("Inspector")
    try:
        ui.component_text("input_text", Rect(400, 40, 250, 32), "Global", TEXT)
        ui.icon("camera", 400, 90, c=MUTED)
        # Cached exact styles retain their panel identity across native windows.
        assert ui.ink(TEXT, scene_style) == (255, 0, 0, 255)
    finally:
        imgui.end()
        imgui.render()
    assert painted["Module"] == painted["Applied theme"] == (1, 0, 0, 1)
    assert painted["Class"] == (0, 1, 0, 1)
    assert painted["Global"] == (32 / 255, 32 / 255, 1, 1)
    assert icons == [(255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255)]


def test_module_palette_caches_and_invalidates_without_remapping_semantic_inks(
    module_ui, monkeypatch
):
    ui, _io = module_ui
    runtime = ui.design.runtime
    spec = deepcopy(runtime.spec)
    spec["tokens"]["muted_color"] = "#102030"
    spec["components"]["navigation"].update(muted_color="#aabbcc", input_color="#314159")
    runtime.update_spec(spec)
    tint = (170, 187, 204, 255)
    assert runtime.ink(MUTED, "navigation") == tint
    assert runtime.ink((16, 32, 48, 255), "navigation") == tint
    themed = runtime.theme_for(THEME, "navigation")
    assert themed.bg_frame == (49 / 255, 65 / 255, 89 / 255, 1)
    assert runtime.theme_for(THEME, "inspector") is THEME
    assert runtime.ink(rgba8("danger"), "navigation") == rgba8("danger")
    assert runtime.ink(rgba8("info"), "navigation") == rgba8("info")
    calls = []
    original = runtime.value

    def resolve(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(runtime, "value", resolve)
    for _ in range(100):
        assert runtime.ink(MUTED, "navigation") == tint
        assert runtime.theme_for(THEME, "navigation") is themed
    assert calls == []
    updated = deepcopy(spec)
    updated["components"]["navigation"]["muted_color"] = "#112233"
    runtime.update_spec(updated)
    assert runtime._module_inks == runtime._module_themes == {}
    assert runtime.ink(MUTED, "navigation") == (17, 34, 51, 255)
    assert runtime.theme_for(THEME, "navigation") is not themed


def test_font_only_modules_do_not_add_a_window_walk_to_global_ink_resolution(
    module_ui, monkeypatch
):
    ui, _io = module_ui
    runtime = ui.design.runtime
    assert not runtime.has_module_palette

    def unexpected_walk():
        pytest.fail("Global palette resolution must not traverse native window ancestors")

    monkeypatch.setattr(ui.design, "family", unexpected_walk)
    assert ui.ink(TEXT) == rgba8("text")
    assert runtime.theme_for(THEME, "navigation") is THEME
