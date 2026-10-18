"""Production workspace menus render registered glyphs and route entity actions."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.navigation import Navigation
from mojive.ui.workspace.widgets import UI


@pytest.mark.parametrize("name", ("Window", "Inspector position", "Simulate", "Playback speed"))
@pytest.mark.parametrize("surface", ("workspace", "guide"))
def test_localized_product_menus_keep_command_identity_and_fit_labels(name, surface, monkeypatch):
    from examples.ui_design.native.guide_localization import translate

    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1000, 900)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=1, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    localizer = Localizer(Language.SIMPLIFIED_CHINESE)
    ui.translate = (
        localizer.text if surface == "workspace" else lambda value: translate(localizer, value)
    )
    session = Session(StaticSceneAdapter(Scene()))
    workspace = SimpleNamespace(
        ui=ui,
        s=1,
        app=SimpleNamespace(session=session, entity=None, panels=[]),
        playing=False,
        saved=False,
        speed=1,
        show_viewport_tab=False,
        open={"Inspector": True, "Timeline": False},
    )
    menus = Menus(workspace, ())
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append(value)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    try:
        rows = [row for row in menus.items(name) if row]
        identity = [(row.label, row.action, row.submenu) for row in rows]
        menus.preview((name,))
        for _ in range(3):
            painted.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Menu host")
            menus.begin_frame()
            menus.popup(name, (20, 20, 100, 30))
            imgui.end()
            imgui.render()
        observed = menus.observed[name]["rows"]
        assert [row["label"] for row in observed] == [row.label for row in rows]
        assert all(row["visible"] for row in observed)
        for row in rows:
            assert ui.translate(row.label) != row.label
            assert ui.translate(row.label) in painted
            assert "menu-item-" + name + "/" + row.label in ui.hits
        localizer.set_language(Language.ENGLISH, persist=False)
        assert all(ui.translate(row.label) == row.label for row in rows)
        assert identity == [
            (row.label, row.action, row.submenu) for row in menus.items(name) if row
        ]
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_scene_context_menu_renders_and_routes_rename(scale, monkeypatch):
    from mojive.ui.workspace import widgets

    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * scale, 600 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    scene = Scene()
    obj = scene.box(name="box")
    session = Session(StaticSceneAdapter(scene))
    app = SimpleNamespace(session=session)
    workspace = SimpleNamespace(app=app, ui=ui, s=scale)
    workspace.menus = Menus(workspace, ())
    painted = []
    original = widgets.draw_icon

    def observe(draw, name, *args, **kwargs):
        painted.append(name)
        return original(draw, name, *args, **kwargs)

    monkeypatch.setattr(widgets, "draw_icon", observe)
    navigation = Navigation.__new__(Navigation)
    navigation.workspace = workspace
    node = session.node_by_object_id(obj.object_id)
    requests = []
    ctx = PanelContext(session, NullBackend(), style_scale=scale, request_rename=requests.append)
    popup = "entity-context-" + str(node.node_id)

    def frame(*, open_popup=False):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Scene")
        ui.button("entity", 20 * scale, 20 * scale, 100, label=node.name)
        if open_popup:
            imgui.open_popup(popup)
        navigation._scene_context(ctx, node)
        imgui.end()
        imgui.render()

    try:
        frame()
        frame(open_popup=True)
        frame()
        assert {"scene-menu-duplicate", "scene-menu-rename", "scene-menu-delete"} <= ui.hits.keys()
        assert "duplicate" in painted
        x, y, width, height = ui.hits["scene-menu-rename"]
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()
        assert requests == [obj.object_id]
    finally:
        session.release()
        imgui.destroy_context(context)
