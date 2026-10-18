"""Application menus use click activation and native popup dismissal semantics."""

from types import SimpleNamespace

import pytest
from examples.ui_design.native.reference_menus import Menus as ReferenceMenus
from imgui_bundle import imgui

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.25))
def scale(request):
    return request.param


@pytest.fixture(params=(False, True), ids=("default", "designed"))
def designed(request):
    return request.param


@pytest.fixture(params=(Menus, ReferenceMenus), ids=("workspace", "reference"))
def menu_bar(request, tmp_path, scale, designed):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    ui = UI(SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)))
    if designed:
        ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "ui.json")))
    workspace = SimpleNamespace(ui=ui, s=scale)
    menus = request.param(workspace, ())
    menus.items = lambda name: [
        Item("Run " + name, action=name),
        Item("Nested", submenu="Nested") if name != "Nested" else Item("Nested action"),
    ]
    actions = []
    menus.dispatch = actions.append
    state = SimpleNamespace(
        menus=menus, ui=ui, io=io, width=900, actions=actions, open_unrelated=False
    )

    def frame(count=1):
        for _ in range(count):
            io.display_size = (state.width * scale, 700 * scale)
            imgui.new_frame()
            if ui.design is not None:
                ui.design.begin_frame()
            menus.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin(
                "Application", flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.menu_bar
            )
            menus.bar((state.width - 30) * scale)
            if state.open_unrelated:
                imgui.open_popup("Unrelated")
                state.open_unrelated = False
            if imgui.begin_popup("Unrelated"):
                imgui.text_unformatted("Other popup")
                imgui.end_popup()
            imgui.end()
            if isinstance(menus, Menus):
                menus.apply_actions()
            imgui.render()

    def hover(rect, count=3):
        x, y, width, height = rect
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        frame(count)

    def click(rect):
        hover(rect)
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame(3)

    state.frame, state.hover, state.click = frame, hover, click
    frame(3)
    try:
        yield state
    finally:
        imgui.destroy_context(context)


def test_idle_menu_hover_does_not_open_the_menu(menu_bar):
    for name in ("File", "Window", "Edit"):
        menu_bar.hover(menu_bar.menus.menu_rects[name])
        assert not menu_bar.menus.observed


def test_activated_menu_switches_between_siblings_on_hover(menu_bar):
    menus = menu_bar.menus
    menu_bar.click(menus.menu_rects["File"])
    assert "File" in menus.observed
    for name in ("Window", "Edit", "Help", "File"):
        menu_bar.hover(menus.menu_rects[name])
        assert set(menus.observed) == {name}


def test_nested_menu_switches_to_another_top_menu_on_hover(menu_bar):
    state, menus = menu_bar, menu_bar.menus
    state.click(menus.menu_rects["File"])
    state.hover(menus.observed["File"]["rows"][1]["rect"], count=8)
    assert "File/Nested" in menus.observed
    state.hover(menus.menu_rects["Window"])
    assert set(menus.observed) == {"Window"}


def test_unrelated_popup_does_not_activate_the_menu_bar(menu_bar):
    state = menu_bar
    state.open_unrelated = True
    state.frame(3)
    state.hover(state.menus.menu_rects["File"])
    assert not state.menus.observed


@pytest.mark.parametrize("dismissal", ("outside", "escape", "same_trigger", "action"))
def test_dismissed_menu_returns_to_idle_hover(menu_bar, dismissal):
    state, menus = menu_bar, menu_bar.menus
    state.click(menus.menu_rects["File"])
    assert "File" in menus.observed
    if dismissal == "outside":
        state.click((600, 500, 30, 30))
    elif dismissal == "escape":
        state.io.add_key_event(imgui.Key.escape, True)
        state.frame()
        state.io.add_key_event(imgui.Key.escape, False)
        state.frame(3)
    elif dismissal == "same_trigger":
        state.click(menus.menu_rects["File"])
    else:
        state.click(menus.observed["File"]["rows"][0]["rect"])
        assert state.actions == ["File"]
    assert not menus.observed
    state.hover(menus.menu_rects["Edit"])
    assert not menus.observed


def test_activated_overflow_and_visible_menus_share_hover_activation(menu_bar):
    state, menus = menu_bar, menu_bar.menus
    state.width = 300
    state.frame(3)
    assert menus.visible_menus and menus.overflow_menus
    first = menus.visible_menus[0]
    state.click(menus.menu_rects[first])
    assert first in menus.observed
    state.hover(menus.menu_rects["More"])
    assert "More" in menus.observed
    state.hover(menus.menu_rects[first])
    assert set(menus.observed) == {first}
