"""Fractional menus use one native layout lattice and complete production text."""

from types import SimpleNamespace

import pytest
from examples.ui_design.native.guide_slices import menus as guide_menus
from imgui_bundle import imgui

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.session import Session
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.panel_host import panel_header
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1.1, 1.2, 1.25, 1.3, 1.5, 2.25), indirect=True)


def observe_text(monkeypatch):
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append(value)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    return painted


def assert_popup(rows, popup, scale):
    first, last = rows[0]["frame_rect"], rows[-1]["frame_rect"]
    x, y, w, h = popup["rect"]
    insets = (first[0] - x, first[1] - y, x + w - last[0] - last[2], y + h - last[1] - last[3])
    assert insets == (native_unit(4 * scale),) * 4
    assert all(not row["clippedLabel"] and not row["clippedShortcut"] for row in rows)
    assert popup["radius"] * scale == pytest.approx(rows[-1]["backgroundRadius"] + insets[0])


@pytest.mark.parametrize("family", ("system", "portable"))
def test_real_guide_edit_menu_paints_complete_chinese_and_equal_native_insets(
    appearance, monkeypatch, family
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            "menu_item": {
                "fontFamily": family,
                "fontSize": 14,
                "height": 30,
                "paddingX": 6,
                "gap": 6,
                "radius": 6,
            }
        }
    )
    ui.translate = Localizer(Language.SIMPLIFIED_CHINESE).text
    imgui.get_io().config_mac_osx_behaviors = True
    session = Session(StaticSceneAdapter(Scene()))
    app = SimpleNamespace(
        session=session, entity={"hidden": False}, model_edits=SimpleNamespace(active=False)
    )
    workspace = SimpleNamespace(ui=ui, s=ui.s, app=app)
    workspace.menus = menus = Menus(workspace, ())
    view = SimpleNamespace(
        ui=ui,
        s=ui.s,
        width=1000,
        hits={},
        text=lambda *_args, **_kwargs: None,
        sample=SimpleNamespace(app=SimpleNamespace(workspace=workspace)),
    )
    view.rect = lambda x, y, w, h: Rect((300 + x) * ui.s, (180 + y) * ui.s, w * ui.s, h * ui.s)
    painted = observe_text(monkeypatch)
    menus.preview(("Edit",))

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        painted.clear()
        menus.begin_frame()
        guide_menus(view)

    try:
        for hidden in (False, True, False):
            app.entity["hidden"] = hidden
            for _ in range(6):
                frame(draw)
            popup = menus.observed["Edit"]
            assert_popup(popup["rows"], popup, ui.s)
            assert {"撤销", "重做", "可见"} <= set(painted)
            assert not any("…" in text for text in painted)
            assert popup["rows"][-1]["checked"] is not hidden
    finally:
        session.release()


@pytest.mark.parametrize("family", ("system", "portable"))
def test_actual_camera_header_actions_submenu_owns_its_text_width_and_insets(
    appearance, monkeypatch, family
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            "menu_item": {
                "fontFamily": family,
                "fontSize": 14,
                "height": 30,
                "paddingX": 6,
                "gap": 6,
                "radius": 6,
            }
        }
    )
    ui.translate = Localizer(Language.SIMPLIFIED_CHINESE).text
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(gizmos=True),
        open={"Cameras": True},
        header_actions={"Cameras": {"frame"}},
        header_add_pending=False,
        header_action=lambda *_args: None,
    )
    workspace.menus = menus = Menus(workspace, ())
    painted = observe_text(monkeypatch)

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        menus.begin_frame()
        painted.clear()
        with group_scope("Cameras"):
            panel_header(workspace, "Cameras", 30 * ui.s, 30 * ui.s, 320 * ui.s)

    def event(method, *args):
        getattr(imgui.get_io(), method)(*args)
        frame(draw)

    def move(key):
        x, y, w, h = ui.hits[key]
        event("add_mouse_pos_event", x + w / 2, y + h / 2)

    for _ in range(3):
        frame(draw)
    move("header-Cameras-more")
    event("add_mouse_button_event", 0, True)
    event("add_mouse_button_event", 0, False)
    for _ in range(4):
        frame(draw)
    move("header-menu-Cameras-actions")
    for _ in range(40):
        frame(draw)
    move("header-action-Cameras-frame")
    for _ in range(4):
        frame(draw)
    key = "header-menu-Cameras-actions"
    popup = menus.observed[key]
    assert len(popup["rows"]) == 1
    assert_popup(popup["rows"], popup, ui.s)
    row = popup["rows"][0]
    assert row["checked"] and row["checkable"] and row["hovered"]
    assert "显示整个场景" in painted
    assert not any("…" in text for text in painted)
    assert [item.label for item in menus._popup_rows[key]] == ["Frame all"]
    assert all(
        item.label != "Frame all" or not item.checkable
        for item in menus._popup_rows["Panel actions###Cameras"]
    )
    actual = next(row for row in runtime.observed if row["key"] == "header-action-Cameras-frame")
    x, y, w, h = actual["backgroundRect"]
    assert x >= actual["clip"][0] and y >= actual["clip"][1]
    assert x + w <= actual["clip"][2] and y + h <= actual["clip"][3]


def test_asymmetric_and_independent_radius_overrides_survive_native_quantization(appearance):
    ui, _runtime, update, frame = appearance
    from mojive.design.native.geometry import Box
    from mojive.ui.workspace.menu_geometry import native_box, popup_radius

    update(
        {
            "menu": {
                "paddingLeft": 3,
                "paddingTop": 7,
                "paddingRight": 11,
                "paddingBottom": 5,
                "radius": 24,
            }
        }
    )

    def draw():
        style = ui.component_style("menu")
        padding = style.edges("padding", ui.s)
        outer = Rect(10, 10, 400, 200)
        box = native_box(Box(outer, outer, outer.inset(*padding), padding, (0, 0, 0, 0)))
        assert box.padding == tuple(native_unit(value * ui.s) for value in (3, 7, 11, 5))
        assert popup_radius(ui, "menu", box.padding) == 24 * ui.s

    frame(draw)
