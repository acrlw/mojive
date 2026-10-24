"""Popup class overrides retain real insets without resetting global corners."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row
from mojive.ui.workspace.popup_style import (
    menu_popup_content,
    menu_popup_style,
    prepare_menu_popup,
)
from mojive.ui.workspace.widgets import UI
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("scale", (1.0, 1.2, 1.25, 2.25))
@pytest.mark.parametrize("part", ("menu", "select_popup"))
@pytest.mark.parametrize("live", (False, True))
def test_popup_class_insets_override_legacy_global_radius(tmp_path, scale, part, live):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        io.display_size = (1000 * scale, 800 * scale)
        ui = UI(
            SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
            portable_fonts=True,
        )
        runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
        spec = deepcopy(runtime.spec)
        spec["tokens"]["radius_popup"] = 8
        spec["parts"]["menu_item"]["radius"] = 6
        spec["parts"]["settings_nav"] = {"iconSize": 20}
        spec["parts"]["segment"] = {"fontSize": 12, "iconSize": 16}
        if live:
            spec["parts"]["settings_nav_button"] = {"height": 36}
        for owner in ("menu", "select_popup"):
            spec["parts"].setdefault(owner, {}).update(
                paddingLeft=8, paddingTop=8, paddingRight=8, paddingBottom=8, radius=14
            )
        runtime.update_spec(spec)
        ui.design = DesignPresentation(ui, runtime)
        for _ in range(5):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Popup profile", flags=imgui.WindowFlags_.no_title_bar)
            if not imgui.is_popup_open("Profile choices"):
                imgui.open_popup("Profile choices")
            with menu_popup_style(ui, part=part):
                prepare_menu_popup(ui, auto_minimum_width=240 * scale, part=part)
                if imgui.begin_popup("Profile choices"):
                    with menu_popup_content(ui, "Profile choices", part=part) as box:
                        rows = []
                        for index, label in enumerate(("Front", "Back", "Top")):
                            row = paint_menu_row(
                                ui,
                                Item(label, icon="view-" + label.lower(), checked=index == 0),
                                "row-" + str(index),
                                content_right=box.content.right,
                                part="select_option" if part == "select_popup" else "menu_item",
                            )
                            rows.append(row.native_rect)
                            advance_menu_row(ui, row, part=part)
                        window = imgui.internal.get_current_window()
                        bounds = (window.pos.x, window.pos.y, window.size.x, window.size.y)
                        rounding = window.window_rounding
                    imgui.end_popup()
            imgui.end()
            imgui.render()
        x, y, width, height = bounds
        first, last = rows[0], rows[-1]
        actual = (
            first[0] - x,
            first[1] - y,
            x + width - first[0] - first[2],
            y + height - last[1] - last[3],
        )
        assert actual == pytest.approx((8 * scale,) * 4, abs=1)
        assert rounding == pytest.approx(
            14 * scale if part == "select_popup" else 6 * scale + native_unit(8 * scale)
        )
        assert runtime.spec == spec
        assert runtime.spec["tokens"]["radius_popup"] == 8
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("appearance", (1.0, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("radius", (None, 0, 19))
def test_select_popup_radius_is_independent_of_its_preserved_inset(appearance, radius):
    ui, runtime, update, frame = appearance
    if radius is not None:
        update({"select_popup": {**runtime.spec["parts"]["select_popup"], "radius": radius}})

    def draw():
        if not imgui.is_popup_open("Moderate choices"):
            imgui.open_popup("Moderate choices")
        before = imgui.get_style().popup_rounding
        actual = None
        with menu_popup_style(ui, part="select_popup"):
            prepare_menu_popup(ui, auto_minimum_width=240 * ui.s, part="select_popup")
            if imgui.begin_popup("Moderate choices"):
                with menu_popup_content(ui, "Moderate choices", part="select_popup") as box:
                    for index, label in enumerate(("Front", "Back", "Top")):
                        row = paint_menu_row(
                            ui,
                            Item(label, checked=index == 0),
                            "moderate-row-" + str(index),
                            content_right=box.content.right,
                            part="select_option",
                        )
                        advance_menu_row(ui, row, part="select_popup")
                    actual = box, imgui.internal.get_current_window().window_rounding
                imgui.end_popup()
        assert imgui.get_style().popup_rounding == before
        return actual

    for _ in range(5):
        box, rounding = frame(draw)
    assert box.padding == (native_unit(4 * ui.s),) * 4
    assert rounding == pytest.approx((8 if radius is None else radius) * ui.s)
