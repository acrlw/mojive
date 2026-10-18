"""Choice selection keeps an accent mark while hover uses a neutral surface."""

import pytest
from imgui_bundle import imgui

from mojive.ui.design_palette import rgba8
from mojive.ui.interaction_paint import ItemPaint, reference_paint
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import paint_menu_row
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 2.25), indirect=True)


@pytest.mark.parametrize(
    "checked,paint,role,alpha",
    (
        (True, ItemPaint(), "primary", 24),
        (True, ItemPaint(hovered=True), "field_hovered", 255),
        (False, ItemPaint(hovered=True), "field_hovered", 255),
        (True, ItemPaint(hovered=True, active=True), "field_active", 255),
        (True, ItemPaint(disabled=True), None, None),
        (False, ItemPaint(), None, None),
    ),
    ids=("selected", "selected-hover", "hover", "selected-press", "disabled", "idle"),
)
def test_choice_surface_alpha_and_selected_mark_remain_independent(
    appearance, checked, paint, role, alpha
):
    ui, runtime, _update, frame = appearance
    observed = {}

    def draw():
        runtime.begin_frame()
        imgui.set_cursor_screen_pos((40 * ui.s, 50 * ui.s))
        draw_list = imgui.get_window_draw_list()
        first_vertex = len(draw_list.vtx_buffer)
        with reference_paint(paint):
            result = paint_menu_row(
                ui,
                Item("Viewport with overlays", checked=checked, checkable=True),
                "choice-state",
                content_right=340 * ui.s,
                part="select_option",
            )
        observed["row"] = result.observation
        observed["colors"] = {
            draw_list.vtx_buffer[index].col
            for index in range(first_vertex, len(draw_list.vtx_buffer))
        }
        style = ui.component_style("select_option")
        if role is not None:
            tint = ui.ink(rgba8(role, alpha), style)
            observed["background"] = imgui.get_color_u32(
                imgui.ImVec4(*(channel / 255 for channel in tint))
            )
            assert tint[3] == alpha
        if checked and not paint.disabled:
            accent = ui.ink(rgba8("primary"), style)
            observed["check"] = imgui.get_color_u32(
                imgui.ImVec4(*(channel / 255 for channel in accent))
            )

    for _ in range(3):
        frame(draw)
    assert observed["row"]["checked"] is checked
    assert (observed["row"]["backgroundRect"] is not None) is (role is not None)
    if role is not None:
        assert observed["background"] in observed["colors"]
    if checked and not paint.disabled:
        assert observed["check"] in observed["colors"]
