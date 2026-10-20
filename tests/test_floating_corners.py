"""Floating surfaces and their painted fields share an actual native corner offset."""

from math import sqrt

import pytest
from imgui_bundle import imgui

from mojive.ui.design_palette import rgba8
from mojive.ui.workspace.floating import floating_window
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.widgets import color
from tests.test_component_appearance import appearance as appearance
from tests.test_menu_corner_offsets import _corner_point, _paint_covers

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2), indirect=True)


@pytest.mark.parametrize("radius", (None, 0, 8, 12, 24, 40))
@pytest.mark.parametrize("control", ("text", "choice"))
@pytest.mark.parametrize("modal", (False, True))
def test_floating_surface_balances_actual_inset_and_painted_field_corners(
    appearance, radius, control, modal
):
    ui, _runtime, update, frame = appearance
    if radius is not None:
        update({"dialog" if modal else "form_popup": {"radius": radius}})

    def paint():
        if modal:
            imgui.open_popup("Floating corners")
        imgui.set_next_window_pos((40 * ui.s, 50 * ui.s))
        imgui.set_next_window_size((320 * ui.s, 0))
        with floating_window(
            ui, "Floating corners", modal=modal, flags=imgui.WindowFlags_.always_auto_resize
        ) as opened:
            assert opened
            window = imgui.get_current_context().current_window
            p = imgui.get_cursor_screen_pos()
            width = imgui.get_content_region_avail().x
            if control == "text":
                ui.input_text("corner-field", p.x, p.y, width, "Value")
            elif ui.begin_combo("corner-field", p.x, p.y, width, "Value"):
                imgui.end_combo()
            return window

    for _ in range(3):
        window = frame(paint)
    field = ui.hits["corner-field"]
    inset = field[0] - window.pos.x
    assert window.pos.x + window.size.x - field[0] - field[2] == pytest.approx(inset)
    assert window.pos.y + window.size.y - field[1] - field[3] == pytest.approx(inset, abs=1)
    inner = window.window_rounding - inset
    assert 0 <= inner <= field[3] / 2 - 1
    if radius is None:
        assert inset == native_unit((12 if modal else 8) * ui.s)
        assert inner == pytest.approx(4 * ui.s)
    else:
        assert window.window_rounding == pytest.approx(radius * ui.s)
        if radius < 12:
            assert inset < native_unit(8 * ui.s)
            assert inner < 4 * ui.s
    ink = color(rgba8("field"))
    diagonal = inner * (1 - sqrt(0.5))
    for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
        assert not _paint_covers(
            window.draw_list, _corner_point(field, right, bottom, diagonal - 0.9), ink
        )
        assert _paint_covers(
            window.draw_list, _corner_point(field, right, bottom, diagonal + 0.9), ink
        )
