"""Actual framed controls fit the floating and Timeline parent's native corner budget."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace.timeline_body import body_corner_scope, body_geometry
from mojive.ui.workspace.widgets import color
from mojive.ui.workspace.window_chrome import panel_content_corners, panel_surface
from tests.test_component_appearance import appearance as appearance
from tests.test_menu_corner_offsets import _corner_point, _paint_covers

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


@pytest.mark.parametrize("consumer", ("ordinary", "timeline"))
@pytest.mark.parametrize("authored", (False, True))
def test_real_control_frame_and_triangles_fit_the_parent_derived_curve(
    appearance, monkeypatch, consumer, authored
):
    ui, _runtime, update, frame = appearance
    update(
        {
            "panel": {"radius": 80} if authored and consumer == "ordinary" else {},
            "timeline_body": {"radius": 23} if authored and consumer == "timeline" else {},
        }
    )
    paints, actual = [], SimpleNamespace()
    rectangle = ui.rect

    def rect(x, y, width, height, tint, r=None, **options):
        draw = options.get("draw") or ui.draw
        ink = tint if options.get("color_resolved") else ui.ink(tint)
        result = rectangle(x, y, width, height, tint, r, **options)
        paints.append(((x, y, width, height), (r or 0) * ui.s, color(ink), draw))
        return result

    monkeypatch.setattr(ui, "rect", rect)

    def draw():
        ui.design.begin_frame()
        paints.clear()
        if consumer == "ordinary":
            imgui.set_next_window_pos((40 * ui.s, 40 * ui.s))
            imgui.set_next_window_size((500 * ui.s, 450 * ui.s))
            with panel_surface(ui, docked=False, name="Custom"):
                imgui.begin("Custom", flags=imgui.WindowFlags_.no_title_bar)
                window = imgui.internal.get_current_window()
                with panel_content_corners(ui, "Custom", window):
                    pos = imgui.get_cursor_screen_pos()
                    ui.input("corner-number", pos.x, pos.y, 120 * ui.s, 0.0)
                    actual.frame = ui.hits["corner-number-frame"]
                    actual.outer, actual.inset = window.window_rounding, window.window_padding.x
                imgui.end()
        else:
            geometry = body_geometry(ui)
            with body_corner_scope(ui, geometry):
                ui.combo("corner-select", 40 * ui.s, 40 * ui.s, 180 * ui.s, 0, ("Scene",))
                actual.frame = ui.hits["corner-select"]
                actual.outer, actual.inset = geometry.radius, geometry.padding[0]

    for _ in range(3):
        frame(draw)
    bounds, radius, ink, draw_list = next(
        paint for paint in paints if paint[0] == pytest.approx(actual.frame)
    )
    assert radius == pytest.approx(max(0, actual.outer - actual.inset))
    assert radius <= min(bounds[2], bounds[3]) / 2 - 1
    if authored:
        assert actual.outer == pytest.approx((80 if consumer == "ordinary" else 23) * ui.s)
    diagonal = radius * (1 - 0.5**0.5)
    for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
        assert not _paint_covers(
            draw_list, _corner_point(bounds, right, bottom, diagonal - 0.9), ink
        )
        assert _paint_covers(draw_list, _corner_point(bounds, right, bottom, diagonal + 0.9), ink)
