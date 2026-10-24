"""Menu rows paint the corner offset of their actual native popup owner."""

from copy import deepcopy
from math import sqrt
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.specimens import ComponentSpecimens
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.widgets import color
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1.0, 1.2, 1.25, 2.25), indirect=True)


def _paint_covers(draw, point, ink):
    """Inspect actual CPU draw triangles, excluding the anti-alias fringe."""
    vertices, indices = list(draw.vtx_buffer), list(draw.idx_buffer)

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for offset in range(0, len(indices), 3):
        triangle = [vertices[indices[offset + i]] for i in range(3)]
        if any(vertex.col != ink for vertex in triangle):
            continue
        a, b, c = (tuple(vertex.pos) for vertex in triangle)
        sides = cross(a, b, point), cross(b, c, point), cross(c, a, point)
        if min(sides) >= 0 or max(sides) <= 0:
            return True
    return False


def _corner_point(bounds, right, bottom, offset):
    x, y, width, height = bounds
    return (
        x + width - offset if right else x + offset,
        y + height - offset if bottom else y + offset,
    )


@pytest.mark.parametrize("path", ("named", "local"))
@pytest.mark.parametrize("radius", (0, 3, 19, 24))
@pytest.mark.parametrize("asymmetric", (False, True))
@pytest.mark.parametrize("uniform_inset", (4, 6))
def test_real_menu_owners_keep_uniform_insets_and_constrain_actual_row_paint(
    appearance, monkeypatch, path, radius, asymmetric, uniform_inset
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    authored = dict(runtime.spec["parts"]["menu"], radius=radius)
    authored.update(
        dict.fromkeys(("paddingLeft", "paddingTop", "paddingRight", "paddingBottom"), uniform_inset)
    )
    if asymmetric:
        authored.update(paddingLeft=3, paddingTop=7, paddingRight=11, paddingBottom=5)
    update({"menu": authored, "menu_item": {"radius": 17}})
    specification = deepcopy(runtime.spec)
    menus = Menus(SimpleNamespace(ui=ui, s=ui.s), ())
    rows = tuple(
        Item(label, icon="camera", shortcut="Ctrl+Shift+K")
        for label in ("Complete first command", "Complete middle command", "Complete final command")
    )
    monkeypatch.setattr(menus, "items", lambda _name: rows)
    if path == "named":
        menus.preview(("Offset menu",))

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        menus.begin_frame()
        before = imgui.get_style().popup_rounding
        with ui.corner_scope(1, "menu_item"):
            if path == "named":
                menus.popup("Offset menu", (40 * ui.s, 40 * ui.s, 120 * ui.s, 30 * ui.s))
            else:
                if not imgui.is_popup_open("Offset menu"):
                    imgui.open_popup("Offset menu")
                with menus.local_popup("Offset menu") as opened:
                    if opened:
                        menus.local_rows(rows)
            assert ui.component_style("menu_item").get("radius") * ui.s == pytest.approx(1)
        assert imgui.get_style().popup_rounding == before
        assert ui.component_style("menu_item").get("radius") == 17

    for _ in range(5):
        frame(draw)
    inset = native_unit((11 if asymmetric else uniform_inset) * ui.s)
    inner = max(0, radius * ui.s - inset)
    for hovered in (0, 2):
        row = menus.observed["Offset menu"]["rows"][hovered]
        x, y, w, h = row["frame_rect"]
        imgui.get_io().add_mouse_pos_event(x + w / 2, y + h / 2)
        for _ in range(3):
            frame(draw)
        popup = menus.observed["Offset menu"]
        first, last = popup["rows"][0]["frame_rect"], popup["rows"][-1]["frame_rect"]
        px, py, pw, ph = popup["rect"]
        assert (
            first[0] - px,
            first[1] - py,
            px + pw - last[0] - last[2],
            py + ph - last[1] - last[3],
        ) == (inset,) * 4
        assert popup["radius"] * ui.s == pytest.approx(radius * ui.s)
        row = popup["rows"][hovered]
        assert row["hovered"] and row["backgroundRect"] is not None
        assert row["backgroundRadius"] == pytest.approx(inner)
        assert all(
            not item["clippedLabel"] and not item["clippedShortcut"] for item in popup["rows"]
        )
        record = next(
            item for item in runtime.observed if item.get("backgroundRect") == row["backgroundRect"]
        )
        window = next(
            item for item in imgui.get_current_context().windows if item.id_ == record["windowId"]
        )
        draw_list, ink = window.draw_list, color((255, 255, 255, 13))
        x, y, w, h = row["frame_rect"]
        assert _paint_covers(draw_list, (x + w / 2, y + h / 2), ink)
        for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
            if inner:
                diagonal = inner * (1 - sqrt(0.5))
                assert not _paint_covers(
                    draw_list, _corner_point((x, y, w, h), right, bottom, diagonal - 0.9), ink
                )
                assert _paint_covers(
                    draw_list, _corner_point((x, y, w, h), right, bottom, diagonal + 0.9), ink
                )
            else:
                assert _paint_covers(
                    draw_list, _corner_point((x, y, w, h), right, bottom, 0.75), ink
                )
    assert runtime.spec == specification


@pytest.mark.parametrize("radius", (0, 24))
@pytest.mark.parametrize("asymmetric", (False, True))
@pytest.mark.parametrize("uniform_inset", (4, 6))
def test_menu_reference_uses_the_same_resolved_inset_and_effective_row_curve(
    appearance, radius, asymmetric, uniform_inset
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    authored = dict(runtime.spec["parts"]["menu"], radius=radius)
    authored.update(
        dict.fromkeys(("paddingLeft", "paddingTop", "paddingRight", "paddingBottom"), uniform_inset)
    )
    if asymmetric:
        authored.update(paddingLeft=3, paddingTop=7, paddingRight=11, paddingBottom=5)
    update({"menu": authored, "menu_item": {"radius": 17}})
    specimens = ComponentSpecimens(runtime)

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        width, height = specimens.measure(ui, "menu", 420 * ui.s)
        return specimens.paint(ui, "menu", Rect(40 * ui.s, 40 * ui.s, width, height))

    for _ in range(3):
        record = frame(draw)
    first, last = record["children"][0], record["children"][-1]
    x, y, width, height = record["frame"]
    inset = native_unit((11 if asymmetric else uniform_inset) * ui.s)
    assert (
        first[0] - x,
        first[1] - y,
        x + width - last[0] - last[2],
        y + height - last[1] - last[3],
    ) == (inset,) * 4
    inner = max(0, radius * ui.s - inset)
    assert all(child["backgroundRadius"] == pytest.approx(inner) for child in record["controls"])
    assert ui.component_style("menu_item").get("radius") == 17
