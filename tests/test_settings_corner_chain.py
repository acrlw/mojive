"""The actual first Settings row and navigation bottom share nested native corners."""

from math import sqrt
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace import panel_host, settings_layout
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.widgets import color
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


def _paint_covers(draw, point, ink):
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


def _assert_native_corners(draw, paint):
    bounds, radius, ink, _window_id = paint
    x, y, width, height = bounds
    diagonal = radius * (1 - sqrt(0.5))
    for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
        for offset, covered in ((diagonal - 0.9, False), (diagonal + 0.9, True)):
            point = (
                x + width - offset if right else x + offset,
                y + height - offset if bottom else y + offset,
            )
            assert _paint_covers(draw, point, ink) is covered


def test_floating_settings_first_active_row_and_bottom_corner_have_real_clearance(
    appearance, monkeypatch
):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        open={"Settings": True},
        widths={},
        focus_next=None,
        bottom_focus=None,
        app=SimpleNamespace(apply_document_edits=lambda: None),
        scene_panel=None,
        inspector=None,
        control_panel=None,
        assets_panel=None,
        pose_panel=None,
        camera_panel=None,
        layers_panel=None,
        timeline=None,
    )
    workspace.settings = WorkspaceSettings(workspace)
    workspace.app.panels = SimpleNamespace(
        get=lambda _name: SimpleNamespace(finish_frame=lambda _ctx: None)
    )
    ctx = ui.value_context
    monkeypatch.setattr(settings_layout, "_draw_page", lambda *_args: None)
    painted = []
    rectangle = ui.rect

    def rect(x, y, width, height, tint, r=None, **options):
        resolved = tint if options.get("color_resolved") else ui.ink(tint)
        painted.append(
            (
                (x, y, width, height),
                (r or 0) * ui.s,
                color(resolved),
                imgui.internal.get_current_window().id_,
            )
        )
        return rectangle(x, y, width, height, tint, r, **options)

    monkeypatch.setattr(ui, "rect", rect)

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        painted.clear()
        imgui.set_next_window_pos((40 * ui.s, 40 * ui.s))
        imgui.set_next_window_size((820 * ui.s, 620 * ui.s))
        panel_host.panel(workspace, "Settings", ctx)
        return imgui.internal.find_window_by_name("Settings")

    for _ in range(4):
        host = frame(draw)
    navigation = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    frame_rect = navigation["hit"]
    active = ui.hits["settings-General"]
    host_inset, row_inset = native_unit(6 * ui.s), native_unit(4 * ui.s)
    assert tuple(host.window_padding) == (host_inset,) * 2
    epsilon = native_epsilon(*host.pos, *host.size, *frame_rect)
    assert frame_rect[0] - host.pos.x == pytest.approx(host_inset, abs=epsilon)
    assert host.pos.y + host.size.y - frame_rect[1] - frame_rect[3] == pytest.approx(
        host_inset, abs=epsilon
    )
    assert navigation["radius"] == pytest.approx(host.window_rounding - host_inset)
    assert navigation["padding"] == [row_inset] * 4
    assert active[0] - frame_rect[0] == row_inset
    assert active[1] - frame_rect[1] == row_inset
    assert navigation["innerRadius"] == pytest.approx(navigation["radius"] - row_inset)
    assert navigation["innerRadius"] >= 3.5 * ui.s
    nav_paint = next(paint for paint in painted if paint[0] == pytest.approx(frame_rect))
    row_paint = next(paint for paint in painted if paint[0] == pytest.approx(active))
    assert nav_paint[1] == pytest.approx(navigation["radius"])
    assert row_paint[1] == pytest.approx(navigation["innerRadius"])
    child = next(
        window for window in imgui.get_current_context().windows if window.id_ == row_paint[3]
    )
    panel_child = next(
        window for window in imgui.get_current_context().windows if window.id_ == nav_paint[3]
    )
    _assert_native_corners(panel_child.draw_list, nav_paint)
    _assert_native_corners(child.draw_list, row_paint)
