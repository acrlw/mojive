"""Panel rules and disclosure rows stay inside their measured content columns."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import ViewportLayers
from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.ui.workspace.panel_host import draw_content, panel_header
from mojive.ui.workspace.property_style import NativePropertyPresentation
from mojive.ui.workspace.scene_panels import layers_panel
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.1, 1.2, 1.25, 2.25), indirect=True)


def _workspace(ui):
    def noop(*_args):
        return None

    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        widths={},
        app=SimpleNamespace(
            panels=SimpleNamespace(get=lambda _name: SimpleNamespace()),
            viewport_layers=ViewportLayers(),
            gizmos=True,
            apply_document_edits=noop,
        ),
        header_actions={"Cameras": (), "Layers": ()},
        menus=SimpleNamespace(local_popup=lambda _key: nullcontext(False)),
        scene_panel=noop,
        inspector=noop,
        control_panel=noop,
        assets_panel=noop,
        pose_panel=noop,
        camera_panel=noop,
        layers_panel=noop,
        timeline=noop,
    )
    workspace.panel_header = lambda name, x, y, w: panel_header(workspace, name, x, y, w)
    return workspace


def _inside(inner, outer, epsilon=0.001):
    x, y, w, h = inner
    ox, oy, ow, oh = outer
    assert x >= ox - epsilon and y >= oy - epsilon
    assert x + w <= ox + ow + epsilon and y + h <= oy + oh + epsilon


@pytest.mark.parametrize("width", (220, 440))
def test_actual_layers_disclosure_hover_has_real_text_and_chevron_padding(appearance, width):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    workspace = _workspace(ui)
    workspace._panel_context = SimpleNamespace(
        tr=ui.translate,
        backend=SimpleNamespace(debug=None),
        set_viewport_layers=lambda _value: None,
    )
    header = "section-" + ui.translate("Editing aids")

    def draw():
        runtime.begin_frame()
        with group_scope("Layers"):
            layers_panel(workspace, 30 * ui.s, 30 * ui.s, width * ui.s)

    for _ in range(3):
        frame(draw)
    x, y, w, h = ui.hits[header]
    imgui.get_io().add_mouse_pos_event(x + w / 2, y + h / 2)
    for _ in range(3):
        frame(draw)
    row = next(row for row in runtime.observed if row["key"] == header)
    assert row["hovered"] and row["backgroundRect"] == list(row["hit"])
    _inside(row["labelRect"], row["hit"])
    _inside(row["iconRect"], row["hit"])
    assert row["labelRect"][0] - x == pytest.approx(6 * ui.s)
    icon = row["iconRect"]
    assert x + w - icon[0] - icon[2] == pytest.approx(6 * ui.s)
    label = row["labelRect"]
    assert icon[0] - label[0] - label[2] == pytest.approx(row["gap"])
    for section in (row for row in runtime.observed if row["kind"] == "section_header"):
        if section["dividerRect"] is not None:
            divider = section["dividerRect"]
            assert divider[0] >= section["hit"][0]
            assert divider[0] + divider[2] <= section["hit"][0] + section["hit"][2] + 0.001


def test_section_independent_asymmetric_padding_and_tiny_width_remain_contained(appearance):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"layers_section_header": {"paddingLeft": 10, "paddingRight": 14}})

    def draw():
        runtime.begin_frame()
        with group_scope("Layers"):
            ui.section("Independent section", 30 * ui.s, 30 * ui.s, 180 * ui.s, True)
            ui.section("Narrow section", 30 * ui.s, 90 * ui.s, 30 * ui.s, True)

    frame(draw)
    row = next(row for row in runtime.observed if row["key"] == "section-Independent section")
    x, _y, w, _h = row["hit"]
    assert row["labelRect"][0] - x == pytest.approx(10 * ui.s)
    icon = row["iconRect"]
    assert x + w - icon[0] - icon[2] == pytest.approx(14 * ui.s)
    for row in (row for row in runtime.observed if row["kind"] == "section_header"):
        _inside(row["labelRect"], row["hit"])
        _inside(row["iconRect"], row["hit"])


@pytest.mark.parametrize("width", (220, 440))
def test_panel_header_rule_and_scroll_content_share_real_parent_budget(appearance, width):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    workspace = _workspace(ui)
    presentation = NativePropertyPresentation(ui, prefix="camera")
    last = "Final camera bookmarks"

    def camera(_x, _y, _width):
        imgui.set_scroll_y(100000)
        for index in range(10):
            presentation.section(last if index == 9 else "Camera section " + str(index))

    workspace.camera_panel = camera

    def draw():
        runtime.begin_frame()
        draw_content(
            workspace,
            "Cameras",
            SimpleNamespace(),
            Rect(30 * ui.s, 30 * ui.s, width * ui.s, 240 * ui.s),
        )

    for _ in range(8):
        frame(draw)
    rows = {row["key"]: row for row in runtime.observed}
    header = rows["panel-header-Cameras"]
    divider = header["dividerRect"]
    hx, _hy, hw, _hh = header["hit"]
    assert divider[0] >= hx
    assert divider[0] + divider[2] <= hx + hw + 0.001
    for action in header["actionRects"]:
        _inside(action, header["content"])
    scroll = rows["panel-scroll-Cameras"]
    section = rows["section-" + last]
    _inside(section["hit"], scroll["innerRect"])
    assert section["hit"][0] + section["hit"][2] <= scroll["contentRight"] + 0.001
    scrollbar = scroll["scrollbarRect"]
    assert scrollbar is not None
    assert scroll["contentRight"] <= scrollbar[0] + 0.001
    window = next(w for w in imgui.get_current_context().windows if w.id_ == scroll["windowId"])
    assert window.scroll.y > 0
    assert section["hit"][1] + section["hit"][3] <= window.inner_clip_rect.max.y + 0.001


@pytest.mark.parametrize("ancestor_clip", (False, True))
def test_panel_native_width_budget_does_not_follow_ancestor_clipping(appearance, ancestor_clip):
    from math import floor

    from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit

    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    workspace = _workspace(ui)
    measured = []

    def camera(_x, _y, _width):
        window = imgui.internal.get_current_window()
        p, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
        own_right = floor(window.inner_rect.max.x - window.window_border_size / 2)
        epsilon = native_epsilon(p.x, available.x, own_right)
        assert p.x + available.x <= own_right + epsilon
        assert window.work_rect.max.x <= own_right + epsilon
        assert window.parent_work_rect.max.x <= own_right + epsilon
        if ancestor_clip:
            assert own_right > window.inner_clip_rect.max.x
        style = ui.component_style("section_header")
        trailing = native_unit(style.edges("margin", ui.s)[2]) + native_unit(
            style.edges("padding", ui.s, (6, 0, 6, 0))[2]
        )
        measured.append((available.x, own_right - trailing - p.x))
        # Beginning and ending a native table must retain the same parent budget.
        if imgui.begin_table("Width budget", 2):
            imgui.table_next_row()
            imgui.table_next_column()
            assert imgui.get_content_region_avail().x <= available.x
            imgui.end_table()
        assert imgui.get_content_region_avail().x == pytest.approx(available.x, abs=epsilon)
        imgui.dummy((0, 600 * ui.s))

    workspace.camera_panel = camera

    def draw():
        if ancestor_clip:
            imgui.push_clip_rect((0, 0), (180 * ui.s, 700 * ui.s), True)
        draw_content(
            workspace,
            "Cameras",
            SimpleNamespace(),
            Rect(30 * ui.s, 30 * ui.s, 240 * ui.s, 240 * ui.s),
        )
        if ancestor_clip:
            imgui.pop_clip_rect()

    for _ in range(5):
        frame(draw)
    assert all(width == pytest.approx(expected, abs=0.001) for width, expected in measured)
