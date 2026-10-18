"""The actual floating owner encloses its rounded content at fractional scales."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.presentation import PANEL_FAMILIES
from mojive.ui.design_palette import rgba8
from mojive.ui.panels import Panel
from mojive.ui.workspace import panel_host
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from mojive.ui.workspace.panel_geometry import floating_panel_geometry
from mojive.ui.workspace.panel_manager import WorkspacePanels
from mojive.ui.workspace.timeline_panel import WorkspaceTimeline
from mojive.ui.workspace.window_chrome import (
    draw_dockspace,
    floating_dock_hosts,
    floating_dock_root,
    panel_surface,
)
from tests.test_surface_hierarchy import surfaces as surfaces

pytestmark = pytest.mark.parametrize("appearance", (1.0, 1.2, 1.25, 2.25), indirect=True)


@pytest.fixture
def appearance(request, tmp_path):
    yield from surfaces.__wrapped__(request, tmp_path)


def _equal(actual, expected, *coordinates):
    assert abs(actual - expected) <= native_epsilon(actual, expected, *coordinates)


def _update(ui):
    def update(parts):
        spec = deepcopy(ui.design.runtime.spec)
        spec["parts"].update(parts)
        ui.design.runtime.update_spec(spec)

    return update


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


def _workspace(ui, *names):
    panel = SimpleNamespace(finish_frame=lambda *_args, **_kwargs: None)
    return SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(panels=SimpleNamespace(get=lambda _name: panel, panels=[])),
        open=dict.fromkeys(names, True),
        new_panel=None,
        left_active="Scene",
        left=0,
        float_next=False,
        focus_next=None,
        bottom_focus=None,
        widths={},
        inspector_grouped_with=lambda _name: False,
    )


def _family(ui, name):
    return ui.design.family_scope(PANEL_FAMILIES.get(name)) if ui.design else nullcontext()


def _frame(workspace, *names, dock_id=None, keep_layout=False):
    with floating_dock_hosts(workspace):
        imgui.new_frame()
        main_dock = []
        workspace.window = SimpleNamespace(
            begin_dockspace=lambda: main_dock.append(imgui.dock_space_over_viewport())
        )
        draw_dockspace(workspace)
        for name in names:
            if keep_layout:
                pass
            elif dock_id is not None:
                imgui.set_next_window_dock_id(dock_id, imgui.Cond_.always)
            else:
                imgui.set_next_window_dock_id(0, imgui.Cond_.always)
                imgui.set_next_window_pos((30 * workspace.s, 30 * workspace.s))
                imgui.set_next_window_size((500 * workspace.s, 450 * workspace.s))
            panel_host.panel(workspace, name, SimpleNamespace(tr=str))
        imgui.render()
    return main_dock[0]


@pytest.mark.parametrize("outer_radius", (None, 0, 20))
def test_actual_floating_timeline_encloses_its_native_body_at_the_same_corner_inset(
    appearance, monkeypatch, outer_radius
):
    ui, runtime = appearance, appearance.design.runtime
    runtime.observe_components = True
    if outer_radius is not None:
        _update(ui)({"panel": {"radius": outer_radius}, "timeline_body": {"radius": 23}})
    workspace = _workspace(ui, "Timeline")
    timeline = WorkspaceTimeline.__new__(WorkspaceTimeline)
    timeline.workspace = workspace

    def draw(_workspace, _name, _ctx, rect):
        imgui.set_cursor_screen_pos((rect.x, rect.y))
        timeline._begin_body(rect.y, rect.height)
        imgui.text_unformatted("Timeline content")
        imgui.end_child()

    monkeypatch.setattr(panel_host, "draw_content", draw)
    for _ in range(3):
        _frame(workspace, "Timeline")
    window = imgui.internal.find_window_by_name("Timeline")
    outer = next(
        record for record in reversed(runtime.observed) if record["key"] == "panel-Timeline"
    )
    body = next(record for record in reversed(runtime.observed) if record["key"] == "timeline-body")
    inset = native_unit(6 * ui.s)
    assert tuple(window.window_padding) == (inset, inset)
    assert outer["nativeWindowPadding"] == [inset, inset]
    _equal(
        window.window_rounding,
        6 * ui.s + native_unit(8 * ui.s) if outer_radius is None else outer_radius * ui.s,
    )
    _equal(body["radius"], max(0, window.window_rounding - inset))
    _equal(outer["nativeWindowRadius"], window.window_rounding)
    _equal(body["hit"][0] - window.pos.x, inset, *body["hit"])
    _equal(window.pos.y + window.size.y - body["hit"][1] - body["hit"][3], inset, *body["hit"])
    assert outer["floatingDockHostId"] is None
    assert outer["nativeTitleBarRect"][3] > 0


@pytest.mark.parametrize("grouped", (False, True))
def test_real_panel_content_child_preserves_the_floating_hosts_bottom_inset(appearance, grouped):
    ui, runtime = appearance, appearance.design.runtime
    imgui.get_io().display_size = (1600 * ui.s, 1200 * ui.s)
    runtime.observe_components = True
    workspace = _workspace(ui, "Timeline", "Output")
    workspace.app.apply_document_edits = lambda: None
    workspace.app.panels.get("Output").draw = lambda _ctx: imgui.text_unformatted("Output")
    for name in (
        "scene_panel",
        "inspector",
        "control_panel",
        "assets_panel",
        "pose_panel",
        "camera_panel",
        "layers_panel",
    ):
        setattr(workspace, name, None)
    timeline = WorkspaceTimeline.__new__(WorkspaceTimeline)
    timeline.workspace = workspace

    def draw(x, y, _width, height):
        imgui.set_cursor_screen_pos((x, y))
        imgui.dummy((0, 77 * ui.s))
        timeline._begin_body(y, height)
        imgui.text_unformatted("Timeline content")
        imgui.end_child()

    workspace.timeline = draw
    dock_id = None
    if grouped:
        imgui.new_frame()
        dock_id = imgui.internal.dock_builder_add_node(0, imgui.DockNodeFlags_.none)
        imgui.internal.dock_builder_set_node_pos(dock_id, (86, 84))
        imgui.internal.dock_builder_set_node_size(dock_id, (1164, 653))
        for name in ("Output", "Timeline"):
            imgui.internal.dock_builder_dock_window(name, dock_id)
        imgui.internal.dock_builder_finish(dock_id)
        imgui.render()
    for _ in range(4):
        workspace.focus_next = "Timeline"
        _frame(workspace, "Timeline", "Output", dock_id=dock_id)
    window = imgui.internal.find_window_by_name("Timeline")
    root = floating_dock_root(window)
    host = root.host_window if root is not None else window
    body = next(row for row in reversed(runtime.observed) if row["key"] == "timeline-body")
    assert body["submittedFrame"] == imgui.get_frame_count()
    scroll = next(
        row for row in reversed(runtime.observed) if row["key"] == "panel-scroll-Timeline"
    )
    bx, by, bw, bh = body["hit"]
    inset = native_unit(6 * ui.s)
    _equal(bx - host.pos.x, inset, *body["hit"])
    _equal(host.pos.x + host.size.x - bx - bw, inset, *body["hit"])
    _equal(host.pos.y + host.size.y - by - bh, inset, *body["hit"])
    _equal(host.window_rounding - body["radius"], inset)
    sx, sy, sw, sh = scroll["windowRect"]
    assert by + bh == sy + sh
    assert bx >= sx and bx + bw <= sx + sw
    assert body["clip"][3] >= by + bh


def test_authored_inner_radius_expands_automatic_window_and_explicit_window_wins(
    appearance, monkeypatch
):
    ui = appearance
    update = _update(ui)
    workspace = _workspace(ui, "Timeline")
    monkeypatch.setattr(panel_host, "draw_content", lambda *_args: imgui.text_unformatted("Body"))
    update({"timeline_body": {"radius": 21}})
    for _ in range(2):
        _frame(workspace, "Timeline")
    window = imgui.internal.find_window_by_name("Timeline")
    _equal(window.window_rounding, 21 * ui.s + native_unit(6 * ui.s))
    update({"panel": {"radius": 9}})
    for _ in range(2):
        _frame(workspace, "Timeline")
    _equal(window.window_rounding, 9 * ui.s)
    assert tuple(window.window_padding) == (native_unit(6 * ui.s),) * 2


def test_surface_scope_restores_native_geometry_even_when_begin_raises(appearance):
    ui = appearance
    style = imgui.get_style()
    before = tuple(style.window_padding), style.window_rounding, style.window_border_size
    with (
        pytest.raises(RuntimeError, match="scope probe"),
        panel_surface(ui, name="Timeline", docked=False),
    ):
        assert tuple(style.window_padding) == (native_unit(6 * ui.s),) * 2
        raise RuntimeError("scope probe")
    assert (tuple(style.window_padding), style.window_rounding, style.window_border_size) == before


def test_main_dockspace_and_its_tiles_remain_square(appearance, monkeypatch):
    ui = appearance
    workspace = _workspace(ui, "Timeline")
    monkeypatch.setattr(panel_host, "draw_content", lambda *_args: imgui.text_unformatted("Body"))
    dock_id = _frame(workspace)
    for _ in range(3):
        _frame(workspace, "Timeline", dock_id=dock_id)
    window = imgui.internal.find_window_by_name("Timeline")
    root = imgui.internal.dock_node_get_root_node(window.dock_node)
    assert root.is_dock_space()
    assert floating_dock_root(window) is None
    assert window.window_rounding == root.host_window.window_rounding == 0


def test_floating_dock_host_uses_shared_member_radius_before_its_border_is_painted(
    appearance, monkeypatch
):
    ui = appearance
    workspace = _workspace(ui, "Timeline", "Output")
    monkeypatch.setattr(panel_host, "draw_content", lambda *_args: imgui.text_unformatted("Body"))
    imgui.new_frame()
    dock_id = imgui.internal.dock_builder_add_node(0, imgui.DockNodeFlags_.none)
    imgui.internal.dock_builder_set_node_pos(dock_id, (30 * ui.s, 30 * ui.s))
    imgui.internal.dock_builder_set_node_size(dock_id, (500 * ui.s, 450 * ui.s))
    for name in workspace.open:
        imgui.internal.dock_builder_dock_window(name, dock_id)
    imgui.internal.dock_builder_finish(dock_id)
    imgui.render()
    for _ in range(4):
        _frame(workspace, *workspace.open, dock_id=dock_id)
    window = imgui.internal.find_window_by_name("Timeline")
    root = floating_dock_root(window)
    assert root is not None and root.host_window is not None
    with group_scope("Timeline"), _family(ui, "Timeline"):
        expected = floating_panel_geometry(ui, "Timeline")
    _equal(root.host_window.window_rounding, expected.radius)
    assert tuple(window.window_padding) == (expected.padding,) * 2
    _equal(window.window_rounding, root.host_window.window_rounding)
    assert root.host_window.draw_list.vtx_buffer.size() > 0
    ink = imgui.color_convert_float4_to_u32(tuple(channel / 255 for channel in rgba8("floating")))
    corner = (root.pos.x + ui.s, root.pos.y + root.size.y - ui.s)
    assert not _paint_covers(root.host_window.draw_list, corner, ink)
    assert not _paint_covers(window.draw_list, corner, ink)


def test_queued_undock_has_the_correct_radius_on_its_first_painted_frame(appearance, monkeypatch):
    ui = appearance
    workspace = _workspace(ui, "Timeline", "Output")
    monkeypatch.setattr(panel_host, "draw_content", lambda *_args: imgui.text_unformatted("Body"))
    dock_id = _frame(workspace)
    for _ in range(3):
        _frame(workspace, *workspace.open, dock_id=dock_id)
    window = imgui.internal.find_window_by_name("Timeline")
    assert window.dock_node.is_dock_space()
    imgui.internal.dock_context_queue_undock_node(imgui.get_current_context(), window.dock_node)
    _frame(workspace, *workspace.open, keep_layout=True)
    root = floating_dock_root(window)
    assert root is not None and root.host_window is not None
    with group_scope("Timeline"), _family(ui, "Timeline"):
        expected = floating_panel_geometry(ui, "Timeline")
    _equal(root.host_window.window_rounding, expected.radius)
    _equal(window.window_rounding, expected.radius)
    assert root.host_window.draw_list.vtx_buffer.size() > 0
    ink = imgui.color_convert_float4_to_u32(tuple(channel / 255 for channel in rgba8("floating")))
    corner = (root.pos.x + ui.s, root.pos.y + root.size.y - ui.s)
    assert not _paint_covers(root.host_window.draw_list, corner, ink)


@pytest.mark.parametrize("radius", (11, 80))
def test_custom_native_panel_uses_the_shared_floating_geometry(appearance, radius):
    ui = appearance
    update = _update(ui)
    manager = WorkspacePanels.__new__(WorkspacePanels)
    manager.workspace = _workspace(ui, "Custom")
    panel = Panel()
    panel.name, panel.standalone = "Custom", True
    manager.workspace.open.clear()
    manager.workspace.app.panels.panels = [panel]
    update({"panel": {"radius": radius}})
    for _ in range(3):
        with floating_dock_hosts(manager.workspace) as shared_radius:
            _equal(shared_radius, radius * ui.s)
            imgui.new_frame()
            manager._begin_panel_window(panel, "Custom", ui.s)
            imgui.text_unformatted("Custom panel")
            imgui.end()
            imgui.render()
    window = imgui.internal.find_window_by_name("Custom")
    _equal(window.window_rounding, radius * ui.s)
    geometry = floating_panel_geometry(ui, "Custom")
    assert tuple(window.window_padding) == (geometry.padding,) * 2
    assert geometry.padding >= native_unit(8 * ui.s)
    assert geometry.inner_radius <= 14 * ui.s - 1 + native_epsilon(geometry.inner_radius, 14 * ui.s)


def test_authored_zero_shared_radius_does_not_trigger_an_automatic_fallback(appearance):
    ui = appearance
    workspace = _workspace(ui, "Timeline")
    _update(ui)({"panel": {"radius": 0}})
    with floating_dock_hosts(workspace) as radius:
        assert radius == 0
