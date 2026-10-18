"""Actual navigation slots retain bottom clearance and coherent corner geometry."""

from contextlib import nullcontext
from copy import deepcopy
from math import ceil
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import NativePartStyle, Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.native.specimens import ComponentSpecimens
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace import chrome
from mojive.ui.workspace.constants import PANELS
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 1.2, 1.25, 2.25))
def rail(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 760 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    state = SimpleNamespace(ui=ui, io=io, runtime=runtime, bottom=False, paints=[])
    visibility = SimpleNamespace(
        panel_scope=lambda *args, **kwargs: nullcontext(), bind_current_window=lambda: None
    )
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(scrollbar_visibility=visibility),
        left_active="Scene",
        bottom_active="Settings",
        open=dict.fromkeys(PANELS, True),
        toggle_activity=lambda _name: None,
    )

    def sidebar(*args):
        visible = chrome.sidebar(workspace, *args)
        if state.bottom:
            imgui.set_scroll_y(imgui.get_scroll_max_y())
        return visible

    workspace.sidebar = sidebar
    workspace.end_sidebar = lambda: chrome.end_sidebar(workspace)
    state.workspace = workspace
    original_rect = ui.rect

    def rectangle(x, y, width, height, *args, **kwargs):
        state.paints.append((x, y, width, height, kwargs.get("r")))
        return original_rect(x, y, width, height, *args, **kwargs)

    monkeypatch.setattr(ui, "rect", rectangle)

    def frame(count=1):
        for _ in range(count):
            state.paints.clear()
            ui.hits.clear()
            runtime.begin_frame()
            ui.design.begin_frame()
            imgui.new_frame()
            chrome._activity_rail(workspace)
            imgui.render()
        state.window = imgui.internal.find_window_by_name("Navigation")
        state.record = next(
            record for record in runtime.observed if record["key"] == "activity-rail"
        )

    state.frame = frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


def _assert_settings(rail, *, bottom_gap, radius):
    scale, record = rail.ui.s, rail.record
    x, y, width, height = rail.ui.hits["rail-Settings"]
    outer = record["outer"]
    assert outer[1] + outer[3] - y - height == pytest.approx(bottom_gap * scale, abs=1)
    assert record["bottomInset"] == pytest.approx(bottom_gap * scale)
    assert record["itemRadius"] == pytest.approx(radius * scale)
    assert any(
        paint[:4] == pytest.approx((x, y, width, height)) and paint[4] == pytest.approx(radius)
        for paint in rail.paints
    )


def test_tall_rail_settings_clearance_and_actual_painted_corner_follow_resolved_inset(rail):
    rail.frame(4)
    assert not rail.record["compact"]
    assert rail.record["outerRadius"] == pytest.approx(16)
    assert rail.record["cornerInset"] == pytest.approx(6 * rail.ui.s)
    _assert_settings(rail, bottom_gap=6, radius=max(0, 16 / rail.ui.s - 6))
    assert rail.record["bottomInset"] == pytest.approx(rail.record["cornerInset"])


def test_rail_native_endpoint_uses_the_actual_fractional_scale_titlebar_size(rail):
    rail.io.display_size = (1280, 800)
    rail.bottom = True
    for _ in range(6):
        rail.runtime.begin_frame()
        rail.ui.design.begin_frame()
        imgui.new_frame()
        chrome.sidebar(rail.workspace, "Application", imgui.Dir.up, 46)
        chrome.end_sidebar(rail.workspace)
        chrome._activity_rail(rail.workspace)
        imgui.render()
    navigation = imgui.internal.find_window_by_name("Navigation")
    application = imgui.internal.find_window_by_name("Application")
    viewport = imgui.get_main_viewport()
    assert application.pos.y + application.size.y == navigation.pos.y
    assert navigation.pos.y + navigation.size.y == viewport.pos.y + viewport.size.y
    record = next(r for r in rail.runtime.observed if r["key"] == "activity-rail")
    assert record["outer"][1] + record["outer"][3] == viewport.pos.y + viewport.size.y
    _x, y, _width, height = rail.ui.hits["rail-Settings"]
    assert viewport.pos.y + viewport.size.y - y - height == pytest.approx(6 * rail.ui.s, abs=1)


def test_full_chrome_reservation_keeps_the_rail_at_the_actual_viewport_endpoint(rail):
    from mojive.ui.workspace.menu_geometry import native_epsilon

    rail.io.display_size = (1280, 800)
    workspace = rail.workspace
    workspace.rects = {}
    workspace.activate = lambda _name: None
    workspace.open["Inspector"] = True
    workspace.app.session = SimpleNamespace(asset_path=None)
    workspace.menus = Menus(workspace, ())
    rail.bottom = True
    for _ in range(6):
        rail.runtime.begin_frame()
        rail.ui.design.begin_frame()
        imgui.new_frame()
        chrome.chrome(workspace)
        # The production status row is reserved after the left rail so the
        # latter reaches the application's actual bottom corner.
        chrome.sidebar(
            workspace,
            "Status###application_status",
            imgui.Dir.down,
            ceil(28 * rail.ui.s) / rail.ui.s,
        )
        chrome.end_sidebar(workspace)
        imgui.render()
    application = imgui.internal.find_window_by_name("Application")
    navigation = imgui.internal.find_window_by_name("Navigation")
    viewport = imgui.get_main_viewport()
    assert application.pos.y + application.size.y == navigation.pos.y
    assert navigation.pos.y + navigation.size.y == viewport.pos.y + viewport.size.y
    record = next(r for r in rail.runtime.observed if r["key"] == "activity-rail")
    if not record["compact"]:
        x, y, _width, height = rail.ui.hits["rail-Settings"]
        bottom = viewport.pos.y + viewport.size.y - y - height
        left = x - viewport.pos.x
        assert bottom == pytest.approx(
            left, abs=native_epsilon(bottom, left, y, height, *viewport.size)
        )


@pytest.mark.parametrize("gap", (3, 17))
@pytest.mark.parametrize("margin", (0, 5))
def test_compact_rail_scroll_end_keeps_bottom_padding_independent_of_row_gap(rail, gap, margin):
    spec = deepcopy(rail.runtime.spec)
    spec["parts"].setdefault("rail", {}).update(rowGap=gap, paddingBottom=12, marginBottom=margin)
    rail.runtime.update_spec(spec)
    rail.io.display_size = (900 * rail.ui.s, 230 * rail.ui.s)
    rail.frame(4)
    assert rail.record["compact"] and rail.window.scroll_max.y > 0
    rail.bottom = True
    rail.frame(5)
    _assert_settings(rail, bottom_gap=12 + margin, radius=max(0, 16 / rail.ui.s - 6))


def test_parent_corner_budget_constrains_generic_button_radius_locally(rail):
    spec = deepcopy(rail.runtime.spec)
    spec["parts"].setdefault("rail", {}).update(
        width=58, paddingLeft=9, paddingRight=13, paddingBottom=19, marginLeft=2, radius=21
    )
    rail.runtime.update_spec(spec)
    rail.frame(4)
    assert rail.record["cornerInset"] == pytest.approx(11 * rail.ui.s)
    _assert_settings(rail, bottom_gap=19, radius=max(0, 21 / rail.ui.s - 11))
    spec["parts"].setdefault("icon_button", {})["radius"] = 7
    rail.runtime.update_spec(spec)
    rail.frame(3)
    _assert_settings(rail, bottom_gap=19, radius=max(0, 21 / rail.ui.s - 11))


@pytest.mark.parametrize("outer_radius", (18, 24, 32, 4))
def test_catalogue_rail_uses_production_slots_and_the_same_corner_relation(rail, outer_radius):
    ui, runtime = rail.ui, rail.runtime
    spec = deepcopy(runtime.spec)
    spec["parts"].setdefault("rail", {})["radius"] = outer_radius
    runtime.update_spec(spec)
    specimen = ComponentSpecimens(runtime)
    for _ in range(3):
        rail.paints.clear()
        runtime.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(rail.io.display_size)
        imgui.begin("Rail specimen", flags=imgui.WindowFlags_.no_title_bar)
        width, height = specimen.measure(ui, "rail", 100 * ui.s)
        result = specimen.paint(ui, "rail", Rect(30 * ui.s, 40 * ui.s, width, height))
        imgui.end()
        imgui.render()
    assert len(result["children"]) == 5
    for control in result["controls"]:
        assert "icon_button" in control["parts"]
        child_radius = max(0, outer_radius - 6 * ui.s)
        slot = max(36 * ui.s, ceil(2 * child_radius + 2))
        assert control["hit"][2:] == pytest.approx((slot,) * 2)
        assert any(
            paint[:4] == pytest.approx(control["hit"])
            and paint[4] == pytest.approx(child_radius / ui.s)
            for paint in rail.paints
        )


@pytest.mark.parametrize("g3", (False, True))
@pytest.mark.parametrize("bottom", (6, 12))
def test_large_authored_rail_corner_grows_real_slots_without_native_arc_clamping(
    rail, monkeypatch, g3, bottom
):
    from mojive.geometry2d import curves

    spec = deepcopy(rail.runtime.spec)
    spec["native"]["g3"] = g3
    spec["parts"].setdefault("rail", {}).update(radius=32, paddingBottom=bottom)
    rail.runtime.update_spec(spec)
    capsules = []
    capsule = curves.smooth_capsule_points

    def capture(*args):
        capsules.append(args)
        return capsule(*args)

    monkeypatch.setattr(curves, "smooth_capsule_points", capture)
    rail.frame(4)
    _assert_settings(rail, bottom_gap=bottom, radius=max(0, 32 / rail.ui.s - 6))
    x, _y, width, height = rail.ui.hits["rail-Settings"]
    assert min(width, height) >= 2 * rail.record["itemRadius"] + 2
    assert x - rail.record["outer"][0] == pytest.approx(6 * rail.ui.s)
    assert rail.record["outer"][2] >= width + 12 * rail.ui.s
    assert not capsules


@pytest.mark.parametrize(
    "part", ("toolbox", "viewport_toolbar", "transport", "pending_edits", "file_receipt")
)
@pytest.mark.parametrize("authored", (False, True))
def test_painted_group_references_keep_children_inside_their_resolved_padding(rail, part, authored):
    ui, runtime = rail.ui, rail.runtime
    if authored:
        spec = deepcopy(runtime.spec)
        spec["parts"].setdefault(part, {}).update(
            paddingLeft=7, paddingRight=9, paddingTop=5, paddingBottom=11
        )
        runtime.update_spec(spec)
    specimen = ComponentSpecimens(runtime)
    for _ in range(3):
        rail.paints.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(rail.io.display_size)
        imgui.begin("Padded group specimen", flags=imgui.WindowFlags_.no_title_bar)
        width, height = specimen.measure(ui, part, 800 * ui.s)
        result = specimen.paint(ui, part, Rect(30 * ui.s, 40 * ui.s, width, height))
        imgui.end()
        imgui.render()
    x, y, w, h = result["frame"]
    pads = (max(native_unit(v) for v in NativePartStyle(runtime, part).edges("padding", ui.s)),) * 4
    children = result["children"]
    assert min(r[0] for r in children) - x == pytest.approx(pads[0])
    assert min(r[1] for r in children) - y == pytest.approx(pads[1])
    assert x + w - max(r[0] + r[2] for r in children) == pytest.approx(pads[2])
    assert y + h - max(r[1] + r[3] for r in children) == pytest.approx(pads[3])
    assert rail.paints[0][:4] == pytest.approx(result["frame"])


@pytest.mark.parametrize(
    "part", ("title_actions", "output_toolbar", "dialog_footer", "tabs", "timeline_details")
)
def test_transparent_action_group_references_do_not_invent_a_rounded_enclosing_plate(rail, part):
    ui, runtime = rail.ui, rail.runtime
    specimen = ComponentSpecimens(runtime)
    for _ in range(3):
        rail.paints.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(rail.io.display_size)
        imgui.begin("Transparent group specimen", flags=imgui.WindowFlags_.no_title_bar)
        width, height = specimen.measure(ui, part, 800 * ui.s)
        result = specimen.paint(ui, part, Rect(30 * ui.s, 40 * ui.s, width, height))
        imgui.end()
        imgui.render()
    assert not any(paint[:4] == pytest.approx(result["frame"]) for paint in rail.paints)
