"""Native windows and popups share surface roles without leaking local styles."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from examples.ui_design.native.guide_workspace import GuideWorkspace
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.native.specimens import ComponentSpecimens
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.design_palette import PALETTE, rgba8
from mojive.ui.workspace import panel_host
from mojive.ui.workspace.panel_host import draw_content
from mojive.ui.workspace.popup_style import menu_popup_style
from mojive.ui.workspace.settings_layout import _draw_page, draw_navigation
from mojive.ui.workspace.widgets import UI
from mojive.ui.workspace.window_chrome import panel_surface, surface_ink


@pytest.fixture(params=(1.0, 2.0))
def surfaces(request, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 700 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.docking_enable
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)))
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json")))
    try:
        yield ui
    finally:
        imgui.destroy_context(context)


def _color(role):
    return tuple(imgui.get_style_color_vec4(role))


def _packed(color):
    return imgui.color_convert_float4_to_u32(tuple(channel / 255 for channel in color))


def _colors(window):
    return {vertex.col for vertex in window.draw_list.vtx_buffer}


def _window_surface(ui, *, docked, family=None):
    painted = set()
    for frame in range(3):
        imgui.new_frame()
        if docked:
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((700 * ui.s, 580 * ui.s))
            imgui.begin("SurfaceHost", flags=imgui.WindowFlags_.no_docking)
            dock_id = imgui.get_id("SurfaceDock")
            if frame == 0:
                imgui.internal.dock_builder_add_node(dock_id, imgui.DockNodeFlags_.none)
                imgui.internal.dock_builder_set_node_size(dock_id, (700 * ui.s, 580 * ui.s))
                imgui.internal.dock_builder_dock_window("SurfacePanel", dock_id)
                imgui.internal.dock_builder_finish(dock_id)
            imgui.dock_space(dock_id)
            imgui.end()
        else:
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            imgui.set_next_window_pos((20, 20))
            imgui.set_next_window_size((500 * ui.s, 400 * ui.s))
        before = (
            _color(imgui.Col_.window_bg),
            _color(imgui.Col_.border),
            imgui.get_style().window_border_size,
        )
        with ui.design.family_scope(family), panel_surface(ui, docked=docked) as ink:
            assert _color(imgui.Col_.window_bg) == pytest.approx(tuple(v / 255 for v in ink))
            assert imgui.get_style().window_border_size == (0 if docked else ui.s)
            imgui.begin("SurfacePanel", flags=imgui.WindowFlags_.no_title_bar)
        assert (
            _color(imgui.Col_.window_bg),
            _color(imgui.Col_.border),
            imgui.get_style().window_border_size,
        ) == before
        window = imgui.internal.get_current_window()
        assert bool(window.dock_id) is docked
        imgui.dummy((20, 20))
        imgui.end()
        imgui.render()
        painted = _colors(window)
        if window.dock_node is not None:
            painted |= _colors(window.dock_node.host_window)
    return ink, painted


def _popup_surface(ui, part, family=None):
    for _ in range(3):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((800 * ui.s, 600 * ui.s))
        imgui.begin("PopupOwner")
        imgui.open_popup("SurfacePopup")
        with ui.design.family_scope(family), menu_popup_style(ui, part=part):
            ink = surface_ink(ui, part, "popup")
            imgui.set_next_window_size((200 * ui.s, 100 * ui.s))
            assert imgui.begin_popup("SurfacePopup")
            window = imgui.internal.get_current_window()
            imgui.dummy((20, 20))
            imgui.end_popup()
        imgui.end()
        imgui.render()
    return ink, _colors(window)


def test_actual_docked_floating_and_menu_surfaces_have_ordered_gray_levels(surfaces):
    ui = surfaces
    docked, docked_inks = _window_surface(ui, docked=True)
    floating, floating_inks = _window_surface(ui, docked=False)
    menu, menu_inks = _popup_surface(ui, "menu")
    assert (docked, floating, menu) == tuple(
        rgba8(role) for role in ("background", "floating", "popup")
    )
    assert all(a < b < c for a, b, c in zip(docked[:3], floating[:3], menu[:3], strict=True))
    assert _packed(docked) in docked_inks
    assert _packed(floating) in floating_inks
    assert _packed(menu) in menu_inks


def _native_panel_workspace(ui, name):
    panel = SimpleNamespace(finish_frame=lambda *args, **kwargs: None)
    return SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(panels=SimpleNamespace(get=lambda _name: panel)),
        open={name: True},
        new_panel=None,
        left_active="Scene",
        left=0,
        float_next=False,
        focus_next=None,
        bottom_focus=None,
        widths={},
        inspector_grouped_with=lambda _name: False,
    )


def _dock_frame(ui, workspace):
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((700 * ui.s, 580 * ui.s))
    imgui.begin("Dock transition host", flags=imgui.WindowFlags_.no_docking)
    dock_id = imgui.get_id("Dock transition node")
    if not workspace.left:
        imgui.internal.dock_builder_add_node(dock_id, imgui.DockNodeFlags_.none)
        imgui.internal.dock_builder_set_node_size(dock_id, (700 * ui.s, 580 * ui.s))
        imgui.internal.dock_builder_finish(dock_id)
        workspace.left = dock_id
    imgui.dock_space(dock_id)
    imgui.end()
    imgui.set_next_window_size((400 * ui.s, 300 * ui.s))


def _assert_native_surface(window, ui, docked, *, verify_vertices=True):
    assert bool(window.dock_id) is docked
    assert window.window_border_size == (0 if docked else ui.s)
    painted = _colors(window)
    if window.dock_node is not None:
        painted |= _colors(window.dock_node.host_window)
    if verify_vertices:
        assert _packed(rgba8("background" if docked else "floating")) in painted


def test_inspector_undocking_uses_floating_surface_in_the_transition_frame(surfaces, monkeypatch):
    ui = surfaces
    workspace = _native_panel_workspace(ui, "Inspector")
    monkeypatch.setattr(panel_host, "draw_content", lambda *args: imgui.dummy((20, 20)))
    _dock_frame(ui, workspace)
    imgui.set_next_window_dock_id(workspace.left, imgui.Cond_.always)
    imgui.begin("Inspector")
    imgui.end()
    imgui.render()
    assert imgui.internal.find_window_by_name("Inspector").dock_id == workspace.left

    workspace.float_next = True
    _dock_frame(ui, workspace)
    panel_host.panel(workspace, "Inspector", SimpleNamespace(tr=str))
    imgui.render()
    _assert_native_surface(imgui.internal.find_window_by_name("Inspector"), ui, False)


@pytest.mark.parametrize("previous", ("new", "saved-floating", "used-floating", "unused-floating"))
def test_first_use_docking_surface_matches_the_condition_imgui_actually_accepts(
    surfaces, monkeypatch, previous
):
    ui = surfaces
    workspace = _native_panel_workspace(ui, "Pose")
    inks = []

    @contextmanager
    def record_surface(ui, *, docked, **options):
        with panel_surface(ui, docked=docked, **options) as ink:
            inks.append(ink)
            assert _color(imgui.Col_.window_bg) == pytest.approx(tuple(v / 255 for v in ink))
            yield ink

    monkeypatch.setattr(panel_host, "panel_surface", record_surface)
    monkeypatch.setattr(panel_host, "draw_content", lambda *args: imgui.dummy((20, 20)))
    if previous == "saved-floating":
        imgui.load_ini_settings_from_memory(
            "[Window][Pose]\nPos=40,40\nSize=400,300\nCollapsed=0\n"
        )
    elif previous in ("used-floating", "unused-floating"):
        _dock_frame(ui, workspace)
        if previous == "used-floating":
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
        imgui.begin("Pose")
        imgui.end()
        imgui.render()
    workspace.new_panel = "Pose"
    _dock_frame(ui, workspace)
    panel_host.panel(workspace, "Pose", SimpleNamespace(tr=str))
    imgui.render()
    docked = previous in ("new", "unused-floating")
    assert inks == [rgba8("background" if docked else "floating")]
    # Docking may hide a newly attached window for this frame. The Begin scope
    # above still checks its selected ink, and its native border is recorded.
    _assert_native_surface(
        imgui.internal.find_window_by_name("Pose"), ui, docked, verify_vertices=not docked
    )


@pytest.mark.parametrize(
    "part,role", (("panel", "floating"), ("menu", "popup"), ("select_popup", "popup"))
)
def test_component_surface_literal_outranks_module_and_global_palette(surfaces, part, role):
    ui, family = surfaces, "navigation"
    runtime = ui.design.runtime
    spec = deepcopy(runtime.spec)
    spec["tokens"].update(panel_color="#101214", floating_color="#212325", popup_color="#313335")
    spec["components"].setdefault(family, {}).update(
        floating_color="#456789", popup_color="#56789A"
    )
    runtime.update_spec(spec)
    global_ink, global_painted = (
        _window_surface(ui, docked=False) if part == "panel" else _popup_surface(ui, part)
    )
    global_expected = (33, 35, 37, 255) if role == "floating" else (49, 51, 53, 255)
    assert global_ink == global_expected and _packed(global_ink) in global_painted
    render = (
        (lambda: _window_surface(ui, docked=False, family=family))
        if part == "panel"
        else (lambda: _popup_surface(ui, part, family))
    )
    expected = (69, 103, 137, 255) if role == "floating" else (86, 120, 154, 255)
    module, ink = render()
    assert module == expected and _packed(expected) in ink
    # Matching a built-in role color is still an explicit component choice.
    spec["parts"][part] = {"surfaceColor": PALETTE[role]}
    runtime.update_spec(spec)
    component, ink = render()
    assert component == rgba8(role) and _packed(component) in ink
    other = "menu" if part != "menu" else "select_popup"
    with ui.design.family_scope(family):
        assert surface_ink(ui, other, "popup") == (86, 120, 154, 255)


@pytest.mark.parametrize(
    "part,role", (("panel", "background"), ("menu", "popup"), ("select_popup", "popup"))
)
@pytest.mark.parametrize("g3", (False, True))
def test_workbench_preview_paints_explicit_role_color_without_remapping(surfaces, part, role, g3):
    ui = surfaces
    runtime = ui.design.runtime
    spec = deepcopy(runtime.spec)
    spec["components"].setdefault("navigation", {}).update(
        panel_color="#A74589", popup_color="#A74589"
    )
    spec["parts"][part] = {"surfaceColor": PALETTE[role]}
    spec["native"]["g3"] = g3
    runtime.update_spec(spec)
    specimens = ComponentSpecimens(runtime)
    bounds = Rect(40 * ui.s, 40 * ui.s, 450 * ui.s, 280 * ui.s)
    painted = []
    for _ in range(2):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((700 * ui.s, 600 * ui.s))
        imgui.begin("Workbench preview", flags=imgui.WindowFlags_.no_title_bar)
        draw = imgui.get_window_draw_list()
        start = len(draw.idx_buffer)
        with ui.design.family_scope("navigation"):
            # Prove this exact literal would otherwise be remapped by UI.rect.
            assert ui.ink(rgba8(role)) == (167, 69, 137, 255)
            result = specimens.paint(ui, part, bounds)
        frame = result["frame"]
        point = (frame[0] + frame[2] / 2, frame[1] + 2 * ui.s)
        painted = _triangle_colors_at(draw, start, point)
        imgui.end()
        imgui.render()
    # Sample inside the preview's top padding, away from all child controls.
    assert painted and all(color == _packed(rgba8(role)) for color in painted)


def _triangle_colors_at(draw, start, point):
    """Return vertex colors for actual draw triangles covering an interior point."""
    indices, vertices = draw.idx_buffer, draw.vtx_buffer
    colors = []
    for offset in range(start, len(indices), 3):
        triangle = [vertices[indices[offset + index]] for index in range(3)]
        signs = []
        for a, b in zip(triangle, triangle[1:] + triangle[:1], strict=True):
            signs.append(
                (b.pos.x - a.pos.x) * (point[1] - a.pos.y)
                - (b.pos.y - a.pos.y) * (point[0] - a.pos.x)
            )
        if (min(signs) >= 0 or max(signs) <= 0) and any(abs(sign) > 1e-6 for sign in signs):
            colors.extend(vertex.col for vertex in triangle)
    return colors


def test_panel_surface_override_does_not_leak_into_its_child_popups(surfaces):
    ui = surfaces
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"]["panel"] = {"surfaceColor": "#718293"}
    ui.design.runtime.update_spec(spec)
    assert surface_ink(ui, "panel", "floating") == (113, 130, 147, 255)
    with ui.component_scope("panel"):
        for part in ("menu", "select_popup"):
            ink, painted = _popup_surface(ui, part)
            assert ink == rgba8("popup") and _packed(ink) in painted


def test_nested_surface_styles_restore_colors_dimensions_and_font_after_failure(surfaces):
    ui = surfaces
    imgui.new_frame()
    imgui.begin("Scope owner")
    colors = tuple(
        _color(role)
        for role in (
            imgui.Col_.window_bg,
            imgui.Col_.border,
            imgui.Col_.popup_bg,
            imgui.Col_.header,
        )
    )
    style = imgui.get_style()
    dimensions = (
        style.window_border_size,
        style.popup_rounding,
        tuple(style.window_padding),
        tuple(style.frame_padding),
    )
    font = imgui.get_font().font_id, imgui.get_font_size()
    with (
        pytest.raises(RuntimeError, match="interrupt"),
        panel_surface(ui, docked=False),
        menu_popup_style(ui, part="select_popup"),
    ):
        raise RuntimeError("interrupt")
    assert (
        tuple(
            _color(role)
            for role in (
                imgui.Col_.window_bg,
                imgui.Col_.border,
                imgui.Col_.popup_bg,
                imgui.Col_.header,
            )
        )
        == colors
    )
    assert (
        style.window_border_size,
        style.popup_rounding,
        tuple(style.window_padding),
        tuple(style.frame_padding),
    ) == dimensions
    assert (imgui.get_font().font_id, imgui.get_font_size()) == font
    imgui.end()
    imgui.render()


def test_production_panel_content_does_not_paint_over_its_host_surface(surfaces):
    ui = surfaces
    children = []
    panel = SimpleNamespace(draw=lambda ctx: children.append(imgui.internal.get_current_window()))
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        widths={},
        app=SimpleNamespace(
            panels=SimpleNamespace(get=lambda name: panel), apply_document_edits=lambda: None
        ),
    )
    for name in (
        "scene_panel",
        "inspector",
        "control_panel",
        "assets_panel",
        "pose_panel",
        "camera_panel",
        "layers_panel",
        "timeline",
    ):
        setattr(workspace, name, None)
    imgui.new_frame()
    imgui.set_next_window_size((500 * ui.s, 400 * ui.s))
    with panel_surface(ui, docked=False):
        imgui.begin("Floating Output", flags=imgui.WindowFlags_.no_title_bar)
    host = imgui.internal.get_current_window()
    position = imgui.get_cursor_screen_pos()
    draw_content(
        workspace, "Output", SimpleNamespace(), Rect(position.x, position.y, 400 * ui.s, 300 * ui.s)
    )
    imgui.end()
    imgui.render()
    assert children and children[0].flags & imgui.WindowFlags_.no_background
    assert not children[0].draw_list.vtx_buffer
    assert _packed(rgba8("floating")) in _colors(host)


def test_settings_navigation_and_page_children_preserve_parent_background(surfaces):
    ui = surfaces
    workspace = SimpleNamespace(ui=ui, s=ui.s)
    panel = SimpleNamespace(workspace=workspace, _category="General", _control_padding=lambda: None)
    for method in (
        "_general",
        "_camera_navigation",
        "_interaction",
        "_rendering",
        "_recording",
        "_mujoco_visuals",
    ):
        setattr(panel, method, lambda ctx: None)
    ctx = SimpleNamespace(tr=str)
    imgui.new_frame()
    imgui.set_next_window_size((850 * ui.s, 600 * ui.s))
    with panel_surface(ui, docked=False):
        imgui.begin("Settings surface", flags=imgui.WindowFlags_.no_title_bar)
    p = imgui.get_cursor_screen_pos()
    draw_navigation(panel, ctx, Rect(p.x, p.y, 800 * ui.s, 500 * ui.s), False)
    _draw_page(panel, ctx, Rect(p.x + 240 * ui.s, p.y, 540 * ui.s, 500 * ui.s), ())
    imgui.end()
    imgui.render()
    children = [
        window
        for window in imgui.get_current_context().windows
        if window.parent_window is not None and window.parent_window.name == "Settings surface"
    ]
    assert len(children) == 2
    assert all(window.flags & imgui.WindowFlags_.no_background for window in children)


@pytest.mark.parametrize(
    "name,family,role",
    (("Scene", "navigation", "background"), ("Settings", "settings", "floating")),
)
def test_guide_specimen_surface_matches_the_native_panel_palette(surfaces, name, family, role):
    ui = surfaces
    spec = deepcopy(ui.design.runtime.spec)
    spec["components"].setdefault(family, {}).update(
        panel_color="#334455", floating_color="#667788"
    )
    ui.design.runtime.update_spec(spec)
    specimen = GuideWorkspace.__new__(GuideWorkspace)
    specimen.guide = SimpleNamespace(ui=ui, s=ui.s, rect=lambda x, y, w, h: Rect(x, y, w, h))
    specimen.app = SimpleNamespace(
        workspace=SimpleNamespace(ui=ui, open={}),
        _draw_rename_popup=lambda: None,
        model_edits=SimpleNamespace(active=False),
    )
    for _ in range(3):
        imgui.new_frame()
        imgui.set_next_window_size((700 * ui.s, 600 * ui.s))
        imgui.begin("Guide host")
        host = imgui.internal.get_current_window()
        p = imgui.get_cursor_screen_pos()
        with specimen.panel_region(name, p.x, p.y, 600 * ui.s, 500 * ui.s):
            child = imgui.internal.get_current_window()
            assert ui.design.family() == family
            expected = surface_ink(ui, "panel", role)
            imgui.dummy((20, 20))
        imgui.end()
        imgui.render()
    assert _packed(expected) in _colors(child) | _colors(host)
    native, colors = _window_surface(ui, docked=role == "background", family=family)
    assert native == expected and _packed(expected) in colors
