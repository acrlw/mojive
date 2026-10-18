"""Native menu overflow preserves popup ownership, capture paths, and font alignment."""

from types import SimpleNamespace

import pytest
from examples.ui_design.native.reference_menus import TOP_MENUS, Menus
from imgui_bundle import imgui

from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.25))
def menu_bar(request, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    scale = request.param
    ui = UI(SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)))
    session = SimpleNamespace(
        can_undo=False,
        can_redo=False,
        state_take_recording=False,
        state_take_times=[],
        adapter=SimpleNamespace(caps=SimpleNamespace(simulation=False, topology_editing=False)),
    )
    study = SimpleNamespace(
        ui=ui,
        s=scale,
        preview=SimpleNamespace(
            entity=None,
            tool="select",
            gizmos=True,
            camera=SimpleNamespace(orthographic=False),
            session=session,
        ),
        output_directory=tmp_path,
        space="World",
        playing=False,
        speed=1.0,
        loop=False,
        saved=False,
        show_viewport_tab=True,
        workspace="Edit",
        open={"Scene": True, "Inspector": True, "Timeline": False},
    )
    menus = Menus(study, [("Scene", "scene")])
    state = SimpleNamespace(menus=menus, ui=ui, io=io, width=400, right_edge=None, padding_y=7)

    def frame(count=1):
        for _ in range(count):
            io.display_size = (state.width * scale, 700 * scale)
            imgui.new_frame()
            menus.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((state.width * scale, 36 * scale))
            imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
            imgui.push_style_var(
                imgui.StyleVar_.frame_padding, (8 * scale, state.padding_y * scale)
            )
            imgui.begin(
                "Application", flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.menu_bar
            )
            menus.bar(state.right_edge)
            state.draw_list = imgui.get_window_draw_list()
            imgui.end()
            imgui.pop_style_var(2)
            imgui.render()

    def click(rect):
        x, y, w, h = rect
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()

    state.frame, state.click = frame, click
    frame(3)
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (230, 330, 400, 700, 759, 760, 800, 1440))
def test_header_menus_fit_the_reserved_controls_and_keep_all_entries_accessible(menu_bar, width):
    state = menu_bar
    state.width = width
    state.frame(2)
    menus, scale = state.menus, state.ui.s
    assert menus.visible_menus + menus.overflow_menus == TOP_MENUS
    reserved_x = (width - (228 if width >= 760 else 57) - 8) * scale
    displayed = (*menus.visible_menus, "More") if menus.overflow_menus else menus.visible_menus
    right = 0
    for name in displayed:
        rect = menus.menu_rects[name]
        x, _y, w, h = rect
        assert x >= right - 1
        assert x + w <= reserved_x + 1
        assert h == pytest.approx(menus.menu_height)
        assert state.ui.hits["menu-" + name] == rect
        right = x + w


def test_explicit_reserved_edge_reflows_the_real_native_menu_items(menu_bar):
    state = menu_bar
    state.width, state.right_edge = 900, 220 * state.ui.s
    state.frame(2)
    assert state.menus.overflow_menus
    displayed = (*state.menus.visible_menus, "More")
    assert max(
        state.menus.menu_rects[name][0] + state.menus.menu_rects[name][2] for name in displayed
    ) <= (state.right_edge - 8 * state.ui.s + 1)


@pytest.mark.parametrize("padding_y", (7, 11))
def test_menu_center_uses_the_visible_font_cap_bounds(menu_bar, padding_y):
    state = menu_bar
    state.padding_y = padding_y
    state.ui.translate = lambda value: "H" if value in (*TOP_MENUS, "More") else value
    state.frame(2)
    rgba = imgui.get_color_u32(imgui.Col_.text)
    vertices = [vertex.pos.y for vertex in state.draw_list.vtx_buffer if vertex.col == rgba]
    assert vertices
    assert (min(vertices) + max(vertices)) / 2 == pytest.approx(state.menus.menu_center_y, abs=0.01)


@pytest.mark.parametrize(
    "path",
    [
        *((name,) for name in TOP_MENUS),
        ("Window", "Workspace"),
        ("Window", "Panels"),
        ("Window", "Inspector position"),
        ("View", "Projection"),
        ("View", "Transform tool"),
        ("View", "Gizmo orientation"),
    ],
)
def test_overflow_preserves_the_existing_gallery_popup_paths(menu_bar, path):
    state = menu_bar
    state.menus.preview(path)
    state.frame(4)
    assert "/".join(path) in state.menus.observed
    for popup in state.menus.observed.values():
        x, y, w, h = popup["rect"]
        assert x >= 0 and y >= 0
        assert x + w <= state.io.display_size.x + 1
        assert y + h <= state.io.display_size.y + 1


def test_more_opens_a_hidden_menu_through_actual_native_mouse_input(menu_bar):
    state = menu_bar
    menus = state.menus
    assert "Window" in menus.overflow_menus
    state.click(menus.menu_rects["More"])
    assert "More" in menus.observed
    state.click(menus.menu_rects["Window"])
    assert "Window" in menus.observed
    workspace = next(row for row in menus.observed["Window"]["rows"] if row["label"] == "Workspace")
    x, y, width, height = workspace["rect"]
    state.io.add_mouse_pos_event(x + width / 2, y + height / 2)
    state.frame(2)
    assert "Window/Workspace" in menus.observed
