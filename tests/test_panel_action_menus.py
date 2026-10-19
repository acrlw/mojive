"""Production action menus share padded rows and only separate nonempty groups."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext, PanelManager
from mojive.ui.workspace.constants import LEFT_PANELS
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.panel_host import panel_header
from mojive.ui.workspace.timeline_track import NativeTimelineTrack
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 1.25, 2.25))
def action_surface(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.display_size = (1000 * request.param, 800 * request.param)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "ui.json")))
    actions = []
    session = Session(StaticSceneAdapter(Scene()))
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(
            gizmos=True, session=session, panels=PanelManager(builtin_ids=("control",))
        ),
        open=dict.fromkeys(LEFT_PANELS, True),
        header_actions={name: set() for name in LEFT_PANELS},
        header_action=lambda *args: actions.append(args),
        header_add_pending=False,
        search_visible=False,
        control_search_visible=False,
        filter="All",
        navigation=SimpleNamespace(
            pose_action_active=lambda _key: False,
            scene_filters=lambda _ctx: (("all", 5), ("camera", 2)),
        ),
        _panel_context=PanelContext(session, NullBackend(), style_scale=ui.s),
    )
    workspace.menus = menus = Menus(workspace, ())
    state = SimpleNamespace(
        workspace=workspace, menus=menus, ui=ui, io=io, actions=actions, draw=lambda: None
    )

    def frame(count=1):
        for _ in range(count):
            ui.hits.clear()
            ui.design.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin(
                "Action menu host",
                flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.menu_bar,
            )
            menus.begin_frame()
            state.draw()
            imgui.end()
            imgui.render()

    def click(key):
        x, y, width, height = ui.hits[key]
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        frame(2)
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame(3)

    def reject_native(*_args, **_kwargs):
        raise AssertionError("Production action menus must use the shared row and separator")

    monkeypatch.setattr(imgui, "menu_item", reject_native)
    monkeypatch.setattr(imgui, "separator", reject_native)
    state.frame, state.click = frame, click
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("name", LEFT_PANELS)
def test_every_sidebar_more_keeps_all_rows_inside_one_padded_content_column(
    action_surface, language, name
):
    state = action_surface
    state.ui.translate = Localizer(Language(language)).text

    def draw():
        with group_scope(name):
            panel_header(state.workspace, name, 30 * state.ui.s, 30 * state.ui.s, 220 * state.ui.s)

    state.draw = draw
    state.frame(3)
    state.click("header-" + name + "-more")
    popup = state.menus.observed["Panel actions###" + name]
    rows = popup["rows"]
    frames = [row["frame_rect"] for row in rows]
    assert rows[-1]["label"] == "Hide panel"
    assert all(row["visible"] and not row["clippedLabel"] for row in rows)
    assert all(row[0] == pytest.approx(frames[0][0], abs=0.01) for row in frames)
    assert all(row[2] == pytest.approx(frames[0][2], abs=0.01) for row in frames)
    x, y, width, height = popup["rect"]
    menu_style = state.ui.component_style("menu")
    padding = menu_style.edges("padding", state.ui.s)
    assert frames[0][0] - x == pytest.approx(padding[0], abs=1)
    assert x + width - frames[-1][0] - frames[-1][2] == pytest.approx(padding[2], abs=1)
    dividers = popup.get("dividers", ())
    if name in ("Assets", "Sensors"):
        assert len(rows) == 1 and not dividers
    else:
        assert dividers
        first_bottom = frames[0][1] + frames[0][3]
        assert all(first_bottom <= divider[1] < frames[-1][1] for divider in dividers)
    assert frames[0][1] - y == pytest.approx(padding[1], abs=1)
    assert y + height - frames[-1][1] - frames[-1][3] == pytest.approx(padding[3], abs=1)
    hx, hy, hw, hh = state.ui.hits["header-menu-" + name + "-hide"]
    state.io.add_mouse_pos_event(hx + hw / 2, hy + hh / 2)
    state.frame(3)
    hovered = state.menus.observed["Panel actions###" + name]["rows"][-1]
    assert hovered["hovered"] and hovered["backgroundRect"] == hovered["frame_rect"]
    assert hovered["backgroundRadius"] == pytest.approx(
        state.ui.component_style("menu_item").get("radius", 6) * state.ui.s
    )
    state.click("header-menu-" + name + "-hide")
    assert not state.workspace.open[name]


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_header_action_preferences_use_the_same_checked_menu_row(action_surface, language):
    state = action_surface
    state.ui.translate = Localizer(Language(language)).text
    state.draw = lambda: panel_header(
        state.workspace, "Layers", 30 * state.ui.s, 30 * state.ui.s, 220 * state.ui.s
    )
    state.frame(3)
    state.click("header-Layers-more")
    x, y, width, height = state.ui.hits["header-menu-Layers-actions"]
    state.io.add_mouse_pos_event(x + width / 2, y + height / 2)
    state.frame(16)
    assert "header-action-Layers-gizmos" in state.ui.hits
    state.click("header-action-Layers-gizmos")
    assert state.workspace.header_actions["Layers"] == {"gizmos"}


def test_pending_separators_trim_empty_edges_and_preserve_row_alignment(action_surface):
    state = action_surface

    def draw():
        if not imgui.is_popup_open("Groups"):
            imgui.open_popup("Groups")
        with state.menus.local_popup("Groups") as opened:
            if opened:
                state.menus.local_separator()
                state.menus.local_item("First", key="first")
                state.menus.local_separator()
                state.menus.local_separator()
                state.menus.local_item("Second", key="second")
                state.menus.local_separator()

    state.draw = draw
    state.frame(6)
    popup = state.menus.observed["Groups"]
    assert len(popup["dividers"]) == 1
    first, second = popup["rows"]
    assert first["rect"][0] == second["rect"][0]
    assert first["rect"][2] == second["rect"][2]
    divider = popup["dividers"][0]
    inset = state.ui.metric("separator", "margin", 0) * state.ui.s
    assert divider[0] == first["rect"][0] + inset
    assert first["rect"][1] + first["rect"][3] < divider[1] < second["rect"][1]


def test_application_more_keeps_existing_hidden_menu_identity_and_preview_path(action_surface):
    state = action_surface
    state.menus.items = lambda name: (Item("Run " + name, action=name),)
    state.draw = lambda: state.menus.bar(280 * state.ui.s)
    state.frame(3)
    assert "Window" in state.menus.overflow_menus
    state.menus.preview(("Window",))
    state.frame(8)
    assert "More" in state.menus.observed and "Window" in state.menus.observed
    assert state.menus.observed["Window"]["rows"][0]["label"] == "Run Window"
    assert "Window" in state.menus.menu_rects


@pytest.mark.parametrize("enabled", (False, True))
def test_timeline_context_uses_shared_rows_and_preserves_selection_capability(
    action_surface, enabled
):
    state, calls = action_surface, []
    editor = SimpleNamespace(
        selection_count=lambda: 1,
        edit_lane="model",
        delete_selection=lambda *args: calls.append(args),
    )
    ctx = SimpleNamespace(
        session=SimpleNamespace(state_take_recording=False), take_video_active=False
    )
    track = NativeTimelineTrack(state.workspace)

    def draw():
        if not imgui.is_popup_open("timeline-selection-menu"):
            imgui.open_popup("timeline-selection-menu")
        track.draw_selection_menu(editor, ctx, enabled)

    state.draw = draw
    state.frame(5)
    row = state.menus.observed["timeline-selection-menu"]["rows"][0]
    assert row["label"] == "Delete selection" and row["icon"] == "delete"
    assert row["enabled"] is enabled
    state.click("timeline-delete-selection")
    assert calls == ([(ctx, True)] if enabled else [])
