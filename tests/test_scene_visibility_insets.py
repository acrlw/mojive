"""Scene eye affordances breathe inside selected rows and retain real input."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.adapters.static import StaticSceneAdapter
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.navigation import Navigation
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 1.2, 1.25, 2.25))
def visibility_row(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (700 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    scene = Scene()
    scene.box(name="sample_geometry")
    session = Session(StaticSceneAdapter(scene))
    node_id = next(node.node_id for node in session.nodes if node.type is NodeType.GEOM)
    assert session.submit(cmd.SelectNode(node_id)).ok
    navigation = Navigation.__new__(Navigation)
    navigation.workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        menus=SimpleNamespace(local_popup=lambda *args, **kwargs: nullcontext(False)),
    )
    navigation.hierarchy = SimpleNamespace(_batch_selected=set())
    ctx = PanelContext(session, NullBackend(), style_scale=ui.s)
    state = SimpleNamespace(
        ui=ui,
        io=io,
        runtime=runtime,
        session=session,
        node_id=node_id,
        paints=[],
        width=300,
        depth=0,
    )
    original = ui.rect

    def rectangle(x, y, width, height, *args, **kwargs):
        state.paints.append((x, y, width, height, kwargs.get("r")))
        return original(x, y, width, height, *args, **kwargs)

    monkeypatch.setattr(ui, "rect", rectangle)

    def frame(count=1):
        for _ in range(count):
            state.paints.clear()
            ui.hits.clear()
            runtime.begin_frame()
            ui.design.begin_frame()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Scene", flags=imgui.WindowFlags_.no_title_bar)
            imgui.set_cursor_screen_pos((20 * ui.s, 60 * ui.s))
            navigation.draw_scene_row(
                ctx, session.node(node_id), state.depth, True, state.width * ui.s
            )
            imgui.end()
            imgui.render()
        state.row = next(r for r in runtime.observed if r["key"] == "tree-row-" + str(node_id))

    state.frame = frame
    try:
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize("width,depth", ((160, 0), (160, 3), (300, 0), (300, 3)))
@pytest.mark.parametrize("authored_radius", (None, 7))
def test_visibility_plate_has_trailing_and_vertical_clearance_without_reindenting_tree(
    visibility_row, width, depth, authored_radius
):
    state, ui = visibility_row, visibility_row.ui
    state.width, state.depth = width, depth
    if authored_radius is not None:
        spec = deepcopy(state.runtime.spec)
        spec["parts"].setdefault("icon_button", {})["radius"] = authored_radius
        state.runtime.update_spec(spec)
    state.io.add_mouse_pos_event((20 + width / 2) * ui.s, 74 * ui.s)
    state.frame(3)
    eye_key = "visible-" + str(state.node_id)
    x, y, w, h = ui.hits[eye_key]
    state.io.add_mouse_pos_event(x + w / 2, y + h / 2)
    state.frame(3)
    x, y, w, h = ui.hits[eye_key]
    rx, ry, rw, rh = state.row["hit"]
    assert (rx, ry, rw, rh) == pytest.approx((20 * ui.s, 60 * ui.s, width * ui.s, 28 * ui.s))
    assert state.row["padding"] == [0, 0, 0, 0]
    assert (rx + rw - x - w, y - ry, ry + rh - y - h) == pytest.approx(
        (4 * ui.s, 3 * ui.s, 3 * ui.s)
    )
    radius = 1 if authored_radius is None else authored_radius
    assert state.row["visibilityRadius"] == pytest.approx(radius * ui.s)
    assert any(
        paint[:4] == pytest.approx((x, y, w, h)) and paint[4] == pytest.approx(radius)
        for paint in state.paints
    )
    label = ui.hits["entity-" + str(state.node_id)]
    assert label[0] + label[2] <= x + 1e-4
    state.io.add_mouse_button_event(0, True)
    state.frame()
    state.io.add_mouse_button_event(0, False)
    state.frame(2)
    assert not state.session.node(state.node_id).visible
