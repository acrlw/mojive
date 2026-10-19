"""Scene-only controls retain their shared commands and ordinary layout."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import WorldSelectionInfo
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.panels import PanelContext, PanelManager
from mojive.ui.panels.hierarchy import HierarchyPanel
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.navigation import Navigation
from mojive.ui.workspace.widgets import UI


class WorldAdapter(StaticSceneAdapter):
    def __init__(self):
        super().__init__(Scene())
        self.info = WorldSelectionInfo(8, (0,), 4)

    def world_selection(self):
        return self.info

    def set_world_selection(self, ids):
        self.info = replace(self.info, world_ids=ids)
        return True


def test_scene_world_selection_is_conditional_and_routes_apply(monkeypatch):
    context = imgui.create_context()
    adapter = WorldAdapter()
    session = Session(adapter)
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (500, 700)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        ui = UI(SimpleNamespace(style_scale=1, font_report=SimpleNamespace(cjk_path=None)))
        workspace = SimpleNamespace(
            app=SimpleNamespace(
                session=session, panels=PanelManager(builtin_ids=("hierarchy", "joints"))
            ),
            ui=ui,
            s=1,
            query="",
            filter="All",
            search_visible=False,
        )
        workspace.menus = Menus(workspace, ())
        navigation = Navigation(workspace)
        ctx = PanelContext(session, NullBackend())
        buttons = {}
        original = imgui.button

        def button(label, *args, **kwargs):
            result = original(label, *args, **kwargs)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            buttons[label] = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
            return result

        monkeypatch.setattr(imgui, "button", button)

        def frame():
            buttons.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Scene")
            navigation.draw_scene(ctx, 20, 40, 460, 600)
            imgui.end()
            imgui.render()

        frame()
        frame()
        ordinary_filters = ui.hits["scene-type-filters"]
        assert "Apply worlds" not in buttons
        adapter.caps = replace(adapter.caps, features=(("world.selection", 1),))
        frame()
        assert ui.hits["scene-type-filters"][1] > ordinary_filters[1]
        control = navigation.hierarchy._world_selection
        control._start, control._count = 2, 2
        frame()
        io.add_mouse_pos_event(*buttons["Apply worlds"])
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()
        assert session.world_selection.world_ids == (2, 3)
        adapter.caps = replace(adapter.caps, features=())
        frame()
        assert "Apply worlds" not in buttons
        assert ui.hits["scene-type-filters"] == ordinary_filters
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.physics
def test_scene_batch_context_deletes_roots_with_one_undo(tmp_path):
    pytest.importorskip("mujoco")
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.workspace import WorkspaceAdapter

    path = tmp_path / "batch.xml"
    path.write_text(
        '<mujoco><worldbody><body name="first"><geom name="child" size=".1"/></body>'
        '<body name="second"><geom name="sibling" size=".1"/></body></worldbody></mujoco>'
    )
    session = Session(WorkspaceAdapter(MuJoCoAdapter(path)), path)
    try:
        assert session.submit(cmd.Pause()).ok
        ctx = PanelContext(session, NullBackend())
        hierarchy = HierarchyPanel()
        hierarchy._refresh(ctx)
        nodes = {node.name: node for node in session.nodes}
        hierarchy._batch_selected = {nodes[name].node_id for name in ("first", "child", "sibling")}
        rows = {}

        def item(label, *_args, key, enabled=True, **_kwargs):
            rows[key] = label, enabled
            return key == "scene-menu-delete-selected" and enabled, False

        menus = SimpleNamespace(
            local_popup=lambda *_args, **_kwargs: nullcontext(True),
            local_item=item,
            local_separator=lambda: None,
        )
        navigation = Navigation.__new__(Navigation)
        navigation.hierarchy = hierarchy
        navigation.workspace = SimpleNamespace(menus=menus)
        navigation._scene_context(ctx, nodes["child"])
        assert rows["scene-menu-delete-selected"] == ("Delete 2 model elements", True)
        assert not hierarchy._batch_selected
        assert {"first", "child", "sibling"}.isdisjoint(node.name for node in session.nodes)
        assert session.submit(cmd.Undo()).ok
        assert {"first", "child", "sibling"} <= {node.name for node in session.nodes}
        assert not session.can_undo
    finally:
        session.release()


@pytest.mark.parametrize("paused, ok", ((False, True), (True, False)))
def test_scene_batch_context_preserves_selection_when_disabled_or_rejected(paused, ok):
    selected = {1, 2}
    submitted = []
    navigation = Navigation.__new__(Navigation)
    navigation.hierarchy = SimpleNamespace(
        _batch_selected=selected, _batch_removable_roots=lambda: (1, 2)
    )
    navigation.workspace = SimpleNamespace(
        menus=SimpleNamespace(local_item=lambda *_args, enabled, **_kwargs: (enabled, False))
    )
    ctx = SimpleNamespace(
        tr=str,
        session=SimpleNamespace(
            paused=paused, adapter=SimpleNamespace(caps=SimpleNamespace(topology_editing=True))
        ),
        submit=lambda command: submitted.append(command) or SimpleNamespace(ok=ok),
    )
    navigation._batch_actions(ctx, SimpleNamespace(node_id=1))
    assert len(submitted) == int(paused)
    assert selected == {1, 2}
