"""Native Add preserves the established creation commands without a renderer."""

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import AdapterCaps, NodeType
from mojive.adapters.static import StaticSceneAdapter
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import CameraView
from mojive.ui.app.menus import _Menus
from mojive.ui.workspace.menu_actions import dispatch
from mojive.ui.workspace.menu_catalog import items


def _menu(session):
    app = _Menus()
    app.session = session
    app.entity = None
    app.localizer = SimpleNamespace(text=lambda text: text)
    app._camera_view = lambda: CameraView(eye=np.array((2, -3, 4), dtype=np.float32))
    return SimpleNamespace(workspace=SimpleNamespace(app=app))


def test_native_add_preserves_all_classic_creation_entries(monkeypatch):
    menu = _menu(SimpleNamespace(adapter=SimpleNamespace(caps=AdapterCaps())))
    classic = []
    menu.workspace.app._menu_action = lambda _actions, label, *_args, **_kwargs: classic.append(
        label
    )
    monkeypatch.setattr(imgui, "separator", lambda: None)
    menu.workspace.app._draw_create_entity_menu([])
    context = imgui.create_context()
    try:
        native = [row for row in items(menu, "Add") if row]
        assert {row.label for row in native} == set(classic)
        assert all(row.action == "create:" + row.label for row in native)
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "label, kind, name",
    [("Point Light", NodeType.LIGHT, "point light"), ("Camera", NodeType.CAMERA, "camera")],
)
def test_add_scene_entity_creates_and_selects_without_physics(label, kind, name):
    session = Session(StaticSceneAdapter(Scene()))
    menu = _menu(session)
    try:
        assert not session.adapter.caps.topology_editing
        for expected_name in (name, name + " 2"):
            dispatch(menu, "create:" + label)
            selected = session.selected_node
            assert selected is not None
            assert (selected.type, selected.name) == (kind, expected_name)
            assert session.node_by_object_id(session.selected) == selected
        assert len([node for node in session.nodes if node.type is kind]) == 2
    finally:
        session.release()


@pytest.mark.physics
def test_add_site_uses_selected_body_and_one_undoable_edit(tmp_path):
    pytest.importorskip("mujoco")
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.workspace import WorkspaceAdapter

    path = tmp_path / "site.xml"
    path.write_text(
        '<mujoco><worldbody><body name="arm"><geom size=".1"/></body></worldbody></mujoco>'
    )
    session = Session(WorkspaceAdapter(MuJoCoAdapter(path)), path)
    try:
        assert session.submit(cmd.Pause()).ok
        body = next(node for node in session.nodes if node.name == "arm")
        assert session.submit(cmd.SelectNode(body.node_id)).ok
        dispatch(_menu(session), "create:Site")
        site = session.selected_node
        assert site is not None
        assert (site.type, site.name, site.parent) == (NodeType.SITE, "site", body.node_id)
        assert session.site_properties(site.node_id) is not None
        assert session.submit(cmd.Undo()).ok
        assert not any(node.type is NodeType.SITE for node in session.nodes)
        assert session.submit(cmd.Redo()).ok
        assert session.selected_node.name == "site"
    finally:
        session.release()
