"""Native Restore snapshots retain load-time transform and color defaults."""

from types import SimpleNamespace

import numpy as np
import pytest

from mojive import Scene, math3d
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.static import StaticSceneAdapter
from mojive.scene.queries import node_world_pose
from mojive.session import Session
from mojive.ui.theme import THEME
from mojive.ui.workspace.entities import EntityProjection


def _app(session):
    return SimpleNamespace(
        session=session,
        theme=THEME,
        workspace=SimpleNamespace(pinned=None),
        model_edits=SimpleNamespace(_checkpoint=None),
    )


def _original(projection, node):
    identity = projection._identity(node)
    return next(entry for entry in projection.initial_entities if entry["identity"] == identity)


def test_initial_snapshots_only_read_restore_fields(monkeypatch):
    scene = Scene()
    for index in range(128):
        scene.box(name=f"box{index}", position=(index, 2, 3))
    session = Session(StaticSceneAdapter(scene))

    def unexpected_projection(*args):
        pytest.fail("Initial Restore snapshots must not build display fields")

    monkeypatch.setattr(EntityProjection, "_project_node", unexpected_projection)
    try:
        projection = EntityProjection(_app(session))
        assert len(projection.initial_entities) == len(session.nodes)
        assert all(
            set(entry) == {"identity", "position", "rotation", "scale", "rgba"}
            for entry in projection.initial_entities
        )
        assert not projection.entities
        before = projection.initial_entities.copy()
        projection.refresh_document()
        assert all(
            old is current for old, current in zip(before, projection.initial_entities, strict=True)
        )
    finally:
        session.release()


def test_restore_retains_load_time_transform_after_edits_and_structure_changes():
    scene = Scene()
    initial_position = np.array((1, 2, 3), np.float64)
    initial_rotation = math3d.euler_xyz_to_mat3(np.radians((10, 20, 30)))
    obj = scene.box(name="original", position=initial_position, rotation=initial_rotation)
    session = Session(StaticSceneAdapter(scene))
    try:
        projection = EntityProjection(_app(session))
        node = session.node_by_object_id(obj.object_id)
        original = _original(projection, node)
        assert session.submit(cmd.SelectNode(node.node_id)).ok
        projection.refresh_document()
        projection.entity["position"] = [5, 6, 7]
        projection.entity["rotation"] = [40, 50, 60]
        projection.apply_document_edits()
        assert original["position"] == pytest.approx(initial_position)
        assert original["rotation"] == pytest.approx((10, 20, 30))
        duplicate = session.submit(cmd.DuplicateSceneEntity(obj.object_id))
        assert duplicate.ok
        projection.refresh_document()
        copy_node = session.node_by_object_id(duplicate.entity_id)
        copy_original = _original(projection, copy_node)
        assert copy_original["position"] == pytest.approx((5, 6, 7))
        assert _original(projection, session.node_by_object_id(obj.object_id)) is original
        assert session.submit(cmd.SelectNode(session.node_by_object_id(obj.object_id).node_id)).ok
        projection.refresh_document()
        for name in ("position", "rotation", "scale"):
            projection.entity[name] = original[name]
        projection.apply_document_edits()
        session.tick(FrameNeeds(poses=True), wall_dt=0)
        position, rotation = node_world_pose(session, session.node_by_object_id(obj.object_id))
        assert position == pytest.approx(initial_position)
        assert rotation == pytest.approx(initial_rotation)
        assert session.submit(cmd.Undo()).ok
        session.tick(FrameNeeds(poses=True), wall_dt=0)
        position, _ = node_world_pose(session, session.node_by_object_id(obj.object_id))
        assert position == pytest.approx((5, 6, 7))
    finally:
        session.release()


def test_deleted_entity_projection_is_evicted_and_undo_preserves_restore_baseline():
    scene = Scene()
    obj = scene.box(name="box", position=(1, 2, 3))
    session = Session(StaticSceneAdapter(scene))
    try:
        projection = EntityProjection(_app(session))
        node = session.node_by_object_id(obj.object_id)
        original = _original(projection, node)
        assert session.submit(cmd.SelectNode(node.node_id)).ok
        projection.refresh_document()
        assert node.node_id in projection._entity_cache
        assert session.submit(cmd.RemoveSceneEntity(obj.object_id)).ok
        projection.refresh_document()
        assert node.node_id not in projection._entity_cache
        assert session.submit(cmd.Undo()).ok
        projection.refresh_document()
        restored = session.node_by_object_id(obj.object_id)
        assert _original(projection, restored) is original
        assert (
            sum(entry["identity"] == original["identity"] for entry in projection.initial_entities)
            == 1
        )
    finally:
        session.release()


def test_new_document_resets_transform_snapshots_even_when_ids_are_reused():
    first, second = Scene(), Scene()
    first_box = first.box(name="box", position=(1, 0, 0))
    second_box = second.box(name="box", position=(9, 0, 0))
    first_session = Session(StaticSceneAdapter(first))
    second_session = Session(StaticSceneAdapter(second))
    try:
        app = _app(first_session)
        projection = EntityProjection(app)
        first_node = first_session.node_by_object_id(first_box.object_id)
        original = _original(projection, first_node)
        app.workspace.pinned = original["identity"]
        app.session = second_session
        projection.refresh_document()
        second_node = second_session.node_by_object_id(second_box.object_id)
        assert projection._identity(first_node) == projection._identity(second_node)
        updated = _original(projection, second_node)
        assert updated is not original
        assert updated["position"] == pytest.approx((9, 0, 0))
        assert len(projection.initial_entities) == len(second_session.nodes)
        assert app.workspace.pinned is None
    finally:
        first_session.release()
        second_session.release()


@pytest.mark.physics
def test_model_node_reindex_preserves_transform_baseline_by_source_identity(tmp_path):
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.workspace import WorkspaceAdapter

    path = tmp_path / "bodies.xml"
    path.write_text(
        '<mujoco><worldbody><body name="first"><freejoint/><geom size=".1"/></body>'
        '<body name="target" pos="1 2 3"><freejoint/><geom size=".1"/></body>'
        "</worldbody></mujoco>"
    )
    session = Session(WorkspaceAdapter(MuJoCoAdapter(path), Scene()))
    assert session.submit(cmd.Pause()).ok
    try:
        projection = EntityProjection(_app(session))
        first = next(node for node in session.nodes if node.name == "first")
        target = next(node for node in session.nodes if node.name == "target")
        original = _original(projection, target)
        assert session.submit(cmd.SelectNode(target.node_id)).ok
        projection.refresh_document()
        assert session.submit(cmd.RemoveModelElement(first.node_id)).ok
        projection.refresh_document()
        remapped = next(node for node in session.nodes if node.name == "target")
        assert remapped.node_id != target.node_id
        assert _original(projection, remapped) is original
        assert session.submit(cmd.SelectNode(remapped.node_id)).ok
        projection.refresh_document()
        assert projection.entity["identity"] == original["identity"]
        assert projection.entity["node_id"] == remapped.node_id
        projection.entity["position"] = [8, 9, 10]
        projection.apply_document_edits()
        session.tick(FrameNeeds(poses=True), wall_dt=0)
        projection.refresh_document()
        for name in ("position", "rotation", "scale"):
            projection.entity[name] = original[name]
        projection.apply_document_edits()
        session.tick(FrameNeeds(poses=True), wall_dt=0)
        position, _ = node_world_pose(session, session.node(remapped.node_id))
        assert position == pytest.approx((1, 2, 3))
    finally:
        session.release()


@pytest.mark.parametrize("scale", (1, 2))
@pytest.mark.parametrize("axis", "RGBA")
def test_material_axis_click_restores_load_time_color_and_undo(scale, axis):
    from imgui_bundle import imgui

    from mojive.adapters.base import NodeType
    from mojive.render.backend import NullBackend
    from mojive.ui.app import ViewerApp
    from mojive.ui.panels import PanelContext
    from mojive.ui.workspace import inspector, transactions
    from mojive.ui.workspace.menus import Menus
    from mojive.ui.workspace.widgets import UI

    initial = np.array((0.2, 0.4, 0.6, 0.8))
    edited = np.array((0.9, 0.7, 0.3, 0.25))
    scene = Scene()
    obj = scene.box(name="box", color=initial)
    session = Session(StaticSceneAdapter(scene))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (420 * scale, 760 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s = scale
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.hits, ui.numeric_edits, ui.translate = {}, {}, str
    ui._section_columns = []
    app = ViewerApp(session, NullBackend())
    projection = EntityProjection(app)
    app.initial_entities, app.entities = projection.initial_entities, projection.entities
    app.apply_document_edits, app.refresh_document = (
        projection.apply_document_edits,
        projection.refresh_document,
    )
    workspace = SimpleNamespace(
        s=scale,
        ui=ui,
        app=app,
        pinned=None,
        open={"Inspector": True},
        inspector_tab="Material",
        playing=False,
        _numeric_edit=None,
    )
    ctx = PanelContext(session, app.backend, style_scale=scale)
    ui.value_context = ctx
    workspace._panel_context = ctx
    workspace.menus = Menus(workspace, ())
    workspace.begin_numeric_edit = lambda key, **kw: transactions.begin_numeric_edit(
        workspace, key, **kw
    )
    workspace.finish_numeric_edit = lambda key, state: transactions.finish_numeric_edit(
        workspace, key, state
    )

    def frame(count=1):
        for _ in range(count):
            session.tick(FrameNeeds(poses=True), wall_dt=0)
            projection.refresh_document()
            app.entity = projection.entity
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((420 * scale, 760 * scale))
            imgui.begin("Inspector")
            pos, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
            inspector.inspector(workspace, pos.x, pos.y, size.x, size.y)
            imgui.end()
            transactions.finish_numeric_frame(workspace, ctx)
            imgui.render()

    try:
        node = session.node_by_object_id(obj.object_id)
        geom = next(
            session.node(child)
            for child in node.children
            if session.node(child).type is NodeType.GEOM
        )
        assert session.submit(cmd.SetGeometryColor(geom.node_id, edited)).ok
        # First presentation happens after the color edit; reset must still use load-time RGBA.
        assert session.submit(cmd.SelectNode(node.node_id)).ok
        frame(3)
        key = "##" + projection.entity["id"] + "color" + axis + "-reset"
        x, y, width, height = ui.hits[key]
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame(2)
        expected = edited.copy()
        expected["RGBA".index(axis)] = initial["RGBA".index(axis)]
        assert projection.entity["rgba"] == pytest.approx(expected)
        assert _original(projection, node)["rgba"] == pytest.approx(initial)
        assert session.submit(cmd.Undo()).ok
        frame()
        assert projection.entity["rgba"] == pytest.approx(edited)
    finally:
        imgui.destroy_context(context)
        session.release()
