"""Native base-color popups retain their own edit transaction and target."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native.reference_menus import Menus as StudyMenus
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.session.model_edits import model_edit_scope
from mojive.types import MeshShape
from mojive.ui.app import ViewerApp
from mojive.ui.workspace.entities import EntityProjection
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(Menus, StudyMenus), ids=("workspace", "study"))
def color_popup(request, tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800, 800)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    scene = Scene()
    first = scene.box(name="first", color=(0.2, 0.4, 0.6, 1))
    second = scene.box(name="second", color=(0.7, 0.3, 0.1, 1))
    session = Session(StaticSceneAdapter(scene))
    app = ViewerApp(session, NullBackend())
    app.live_model_updates = True
    projection = EntityProjection(app)
    app.refresh_document = projection.refresh_document
    app.apply_document_edits = projection.apply_document_edits
    ui = UI(
        SimpleNamespace(style_scale=1, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    workspace = SimpleNamespace(app=app, preview=app, ui=ui, s=1)
    menu = request.param(workspace, ())
    state = SimpleNamespace(
        app=app,
        projection=projection,
        ui=ui,
        io=io,
        menu=menu,
        first=session.node_by_object_id(first.object_id),
        second=session.node_by_object_id(second.object_id),
    )

    def frame(count=1):
        for _ in range(count):
            projection.refresh_document()
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Inspector")
            with model_edit_scope(session, app._intercept_model_edit):
                menu.begin_frame()
                menu.color_control(projection.entity, 30, 30, 300)
                projection.apply_document_edits()
                menu.finish_color_edit()
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def click(key, *, fraction=(0.5, 0.5)):
        x, y, width, height = ui.hits[key]
        event("add_mouse_pos_event", x + width * fraction[0], y + height * fraction[1])
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)

    state.frame, state.event, state.click = frame, event, click
    assert session.submit(cmd.SelectNode(state.first.node_id)).ok
    try:
        frame(3)
        yield state
    finally:
        app.release()
        imgui.destroy_context(context)


def _open(state):
    state.click("material-color")
    state.frame(2)
    assert "Color" in state.menu.observed


def test_failed_color_transaction_does_not_open_or_finish_another_edit(color_popup):
    state, session = color_popup, color_popup.app.session
    assert session.submit(cmd.BeginEditTransaction("Other gesture")).ok
    assert session.submit(cmd.SetPose(state.first.node_id, np.array((3, 4, 5)), np.eye(3))).ok
    state.click("material-color")
    state.frame(2)
    assert "Color" not in state.menu.observed
    assert session.editing
    assert session.submit(cmd.CancelEditTransaction()).ok
    state.frame()
    assert state.projection.entity["position"] == pytest.approx((0, 0, 0))
    assert not session.can_undo


@pytest.mark.parametrize("close", ("done", "escape", "selection"))
def test_color_popup_commit_cancel_and_target_change(color_popup, close):
    state, session = color_popup, color_popup.app.session
    initial = state.projection.entity["rgba"].copy()
    _open(state)
    state.click("color-editor-picker", fraction=(0.2, 0.3))
    state.click("color-editor-picker", fraction=(0.6, 0.7))
    assert session.editing
    assert not session.can_undo

    assert state.projection.entity["rgba"] != pytest.approx(initial)
    if close == "done":
        state.click("done-color")
    elif close == "escape":
        state.event("add_key_event", imgui.Key.escape, True)
        state.event("add_key_event", imgui.Key.escape, False)
    else:
        second_color = state.second.object_id
        assert session.submit(cmd.SelectNode(state.second.node_id)).ok
    state.frame(3)
    assert not session.editing
    if close == "selection":
        assert session.selected_node.object_id == second_color
        assert state.projection.entity["rgba"] == pytest.approx((0.7, 0.3, 0.1, 1))
        assert session.submit(cmd.SelectNode(state.first.node_id)).ok
        state.frame()
    if close in ("done", "selection"):
        assert session.can_undo and session.submit(cmd.Undo()).ok
        state.frame()
        assert session.submit(cmd.Redo()).ok
        state.frame()
        assert state.projection.entity["rgba"] != pytest.approx(initial)
        assert session.submit(cmd.Undo()).ok
    state.frame()
    assert state.projection.entity["rgba"] == pytest.approx(initial)
    assert not session.can_undo


@pytest.mark.parametrize("edit_again", (False, True))
def test_restore_original_color_stays_in_the_existing_gesture(color_popup, edit_again):
    state, session = color_popup, color_popup.app.session
    initial = state.projection.entity["rgba"].copy()
    _open(state)
    state.click("color-editor-picker", fraction=(0.2, 0.3))
    assert state.projection.entity["rgba"] != pytest.approx(initial)
    state.click("color-editor-original")
    assert session.editing and not session.can_undo
    assert "Color" in state.menu.observed
    assert state.projection.entity["rgba"] == pytest.approx(initial)
    if edit_again:
        state.click("color-editor-picker", fraction=(0.6, 0.7))
        assert state.projection.entity["rgba"] != pytest.approx(initial)
        state.click("done-color")
    else:
        state.event("add_key_event", imgui.Key.escape, True)
        state.event("add_key_event", imgui.Key.escape, False)
    state.frame(3)
    assert not session.editing
    assert session.can_undo == edit_again
    if edit_again:
        assert session.submit(cmd.Undo()).ok
        state.frame()
        assert not session.can_undo
    assert state.projection.entity["rgba"] == pytest.approx(initial)


@pytest.mark.parametrize("finish", (cmd.EndEditTransaction, cmd.CancelEditTransaction))
def test_stale_color_popup_does_not_end_a_replacement_transaction(color_popup, finish):
    state, session = color_popup, color_popup.app.session
    _open(state)
    assert session.submit(finish()).ok
    assert session.submit(cmd.BeginEditTransaction("Replacement gesture")).ok
    assert session.submit(cmd.SetPose(state.second.node_id, np.array((3, 4, 5)), np.eye(3))).ok
    state.frame(3)
    assert "Color" not in state.menu.observed
    assert session.editing
    state.event("add_key_event", imgui.Key.escape, True)
    state.event("add_key_event", imgui.Key.escape, False)
    assert session.editing
    assert session.submit(cmd.CancelEditTransaction()).ok
    state.frame()
    assert session.node(state.second.node_id) is not None


@pytest.mark.parametrize("close", ("done", "escape"))
def test_color_popup_works_without_adapter_edit_history(color_popup, monkeypatch, close):
    state, session = color_popup, color_popup.app.session
    monkeypatch.setattr(session.adapter, "caps", replace(session.adapter.caps, edit_history=False))
    initial = state.projection.entity["rgba"].copy()
    _open(state)
    state.click("color-editor-picker", fraction=(0.2, 0.3))
    assert state.projection.entity["rgba"] != pytest.approx(initial)
    assert not session.editing
    if close == "done":
        state.click("done-color")
    else:
        state.event("add_key_event", imgui.Key.escape, True)
        state.event("add_key_event", imgui.Key.escape, False)
    state.frame(3)
    assert state.menu.color_edit is None and not session.can_undo
    if close == "escape":
        assert state.projection.entity["rgba"] == pytest.approx(initial)
    else:
        assert state.projection.entity["rgba"] != pytest.approx(initial)


def test_color_popup_does_not_end_new_document_transaction(color_popup):
    state, session = color_popup, color_popup.app.session
    _open(state)
    assert session.submit(cmd.EndEditTransaction()).ok
    assert session.submit(cmd.NewScene()).ok
    result = session.submit(cmd.AddSceneObject(MeshShape.BOX, "replacement"))
    assert result.ok
    target = session.node_by_object_id(result.entity_id)
    assert session.submit(cmd.SelectNode(target.node_id)).ok
    assert session.submit(cmd.BeginEditTransaction("New document gesture")).ok
    state.frame(3)
    assert session.editing and "Color" not in state.menu.observed
    assert session.submit(cmd.CancelEditTransaction()).ok


@pytest.mark.parametrize("close", ("done", "escape"))
def test_color_popup_tracks_its_deferred_transaction(color_popup, close):
    state, session = color_popup, color_popup.app.session
    state.app.live_model_updates = False
    initial = state.projection.entity["rgba"].copy()
    _open(state)
    assert state.app.model_edits._checkpoint is not None
    state.click("color-editor-picker", fraction=(0.2, 0.3))
    assert state.projection.entity["rgba"] != pytest.approx(initial) and session.editing
    if close == "done":
        state.click("done-color")
    else:
        state.event("add_key_event", imgui.Key.escape, True)
        state.event("add_key_event", imgui.Key.escape, False)
    state.frame(3)
    assert not session.editing and state.app.model_edits._checkpoint is None
    if close == "done":
        assert session.submit(cmd.Undo()).ok
        state.frame()
    assert state.projection.entity["rgba"] == pytest.approx(initial)


def test_color_cancel_reports_restore_failure_without_history(color_popup, monkeypatch):
    state, session = color_popup, color_popup.app.session
    monkeypatch.setattr(session.adapter, "caps", replace(session.adapter.caps, edit_history=False))
    _open(state)
    state.click("color-editor-picker", fraction=(0.2, 0.3))
    changed = state.projection.entity["rgba"].copy()
    submit = session.submit

    def reject_restore(command):
        if isinstance(command, cmd.SetGeometryColor):
            return session._record_result(cmd.CommandResult.bad("Color restore failed"))
        return submit(command)

    monkeypatch.setattr(session, "submit", reject_restore)
    state.event("add_key_event", imgui.Key.escape, True)
    state.event("add_key_event", imgui.Key.escape, False)
    state.frame(3)
    assert session.last_message == "Color restore failed"
    assert state.projection.entity["rgba"] == pytest.approx(changed)


@pytest.mark.parametrize("history", (False, True))
@pytest.mark.parametrize("close", ("frame", "escape"))
def test_color_popup_rejects_reused_geometry_identity(color_popup, monkeypatch, history, close):
    state, session = color_popup, color_popup.app.session
    monkeypatch.setattr(
        session.adapter, "caps", replace(session.adapter.caps, edit_history=history)
    )
    original = session.node(state.first.children[0])
    assert session.submit(cmd.SelectNode(original.node_id)).ok
    state.frame()
    identity = state.projection.entity["identity"]
    _open(state)
    state.click("color-editor-picker", fraction=(0.2, 0.3))
    assert session.submit(cmd.RemoveSceneEntity(state.first.object_id)).ok
    assert session.submit(cmd.RemoveSceneEntity(state.second.object_id)).ok
    rgba = (0.8, 0.1, 0.3, 1)
    result = session.submit(cmd.AddSceneObject(MeshShape.BOX, "first", color=rgba))
    assert result.ok
    replacement = session.node_by_object_id(result.entity_id)
    target = session.node(replacement.children[0])
    assert session.submit(cmd.SelectNode(target.node_id)).ok
    assert state.projection._identity(target) == identity
    if close == "escape":
        state.event("add_key_event", imgui.Key.escape, True)
        state.event("add_key_event", imgui.Key.escape, False)
    state.frame(3)
    assert not session.editing and state.menu.color_edit is None
    assert "Color" not in state.menu.observed
    assert state.projection.entity["rgba"] == pytest.approx(rgba)
    assert session.last_message == "The scene changed; the color editor was closed"
