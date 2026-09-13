"""Shared edit execution preserves order, ownership, history and failure evidence."""

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.control.operations import OPERATIONS, apply_session_operation
from mojive.control.rpc import ControlService, RpcError
from mojive.control.schema import Validator
from mojive.session import Session
from mojive.session.model_edits import ModelEditDraft


@pytest.fixture
def edited():
    scene = Scene()
    box = scene.box(size=(0.2, 0.3, 0.4))
    session = Session(StaticSceneAdapter(scene))
    service = ControlService(session=session)
    draft = ModelEditDraft(session)
    node = session.node_by_object_id(box.object_id)
    yield scene, session, service, draft, node
    service.close()
    session.release()


def transaction(service, operations):
    return service.dispatch("edit_scene", {"operations": operations})


def operation(method, **params):
    return {"method": method, "params": params}


@pytest.mark.parametrize("entry", ["ui", "rpc"])
def test_mixed_dimensions_and_scale_share_one_group_and_preserve_order(edited, monkeypatch, entry):
    scene, session, service, draft, node = edited
    geometry = session.scale_target(node.node_id)
    commands = [
        cmd.SetGeometrySize(geometry.node_id, [0.4, 0.6, 0.8]),
        cmd.SetScale(node.node_id, [2, 0.5, 3]),
        cmd.SetGeometrySize(geometry.node_id, [0.3, 0.8, 0.9]),
    ]
    groups = []

    @contextmanager
    def batch():
        groups.append(True)
        yield

    monkeypatch.setattr(session.adapter, "model_edit_batch", batch)
    if entry == "ui":
        for command in commands:
            assert draft.stage(command).ok
        draft.applying = True
        result = session.apply_model_edits(draft.resolve_commands(session))
        assert result.ok, result.message
        draft.clear()
    else:
        result = transaction(
            service,
            [
                operation("set_geometry_size", node_id=geometry.node_id, size=[0.4, 0.6, 0.8]),
                operation("set_scale", node_id=node.node_id, scale=[2, 0.5, 3]),
                operation("set_geometry_size", node_id=geometry.node_id, size=[0.3, 0.8, 0.9]),
            ],
        )
        assert len(result["results"]) == 3
        assert all(item["document"] == result["document"] for item in result["results"])
    assert groups == [True]
    np.testing.assert_allclose(scene.source.geom_size[0], [0.3, 0.8, 0.9])
    assert session.scale_factors(node.node_id) == (1, 1, 1)
    assert session.submit(cmd.Undo()).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
    assert not session.can_undo
    assert session.submit(cmd.Redo()).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.3, 0.8, 0.9])


@pytest.mark.parametrize("stage", ["group_exit", "history"])
def test_finalization_failure_retains_checkpoint_until_rollback(edited, monkeypatch, stage):
    scene, session, service, _draft, node = edited
    before = service.dispatch("get_scene", {})["document"]
    if stage == "group_exit":

        @contextmanager
        def batch():
            yield
            raise RuntimeError("injected rebuild failure")

        monkeypatch.setattr(session.adapter, "model_edit_batch", batch)
    else:
        capture = session.adapter.capture_edit_state
        calls = 0

        def capture_edit_state():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected history failure")
            return capture()

        monkeypatch.setattr(session.adapter, "capture_edit_state", capture_edit_state)
    with pytest.raises(RpcError, match="injected") as error:
        transaction(service, [operation("set_scale", node_id=node.node_id, scale=[2, 2, 2])])
    assert error.value.code == "command_failed"
    assert not session.editing and not session.can_undo
    assert service.dispatch("get_scene", {})["document"] == before
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])


def test_command_exception_and_recovery_failure_are_both_reported(edited, monkeypatch):
    _scene, session, service, _draft, node = edited
    restore = session.adapter.restore_edit_state

    def fail_scale(*args):
        raise RuntimeError("original scale failure")

    def fail_restore(*args):
        raise RuntimeError("snapshot restore failure")

    monkeypatch.setattr(session.adapter, "set_scale", fail_scale)
    monkeypatch.setattr(session.adapter, "restore_edit_state", fail_restore)
    with pytest.raises(RpcError) as error:
        transaction(service, [operation("set_scale", node_id=node.node_id, scale=[2, 2, 2])])
    assert error.value.code == "rollback_failed"
    assert "original scale failure" in str(error.value)
    assert error.value.details == {
        "index": 0,
        "method": "set_scale",
        "rollback_error": "snapshot restore failure",
    }
    assert session.editing and not session.can_undo
    monkeypatch.setattr(session.adapter, "restore_edit_state", restore)
    assert session.submit(cmd.CancelEditTransaction()).ok


@pytest.mark.parametrize(
    "method",
    [
        "set_scale",
        "set_pose",
        "set_geometry_size",
        "add_scene_object",
        "remove_scene_entity",
        "save_scene",
        "new_scene",
        "open_scene",
        "edit_scene",
        "undo",
        "redo",
    ],
)
def test_pending_draft_blocks_document_writes_before_dispatch(edited, tmp_path, method):
    scene, session, service, draft, node = edited
    geometry = session.scale_target(node.node_id)
    assert draft.stage(cmd.SetScale(node.node_id, [2, 3, 4])).ok
    params = {
        "set_scale": {"node_id": node.node_id, "scale": [5, 5, 5]},
        "set_pose": {"node_id": node.node_id, "position": [9, 8, 7], "rotation": np.eye(3)},
        "set_geometry_size": {"node_id": geometry.node_id, "size": [9, 8, 7]},
        "add_scene_object": {"shape": "box"},
        "remove_scene_entity": {"object_id": node.object_id},
        "save_scene": {"path": str(tmp_path / "unsaved.json")},
        "new_scene": {},
        "open_scene": {"path": str(tmp_path / "missing.json")},
        "edit_scene": {
            "operations": [operation("set_scale", node_id=node.node_id, scale=[5, 5, 5])]
        },
        "undo": {},
        "redo": {},
    }[method]
    spec = service.dispatch("describe_operations", {"name": method})["operations"][0]
    Validator(OPERATIONS["describe_operations"].output_schema).validate(
        {
            "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
            "document": service.dispatch("get_scene", {})["document"],
            "operations": [spec],
        }
    )
    assert spec["writes_document"] and not spec["available"]
    before = service.dispatch("inspect_object", {"object_id": node.object_id})
    for invoke in (
        service.dispatch,
        lambda name, values: apply_session_operation(session, name, values),
    ):
        with pytest.raises(RpcError) as error:
            invoke(method, params)
        assert error.value.code == "pending_edits"
    assert service.dispatch("inspect_object", {"object_id": node.object_id}) == before
    assert before["scale"] == [2, 3, 4] and draft.compatible()
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
    assert not (tmp_path / "unsaved.json").exists()


def test_pending_gesture_blocks_writes_but_keeps_selection_and_visibility(edited):
    _scene, _session, service, draft, node = edited
    draft.begin_transaction("Scale gesture")
    assert not draft.active
    with pytest.raises(RpcError) as error:
        service.dispatch("set_scale", {"node_id": node.node_id, "scale": [2, 2, 2]})
    assert error.value.code == "pending_edits"
    assert service.dispatch("select_node", {"node_id": node.node_id})["ok"]
    assert service.dispatch("set_visible", {"node_id": node.node_id, "visible": False})["ok"]
    draft.end_transaction(cancel=True)
    assert service.dispatch("set_scale", {"node_id": node.node_id, "scale": [2, 2, 2]})["ok"]


def test_no_history_adapter_restores_document_on_failure(edited):
    scene, session, _service, _draft, node = edited
    session.adapter.caps = replace(session.adapter.caps, edit_history=False)
    revision = session.document_revision
    result = session.apply_edits(
        [cmd.SetScale(node.node_id, [2, 2, 2]), cmd.SetScale(2**30, [2, 2, 2])]
    )
    assert not result.result.ok and result.failed_index == 1
    assert not result.rollback_error and not session.editing
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
    assert session.document_revision == revision


def test_no_history_batch_advances_document_once(edited):
    scene, session, _service, _draft, node = edited
    session.adapter.caps = replace(session.adapter.caps, edit_history=False)
    revision = session.document_revision
    result = session.apply_edits(
        [cmd.SetScale(node.node_id, [2, 2, 2]), cmd.SetScale(node.node_id, [3, 3, 3])]
    )
    assert result.result.ok, result.result.message
    assert session.document_revision == revision + 1 and not session.can_undo
    np.testing.assert_allclose(scene.source.geom_size[0], [1.2, 1.8, 2.4])


def test_queued_draft_rechecks_document_before_any_write(edited):
    scene, session, _service, draft, node = edited
    assert draft.stage(cmd.SetScale(node.node_id, [2, 2, 2])).ok
    plan = draft.resolve_commands(session)
    draft.applying = True
    assert session.submit(cmd.SetScale(node.node_id, [3, 3, 3])).ok
    revision = session.document_revision
    result = session.apply_model_edits(plan)
    assert not result.ok and "scene changed" in result.message
    assert session.document_revision == revision and not session.editing
    np.testing.assert_allclose(scene.source.geom_size[0], [0.6, 0.9, 1.2])


def test_rpc_rechecks_document_after_preparing_commands(edited, monkeypatch):
    scene, session, service, _draft, node = edited
    original = OPERATIONS["set_scale"]

    def command(values):
        result = original.command(values)
        assert session.submit(cmd.SetScale(node.node_id, [3, 3, 3])).ok
        return result

    monkeypatch.setitem(OPERATIONS, "set_scale", replace(original, command=command))
    with pytest.raises(RpcError) as error:
        transaction(service, [operation("set_scale", node_id=node.node_id, scale=[2, 2, 2])])
    assert error.value.code == "stale_document"
    np.testing.assert_allclose(scene.source.geom_size[0], [0.6, 0.9, 1.2])
    assert not session.editing


@pytest.mark.parametrize("raises", [False, True])
def test_gesture_finalization_keeps_original_and_recovery_failures(edited, monkeypatch, raises):
    _scene, session, _service, _draft, _node = edited
    assert session.submit(cmd.BeginEditTransaction("Failing gesture")).ok
    original = session.submit(cmd.SetScale(2**30, [2, 2, 2]))
    assert not original.ok
    restore = session.adapter.restore_edit_state

    def reject_restore(*args):
        if raises:
            raise RuntimeError("injected restore failure")
        return False

    monkeypatch.setattr(session.adapter, "restore_edit_state", reject_restore)
    result = session.submit(cmd.EndEditTransaction())
    assert not result.ok and original.message in result.message
    assert "rollback failed" in result.message and session.editing
    monkeypatch.setattr(session.adapter, "restore_edit_state", restore)
    assert session.submit(cmd.CancelEditTransaction()).ok


def test_failed_history_accounting_preserves_redo_branch(edited, monkeypatch):
    from mojive.session import history

    scene, session, service, _draft, node = edited
    assert session.submit(cmd.SetScale(node.node_id, [2, 2, 2])).ok
    assert session.submit(cmd.Undo()).ok and session.can_redo
    revision = session.document_revision

    def fail_accounting(*args):
        raise RuntimeError("injected history accounting failure")

    monkeypatch.setattr(history, "_snapshot_allocations", fail_accounting)
    with pytest.raises(RpcError, match="injected history accounting failure"):
        transaction(service, [operation("set_scale", node_id=node.node_id, scale=[3, 3, 3])])
    assert not session.can_undo and session.can_redo and not session.editing
    assert session.document_revision == revision
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
    assert session.submit(cmd.Redo()).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.4, 0.6, 0.8])


def test_no_history_recovery_failure_keeps_checkpoint_for_retry(edited, monkeypatch):
    scene, session, _service, _draft, node = edited
    session.adapter.caps = replace(session.adapter.caps, edit_history=False)
    restore = session.adapter.restore_edit_state
    monkeypatch.setattr(session.adapter, "restore_edit_state", lambda state: False)
    result = session.apply_edits(
        [cmd.SetScale(node.node_id, [2, 2, 2]), cmd.SetScale(2**30, [2, 2, 2])]
    )
    assert not result.result.ok and result.rollback_error and session.editing
    assert not session.apply_edits([cmd.SetScale(node.node_id, [3, 3, 3])]).result.ok
    monkeypatch.setattr(session.adapter, "restore_edit_state", restore)
    assert session.submit(cmd.CancelEditTransaction()).ok
    assert not session.editing and session.document_revision == 0
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])


@pytest.mark.parametrize("command", [cmd.EndEditTransaction(), cmd.NewScene(), cmd.Play()])
def test_batch_rejects_lifecycle_commands_before_writing(edited, command):
    scene, session, _service, _draft, node = edited
    result = session.apply_edits([cmd.SetScale(node.node_id, [2, 2, 2]), command])
    assert not result.result.ok and result.failed_index == 1
    assert not result.results and not session.editing and not session.can_undo
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])


def test_remote_write_cannot_join_an_owned_live_gesture(edited):
    scene, session, service, draft, node = edited
    assert not draft.active
    assert session.submit(cmd.BeginEditTransaction("Local resize")).ok
    assert session.submit(cmd.SetScale(node.node_id, [2, 2, 2])).ok
    with pytest.raises(RpcError) as error:
        service.dispatch("set_scale", {"node_id": node.node_id, "scale": [3, 3, 3]})
    assert error.value.code == "pending_edits"
    assert session.editing and not draft.active
    np.testing.assert_allclose(scene.source.geom_size[0], [0.4, 0.6, 0.8])
    assert session.submit(cmd.CancelEditTransaction()).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
