from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.mujoco_adapter import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.model_edits import ModelEditDraft, model_edit_scope
from mojive.session import Session

pytestmark = pytest.mark.physics


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "pending.xml"
    path.write_text(
        '<mujoco><worldbody><body name="arm"><joint name="hinge"/><geom name="box" type="box" size=".1 .2 .3"/></body></worldbody></mujoco>'
    )
    primary = MuJoCoAdapter(path)
    session = Session(WorkspaceAdapter(primary), path)
    session.submit(cmd.Pause())
    draft = ModelEditDraft(session)
    yield session, primary, draft


def test_staging_coalesces_without_physics_writes_and_apply_has_one_undo(scene, monkeypatch):
    session, adapter, draft = scene
    box = next(n for n in session.nodes if n.name == "box")
    original = adapter._m
    compiles = []
    compile_model = adapter._compile_composed_model

    def compile_counted():
        if not adapter._model_edit_batch_depth:
            compiles.append(True)
        return compile_model()

    monkeypatch.setattr(adapter, "_compile_composed_model", compile_counted)
    for value in np.linspace(0.1, 0.5, 60):
        assert draft.stage(cmd.SetGeometrySize(box.node_id, np.full(3, value))).ok
    assert draft.stage(cmd.RenameModelElement(box.node_id, "edited_box")).ok
    assert len(draft.commands) == 2
    assert compiles == [] and adapter._m is original
    np.testing.assert_allclose(original.geom_size[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(session.source.geom_size[0], [0.5, 0.5, 0.5])
    assert session.node(box.node_id).name == "edited_box"
    adapter._d.qpos[0], adapter._d.qvel[0] = 0.37, 0.2
    draft.applying = True
    result = session.apply_model_edits(draft.commands)
    assert result.ok, result.message
    draft.clear()
    assert len(compiles) == 1
    np.testing.assert_allclose(adapter._m.geom_size[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(adapter._d.qpos, [0.37])
    np.testing.assert_allclose(adapter._d.qvel, [0.2])
    assert session.submit(cmd.Undo()).ok
    assert next(n for n in session.nodes if n.geom_index == 0).name == "box"
    np.testing.assert_allclose(adapter._m.geom_size[0], [0.1, 0.2, 0.3])
    assert not session.submit(cmd.Undo()).ok


def test_failed_compile_rolls_back_and_can_be_corrected(scene):
    session, adapter, draft = scene
    box = next(n for n in session.nodes if n.name == "box")
    assert draft.stage(cmd.SetGeometryShape(box.node_id, "mesh", "missing")).ok
    draft.applying = True
    result = session.apply_model_edits(draft.commands)
    assert not result.ok
    draft.rebase_after_failure()
    assert draft.active and draft.compatible()
    np.testing.assert_allclose(adapter._m.geom_size[0], [0.1, 0.2, 0.3])
    assert draft.stage(cmd.SetGeometryShape(box.node_id, "sphere")).ok
    draft.applying = True
    result = session.apply_model_edits(draft.commands)
    assert result.ok, result.message
    draft.clear()
    assert session.geometry_shape_properties(box.node_id).type == "sphere"


def test_cancel_gesture_only_removes_its_pending_edits(scene):
    session, _adapter, draft = scene
    box = next(n for n in session.nodes if n.name == "box")
    draft.stage(cmd.RenameModelElement(box.node_id, "kept"))
    draft.begin_transaction("Resize")
    draft.stage(cmd.SetGeometrySize(box.node_id, np.ones(3)))
    draft.end_transaction(cancel=True)
    assert len(draft.commands) == 1
    assert session.node(box.node_id).name == "kept"
    np.testing.assert_allclose(session.source.geom_size[0], [0.1, 0.2, 0.3])
    draft.clear()
    assert session.node(box.node_id).name == "box"


def test_command_scope_does_not_change_rpc_or_worker_api(scene):
    session, adapter, draft = scene
    box = next(n for n in session.nodes if n.name == "box")
    command = cmd.RenameModelElement(box.node_id, "direct")
    with model_edit_scope(session, draft.stage):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(session.submit, command).result().ok
        assert not draft.active
        with model_edit_scope(session, None):
            assert session.submit(replace(command, name="rpc")).ok
        assert not draft.active
        assert session.submit(replace(command, name="pending")).ok
    assert draft.active
    assert adapter._m.geom(0).name == "rpc"


def test_model_placement_preserves_transformed_free_joint_state(scene, tmp_path):
    session, adapter, draft = scene
    path = tmp_path / "free.xml"
    path.write_text(
        '<mujoco><worldbody><body name="floating"><freejoint/><geom type="sphere" size=".2"/></body></worldbody></mujoco>'
    )
    added = session.submit(cmd.AddSceneModel(path))
    assert added.ok, added.message
    model_id = session.scene_models[-1].model_id
    position = np.array([2.0, 3.0, 4.0])
    assert draft.stage(cmd.SetSceneModelTransform(model_id, position, np.eye(3))).ok
    draft.applying = True
    result = session.apply_model_edits(draft.commands)
    assert result.ok, result.message
    draft.clear()
    np.testing.assert_allclose(adapter._d.qpos[-7:-4], position)
    np.testing.assert_allclose(session.scene_models[-1].position, position)


def test_final_compile_failure_retains_draft_and_original_document(scene, monkeypatch):
    session, adapter, draft = scene
    node = next(n for n in session.nodes if n.name == "box")
    draft.stage(cmd.RenameModelElement(node.node_id, "changed"))
    compile_model = adapter._compile_composed_model
    failed = False

    def fail_once():
        nonlocal failed
        if not adapter._model_edit_batch_depth and not failed:
            failed = True
            raise RuntimeError("Injected compile failure")
        return compile_model()

    monkeypatch.setattr(adapter, "_compile_composed_model", fail_once)
    draft.applying = True
    assert not session.apply_model_edits(draft.commands).ok
    draft.rebase_after_failure()
    assert draft.active and draft.compatible()
    assert adapter._m.geom(0).name == "box"
    draft.applying = True
    assert session.apply_model_edits(draft.commands).ok
    draft.clear()
    assert adapter._m.geom(0).name == "changed"


def test_source_reordering_migrates_state_by_joint_identity(scene):
    session, adapter, draft = scene
    xml = '<mujoco><worldbody><body><joint name="a"/><joint name="b" axis="1 0 0"/><geom type="sphere" size=".2"/></body></worldbody></mujoco>'
    assert session.submit(cmd.SetModelSource(0, xml)).ok
    adapter._d.qpos[:] = [0.2, 0.7]
    reordered = xml.replace(
        '<joint name="a"/><joint name="b" axis="1 0 0"/>',
        '<joint name="b" axis="1 0 0"/><joint name="a"/>',
    )
    assert draft.stage(cmd.SetModelSource(0, reordered)).ok
    draft.applying = True
    assert session.apply_model_edits(draft.commands).ok
    draft.clear()
    np.testing.assert_allclose(adapter._d.qpos, [0.7, 0.2])
