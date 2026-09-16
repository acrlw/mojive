"""Partial document support agrees across commands, discovery and composition."""

from dataclasses import replace

import pytest

from mojive import commands as cmd
from mojive.adapters.base import AdapterCaps
from mojive.adapters.conformance import check_adapter
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene import Scene
from mojive.session import Session


def test_legacy_file_capability_and_explicit_operation_overrides():
    legacy = AdapterCaps(scene_files=True)
    assert all(legacy.supports(name) for name in ("scene_new", "scene_open", "scene_save"))
    partial = replace(legacy, scene_open=False)
    assert partial.supports("scene_save") and not partial.supports("scene_open")
    assert not partial.supports("scene_save", 2)


def test_lightweight_workspace_rejects_files_before_writes_but_keeps_edit_history(tmp_path):
    adapter = WorkspaceAdapter(ToyPhysicsAdapter())
    session = Session(adapter)
    try:
        assert check_adapter(adapter).ok
        assert not any(
            adapter.caps.supports(name) for name in ("scene_new", "scene_open", "scene_save")
        )
        source = session.source
        document = session.document_id
        for command in (
            cmd.NewScene(),
            cmd.OpenScene(tmp_path / "scene.mojive.json"),
            cmd.SaveScene(tmp_path / "scene.mojive.json"),
        ):
            result = session.submit(command)
            assert not result.ok and "does not support" in result.message
        assert session.source is source and session.document_id == document
        assert session.submit(cmd.AddSceneObject("box", "authored box")).ok
        assert session.can_undo
        assert session.submit(cmd.Undo()).ok
        assert "authored box" not in {node.name for node in session.nodes}
    finally:
        session.release()


def test_save_only_adapter_does_not_need_open_or_new(tmp_path):
    adapter = StaticSceneAdapter(Scene())
    adapter.caps = replace(
        adapter.caps, scene_files=False, scene_new=False, scene_open=False, scene_save=True
    )
    session = Session(adapter)
    try:
        assert check_adapter(adapter).ok
        path = tmp_path / "saved.mojive.json"
        assert session.submit(cmd.SaveScene(path)).ok
        assert path.exists()
        assert not session.submit(cmd.OpenScene(path)).ok
        assert not session.submit(cmd.NewScene()).ok
    finally:
        session.release()


def test_new_scene_reports_backend_failure_without_clearing_session():
    class Rejecting(StaticSceneAdapter):
        def new_scene(self):
            raise RuntimeError("document replacement failed")

    scene = Scene()
    scene.box(name="original")
    session = Session(Rejecting(scene))
    try:
        document = session.document_id
        result = session.submit(cmd.NewScene())
        assert not result.ok and result.message == "document replacement failed"
        assert session.document_id == document
        assert "original" in {node.name for node in session.nodes}
    finally:
        session.release()


def test_workspace_capabilities_do_not_copy_models_to_probe_support():
    class Checkpoints(StaticSceneAdapter):
        def capture_edit_state(self):
            pytest.fail("Capability discovery must not clone a document")

    adapter = WorkspaceAdapter(Checkpoints(Scene()))
    try:
        adapter.scene_source()
        adapter.primary.scene.box()
        adapter.scene_source()
        assert adapter.caps.edit_history and adapter.caps.supports("scene_open")
    finally:
        adapter.release()


def test_workspace_does_not_save_an_unrepresented_primary_scene(tmp_path):
    primary_scene = Scene()
    primary_scene.box(name="must not disappear")
    primary = StaticSceneAdapter(primary_scene)
    adapter = WorkspaceAdapter(primary)
    session = Session(adapter)
    target = tmp_path / "existing.mojive.json"
    target.write_text("existing document")
    try:
        assert primary.caps.supports("scene_save")
        assert not adapter.caps.supports("scene_save")
        assert not session.submit(cmd.SaveScene(target)).ok
        assert target.read_text() == "existing document"
        assert "must not disappear" in {node.name for node in session.nodes}
    finally:
        session.release()


@pytest.mark.physics
def test_workspace_does_not_save_compiled_only_or_nested_primary(tmp_path):
    import mujoco

    from mojive.adapters.mujoco import MuJoCoAdapter

    primary = MuJoCoAdapter()
    primary.load_model(
        mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
    )
    compiled = WorkspaceAdapter(primary)
    try:
        assert not compiled.caps.supports("scene_save")
    finally:
        compiled.release()
    primary = MuJoCoAdapter()
    primary.new_scene()
    inner = WorkspaceAdapter(primary)
    inner.scene.box(name="inner authored entity")
    outer = WorkspaceAdapter(inner)
    try:
        assert inner.caps.supports("scene_save")
        assert not outer.caps.supports("scene_save")
        with pytest.raises(RuntimeError, match="scene_save"):
            outer.save_scene(tmp_path / "must-not-be-created.mojive.json")
        assert not (tmp_path / "must-not-be-created.mojive.json").exists()
    finally:
        outer.release()
