"""Document replacement must reject unsupported sources and roll back partial loads."""

import json
from dataclasses import replace

import pytest

from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene import Scene
from mojive.scene.io import scene_to_document
from mojive.scene.workspace import FORMAT, VERSION

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("source_kind", ["root", "attached"])
def test_rejected_workspace_model_source_restores_both_owners_and_document_metadata(
    tmp_path, source_kind
):
    class RejectingSource(StaticSceneAdapter):
        caps = replace(StaticSceneAdapter.caps, features=(("mujoco.mjcf", 1),))

        def set_scene_model_xml(self, model_id, xml):
            return False

        def add_scene_model(self, path, position, rotation):
            self.scene.box(name="partially loaded")
            return 1

    primary_scene, authored = Scene(), Scene()
    primary_scene.box(name="original model")
    authored.box(name="original entity")
    primary = RejectingSource(primary_scene)
    primary_path = tmp_path / "primary.mojive.json"
    primary.save_scene(primary_path)
    workspace = WorkspaceAdapter(primary, authored)
    workspace.set_resource_roots((tmp_path,))
    original_path = tmp_path / "original.mojive.json"
    workspace._path = original_path
    document = {
        "format": FORMAT,
        "version": VERSION,
        "models": [],
        "scene": scene_to_document(authored),
    }
    if source_kind == "root":
        document["root_mjcf"] = "<mujoco/>"
    else:
        model_path = tmp_path / "model.xml"
        model_path.write_text("<mujoco/>")
        document["models"] = [{"path": "model.xml", "mjcf": "<mujoco/>"}]
    target = tmp_path / "rejected.mojive.json"
    target.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="Failed to restore"):
        workspace.open_scene(target)
    assert workspace.scene is authored
    assert "original entity" in {node.name for node in workspace.scene.source.nodes}
    assert "original model" in {node.name for node in workspace.primary.scene_source().nodes}
    assert workspace.resource_roots == (tmp_path,)
    assert workspace._path == original_path
    assert primary.scene is primary_scene
    assert primary._path == primary_path


def test_workspace_source_capability_is_checked_before_clearing_primary(tmp_path):
    class ReadOnlySource(StaticSceneAdapter):
        cleared = False

        def new_scene(self):
            self.cleared = True
            super().new_scene()

    primary = ReadOnlySource(Scene())
    workspace = WorkspaceAdapter(primary)
    path = tmp_path / "source.mojive.json"
    document = {
        "format": FORMAT,
        "version": VERSION,
        "models": [],
        "scene": scene_to_document(Scene()),
    }
    document["root_mjcf"] = "<mujoco/>"
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="does not support workspace MJCF"):
        workspace.open_scene(path)
    assert not primary.cleared


def test_workspace_refuses_replacement_without_rollback_support(tmp_path):
    class NoHistory(StaticSceneAdapter):
        def capture_edit_state(self):
            return None

        def new_scene(self):
            pytest.fail("primary must not be cleared without rollback support")

    workspace = WorkspaceAdapter(NoHistory(Scene()))
    with pytest.raises(RuntimeError, match="document checkpoint"):
        workspace.open_scene(tmp_path / "document.mojive.json")
