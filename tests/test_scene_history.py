"""Programmatic handles and immutable resources across authored edit history."""

from dataclasses import replace

import numpy as np
import pytest

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.session import Session
from mojive.types import CameraView, Light, MeshData, MeshShape, TextureData, TextureType


def triangle():
    return MeshData(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.tile([0, 0, 1], (3, 1)).astype(np.float32),
        np.zeros((3, 2), np.float32),
        np.arange(3, dtype=np.uint32),
    )


def test_undo_redo_preserves_programmatic_scene_and_handles():
    scene = Scene()
    box = scene.box()
    light = scene.add_light("key", Light())
    camera_id = scene.add_camera("shot", CameraView())
    session = Session(StaticSceneAdapter(scene))
    session.submit(cmd.RenameSceneEntity(box.object_id, "renamed"))
    for command in (cmd.Undo(), cmd.Redo(), cmd.Undo()):
        assert session.submit(command).ok
        assert session.adapter.scene is scene
        box.set_pose((10, 2, 3))
        light.set(replace(light.value, intensity=2))
        assert scene.set_camera(camera_id, CameraView(eye=np.ones(3)))
        frame = session.tick(FrameNeeds())
        np.testing.assert_allclose(frame.geom_xpos[0], [10, 2, 3])
        assert frame.lights.lights[0].intensity == 2
        np.testing.assert_allclose(frame.cameras[0].eye, 1)


def test_undo_does_not_reuse_removed_entity_ids():
    scene = Scene()
    session = Session(StaticSceneAdapter(scene))
    created = session.submit(cmd.AddSceneObject(shape=MeshShape.BOX))
    assert created.ok
    handle = scene.object("object")
    assert session.submit(cmd.Undo()).ok
    with pytest.raises(KeyError):
        handle.set_pose((1, 0, 0))
    another = scene.box()
    assert another.object_id != handle.object_id
    with pytest.raises(KeyError):
        handle.set_pose((1, 0, 0))


def test_handles_from_different_scene_owners_are_distinct():
    a, b = Scene(), Scene()
    assert a.box() != b.box()
    assert a.add_light("key", Light()) != b.add_light("key", Light())


def test_reload_preserves_handles_for_the_same_document(tmp_path):
    scene = Scene()
    box = scene.box()
    adapter = StaticSceneAdapter(scene)
    adapter.save_scene(tmp_path / "scene.mojive.json")
    box.set_pose((5, 0, 0))
    adapter.reload()
    assert adapter.scene is scene
    np.testing.assert_allclose(scene.frame.geom_xpos[0], 0)
    box.set_pose((2, 0, 0))
    np.testing.assert_allclose(adapter.frame(FrameNeeds()).geom_xpos[0], [2, 0, 0])


def test_workspace_history_preserves_authored_scene_owner():
    scene = Scene()
    box = scene.box()
    adapter = WorkspaceAdapter(ToyPhysicsAdapter(), scene)
    before = adapter.capture_edit_state()
    box.set_pose((5, 0, 0))
    assert adapter.restore_edit_state(before)
    assert adapter.scene is scene
    box.set_pose((2, 0, 0))
    np.testing.assert_allclose(adapter.scene.frame.geom_xpos[0], [2, 0, 0])


def test_workspace_history_restores_primary_pose_color_and_simulation_state():
    primary = ToyPhysicsAdapter()
    session = Session(WorkspaceAdapter(primary, Scene()))
    session.submit(cmd.Pause())
    body = next(node for node in session.nodes if node.name == "red ball")
    geom = next(node for node in session.nodes if node.parent == body.node_id)
    position = primary._positions.copy()
    velocity = primary._velocities.copy()
    colors = session.source.geom_rgba.copy()
    try:
        with session.edit("Edit primary"):
            assert session.submit(cmd.SetPose(body.node_id, (2, 3, 4), np.eye(3))).ok
            assert session.submit(cmd.SetGeometryColor(geom.node_id, (0, 1, 0, 1))).ok
        assert session.submit(cmd.Undo()).ok
        np.testing.assert_allclose(primary._positions, position)
        np.testing.assert_allclose(primary._velocities, velocity)
        np.testing.assert_allclose(session.source.geom_rgba, colors)
        assert session.submit(cmd.Redo()).ok
        np.testing.assert_allclose(primary._positions[0], [2, 3, 4])
        assert np.all(session.source.geom_rgba[1] == [0, 1, 0, 1])
        primary.scene.add_camera("rebuild", CameraView())
        session.tick(FrameNeeds(), wall_dt=0)
        assert np.all(session.source.geom_rgba[1] == [0, 1, 0, 1])
    finally:
        session.release()


def test_workspace_does_not_advertise_history_without_restorable_primary():
    primary = ToyPhysicsAdapter()
    primary.capture_edit_state = lambda: None
    session = Session(WorkspaceAdapter(primary, Scene()))
    try:
        assert not session.adapter.caps.edit_history
        assert session.adapter.capture_edit_state() is None
        result = session.submit(cmd.BeginEditTransaction())
        assert not result.ok and not session.editing
    finally:
        session.release()


def test_redo_retains_session_color_override_after_structure_rebuild(monkeypatch):
    scene = Scene()
    scene.box()
    adapter = StaticSceneAdapter(scene)
    monkeypatch.setattr(adapter, "set_geometry_color", lambda *_args: False)
    session = Session(adapter)
    geom = next(node for node in session.nodes if node.type.value == "geom")
    try:
        assert session.submit(cmd.SetGeometryColor(geom.node_id, (0, 1, 0, 1))).ok
        assert session.submit(cmd.Undo()).ok
        assert session.submit(cmd.Redo()).ok
        scene.add_camera("rebuild", CameraView())
        session.tick(FrameNeeds(), wall_dt=0)
        assert np.all(session.source.geom_rgba[0] == [0, 1, 0, 1])
    finally:
        session.release()


def test_scene_assets_are_owned_immutable_values_shared_by_history():
    scene = Scene()
    mesh = triangle()
    obj = scene.mesh(mesh)
    pixels = np.zeros((512, 512, 4), np.uint8)
    scene.add_texture(TextureData("texture", TextureType.TWO_D, pixels))
    session = Session(StaticSceneAdapter(scene))
    stored_mesh = scene.source.meshes[obj.mesh_key]
    stored_texture = scene.textures["texture"]
    mesh.positions[:] = 10
    pixels[:] = 255
    np.testing.assert_allclose(stored_mesh.positions[0], 0)
    assert not stored_texture.pixels.any()
    for array in (stored_mesh.positions, stored_mesh.indices, stored_texture.pixels):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    for index in range(20):
        assert session.submit(cmd.RenameSceneEntity(obj.object_id, f"mesh-{index}")).ok
    for record in session._undo_stack:
        for snapshot in (record.before.adapter_state, record.after.adapter_state):
            assert snapshot.textures["texture"] is stored_texture
            assert snapshot.source.meshes[obj.mesh_key] is stored_mesh


def test_resource_replacements_and_restoration_do_not_mutate_saved_resources():
    scene = Scene()
    obj = scene.mesh(triangle())
    scene.add_texture(TextureData("surface", TextureType.TWO_D, np.zeros((4, 4, 3), np.uint8)))
    before = scene.clone()
    changed_mesh = triangle()
    changed_mesh.positions[:] += 4
    scene.replace_mesh(obj.mesh_key, changed_mesh)
    scene.add_texture(TextureData("surface", TextureType.TWO_D, np.full((4, 4, 3), 127, np.uint8)))
    after = scene.clone()
    scene.restore(before)
    np.testing.assert_allclose(scene.source.meshes[obj.mesh_key].positions[0], 0)
    assert not scene.textures["surface"].pixels.any()
    scene.restore(after)
    np.testing.assert_allclose(scene.source.meshes[obj.mesh_key].positions[0], 4)
    assert (scene.textures["surface"].pixels == 127).all()


def test_loaded_scene_snapshots_share_resources(tmp_path):
    scene = Scene()
    obj = scene.mesh(triangle())
    scene.add_texture(TextureData("surface", TextureType.TWO_D, np.zeros((4, 4, 3), np.uint8)))
    loaded = Scene.load(scene.save(tmp_path / "scene.mojive.json"))
    a, b = loaded.clone(), loaded.clone()
    assert a.source.meshes[obj.mesh_key] is b.source.meshes[obj.mesh_key]
    assert a.textures["surface"] is b.textures["surface"]


def test_edit_context_commits_one_record_and_rolls_back_exceptions():
    scene = Scene()
    box = scene.box()
    session = Session(StaticSceneAdapter(scene))
    with session.edit("Rename twice"):
        session.submit(cmd.RenameSceneEntity(box.object_id, "first"))
        session.submit(cmd.RenameSceneEntity(box.object_id, "second"))
    assert len(session._undo_stack) == 1
    retained = session.history_bytes
    with pytest.raises(ValueError, match="stop edit"), session.edit("Interrupted rename"):
        session.submit(cmd.RenameSceneEntity(box.object_id, "temporary"))
        raise ValueError("stop edit")
    assert scene.object("second") == box
    assert not session.editing
    assert session.history_bytes == retained
    assert session.submit(cmd.Undo()).ok
    assert scene.object("object") == box


def test_failed_command_rolls_back_the_entire_edit():
    scene = Scene()
    box = scene.box()
    session = Session(StaticSceneAdapter(scene))
    with pytest.raises(RuntimeError, match="rolled back"), session.edit("Invalid batch"):
        session.submit(cmd.RenameSceneEntity(box.object_id, "temporary"))
        assert not session.submit(cmd.RenameSceneEntity(999, "missing")).ok
    assert scene.object("object") == box
    assert not session.can_undo
    assert not session.dirty
    assert not session.editing


def test_cancelled_edit_preserves_the_redo_branch():
    scene = Scene()
    box = scene.box()
    session = Session(StaticSceneAdapter(scene))
    session.submit(cmd.RenameSceneEntity(box.object_id, "renamed"))
    session.submit(cmd.Undo())
    retained = session.history_bytes
    session.submit(cmd.BeginEditTransaction())
    session.submit(cmd.RenameSceneEntity(box.object_id, "temporary"))
    assert session.submit(cmd.CancelEditTransaction()).ok
    assert session.can_redo
    assert session.history_bytes == retained
    assert session.submit(cmd.Redo()).ok
    assert scene.object("renamed") == box


def test_edit_history_prunes_contiguous_records_and_counts_redo_memory():
    scene = Scene()
    box = scene.box()
    session = Session(StaticSceneAdapter(scene), history_record_limit=2)
    for name in ("one", "two", "three"):
        session.submit(cmd.RenameSceneEntity(box.object_id, name))
    assert len(session._undo_stack) == 2
    retained = session.history_bytes
    for name in ("two", "one"):
        assert session.submit(cmd.Undo()).ok
        assert scene.object(name) == box
        assert session.history_bytes == retained
    assert not session.can_undo
    session.submit(cmd.RenameSceneEntity(box.object_id, "new branch"))
    assert not session.can_redo
    assert 0 < session.history_bytes < retained


def test_oversized_edit_clears_history_without_reverting_the_applied_edit():
    scene = Scene()
    box = scene.box()
    session = Session(StaticSceneAdapter(scene), history_byte_limit=1024)
    result = session.submit(cmd.RenameSceneEntity(box.object_id, "renamed"))
    assert result.ok
    assert "memory budget" in result.message
    assert scene.object("renamed") == box
    assert not session.can_undo and not session.can_redo
    assert session.history_bytes == 0
    assert session.dirty


def test_shared_texture_storage_is_charged_once_to_the_history_budget():
    scene = Scene()
    box = scene.box()
    pixels = np.zeros((1024, 1024, 4), np.uint8)
    scene.add_texture(TextureData("surface", TextureType.TWO_D, pixels))
    session = Session(StaticSceneAdapter(scene), history_byte_limit=8 * 1024 * 1024)
    for index in range(10):
        session.submit(cmd.RenameSceneEntity(box.object_id, f"name-{index}"))
    assert len(session._undo_stack) == 10
    assert pixels.nbytes <= session.history_bytes < 8 * 1024 * 1024


@pytest.mark.parametrize("operation", ["new", "reload", "save", "open", "load"])
def test_document_operations_cannot_escape_an_active_edit(tmp_path, operation):
    scene = Scene()
    box = scene.box()
    session = Session(StaticSceneAdapter(scene))
    path = tmp_path / "scene.mojive.json"
    command = {
        "new": cmd.NewScene(),
        "reload": cmd.Reload(),
        "save": cmd.SaveScene(path),
        "open": cmd.OpenScene(path),
        "load": cmd.LoadAsset(path),
    }[operation]
    with pytest.raises(RuntimeError, match="rolled back"), session.edit("Unfinished"):
        session.submit(cmd.RenameSceneEntity(box.object_id, "temporary"))
        assert not session.submit(command).ok
    assert scene.object("object") == box
    assert not path.exists()
    assert not session.dirty
