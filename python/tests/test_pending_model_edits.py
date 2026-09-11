from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.base import NodeType
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


def test_pending_model_keyframes_reserve_generated_names_until_apply(scene):
    from mojive.ui.panels.keyframes import unique_keyframe_name

    session, _adapter, draft = scene
    first = unique_keyframe_name(draft.model_keyframe_names(0))
    assert first == "key1"
    assert draft.stage(cmd.AddModelKeyframe(0, first)).ok

    second = unique_keyframe_name(draft.model_keyframe_names(0))
    assert second == "key2"
    assert draft.stage(cmd.AddModelKeyframe(0, second)).ok
    duplicate = draft.stage(cmd.AddModelKeyframe(0, first))
    assert not duplicate.ok and "already exists" in duplicate.message

    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    assert [key.name for key in session.keyframes] == ["key1", "key2"]


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


def _creation_handler(session, draft):
    """Route creation, styling and selection through one pending draft."""

    def handler(command):
        if isinstance(command, cmd.BeginEditTransaction):
            draft.begin_transaction(command.label)
            return cmd.CommandResult.good()
        if isinstance(command, cmd.EndEditTransaction):
            draft.end_transaction()
            return cmd.CommandResult.good()
        if isinstance(command, (cmd.AddModelElement, cmd.SetGeometryColor, cmd.SelectNode)):
            return draft.stage(command)
        return None

    return handler


def test_created_element_binds_its_style_and_selection_on_apply(scene):
    session, adapter, draft = scene
    world = next(node for node in session.nodes if node.type is NodeType.WORLD and node.parent < 0)
    color = (0.25, 0.5, 0.75, 1.0)
    with model_edit_scope(session, _creation_handler(session, draft)):
        assert session.submit(cmd.BeginEditTransaction("Create plane")).ok
        created = session.submit(cmd.AddModelElement(world.node_id, "geom:plane", "plane"))
        assert created.ok, created.message
        assert created.entity_id >= 0 and created.entity_key
        assert session.submit(cmd.SetGeometryColor(-1, color, created.entity_key)).ok
        assert session.submit(cmd.SelectNode(-1, created.entity_key)).ok
        assert session.submit(cmd.EndEditTransaction()).ok
    assert session.node(created.entity_id).name == "plane"
    assert adapter._m.ngeom == 1

    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()

    plane = next(node for node in session.nodes if node.name == "plane")
    assert session.selected_node is plane
    np.testing.assert_allclose(
        session.source.geom_rgba[session.source.geom_node == plane.node_id], [color]
    )
    np.testing.assert_allclose(adapter._m.geom_rgba[plane.geom_index], color)
    assert session.submit(cmd.Undo()).ok
    assert not any(node.name == "plane" for node in session.nodes)


def test_external_topology_change_does_not_rebind_a_stale_draft(scene):
    session, adapter, draft = scene
    box = next(node for node in session.nodes if node.name == "box")
    assert draft.stage(cmd.SetGeometrySize(box.node_id, np.ones(3))).ok
    assert session.submit(cmd.RenameModelElement(box.node_id, "external")).ok
    assert not draft.compatible()
    app = _editor(session, draft)
    app._apply_model_edits_requested = True
    app._start_pending_model_edits()
    assert draft.active and not draft.applying and not draft.compatible()
    assert "The scene changed; discard" in session.last_message
    assert adapter.model.geom(0).name == "external"
    np.testing.assert_allclose(adapter.model.geom_size[0], (0.1, 0.2, 0.3))
    draft.clear()
    assert session.node(box.node_id).name == "external"


@pytest.mark.parametrize("live", [False, True])
def test_empty_editor_creates_plane_with_one_undo_and_redo(live):
    from mojive.types import MeshShape

    adapter = MuJoCoAdapter()
    adapter.new_scene()
    session = Session(WorkspaceAdapter(adapter))
    assert session.submit(cmd.Pause()).ok
    draft = ModelEditDraft(session)
    app = _editor(session, draft)
    app.live_model_updates = live
    original = adapter.model
    with model_edit_scope(session, app._intercept_model_edit):
        app._add_scene_object(MeshShape.PLANE, "plane")
    assert not session.editing and draft._checkpoint is None
    assert session.selected_node.name == "plane"
    before = _geometry_state(session, session.selected_node)
    if not live:
        assert draft.active and draft.visible and draft.compatible()
        assert adapter.model is original and adapter.model.ngeom == 0
        draft.applying = True
        result = session.apply_model_edits(draft.resolve_commands(session))
        assert result.ok, result.message
        app._finish_pending_model_edits(result)
    assert not draft.active and adapter.model.ngeom == 1
    np.testing.assert_allclose(_geometry_state(session, session.selected_node), before)
    assert session.submit(cmd.Undo()).ok
    assert adapter.model.ngeom == 0 and not draft.active
    assert not session.submit(cmd.Undo()).ok
    assert session.submit(cmd.Redo()).ok
    assert adapter.model.ngeom == 1


def _geometry_state(session, node):
    from mojive.bounds import _instance_world_corners

    source = session.source
    rows = np.flatnonzero(source.geom_node == node.node_id)
    return np.concatenate([_instance_world_corners(source, session.frame, int(i)) for i in rows])


def _editor(session, draft):
    from types import SimpleNamespace

    from mojive.types import CameraView
    from mojive.ui.app import ViewerApp

    app = ViewerApp.__new__(ViewerApp)
    app.session, app.model_edits, app.live_model_updates = session, draft, False
    app.localizer = SimpleNamespace(text=str)
    app._camera_view = lambda: CameraView()
    return app


def test_duplicate_follow_ups_bind_to_the_returned_entity(scene):
    session, adapter, draft = scene
    box = next(node for node in session.nodes if node.name == "box")
    color = (0.2, 0.6, 0.9, 1)
    created = draft.stage(cmd.DuplicateModelElement(box.node_id))
    assert created.ok
    assert draft.stage(cmd.SetGeometryColor(-1, color, created.entity_key)).ok
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    duplicate = next(node for node in session.nodes if node.name == "box_copy")
    np.testing.assert_allclose(adapter.model.geom_rgba[duplicate.geom_index], color)


def test_creation_follow_ups_share_one_rebuild_and_preserve_renamed_selection(scene, monkeypatch):
    session, adapter, draft = scene
    app = _editor(session, draft)
    compiles = []
    compile_model = adapter._compile_composed_model

    def compile_counted():
        if not adapter._model_edit_batch_depth:
            compiles.append(True)
        return compile_model()

    monkeypatch.setattr(adapter, "_compile_composed_model", compile_counted)
    with model_edit_scope(session, app._intercept_model_edit):
        app._add_model_primitive("plane", "plane")
        plane = session.selected_node
        assert session.submit(cmd.SetGeometrySize(plane.node_id, np.array((2, 3, 0.02)))).ok
        assert session.submit(cmd.RenameModelElement(plane.node_id, "ground")).ok
    assert not compiles
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    assert session.selected_node.name == "ground"
    assert len(compiles) == 2


def test_failed_creation_follow_up_rolls_back_and_retains_the_original_error(scene):
    session, adapter, draft = scene
    world = next(node for node in session.nodes if node.type is NodeType.WORLD)
    created = draft.stage(cmd.AddModelElement(world.node_id, "geom:plane", "ground"))
    assert created.ok
    assert draft.stage(cmd.SetGeometryShape(created.entity_id, "mesh", "missing")).ok
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert not result.ok and "missing" in result.message
    assert adapter.model.ngeom == 1 and adapter.model.geom(0).name == "box"
    assert not session.editing and not draft.applying and draft.compatible()
    assert draft.stage(cmd.SetGeometryShape(created.entity_id, "sphere")).ok
    draft.applying = True
    assert session.apply_model_edits(draft.resolve_commands(session)).ok
    draft.clear()
    assert adapter.model.ngeom == 2
    assert session.submit(cmd.Undo()).ok
    assert adapter.model.ngeom == 1


def test_creation_rebinds_later_source_after_node_indices_shift(scene):
    session, adapter, draft = scene
    box = next(node for node in session.nodes if node.name == "box")
    world = next(node for node in session.nodes if node.type is NodeType.WORLD)
    assert draft.stage(cmd.AddModelElement(world.node_id, "geom:plane", "ground")).ok
    duplicate = draft.stage(cmd.DuplicateModelElement(box.node_id))
    color = (0.2, 0.6, 0.9, 1)
    assert draft.stage(cmd.SetGeometryColor(-1, color, duplicate.entity_key)).ok
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    np.testing.assert_allclose(adapter.model.geom("box_copy").rgba, color)


def test_failed_mixed_gesture_keeps_only_the_earlier_draft(scene):
    session, adapter, draft = scene
    app = _editor(session, draft)
    box = next(node for node in session.nodes if node.name == "box")
    original_color = adapter.model.geom("box").rgba.copy()
    assert draft.stage(cmd.SetGeometrySize(box.node_id, np.full(3, 0.4))).ok
    with model_edit_scope(session, app._intercept_model_edit):
        assert session.submit(cmd.BeginEditTransaction("Mixed edit")).ok
        app._add_model_primitive("plane", "temporary")
        assert session.submit(cmd.SetGeometryColor(box.node_id, (1, 0, 0, 1))).ok
        failed = session.submit(cmd.RenameSceneEntity(999, "missing"))
        assert not failed.ok
        result = session.submit(cmd.EndEditTransaction())
    assert not result.ok and failed.message in result.message
    assert not session.editing and draft.compatible() and draft.visible
    assert len(draft.commands) == 1
    assert not any(node.name == "temporary" for node in session.nodes)
    np.testing.assert_allclose(adapter.model.geom("box").rgba, original_color)
    np.testing.assert_allclose(session.source.geom_size[0], (0.4, 0.4, 0.4))


@pytest.mark.parametrize("shape", ["capsule", "cylinder", "sphere", "box", "plane"])
def test_pending_primitive_can_be_created_moved_resized_and_applied_without_visual_change(
    scene, shape
):
    from mojive import math3d
    from mojive.adapters.base import FrameNeeds

    session, adapter, draft = scene
    app = _editor(session, draft)
    original = adapter._m
    with model_edit_scope(session, app._intercept_model_edit):
        app._add_model_primitive(shape, "new_shape")
        node = session.selected_node
        assert node.name == "new_shape"
        assert session.node_by_object_id(node.object_id) is node
        rotation = math3d.axis_angle_to_mat3((0, 1, 0), 0.7)
        assert session.submit(cmd.SetPose(node.node_id, np.array((2, 1, 3)), rotation)).ok
        size = np.array((0.4, 0.4, 0.4 if shape == "sphere" else 0.7))
        assert session.submit(cmd.SetGeometrySize(node.node_id, size)).ok
        assert adapter._m is original and adapter._m.ngeom == 1
        session.tick(FrameNeeds())
        before = _geometry_state(session, node)
        bounds = session.node_world_bounds(node.node_id)
        assert bounds is not None
        assert bounds.center == pytest.approx((2, 1, 3), abs=1e-6)

    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    final = next(node for node in session.nodes if node.name == "new_shape")
    np.testing.assert_allclose(_geometry_state(session, final), before, atol=1e-6)
    assert session.submit(cmd.Undo()).ok
    assert not any(node.name == "new_shape" for node in session.nodes)


def test_capsule_preview_matches_rebuilt_caps_and_discard_restores_original(scene):
    session, adapter, draft = scene
    world = next(node for node in session.nodes if node.type is NodeType.WORLD and node.parent < 0)
    created = session.submit(cmd.AddModelElement(world.node_id, "geom:capsule", "capsule"))
    node = session.node(created.entity_id)
    original = _geometry_state(session, node)
    compiled = adapter._m
    for size in ((0.2, 0.2, 0.6), (0.8, 0.8, 0.05)):
        assert draft.stage(cmd.SetGeometrySize(node.node_id, np.array(size))).ok
        preview = _geometry_state(session, node)
        assert adapter._m is compiled
        assert not np.allclose(preview, original)
        draft.clear()
        np.testing.assert_allclose(_geometry_state(session, node), original)
    assert draft.stage(cmd.SetGeometrySize(node.node_id, np.array((0.2, 0.2, 0.6)))).ok
    preview = _geometry_state(session, node)
    draft.applying = True
    assert session.apply_model_edits(draft.resolve_commands(session)).ok
    draft.clear()
    np.testing.assert_allclose(_geometry_state(session, session.node(node.node_id)), preview)


def test_multiple_pending_creations_keep_distinct_colors_and_allow_authored_scene_edits(scene):
    from mojive.types import MeshShape

    session, adapter, draft = scene
    app = _editor(session, draft)
    with model_edit_scope(session, app._intercept_model_edit):
        for name, color in (("red", (1, 0, 0, 1)), ("green", (0, 1, 0, 1))):
            app._add_model_primitive("capsule", name)
            node = session.selected_node
            assert session.submit(cmd.SetGeometryColor(node.node_id, color)).ok
        app._add_scene_object(MeshShape.BOX, "box2")
        box_object = session.selected
        box_geometry = session.node(session.selected_node.children[0])
        assert session.submit(cmd.SetGeometryColor(box_geometry.node_id, (0.1, 0.2, 0.8, 1))).ok
        assert draft.visible
        assert {"red", "green", "box2"} <= {node.name for node in session.nodes}
        for name, color in (("red", (1, 0, 0, 1)), ("green", (0, 1, 0, 1))):
            node = next(node for node in session.nodes if node.name == name)
            rows = session.source.geom_node == node.node_id
            assert np.all(session.source.geom_rgba[rows] == color)
        assert adapter._m.ngeom == 1
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    assert session.selected == box_object
    for name, color in (("red", (1, 0, 0, 1)), ("green", (0, 1, 0, 1))):
        node = next(node for node in session.nodes if node.name == name)
        assert np.all(session.source.geom_rgba[session.source.geom_node == node.node_id] == color)
    box_geometry = session.node(session.selected_node.children[0])
    np.testing.assert_allclose(
        session.source.geom_rgba[session.source.geom_node == box_geometry.node_id],
        [[0.1, 0.2, 0.8, 1]],
    )


def test_creating_world_geometry_keeps_existing_geometry_edits_bound_to_their_target(scene):
    session, adapter, draft = scene
    app = _editor(session, draft)
    box = next(node for node in session.nodes if node.name == "box")
    before_id = box.node_id
    with model_edit_scope(session, app._intercept_model_edit):
        assert session.submit(cmd.SetGeometrySize(box.node_id, np.array((0.4, 0.5, 0.6))))
        app._add_model_primitive("capsule", "new_shape")
        preview = _geometry_state(session, box)
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    box = next(node for node in session.nodes if node.name == "box")
    assert box.node_id != before_id
    np.testing.assert_allclose(_geometry_state(session, box), preview)
    np.testing.assert_allclose(adapter._m.geom("box").size, (0.4, 0.5, 0.6))
    assert session.submit(cmd.Undo())
    np.testing.assert_allclose(adapter._m.geom("box").size, (0.1, 0.2, 0.3))
