"""Local geometry scaling across authored scenes, previews, persistence and history."""

import numpy as np
import pytest

from mojive import Scene, math3d
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, NodeType
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene.geometry import geometry_dimensions, geometry_size_from_dimensions
from mojive.session import Session
from mojive.session.model_edits import ModelEditDraft
from mojive.types import MeshShape


@pytest.fixture(params=[False, True], ids=["static", "workspace"])
def authored(request):
    scene = Scene()
    adapter = (
        WorkspaceAdapter(StaticSceneAdapter(Scene()), scene)
        if request.param
        else StaticSceneAdapter(scene)
    )
    session = Session(adapter)
    session.submit(cmd.Pause())
    result = session.submit(
        cmd.AddSceneObject(
            MeshShape.BOX,
            "box",
            (0.2, 0.3, 0.4),
            (1, 2, 3),
            math3d.euler_xyz_to_mat3([0.3, 0.4, 0.5]),
        )
    )
    assert result.ok
    node = next(n for n in session.nodes if n.name == "box")
    return scene, session, node


def test_scale_preview_coalesces_bakes_once_and_roundtrips_history(authored, tmp_path):
    scene, session, node = authored
    original = session.source
    pose = scene.frame.geom_xpos.copy(), scene.frame.geom_xmat.copy()
    draft = ModelEditDraft(session)
    assert draft.stage(cmd.SetScale(node.node_id, [2, 1, 1])).ok
    preview = draft.source
    geometry = session.scale_target(node.node_id)
    assert draft.stage(cmd.SetScale(geometry.node_id, [3, 0.5, 2])).ok
    assert draft.source is preview
    assert len(draft.commands) == 1
    np.testing.assert_allclose(session.scale_factors(node.node_id), [3, 0.5, 2])
    np.testing.assert_allclose(session.source.geom_size[-1], [0.6, 0.15, 0.8])
    np.testing.assert_allclose(original.geom_size[-1], [0.2, 0.3, 0.4])
    draft.applying = True
    assert session.apply_model_edits(draft.resolve_commands(session)).ok
    draft.clear()
    assert session.scale_factors(node.node_id) == (1, 1, 1)
    np.testing.assert_allclose(scene.source.geom_size[0], [0.6, 0.15, 0.8])
    np.testing.assert_array_equal(scene.frame.geom_xpos, pose[0])
    np.testing.assert_array_equal(scene.frame.geom_xmat, pose[1])
    path = scene.save(tmp_path / "scaled.mojive.json")
    np.testing.assert_array_equal(Scene.load(path).source.geom_size, scene.source.geom_size)
    assert session.submit(cmd.Undo()).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
    assert session.submit(cmd.Redo()).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.6, 0.15, 0.8])


def test_scale_discard_and_stale_document_do_not_write(authored):
    scene, session, node = authored
    draft = ModelEditDraft(session)
    assert draft.stage(cmd.SetScale(node.node_id, [2, 3, 4])).ok
    draft.clear()
    np.testing.assert_allclose(session.source.geom_size[-1], [0.2, 0.3, 0.4])
    assert draft.stage(cmd.SetScale(node.node_id, [2, 3, 4])).ok
    scene.box(name="other")
    assert not draft.stage(cmd.SetScale(node.node_id, [5, 5, 5])).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "factors", [[0, 1, 1], [-1, 1, 1], [np.nan, 1, 1], [np.inf, 1, 1], [1, 2], [1e300, 1, 1]]
)
def test_invalid_scale_is_atomic(authored, factors):
    scene, session, node = authored
    draft = ModelEditDraft(session)
    assert not draft.stage(cmd.SetScale(node.node_id, factors)).ok
    assert not draft.active
    assert not session.submit(cmd.SetScale(node.node_id, factors)).ok
    np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "shape", [MeshShape.BOX, MeshShape.SPHERE, MeshShape.CYLINDER, MeshShape.PLANE]
)
def test_programmatic_scale_preserves_local_axes_and_dimensions(shape):
    scene = Scene()
    item = scene.add(shape, size=(0.2, 0.3, 0.4))
    item.scale((2, 3, 4))
    np.testing.assert_allclose(scene.source.geom_size[0], [0.4, 0.9, 1.6])
    assert scene.source.geom_mesh[0].shape == shape
    size = scene.source.geom_size[0]
    dimensions = geometry_dimensions(shape, size)
    np.testing.assert_allclose(geometry_size_from_dimensions(shape, size, dimensions.values), size)


def test_scale_requires_explicit_entity_capability(authored):
    _scene, session, _node = authored
    world = next(n for n in session.nodes if n.type is NodeType.WORLD)
    assert session.scale_target(world.node_id) is None
    assert not session.submit(cmd.SetScale(world.node_id, [2, 2, 2])).ok
    session.tick(FrameNeeds())


@pytest.mark.parametrize("pending_dimensions", [False, True])
def test_absolute_dimensions_during_scale_preview_do_not_scale_twice(authored, pending_dimensions):
    from mojive.session.model_edits import model_edit_scope
    from mojive.ui.app import ViewerApp

    _scene, session, node = authored
    app = ViewerApp.__new__(ViewerApp)
    app.session = session
    app.model_edits = ModelEditDraft(session)
    app.live_model_updates = True
    geometry = session.scale_target(node.node_id)
    if pending_dimensions:
        assert app.model_edits.stage(cmd.SetGeometrySize(geometry.node_id, [0.4, 0.6, 0.8])).ok
    with model_edit_scope(session, app._intercept_model_edit):
        assert session.submit(cmd.SetScale(node.node_id, [2, 2, 2])).ok
        assert session.submit(cmd.SetGeometrySize(geometry.node_id, np.array([0.8, 0.9, 1.2]))).ok
    np.testing.assert_allclose(session.source.geom_size[-1], [0.8, 0.9, 1.2])
    expected_factors = [2, 1.5, 1.5] if pending_dimensions else [4, 3, 3]
    np.testing.assert_allclose(session.scale_factors(node.node_id), expected_factors)
    np.testing.assert_allclose(session._source.geom_size[-1], [0.2, 0.3, 0.4])
    assert len(app.model_edits.commands) == 1 + pending_dimensions
    app.model_edits.applying = True
    assert session.apply_model_edits(app.model_edits.resolve_commands(session)).ok
    app.model_edits.clear()
    np.testing.assert_allclose(session.source.geom_size[-1], [0.8, 0.9, 1.2])
    assert session.scale_factors(node.node_id) == (1, 1, 1)


def test_composed_scale_overflow_keeps_the_previous_preview(authored):
    _scene, session, node = authored
    draft = ModelEditDraft(session)
    target = session.scale_target(node.node_id)
    assert draft.stage(cmd.SetGeometrySize(target.node_id, np.full(3, 1e30))).ok
    before = session.source.geom_size.copy()
    assert not draft.stage(cmd.SetScale(node.node_id, [1e30, 1, 1])).ok
    assert len(draft.commands) == 1
    np.testing.assert_array_equal(session.source.geom_size, before)
