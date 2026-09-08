"""Independent display worlds retain shared resources and complete batch updates."""

import numpy as np
import pytest

from mojive.adapters.base import FrameNeeds, SceneFrame, SceneSource
from mojive.adapters.worlds import WorldInstances
from mojive.types import Material, MeshKey, MeshShape


def template():
    source = SceneSource(
        materials=[Material()],
        geom_mesh=[MeshKey(MeshShape.BOX)] * 4,
        geom_material=[0] * 4,
        geom_size=np.ones((4, 3), np.float32),
        geom_rgba=np.ones((4, 4), np.float32),
        geom_source=np.array([3, 1, 4, 2], np.int32),
        geom_static=np.array([True, False, False, False]),
        geom_local=np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0),
        geom_visual=np.zeros(4, np.uint8),
        geom_infinite_plane=np.zeros(4, bool),
    )
    frame = SceneFrame(
        geom_xpos=np.arange(15, dtype=np.float32).reshape(5, 3),
        geom_xmat=np.repeat(np.eye(3, dtype=np.float32)[None], 5, axis=0),
    )
    return source, frame


def test_worlds_share_resources_and_keep_static_geometry_once():
    source, frame = template()
    offsets = np.array([[0, 0, 0], [7, 0, 0]], np.float32)
    adapter = WorldInstances(source, frame, offsets)
    repeated = adapter.scene_source()
    assert repeated.instance_count == 7
    assert repeated.meshes is source.meshes
    assert repeated.materials is source.materials
    np.testing.assert_array_equal(adapter.pose_indices, [1, 4, 2])
    np.testing.assert_array_equal(repeated.geom_object_id, [1, 1, 1, 2, 2, 2, 0])
    np.testing.assert_array_equal(
        repeated.geom_segmentation, [[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [-1, 0]]
    )
    actual = adapter.frame(FrameNeeds())
    np.testing.assert_array_equal(actual.geom_xpos[-1], frame.geom_xpos[3])
    np.testing.assert_array_equal(actual.geom_xpos[3:6], frame.geom_xpos[[1, 4, 2]] + offsets[1])
    assert not adapter.caps.perturb and not adapter.caps.asset_loading
    assert not adapter.caps.clock_control


def test_world_pose_updates_are_atomic_and_reuse_frame_buffers():
    source, frame = template()
    adapter = WorldInstances(source, frame, np.array([[0, 0, 0], [7, 0, 0]]))
    actual = adapter.frame(FrameNeeds())
    previous_buffer = actual.geom_xpos
    positions = np.zeros((2, 3, 3), np.float32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3, 3))
    positions[1, :, 2] = 3
    adapter.set_poses(positions, rotations, time=2.5)
    assert adapter.frame(FrameNeeds()) is actual
    assert actual.geom_xpos is previous_buffer
    np.testing.assert_array_equal(actual.geom_xpos[:3], 0)
    np.testing.assert_array_equal(actual.geom_xpos[3:6], [[7, 0, 3]] * 3)
    before = actual.geom_xpos.copy()
    positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.set_poses(positions, rotations)
    np.testing.assert_array_equal(actual.geom_xpos, before)
    assert actual.time == 2.5
    with pytest.raises(ValueError, match="shape"):
        adapter.set_poses(positions[:1], rotations)


def test_worlds_reject_unsupported_deforming_sources():
    source, frame = template()
    source.dynamic_meshes = frozenset(source.geom_mesh)
    with pytest.raises(ValueError, match="rigid"):
        WorldInstances(source, frame, np.zeros((1, 3)))


def test_worlds_preserve_empty_optional_source_arrays():
    from dataclasses import replace

    source, frame = template()
    defaults = SceneSource()
    source = replace(
        source,
        geom_local=defaults.geom_local,
        geom_visual=defaults.geom_visual,
        geom_infinite_plane=defaults.geom_infinite_plane,
    )
    adapter = WorldInstances(source, frame, np.zeros((2, 3)))
    assert not len(adapter.scene_source().geom_local)
    assert not len(adapter.scene_source().geom_visual)
    source.geom_visual = np.zeros(2, np.uint8)
    with pytest.raises(ValueError, match="instance count"):
        WorldInstances(source, frame, np.zeros((2, 3)))
