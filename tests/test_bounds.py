"""World bounds for supported instance pose sources and mesh resources."""

from dataclasses import replace

import numpy as np

from mojive import Scene
from mojive.adapters.base import FrameNeeds
from mojive.adapters.static import StaticSceneAdapter
from mojive.session import Session
from mojive.types import Bounds, CenteredBounds, InstancePoseSource, MeshData, MeshUpdate


def mesh_at(x):
    return MeshData(
        np.array([[x, 0, 0], [x + 2, 0, 0], [x, 2, 0]], np.float32),
        np.tile([0, 0, 1], (3, 1)).astype(np.float32),
        np.zeros((3, 2), np.float32),
        np.arange(3, dtype=np.uint32),
    )


def test_offset_mesh_bounds_match_world_geometry():
    scene = Scene()
    obj = scene.mesh(mesh_at(100), size=(1, 1, 1))
    session = Session(StaticSceneAdapter(scene))
    bounds = session.bounds()
    assert isinstance(bounds, Bounds)
    np.testing.assert_allclose(bounds.minimum, [100, 0, 0])
    np.testing.assert_allclose(bounds.maximum, [102, 2, 0])
    node = session.node_by_object_id(obj.object_id)
    selected = session.node_world_bounds(node.node_id)
    assert isinstance(selected, CenteredBounds)
    np.testing.assert_allclose(selected.minimum, bounds.minimum)
    np.testing.assert_allclose(selected.maximum, bounds.maximum)


def test_rotated_box_uses_all_corners():
    scene = Scene()
    c = 2**-0.5
    scene.box(size=(1, 1, 1), rotation=np.array([[c, -c, 0], [c, c, 0], [0, 0, 1]]))
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().half_extent, [2**0.5, 2**0.5, 1])


def test_shared_pose_sources_keep_each_instances_scale_and_local_offset():
    scene = Scene()
    scene.box(size=(1, 1, 1), position=(10, 0, 0))
    scene.box(size=(4, 2, 1))
    source, frame = scene.source, scene.frame
    source.geom_source[:] = 0
    source.geom_local[1, 0, 3] = 2
    frame.geom_xpos = frame.geom_xpos[:1]
    frame.geom_xmat = frame.geom_xmat[:1]
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().minimum, [8, -2, -1])
    np.testing.assert_allclose(session.bounds().maximum, [16, 2, 1])


def test_world_and_site_instances_do_not_require_a_geom_pose():
    scene = Scene()
    scene.box(size=(1, 1, 1))
    scene.box(size=(1, 1, 1))
    source, frame = scene.source, scene.frame
    source.geom_pose_source[:] = [InstancePoseSource.WORLD, InstancePoseSource.SITE]
    source.geom_source[:] = [99, 0]
    source.geom_local[0, 0, 3] = -5
    frame.geom_xpos = np.zeros((0, 3), np.float32)
    frame.geom_xmat = np.zeros((0, 3, 3), np.float32)
    frame.site_xpos = np.array([[10, 0, 0]], np.float32)
    frame.site_xmat = np.eye(3, dtype=np.float32)[None]
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().minimum, [-6, -1, -1])
    np.testing.assert_allclose(session.bounds().maximum, [11, 1, 1])


def test_dynamic_mesh_bounds_follow_current_frame_updates():
    scene = Scene()
    obj = scene.mesh(mesh_at(0), size=(1, 1, 1), position=(10, 0, 0))
    source = scene.source
    source.dynamic_meshes = frozenset({obj.mesh_key})
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().minimum, [10, 0, 0])
    mesh = mesh_at(100)
    scene.frame.mesh_updates = {obj.mesh_key: MeshUpdate(mesh.positions, mesh.normals)}
    np.testing.assert_allclose(session.bounds().minimum, [110, 0, 0])


def test_replacing_a_static_mesh_invalidates_cached_bounds():
    scene = Scene()
    obj = scene.mesh(mesh_at(0), size=(1, 1, 1))
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().minimum, [0, 0, 0])
    scene.replace_mesh(obj.mesh_key, replace(mesh_at(20)))
    session.tick(FrameNeeds())
    np.testing.assert_allclose(session.bounds().minimum, [20, 0, 0])


def test_infinite_planes_do_not_expand_framing_bounds():
    scene = Scene()
    scene.box(size=(1, 1, 1))
    scene.plane(size=(1000, 1000, 1))
    scene.source.geom_infinite_plane[1] = True
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().maximum, 1)


def test_local_and_world_rotations_cancel_without_expanding_bounds():
    scene = Scene()
    c = 2**-0.5
    rotation = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1]], np.float32)
    scene.box(size=(2, 1, 0.5), rotation=rotation)
    scene.source.geom_local[0, :3, :3] = rotation.T
    session = Session(StaticSceneAdapter(scene))
    np.testing.assert_allclose(session.bounds().half_extent, [2, 1, 0.5], atol=1e-6)


def test_authored_source_framing_metadata_includes_offset_mesh_vertices():
    scene = Scene()
    scene.mesh(mesh_at(100), size=(1, 1, 1))
    np.testing.assert_allclose(scene.source.scene_center, [101, 1, 0])
    np.testing.assert_allclose(scene.source.scene_extent, 2**0.5)
