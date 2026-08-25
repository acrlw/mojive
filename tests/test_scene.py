from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

from forge_viewer import math3d as M
from forge_viewer.render.scene import BACKGROUND_ID, INSTANCE_FLOATS, RenderScene, SceneBuilder
from forge_viewer.types import CameraView, LightSet, Material, MeshKey, MeshShape

OPAQUE = [1.0, 1.0, 1.0, 1.0]
GLASS = [1.0, 1.0, 1.0, 0.4]
MAT = [0.0, 0.5, 0.5, 0.0]


def build(rows, camera=None):
    b = SceneBuilder()
    mats = {}
    for shape, matname, rgba in rows:
        if matname not in mats:
            mats[matname] = b.material_id(Material(name=matname))
        b.add(
            MeshKey(shape),
            mats[matname],
            M.compose([0.0, 0.0, 0.0], np.eye(3), [1, 1, 1]),
            rgba,
            MAT,
            object_id=1,
        )
    return b, b.build(camera or CameraView(), LightSet(), 1.0, np.zeros(3))


def test_validate_accepts_a_well_formed_scene():
    _, scene = build(
        [
            (MeshShape.BOX, "a", OPAQUE),
            (MeshShape.SPHERE, "a", OPAQUE),
            (MeshShape.BOX, "b", OPAQUE),
        ]
    )
    scene.validate()
    assert scene.count == 3
    assert scene.bucket_count() == 3


def test_bucket_ranges_are_contiguous_and_cover_everything():

    _, scene = build([(MeshShape.BOX, "a", OPAQUE)] * 3 + [(MeshShape.SPHERE, "a", OPAQUE)] * 2)
    cursor = 0
    for start, stop in scene.bucket_ranges:
        assert start == cursor
        cursor = stop
    assert cursor == scene.count

    broken = RenderScene(**{k: getattr(scene, k) for k in scene.__dataclass_fields__})
    broken.bucket_ranges = ((0, 2), (3, 5))
    with pytest.raises(ValueError, match="starts at"):
        broken.validate()


def test_transforms_stay_row_major():

    b = SceneBuilder()
    m = b.material_id(Material(name="a"))
    b.add(
        MeshKey(MeshShape.BOX), m, M.compose([1.0, 2.0, 3.0], np.eye(3), [1, 1, 1]), OPAQUE, MAT, 7
    )
    scene = b.build(CameraView(), LightSet(), 1.0, np.zeros(3))
    assert np.allclose(scene.transforms[0, :3, 3], [1.0, 2.0, 3.0])
    assert scene.transforms.dtype == np.float32


def test_object_id_is_uint32_and_zero_is_reserved():

    assert BACKGROUND_ID == 0
    _, scene = build([(MeshShape.BOX, "a", OPAQUE)])
    assert scene.object_id.dtype == np.uint32


def test_scale_does_not_enter_the_bucket_key():

    b = SceneBuilder()
    m = b.material_id(Material(name="a"))
    for s in (0.1, 1.0, 17.0):
        b.add(MeshKey(MeshShape.BOX), m, M.compose([0, 0, 0], np.eye(3), [s, s, s]), OPAQUE, MAT, 1)
    scene = b.build(CameraView(), LightSet(), 1.0, np.zeros(3))
    assert scene.bucket_count() == 1
    assert scene.bucket_ranges == ((0, 3),)


def test_same_mesh_and_material_but_different_alpha_split_into_two_buckets():

    _, scene = build(
        [
            (MeshShape.BOX, "a", OPAQUE),
            (MeshShape.BOX, "a", GLASS),
            (MeshShape.BOX, "a", OPAQUE),
        ]
    )
    assert scene.bucket_count() == 2
    assert len(scene.opaque_buckets) == 1
    assert len(scene.transparent_buckets) == 1

    assert scene.bucket_keys[0] == scene.bucket_keys[1]


def test_opaque_buckets_all_come_before_transparent_ones():

    _, scene = build(
        [
            (MeshShape.BOX, "a", GLASS),
            (MeshShape.SPHERE, "a", OPAQUE),
            (MeshShape.CONE, "a", GLASS),
            (MeshShape.BOX, "b", OPAQUE),
        ]
    )
    assert max(scene.opaque_buckets) < min(scene.transparent_buckets)


def test_transparent_buckets_draw_far_to_near():

    b = SceneBuilder()
    m = b.material_id(Material(name="a"))
    for i, x in enumerate((1.0, 9.0, 5.0)):
        b.add(
            MeshKey([MeshShape.BOX, MeshShape.SPHERE, MeshShape.CONE][i]),
            m,
            M.compose([x, 0, 0], np.eye(3), [1, 1, 1]),
            GLASS,
            MAT,
            1,
        )
    cam = CameraView(eye=np.zeros(3, np.float32), target=np.array([1.0, 0, 0], np.float32))
    scene = b.build(cam, LightSet(), 1.0, np.zeros(3))
    order = scene.transparent_draw_order()
    dists = [
        float(np.linalg.norm(scene.transforms[scene.bucket_ranges[bk][0], :3, 3] - cam.eye))
        for bk in order
    ]
    assert dists == sorted(dists, reverse=True)


def test_transparent_instances_with_shared_mesh_are_sorted_individually():
    b = SceneBuilder()
    material = b.material_id(Material(name="glass"))
    for x in (1.0, 9.0, 5.0):
        b.add(
            MeshKey(MeshShape.BOX),
            material,
            M.compose([x, 0, 0], np.eye(3), [1, 1, 1]),
            GLASS,
            MAT,
            1,
        )
    scene = b.build(CameraView(eye=np.zeros(3)), LightSet(), 1.0, np.zeros(3))

    assert len(scene.transparent_buckets) == 3
    assert all(stop - start == 1 for start, stop in scene.bucket_ranges)
    centers = [
        scene.transforms[scene.bucket_ranges[bucket][0], 0, 3]
        for bucket in scene.transparent_draw_order()
    ]
    assert centers == [9.0, 5.0, 1.0]


def test_transparent_order_accepts_a_reflected_eye():
    b = SceneBuilder()
    m = b.material_id(Material(name="a"))
    for shape, x in ((MeshShape.BOX, -2.0), (MeshShape.SPHERE, 3.0)):
        b.add(
            MeshKey(shape),
            m,
            M.compose([x, 0, 0], np.eye(3), [1, 1, 1]),
            GLASS,
            MAT,
            1,
        )
    scene = b.build(CameraView(), LightSet(), 1.0, np.zeros(3))
    eye = np.array([10.0, 0.0, 0.0], np.float32)
    order = scene.transparent_draw_order(eye)
    centers = [scene.transforms[scene.bucket_ranges[bucket][0], 0, 3] for bucket in order]
    assert centers == [-2.0, 3.0]


def test_write_index_is_a_permutation_computed_once():

    b, scene = build(
        [
            (MeshShape.BOX, "a", OPAQUE),
            (MeshShape.SPHERE, "a", OPAQUE),
            (MeshShape.BOX, "a", GLASS),
            (MeshShape.BOX, "b", OPAQUE),
            (MeshShape.SPHERE, "a", OPAQUE),
        ]
    )
    wi = b.write_index
    assert len(wi) == scene.count
    assert sorted(wi.tolist()) == list(range(scene.count))


def test_write_index_actually_places_each_instance_in_its_bucket():

    rows = [
        (MeshShape.BOX, "a", OPAQUE),
        (MeshShape.SPHERE, "a", OPAQUE),
        (MeshShape.BOX, "a", OPAQUE),
        (MeshShape.SPHERE, "b", OPAQUE),
    ]
    b = SceneBuilder()
    mats, ids = {}, []
    for i, (shape, name, rgba) in enumerate(rows):
        if name not in mats:
            mats[name] = b.material_id(Material(name=name))
        b.add(MeshKey(shape), mats[name], M.identity(), rgba, MAT, object_id=i + 1)
        ids.append(i + 1)
    scene = b.build(CameraView(), LightSet(), 1.0, np.zeros(3))
    for src, oid in enumerate(ids):
        dst = int(b.write_index[src])
        assert int(scene.object_id[dst]) == oid
        bkt = int(scene.bucket[dst])
        start, stop = scene.bucket_ranges[bkt]
        assert start <= dst < stop


def test_refilling_transforms_does_not_change_bucketing():

    b, scene = build([(MeshShape.BOX, "a", OPAQUE), (MeshShape.SPHERE, "a", GLASS)])
    keys, ranges, wi = scene.bucket_keys, scene.bucket_ranges, b.write_index.copy()
    for f in range(100):
        scene.transforms[:, :3, 3] = float(f)
    assert scene.bucket_keys == keys
    assert scene.bucket_ranges == ranges
    assert np.array_equal(b.write_index, wi)
    scene.validate()


def test_scatter_refill_does_not_allocate():

    n = 400
    b = SceneBuilder()
    m = b.material_id(Material(name="a"))
    for i in range(n):
        b.add(
            MeshKey(MeshShape.BOX if i % 2 else MeshShape.SPHERE), m, M.identity(), OPAQUE, MAT, 1
        )
    scene = b.build(CameraView(), LightSet(), 1.0, np.zeros(3))
    wi = b.write_index.astype(np.intp)
    stage = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))

    def refill():
        scene.transforms[wi] = stage

    for _ in range(5):
        refill()
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    for _ in range(200):
        refill()
    gc.collect()
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grown = sum(d.size_diff for d in snap.compare_to(base, "lineno"))
    assert grown < 20_000


def test_instance_float_count_matches_the_documented_layout():

    assert INSTANCE_FLOATS == 16 + 4 + 4 + 4 + 4 == 32


def test_material_quadruple_keeps_the_fourth_slot_reserved():

    _, scene = build([(MeshShape.BOX, "a", OPAQUE)])
    assert scene.material.shape[1] == 4
    assert scene.material[0, 3] == 0.0
