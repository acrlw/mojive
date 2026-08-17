from __future__ import annotations

import gc
import tracemalloc
from dataclasses import replace

import numpy as np
import pytest

from forge_viewer.adapters.base import NodeKind as NK
from forge_viewer.adapters.base import SceneFrame, SceneNode, SceneSource
from forge_viewer.render.builder import SceneSourceBuilder
from forge_viewer.types import CameraView, InstanceVisual, Material, MeshKey, MeshShape


def _grid_material(repeat: float = 1.0, uniform: bool = True) -> Material:
    return Material(
        name="grid",
        rgba=np.ones(4, np.float32),
        texture="grid",
        tex_repeat=np.array([repeat, repeat], np.float32),
        tex_uniform=uniform,
    )


def make_source(
    *,
    bodies: int = 4,
    with_plane: bool = True,
    plane_repeat: float = 1.0,
) -> SceneSource:

    src = SceneSource()
    src.materials = [_grid_material(plane_repeat), Material(name="paint")]

    keys: list[MeshKey] = []
    mats: list[int] = []
    sizes: list[list[float]] = []
    rgba: list[list[float]] = []
    obj: list[int] = []
    body: list[int] = []
    source: list[int] = []
    local: list[np.ndarray] = []
    infinite: list[bool] = []
    nodes: list[SceneNode] = [SceneNode(node_id=0, name="world", kind=NK.WORLD, parent=-1)]

    geom = 0
    if with_plane:
        keys.append(MeshKey(MeshShape.PLANE))
        mats.append(0)
        sizes.append([0.0, 0.0, 1.0])
        rgba.append([1.0, 1.0, 1.0, 1.0])
        obj.append(0)
        body.append(0)
        source.append(geom)
        local.append(np.eye(4, dtype=np.float32))
        infinite.append(True)
        nodes.append(
            SceneNode(node_id=len(nodes), name="floor", kind=NK.GEOM, parent=0, body_index=0)
        )
        geom += 1

    for b in range(1, bodies + 1):
        body_node = len(nodes)
        nodes.append(
            SceneNode(
                node_id=body_node,
                name=f"link{b}",
                kind=NK.LINK,
                parent=0,
                object_id=b,
                body_index=b,
            )
        )

        keys.append(MeshKey(MeshShape.BOX))
        mats.append(1)
        sizes.append([0.1 + 0.01 * b, 0.2, 0.3])
        rgba.append([0.8, 0.2, 0.2, 1.0])
        obj.append(b)
        body.append(b)
        source.append(geom)
        local.append(np.eye(4, dtype=np.float32))
        infinite.append(False)
        nodes.append(
            SceneNode(
                node_id=len(nodes), name=f"box{b}", kind=NK.GEOM, parent=body_node, body_index=b
            )
        )
        geom += 1

        half, radius = 0.25, 0.05
        for part, offset in (
            (MeshShape.CAPSULE_SHAFT, 0.0),
            (MeshShape.CAPSULE_CAP, +half),
            (MeshShape.CAPSULE_CAP, -half),
        ):
            keys.append(MeshKey(part))
            mats.append(1)
            sizes.append(
                [radius, radius, half] if part is MeshShape.CAPSULE_SHAFT else [radius] * 3
            )
            rgba.append([0.2, 0.4, 0.9, 1.0])
            obj.append(b)
            body.append(b)
            source.append(geom)
            m = np.eye(4, dtype=np.float32)
            m[2, 3] = offset
            if offset < 0:
                m[1, 1] = m[2, 2] = -1.0
            local.append(m)
            infinite.append(False)
        nodes.append(
            SceneNode(
                node_id=len(nodes), name=f"cap{b}", kind=NK.GEOM, parent=body_node, body_index=b
            )
        )
        geom += 1

    for node in nodes:
        if node.parent >= 0:
            nodes[node.parent].children.append(node.node_id)

    src.geom_mesh = keys
    src.geom_material = mats
    src.geom_size = np.array(sizes, np.float32)
    src.geom_rgba = np.array(rgba, np.float32)
    src.geom_object_id = np.array(obj, np.uint32)
    src.geom_body = np.array(body, np.int32)
    src.geom_source = np.array(source, np.int32)
    src.geom_local = np.stack(local)
    src.geom_infinite_plane = np.array(infinite, bool)
    src.nodes = nodes
    src.scene_extent = 3.0
    return src


def make_frame(source: SceneSource, seed: int = 0) -> SceneFrame:

    n = int(source.geom_source.max()) + 1
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    mats = np.zeros((n, 3, 3), np.float32)
    for i in range(n):
        a = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(a), np.sin(a)
        mats[i] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
    f = SceneFrame()
    f.geom_xpos = pos
    f.geom_xmat = mats
    return f


def naive_transforms(source: SceneSource, frame: SceneFrame) -> np.ndarray:

    out = []
    for i in range(source.instance_count):
        g = int(source.geom_source[i])
        w = np.eye(4, dtype=np.float64)
        w[:3, :3] = frame.geom_xmat[g]
        w[:3, 3] = frame.geom_xpos[g]
        ls = np.asarray(source.geom_local[i], np.float64).copy()
        ls[:3, :3] = ls[:3, :3] * np.asarray(source.geom_size[i], np.float64)[None, :]
        out.append(w @ ls)
    return np.array(out)


def test_build_validates():

    src = make_source(bodies=6)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    scene.validate()
    assert scene.count == src.instance_count

    assert scene.bucket_count() < scene.count


def test_planar_reflection_is_limited_to_planes_and_box_top_faces():
    src = make_source(bodies=1)
    src.materials[0] = replace(src.materials[0], reflectance=0.3)
    src.materials[1] = replace(src.materials[1], reflectance=0.8)
    builder = SceneSourceBuilder()
    scene = builder.set_source(src, CameraView())

    assert scene.material[builder.write_index[0], 3] == pytest.approx(0.3)
    assert scene.material[builder.write_index[1], 3] == pytest.approx(0.8)
    assert np.all(scene.material[builder.write_index[2:], 3] == 0.0)


def test_transforms_match_naive():

    src = make_source(bodies=5)
    frame = make_frame(src, seed=3)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    b.update(frame, CameraView())

    expect = naive_transforms(src, frame)
    got = scene.transforms[b.write_index]

    keep = ~src.geom_infinite_plane
    assert np.allclose(got[keep], expect[keep], atol=1e-5)


def test_capsule_caps_ride_the_shaft():

    src = make_source(bodies=1, with_plane=False)
    frame = make_frame(src, seed=7)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    b.update(frame)

    rows = b.write_index
    shaft, cap_a, cap_b = (scene.transforms[rows[i]] for i in (1, 2, 3))
    axis = frame.geom_xmat[int(src.geom_source[1])][:, 2]
    center = frame.geom_xpos[int(src.geom_source[1])]
    assert np.allclose(shaft[:3, 3], center, atol=1e-5)
    assert np.allclose(cap_a[:3, 3], center + 0.25 * axis, atol=1e-5)
    assert np.allclose(cap_b[:3, 3], center - 0.25 * axis, atol=1e-5)

    assert np.allclose(np.linalg.norm(cap_a[:3, :3], axis=0), 0.05, atol=1e-6)

    assert np.linalg.det(cap_b[:3, :3]) > 0


def test_flipped_capsule_cap_flips_v():

    src = make_source(bodies=1, with_plane=False)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    rows = b.write_index
    top = scene.tex_coef[rows[2]]
    bottom = scene.tex_coef[rows[3]]
    assert top[1] > 0 and bottom[1] < 0

    def mapped(coef, v):
        return coef[1] * v + coef[3]

    assert mapped(bottom, 0.75) == pytest.approx(0.25)
    assert mapped(bottom, 1.0) == pytest.approx(0.0)
    assert mapped(top, 0.75) == pytest.approx(0.75)


def test_buckets_never_move_across_frames():

    src = make_source(bodies=8)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())

    keys0 = scene.bucket_keys
    ranges0 = scene.bucket_ranges
    write0 = b.write_index.copy()
    bucket0 = scene.bucket.copy()
    first = None

    for i in range(100):
        frame = make_frame(src, seed=i)
        cam = CameraView(eye=np.array([3.0 + i, -3.0, 2.0 + 0.1 * i], np.float32))
        b.update(frame, cam)
        assert scene.bucket_keys == keys0
        assert scene.bucket_ranges == ranges0
        assert np.array_equal(b.write_index, write0)
        assert np.array_equal(scene.bucket, bucket0)
        if first is None:
            first = scene.transforms.copy()

    assert not np.allclose(scene.transforms, first)


def test_hot_path_does_not_allocate():

    src = make_source(bodies=150)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    assert scene.count > 500

    frames = [make_frame(src, seed=i % 7) for i in range(4)]
    cams = [CameraView(eye=np.array([3.0 + i, -3.0, 2.0], np.float32)) for i in range(4)]
    holder = SceneFrame()
    for i in range(4):
        holder.geom_xpos, holder.geom_xmat = frames[i].geom_xpos, frames[i].geom_xmat
        b.update(holder, cams[i])

    budget = scene.count * 4 * 4 * 4
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    start, _ = tracemalloc.get_traced_memory()
    for i in range(100):
        holder.geom_xpos, holder.geom_xmat = frames[i % 4].geom_xpos, frames[i % 4].geom_xmat
        b.update(holder, cams[i % 4])
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak - start < budget
    assert current - start < 64 * 1024


def test_infinite_plane_grows_with_camera():

    repeat = 2.0
    src = make_source(bodies=2, plane_repeat=repeat)
    frame = make_frame(src, seed=1)
    frame.geom_xpos[0] = 0.0
    frame.geom_xmat[0] = np.eye(3, dtype=np.float32)
    b = SceneSourceBuilder()
    b.set_source(src, CameraView())
    period = 2.0 / repeat

    near_cam = CameraView(eye=np.array([2.37, 1.63, 2.0], np.float32), far=50.5)
    b.update(frame, near_cam)
    ((hx0, hy0),) = b.infinite_plane_half_extents()

    far_cam = CameraView(eye=np.array([20.7, 19.3, 20.0], np.float32), far=505.5)
    b.update(frame, far_cam)
    ((hx1, hy1),) = b.infinite_plane_half_extents()

    assert hx1 > hx0 * 5 and hy1 > hy0 * 5
    for h in (hx0, hy0, hx1, hy1):
        assert h >= 50.0
        assert abs(h / period - round(h / period)) < 1e-6

    row = b.scene.infinite_planes[0]
    assert b.scene.transforms[row][0, 0] == pytest.approx(hx1, rel=1e-5)


def test_infinite_plane_phase_is_stable():

    src = make_source(bodies=1, plane_repeat=4.0)
    frame = make_frame(src, seed=2)
    frame.geom_xpos[0] = 0.0
    frame.geom_xmat[0] = np.eye(3, dtype=np.float32)
    b = SceneSourceBuilder()
    b.set_source(src, CameraView())
    row = b.scene.infinite_planes[0]

    periods = []
    for d in (5.0, 50.0, 500.0):
        b.update(frame, CameraView(eye=np.array([d, d, d], np.float32), far=100.0))
        half = b.infinite_plane_half_extents()[0][0]
        repeats = b.scene.tex_coef[row][0]
        periods.append(2.0 * half / repeats)
    assert np.allclose(periods, periods[0], rtol=1e-6)


@pytest.mark.parametrize(("repeat", "expect"), [(1.0, 0.5), (2.0, 1.0), (4.0, 2.0)])
def test_texuniform_calibration(repeat: float, expect: float):

    src = SceneSource()
    src.materials = [_grid_material(repeat)]
    src.geom_mesh = [MeshKey(MeshShape.BOX)]
    src.geom_material = [0]
    src.geom_size = np.array([[0.5, 0.5, 0.5]], np.float32)
    src.geom_rgba = np.ones((1, 4), np.float32)
    src.geom_object_id = np.array([1], np.uint32)
    src.geom_body = np.array([1], np.int32)
    src.geom_source = np.array([0], np.int32)
    src.geom_local = np.eye(4, dtype=np.float32)[None]
    src.geom_infinite_plane = np.zeros(1, bool)

    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    assert scene.tex_coef[0][0] == pytest.approx(expect)
    assert scene.tex_coef[0][1] == pytest.approx(expect)


def test_texuniform_off_uses_repeat_directly():

    src = SceneSource()
    src.materials = [_grid_material(3.0, uniform=False)]
    src.geom_mesh = [MeshKey(MeshShape.BOX)]
    src.geom_material = [0]
    src.geom_size = np.array([[2.0, 2.0, 2.0]], np.float32)
    src.geom_rgba = np.ones((1, 4), np.float32)
    src.geom_object_id = np.array([1], np.uint32)
    src.geom_body = np.array([1], np.int32)
    src.geom_source = np.array([0], np.int32)
    src.geom_local = np.eye(4, dtype=np.float32)[None]
    src.geom_infinite_plane = np.zeros(1, bool)

    scene = SceneSourceBuilder().set_source(src, CameraView())
    assert scene.tex_coef[0][0] == pytest.approx(3.0)


def test_set_visible_rebuilds_and_drops_instances():

    src = make_source(bodies=3)
    b = SceneSourceBuilder()
    scene0 = b.set_source(src, CameraView())
    n0 = scene0.count

    cap_node = next(n for n in src.nodes if n.name == "cap1")
    assert b.set_visible(cap_node.node_id, False)
    assert b.scene.count == n0 - 3
    b.scene.validate()
    assert b.stats().hidden == 3

    link = next(n for n in src.nodes if n.name == "link2")
    assert b.set_visible(link.node_id, False)
    assert b.scene.count == n0 - 3 - 4

    assert b.set_visible(cap_node.node_id, True)
    assert b.scene.count == n0 - 4
    assert not b.set_visible(9999, False)

    assert b.set_visible(1, False)  # node 1 = floor
    assert b.scene.infinite_planes == ()
    assert b.infinite_plane_half_extents() == ()
    b.update(make_frame(src, seed=0), CameraView())
    b.scene.validate()


def test_visual_options_filter_static_skin_and_flex_instances():
    src = make_source(bodies=1)
    src.geom_static = np.array([True, False, False, False, False])
    src.geom_visual = np.array(
        [
            InstanceVisual.DEFAULT,
            InstanceVisual.SKIN,
            InstanceVisual.FLEX_SKIN,
            InstanceVisual.FLEX_FACE,
            InstanceVisual.FLEX_EDGE,
        ],
        np.uint8,
    )
    builder = SceneSourceBuilder()
    scene = builder.set_source(src, CameraView())

    assert scene.count == 4

    assert builder.set_visual_options(
        static=True,
        skin=True,
        flex_face=True,
        flex_skin=False,
    )
    assert builder.scene.count == 4

    builder.set_visual_options(static=True, skin=True, flex_face=False, flex_skin=False)
    assert builder.scene.count == 3

    builder.set_visual_options(static=False, skin=True, flex_face=False, flex_skin=True)
    assert builder.scene.count == 3

    builder.set_visual_options(static=True, skin=False, flex_face=False, flex_skin=True)
    assert builder.scene.count == 3


def test_convex_hull_option_switches_mesh_keys():
    src = make_source(bodies=1, with_plane=False)
    original = src.geom_mesh[0]
    hull = MeshKey(MeshShape.CONVEX_HULL, 7)
    src.geom_convex_mesh = src.geom_mesh.copy()
    src.geom_convex_mesh[0] = hull
    builder = SceneSourceBuilder()
    builder.set_source(src, CameraView())

    assert original in {key for key, _ in builder.scene.bucket_keys}
    assert hull not in {key for key, _ in builder.scene.bucket_keys}
    assert builder.set_visual_options(
        static=True,
        skin=True,
        flex_face=False,
        flex_skin=True,
        convex_hull=True,
    )
    assert hull in {key for key, _ in builder.scene.bucket_keys}
    assert original not in {key for key, _ in builder.scene.bucket_keys}


def test_island_colors_replace_dynamic_instance_color_and_texture():
    src = make_source(bodies=1)
    src.materials[1] = _grid_material()
    src.instance_island_body = np.array([-1, 1, 1, 1, 1], np.int32)
    src.geom_rgba[1:, 3] = 0.25
    builder = SceneSourceBuilder()
    builder.set_source(src, CameraView())
    builder.set_visual_options(
        static=True,
        skin=True,
        flex_face=False,
        flex_skin=True,
        island=True,
    )

    island_rgba = np.array(
        [
            [0.1, 0.2, 0.3, 1.0],
            [0.8, 0.6, 0.4, 1.0],
            [0.7, 0.5, 0.3, 1.0],
            [0.6, 0.4, 0.2, 1.0],
            [0.5, 0.3, 0.1, 1.0],
        ],
        np.float32,
    )
    scene = builder.update(make_frame(src), instance_rgba=island_rgba)

    for instance in range(src.instance_count):
        row = builder.write_index[instance]
        assert scene.colors[row, :3] == pytest.approx(island_rgba[instance, :3] ** 2.2)
        assert scene.colors[row, 3] == pytest.approx(1.0)
    assert scene.materials[scene.bucket_keys[scene.bucket[builder.write_index[0]]][1]].texture
    for instance in range(1, src.instance_count):
        row = builder.write_index[instance]
        material = scene.materials[scene.bucket_keys[scene.bucket[row]][1]]
        assert material.texture is None
    assert not scene.transparent_buckets


def test_stats_reports_batching_numbers():

    mesh = pytest.importorskip("forge_viewer.render.mesh", reason="built-in meshes are unavailable")
    src = make_source(bodies=10)
    b = SceneSourceBuilder()
    scene = b.set_source(src, CameraView())
    stats = b.stats()
    assert stats.instances == scene.count
    assert stats.buckets == scene.bucket_count()
    assert not stats.notes

    expect = 0
    for bucket, (start, stop) in enumerate(scene.bucket_ranges):
        key = scene.bucket_keys[bucket][0]
        expect += mesh.builtin_mesh(key).triangle_count * (stop - start)
    assert stats.triangles == expect > 0
