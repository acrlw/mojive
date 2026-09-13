from __future__ import annotations

import numpy as np
import pytest

from mojive import math3d
from mojive.types import MeshKey, MeshShape

moderngl = pytest.importorskip("moderngl")

from mojive.render.opengl.passes.reflect import ReflectPass  # noqa: E402
from mojive.render.scene import RenderScene  # noqa: E402


@pytest.mark.parametrize(
    ("point", "normal", "src", "want"),
    [
        ((0, 0, 0), (0, 0, 1), (1, 2, 3), (1, 2, -3)),
        ((0, 0, 2), (0, 0, 1), (1, 2, 3), (1, 2, 1)),
        ((0, 0, 2), (0, 0, 5), (1, 2, 3), (1, 2, 1)),
        ((1, 0, 0), (1, 0, 0), (3, 5, 7), (-1, 5, 7)),
    ],
)
def test_mirror_reflects_points_across_the_plane(point, normal, src, want):

    m = math3d.mirror(point, normal)
    got = (m @ np.array([*src, 1.0]))[:3]
    assert np.allclose(got, want, atol=1e-5)


def test_mirror_keeps_points_on_the_plane_put():

    m = math3d.mirror((0, 0, 1.5), (0, 0, 1))
    for p in ((0, 0, 1.5), (3, -2, 1.5), (-1, 7, 1.5)):
        got = (m @ np.array([*p, 1.0]))[:3]
        assert np.allclose(got, p, atol=1e-5)


def test_mirror_flips_handedness():

    m = math3d.mirror((0, 0, 0), (0.3, -0.6, 0.74))
    assert np.linalg.det(m[:3, :3]) == pytest.approx(-1.0, abs=1e-5)


def test_mirror_is_its_own_inverse():

    m = math3d.mirror((0.5, -1.0, 2.0), (0.2, 0.3, 0.9))
    assert np.allclose(m @ m, np.eye(4), atol=1e-5)


def test_degenerate_normal_falls_back_to_identity():

    m = math3d.mirror((0, 0, 0), (0, 0, 0))
    assert np.allclose(m, np.eye(4))


def make_scene(rows, shapes=None) -> RenderScene:

    scene = RenderScene()
    scene.count = len(rows)
    scene.transforms = np.stack([np.asarray(t, np.float32) for t, _ in rows])
    scene.material = np.array([[0.0, 0.0, 0.0, r] for _, r in rows], np.float32)
    scene.bucket = np.arange(len(rows), dtype=np.int32)
    shapes = shapes or [MeshShape.PLANE] * len(rows)
    scene.bucket_keys = tuple((MeshKey(shape), 0) for shape in shapes)
    return scene


def flat(z: float = 0.0, scale: float = 10.0) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = scale
    m[2, 3] = z
    return m


def test_no_reflective_material_means_no_plane():

    assert ReflectPass.find_plane(make_scene([(flat(), 0.0), (flat(1.0), 0.0)])) is None
    assert ReflectPass.find_plane(RenderScene()) is None


def test_plane_is_read_off_the_instance_transform():

    found = ReflectPass.find_plane(make_scene([(flat(z=1.5), 0.4)]))
    assert found is not None
    index, plane = found
    assert index == 0
    assert np.allclose(plane[:3], (0.0, 0.0, 1.0), atol=1e-6)
    assert plane[3] == pytest.approx(-1.5)

    assert np.dot(plane[:3], (0.0, 0.0, 2.0)) + plane[3] > 0.0
    assert np.dot(plane[:3], (0.0, 0.0, 1.0)) + plane[3] < 0.0


def test_normal_uses_the_inverse_transpose_not_the_third_column():

    m = np.array(
        [
            [1.0, 0.6, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    found = ReflectPass.find_plane(make_scene([(m, 0.3)]))
    assert found is not None
    _idx, plane = found

    basis = np.asarray(m[:3, :3], np.float64)
    want = np.linalg.inv(basis).T @ np.array([0.0, 0.0, 1.0])
    want = want / np.linalg.norm(want)
    third = basis @ np.array([0.0, 0.0, 1.0])
    third = third / np.linalg.norm(third)
    assert not np.allclose(want, third, atol=1e-3)
    assert np.allclose(plane[:3], want, atol=1e-5)


def test_the_strongest_reflector_wins():

    found = ReflectPass.find_plane(
        make_scene([(flat(z=0.0), 0.1), (flat(z=3.0), 0.6), (flat(z=1.0), 0.2)])
    )
    assert found is not None
    assert found[0] == 1
    assert found[1][3] == pytest.approx(-3.0)


def test_distinct_reflection_planes_keep_separate_layers():
    scene = make_scene([(flat(z=0.0), 0.4), (flat(z=2.0), 0.7), (flat(z=0.0), 0.2)])

    groups = ReflectPass.find_planes(scene)

    assert len(groups) == 2
    assert groups[0].indices == [1]
    assert groups[1].indices == [0, 2]
    assert groups[0].plane[3] == pytest.approx(-2.0)
    assert groups[1].plane[3] == pytest.approx(0.0)


def test_box_reflection_uses_the_positive_z_face():
    transform = flat(z=1.5, scale=2.0)
    transform[2, 2] = 0.4
    scene = make_scene([(transform, 0.6)], [MeshShape.BOX])

    groups = ReflectPass.find_planes(scene)

    assert len(groups) == 1
    assert groups[0].plane[3] == pytest.approx(-1.9)


def test_reflection_metadata_does_not_mutate_material_reflectance():
    scene = make_scene(
        [(flat(z=0.0), 0.4), (flat(z=2.0), 0.7)],
        [MeshShape.PLANE, MeshShape.BOX],
    )
    original = scene.material.copy()
    reflection = ReflectPass()

    reflection._build_reflection_info(scene, ReflectPass.find_planes(scene))

    assert np.array_equal(scene.material, original)
    # Strongest box is layer zero (+Z-only bit); plane is layer one.
    assert tuple(reflection.reflection_info) == (2, 9)


def test_nonplanar_reflective_material_does_not_replace_the_floor():
    found = ReflectPass.find_plane(
        make_scene(
            [(flat(z=0.0), 0.3), (flat(z=1.0), 0.5)],
            [MeshShape.PLANE, MeshShape.CAPSULE_SHAFT],
        )
    )
    assert found is not None
    assert found[0] == 0
    assert found[1][3] == pytest.approx(0.0)
