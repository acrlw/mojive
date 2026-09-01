from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from mojive.render import mesh as mesh_mod
from mojive.render.mesh import BUILTIN_SHAPES, all_builtin, builtin_mesh, gizmo_mesh
from mojive.types import MeshData, MeshKey, MeshShape

CLOSED_VOLUME = {
    MeshShape.SPHERE: 4.0 * math.pi / 3.0,
    MeshShape.BOX: 8.0,
    MeshShape.CYLINDER: 2.0 * math.pi,
    MeshShape.CONE: 2.0 * math.pi / 3.0,
    MeshShape.ARROW_SHAFT: math.pi,
    MeshShape.ARROW_HEAD: math.pi / 3.0,
}


OPEN_SHAPES = {
    MeshShape.PLANE,
    MeshShape.DISK,
    MeshShape.TUBE,
    MeshShape.CAPSULE_SHAFT,
    MeshShape.CAPSULE_CAP,
}


def _mesh(shape: MeshShape) -> MeshData:
    return builtin_mesh(MeshKey(shape=shape))


def _triangles(md: MeshData) -> np.ndarray:

    return md.positions[md.indices.reshape(-1, 3)].astype(np.float64)


def signed_volume(md: MeshData) -> float:

    p = _triangles(md)
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def face_normals(md: MeshData) -> np.ndarray:
    p = _triangles(md)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@pytest.mark.parametrize("shape", BUILTIN_SHAPES, ids=lambda s: s.value)
def test_builtin_mesh_is_well_formed(shape: MeshShape) -> None:
    md = _mesh(shape)

    assert md.triangle_count > 0
    assert len(md.indices) % 3 == 0
    assert int(md.indices.max()) < len(md.positions)
    assert md.indices.dtype == np.uint32

    norms = np.linalg.norm(md.normals.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    assert np.abs(md.positions).max() <= 1.0 + 1e-5

    assert md.uvs.min() >= -1e-6 and md.uvs.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("shape", BUILTIN_SHAPES, ids=lambda s: s.value)
def test_no_degenerate_triangles(shape: MeshShape) -> None:

    p = _triangles(_mesh(shape))
    area2 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    assert area2.min() > 1e-9


@pytest.mark.parametrize("shape", sorted(CLOSED_VOLUME, key=str), ids=lambda s: s.value)
def test_closed_mesh_has_positive_volume(shape: MeshShape) -> None:

    assert signed_volume(_mesh(shape)) > 0.0


@pytest.mark.parametrize("shape", sorted(CLOSED_VOLUME, key=str), ids=lambda s: s.value)
def test_closed_mesh_volume_matches_analytic(shape: MeshShape) -> None:

    v = signed_volume(_mesh(shape))
    ratio = v / CLOSED_VOLUME[shape]
    assert 0.95 <= ratio <= 1.0 + 1e-6


@pytest.mark.parametrize("shape", BUILTIN_SHAPES, ids=lambda s: s.value)
def test_vertex_normals_agree_with_winding(shape: MeshShape) -> None:

    md = _mesh(shape)
    vn = md.normals[md.indices.reshape(-1, 3)].astype(np.float64).mean(axis=1)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    dots = np.einsum("ij,ij->i", face_normals(md), vn)
    assert dots.min() > 0.9


def test_sphere_normals_are_analytic() -> None:

    md = _mesh(MeshShape.SPHERE)
    p = md.positions.astype(np.float64)
    expect = p / np.linalg.norm(p, axis=1, keepdims=True)
    assert np.allclose(md.normals, expect, atol=1e-6)


def test_cone_normals_are_analytic() -> None:

    md = _mesh(MeshShape.CONE)
    n = md.normals.astype(np.float64)
    side = n[:, 2] > 0.0
    assert side.sum() > 0
    assert np.allclose(n[side, 2], 1.0 / math.sqrt(5.0), atol=1e-5)
    horizontal = np.linalg.norm(n[side, :2], axis=1)
    assert np.allclose(horizontal, 2.0 / math.sqrt(5.0), atol=1e-5)


def test_capsule_is_two_meshes() -> None:

    shaft = _mesh(MeshShape.CAPSULE_SHAFT)
    cap = _mesh(MeshShape.CAPSULE_CAP)
    assert shaft is not cap
    assert shaft.triangle_count > 0 and cap.triangle_count > 0

    r = np.linalg.norm(shaft.positions[:, :2], axis=1)
    assert np.allclose(r, 1.0, atol=1e-5)


def test_capsule_cap_is_a_hemisphere() -> None:

    cap = _mesh(MeshShape.CAPSULE_CAP)
    assert cap.positions[:, 2].min() >= -1e-6
    radii = np.linalg.norm(cap.positions.astype(np.float64), axis=1)
    assert np.allclose(radii, 1.0, atol=1e-5)


def test_capsule_uv_seam_is_continuous() -> None:

    shaft = _mesh(MeshShape.CAPSULE_SHAFT)
    cap = _mesh(MeshShape.CAPSULE_CAP)
    assert shaft.uvs[:, 1].max() == pytest.approx(cap.uvs[:, 1].min(), abs=1e-6)


def test_plane_uv_matches_the_calibration() -> None:

    md = _mesh(MeshShape.PLANE)
    assert md.triangle_count == 2
    assert np.allclose(md.positions[:, 2], 0.0)
    assert np.allclose(md.normals, [0.0, 0.0, 1.0])

    for (x, y, _), uv in zip(md.positions, md.uvs, strict=True):
        s, t = 0.5 * x, 0.5 * y
        assert uv == pytest.approx((0.5 + s, 0.5 - t), abs=1e-6)


def test_builtin_mesh_is_generated_once() -> None:

    a = builtin_mesh(MeshKey(shape=MeshShape.SPHERE))
    b = builtin_mesh(MeshKey(shape=MeshShape.SPHERE))
    assert a is b

    assert builtin_mesh(MeshKey(shape=MeshShape.SPHERE, index=7)) is a


def test_nothing_is_generated_at_import_time() -> None:

    importlib.reload(mesh_mod)
    assert mesh_mod._CACHE == {}


def test_all_builtin_covers_every_shape() -> None:
    table = all_builtin()
    assert set(table) == {MeshKey(shape=s) for s in BUILTIN_SHAPES}
    assert MeshKey(shape=MeshShape.ASSET) not in table

    for key, md in table.items():
        assert builtin_mesh(key) is md


def test_asset_shape_is_rejected_with_the_available_names() -> None:

    with pytest.raises(KeyError) as err:
        builtin_mesh(MeshKey(shape=MeshShape.ASSET, index=3))
    assert "sphere" in str(err.value) and "capsule_cap" in str(err.value)


def test_mesh_arrays_are_read_only() -> None:

    md = _mesh(MeshShape.BOX)
    for arr in (md.positions, md.normals, md.uvs, md.indices):
        assert not arr.flags.writeable


@pytest.mark.parametrize(
    "name", ("arrow", "plane", "ring", "half_ring", "screen_ring", "screen_ring_edge")
)
def test_gizmo_mesh_is_well_formed_and_cached(name: str) -> None:
    md = gizmo_mesh(name)
    assert md is gizmo_mesh(name)
    assert md.triangle_count > 0
    assert len(md.indices) % 3 == 0
    assert int(md.indices.max()) < len(md.positions)
    assert np.linalg.norm(md.normals, axis=1) == pytest.approx(1.0, abs=1e-5)
    tri = _triangles(md)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert area2.min() > 1e-9


@pytest.mark.parametrize(
    "name", ("arrow", "plane", "ring", "half_ring", "screen_ring", "screen_ring_edge")
)
def test_gizmo_mesh_winding_matches_its_normals(name: str) -> None:
    md = gizmo_mesh(name)
    triangles = md.indices.reshape(-1, 3)
    p = md.positions[triangles]
    face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal = md.normals[triangles].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", face, normal) > 0.0)


def test_gizmo_arrow_is_one_continuous_silhouette() -> None:

    from mojive.gizmo import (
        AXIS_HEAD_HALF_PT,
        AXIS_HEAD_LENGTH_PT,
        AXIS_SHAFT_HALF_PT,
        SIZE_PT,
    )

    p = gizmo_mesh("arrow").positions
    head_base = 1.0 - AXIS_HEAD_LENGTH_PT / SIZE_PT
    at_join = p[np.isclose(p[:, 2], head_base)]
    radii = np.linalg.norm(at_join[:, :2], axis=1)
    assert radii.min() * SIZE_PT == pytest.approx(AXIS_SHAFT_HALF_PT, abs=1e-6)
    assert radii.max() * SIZE_PT == pytest.approx(AXIS_HEAD_HALF_PT, abs=1e-6)
    assert np.unique(np.round(p[:, 2], 5)) == pytest.approx(np.round((0.0, head_base, 1.0), 5))


def test_solid_gizmo_ring_widths_match_the_flat_overlay() -> None:
    from mojive.gizmo import CONTRAST_EDGE_PT, RING_WIDTH_PT, SCREEN_RING_WIDTH_PT, SIZE_PT

    for name, width in (
        ("ring", RING_WIDTH_PT),
        ("screen_ring", SCREEN_RING_WIDTH_PT),
        ("screen_ring_edge", SCREEN_RING_WIDTH_PT + 2.0 * CONTRAST_EDGE_PT),
    ):
        p = gizmo_mesh(name).positions
        radial = np.linalg.norm(p[:, :2], axis=1)
        assert (radial.max() - radial.min()) * SIZE_PT == pytest.approx(width, abs=1e-5)


def test_solid_gizmo_half_ring_has_round_caps() -> None:
    from mojive.gizmo import RING_TUBE

    points = gizmo_mesh("half_ring").positions

    assert points[:, 1].min() == pytest.approx(-RING_TUBE, abs=1e-6)
