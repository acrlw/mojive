"""Environment geometry shared across renderer implementations."""

from types import SimpleNamespace

import numpy as np
import pytest

from mojive.render.environment import classic_skybox_vertices, reflection_plane
from mojive.types import MeshShape


@pytest.mark.parametrize("shape,offset", ((MeshShape.PLANE, -5.0), (MeshShape.BOX, -9.0)))
def test_reflection_plane_uses_transformed_surface_not_geometry_center(shape, offset):
    transform = np.diag((2.0, 3.0, 4.0, 1.0))
    transform[2, 3] = 5.0
    scene = SimpleNamespace(
        transforms=[transform], bucket=[0], bucket_keys=[(SimpleNamespace(shape=shape),)]
    )
    assert reflection_plane(scene, 0) == pytest.approx((0, 0, 1, offset))
    transform[2, 2] = 0
    assert reflection_plane(scene, 0) is None


def test_classic_skybox_geometry_has_closed_caps_and_nondegenerate_triangles():
    vertices = classic_skybox_vertices(64)
    assert vertices.shape == (64 * 12, 3)
    assert vertices.dtype == np.float32
    assert set(vertices[:, 2]) == {-1, 1}
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    assert np.all(np.isclose(radii, 0) | np.isclose(radii, 1))
    triangles = vertices.reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.linalg.norm(normals, axis=1) > 0)
