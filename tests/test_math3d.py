from __future__ import annotations

import numpy as np
import pytest

from mojive import math3d


@pytest.mark.parametrize(
    "degrees",
    ((0.0, 0.0, 0.0), (20.0, -35.0, 70.0), (-120.0, 45.0, 179.0), (15.0, 90.0, 25.0)),
)
def test_euler_xyz_round_trips_through_the_rotation_matrix(degrees) -> None:
    matrix = math3d.euler_xyz_to_mat3(np.radians(degrees))
    rebuilt = math3d.euler_xyz_to_mat3(math3d.mat3_to_euler_xyz(matrix))
    assert rebuilt == pytest.approx(matrix, abs=1e-6)


@pytest.mark.parametrize(
    "bounds",
    ((-1.0, 1.0, -2.0, 2.0, 0.1, 10.0), (2.0, 7.0, -4.0, 3.0, 1.5, 40.0)),
)
def test_inverse_orthographic_box_inverts_the_projection(bounds) -> None:
    projection = math3d.ortho_box(*bounds)
    inverse = math3d.inverse_orthographic_box(*bounds)

    assert inverse @ projection == pytest.approx(np.eye(4), abs=1e-6)
    assert projection @ inverse == pytest.approx(np.eye(4), abs=1e-6)
