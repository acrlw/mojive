from __future__ import annotations

import numpy as np
import pytest

from mojive.geometry import geometry_dimensions, geometry_size_from_dimensions
from mojive.types import MeshShape


@pytest.mark.parametrize(
    ("shape", "size", "values", "axes", "factors"),
    (
        (MeshShape.PLANE, (0.5, 1.0, 0.1), (1.0, 2.0), (0, 1), (2.0, 2.0)),
        (MeshShape.BOX, (0.5, 1.0, 1.5), (1.0, 2.0, 3.0), (0, 1, 2), (2.0,) * 3),
        (MeshShape.SPHERE, (0.5, 0.5, 0.5), (0.5,), (None,), (1.0,)),
        (MeshShape.SPHERE, (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0, 1, 2), (1.0,) * 3),
        (MeshShape.CYLINDER, (0.5, 0.5, 1.5), (1.0, 3.0), (0, 2), (2.0, 2.0)),
        (MeshShape.CAPSULE_SHAFT, (0.5, 0.5, 1.5), (1.0, 3.0), (0, 2), (2.0, 2.0)),
    ),
)
def test_geometry_dimensions_define_only_independent_primitive_parameters(
    shape: MeshShape,
    size,
    values,
    axes,
    factors,
) -> None:
    editor = geometry_dimensions(shape, size)

    assert editor is not None
    assert editor.values == pytest.approx(values)
    assert tuple(handle.axis for handle in editor.handles) == axes
    assert tuple(handle.world_to_value for handle in editor.handles) == factors
    assert geometry_size_from_dimensions(shape, size, editor.values) == pytest.approx(size)


def test_mesh_dimensions_are_not_misrepresented_as_primitive_scale() -> None:
    assert geometry_dimensions(MeshShape.CONVEX_HULL, np.ones(3)) is None
