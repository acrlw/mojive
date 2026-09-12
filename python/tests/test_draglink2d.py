"""Continuity, topology, and approximation checks for implicit drag links."""

import numpy as np
import pytest

from mojive.drawing.curves import CURVE_TOLERANCE


def test_implicit_union_matches_three_derivatives_at_its_blend_limits():
    from numpy.polynomial import Polynomial

    from mojive.drawing.drag_link import smooth_union

    x = np.linspace(-1.0, 1.0, 25)
    p = Polynomial.fit(x, smooth_union(x, 0.0, 1.0), 6).convert()
    for point, expected in ((-1.0, (-1.0, 1.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0, 0.0))):
        assert [p.deriv(order)(point) for order in range(4)] == pytest.approx(expected, abs=1e-10)
    assert smooth_union(x, 0.0, 0.0) == pytest.approx(np.minimum(x, 0.0))


@pytest.mark.parametrize("smoothing", (0.0, 0.05, 0.6, 1.0))
@pytest.mark.parametrize("distance", (0.0, 5.0, 8.0, 20.0, 100.0))
def test_drag_link_mesh_follows_the_field_and_keeps_the_hole_empty(smoothing, distance):
    from mojive.drawing.drag_link import drag_link_field, smooth_drag_link_mesh

    vertices, indices, outer, hole = smooth_drag_link_mesh(distance, 5.0, 2.0, smoothing)
    for contour in (outer, hole):
        if not contour:
            continue
        points = np.asarray(contour)
        values = drag_link_field(points[:, 0], points[:, 1], distance, 5.0, 2.0, smoothing)
        assert max(abs(values)) < 0.001
    triangles = np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)]
    centers = triangles.mean(axis=1)
    field = drag_link_field(centers[:, 0], centers[:, 1], distance, 5.0, 2.0, smoothing)
    assert field.max() < CURVE_TOLERANCE


def test_stretching_a_long_drag_link_reuses_the_same_mesh_topology():
    from mojive.drawing.drag_link import smooth_drag_link_mesh

    a = smooth_drag_link_mesh(100.0, 5.0, 2.0)
    b = smooth_drag_link_mesh(10000.0, 5.0, 2.0)
    assert len(a[0]) == len(b[0])
    assert a[1] is b[1]
    assert a[3] == b[3]
