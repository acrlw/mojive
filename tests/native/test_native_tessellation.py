"""Fill topology against analytic areas and independent winding-number samples."""

import gc

import numpy as np
import pytest

from mojive.geometry2d import (
    Contour2D,
    PathBuilder2D,
    compile_fill,
    tessellate_contours,
)


def contour(points):
    return Contour2D(np.asarray(points, dtype=np.float64).reshape(-1, 2), True)


def area(mesh):
    triangles = mesh.positions[mesh.indices.reshape(-1, 3)]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    return ((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2).sum()


def winding_samples(points, contours):
    winding = np.zeros(len(points), dtype=int)
    for part in contours:
        for a, b in zip(part.points, np.roll(part.points, -1, axis=0), strict=True):
            cross = (b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])
            upward = (a[1] <= points[:, 1]) & (points[:, 1] < b[1]) & (cross > 0)
            downward = (b[1] <= points[:, 1]) & (points[:, 1] < a[1]) & (cross < 0)
            winding += upward.astype(int) - downward.astype(int)
    return winding


def triangle_samples(points, mesh):
    hits = np.zeros(len(points), dtype=int)
    for triangle in mesh.positions[mesh.indices.reshape(-1, 3)]:
        inside = np.ones(len(points), dtype=bool)
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            cross = (b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])
            inside &= cross >= 0
        hits += inside
    return hits


@pytest.mark.parametrize(
    "fill_rule,reverse,expected",
    [("nonzero", False, 100), ("nonzero", True, 84), ("evenodd", False, 84)],
)
def test_holes_winding_and_external_boundary_edges(native, fill_rule, reverse, expected):
    outer = contour([(0, 0), (10, 0), (10, 10), (0, 10)])
    inner = contour([(3, 3), (7, 3), (7, 7), (3, 7)])
    if reverse:
        inner = contour(inner.points[::-1])
    mesh = tessellate_contours([outer, inner], fill_rule=fill_rule)
    assert area(mesh) == pytest.approx(expected, abs=1e-5)
    edges = mesh.positions[mesh.boundary_edges]
    perimeter = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).sum()
    assert perimeter == pytest.approx(56 if expected == 84 else 40)
    counts = {}
    for tri in mesh.indices.reshape(-1, 3):
        for a, b in zip(tri, np.roll(tri, -1), strict=True):
            edge = tuple(sorted((a, b)))
            counts[edge] = counts.get(edge, 0) + 1
    expected_boundary = {edge for edge, count in counts.items() if count == 1}
    assert {tuple(sorted(e)) for e in mesh.boundary_edges} == expected_boundary


@pytest.mark.parametrize("rule", ["evenodd", "nonzero"])
def test_self_intersections_and_overlap_match_independent_coverage(native, rule):
    parts = [
        contour([(0, 0), (8, 8), (0, 8), (8, 0)]),
        contour([(2, -1), (6, -1), (6, 9), (2, 9)]),
    ]
    mesh = tessellate_contours(parts, fill_rule=rule)
    samples = np.random.default_rng(17).uniform((-1, -2), (9, 10), (10000, 2))
    winding = winding_samples(samples, parts)
    expected = (winding % 2 != 0) if rule == "evenodd" else winding != 0
    hits = triangle_samples(samples, mesh)
    np.testing.assert_array_equal(hits, expected.astype(int))


def test_large_translation_mirror_and_degenerate_contours(native):
    points = np.array([(0, 0), (7, 0), (7, 5), (0, 5)], dtype=np.float64)
    transform = np.array([[-2, 3], [0, 4]])
    part = contour(points @ transform.T + 1e9)
    assert area(tessellate_contours([part])) == pytest.approx(35 * 8, abs=2e-5)
    for points in ([], [(1, 1)], [(0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)]):
        mesh = tessellate_contours([contour(points)])
        assert mesh.indices.size == mesh.boundary_edges.size == 0


def test_compile_fill_keeps_authored_rule_and_owns_immutable_arrays(native):
    builder = PathBuilder2D(fill_rule="evenodd")
    for x0, x1 in ((0, 10), (3, 7)):
        builder.move_to(x0, x0).line_to(x1, x0).line_to(x1, x1).line_to(x0, x1).close()
    mesh = compile_fill(builder.finish())
    assert area(mesh) == pytest.approx(84, abs=1e-5)
    saved = mesh.positions.copy()
    builder.move_to(-10, -10).line_to(90, 90)
    gc.collect()
    np.testing.assert_array_equal(mesh.positions, saved)
    for array in (mesh.positions, mesh.indices, mesh.boundary_edges):
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_input_and_output_budgets_reject_without_poisoning_later_compiles(native):
    square = contour([(0, 0), (10, 0), (10, 10), (0, 10)])
    baseline = tessellate_contours([square])
    for scratch in range(1, baseline.scratch_peak_bytes, 379):
        with pytest.raises(RuntimeError, match="budget"):
            tessellate_contours([square], max_scratch_bytes=scratch)
    for limits in ({"max_vertices": 3}, {"max_indices": 3}):
        with pytest.raises(ValueError, match="budget"):
            tessellate_contours([square], **limits)
    after = tessellate_contours([square], max_scratch_bytes=baseline.scratch_peak_bytes)
    assert area(after) == 100
    np.testing.assert_array_equal(after.indices, baseline.indices)
    with pytest.raises(ValueError, match="offsets"):
        native.tessellate_contours(square.points, np.array([1, 4], dtype=np.uint32))
    with pytest.raises(TypeError):
        native.tessellate_contours(
            square.points.astype(np.float32), np.array([0, 4], dtype=np.uint32)
        )
