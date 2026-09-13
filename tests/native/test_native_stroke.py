"""Stroke coverage against analytic geometry, affine widths and failure budgets."""

import math
from itertools import pairwise

import numpy as np
import pytest

from mojive.geometry2d import Affine2D, PathBuilder2D, StrokeStyle, compile_stroke


def path(points, *, closed=False):
    builder = PathBuilder2D().move_to(*points[0])
    for point in points[1:]:
        builder.line_to(*point)
    if closed:
        builder.close()
    return builder.finish()


def area(mesh):
    triangles = mesh.positions[mesh.indices.reshape(-1, 3)]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    return ((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2).sum()


def coverage(mesh, points):
    hits = np.zeros(len(points), dtype=np.int32)
    for triangle in mesh.positions[mesh.indices.reshape(-1, 3)]:
        inside = np.ones(len(points), dtype=bool)
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            cross = (b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])
            inside &= cross >= 0
        hits += inside
    return hits


@pytest.mark.parametrize(
    "cap,expected", [("butt", 40), ("square", 56), ("round", 40 + 4 * math.pi)]
)
def test_caps_area_and_extrema(native, cap, expected):
    mesh = compile_stroke(path([(0, 0), (10, 0)]), StrokeStyle(4, cap=cap), tolerance=0.002)
    assert area(mesh) == pytest.approx(expected, abs=0.02)
    assert mesh.positions[:, 1].min() == pytest.approx(-2)
    assert mesh.positions[:, 1].max() == pytest.approx(2)
    assert mesh.positions[:, 0].min() == pytest.approx(0 if cap == "butt" else -2)
    assert mesh.positions[:, 0].max() == pytest.approx(10 if cap == "butt" else 12)


@pytest.mark.parametrize("join,expected", [("miter", 80), ("bevel", 78), ("round", 76 + math.pi)])
def test_joins_and_reversed_path(native, join, expected):
    points = [(0, 0), (10, 0), (10, 10)]
    for direction in (points, points[::-1]):
        mesh = compile_stroke(path(direction), StrokeStyle(4, join=join), tolerance=0.001)
        assert area(mesh) == pytest.approx(expected, abs=0.01)


def test_sharp_miter_limit_becomes_bevel_without_spikes(native):
    shape = path([(0, 0), (10, 0), (0, 0.1)])
    bevel = compile_stroke(shape, StrokeStyle(4, join="bevel"))
    limited = compile_stroke(shape, StrokeStyle(4, join="miter", miter_limit=4))
    assert area(limited) == pytest.approx(area(bevel))
    assert limited.positions[:, 0].max() < 11
    # A higher requested limit deliberately admits the long miter.
    unlimited = compile_stroke(shape, StrokeStyle(4, join="miter", miter_limit=300))
    assert unlimited.positions[:, 0].max() > 400


def test_self_crossing_round_stroke_is_one_coverage_union(native):
    points = np.array([(0, 0), (12, 12), (0, 12), (12, 0), (0, 0)], dtype=float)
    mesh = compile_stroke(path(points), StrokeStyle(3, cap="round", join="round"), tolerance=0.002)
    samples = np.random.default_rng(32).uniform(-2, 14, (9000, 2))
    distance = np.full(len(samples), np.inf)
    for a, b in pairwise(points):
        t = np.clip(((samples - a) @ (b - a)) / np.dot(b - a, b - a), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(samples - a - t[:, None] * (b - a), axis=1))
    keep = np.abs(distance - 1.5) > 0.003
    np.testing.assert_array_equal(coverage(mesh, samples[keep]), (distance[keep] < 1.5).astype(int))


def test_closed_stroke_preserves_hole_and_ignores_open_caps(native):
    shape = path([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], closed=True)
    for cap in ("butt", "round", "square"):
        mesh = compile_stroke(shape, StrokeStyle(2, cap=cap))
        assert area(mesh) == pytest.approx(80)
        assert coverage(mesh, np.array([(5, 5)]))[0] == 0


def test_full_affine_local_and_physical_screen_widths(native):
    shape = path([(0, 0), (10, 0)])
    projection = Affine2D((-3, 2, 20, 4, 1, 5))
    local = compile_stroke(shape, StrokeStyle(4), projection=projection)
    assert area(local) == pytest.approx(40)
    assert np.ptp(local.positions[:, 1]) == pytest.approx(4)
    screen = compile_stroke(shape, StrokeStyle(4, space="screen"), projection=projection)
    assert area(screen) == pytest.approx(50 * 4)
    # Projected centerline (length 50) has a physical 4-pixel width, including shear/reflection.
    perpendicular = np.array([-0.8, -0.6])
    assert np.ptp(screen.positions @ perpendicular) == pytest.approx(4, abs=1e-5)


def test_singular_transform_duplicate_points_and_zero_width(native):
    shape = path([(0, 0), (0, 0), (10, 0)])
    for cap, expected in (("butt", 0), ("round", 4 * math.pi), ("square", 16)):
        mesh = compile_stroke(
            shape,
            StrokeStyle(4, cap=cap, space="screen"),
            projection=Affine2D.scale(0),
            tolerance=0.001,
        )
        assert area(mesh) == pytest.approx(expected, abs=0.01)
    mesh = compile_stroke(shape, StrokeStyle(0, cap="round"))
    assert mesh.indices.size == mesh.boundary_edges.size == 0


def test_circle_quality_uses_affine_but_not_translation(native):
    shape = path([(0, 0)])
    style = StrokeStyle(4, cap="round")
    ordinary = compile_stroke(shape, style, tolerance=0.01)
    translated = compile_stroke(
        shape, style, projection=Affine2D.translation(1e15, -1e15), tolerance=0.01
    )
    np.testing.assert_array_equal(ordinary.positions, translated.positions)
    projected = compile_stroke(
        shape, style, projection=Affine2D((20, 30, 0, 0, 3, 0)), tolerance=0.01
    )
    assert len(projected.positions) > len(ordinary.positions)


def test_width_scale_is_proportional_below_one_pixel(native):
    shape = path([(0, 0), (10, 0)])
    for width in (0.001, 0.1, 1, 8):
        mesh = compile_stroke(shape, StrokeStyle(width, cap="round"), tolerance=width / 1000)
        assert np.ptp(mesh.positions[:, 1]) == pytest.approx(width, rel=2e-5)


def test_stroke_failures_do_not_poison_later_compilation(native):
    shape = path([(0, 0), (10, 10), (0, 10), (10, 0)])
    style = StrokeStyle(3, cap="round", join="round")
    for kwargs in (
        {"max_vertices": 8},
        {"max_indices": 3},
        {"max_scratch_bytes": 128},
        {"tolerance": 0},
        {"tolerance": 1e-100},
        {"max_vertices": True},
    ):
        with pytest.raises((ValueError, RuntimeError, OverflowError)):
            compile_stroke(shape, style, **kwargs)
    assert area(compile_stroke(shape, style)) > 0
    for width, cap, join in ((-1, 0, 0), (1, -1, 0), (1, 0, 5), (float("nan"), 0, 0)):
        with pytest.raises(ValueError):
            native.compile_stroke(shape.packed(), width, cap, join)
