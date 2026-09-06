"""Continuity, bounded approximation, and shared shape geometry."""

import math

import numpy as np
import pytest

from mojive.curves2d import (
    CORNER_SMOOTHING,
    CURVE_TOLERANCE,
    _ellipse_advance,
    _ellipse_frame,
    clip_polygon_rect,
    smooth_capsule_points,
    smooth_ellipse_stroke,
    smooth_line_cap,
    smooth_rect_points,
    smooth_turn_points,
    turn_curvature,
)
from tests.curve_assertions import assert_paths_close, distance_to_path


@pytest.mark.parametrize("angle", (math.pi / 7, math.pi / 2, -math.pi / 2, math.pi))
@pytest.mark.parametrize("q", (0.1, 0.6, 1.0))
def test_turn_has_g3_joins_and_exact_total_rotation(angle, q):
    phi = abs(angle)
    total = (1 + q) * phi
    joins = np.array((0.0, q * phi, phi, total))
    theta, curvature, slope = turn_curvature(joins, angle, q)
    sign = math.copysign(1, angle)
    assert theta == pytest.approx(sign * np.array((0, q * phi / 2, phi - q * phi / 2, phi)))
    assert curvature == pytest.approx(sign * np.array((0, 1, 1, 0)), abs=1e-13)
    assert slope == pytest.approx(np.zeros(4), abs=1e-12)
    # Both one-sided limits agree, including where the circular interval vanishes.
    for epsilon in (-1e-8, 1e-8):
        t, k, derivative = turn_curvature(joins + epsilon, angle, q)
        assert t == pytest.approx(theta, abs=1e-7)
        assert k == pytest.approx(curvature, abs=1e-12)
        assert derivative == pytest.approx(slope, abs=4e-5)


@pytest.mark.parametrize("angle", (0.4, math.pi / 2, -math.pi, math.pi))
@pytest.mark.parametrize("scale", (0.1, 1.0, 26.0, 104.0))
def test_turn_chord_error_against_dense_independent_arc_length_integration(angle, scale):
    # Composite trapezoids on a dense uniform arc-length grid are independent
    # of the production Gaussian quadrature and adaptive interval layout.
    s = np.linspace(0, (1 + CORNER_SMOOTHING) * abs(angle), 16001)
    theta = turn_curvature(s, angle)[0]
    tangent = np.column_stack((np.cos(theta), np.sin(theta)))
    reference = (
        np.vstack(
            (
                np.zeros(2),
                np.cumsum((tangent[1:] + tangent[:-1]) * 0.5 * np.diff(s)[:, None], axis=0),
            )
        )
        * scale
    )
    path = smooth_turn_points(angle, scale)
    assert np.asarray(path)[-1] == pytest.approx(reference[-1], abs=1e-6)
    assert distance_to_path(reference, path, closed=False).max() <= CURVE_TOLERANCE


@pytest.mark.parametrize("ratio", (1, 1.0001, 1.1, 1.8, 4.0))
def test_capsule_bounds_symmetry_and_non_overlapping_edges(ratio):
    points = np.asarray(smooth_capsule_points(3, 7, 40 * ratio, 40))
    assert points.min(axis=0) == pytest.approx((3, 7), abs=1e-8)
    assert points.max(axis=0) == pytest.approx((3 + 40 * ratio, 47), abs=1e-8)
    assert_paths_close((6 + 40 * ratio, 54) - points, points, 1e-8)
    edges = np.roll(points, -1, axis=0) - points
    assert np.linalg.norm(edges, axis=1).min() > 1e-9
    turn = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    assert turn.min() >= -1e-8


def test_rectangle_does_not_switch_shape_when_radius_saturates():
    almost = smooth_rect_points(0, 0, 100, 40, 20 - 1e-7)
    saturated = smooth_rect_points(0, 0, 100, 40, 20)
    assert_paths_close(almost, saturated, 1e-6)
    assert saturated == smooth_rect_points(0, 0, 100, 40, 1000)


def test_short_line_cap_preserves_tip_and_fits_its_available_straight_edge():
    cap = smooth_line_cap((10, 0), (1, 0), 4, max_inset=0.01)
    assert cap[:, 0].max() == pytest.approx(12)
    assert cap[0] == pytest.approx((9.99, 2), abs=1e-8)
    assert cap[-1] == pytest.approx((9.99, -2), abs=1e-8)


def test_short_cap_samples_only_the_selected_profile(monkeypatch):
    from mojive import curves2d

    calls = []
    original = curves2d.smooth_turn_points

    def sample(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    curves2d._cap_geometry.cache_clear()
    monkeypatch.setattr(curves2d, "smooth_turn_points", sample)
    direction = np.array((3.0, 0.0))
    cap = smooth_line_cap((10, 0), direction, 4, max_inset=0.012345)
    assert cap[0, 0] == pytest.approx(10 - 0.012345, abs=1e-8)
    assert len(calls) == 1
    assert np.array_equal(direction, (3.0, 0.0))


@pytest.mark.parametrize("shape", ("capsule", "rectangle"))
def test_translation_reuses_shape_without_resampling(monkeypatch, shape):
    from mojive import curves2d

    def path(x, y):
        if shape == "capsule":
            return smooth_capsule_points(x, y, 43.25, 39.75)
        return smooth_rect_points(x, y, x + 43.25, y + 39.75, 4.625)

    reference = np.asarray(path(0, 0))

    def unexpected(*args, **kwargs):
        pytest.fail("translation must not resample the corner profile")

    monkeypatch.setattr(curves2d, "smooth_turn_points", unexpected)
    for x in range(70):
        assert np.asarray(path(x, -x)) == pytest.approx(reference + np.array((x, -x)), abs=1e-12)


@pytest.mark.parametrize("q", (0.05, 0.2, 0.6, 1.0))
@pytest.mark.parametrize("radius", (2.0, 10.0))
def test_view_gizmo_neck_matches_straight_and_circle_through_g3(q, radius):
    from mojive.curves2d import _lollipop_bridge

    p, angle = _lollipop_bridge(radius, radius * 0.1, radius * 0.6 * q, q)
    derivatives = [p]
    for degree in (7, 6, 5):
        derivatives.append(degree * np.diff(derivatives[-1], axis=0))
    for endpoint, curvature in ((0, 0.0), (-1, 1.0 / radius)):
        v, a, j = (values[endpoint] for values in derivatives[1:])
        speed = np.linalg.norm(v)
        cross = v[0] * a[1] - v[1] * a[0]
        k = cross / speed**3
        slope = (v[0] * j[1] - v[1] * j[0]) / speed**4 - 3 * cross * (v @ a) / speed**6
        assert k == pytest.approx(curvature, abs=1e-8)
        assert slope == pytest.approx(0.0, abs=2e-6)
    assert derivatives[1][0, 1] == 0.0
    expected_tangent = np.array((-math.sin(angle), math.cos(angle)))
    assert derivatives[1][-1] / np.linalg.norm(derivatives[1][-1]) == pytest.approx(
        expected_tangent
    )
    # Positive x derivatives throughout the control hull exclude cusps and loops.
    assert np.min(derivatives[1][:, 0]) > 0.0


@pytest.mark.parametrize("q", (0.0, 0.2, 0.6, 1.0))
def test_short_capsule_has_no_edges_that_collapse_in_imgui(q):
    points = np.asarray(smooth_capsule_points(100, 200, 88, 54, q), np.float32)
    assert np.min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) > 0.0
    assert points.min(axis=0) == pytest.approx((100, 200))
    assert points.max(axis=0) == pytest.approx((188, 254))


def test_clipped_progress_fill_is_inside_the_same_full_boundary():
    full = np.asarray(smooth_rect_points(0, 0, 100, 40, 12))
    clipped = np.asarray(clip_polygon_rect(full, (3, 0, 64, 40)))
    assert clipped[:, 0].min() >= 3 and clipped[:, 0].max() <= 64
    interior = clipped[(clipped[:, 0] > 3) & (clipped[:, 0] < 64)]
    assert distance_to_path(interior, full).max() < 1e-10


def test_ellipse_tubular_map_has_regular_offsets_and_accurate_arc_lengths():
    a, b = np.array((3.177, 4.765)), np.array((2.594, -2.594))
    offsets = np.linspace(-0.5, 0.5, 9)
    t = _ellipse_advance(a, b, 0.2, offsets)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    half = (t - 0.2) * 0.5
    distance = half * (_ellipse_frame(a, b, 0.2 + half[:, None] * (1 + nodes))[3] @ weights)
    assert distance == pytest.approx(offsets, abs=1e-12)
    # The coordinate map's Jacobian stays away from zero across the stroke.
    for t0 in np.linspace(-0.3, math.pi + 0.3, 100):
        speed = _ellipse_frame(a, b, t0)[3]
        curvature = np.linalg.det(np.column_stack((a, b))) / speed**3
        assert 1 - abs(curvature) * 0.62 > 0.5
    coarse = smooth_ellipse_stroke(tuple(a), tuple(b), 1.24)
    fine = smooth_ellipse_stroke(tuple(a), tuple(b), 1.24, tolerance=0.0005)
    assert_paths_close(coarse, fine)


@pytest.mark.parametrize("smoothing", (0.0, 0.6, 1.0))
def test_drag_connector_does_not_dent_the_circular_origin_hole(smoothing):
    from mojive.draglink2d import drag_link_field

    angles = np.linspace(0.0, 2.0 * np.pi, 200)
    xy = 4.0 * np.column_stack((np.cos(angles), np.sin(angles)))
    assert drag_link_field(xy[:, 0], xy[:, 1], 100.0, 5.0, 2.0, smoothing) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("scale", (0.75, 1.0, 4.0))
@pytest.mark.parametrize("level", (0.0, 0.75))
@pytest.mark.parametrize("smoothing", (0.0, 0.05, 0.6, 1.0))
def test_drag_hole_topology_changes_preserve_the_field_boundary(scale, level, smoothing):
    from mojive.draglink2d import drag_link_field, smooth_drag_link_mesh

    threshold = 1.0 + 2.0 * level + 5.0 * smoothing * 5.0 / 16.0
    for distance in (threshold - 0.001, threshold + 0.001, 4.9, 6.3, 9.7, 30.0):
        vertices, indices, outer, hole = smooth_drag_link_mesh(
            distance * scale, 5.0 * scale, 2.0 * scale, smoothing, level * scale
        )
        for contour in (outer, hole):
            if not contour:
                continue
            points = np.asarray(contour)
            residual = (
                drag_link_field(
                    points[:, 0],
                    points[:, 1],
                    distance * scale,
                    5.0 * scale,
                    2.0 * scale,
                    smoothing,
                )
                - level * scale
            )
            assert np.abs(residual).max() < 0.001
        centers = np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)].mean(axis=1)
        residual = (
            drag_link_field(
                centers[:, 0], centers[:, 1], distance * scale, 5.0 * scale, 2.0 * scale, smoothing
            )
            - level * scale
        )
        assert residual.max() < CURVE_TOLERANCE


@pytest.mark.parametrize("smoothing", (0.0, 0.6, 1.0))
@pytest.mark.parametrize("radius", (0.0, 0.5, 3.0))
def test_arrow_fan_has_no_inverted_faces_or_internal_antialiasing(smoothing, radius):
    from mojive.curves2d import arrow_triangles

    mesh = arrow_triangles(40.0, 4.0, 12.0, 14.0, radius, smoothing, False, True)
    solid = mesh[np.all(mesh[:, :, 2] == 1.0, axis=1), :, :2]
    ab, ac = solid[:, 1] - solid[:, 0], solid[:, 2] - solid[:, 0]
    assert np.all(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0] >= -1e-8)
    assert len(mesh) == 3 * len(solid)
    assert not mesh.flags.writeable


@pytest.mark.parametrize("length", (0.1, 1.0, 2.0))
@pytest.mark.parametrize("round_tail", (False, True))
def test_foreshortened_arrow_keeps_an_antialias_contour(length, round_tail):
    from mojive.curves2d import arrow_points, polygon_fringe

    points = arrow_points((20, 30), (20 + length, 30), round_tail=round_tail)
    fringe = polygon_fringe(points)
    assert len(fringe) == len(points)
    assert np.all(np.isfinite(fringe))


@pytest.mark.parametrize("round_tail", (False, True))
def test_zero_length_arrow_head_is_a_plain_shaft(round_tail):
    from mojive.curves2d import arrow_points, arrow_triangles

    path = arrow_points((0, 0), (20, 0), 2, head_length=0, round_tail=round_tail)
    assert path[:, 1].min() == pytest.approx(-1.0)
    assert path[:, 1].max() == pytest.approx(1.0)
    mesh = arrow_triangles(20, 2, 0, 8, 0.5, 0.6, round_tail, True)
    assert np.isfinite(mesh).all()


@pytest.mark.parametrize(
    "name", ("width", "head_length", "head_width", "corner_radius", "smoothing")
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0))
def test_arrow_rejects_invalid_style_at_the_shared_geometry_boundary(name, value):
    from mojive.curves2d import arrow_points

    with pytest.raises(ValueError, match="arrow dimensions"):
        arrow_points((0, 0), (20, 0), **{name: value})


def test_arc_translation_reuses_profiles_and_preserves_caller_radials():
    from mojive.curves2d import _arc_ribbon_shape, arc_ribbon_points

    points = np.array(((2, 3), (6, 4), (10, 8)), dtype=float)
    radial = np.array((3.0, 4.0))
    _arc_ribbon_shape.cache_clear()
    a = arc_ribbon_points(points, radial, radial, 2, round_caps=True)
    b = arc_ribbon_points(np.add(points, (120, 50)), radial, radial, 2, round_caps=True)
    assert b == pytest.approx(np.add(a, (120, 50)))
    assert radial == pytest.approx((3, 4))
    assert _arc_ribbon_shape.cache_info().misses == 1
    assert _arc_ribbon_shape.cache_info().hits == 1


@pytest.mark.parametrize("smoothing", (0.0, 0.6, 1.0))
def test_arrow_join_radius_is_independent_of_tip_radius(smoothing):
    from mojive.curves2d import arrow_points

    options = {
        "width": 4,
        "head_length": 12,
        "head_width": 14,
        "corner_radius": 0.5,
        "smoothing": smoothing,
    }
    rounded = arrow_points((0, 0), (100, 0), **options, join_radius=4)
    sharp_join = arrow_points((0, 0), (100, 0), **options, join_radius=0)
    for shoulder in ((88, -2), (88, 2)):
        assert distance_to_path([shoulder], rounded)[0] > 0.4
        assert distance_to_path([shoulder], sharp_join)[0] < 1e-8
    # Changing the concave transition leaves the forward tip profile untouched.
    assert_paths_close(rounded[rounded[:, 0] > 90], sharp_join[sharp_join[:, 0] > 90], closed=False)


@pytest.mark.parametrize("smoothing", (0.0, 0.6, 1.0))
def test_default_arrow_shoulders_are_subtler_than_the_head_corners(smoothing):
    from mojive.curves2d import arrow_points

    path = arrow_points(
        (0, 0),
        (100, 0),
        4,
        head_length=12,
        head_width=14,
        corner_radius=1.2,
        smoothing=smoothing,
    )
    shoulder, wing, tip = distance_to_path([(88, -2), (88, -7), (100, 0)], path)
    assert 0.1 < shoulder < min(wing, tip) * 0.5
    assert min(wing, tip) > 0.7


@pytest.mark.parametrize("smoothing", (0.001, 0.1, 0.6, 1.0))
@pytest.mark.parametrize("fraction", (0.000001, 0.1, 0.5, 0.9, 0.999999))
def test_cap_fit_matches_high_precision_bracketing(smoothing, fraction):
    from mojive.curves2d import _cap_depth_ratio, _fit_cap_smoothing

    target = 0.5 + (_cap_depth_ratio(smoothing) - 0.5) * fraction
    lo, hi = 0.0, smoothing
    for _ in range(50):
        mid = (lo + hi) * 0.5
        if _cap_depth_ratio(mid) > target:
            hi = mid
        else:
            lo = mid
    fitted = _fit_cap_smoothing(target, smoothing)
    assert fitted == pytest.approx(lo, abs=2e-10)
    assert abs(_cap_depth_ratio(fitted) - target) < 2e-12
    assert 0 <= fitted <= smoothing


@pytest.mark.parametrize("smoothing", (0.001, 0.1, 0.6, 0.999))
def test_cap_depth_derivative_matches_finite_difference(smoothing):
    from mojive.curves2d import _cap_depth_and_slope, _cap_depth_ratio

    step = 1e-5
    numerical = (_cap_depth_ratio(smoothing + step) - _cap_depth_ratio(smoothing - step)) / (
        2 * step
    )
    assert _cap_depth_and_slope(smoothing)[1] == pytest.approx(numerical, rel=1e-8)


@pytest.mark.parametrize("length", (1.0, 5.0, 5.1, 7.0, 60.0))
@pytest.mark.parametrize("width", (2.5, 10.0))
def test_box_handle_shortening_preserves_valid_bounds(length, width):
    from mojive.curves2d import box_handle_points

    shape = box_handle_points((0, 0), (length, 0), width, 10, corner_radius=1)
    assert np.isfinite(shape).all()
    assert shape.min(axis=0) == pytest.approx((min(0, length - 5), -5))
    assert shape.max(axis=0) == pytest.approx((length + 5, 5))
    assert np.linalg.norm(shape - np.roll(shape, 1, axis=0), axis=1).min() > 1e-8


@pytest.mark.parametrize("sample_count", (2, 8, 64, 256))
@pytest.mark.parametrize("caps", ((False, False), (True, False), (False, True), (True, True)))
@pytest.mark.parametrize("sweep", (0.05, 1.5, -4.0))
def test_arc_strips_cover_boundary_once_without_internal_fringes(sample_count, caps, sweep):
    from collections import Counter

    from mojive.curves2d import arc_ribbon_mesh

    theta = np.linspace(0, sweep, sample_count)
    points = np.column_stack((np.cos(theta), np.sin(theta))) * 60
    vertices, indices = arc_ribbon_mesh(
        points, None, None, 4, round_start=caps[0], round_end=caps[1]
    )
    triangles = np.asarray(indices).reshape(-1, 3)
    assert triangles.min() >= 0 and triangles.max() < len(vertices)
    xyz = vertices[triangles]
    edge1, edge2 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    areas = (edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]) * 0.5
    polygon_area = (
        np.sum(
            vertices[:, 0] * np.roll(vertices[:, 1], -1)
            - vertices[:, 1] * np.roll(vertices[:, 0], -1)
        )
        * 0.5
    )
    assert np.abs(areas).sum() == pytest.approx(abs(polygon_area), abs=1e-7)
    edges = Counter(
        tuple(sorted(pair))
        for t in triangles
        for pair in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))
    )
    boundary = {tuple(sorted((i, (i + 1) % len(vertices)))) for i in range(len(vertices))}
    assert {edge for edge, count in edges.items() if count == 1} == boundary
    assert all(count in (1, 2) for count in edges.values())
