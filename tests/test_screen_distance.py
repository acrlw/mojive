"""Screen hit distances preserve topology and degenerate-segment behavior."""

import math

import numpy as np
import pytest

from mojive.gizmo import screen_path_distance, screen_polygon_distance


def reference_distance(point, path, closed):
    distances = []
    for i in range(len(path) if closed else len(path) - 1):
        a, b = path[i], path[(i + 1) % len(path)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = (
            min(1.0, max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
            if length2 > 1e-12
            else 0.0
        )
        distances.append(math.hypot(point[0] - a[0] - dx * t, point[1] - a[1] - dy * t))
    return min(distances, default=float("inf"))


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("count", [2, 4, 33, 128, 512])
def test_batched_distance_matches_scalar_segments_without_mutating_inputs(closed, count):
    rng = np.random.default_rng(count)
    # A sliced read-only path also exercises the projected XYZ-to-XY view.
    path = rng.uniform(-100, 100, (count, 3))[:, :2]
    path[1] = path[0]
    path.setflags(write=False)
    for point in rng.uniform(-150, 150, (20, 2)):
        point.setflags(write=False)
        assert screen_path_distance(point, path, closed=closed) == pytest.approx(
            reference_distance(point, path, closed), abs=1e-11
        )


def test_open_path_does_not_gain_a_closing_edge():
    path = [(0, 0), (0, 10), (10, 10)]
    assert screen_path_distance((5, 5), path) == pytest.approx(5)
    assert screen_path_distance((5, 5), path, closed=True) == pytest.approx(0)
    assert math.isinf(screen_path_distance((0, 0), []))
    assert math.isinf(screen_path_distance((0, 0), [(0, 0)]))
    assert screen_path_distance((0, 0), [(3, 4), (3, 4)]) == pytest.approx(5)


@pytest.mark.parametrize("reverse", [False, True])
def test_polygon_distance_preserves_concave_notch_and_interior(reverse):
    polygon = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    if reverse:
        polygon.reverse()
    assert screen_polygon_distance((0.5, 3), polygon) == 0
    assert screen_polygon_distance((2, 2), polygon) == pytest.approx(1)
    assert screen_polygon_distance((1, 1), polygon) == 0
    assert math.isinf(screen_polygon_distance((0, 0), [(0, 0), (1, 1)]))
