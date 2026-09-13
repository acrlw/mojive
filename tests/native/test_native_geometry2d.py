"""Projected error bounds, independent native compilation, and buffer ownership."""

import gc
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mojive.geometry2d import Affine2D, PathBuilder2D, flatten_path


def distance_to_polyline(points, line):
    start, vector = line[:-1], np.diff(line, axis=0)
    length2 = (vector * vector).sum(axis=1)
    delta = points[:, None] - start
    fraction = np.divide(
        (delta * vector).sum(axis=2), length2, out=np.zeros(delta.shape[:2]), where=length2 > 0
    )
    nearest = start + fraction.clip(0, 1)[..., None] * vector
    return np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1)


@pytest.mark.parametrize(
    "transform",
    [
        Affine2D(),
        Affine2D.scale(-12, 1.5),
        Affine2D((4, 8, 32, 2, 1, -17)),
    ],
)
def test_curve_error_is_bounded_after_full_projection(native, transform):
    path = PathBuilder2D().move_to(0, 0).cubic_to(0, 50, 50, 50, 50, 0).finish()
    (contour,) = flatten_path(path, projection=transform, tolerance=0.1)
    t = np.linspace(0, 1, 2001)
    exact = np.column_stack(
        (150 * (1 - t) * t * t + 50 * t**3, 150 * (1 - t) ** 2 * t + 150 * (1 - t) * t * t)
    )
    error = distance_to_polyline(transform.apply(exact), transform.apply(contour.points))
    assert error.max() <= 0.1 + 1e-8


def test_arc_error_closed_paths_and_immutable_result(native):
    path = PathBuilder2D().move_to(10, 0).arc(0, 0, 10, 3, 0, math.tau).close().finish()
    (contour,) = flatten_path(path, projection=Affine2D.scale(8, 2), tolerance=0.2)
    points = contour.points
    assert contour.closed
    assert points.dtype == np.float64 and not points.flags.writeable
    with pytest.raises(ValueError):
        points.flags.writeable = True
    t = np.linspace(0, math.tau, 2001)
    exact = np.column_stack((80 * np.cos(t), 6 * np.sin(t)))
    line = Affine2D.scale(8, 2).apply(points)
    line = np.vstack((line, line[0]))
    assert distance_to_polyline(exact, line).max() <= 0.2 + 1e-8
    before = points.copy()
    del path, contour
    gc.collect()
    np.testing.assert_array_equal(points, before)


def test_viewport_translation_never_changes_curve_quality(native):
    path = PathBuilder2D().move_to(0, 0).cubic_to(0, 50, 50, 50, 50, 0).finish()
    base = flatten_path(path, tolerance=0.01)[0].points
    moved = flatten_path(path, projection=Affine2D.translation(1e15, -1e15), tolerance=0.01)[
        0
    ].points
    np.testing.assert_array_equal(moved, base)


def test_large_quadratic_coordinates_do_not_overflow_intermediate_weights(native):
    path = PathBuilder2D().move_to(1e308, 0).quadratic_to(1e308, 1e308, 0, 1e308).finish()
    result = flatten_path(path, projection=Affine2D.scale(1e-308))[0]
    assert np.isfinite(result.points).all()


def test_budget_and_native_input_fail_before_returning_partial_geometry(native):
    path = PathBuilder2D().move_to(0, 0).cubic_to(0, 50, 50, 50, 50, 0).finish()
    with pytest.raises(ValueError, match="budget"):
        flatten_path(path, max_points=2)
    with pytest.raises(ValueError):
        flatten_path(path, tolerance=0)
    with pytest.raises(TypeError):
        native.flatten_path(path.packed().astype(np.float32))
    invalid = path.packed()
    invalid[0, 0] = 0.5
    with pytest.raises(ValueError, match="verb"):
        native.flatten_path(invalid)


def test_cpu_extension_compiles_without_importing_a_graphics_stack(native):
    program = """
import sys
from mojive.geometry2d import PathBuilder2D, flatten_path
path = PathBuilder2D().move_to(0, 0).quadratic_to(5, 10, 10, 0).finish()
assert len(flatten_path(path)[0].points) > 2
blocked = ('imgui_bundle', 'glfw', 'mujoco', 'moderngl', 'wgpu')
assert not any(name in sys.modules for name in blocked)
"""
    env = dict(os.environ, MOJIVE_NATIVE_BUILD=str(Path(native.__file__).parents[2]))
    subprocess.run([sys.executable, "-c", program], env=env, check=True)
