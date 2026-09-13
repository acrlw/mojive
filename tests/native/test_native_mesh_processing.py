"""The optional simplifier preserves source ownership, surfaces and error bounds."""

import gc

import numpy as np
import pytest


def grid(side=21):
    y, x = np.mgrid[:side, :side]
    positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32) / (
        side - 1
    )
    normals = np.tile(np.array([0, 0, 1], np.float32), (len(positions), 1))
    a = (np.arange(side - 1)[:, None] * side + np.arange(side - 1)).ravel()
    indices = (
        np.column_stack((a, a + 1, a + side + 1, a, a + side + 1, a + side))
        .ravel()
        .astype(np.uint32)
    )
    return positions, normals, positions[:, :2].copy(), indices


def test_simplification_retains_planar_coverage_and_owned_indices(native):
    p, n, uv, original = grid()
    indices, error = native.simplify_mesh_indices(p, n, uv, original, 0.1, 0.005)
    assert len(indices) <= len(original) * 0.11
    assert error <= 0.005
    assert indices.dtype == np.uint32 and indices.flags.c_contiguous
    tri = p[indices.reshape(-1, 3)]
    area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2].sum() / 2
    assert area == pytest.approx(1.0)
    assert indices.max() < len(p)
    np.testing.assert_array_equal(original, grid()[3])
    before = indices.copy()
    del p, n, uv, original
    gc.collect()
    np.testing.assert_array_equal(indices, before)


def test_mesh_processing_rejects_invalid_input_before_native_algorithm(native):
    p, n, uv, indices = grid(3)
    for ratio, error in ((0, 0.01), (2, 0.01), (0.5, -1), (float("nan"), 0.01)):
        with pytest.raises(ValueError):
            native.simplify_mesh_indices(p, n, uv, indices, ratio, error)
    bad = indices.copy()
    bad[-1] = 10000
    with pytest.raises(ValueError, match="index"):
        native.simplify_mesh_indices(p, n, uv, bad, 0.5, 0.01)
    with pytest.raises(ValueError, match="dimensions"):
        native.simplify_mesh_indices(p, n[:-1], uv, indices, 0.5, 0.01)
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        native.simplify_mesh_indices(p, n, uv, indices, 0.5, 0.01)
