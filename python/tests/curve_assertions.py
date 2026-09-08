"""Geometric comparisons independent of a curve's adaptive vertex count."""

import numpy as np


def distance_to_path(points, path, *, closed=True):
    points, path = np.asarray(points), np.asarray(path)
    starts = path if closed else path[:-1]
    ends = np.roll(path, -1, axis=0) if closed else path[1:]
    edges = ends - starts
    delta = points[:, None, :] - starts
    denominator = np.maximum(np.sum(edges * edges, axis=1), 1e-30)
    u = np.clip(np.sum(delta * edges, axis=2) / denominator, 0, 1)
    return np.linalg.norm(delta - u[..., None] * edges, axis=2).min(axis=1)


def assert_paths_close(first, second, tolerance=0.025, *, closed=True):
    assert distance_to_path(first, second, closed=closed).max() <= tolerance
    assert distance_to_path(second, first, closed=closed).max() <= tolerance
