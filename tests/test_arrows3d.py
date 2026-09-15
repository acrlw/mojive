"""Shared 3D arrow geometry, retained updates, and public bridge contracts."""

import numpy as np
import pytest

from mojive.remote.bridge import DebugBridge
from mojive.render.debugdraw import DebugDraw, DrawPath, Occlusion, PrimitiveType
from mojive.render.mesh import arrow_mesh


def test_arrow_mesh_has_smooth_unit_normals_and_closed_surface():
    for bend in (0.0, 0.5):
        mesh = arrow_mesh(1.0, 0.02, 0.06, 0.12, bend)
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1, atol=1e-6)
        tri = mesh.positions[mesh.indices.reshape(-1, 3)]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = mesh.normals[mesh.indices.reshape(-1, 3)].mean(axis=1)
        nonzero = np.linalg.norm(cross, axis=1) > 1e-9
        assert (np.sum(cross[nonzero] * normals[nonzero], axis=1) > 0).all()
        # Divergence theorem: a watertight oriented surface has zero net area vector.
        np.testing.assert_allclose(cross.sum(axis=0), 0, atol=1e-6)


def test_retained_arrows_replace_clear_and_reject_invalid_updates_atomically():
    draw = DebugDraw()
    layer = draw.layer("velocity", Occlusion.ALWAYS)
    layer.arrow_3d("v", (0, 0, 0), (1, 0, 0), (0, 1, 0, 1))
    first = draw.build().stream(DrawPath.LIT_TRIANGLE).copy()
    for _ in range(3):
        layer.arrow_3d("v", (0, 0, 0), (1, 0, 0), (0, 1, 0, 1))
        assert draw.primitives == len(first)
    with pytest.raises(ValueError, match="opaque"):
        layer.arrow_3d("v", (0, 0, 0), (1, 0, 0), (1, 0, 0, 0.5))
    np.testing.assert_array_equal(first, draw.build().stream(DrawPath.LIT_TRIANGLE))
    layer.arc_arrow_3d("v", (0, 0, 0), (0, 0, 1), (1, 0, 0), -1, (1, 0, 0, 1))
    points = layer.positions_of(PrimitiveType.LIT_TRIANGLE)
    assert points[..., 1].min() < -0.3
    layer.arc_arrow_3d("v", (0, 0, 0), (0, 0, 1), (1, 0, 0), 0, (1, 0, 0, 1))
    assert draw.primitives == 0


def test_bridge_accepts_same_3d_arrow_parameters():
    from types import SimpleNamespace

    draw = DebugDraw()
    bridge = DebugBridge(SimpleNamespace(debug=draw, caps=SimpleNamespace(debug_draw=True)))
    records = [
        {"op": "arrow_3d", "id": "linear", "a": [0, 0, 0], "b": [1, 0, 0]},
        {
            "op": "arc_arrow_3d",
            "id": "yaw",
            "center": [0, 0, 0],
            "start_direction": [1, 0, 0],
            "sweep": 1.5,
        },
    ]
    assert bridge.apply_batch(records) == 2
    assert draw.build().counts[DrawPath.LIT_TRIANGLE] > 0
    assert bridge.stats.invalid == 0
