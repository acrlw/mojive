"""Plane mapping and retained-layer contracts for the 2D debug canvas."""

import numpy as np
import pytest

from mojive import Canvas2D
from mojive.render.debugdraw import DebugDraw, DrawPath, PrimitiveType


def test_canvas_maps_xy_and_depth_onto_an_arbitrary_plane() -> None:
    canvas = Canvas2D(
        DebugDraw(),
        origin=(1.0, 2.0, 3.0),
        x_axis=(0.0, 1.0, 0.0),
        y_axis=(0.0, 0.0, 1.0),
    )
    layer = canvas.layer("geometry", depth=0.25)

    assert layer.world((2.0, 4.0)) == pytest.approx((1.25, 4.0, 7.0))


def test_canvas_primitives_update_retained_debug_storage() -> None:
    draw = DebugDraw()
    canvas = Canvas2D(draw)
    layer = canvas.layer("physics")
    layer.circle("body", (1.0, 2.0), 0.5, (1.0, 0.0, 0.0, 1.0), segments=12)
    layer.points("contacts", ((0.0, 0.0), (1.0, 1.0)), (1.0, 1.0, 0.0, 1.0))

    assert layer._layer.count_of(PrimitiveType.STROKE) == 12
    assert layer._layer.count_of(PrimitiveType.POINT) == 2
    assert draw.build().counts[DrawPath.STROKE] == 12

    layer.visible = False
    assert draw.build().counts[DrawPath.STROKE] == 0
    layer.visible = True
    assert draw.build().counts[DrawPath.STROKE] == 12


def test_canvas_camera_fits_bounds_and_uses_canvas_orientation() -> None:
    canvas = Canvas2D(DebugDraw())
    camera = canvas.camera((-4.0, -1.0, 4.0, 1.0), aspect=2.0, padding=0.0)

    assert camera.orthographic
    assert camera.ortho_height == pytest.approx(4.0)
    assert camera.target == pytest.approx(np.zeros(3))
    assert camera.forward() == pytest.approx((0.0, 0.0, -1.0))
    viewport = (20.0, 30.0, 800.0, 400.0)
    screen = canvas.canvas_to_screen((1.25, -0.5), camera, viewport)
    assert screen is not None
    assert canvas.screen_to_canvas(screen, camera, viewport) == pytest.approx((1.25, -0.5))


def test_canvas_rejects_degenerate_basis_and_bounds() -> None:
    with pytest.raises(ValueError, match="axes"):
        Canvas2D(DebugDraw(), x_axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="axes"):
        Canvas2D(DebugDraw(), x_axis=(1.0, 0.0, 0.0), y_axis=(2.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="bounds"):
        Canvas2D(DebugDraw()).camera((0.0, 0.0, 0.0, 1.0))
