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


def _hole_path():
    from mojive.canvas2d import PathBuilder2D

    builder = PathBuilder2D(fill_rule="evenodd")
    for lo, hi in ((0, 4), (1, 3)):
        builder.move_to(lo, lo).line_to(hi, lo).line_to(hi, hi).line_to(lo, hi).close()
    return builder.finish()


def test_filled_hole_has_correct_area_and_reuses_compilation(monkeypatch):
    import mojive.render.canvas_geometry as geometry
    from mojive.canvas2d import Affine2D

    draw = DebugDraw()
    canvas = Canvas2D(draw)
    layer = canvas.layer("fill")
    path = _hole_path()
    layer.fill_path("hole", path, (1, 0, 0, 0.5))
    vertices = layer._layer.positions_of(PrimitiveType.TRIANGLE).copy()
    a, b, c = vertices[:, 0, :2], vertices[:, 1, :2], vertices[:, 2, :2]
    assert (
        np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) / 2
    ).sum() == pytest.approx(12)
    initial_count = draw.primitives

    def unexpected_compile(*args, **kwargs):
        pytest.fail("A translated or recolored path was compiled again")

    monkeypatch.setattr(geometry, "compile_fill", unexpected_compile)
    with layer.transformed(Affine2D.translation(5, 2)):
        layer.fill_path("hole", path, (0, 1, 0, 0.5))
    assert draw.primitives == initial_count
    assert layer._layer.positions_of(PrimitiveType.TRIANGLE) == pytest.approx(
        vertices + np.array((5, 2, 0))
    )
    layer.visible = False
    assert draw.build().counts[DrawPath.TRIANGLE] == 0
    canvas.clear()
    assert draw.primitives == canvas._paths.bytes == 0


def test_transform_context_restores_after_exception_and_invalid_mesh_is_atomic():
    from mojive.canvas2d import Affine2D

    layer = Canvas2D(DebugDraw()).layer("batch")
    triangle = np.array([[[0, 0], [1, 0], [0, 1]]])
    layer.triangles("mesh", triangle, (1, 0, 0, 1))
    original = layer._layer.positions_of(PrimitiveType.TRIANGLE).copy()
    with (
        pytest.raises(ValueError),
        layer.transformed(Affine2D.translation(2, 3)),
        layer.transformed(Affine2D.scale(2)),
    ):
        assert layer.world((1, 1)) == pytest.approx((4, 5, 0))
        layer.triangles("mesh", triangle, (1, float("nan"), 0, 1))
    assert layer.world((1, 1)) == pytest.approx((1, 1, 0))
    assert layer._layer.positions_of(PrimitiveType.TRIANGLE) == pytest.approx(original)
    layer.triangles("mesh", np.empty((0, 3, 2)), (1, 1, 1, 1))
    assert layer._layer.count_of(PrimitiveType.TRIANGLE) == 0


def test_curves_and_strokes_preserve_endpoints_and_replace_id():
    from mojive.canvas2d import PathBuilder2D

    layer = Canvas2D(DebugDraw()).layer("curves")
    layer.bezier("curve", ((0, 0), (1, 2), (3, 0)), (1, 1, 1, 1))
    vertices = layer._layer.positions_of(PrimitiveType.STROKE)
    assert vertices[0, 1] == pytest.approx((0, 0, 0))
    assert vertices[-1, 2] == pytest.approx((3, 0, 0))
    path = PathBuilder2D().move_to(0, 0).line_to(3, 0).finish()
    layer.stroke_path("curve", path, (1, 1, 1, 0.5), width=1, cap="square")
    assert layer._layer.count_of(PrimitiveType.STROKE) == 0
    vertices = layer._layer.positions_of(PrimitiveType.TRIANGLE)
    assert vertices[..., 0].min() == pytest.approx(-0.5)
    assert vertices[..., 0].max() == pytest.approx(3.5)
    assert vertices[..., 1].max() == pytest.approx(0.5)


def test_path_cache_evicts_and_does_not_retain_oversized_entries(monkeypatch):
    import mojive.render.canvas_geometry as geometry
    from mojive.canvas2d import PathBuilder2D

    monkeypatch.setattr(geometry, "_CACHE_ENTRIES", 2)
    cache = geometry.PathCache()
    for x in range(5):
        path = PathBuilder2D().move_to(x, 0).line_to(x + 1, 0).line_to(x, 1).close().finish()
        cache.prepare(path, 0.01)
    assert len(cache._entries) == 2
    assert cache.bytes <= geometry._CACHE_BYTES
    monkeypatch.setattr(geometry, "_CACHE_BYTES", 1)
    cache.clear()
    cache.prepare(path, 0.01)
    assert cache.bytes == 0
