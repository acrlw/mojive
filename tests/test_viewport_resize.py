"""Keep the viewport target and projection current through continuous layout changes."""

from types import SimpleNamespace

import pytest

from mojive.ui.app import ViewerApp
from mojive.ui.camera import OrbitCamera
from mojive.ui.viewport_surface import ViewportSurface
from mojive.ui.window import Window


@pytest.fixture
def app():
    viewer = ViewerApp.__new__(ViewerApp)
    viewer.window = Window.__new__(Window)
    viewer.window._pixel_scale = 1.0
    viewer.viewport_surface = ViewportSurface((10.0, 20.0), (800.0, 600.0))
    viewer.camera = OrbitCamera()
    viewer._fixed_render_size = None
    viewer._viewport_image = None
    viewer._started = True
    target = SimpleNamespace(width=800, height=600)
    calls = []

    def resize(width, height):
        calls.append((width, height))
        target.width, target.height = width, height

    viewer.backend = SimpleNamespace(target=target, resize=resize, resize_calls=calls)
    return viewer


def test_viewport_target_and_projection_follow_every_drag_frame(app):
    for index in range(120):
        width, height = 810 + index * 3, 610 + index * 2
        app.viewport_surface.size = (float(width), float(height))

        app._sync_viewport_size()

        assert app._current_viewport_render_size() == (width, height)
        assert app.camera._aspect == pytest.approx(width / height)
        assert app._viewport_rect == pytest.approx((10.0, 20.0, width, height))
    assert len(app.backend.resize_calls) == 120


def test_unchanged_viewport_does_not_reallocate_and_still_initializes_projection(app):
    for _ in range(120):
        app._sync_viewport_size()

    assert app.backend.resize_calls == []
    assert app.camera._aspect == pytest.approx(800 / 600)


def test_viewport_resize_uses_current_framebuffer_density_and_pixel_extent(app):
    app.window._pixel_scale = 2.0
    app.viewport_surface.size = (407.25, 305.75)

    app._sync_viewport_size()

    assert app.backend.resize_calls == [(814, 611)]
    assert app.camera._aspect == pytest.approx(814 / 611)
    x, y, width, height = app._viewport_rect
    assert 0 <= x - 10.0 < 0.5
    assert 0 <= y - 20.0 < 0.5
    assert abs(width - 407.25) < 0.5
    assert abs(height - 305.75) < 0.5


def test_fixed_output_preserves_aspect_fit_and_clears_on_the_same_frame(app):
    app.set_fixed_render_size(1600, 900)
    app.viewport_surface.size = (900.0, 900.0)
    app._sync_viewport_size()

    assert app.backend.resize_calls == [(1600, 900)]
    assert app._viewport_rect == pytest.approx((10.0, 216.875, 900.0, 506.25))
    assert app.camera._aspect == pytest.approx(1600 / 900)

    app.clear_fixed_render_size()

    assert app.backend.resize_calls == [(1600, 900), (900, 900)]
    assert app._viewport_rect == pytest.approx((10.0, 20.0, 900.0, 900.0))
    assert app.camera._aspect == pytest.approx(1.0)


def test_failed_resize_retries_the_actual_target_instead_of_committing_a_pending_size(
    app, monkeypatch
):
    app.viewport_surface.size = (1000.0, 700.0)
    resize = app.backend.resize

    def fail(width, height):
        raise RuntimeError("Target allocation failed")

    monkeypatch.setattr(app.backend, "resize", fail)
    with pytest.raises(RuntimeError, match="Target allocation failed"):
        app._sync_viewport_size()
    assert app._current_viewport_render_size() == (800, 600)

    monkeypatch.setattr(app.backend, "resize", resize)
    app._sync_viewport_size()
    assert app.backend.resize_calls == [(1000, 700)]
    assert app.camera._aspect == pytest.approx(1000 / 700)


def test_collapsed_viewport_keeps_a_positive_render_extent(app):
    app.viewport_surface.size = (0.0, 0.0)
    app._sync_viewport_size()
    assert app._current_viewport_render_size() == (1, 1)
    assert app.camera._aspect == pytest.approx(1.0)
