"""The presented viewport follows every resize frame without a settling delay."""

import pytest

from mojive.app.composition import build_scene
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene import Scene
from mojive.ui import window as window_module

pytestmark = pytest.mark.gpu


def test_viewport_image_matches_each_current_layout_after_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    scene = Scene()
    scene.box(name="resize-box")
    viewer = build_scene(
        scene,
        width=900,
        height=650,
        vsync=False,
        show_window=False,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    )
    try:
        # The former policy only began delaying size changes after 30 frames.
        for _ in range(35):
            viewer.sync()
        observed = set()
        for width, height in ((1000, 700), (1100, 760), (820, 720), (960, 640), (900, 650)):
            window_module.glfw.set_window_size(viewer.window._window, width, height)
            viewer.sync()
            app = viewer.app
            expected = tuple(
                max(1, int(value))
                for value in viewer.window.points_to_pixels(app.viewport_surface.size)
            )
            image = app._viewport_image
            assert image is not None
            assert (image.width, image.height) == expected
            assert app._current_viewport_render_size() == expected
            assert app.camera.view().aspect == pytest.approx(expected[0] / expected[1])
            panel_x, panel_y = app.viewport_surface.position
            panel_w, panel_h = app.viewport_surface.size
            x, y, w, h = app._viewport_rect
            assert abs(x - panel_x) < 1
            assert abs(y - panel_y) < 1
            assert abs(w - panel_w) < 1
            assert abs(h - panel_h) < 1
            observed.add(expected)
        assert len(observed) == 5
    finally:
        viewer.release()
