"""Automatic reveal waits for a fully drawn native layout without extra app ticks."""

from dataclasses import replace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.app.composition import build
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig

pytestmark = pytest.mark.gpu


@pytest.fixture(params=("native", "classic"))
def startup_viewer(request, backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1.25")
    config = ViewerConfig(
        ui=request.param,
        layout=LayoutConfig(persistence=False),
        debug_server=False,
        threaded_physics=False,
    )
    with build(
        resolve("joint_gizmo"),
        renderer=backend_name,
        config=config,
        width=1280,
        height=800,
        vsync=False,
        show_window=False,
    ) as viewer:
        yield viewer


def test_first_reveal_has_complete_dock_layout_and_one_tick_per_sync(startup_viewer, monkeypatch):
    viewer = startup_viewer
    window, app = viewer.window, viewer.app
    window.config = replace(window.config, show_on_start=True)
    events, images, ticks = [], [], []
    tick, end = viewer.session.tick, window.end_frame

    def tick_once(*args, **kwargs):
        ticks.append(app._frame_index)
        return tick(*args, **kwargs)

    def show():
        viewport = imgui.internal.find_window_by_name("Viewport")
        assert not viewport.hidden and not viewport.skip_items
        assert min(app.viewport_surface.size) > 200
        assert window._ready_to_show()
        assert not window._default_tabs_frame
        assert imgui.get_draw_data().total_vtx_count > 1000
        events.append(app._frame_index)
        # Verify reveal decisions without mapping a test window onto the desktop.
        window._shown, window._show_pending = True, False

    def completed_frame(**kwargs):
        image = end(readback=True)
        if events and not images:
            images.append(image.copy())
        return image

    monkeypatch.setattr(window, "show", show)
    monkeypatch.setattr(window, "end_frame", completed_frame)
    monkeypatch.setattr(viewer.session, "tick", tick_once)
    assert not window.shown
    for index in range(8):
        viewer.sync()
        assert len(ticks) == index + 1
        assert app._frame_index == index + 1
        if index == 0:
            assert not events, "The dock host is still measuring on its first frame"
    assert len(events) == 1 and events[0] < 4
    assert images[0].shape[:2] == window.size_pixels[::-1]
    assert np.std(images[0]) > 10
    # Hiding an already prepared viewer must not arm automatic reveal again.
    window.hide()
    viewer.sync()
    assert len(events) == 1 and not window.shown


def test_hidden_configuration_keeps_rendering_without_automatic_reveal(startup_viewer, monkeypatch):
    viewer = startup_viewer
    monkeypatch.setattr(viewer.window, "show", lambda: pytest.fail("Hidden viewer was revealed"))
    for _ in range(4):
        viewer.sync()
    assert viewer.window._ready_to_show()
    assert not viewer.window.shown and not viewer.window._show_pending
    assert viewer.app._viewport_image is not None


def test_failed_render_does_not_reveal_then_can_retry(startup_viewer, monkeypatch):
    viewer = startup_viewer
    window = viewer.window
    window.config = replace(window.config, show_on_start=True)
    viewer.sync()
    shown = []
    monkeypatch.setattr(window, "show", lambda: shown.append(True))
    # Fail during GPU submission, after ImGui has resolved the next layout.
    renderer = window._imgui_backend if hasattr(window, "_imgui_backend") else window._impl
    method = "prepare" if hasattr(window, "_imgui_backend") else "render"

    def fail(*_args):
        raise RuntimeError("injected UI submission failure")

    with monkeypatch.context() as patch:
        patch.setattr(renderer, method, fail)
        with pytest.raises(RuntimeError, match="injected UI submission failure"):
            viewer.sync()
    assert not shown and not window.shown
    viewer.sync()
    assert shown
