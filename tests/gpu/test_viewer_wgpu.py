"""Window-stack smoke tests for the wgpu viewer path.

Runs only under ``FORGE_VIEWER_BACKEND=wgpu`` (see Makefile ``gpu-wgpu``);
like the GL window tests it needs a display server for GLFW.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")
pytest.importorskip("wgpu")
pytest.importorskip("rendercanvas")

from forge_viewer.composition import build_scene  # noqa: E402
from forge_viewer.demos import canvas_scene  # noqa: E402
from forge_viewer.render.backend import RenderFlag  # noqa: E402
from forge_viewer.render.webgpu.backend import WgpuBackend  # noqa: E402
from forge_viewer.ui.window_wgpu import WgpuWindow  # noqa: E402


@pytest.fixture(scope="module")
def viewer(backend_name):
    if backend_name != "wgpu":
        pytest.skip("wgpu window-stack test; run with FORGE_VIEWER_BACKEND=wgpu")
    scene = canvas_scene()
    instance = build_scene(scene, vsync=False, width=960, height=640)
    try:
        instance.app.camera.look_from(-135.0, 25.0, instance.app.camera_out, animate=False)
        for _ in range(8):
            instance.sync()
        yield instance, scene
    finally:
        instance.release()


def snap(v) -> np.ndarray:
    px = v.window.read_frame()
    assert px is not None
    return np.asarray(px)[::-1][..., :3].copy()


def viewport_snap(v) -> np.ndarray:
    image = snap(v)
    x, y, w, h = v.window.points_to_pixels(v.app._viewport_rect)
    x0, y0, x1, y1 = round(x), round(y), round(x + w), round(y + h)
    return image[y0:y1, x0:x1]


def test_window_and_backend_are_wgpu(viewer):
    v, _scene = viewer
    assert isinstance(v.backend, WgpuBackend)
    assert isinstance(v.window, WgpuWindow)
    assert not v.window.shown
    assert v.backend.caps.name == "wgpu"
    assert v.backend.device is v.window.device
    fb_w, fb_h = v.window.size_pixels
    pt_w, pt_h = v.window.size_points
    assert fb_w >= pt_w and fb_h >= pt_h


def test_gpu_pass_timing_is_reported_when_supported(viewer):
    v, _scene = viewer
    supported = "timestamp-query" in v.backend.device.features
    assert v.backend.caps.gpu_timing is supported
    if not supported:
        assert not v.backend.stats.gpu_ms
        return
    for _ in range(20):
        v.sync()
        if {"scene", "export"} <= v.backend.stats.gpu_ms.keys():
            break
        time.sleep(0.005)
    assert v.backend.stats.gpu_ms["scene"] > 0.0
    assert v.backend.stats.gpu_ms["export"] > 0.0
    assert all(np.isfinite(value) and value < 1000.0 for value in v.backend.stats.gpu_ms.values())


def test_window_dependencies_use_the_imgui_glfw_library(backend_name):
    if backend_name != "wgpu":
        pytest.skip("wgpu window-stack test; run with FORGE_VIEWER_BACKEND=wgpu")

    env = os.environ.copy()
    env.pop("PYGLFW_LIBRARY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from forge_viewer.ui.window_wgpu import _load_window_deps; _load_window_deps()",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "implemented in both" not in result.stderr


def test_read_frame_contains_scene_pixels(viewer):
    v, _scene = viewer
    px = v.window.read_frame()
    fb_w, fb_h = v.window.size_pixels
    assert px is not None
    assert px.shape == (fb_h, fb_w, 3)
    assert px.dtype == np.uint8
    # The scene is drawn into the Viewport panel, not just UI chrome.
    assert float(viewport_snap(v).std()) > 10.0


def test_viewport_image_carries_the_wgpu_color_view(viewer):
    v, _scene = viewer
    image = v.app._viewport_image
    assert image is not None
    assert image.flip_y is False
    assert (image.width, image.height) == (v.backend.target.width, v.backend.target.height)
    assert image.payload is v.backend.target.color_view
    # Registering through the window is cached while the target is unchanged.
    ref = v.window.viewport_texture_ref(image)
    assert ref is v.window.viewport_texture_ref(image)


def test_msaa_flag_rebuilds_wgpu_targets_and_pipelines(viewer):
    v, _scene = viewer
    original_color = v.backend.target.color
    assert v.backend.target.samples == 4

    assert v.backend.set_flag(RenderFlag.MSAA, False)
    assert v.backend.target.samples == 1
    assert v.backend.caps.msaa_samples == 1
    assert v.backend.target.color is not original_color
    v.sync()
    assert float(viewport_snap(v).std()) > 10.0

    assert v.backend.set_flag(RenderFlag.MSAA, True)
    assert v.backend.target.samples == 4
    assert v.backend.caps.msaa_samples == 4
    v.sync()
    assert float(viewport_snap(v).std()) > 10.0


def test_scene_edit_reaches_the_window(viewer):
    v, scene = viewer
    before = viewport_snap(v)
    scene.object("ball").set_pose((1.4, -0.3, 0.42))
    v.sync()
    v.sync()
    after = viewport_snap(v)
    assert after.shape == before.shape
    diff = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    assert np.count_nonzero(diff > 10) > 500


def test_injected_mouse_drag_rotates_the_camera(viewer):
    from imgui_bundle import imgui

    v, _scene = viewer
    io = imgui.get_io()
    x, y, w, h = v.app._viewport_rect
    cx, cy = x + w * 0.5, y + h * 0.5
    before = viewport_snap(v)
    io.add_mouse_pos_event(cx, cy)
    io.add_mouse_button_event(0, True)
    v.sync()
    for i in range(1, 13):
        io.add_mouse_pos_event(cx + i * 5.0, cy + i * 2.0)
        v.sync()
    io.add_mouse_button_event(0, False)
    v.sync()
    after = viewport_snap(v)
    assert after.shape == before.shape
    diff = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    assert np.count_nonzero(diff > 10) > 500


def test_vendored_imgui_render_path(viewer, monkeypatch):
    from forge_viewer.ui import window_wgpu

    # Force the vendored imgui-1.92 draw-data fix even where the installed
    # wgpu no longer needs it, so the vendored code path stays exercised.
    monkeypatch.setattr(window_wgpu, "_upstream_needs_imgui_192_fix", lambda: True)
    v, _scene = viewer
    v.sync()
    assert float(viewport_snap(v).std()) > 10.0


def test_native_resize_between_draw_and_submit_stays_within_wgpu_target(viewer, monkeypatch):
    from forge_viewer.ui import window_wgpu

    v, _scene = viewer
    window = v.window
    original_size = window.size_points
    target_size = (max(480, original_size[0] - 240), max(360, original_size[1] - 160))
    original_end_frame = window.end_frame
    resized = False

    def resize_before_submit(*args, **kwargs):
        nonlocal resized
        if not resized:
            resized = True
            window_wgpu.glfw.set_window_size(window._window, *target_size)
            window_wgpu.glfw.poll_events()
        return original_end_frame(*args, **kwargs)

    monkeypatch.setattr(window, "end_frame", resize_before_submit)
    try:
        # The old ImGui draw data and the new framebuffer deliberately coexist
        # in this frame, matching a native maximize/restore event.
        v.sync()
        assert resized
        v.sync()
        image = v.app._viewport_image
        assert image is not None
        assert v.app._viewport_rect[2] / v.app._viewport_rect[3] == pytest.approx(image.aspect)
    finally:
        monkeypatch.setattr(window, "end_frame", original_end_frame)
        window_wgpu.glfw.set_window_size(window._window, *original_size)
        window_wgpu.glfw.poll_events()
        for _ in range(3):
            v.sync()


def test_occluded_surface_keeps_the_frame_loop_alive(viewer):
    import wgpu

    class OccludedSurface:
        def __init__(self, inner):
            self._inner = inner

        def get_current_texture(self):
            raise wgpu.DrawCancelled("Occluded")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    v, _scene = viewer
    context = v.window._gpu_context
    v.window._gpu_context = OccludedSurface(context)
    try:
        v.sync()
        assert float(viewport_snap(v).std()) > 10.0
    finally:
        v.window._gpu_context = context


def test_windows_keep_independent_imgui_contexts(backend_name):
    if backend_name != "wgpu":
        pytest.skip("wgpu window-stack test; run with FORGE_VIEWER_BACKEND=wgpu")

    left = build_scene(canvas_scene(), vsync=False, width=480, height=360)
    right = build_scene(canvas_scene(), vsync=False, width=640, height=420)
    try:
        assert left.window._imgui_context != right.window._imgui_context
        for _ in range(3):
            left.sync()
            right.sync()
        assert left.window.size_points == (480, 360)
        assert right.window.size_points == (640, 420)
        assert float(viewport_snap(left).std()) > 10.0
        assert float(viewport_snap(right).std()) > 10.0
    finally:
        left.release()
        right.release()
