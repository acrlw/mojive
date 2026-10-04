"""Persistent focus preferences and bounded zoom through native UI input."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import CameraNavigationConfig, CameraView, Scene, ViewerConfig, build, build_scene
from mojive.tools.keyframe_timeline import show_settings
from mojive.tools.ui_runtime import (
    _click,
    _item_center,
    _item_rect,
    _park_cursor,
    _save_window_crop,
)

pytestmark = pytest.mark.gpu


def test_explicit_camera_before_startup_survives_first_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    scene.box(position=(80, 0, 0))
    view = CameraView(eye=np.array([2, -3, 1.5]), target=np.array([0, 0, 0.2]), far=120)
    with build_scene(scene, width=640, height=480, vsync=False, show_window=False) as viewer:
        assert not viewer.app._started
        viewer.set_camera(view)
        for _ in range(3):
            viewer.sync()
            np.testing.assert_allclose(viewer.session.camera.eye, view.eye, atol=1e-5)
            np.testing.assert_allclose(viewer.session.camera.target, view.target, atol=1e-5)
        assert viewer.session.camera.far == view.far


def _enter_number(viewer, name, value):
    point = _item_center(viewer, "drag_float", f"##camera_{name}")
    io = imgui.get_io()
    # A new double-click must not become the third/fourth click of the previous edit.
    time.sleep(io.mouse_double_click_time + 0.01)
    _click(viewer, point)
    _click(viewer, point)
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    io.add_key_event(imgui.Key.a, True)
    viewer.sync()
    io.add_key_event(imgui.Key.a, False)
    io.add_key_event(modifier, False)
    io.add_input_characters_utf8(str(value))
    viewer.sync()
    io.add_key_event(imgui.Key.enter, True)
    viewer.sync()
    io.add_key_event(imgui.Key.enter, False)
    viewer.sync()


@pytest.mark.parametrize("scale,language", [(1, "en"), (1.5, "zh_CN")])
def test_navigation_settings_edit_save_restore_and_override(
    tmp_path, monkeypatch, backend_name, scale, language
):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("MOJIVE_SETTINGS", str(path))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    scene = Scene()
    box = scene.box(name="target", position=(1, 0, 1))
    with build_scene(
        scene, width=round(1440 * scale), height=round(1000 * scale), vsync=False, show_window=False
    ) as viewer:
        viewer.app.set_language(language)
        show_settings(viewer, "Camera")
        for name, value in (
            ("focus_margin", 1.8),
            ("focus_duration", 0.8),
            ("zoom_speed", 2),
            ("min_distance", 0.05),
            ("max_distance", 500),
        ):
            _enter_number(viewer, name, value)
            assert getattr(viewer.app.camera.navigation, name) == pytest.approx(value)
        for name, label in (("focus_easing", "Smootherstep"), ("zoom_mode", "Linear")):
            _click(viewer, _item_center(viewer, "begin_combo", f"##camera_{name}"))
            _click(viewer, _item_center(viewer, "selectable", label))
        expected = viewer.app.camera.navigation
        assert expected.focus_easing == "smootherstep" and expected.zoom_mode == "linear"
        assert (
            CameraNavigationConfig.from_mapping(json.loads(path.read_text())["camera_navigation"])
            == expected
        )
        output = Path("output/camera-navigation") / f"{backend_name}-{language}-{scale:g}"
        output.mkdir(parents=True, exist_ok=True)
        _park_cursor(viewer)
        for _ in range(3):
            viewer.sync()
        _save_window_crop(viewer, "Settings", output / "settings.png")
        # Camera-panel distance uses the same bounds and a usable logarithmic rail.
        viewer.panels.get("Settings").open = False
        viewer.panels.open_panel("Camera")
        for _ in range(3):
            viewer.sync()
        imgui.internal.focus_window(imgui.internal.find_window_by_name("Camera"))
        viewer.sync()
        low, high = _item_rect(viewer, "slider_float", "##camera-distance")
        # Clicking the existing grab preserves its offset; acquire away from it first.
        _click(viewer, (low[0] + (high[0] - low[0]) * 0.1, (low[1] + high[1]) * 0.5))
        _click(viewer, ((low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5))
        assert viewer.app.camera.distance == pytest.approx(
            np.sqrt(expected.min_distance * expected.max_distance), rel=0.08
        )
        _park_cursor(viewer)
        for _ in range(3):
            viewer.sync()
        _save_window_crop(viewer, "Camera", output / "camera-distance.png")
        show_settings(viewer, "Camera")
        _enter_number(viewer, "focus_duration", 0)
        node = viewer.session.node_by_object_id(box.object_id)
        assert viewer.app.request_node_focus(node.node_id)
        viewer.sync()
        assert not viewer.app.camera.animating
        assert viewer.app.camera.pivot == pytest.approx((1, 0, 1))
        _click(
            viewer,
            _item_center(viewer, "button", viewer.app.localizer.text("Reset camera navigation")),
        )
        assert viewer.app.camera.navigation == CameraNavigationConfig()
        viewer.configure_navigation(expected, persist=True)
    with build_scene(scene, vsync=False, show_window=False) as restored:
        assert restored.app.camera.navigation == expected
    explicit = CameraNavigationConfig(focus_duration=1.2)
    with build_scene(
        scene, config=ViewerConfig(navigation=explicit), vsync=False, show_window=False
    ) as viewer:
        assert viewer.app.camera.navigation == explicit
    assert (
        CameraNavigationConfig.from_mapping(json.loads(path.read_text())["camera_navigation"])
        == expected
    )


@pytest.mark.physics
@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("mode", ["proportional", "linear"])
def test_extreme_wheel_zoom_keeps_native_ground_finite_and_recovers(
    tmp_path, monkeypatch, backend_name, orthographic, mode
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    model = tmp_path / "textured-ground.xml"
    model.write_text("""<mujoco><asset>
      <texture name="checker" type="2d" builtin="checker" width="32" height="32" rgb1=".3 .4 .5" rgb2=".6 .7 .8"/>
      <material name="floor" texture="checker" texrepeat="4 4" texuniform="true"/>
      </asset><worldbody><geom type="plane" size="0 0 .1" material="floor"/>
      <body pos="0 0 1"><freejoint/><geom type="box" size=".2 .2 .2"/></body>
      </worldbody></mujoco>""")
    model = Path(os.environ.get("MOJIVE_CAMERA_NAVIGATION_MODEL", model))
    with build(
        model, paused=True, vsync=False, width=1280, height=900, show_window=False
    ) as viewer:
        for _ in range(6):
            viewer.sync()
        camera = viewer.app.camera
        viewer.configure_navigation(
            CameraNavigationConfig(zoom_mode=mode, max_distance=1000 if mode == "linear" else 1e6)
        )
        camera.set_orthographic(orthographic)
        io = imgui.get_io()
        x, y, width, height = viewer.app._viewport_rect
        io.add_mouse_pos_event(x + width * 0.5, y + height * 0.6)
        viewer.sync()
        for _ in range(100):
            io.add_mouse_wheel_event(0, -10000 if mode == "linear" else -10)
            viewer.sync()
            assert np.isfinite(viewer.session.camera.eye).all()
            assert np.isfinite(viewer.session.camera.view_matrix()).all()
        assert camera.distance == pytest.approx(camera.navigation.max_distance)
        io.add_mouse_wheel_event(0, 1)
        viewer.sync()
        assert camera.distance < camera.navigation.max_distance
        for _ in range(30):
            io.add_mouse_wheel_event(0, 10000 if mode == "linear" else 10)
            viewer.sync()
        assert camera.distance == pytest.approx(camera.navigation.min_distance)
        io.add_mouse_wheel_event(0, -1)
        viewer.sync()
        assert camera.distance > camera.navigation.min_distance
        viewer.app._frame_scene(animate=False)
        for _ in range(3):
            viewer.sync()
        output = Path("output/camera-navigation")
        output.mkdir(parents=True, exist_ok=True)
        viewer.capture(
            output / f"{backend_name}-zoom-recovered-{mode}-{orthographic}.png", surface="viewport"
        )


def test_return_from_model_camera_preserves_viewport_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    scene.box(position=(0, 0, 0.5))
    view = CameraView(eye=np.array([3, -5, 2.5]), target=np.array([0, 0, 0.5]))
    camera_id = scene.add_camera("shot", view)
    with build_scene(scene, width=1000, height=600, vsync=False, show_window=False) as viewer:
        for _ in range(5):
            viewer.sync()
        # Public cameras default to a square aspect; the viewport owns presentation.
        viewer.set_camera(view)
        _, _, width, height = viewer.app._viewport_rect
        assert width / height != pytest.approx(view.aspect)
        expected = view.with_aspect(width / height).proj_matrix()
        assert viewer.app.camera.view().proj_matrix() == pytest.approx(expected)
        viewer.app.select_model_camera(camera_id)
        viewer.sync()
        viewer.app.select_model_camera(-1, animate=True)
        for _ in range(20):
            viewer.app._advance_camera(1 / 30)
            assert viewer.session.camera.proj_matrix() == pytest.approx(expected, abs=1e-5)
        assert viewer.app._camera_transition is None


def test_settings_save_failure_keeps_window_usable_and_reports_unsaved_value(
    tmp_path, monkeypatch, backend_name
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    scene.box()
    with build_scene(scene, width=1440, height=1000, vsync=False, show_window=False) as viewer:
        show_settings(viewer, "Camera")
        viewer.app.set_camera_navigation(viewer.app.camera.navigation)
        before = viewer.app.preferences.path.read_bytes()

        def reject_save(values):
            raise OSError("settings destination unavailable")

        with monkeypatch.context() as failing:
            failing.setattr(viewer.app.preferences, "_save", reject_save)
            _enter_number(viewer, "zoom_speed", 2.5)
            for _ in range(3):
                viewer.sync()
            assert viewer.app.camera.navigation.zoom_speed == 2.5
            assert viewer.app.preferences.path.read_bytes() == before
            assert "could not be saved" in viewer.session.last_message
            assert "settings destination unavailable" in viewer.session.last_message
            output = Path("output/camera-navigation") / backend_name
            output.mkdir(parents=True, exist_ok=True)
            _save_window_crop(viewer, "Settings", output / "unsaved-preview.png")
        _enter_number(viewer, "zoom_speed", 3.0)
        assert viewer.app.camera.navigation.zoom_speed == 3.0
        assert viewer.app.preferences.get("camera_navigation")["zoom_speed"] == 3.0
        assert (
            json.loads(viewer.app.preferences.path.read_text())["camera_navigation"]["zoom_speed"]
            == 3.0
        )
