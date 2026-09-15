"""View-cube navigation and raster output use the production input and drawing paths."""

from dataclasses import replace

import numpy as np
import pytest

from mojive.app.ui.window import create_window
from mojive.tools.viewcube_transitions import capture_orientation, circle_limits, depth_swaps
from mojive.ui.viewcube import (
    ORIGIN_BORDER_PT,
    ORIGIN_GAP_PT,
    ORIGIN_RADIUS_PT,
    ViewCube,
    widget_center,
)
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("scale", (0.65, 1.0, 1.5, 2.5))
def test_transitions_and_transparent_origin_shell(backend_name, scale):
    window = create_window(
        WindowConfig(
            width=300,
            height=300,
            vsync=False,
            docking=False,
            ini_path="",
            show_on_start=False,
            ui_scale=scale,
        ),
        backend_name,
    )
    cube = ViewCube()
    try:
        for _ in range(3):
            capture_orientation(window, cube, (1, 1, 0.6), scale)
        for name, eyes in (*depth_swaps(), *circle_limits()):
            before, after = [capture_orientation(window, cube, eye, scale)[0] for eye in eyes]
            difference = np.abs(before.astype(int) - after.astype(int))
            assert difference.max() <= 4, (name, scale, difference.max())
        for background in (
            (0.2, 0.4, 0.7, 1.0),
            (0.7, 0.3, 0.15, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        ):
            pixels = capture_orientation(window, cube, (1, 1, 0.6), scale, background=background)[0]
            height, width = pixels.shape[:2]
            center = pixels[height // 2 - 1 : height // 2 + 1, width // 2 - 1 : width // 2 + 1, :3]
            assert center.mean() > 230
            if scale >= 1.0:
                y, x = np.indices((height, width), dtype=float)
                radius = (
                    np.hypot(x + 0.5 - width * 0.5, y + 0.5 - height * 0.5) / window.pixel_scale
                )
                # Exclude both AA fringes; the remaining annulus must expose the
                # actual background, not a flat-color disk painted over the axes.
                origin_fringe = min(1.0, ORIGIN_BORDER_PT * scale * 0.5)
                border_radius = (ORIGIN_RADIUS_PT + ORIGIN_BORDER_PT) * scale
                clear = (radius > border_radius + origin_fringe + 0.1) & (
                    radius < (ORIGIN_RADIUS_PT + ORIGIN_GAP_PT) * scale - 1.05
                )
                assert clear.any()
                assert np.max(np.abs(pixels[clear].astype(int) - pixels[0, 0].astype(int))) <= 1
            if background[:3] == (1.0, 1.0, 1.0):
                # The old white-only disk disappeared here. Inspect only the
                # center neighborhood, away from the colored spoke ends.
                radius_px = (ORIGIN_RADIUS_PT + ORIGIN_BORDER_PT) * scale * window.pixel_scale
                extent = max(2, int(np.ceil(radius_px)))
                origin = pixels[
                    height // 2 - extent : height // 2 + extent,
                    width // 2 - extent : width // 2 + extent,
                    :3,
                ]
                gray = origin.max(axis=-1) - origin.min(axis=-1) <= 2
                assert np.any(gray & (origin.mean(axis=-1) < 225))
        plain = capture_orientation(window, cube, (1, 1, 0.6), scale)[0]
        hovered = capture_orientation(window, cube, (1, 1, 0.6), scale, hover_origin=True)[0]
        changed_y, changed_x = np.where(np.any(plain != hovered, axis=-1))
        assert len(changed_x) > 0
        distance = (
            np.hypot(
                changed_x + 0.5 - plain.shape[1] * 0.5,
                changed_y + 0.5 - plain.shape[0] * 0.5,
            )
            / window.pixel_scale
        )
        # Hover affects the white disk only, never the axes or the whole backdrop.
        assert distance.max() < (ORIGIN_RADIUS_PT + ORIGIN_GAP_PT) * scale
    finally:
        window.close()


@pytest.fixture
def origin_viewer(backend_name, tmp_path, monkeypatch, request):
    from imgui_bundle import imgui

    from mojive import build_scene
    from mojive.scene import Scene
    from mojive.types import CameraView

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(getattr(request, "param", 1.0)))
    scene = Scene()
    scene.box(name="box")
    scene.add_camera("inspection", CameraView(eye=np.array((3.0, -6.0, 2.0)), target=np.zeros(3)))
    with build_scene(scene, renderer=backend_name, vsync=False, width=1280, height=800) as viewer:
        for _ in range(12):
            viewer.sync()
        viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
        viewer.sync()
        center = widget_center(viewer.app._viewport_rect, viewer.window.style_scale)
        yield viewer, imgui.get_io(), center


def _press_origin(viewer, io, point):
    from mojive.ui.gestures import Claim

    io.add_mouse_pos_event(*point)
    viewer.sync()
    assert viewer.app.view_cube.origin_hovered
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.router.claim is Claim.VIEW_CUBE


@pytest.mark.parametrize("origin_viewer", (0.65, 1.0, 1.5), indirect=True)
def test_origin_toggles_projection_once_on_release_without_reframing(origin_viewer):
    viewer, io, center = origin_viewer
    camera = viewer.app.camera
    original = camera.view()
    selection = viewer.session.selected
    # The hit target includes transparent padding outside the visible disk.
    point = (center[0] + 5.0 * viewer.window.style_scale, center[1])
    for target in (True, False):
        before = camera.orthographic
        _press_origin(viewer, io, point)
        for _ in range(3):
            viewer.sync()
            assert camera.orthographic is before
        io.add_mouse_button_event(0, False)
        viewer.sync()
        assert camera.orthographic is target
        assert camera.animating
        camera.advance(1.0, viewer.app.camera_out)
        for _ in range(3):
            viewer.sync()
        assert camera.orthographic is target
        assert camera.pivot == pytest.approx(original.target)
        assert camera.view().eye == pytest.approx(original.eye, abs=1e-5)
        assert camera.matched_ortho_height() == pytest.approx(original.matched_ortho_height())
        assert viewer.session.selected == selection


@pytest.mark.parametrize("release_motion", (False, True))
def test_origin_drag_never_toggles_projection_on_release(origin_viewer, release_motion):
    viewer, io, center = origin_viewer
    camera = viewer.app.camera
    yaw = camera.yaw
    _press_origin(viewer, io, center)
    if not release_motion:
        io.add_mouse_pos_event(center[0] + 15.0, center[1] + 8.0)
        viewer.sync()
        assert camera.yaw != yaw
        # Returning to the origin is still a drag, not a projection click.
        io.add_mouse_pos_event(*center)
        viewer.sync()
    else:
        # Motion and release may arrive together inside the padded hit target.
        io.add_mouse_pos_event(center[0] + 5.0, center[1])
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not camera.orthographic
    if release_motion:
        assert camera.yaw != yaw


def test_origin_release_outside_cancels_the_click(origin_viewer):
    viewer, io, center = origin_viewer
    _press_origin(viewer, io, (center[0] + 5.5, center[1]))
    io.add_mouse_pos_event(center[0] + 7.0, center[1])
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not viewer.app.view_cube.origin_hovered
    assert not viewer.app.camera.orthographic


def test_axis_covering_origin_keeps_its_click_action(origin_viewer):
    viewer, io, center = origin_viewer
    camera = viewer.app.camera
    camera.look_from(0.0, 0.0, viewer.app.camera_out, animate=False)
    viewer.sync()
    io.add_mouse_pos_event(*center)
    viewer.sync()
    assert not viewer.app.view_cube.origin_hovered
    assert viewer.app.view_cube.hovered.axis == 0
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not camera.orthographic
    assert camera.animating


@pytest.mark.parametrize("start_on_origin", (False, True))
def test_view_change_during_press_does_not_activate_a_different_part(
    origin_viewer, start_on_origin
):
    from mojive.ui.gestures import Claim

    viewer, io, center = origin_viewer
    camera = viewer.app.camera
    if not start_on_origin:
        camera.look_from(0.0, 0.0, viewer.app.camera_out, animate=False)
    io.add_mouse_pos_event(*center)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.router.claim is Claim.VIEW_CUBE
    yaw, pitch = (0.0, 0.0) if start_on_origin else (-135.0, 25.0)
    camera.look_from(yaw, pitch, viewer.app.camera_out, animate=False)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not camera.orthographic
    assert not camera.animating


@pytest.mark.parametrize("disabled", ("view_cube", "viewport_ui", "projection", "modal"))
def test_origin_obeys_input_visibility_and_backend_capability(origin_viewer, monkeypatch, disabled):
    viewer, io, center = origin_viewer
    app = viewer.app
    if disabled == "view_cube":
        app.set_interactions(
            replace(app.interactions, camera=replace(app.interactions.camera, view_cube=False)),
            persist=False,
        )
    elif disabled == "viewport_ui":
        app.set_viewport_layers(replace(app.viewport_layers, viewport_ui=False), persist=False)
    elif disabled == "projection":
        monkeypatch.setattr(
            viewer.backend, "caps", replace(viewer.backend.caps, orthographic=False)
        )
    else:
        monkeypatch.setattr(app, "_scene_input_blocked", lambda: True)
    io.add_mouse_pos_event(*center)
    viewer.sync()
    if disabled != "projection":
        assert not app.view_cube.origin_hovered
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not app.camera.orthographic


def test_origin_projection_from_scene_camera_keeps_the_displayed_view(origin_viewer):
    viewer, io, center = origin_viewer
    app = viewer.app
    camera_id = viewer.session.cameras[0].camera_id
    app.select_model_camera(camera_id)
    viewer.sync()
    original = viewer.session.camera
    _press_origin(viewer, io, center)
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert app._model_camera_id == -1
    assert app.camera.orthographic
    assert app.camera.view().eye == pytest.approx(original.eye, abs=1e-5)
    assert app.camera.pivot == pytest.approx(original.target)
    assert not viewer.session.camera_view(camera_id).orthographic
