"""Camera following keeps navigation stable across target motion and display cadences."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mojive import CameraTrackingConfig
from mojive.adapters.base import NodeType, SceneFrame, SceneNode
from mojive.composition import Viewer
from mojive.types import CameraView
from mojive.ui.app import ViewerApp
from mojive.ui.camera import OrbitCamera
from mojive.ui.camera_tracking import CameraTracker, tracking_position


@pytest.mark.parametrize("fps", [30, 60, 144, 240])
def test_tracking_half_life_is_independent_of_display_rate(fps):
    camera = OrbitCamera(pivot=(0, 0, 2), distance=5, yaw=25, pitch=15)
    tracker = CameraTracker(CameraTrackingConfig(axes="xyz", smoothing=0.5))
    tracker.start(1, camera)
    before = camera.view()
    for _ in range(fps):
        tracker.advance(camera, np.array((8, -4, 6)), 1 / fps)
    assert camera.pivot == pytest.approx((6, -3, 5))
    assert camera.view().eye - camera.view().target == pytest.approx(before.eye - before.target)
    assert camera.distance == 5
    assert camera.yaw == 25
    assert camera.pitch == 15


def test_xy_holds_height_through_large_vertical_motion_and_xyz_smooths_the_switch():
    camera = OrbitCamera(pivot=(0, 0, 2))
    tracker = CameraTracker(CameraTrackingConfig(smoothing=0.25))
    tracker.start(1, camera)
    for z in (6, -2, 9, -1, 10):
        tracker.advance(camera, np.array((0, 0, z)), 0.25)
        assert camera.pivot[2] == 2
    tracker.config = replace(tracker.config, axes="xyz")
    tracker.advance(camera, np.array((0, 0, 10)), 0.25)
    assert camera.pivot[2] == pytest.approx(6)
    tracker.config = replace(tracker.config, axes="xy")
    tracker.advance(camera, np.array((2, 2, -20)), 0.25)
    assert camera.pivot == pytest.approx((1, 1, 6))


def test_tracking_start_switch_stop_and_zero_smoothing_do_not_leave_motion_behind():
    camera = OrbitCamera(pivot=(2, 3, 4))
    tracker = CameraTracker(CameraTrackingConfig(axes="xyz", smoothing=0.5))
    tracker.start(1, camera)
    assert camera.pivot == pytest.approx((2, 3, 4))
    tracker.advance(camera, np.array((6, 7, 8)), 0)
    assert camera.pivot == pytest.approx((2, 3, 4))
    tracker.advance(camera, np.array((6, 7, 8)), 0.5)
    tracker.start(2, camera)
    assert camera.pivot == pytest.approx((4, 5, 6))
    tracker.advance(camera, np.array((0, 1, 2)), 0.5)
    assert camera.pivot == pytest.approx((2, 3, 4))
    tracker.config = replace(tracker.config, smoothing=0)
    tracker.advance(camera, np.array((9, 8, 7)), 1 / 60)
    assert camera.pivot == pytest.approx((9, 8, 7))
    tracker.stop()
    assert not tracker.advance(camera, np.zeros(3), 1)
    assert camera.pivot == pytest.approx((9, 8, 7))


def test_pan_remains_a_composition_offset_and_orbit_zoom_remain_available():
    camera = OrbitCamera(pivot=(0, 0, 1))
    tracker = CameraTracker(CameraTrackingConfig(axes="xyz", smoothing=0))
    tracker.start(1, camera)
    tracker.advance(camera, np.array((0, 0, 1)), 1 / 60)
    camera.pan(70, 30, 800)
    offset = camera.pivot.copy() - (0, 0, 1)
    camera.orbit(40, 20)
    camera.dolly(2)
    orbit = camera.yaw, camera.pitch, camera.distance
    for x in range(4):
        tracker.advance(camera, np.array((x, 0, 1)), 1 / 60)
        assert camera.pivot == pytest.approx(np.array((x, 0, 1)) + offset)
        assert (camera.yaw, camera.pitch, camera.distance) == orbit


def test_tracking_translation_preserves_exact_roll_and_intrinsics():
    camera = OrbitCamera()
    view = CameraView(
        eye=np.array((3, 0, 1)),
        target=np.array((0, 0, 1)),
        up=np.array((0, 1, 1)),
        focal_length=np.array((0.05, 0.05)),
        sensor_size=np.array((0.036, 0.024)),
        principal_offset=np.array((0.002, 0.001)),
    )
    camera.adopt(view, exact=True)
    tracker = CameraTracker(CameraTrackingConfig(axes="xyz", smoothing=0))
    tracker.start(1, camera)
    tracker.advance(camera, np.array((1, 2, 3)), 0.1)
    moved = camera.view()
    assert moved.eye - moved.target == pytest.approx(view.eye - view.target)
    assert moved.up == pytest.approx(view.up)
    assert moved.focal_length == pytest.approx(view.focal_length)
    assert moved.principal_offset == pytest.approx(view.principal_offset)


def test_bobbing_is_attenuated_without_overshoot_on_a_stationary_step():
    camera = OrbitCamera()
    tracker = CameraTracker(CameraTrackingConfig(axes="xyz", smoothing=0.25))
    tracker.start(1, camera)
    heights = []
    for frame in range(240):
        tracker.advance(camera, np.array((0, 0, np.sin(2 * np.pi * 4 * frame / 60))), 1 / 60)
        heights.append(camera.pivot[2])
    assert np.ptp(heights[120:]) < 0.25  # Input peak-to-peak height is 2 m.
    previous = camera.pivot.copy()
    for _ in range(120):
        tracker.advance(camera, np.array((2, 0, 4)), 1 / 60)
        assert previous[2] <= camera.pivot[2] <= 4
        previous = camera.pivot.copy()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, "bad"])
def test_invalid_persisted_smoothing_uses_default(value):
    assert CameraTrackingConfig.from_mapping({"smoothing": value}) == CameraTrackingConfig()


@pytest.mark.parametrize("settings", [{"axes": "xz"}, {"smoothing": -1}, {"smoothing": np.nan}])
def test_invalid_programmatic_tracking_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        CameraTrackingConfig(**settings)


def test_missing_or_nonfinite_target_pose_is_not_replaced_with_world_origin():
    node = SceneNode(1, "root", NodeType.LINK, body_index=1)
    session = SimpleNamespace(frame=SceneFrame())
    assert tracking_position(session, node) is None
    session.frame.body_xpos = np.array(((0, 0, 0), (1, 2, np.nan)))
    assert tracking_position(session, node) is None
    session.frame.body_xpos[1, 2] = 3
    assert tracking_position(session, node) == pytest.approx((1, 2, 3))


def _app():
    app = ViewerApp.__new__(ViewerApp)
    node = SceneNode(11, "root", NodeType.LINK, body_index=1)
    app.session = SimpleNamespace(
        nodes=[node],
        node=lambda node_id: next((n for n in app.session.nodes if n.node_id == node_id), None),
        frame=SceneFrame(body_xpos=np.array(((0, 0, 0), (8, 4, 10)))),
        adapter=object(),
    )
    app.camera = OrbitCamera(pivot=(0, 0, 2))
    app.camera_tracker = CameraTracker()
    app._model_camera_id = -1
    app.camera_out = SimpleNamespace(set_camera=lambda view: setattr(app.session, "camera", view))
    return app


def test_app_tracks_the_displayed_frame_and_drops_deleted_or_replaced_targets():
    app = _app()
    app.track_node(11)
    app._sync_camera_tracking(0.25)
    assert app.session.camera.target == pytest.approx((4, 2, 2))
    app.session.nodes.clear()
    app._sync_camera_tracking(0.25)
    assert app.tracking_node_id is None
    assert app.camera.pivot == pytest.approx((4, 2, 2))
    app.session.nodes.append(SceneNode(11, "new root", NodeType.LINK, body_index=1))
    app.track_node(11)
    app.session.adapter = object()
    app._sync_camera_tracking(0.25)
    assert app.tracking_node_id is None


def test_body_lookup_rejects_missing_and_ambiguous_names_without_losing_target():
    app = _app()
    viewer = Viewer(app, app.session, None, None, None)
    viewer.track_body("root")
    assert viewer.tracking_node_id == 11
    with pytest.raises(ValueError):
        viewer.track_body("missing")
    app.session.nodes.append(SceneNode(12, "root", NodeType.LINK, body_index=2))
    with pytest.raises(ValueError):
        viewer.track_body("root")
    assert viewer.tracking_node_id == 11
    viewer.track_body(2)
    assert viewer.tracking_node_id == 12
    viewer.track_body(None)
    assert viewer.tracking_node_id is None
