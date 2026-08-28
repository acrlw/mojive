from __future__ import annotations

import numpy as np
import pytest

from forge_viewer import math3d
from forge_viewer.adapters.base import CameraInfo, NodeType, SceneFrame, SceneNode, SceneSource
from forge_viewer.commands import SetCamera
from forge_viewer.render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from forge_viewer.types import CameraView
from forge_viewer.ui.camera import CameraOut, OrbitCamera, camera_basis
from forge_viewer.ui.camera_preview import CameraPreview


class RecordingSink:
    def __init__(self) -> None:
        self.views: list = []

    def set_camera(self, camera) -> None:
        self.views.append(camera)


class RecordingSession:
    def __init__(self) -> None:
        self.commands: list = []

    def submit(self, command):
        self.commands.append(command)
        return None


class PreviewSession:
    def __init__(self, camera: CameraView) -> None:
        self.selected_node = SceneNode(
            node_id=1,
            name="inspection",
            type=NodeType.CAMERA,
            camera_index=0,
        )
        self.camera = camera
        self.cameras = [CameraInfo(camera_id=42, name="inspection")]

    def camera_view(self, camera_id: int) -> CameraView | None:
        return self.camera if camera_id == 42 else None


def visible_height_at_target(view) -> float:

    v, p = view.view_matrix(), view.proj_matrix()
    forward = np.asarray(view.forward(), np.float64)
    eye = np.asarray(view.eye, np.float64)
    depth = float(np.dot(np.asarray(view.target, np.float64) - eye, forward))
    hits = []
    for ndc_y in (-1.0, 1.0):
        o, d = math3d.unproject_ray(0.0, ndc_y, v, p)
        o = np.asarray(o, np.float64)
        d = np.asarray(d, np.float64)
        t = (depth - float(np.dot(o - eye, forward))) / float(np.dot(d, forward))
        hits.append(o + d * t)
    return float(np.linalg.norm(hits[1] - hits[0]))


def corners(lo, hi) -> np.ndarray:
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def inside_frustum(view, pts) -> bool:

    mvp = np.asarray(view.proj_matrix(), np.float64) @ np.asarray(view.view_matrix(), np.float64)
    h = np.concatenate([np.asarray(pts, np.float64), np.ones((len(pts), 1))], axis=1) @ mvp.T
    w = h[:, 3]
    if np.any(w <= 0.0):
        return False
    return bool(
        np.all(np.abs(h[:, 0]) <= w)
        and np.all(np.abs(h[:, 1]) <= w)
        and np.all(h[:, 2] >= -w)
        and np.all(h[:, 2] <= w)
    )


def test_camera_intrinsics_create_an_off_center_projection():
    view = CameraView(
        near=0.02,
        far=100.0,
        focal_length=np.array([0.05, 0.04], np.float32),
        sensor_size=np.array([0.036, 0.024], np.float32),
        principal_offset=np.array([0.003, -0.002], np.float32),
    )
    projection = view.proj_matrix()
    assert view.uses_intrinsics()
    assert projection[0, 0] == pytest.approx(2.0 * 0.05 / 0.036)
    assert projection[1, 1] == pytest.approx(2.0 * 0.04 / 0.024)
    assert projection[0, 2] == pytest.approx(2.0 * 0.003 / 0.036)
    assert projection[1, 2] == pytest.approx(-2.0 * -0.002 / 0.024)
    assert view.with_aspect(3.0).proj_matrix() == pytest.approx(projection)


def test_orbit_keeps_the_distance_to_the_pivot():

    cam = OrbitCamera(pivot=np.array([1.0, -2.0, 0.5]), distance=3.7)
    before = np.linalg.norm(cam.eye() - cam.pivot)
    for dx, dy in ((30.0, 12.0), (-80.0, 40.0), (200.0, -300.0)):
        cam.orbit(dx, dy)
        assert np.linalg.norm(cam.eye() - cam.pivot) == pytest.approx(before, rel=1e-9)
        assert np.allclose(cam.pivot, [1.0, -2.0, 0.5])


def test_default_camera_looks_from_positive_x_with_positive_y_on_the_right():
    cam = OrbitCamera()
    view = CameraView()

    assert cam.eye() == pytest.approx((4.0, 0.0, 0.0), abs=1e-7)
    assert cam.view().forward() == pytest.approx((-1.0, 0.0, 0.0), abs=1e-7)
    assert cam.view().up == pytest.approx((0.0, 0.0, 1.0))
    right, _up, _forward = camera_basis(cam.view())
    assert right == pytest.approx((0.0, 1.0, 0.0), abs=1e-7)
    assert view.eye == pytest.approx(cam.eye(), abs=1e-7)
    assert view.forward() == pytest.approx(cam.view().forward(), abs=1e-7)


def test_dolly_only_changes_the_distance():

    cam = OrbitCamera(pivot=np.array([0.4, 0.2, -0.1]), distance=5.0)
    dir_before = cam.direction().copy()
    pivot_before = cam.pivot.copy()
    cam.dolly(4.0)
    assert cam.distance < 5.0
    assert np.allclose(cam.pivot, pivot_before)
    assert np.allclose(cam.direction(), dir_before)
    cam.dolly(-4.0)
    assert cam.distance == pytest.approx(5.0, rel=1e-9)


def test_dolly_never_reaches_zero():

    cam = OrbitCamera(distance=5.0)
    for _ in range(100):
        cam.dolly(10.0)
    assert cam.distance > 0.0


def test_near_plane_follows_the_camera_in_for_close_inspection():

    cam = OrbitCamera(distance=10.0, near=0.5, far=100.0)
    far_view = cam.view()
    cam.distance = 0.1
    close_view = cam.view()

    assert close_view.near < far_view.near / 20.0
    assert close_view.near < cam.distance * 0.01
    assert close_view.near > 0.0


def test_angles_are_degrees():

    cam = OrbitCamera(distance=4.0, yaw=0.0, pitch=30.0)
    assert cam.eye()[2] - cam.pivot[2] == pytest.approx(4.0 * np.sin(np.deg2rad(30.0)))
    cam.yaw = 90.0
    assert cam.eye()[1] - cam.pivot[1] > 0.0
    assert abs(cam.eye()[0] - cam.pivot[0]) < 1e-9
    assert cam.fov_y_deg == pytest.approx(45.0)
    assert cam.fov_y == pytest.approx(np.deg2rad(45.0))


def test_writing_a_property_marks_the_camera_dirty():

    for attr, value in (("distance", 7.0), ("yaw", 12.0), ("pitch", -20.0), ("aspect", 1.9)):
        cam = OrbitCamera()
        sink = RecordingSink()
        cam.publish(sink)
        assert cam.dirty is False
        setattr(cam, attr, value)
        assert cam.dirty is True, attr
        assert cam.advance(0.0, sink) is True
        assert len(sink.views) == 2


def test_assigning_orthographic_directly_still_does_not_jump():

    cam = OrbitCamera(distance=11.0, aspect=1.3)
    before = visible_height_at_target(cam.view())
    cam.orthographic = True
    assert visible_height_at_target(cam.view()) == pytest.approx(before, rel=1e-6)


def test_q_lifts_along_world_z_even_when_looking_straight_down():

    cam = OrbitCamera(pitch=85.0, distance=4.0)
    assert cam.pitch > 80.0
    speed = cam.distance * 1.6
    before = cam.eye().copy()
    cam.fly(0.5, up=1.0)
    delta = cam.eye() - before
    assert delta[2] == pytest.approx(speed * 0.5, rel=1e-9)
    assert np.linalg.norm(delta[:2]) == pytest.approx(0.0, abs=1e-12)


def test_fly_speed_scales_with_the_viewing_distance():

    near = OrbitCamera(distance=1.0)
    far = OrbitCamera(distance=10.0)
    near.fly(0.1, forward=1.0)
    far.fly(0.1, forward=1.0)
    assert np.linalg.norm(far.pivot) == pytest.approx(10.0 * np.linalg.norm(near.pivot), rel=1e-9)


def test_wasd_moves_along_the_view_axes():

    cam = OrbitCamera(pitch=0.0, yaw=0.0, distance=4.0)
    right, _, forward = cam.basis()
    before = cam.pivot.copy()
    cam.fly(1.0, forward=1.0)
    assert np.dot(cam.pivot - before, forward) > 0.0
    before = cam.pivot.copy()
    cam.fly(1.0, right=1.0)
    assert np.dot(cam.pivot - before, right) > 0.0


@pytest.mark.parametrize("distance", [0.7, 4.0, 25.0])
def test_switching_to_orthographic_does_not_jump(distance):

    cam = OrbitCamera(distance=distance, aspect=1.6)
    before = visible_height_at_target(cam.view())
    cam.set_orthographic(True)
    after = visible_height_at_target(cam.view())
    assert after == pytest.approx(before, rel=1e-6)

    cam.set_orthographic(False)
    assert visible_height_at_target(cam.view()) == pytest.approx(before, rel=1e-6)


def test_orthographic_dolly_still_zooms():

    cam = OrbitCamera(distance=4.0)
    cam.set_orthographic(True)
    before = visible_height_at_target(cam.view())
    cam.dolly(4.0)
    assert visible_height_at_target(cam.view()) < before


def test_frame_scene_grows_with_the_bounds_and_keeps_the_box_in_view():

    small = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    big = (np.array([-10.0, -10.0, -10.0]), np.array([10.0, 10.0, 10.0]))

    cam = OrbitCamera(aspect=1.6)
    sink = RecordingSink()
    cam.frame_scene(small, sink, animate=False)
    d_small = cam.distance
    assert inside_frustum(cam.view(), corners(*small))

    cam.frame_scene(big, sink, animate=False)
    assert cam.distance > d_small * 5.0
    assert inside_frustum(cam.view(), corners(*big))


def test_frame_scene_handles_a_lopsided_box():

    bounds = (np.array([-8.0, -0.2, -0.2]), np.array([8.0, 0.2, 0.2]))
    cam = OrbitCamera(aspect=0.6)
    cam.frame_scene(bounds, RecordingSink(), animate=False)
    assert inside_frustum(cam.view(), corners(*bounds))


def test_frame_scene_keeps_distant_ground_in_view():

    bounds = (np.full(3, -1.0), np.full(3, 1.0))
    cam = OrbitCamera(aspect=1.6)
    cam.frame_scene(bounds, RecordingSink(), animate=False)
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.5)
    assert cam.far >= (cam.distance + radius) * 30.0


def test_frame_scene_preserves_scene_clip_planes_when_provided():
    bounds = (np.full(3, -1.0), np.full(3, 1.0))
    clip = CameraView(near=0.016, far=50.0)
    cam = OrbitCamera(aspect=1.6)

    view = cam.frame_scene(bounds, RecordingSink(), animate=False, clip=clip)

    assert view.near == pytest.approx(clip.near)
    assert view.far == pytest.approx(clip.far)


def test_look_from_target_centers_selection_and_padding_controls_distance():
    center = np.array((3.0, -2.0, 1.0))
    cam = OrbitCamera(aspect=1.6)
    sink = RecordingSink()

    cam.look_from_target(90.0, 0.0, center, 2.0, sink, margin=1.0, animate=False)
    tight_distance = cam.distance
    assert cam.pivot == pytest.approx(center)
    assert cam.yaw == pytest.approx(90.0)
    assert cam.pitch == pytest.approx(0.0)

    cam.look_from_target(0.0, 30.0, center, 2.0, sink, margin=1.75, animate=False)
    assert cam.distance == pytest.approx(tight_distance * 1.75)
    assert cam.pivot == pytest.approx(center)
    assert cam.yaw == pytest.approx(0.0)
    assert cam.pitch == pytest.approx(30.0)


def test_look_from_target_updates_orthographic_framing_height():
    cam = OrbitCamera(orthographic=True, aspect=1.0)
    sink = RecordingSink()

    cam.look_from_target(0.0, 0.0, np.zeros(3), 1.0, sink, margin=1.0, animate=False)
    tight_height = cam.ortho_height
    cam.look_from_target(0.0, 0.0, np.zeros(3), 1.0, sink, margin=2.0, animate=False)

    assert cam.ortho_height == pytest.approx(tight_height * 2.0)


def test_look_from_target_uses_fast_ease_out_quart():
    from forge_viewer.ui.camera import FRAME_DURATION, _ease_out_quart

    cam = OrbitCamera(pivot=np.zeros(3))
    sink = RecordingSink()
    target = np.array((8.0, -4.0, 2.0))

    cam.look_from_target(90.0, 0.0, target, 1.0, sink, animate=True)
    cam.advance(FRAME_DURATION * 0.25, sink)

    assert cam.pivot == pytest.approx(target * _ease_out_quart(0.25))
    assert _ease_out_quart(0.25) > 0.65


def test_adopted_scene_clip_planes_survive_free_camera_navigation():
    clip = CameraView(
        eye=np.array([1.2, -2.1, 1.55], np.float32),
        target=np.array([0.0, 0.0, 0.7], np.float32),
        near=0.01667475,
        far=50.024253,
    )
    cam = OrbitCamera()
    cam.adopt(clip)

    cam.orbit(80.0, -25.0)
    cam.pan(30.0, -15.0, 900.0)
    cam.dolly(-2.0)
    view = cam.view()

    assert view.near == pytest.approx(clip.near)
    assert view.far == pytest.approx(clip.far)


def test_frame_scene_publishes_the_camera_to_the_backend():

    cam = OrbitCamera()
    sink = RecordingSink()
    view = cam.frame_scene((np.full(3, -1.0), np.full(3, 1.0)), sink, animate=False)
    assert len(sink.views) == 1
    assert np.allclose(sink.views[-1].eye, view.eye)


def test_camera_reaches_both_the_backend_and_the_session():

    sink = RecordingSink()
    session = RecordingSession()
    out = CameraOut(backend=sink, session=session)

    cam = OrbitCamera()
    cam.frame_scene((np.full(3, -1.0), np.full(3, 1.0)), out, animate=False)

    assert len(sink.views) == 1
    assert [type(c) for c in session.commands] == [SetCamera]
    assert np.allclose(session.commands[-1].camera.eye, sink.views[-1].eye)

    cam.orbit(20.0, 5.0)
    cam.advance(0.016, out)
    assert len(sink.views) == 2
    assert len(session.commands) == 2


def test_free_camera_can_adopt_a_rolled_model_camera_without_an_eye_jump():
    view = CameraView(
        eye=np.array([2.0, -3.0, 4.0], np.float32),
        target=np.array([0.5, 0.25, 1.0], np.float32),
        up=np.array([0.3, 0.7, 0.6], np.float32),
        fov_y=np.deg2rad(53.0),
        near=0.003,
        far=700.0,
        aspect=1.7,
    )
    cam = OrbitCamera()
    cam.adopt(view)
    adopted = cam.view()
    assert adopted.eye == pytest.approx(view.eye)
    assert adopted.target == pytest.approx(view.target)
    assert adopted.fov_y == pytest.approx(view.fov_y)
    assert adopted.near == pytest.approx(view.near)
    assert adopted.far == pytest.approx(view.far)


def test_advance_publishes_every_frame_of_the_easing():

    cam = OrbitCamera(distance=1.0)
    sink = RecordingSink()
    cam.frame_scene((np.full(3, -5.0), np.full(3, 5.0)), sink, animate=True)
    published = len(sink.views)
    for _ in range(12):
        cam.advance(0.05, sink)
    assert len(sink.views) > published + 5
    assert not cam.animating

    settled = len(sink.views)
    cam.advance(0.05, sink)
    assert len(sink.views) == settled


def test_easing_takes_the_short_way_around():

    cam = OrbitCamera(yaw=-179.0)
    sink = RecordingSink()
    cam.look_from(179.0, 0.0, sink, animate=True)
    seen = []
    for _ in range(12):
        cam.advance(0.05, sink)
        seen.append(cam.yaw)
    assert min(abs(y) for y in seen) > 170.0


def test_easing_is_ease_out_not_ease_in_out():

    import itertools

    from forge_viewer.ui.camera import _ease_out_quad as ease

    assert ease(0.0) == 0.0 and ease(1.0) == 1.0
    assert ease(0.3) > 0.45

    steps = [ease((i + 1) / 20.0) - ease(i / 20.0) for i in range(20)]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(steps))


def test_look_from_can_be_retargeted_mid_flight():

    cam = OrbitCamera(yaw=0.0, pitch=0.0)
    sink = RecordingSink()
    cam.look_from(90.0, 40.0, sink, animate=True)
    for _ in range(3):
        cam.advance(0.05, sink)
    midway = cam.yaw
    assert 0.0 < midway < 90.0

    cam.look_from(-60.0, -20.0, sink, animate=True)
    assert abs(cam.yaw - midway) < 1e-6
    for _ in range(40):
        cam.advance(0.05, sink)
    assert abs(cam.yaw - (-60.0)) < 0.5 and abs(cam.pitch - (-20.0)) < 0.5


def test_every_public_setter_marks_dirty_and_publishes():

    writable = [
        name
        for name, attr in vars(OrbitCamera).items()
        if isinstance(attr, property) and attr.fset is not None
    ]
    assert {"pivot", "yaw", "pitch", "distance"} <= set(writable)

    samples = {
        "pivot": np.array([1.0, 2.0, 3.0]),
        "yaw": 12.5,
        "pitch": 7.5,
        "distance": 6.25,
        "fov_y_deg": 55.0,
        "aspect": 1.75,
        "orthographic": True,
        "ortho_height": 9.0,
    }
    for name in writable:
        if name not in samples:
            continue
        cam = OrbitCamera()
        sink = RecordingSink()
        cam.advance(0.0, sink)
        before = len(sink.views)
        setattr(cam, name, samples[name])
        cam.advance(0.0, sink)
        assert len(sink.views) == before + 1


def test_pinned_camera_preview_keeps_camera_and_selection() -> None:
    camera = CameraView(eye=np.array((2.0, -3.0, 1.0), np.float32))
    session = PreviewSession(camera)
    preview = CameraPreview()

    name, pinned_view = preview.selected_camera(session)
    assert name == "inspection"
    assert pinned_view is not None
    preview.set_pinned(True)
    camera.eye[:] = 9.0
    session.selected_node = None

    name, pinned_view = preview.selected_camera(session)
    assert preview.pinned
    assert name == "inspection"
    assert pinned_view.eye == pytest.approx((2.0, -3.0, 1.0))

    preview.set_pinned(False)
    assert preview.selected_camera(session) == ("", None)


def test_locked_camera_preview_tracks_camera_after_selection_changes() -> None:
    camera = CameraView(eye=np.array((2.0, -3.0, 1.0), np.float32))
    session = PreviewSession(camera)
    preview = CameraPreview()

    assert preview.selected_camera(session)[0] == "inspection"
    preview.set_locked(True)
    camera.eye[:] = (4.0, 5.0, 6.0)
    session.selected_node = None

    name, locked_view = preview.selected_camera(session)
    assert preview.locked and not preview.pinned
    assert name == "inspection"
    assert locked_view is not None
    assert locked_view.eye == pytest.approx((4.0, 5.0, 6.0))

    preview.set_pinned(True)
    camera.eye[:] = 9.0
    assert preview.pinned and not preview.locked
    assert preview.selected_camera(session)[1].eye == pytest.approx((4.0, 5.0, 6.0))


def test_camera_preview_copies_the_main_render_state() -> None:
    class Peer:
        def __init__(self) -> None:
            self.flags = {}
            self.debug_view = None
            self.label_mode = None
            self.frame_mode = None
            self.bvh_depth = None
            self.camera = None

        def resize(self, *_size) -> None:
            pass

        def set_scene(self, _source) -> None:
            pass

        def set_flag(self, flag, value) -> None:
            self.flags[flag] = value

        def set_debug_view(self, value) -> None:
            self.debug_view = value

        def set_label_mode(self, value) -> None:
            self.label_mode = value

        def set_frame_mode(self, value) -> None:
            self.frame_mode = value

        def set_bvh_depth(self, value) -> None:
            self.bvh_depth = value

        def set_camera(self, value) -> None:
            self.camera = value

        def highlight(self, _value) -> None:
            pass

        def update(self, _frame) -> None:
            pass

        def render(self):
            return None

    peer = Peer()

    class Main:
        def create_peer(self, *_size):
            return peer

        def render_options(self):
            return (RenderFlag.HAZE, RenderFlag.SHADOW)

        def get_flag(self, flag):
            return flag is RenderFlag.HAZE

        def get_debug_view(self):
            return DebugView.NORMAL

        def get_label_mode(self):
            return LabelMode.BODY

        def get_frame_mode(self):
            return FrameMode.WORLD

        def get_bvh_depth(self):
            return 3

    preview = CameraPreview()
    preview.update(Main(), SceneSource(), 1, SceneFrame(), CameraView(), (320, 180))

    assert peer.flags == {RenderFlag.HAZE: True, RenderFlag.SHADOW: False}
    assert peer.debug_view is DebugView.NORMAL
    assert peer.label_mode is LabelMode.BODY
    assert peer.frame_mode is FrameMode.WORLD
    assert peer.bvh_depth == 3
    assert peer.camera.aspect == pytest.approx(16.0 / 9.0)
