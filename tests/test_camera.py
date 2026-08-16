from __future__ import annotations

import numpy as np
import pytest

from forge_viewer import math3d
from forge_viewer.commands import SetCamera
from forge_viewer.types import CameraView
from forge_viewer.ui.camera import CameraOut, OrbitCamera


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


def test_orbit_keeps_the_distance_to_the_pivot():

    cam = OrbitCamera(pivot=np.array([1.0, -2.0, 0.5]), distance=3.7)
    before = np.linalg.norm(cam.eye() - cam.pivot)
    for dx, dy in ((30.0, 12.0), (-80.0, 40.0), (200.0, -300.0)):
        cam.orbit(dx, dy)
        assert np.linalg.norm(cam.eye() - cam.pivot) == pytest.approx(before, rel=1e-9)
        assert np.allclose(cam.pivot, [1.0, -2.0, 0.5])


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
    assert ease(0.3) > 0.45, f"t=0.3 才走了 {ease(0.3):.3f}，起步太肉"

    steps = [ease((i + 1) / 20.0) - ease(i / 20.0) for i in range(20)]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(steps)), "增量不是单调不增——中间在加速"


def test_look_from_can_be_retargeted_mid_flight():

    cam = OrbitCamera(yaw=0.0, pitch=0.0)
    sink = RecordingSink()
    cam.look_from(90.0, 40.0, sink, animate=True)
    for _ in range(3):
        cam.advance(0.05, sink)
    midway = cam.yaw
    assert 0.0 < midway < 90.0, "第一段缓动就没动"

    cam.look_from(-60.0, -20.0, sink, animate=True)
    assert abs(cam.yaw - midway) < 1e-6, "重新定向时把相机弹回了别处"
    for _ in range(40):
        cam.advance(0.05, sink)
    assert abs(cam.yaw - (-60.0)) < 0.5 and abs(cam.pitch - (-20.0)) < 0.5, (
        f"重新定向之后停在 yaw={cam.yaw:.1f} pitch={cam.pitch:.1f}"
    )


def test_every_public_setter_marks_dirty_and_publishes():

    writable = [
        name
        for name, attr in vars(OrbitCamera).items()
        if isinstance(attr, property) and attr.fset is not None
    ]
    assert {"pivot", "yaw", "pitch", "distance"} <= set(writable), (
        f"这些可写属性不见了，判据要跟着改：{writable}"
    )

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
        assert len(sink.views) == before + 1, f"写 {name} 之后没有下发——改了不生效且不报错"
