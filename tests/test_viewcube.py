from __future__ import annotations

import numpy as np
import pytest

from forge_viewer.types import CameraView
from forge_viewer.ui import viewcube as vc


def cam(eye, target=(0.0, 0.0, 0.0)) -> CameraView:
    return CameraView(
        eye=np.array(eye, np.float32),
        target=np.array(target, np.float32),
        up=np.array([0.0, 0.0, 1.0], np.float32),
    )


CENTER = (100.0, 100.0)
R = 34.0
BALL = 9.5


def balls_from(eye):
    return vc.layout(cam(eye), CENTER, R, BALL)


def find(balls, axis, sign):
    return next(b for b in balls if b.axis == axis and b.sign == sign)


def test_layout_is_sorted_far_to_near():

    balls = balls_from((4.0, -4.0, 3.0))
    depths = [b.depth for b in balls]
    assert depths == sorted(depths, reverse=True)


def test_all_balls_are_the_same_size():

    balls = balls_from((4.0, -4.0, 3.0))
    radii = {round(b.radius, 9) for b in balls}
    assert len(radii) == 1


def test_layout_is_orthographic_not_perspective():

    near = balls_from((4.0, -4.0, 3.0))
    far = balls_from((40.0, -40.0, 30.0))
    for a, b in zip(near, far, strict=True):
        assert abs(a.screen[0] - b.screen[0]) < 1e-9
        assert abs(a.screen[1] - b.screen[1]) < 1e-9
        assert abs(a.radius - b.radius) < 1e-9


def test_opposite_balls_are_symmetric_about_the_center():

    balls = balls_from((3.0, -5.0, 2.0))
    for axis in range(3):
        p = find(balls, axis, 1.0).screen
        n = find(balls, axis, -1.0).screen
        assert abs((p[0] + n[0]) * 0.5 - CENTER[0]) < 1e-6
        assert abs((p[1] + n[1]) * 0.5 - CENTER[1]) < 1e-6


def test_hit_test_prefers_the_nearer_ball():

    a = vc.Ball(axis=0, sign=1.0, screen=(50.0, 50.0), radius=10.0, depth=-0.9)
    b = vc.Ball(axis=1, sign=-1.0, screen=(52.0, 50.0), radius=10.0, depth=+0.9)
    assert vc.hit_test([b, a], (51.0, 50.0)) is a


def test_back_ball_fades_instead_of_hiding_only_its_label():
    balls = balls_from((5.0, 0.0, 0.0))
    assert find(balls, 0, 1.0).alpha == 1.0
    assert find(balls, 0, -1.0).alpha == 0.0


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_nearest_ball_is_drawn_last_on_top(axis, sign):
    # Painter's order: the ball on the camera's side is drawn last (on top)
    # and fully opaque; the far ball is drawn first and fully faded out.
    eye = [0.0, 0.0, 0.0]
    eye[axis] = 5.0 * sign
    balls = balls_from(eye)
    nearest, farthest = balls[-1], balls[0]
    assert (nearest.axis, nearest.sign) == (axis, sign)
    assert (farthest.axis, farthest.sign) == (axis, -sign)
    assert nearest.alpha == 1.0
    assert farthest.alpha == 0.0


def test_back_ball_fade_is_continuous():
    assert vc._back_alpha(vc.BACK_FADE_START) == pytest.approx(1.0)
    assert vc._back_alpha((vc.BACK_FADE_START + vc.BACK_FADE_END) * 0.5) == pytest.approx(0.5)
    assert vc._back_alpha(vc.BACK_FADE_END) == pytest.approx(0.0)


def test_axis_line_and_ball_are_one_lollipop_outline():
    outline = vc._lollipop_outline((10.0, 20.0), (40.0, 60.0), 10.0, 2.0)
    assert len(outline) > 24
    assert np.linalg.norm(np.asarray(outline[0]) - (10.0, 20.0)) == pytest.approx(1.0)
    assert np.linalg.norm(np.asarray(outline[-1]) - (10.0, 20.0)) == pytest.approx(1.0)


def test_lollipop_becomes_one_circle_when_the_ball_covers_the_center():
    ball = (CENTER[0] + BALL * 0.5, CENTER[1])
    outline = vc._lollipop_outline(CENTER, ball, BALL, 2.0)
    radii = np.linalg.norm(np.asarray(outline) - ball, axis=1)
    assert radii == pytest.approx(np.full(24, BALL))


def test_hover_does_not_resize_the_ball():
    class Overlay:
        def __init__(self) -> None:
            self.outline = None

        def circle_filled(self, *args, **kwargs) -> None:
            pass

        def fringed_concave_fill(self, points, _color) -> None:
            self.outline = np.asarray(points)

        def centered_label(self, *args, **kwargs) -> None:
            pass

    ball = vc.Ball(axis=0, sign=1.0, screen=(140.0, 100.0), radius=BALL, depth=-0.5)
    cube = vc.ViewCube()
    cube._center = CENTER
    cube._balls = [ball]
    cube._hover = ball
    overlay = Overlay()

    cube.draw(overlay)

    expected = np.asarray(vc._lollipop_outline(CENTER, ball.screen, BALL, vc.LINE_PT))
    assert overlay.outline == pytest.approx(expected)


def test_negative_label_crossfades_in_when_looking_down_the_negative_axis():
    normal = vc.Ball(axis=0, sign=-1.0, screen=CENTER, radius=BALL, depth=-0.5)
    aligned = vc.Ball(axis=0, sign=-1.0, screen=CENTER, radius=BALL, depth=-1.0)
    assert vc._label_alpha(normal, False) == 0.0
    assert vc._label_alpha(aligned, False) == 1.0


def test_fully_faded_ball_is_not_clickable():
    faded = vc.Ball(axis=0, sign=-1.0, screen=CENTER, radius=BALL, depth=1.0, alpha=0.0)
    assert vc.hit_test([faded], CENTER) is None


@pytest.mark.parametrize(
    ("axis", "sign", "yaw", "pitch"),
    [
        (0, 1.0, 0.0, 0.0),
        (0, -1.0, 180.0, 0.0),
        (1, 1.0, 90.0, 0.0),
        (1, -1.0, 270.0, 0.0),
        (2, 1.0, vc.TOP_YAW, vc.PITCH_LIMIT),
        (2, -1.0, vc.TOP_YAW, -vc.PITCH_LIMIT),
    ],
)
def test_click_target_is_the_axis_you_clicked(axis, sign, yaw, pitch):

    got_yaw, got_pitch = vc.yaw_pitch_for(axis, sign, current_yaw=37.5)
    assert abs((got_yaw - yaw + 180.0) % 360.0 - 180.0) < 1e-6
    assert abs(got_pitch - pitch) < 1e-6


def test_top_view_yaw_matches_the_camera_preset():

    from forge_viewer.ui.camera import PRESETS

    assert vc.yaw_pitch_for(2, 1.0, 0.0) == PRESETS["top"]
    assert vc.yaw_pitch_for(2, -1.0, 0.0) == PRESETS["bottom"]


def test_top_view_is_right_handed_x_right_y_up():

    from forge_viewer.ui.camera import camera_basis

    yaw, pitch = vc.yaw_pitch_for(2, 1.0, 0.0)
    y, p = np.radians(yaw), np.radians(pitch)
    direction = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    right, up, _f = camera_basis(cam(direction * 5.0))
    assert right[0] > 0.99
    assert up[1] > 0.99


def test_widget_sits_inside_the_viewport_corner():

    rect = (291.0, 28.0, 680.0, 554.0)
    cx, cy = vc.widget_center(rect, 1.0)
    outer = vc.RADIUS_PT + vc.BALL_PT
    assert rect[0] <= cx - outer and cx + outer <= rect[0] + rect[2]
    assert rect[1] <= cy - outer and cy + outer <= rect[1] + rect[3]
    assert abs((rect[0] + rect[2]) - (cx + outer)) - vc.MARGIN_PT < 1e-6


def test_view_gizmo_geometry_follows_layout_scale():
    rect = (291.0, 28.0, 680.0, 554.0)
    cube = vc.ViewCube()
    view = cam((4.0, -4.0, 3.0))

    cube.update(view, rect, cursor=(-1.0, -1.0), style_scale=1.0)
    center_1 = vc.widget_center(rect, 1.0)
    balls_1 = {(ball.axis, ball.sign): ball for ball in cube.balls}

    cube.update(view, rect, cursor=(-1.0, -1.0), style_scale=2.0)
    center_2 = vc.widget_center(rect, 2.0)
    balls_2 = {(ball.axis, ball.sign): ball for ball in cube.balls}

    for key, ball_1 in balls_1.items():
        ball_2 = balls_2[key]
        offset_1 = np.asarray(ball_1.screen) - center_1
        offset_2 = np.asarray(ball_2.screen) - center_2
        assert ball_2.radius == pytest.approx(ball_1.radius * 2.0)
        assert offset_2 == pytest.approx(offset_1 * 2.0)
