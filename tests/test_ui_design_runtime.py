"""Native timeline sampling uses one atomic Session joint update."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from examples.ui_design.native import docking_study
from examples.ui_design.native.scene_preview import ReferenceViewCube

from mojive import CameraView, Scene
from mojive import commands as cmd
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.static import StaticSceneAdapter
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.ui import viewcube


def timeline_study(session):
    study = docking_study.Study.__new__(docking_study.Study)
    document = json.loads(docking_study.ROOT.joinpath("document.json").read_text())
    study.preview = SimpleNamespace(session=session, document=document)
    study.pose_time = 4.0
    return study


@pytest.fixture
def joint_study():
    pytest.importorskip("mujoco")
    session = Session(MuJoCoAdapter(resolve("joint_types")))
    assert session.submit(cmd.Pause()).ok
    try:
        yield timeline_study(session)
    finally:
        session.release()


@pytest.mark.physics
@pytest.mark.parametrize("pose_time", [2.0, 4.0])
def test_timeline_samples_multiple_joints_in_one_atomic_command(
    monkeypatch, joint_study, pose_time
):
    study = joint_study
    session = study.preview.session
    before = session.adapter.data.qpos.copy()
    submit = Mock(wraps=session.submit)
    monkeypatch.setattr(session, "submit", submit)
    assert study.sample_pose(pose_time)
    assert study.pose_time == pose_time
    submit.assert_called_once()
    assert isinstance(submit.call_args.args[0], cmd.SetQposBatch)
    expected = before.copy()
    joints = {joint.name: joint for joint in session.joints}
    ratio = pose_time / 4
    expected[joints["hinge_limited"].qpos_adr] = np.radians(-35 * ratio)
    expected[joints["slide"].qpos_adr] = 0.35 * ratio
    ball_address = joints["ball"].qpos_adr
    half_angle = np.radians(20 * ratio) / 2
    expected[ball_address : ball_address + 4] = (np.cos(half_angle), 0, np.sin(half_angle), 0)
    np.testing.assert_allclose(session.adapter.data.qpos, expected, atol=1e-12)


@pytest.mark.physics
@pytest.mark.parametrize("blocked", ["playing", "read_only"])
def test_timeline_keeps_cursor_and_joints_when_editing_is_unavailable(
    monkeypatch, joint_study, blocked
):
    study = joint_study
    session = study.preview.session
    if blocked == "playing":
        assert session.submit(cmd.Play()).ok
    else:
        session.adapter.caps = replace(session.adapter.caps, write_qpos=False)
    before = session.adapter.data.qpos.copy()
    submit = Mock(wraps=session.submit)
    monkeypatch.setattr(session, "submit", submit)
    assert not study.sample_pose(2.0)
    assert study.pose_time == 4.0
    np.testing.assert_array_equal(session.adapter.data.qpos, before)
    submit.assert_not_called()


def test_timeline_without_matching_joints_keeps_its_cursor(monkeypatch):
    adapter = StaticSceneAdapter(Scene())
    adapter.caps = replace(adapter.caps, write_qpos=True)
    session = Session(adapter)
    try:
        study = timeline_study(session)
        submit = Mock(wraps=session.submit)
        monkeypatch.setattr(session, "submit", submit)
        assert not study.sample_pose(2.0)
        assert study.pose_time == 4.0
        submit.assert_not_called()
    finally:
        session.release()


@pytest.mark.physics
def test_timeline_rejected_batch_keeps_cursor_and_joint_state(monkeypatch, joint_study):
    study = joint_study
    session = study.preview.session
    before = session.adapter.data.qpos.copy()
    monkeypatch.setattr(session.adapter, "set_qpos_batch", lambda indices, values: False)
    assert not study.sample_pose(2.0)
    assert study.pose_time == 4.0
    np.testing.assert_array_equal(session.adapter.data.qpos, before)


@pytest.mark.parametrize(
    "rect,effective_scale", [((0, 0, 342, 424), 1.4), ((0, 0, 1200, 900), 2.0)]
)
def test_reference_viewcube_draw_and_hit_regions_use_the_same_scale(rect, effective_scale):
    camera = CameraView(eye=np.array([4.0, -4.0, 3.0]), target=np.zeros(3))
    cube = ReferenceViewCube(selection_padding=1.2)
    cube.update(camera, rect, (-100, -100), 2.0)
    assert cube.reference_scale == effective_scale
    center = viewcube.widget_center(rect, effective_scale)
    expected = viewcube.layout(
        camera, center, viewcube.RADIUS_PT * effective_scale, viewcube.BALL_PT * effective_scale
    )
    for actual, reference in zip(cube.balls, expected, strict=True):
        assert (actual.axis, actual.sign) == (reference.axis, reference.sign)
        assert actual.screen == pytest.approx(reference.screen)
        assert actual.radius == pytest.approx(reference.radius)
    front = min(cube.balls, key=lambda ball: ball.depth)
    hovered = cube.update(camera, rect, front.screen, 2.0)
    assert hovered is not None
    assert (hovered.axis, hovered.sign) == (front.axis, front.sign)
    overlay = Mock()
    cube.draw(overlay, 2.0)
    backdrop = overlay.circle_filled.call_args_list[0]
    assert backdrop.args[0] == pytest.approx(center)
    assert backdrop.args[1] == pytest.approx(viewcube.BACKDROP_RADIUS_PT * effective_scale)
