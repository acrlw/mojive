"""Physics ownership across commands, state history, and model changes."""

import threading
import time

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene.assets import resolve
from mojive.session import Session

pytestmark = pytest.mark.physics
mujoco = pytest.importorskip("mujoco")


@pytest.fixture(params=[False, True], ids=["model", "workspace"])
def session(request):
    adapter = MuJoCoAdapter(resolve("joint_types"))
    if request.param:
        adapter = WorkspaceAdapter(adapter)
    result = Session(adapter)
    assert result.set_threaded_physics(True)
    yield result
    result.release()


def advance(session, after=None):
    after = session.frame.step if after is None else after
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        frame = session.tick(FrameNeeds(), wall_dt=1 / 120)
        if frame.step > after:
            return frame
        time.sleep(0.002)
    pytest.fail("Physics worker did not publish a new state")


def test_snapshot_stays_immutable_and_trajectory_matches_serial(session):
    primary = getattr(session.adapter, "primary", session.adapter)
    reference = mujoco.MjData(primary.model)
    mujoco.mj_copyData(reference, primary.model, primary.data)
    advance(session)
    displayed = primary.data
    state, stamp = displayed.qpos.copy(), displayed.time
    time.sleep(0.03)
    assert displayed.time == stamp
    np.testing.assert_array_equal(displayed.qpos, state)
    assert session.submit(cmd.Pause())
    frame = session.tick(FrameNeeds())
    mujoco.mj_step(primary.model, reference, nstep=frame.step)
    np.testing.assert_array_equal(primary.data.qpos, reference.qpos)
    np.testing.assert_array_equal(primary.data.qvel, reference.qvel)
    assert primary.data.time == reference.time


def test_pause_step_back_control_reset_and_resume(session):
    first = advance(session).step
    advance(session, first)
    assert session.submit(cmd.Pause())
    before = session.tick(FrameNeeds()).time
    time.sleep(0.02)
    assert session.tick(FrameNeeds(), wall_dt=0.02).time == before
    assert session.submit(cmd.SetCtrl(0, 0.1))
    assert session.submit(cmd.Step(3))
    assert session.tick(FrameNeeds()).time == pytest.approx(before + 3 * session.adapter.timestep())
    assert session.submit(cmd.StepBack())
    assert session.tick(FrameNeeds()).time == pytest.approx(before)
    assert session.adapter.capture_state().ctrl[0] == 0
    assert session.submit(cmd.SetCtrl(0, 0.1))
    assert session.submit(cmd.Play())
    advance(session)
    assert session.submit(cmd.Pause())
    assert session.adapter.capture_state().ctrl[0] == pytest.approx(0.1)
    assert session.submit(cmd.Reset())
    frame = session.tick(FrameNeeds())
    assert frame.step == 0
    assert frame.time == 0
    assert session.submit(cmd.Play())
    assert advance(session).step > 0


def test_model_replacement_invalidates_snapshots_and_failed_load_is_safe(session, tmp_path):
    advance(session)
    assert session.submit(cmd.LoadAsset(resolve("deformables")))
    assert session.paused
    frame = session.tick(FrameNeeds())
    assert frame.step == 0
    assert frame.time == 0
    assert session.submit(cmd.Play())
    advance(session)
    assert not session.submit(cmd.LoadAsset(tmp_path / "missing.xml"))
    assert session.paused
    session.tick(FrameNeeds())
    assert session.submit(cmd.Play())
    advance(session)


def test_take_recording_replay_and_shutdown(session):
    assert session.submit(cmd.StartStateTakeRecording())
    for _ in range(3):
        advance(session)
    assert session.submit(cmd.StopStateTakeRecording())
    assert len(session.state_take_times) >= 3
    first_time = session.state_take_times[0]
    assert session.submit(cmd.SeekStateTake(0))
    assert session.tick(FrameNeeds()).time == first_time
    assert session.submit(cmd.PlayStateTake())
    session.tick(FrameNeeds(), wall_dt=0.01)
    assert session.submit(cmd.PauseStateTake())
    assert session.submit(cmd.Play())
    advance(session)
    thread = session._simulation_driver._thread
    session.release()
    assert not thread.is_alive()


def test_selection_stays_responsive_while_pause_fences_native_work(session, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    original = mujoco.mj_step

    def blocked(*args, **kwargs):
        entered.set()
        assert finish.wait(timeout=3)
        original(*args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_step", blocked)
    session.tick(FrameNeeds(), wall_dt=0.01)
    command = None
    paused = threading.Event()

    def pause():
        session.submit(cmd.Pause())
        paused.set()

    try:
        assert entered.wait(timeout=1)
        assert session.submit(cmd.Select(0))
        command = threading.Thread(target=pause)
        command.start()
        assert not paused.wait(timeout=0.02)
    finally:
        finish.set()
        if command is not None:
            command.join(timeout=2)
    assert paused.is_set()
    assert session.paused


def test_worker_failure_reports_error_and_allows_serial_recovery(session, monkeypatch):
    original = mujoco.mj_step

    def fail(*args, **kwargs):
        raise RuntimeError("test physics failure")

    monkeypatch.setattr(mujoco, "mj_step", fail)
    session.tick(FrameNeeds(), wall_dt=0.01)
    deadline = time.monotonic() + 2
    while not session.paused and time.monotonic() < deadline:
        time.sleep(0.002)
        session.tick(FrameNeeds(), wall_dt=0.01)
    assert session.paused
    assert session._simulation_driver is None
    assert "Physics worker stopped" in session.last_message
    monkeypatch.setattr(mujoco, "mj_step", original)
    assert session.submit(cmd.Play())
    assert session.tick(FrameNeeds()).step > 0


def test_external_clock_does_not_create_another_physics_owner():
    adapter = MuJoCoAdapter(external_clock=True)
    model = mujoco.MjModel.from_xml_path(str(resolve("joint_types")))
    data = mujoco.MjData(model)
    adapter.load_model(model, data)
    session = Session(adapter)
    try:
        assert not session.set_threaded_physics(True)
        session.tick(FrameNeeds(), wall_dt=1)
        assert data.time == 0
    finally:
        session.release()


def test_control_edits_keep_the_clock_and_pause_rebases_it(session):
    advance(session)
    driver = session._simulation_driver
    epoch = driver._epoch
    for value in (0.1, 0.2, 0.3):
        assert session.submit(cmd.SetCtrl(0, value))
        assert driver._epoch == epoch
    assert session.submit(cmd.Pause())
    assert session.submit(cmd.Play())
    assert driver._epoch > epoch
    epoch = driver._epoch
    assert session.submit(cmd.SetSpeed(2.0))
    assert driver._epoch > epoch
    assert driver._speed == 2.0


def test_topology_undo_and_redo_rebind_the_worker(session):
    if not session.adapter.caps.edit_history:
        pytest.skip("Undo history is provided by the workspace composition")
    assert session.submit(cmd.Pause())
    parent = next(n for n in session.nodes if n.source_editable and n.type.value == "link")
    count = len(session.nodes)
    assert session.submit(cmd.AddModelElement(parent.node_id, "geom", "threaded_test_geom"))
    assert len(session.nodes) > count
    assert session.submit(cmd.Play())
    advance(session)
    assert session.submit(cmd.Pause())
    assert session.submit(cmd.Undo())
    assert len(session.nodes) == count
    assert session.submit(cmd.Redo())
    assert len(session.nodes) > count
    assert session.submit(cmd.Play())
    advance(session)
