"""Numerical failures must not corrupt editor timelines or leave physics running."""

import time
from dataclasses import replace

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


@pytest.mark.parametrize("threaded", (False, True))
@pytest.mark.parametrize("mode", ("record", "live", "step"))
@pytest.mark.parametrize("field", ("qpos", "qvel", "qfrc_applied", "ctrl", "time"))
def test_instability_preserves_takes_and_allows_recording_again(threaded, mode, field):
    primary = MuJoCoAdapter(resolve("joint_types" if field == "ctrl" else "joint_gizmo"))
    session = Session(WorkspaceAdapter(primary))
    try:
        assert session.submit(cmd.StartStateTakeRecording())
        for _ in range(5):
            session.tick(FrameNeeds.none(), wall_dt=0.1)
        assert session.submit(cmd.StopStateTakeRecording())
        take_id = session.active_state_take_id
        times = tuple(session.state_take_times)
        model_id = session.scene_models[0].model_id
        key = session.submit(cmd.AddModelKeyframe(model_id, "before_failure"))
        assert key
        if mode == "record":
            assert session.submit(cmd.StartStateTakeRecording(new_take=False))
        elif mode == "live":
            assert session.submit(cmd.Play())
        else:
            assert session.submit(cmd.Step(5))
        # The same native instability as a divergent joint perturbation, without
        # depending on mouse trajectory, solver timing, or machine speed.
        if field == "time":
            primary.data.time = float("nan")
        else:
            getattr(primary.data, field)[0] = float("nan")
        session.set_threaded_physics(threaded)
        deadline = time.monotonic() + 3
        while True:
            session.tick(FrameNeeds.none(), wall_dt=0.02)
            if "unstable" in session.last_message.lower():
                break
            if time.monotonic() >= deadline:
                pytest.fail("Numerical failure was not reported to the editor")
            time.sleep(0.002)
        assert session.paused
        assert not session.state_take_recording and not session.state_take_playing
        assert tuple(session.state_take_times) == times
        assert session.active_state_take_id == take_id
        assert session.state_take_cursor == -1
        assert session.frame.time == 0
        assert np.isfinite(primary.data.qpos).all()
        assert np.isfinite(primary.data.qvel).all()
        assert session._pending_steps == 0
        assert session._sim_time_credit == 0
        assert session.submit(cmd.LoadKeyframe(key.entity_id))
        assert session.submit(cmd.SeekStateTake(2))
        assert session.submit(cmd.PlayStateTake())
        session.tick(FrameNeeds.none(), wall_dt=0.1)
        assert session.submit(cmd.PauseStateTake())
        assert session.submit(cmd.StartStateTakeRecording(new_take=False, frame_index=2))
        session.tick(FrameNeeds.none(), wall_dt=0.05)
        assert session.submit(cmd.StopStateTakeRecording())
        assert session.state_take_times[:3] == list(times[:3])
        assert np.all(np.diff(session.state_take_times) > 0)
        assert session.submit(cmd.CreateStateTake())
        assert session.submit(cmd.StartStateTakeRecording(new_take=False))
        session.tick(FrameNeeds.none(), wall_dt=0.02)
        assert session.submit(cmd.StopStateTakeRecording())
        assert session.submit(cmd.RemoveStateTake(take_id))
        assert len(session.state_takes) == 1
    finally:
        session.release()


@pytest.mark.parametrize("auto_reset", (False, True))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), 1e30))
def test_repeated_instability_at_time_zero_recovers_with_or_without_native_auto_reset(
    auto_reset, value
):
    adapter = MuJoCoAdapter(resolve("joint_gizmo"))
    session = Session(adapter)
    try:
        if not auto_reset:
            adapter.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
        for _ in range(2):
            assert session.submit(cmd.StartStateTakeRecording())
            adapter.data.qfrc_applied[0] = value
            session.tick(FrameNeeds.none())
            assert session.paused and not session.state_take_recording
            assert session.frame.time == 0
            assert session.state_take_times == [0]
            assert "QACC" in session.last_message
            assert np.isfinite(adapter.data.qpos).all()
        assert len(session.state_takes) == 2
    finally:
        session.release()


def test_command_fence_recovers_worker_failure_before_saving_the_last_recorded_frame():
    adapter = MuJoCoAdapter(resolve("joint_gizmo"))
    session = Session(adapter)
    try:
        assert session.submit(cmd.StartStateTakeRecording())
        session.tick(FrameNeeds.none(), wall_dt=0.1)
        times = tuple(session.state_take_times)
        adapter.data.qfrc_applied[0] = float("nan")
        session.set_threaded_physics(True)
        session.tick(FrameNeeds.none(), wall_dt=0.01)
        deadline = time.monotonic() + 2
        while session._simulation_driver._error is None:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        result = session.submit(cmd.StopStateTakeRecording())
        assert not result and "QACC" in result.message
        assert session.paused and not session.state_take_recording
        assert tuple(session.state_take_times) == times
        session.tick(FrameNeeds.none())
        assert session.frame.time == 0
        assert session.submit(cmd.SeekStateTake(0))
    finally:
        session.release()


def test_invalid_paused_state_is_not_saved_as_a_model_key_or_restored():
    adapter = MuJoCoAdapter(resolve("joint_gizmo"))
    session = Session(adapter)
    try:
        assert session.submit(cmd.Pause())
        valid = adapter.capture_state()
        invalid = replace(valid, time=float("nan"))
        assert not adapter.restore_state(invalid)
        assert adapter.data.time == valid.time
        adapter.data.qpos[0] = float("nan")
        result = session.submit(cmd.AddModelKeyframe(session.scene_models[0].model_id, "invalid"))
        assert not result and "finite" in result.message
        assert not session.keyframes
        assert session.submit(cmd.Reset())
        assert session.submit(cmd.AddModelKeyframe(session.scene_models[0].model_id, "valid"))
    finally:
        session.release()


def test_native_auto_reset_is_detected_when_warning_count_repeats_and_batch_time_still_increases():
    adapter = MuJoCoAdapter(resolve("joint_gizmo"))
    session = Session(adapter)
    try:
        adapter.data.time = 0.1
        adapter.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
        assert session.submit(cmd.StartStateTakeRecording())
        adapter.data.qfrc_applied[0] = float("nan")
        session.tick(FrameNeeds.none(), wall_dt=0.2)
        assert session.paused and not session.state_take_recording
        assert "instead of" in session.last_message
        assert session.state_take_times == [0.1]
        assert session.frame.time == 0
    finally:
        session.release()
