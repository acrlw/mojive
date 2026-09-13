"""Session commands: playback."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


from ..state import PerturbState


def scene_snapshot(
    self: Session,
    c: cmd.CaptureSceneSnapshot | cmd.RestoreSceneSnapshot | cmd.RemoveSceneSnapshot,
) -> CommandResult:
    return self._snapshot_command(c)


def start_state_take_recording(self: Session, c: cmd.StartStateTakeRecording) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation or not caps.state_snapshots:
        return CommandResult.bad(f"{caps.name} cannot record simulation-state takes")
    state = self._adapter.capture_state()
    if state is None:
        return CommandResult.bad("physics backend could not capture the current state")
    if self._paused and not self._adapter.set_paused(False):
        return CommandResult.bad("physics backend rejected play before recording")
    self._clear_state_take()
    self._clear_frame_history()
    self._paused = False
    self._sim_time_credit = 0.0
    self._state_take_recording = True
    if not self._append_state_take_frame(state):
        self._state_take_recording = False
        self._adapter.set_paused(True)
        self._paused = True
        return CommandResult.bad(
            self._state_take_append_error or "Physics backend could not start state-take recording"
        )
    return CommandResult.good("Recording simulation take")


def stop_state_take_recording(self: Session, c: cmd.StopStateTakeRecording) -> CommandResult:
    if not self._state_take_recording:
        return CommandResult.good("State-take recording is already stopped")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause after recording")
    self._paused = True
    self._sim_time_credit = 0.0
    self._append_state_take_frame()
    self._state_take_recording = False
    return CommandResult.good(f"Recorded {len(self._state_take)} frame(s)")


def play_state_take(self: Session, c: cmd.PlayStateTake) -> CommandResult:
    if not self._state_take:
        return CommandResult.bad("Record a simulation take before replaying it")
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before replaying the take")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause before take replay")
    self._paused = True
    loop = self._state_take_loop if c.loop else None
    first, last = loop or (0, len(self._state_take) - 1)
    previous_index = index = self._state_take_cursor
    if index < first or index >= last:
        index = first
    if not self._restore_state_take_frame(index):
        return CommandResult.bad("Recorded take is incompatible with the current scene")
    if index != previous_index or c.loop != self._state_take_use_loop:
        self._state_take_elapsed = 0.0
    self._state_take_use_loop = c.loop
    self._state_take_playing = len(self._state_take) > 1
    return CommandResult.good("Replaying simulation take")


def pause_state_take(self: Session, c: cmd.PauseStateTake) -> CommandResult:
    self._state_take_playing = False
    return CommandResult.good("Take replay paused")


def seek_state_take(self: Session, c: cmd.SeekStateTake) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before seeking the take")
    if not self._state_take:
        return CommandResult.bad("No recorded take is available")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause before seeking the take")
    self._paused = True
    self._state_take_playing = False
    self._state_take_elapsed = 0.0
    index = min(len(self._state_take) - 1, max(0, int(c.frame_index)))
    if not self._restore_state_take_frame(index):
        return CommandResult.bad("Recorded take is incompatible with the current scene")
    return CommandResult.good(f"Take frame {index + 1}/{len(self._state_take)}")


def set_state_take_loop(self: Session, c: cmd.SetStateTakeLoop) -> CommandResult:
    if c.first_frame is None and c.last_frame is None:
        self._state_take_loop = None
        return CommandResult.good("Cleared take loop range")
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before selecting a loop range")
    try:
        first, last = operator.index(c.first_frame), operator.index(c.last_frame)
    except TypeError:
        return CommandResult.bad("Loop endpoints must both be recorded frame indices")
    if not 0 <= first < last < len(self._state_take):
        return CommandResult.bad("Select at least two frames within the recorded take")
    self._state_take_loop = first, last
    return CommandResult.good(f"Looping take frames {first + 1}–{last + 1}")


def clear_state_take(self: Session, c: cmd.ClearStateTake) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before clearing the take")
    self._clear_state_take()
    return CommandResult.good("Cleared recorded take")


def pause(self: Session, c: cmd.Pause) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation:
        return CommandResult.bad(f"{caps.name} has no simulation to pause")
    if self._state_take_playing:
        self._state_take_playing = False
        self._state_take_elapsed = 0.0
        return CommandResult.good("Take replay paused")
    if self._paused:
        return CommandResult.good("Simulation is already paused")
    if not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause")
    self._paused = True
    self._sim_time_credit = 0.0
    self._state_take_recording = False
    return CommandResult.good("Simulation paused")


def play(self: Session, c: cmd.Play) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation:
        return CommandResult.bad(f"{caps.name} has no simulation to resume")
    if not self._paused:
        return CommandResult.good("Simulation is already running")
    if not self._adapter.set_paused(False):
        return CommandResult.bad("physics backend rejected play")
    self._paused = False
    self._sim_time_credit = 0.0
    self._perturb = PerturbState()
    self._state_take_playing = False
    self._state_take_cursor = -1
    return CommandResult.good("Simulation resumed")


def step(self: Session, c: cmd.Step) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation:
        return CommandResult.bad(f"{caps.name} has no simulation to step")
    if not self._paused:
        return CommandResult.bad("Pause the simulation before stepping")
    count = int(c.count)
    if count <= 0:
        return CommandResult.bad("step count must be positive")
    self._state_take_playing = False
    self._state_take_cursor = -1
    self._pending_steps += count
    return CommandResult.good(f"Stepped {count} frame(s)")


def step_back(self: Session, c: cmd.StepBack) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation or not caps.state_snapshots:
        return CommandResult.bad(f"{caps.name} has no frame history")
    if not self._paused:
        return CommandResult.bad("Pause the simulation before stepping backward")
    if self._state_take_recording or self._state_take_playing:
        return CommandResult.bad("Stop take recording or replay before stepping backward")
    if not self._restore_previous_frame():
        return CommandResult.bad("No previous frame is available")
    return CommandResult.good("Restored previous frame")


def reset(self: Session, c: cmd.Reset) -> CommandResult:
    caps = self._adapter.caps
    try:
        self._adapter.reset()
    except Exception as exc:
        return CommandResult.bad(str(exc))
    self._step_counter = 0
    self._sim_time_credit = 0.0
    self._perturb = PerturbState()
    self._active_keyframe = -1
    self._state_take_recording = False
    self._state_take_playing = False
    self._state_take_cursor = -1
    self._frame_history_dirty = True
    self._equality_constraints = (
        self._adapter.equality_constraints() if caps.equality_constraints else []
    )

    return CommandResult.good("Scene reset")


def set_speed(self: Session, c: cmd.SetSpeed) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation:
        return CommandResult.bad(f"{caps.name} has no simulation speed")
    factor = float(c.factor)
    if not np.isfinite(factor) or factor <= 0.0:
        return CommandResult.bad("simulation speed must be finite and positive")
    self._speed = max(0.05, factor)
    return CommandResult.good(f"Speed ×{self._speed:g}")


def set_camera(self: Session, c: cmd.SetCamera) -> CommandResult:
    self._camera = c.camera
    return CommandResult.good("")
