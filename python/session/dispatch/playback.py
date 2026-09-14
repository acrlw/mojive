"""Session commands: playback."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


from ..state import PerturbState, _StateTake


def scene_snapshot(
    self: Session,
    c: cmd.CaptureSceneSnapshot | cmd.RestoreSceneSnapshot | cmd.RemoveSceneSnapshot,
) -> CommandResult:
    return self._snapshot_command(c)


def start_state_take_recording(self: Session, c: cmd.StartStateTakeRecording) -> CommandResult:
    caps = self._adapter.caps
    if not caps.simulation or not caps.state_snapshots:
        return CommandResult.bad(f"{caps.name} cannot record simulation-state takes")
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before starting another take")
    state = self._adapter.capture_state()
    if state is None:
        return CommandResult.bad("physics backend could not capture the current state")
    previous = self._take
    index = None
    if not c.new_take and previous.frames:
        try:
            index = operator.index(previous.cursor if c.frame_index is None else c.frame_index)
        except TypeError:
            return CommandResult.bad("Recording requires a recorded frame index")
        if not 0 <= index < len(previous.frames):
            return CommandResult.bad("Move the playhead to a recorded frame before recording")
        if not self._adapter.restore_state(previous.frames[index].state):
            return CommandResult.bad("Recorded take is incompatible with the current scene")
    else:
        # Validate the first sample before replacing the active take or starting physics.
        self._take = _StateTake()
        try:
            captured = self._append_state_take_frame(state)
            candidate = self._take
        finally:
            self._take = previous
        if not captured:
            return CommandResult.bad(self._state_take_append_error)
    if self._paused and not self._adapter.set_paused(False):
        if index is not None and not self._adapter.restore_state(state):
            return CommandResult.bad(
                "Physics backend rejected play and failed to restore the previous state"
            )
        return CommandResult.bad("physics backend rejected play before recording")
    if index is not None:
        self._delete_state_take_frames(index + 1, len(previous.frames) - 1)
        self._take.cursor = index
        self._step_counter = self._take.frames[index].step
    else:
        if c.new_take or previous.info.take_id < 0:
            self._new_state_take()
        candidate.info = self._take.info
        self._take = candidate
        self._state_takes[candidate.info.take_id] = candidate
    self._clear_frame_history()
    self._paused = False
    self._sim_time_credit = 0.0
    self._pending_steps = 0
    self._perturb = PerturbState()
    self._active_keyframe = -1
    self._state_take_playing = False
    self._state_take_elapsed = 0.0
    self._state_take_recording = True
    self._playback_source = "simulation"
    self._state_take_recording_start_frame = index if index is not None else 0
    return CommandResult.good("Recording simulation take")


def create_state_take(self: Session, c: cmd.CreateStateTake) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before creating a take")
    take = self._new_state_take(c.name)
    return CommandResult.good(f"Created {take.info.name}", take.info.take_id)


def select_state_take(self: Session, c: cmd.SelectStateTake) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before switching takes")
    take = self._state_takes.get(c.take_id)
    if take is None:
        return CommandResult.bad("Take is no longer available")
    if take is self._take:
        return CommandResult.good(f"Selected {take.info.name}")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("Physics backend rejected pause before switching takes")
    self._paused = True
    previous = self._take
    self._take = take
    if take.frames and not self._restore_state_take_frame(max(0, take.cursor)):
        self._take = previous
        return CommandResult.bad("Recorded take is incompatible with the current scene")
    self._state_take_playing = False
    self._state_take_elapsed = 0.0
    self._frame_history_dirty = True
    return CommandResult.good(f"Selected {take.info.name}")


def remove_state_take(self: Session, c: cmd.RemoveStateTake) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before removing a take")
    if c.take_id not in self._state_takes:
        return CommandResult.bad("Take is no longer available")
    if c.take_id == self.active_state_take_id:
        replacement = next(
            (take_id for take_id in reversed(self._state_takes) if take_id != c.take_id), None
        )
        if replacement is not None:
            result = select_state_take(self, cmd.SelectStateTake(replacement))
            if not result.ok:
                return result
    del self._state_takes[c.take_id]
    if c.take_id == self.active_state_take_id:
        self._take = _StateTake()
        self._state_take_playing = False
        self._state_take_elapsed = 0.0
    return CommandResult.good("Removed take")


def delete_state_take_frames(self: Session, c: cmd.DeleteStateTakeFrames) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before deleting take frames")
    try:
        first, last = operator.index(c.first_frame), operator.index(c.last_frame)
    except TypeError:
        return CommandResult.bad("Selection endpoints must be recorded frame indices")
    if not 0 <= first <= last < len(self._take.frames):
        return CommandResult.bad("Selected take frames are no longer available")
    self._delete_state_take_frames(first, last)
    return CommandResult.good(f"Deleted {last - first + 1} take frame(s)")


def stop_state_take_recording(self: Session, c: cmd.StopStateTakeRecording) -> CommandResult:
    if not self._state_take_recording:
        return CommandResult.good("State-take recording is already stopped")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause after recording")
    self._paused = True
    self._sim_time_credit = 0.0
    captured = self._append_state_take_frame()
    self._state_take_recording = False
    if not captured:
        return CommandResult.bad(self._state_take_append_error)
    return CommandResult.good(f"Recorded {len(self._take.frames)} frame(s)")


def play_state_take(self: Session, c: cmd.PlayStateTake) -> CommandResult:
    if not self._take.frames:
        return CommandResult.bad("Record a simulation take before replaying it")
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before replaying the take")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause before take replay")
    self._paused = True
    play_range = self._take.play_range if c.loop else None
    first, last = play_range or (0, len(self._take.frames) - 1)
    pause_at_end = play_range is not None or (
        self._state_take_pause_at_end if c.pause_at_end is None else c.pause_at_end
    )
    previous_index = index = self._take.cursor
    if play_range is not None:
        index = min(last, max(first, index))
    elif index < first or index > last or (index == last and pause_at_end):
        index = first
    if not self._restore_state_take_frame(index):
        return CommandResult.bad("Recorded take is incompatible with the current scene")
    if index != previous_index or c.loop != self._state_take_use_range:
        self._state_take_elapsed = 0.0
    if play_range is not None:
        self._state_take_elapsed = min(
            self._state_take_elapsed, self._take.times[last] - self._take.times[index]
        )
    self._state_take_use_range = c.loop
    self._state_take_end_override = c.pause_at_end
    self._playback_source = "take"
    self._state_take_playing = len(self._take.frames) > 1 or not pause_at_end
    return CommandResult.good("Replaying simulation take")


def set_state_take_pause_at_end(self: Session, c: cmd.SetStateTakePauseAtEnd) -> CommandResult:
    self._state_take_pause_at_end = bool(c.enabled)
    return CommandResult.good("")


def pause_state_take(self: Session, c: cmd.PauseStateTake) -> CommandResult:
    self._state_take_playing = False
    return CommandResult.good("Take replay paused")


def seek_state_take(self: Session, c: cmd.SeekStateTake) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before seeking the take")
    if not self._take.frames:
        return CommandResult.bad("No recorded take is available")
    if not self._paused and not self._adapter.set_paused(True):
        return CommandResult.bad("physics backend rejected pause before seeking the take")
    self._paused = True
    self._state_take_playing = False
    self._state_take_elapsed = 0.0
    index = min(len(self._take.frames) - 1, max(0, int(c.frame_index)))
    if not self._restore_state_take_frame(index):
        return CommandResult.bad("Recorded take is incompatible with the current scene")
    return CommandResult.good(f"Take frame {index + 1}/{len(self._take.frames)}")


def set_state_take_loop(self: Session, c: cmd.SetStateTakeLoop) -> CommandResult:
    result = set_state_take_range(self, cmd.SetStateTakeRange(c.first_frame, c.last_frame))
    if result.ok:
        self._take.loop_enabled = self._take.play_range is not None
    return result


def set_state_take_range(self: Session, c: cmd.SetStateTakeRange) -> CommandResult:
    if c.first_frame is None and c.last_frame is None:
        self._take.play_range = None
        return CommandResult.good("Cleared take playback range")
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before selecting a playback range")
    try:
        first, last = operator.index(c.first_frame), operator.index(c.last_frame)
    except TypeError:
        return CommandResult.bad("Range endpoints must both be recorded frame indices")
    if not 0 <= first < last < len(self._take.frames):
        return CommandResult.bad("Select at least two frames within the recorded take")
    self._take.play_range = first, last
    return CommandResult.good(f"Selected take frames {first + 1}–{last + 1}")


def set_state_take_loop_enabled(self: Session, c: cmd.SetStateTakeLoopEnabled) -> CommandResult:
    if self._state_take_recording:
        return CommandResult.bad("Stop recording before changing repetition")
    if c.enabled and len(self._take.frames) < 2:
        return CommandResult.bad("Record at least two frames before enabling repetition")
    self._take.loop_enabled = bool(c.enabled)
    return CommandResult.good("")


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
    self._playback_source = "simulation"
    self._state_take_elapsed = 0.0
    self._take.cursor = -1
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
    self._take.cursor = -1
    self._pending_steps += count
    self._playback_source = "simulation"
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
    self._playback_source = "simulation"
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
    self._take.cursor = -1
    self._playback_source = "simulation"
    self._state_take_elapsed = 0.0
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
