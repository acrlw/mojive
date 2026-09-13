"""Session: playback."""

from __future__ import annotations

import bisect
from copy import deepcopy

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import (
    FrameNeeds,
    PhysicsState,
    SceneFrame,
)
from mojive.commands import CommandResult

from .state import (
    FRAME_HISTORY_BYTE_LIMIT,
    FRAME_HISTORY_LIMIT,
    FRAME_HISTORY_TRIM_RATIO,
    STATE_TAKE_BYTE_LIMIT,
    STATE_TAKE_FRAME_LIMIT,
    PerturbState,
    SceneSnapshotInfo,
    _SceneSnapshot,
    _StateTakeFrame,
)


class _Playback:
    """Private playback methods of Session; state belongs to its owner."""

    @staticmethod
    def _physics_state_signature(state: PhysicsState) -> tuple[tuple[int, ...], ...]:
        return tuple(
            np.shape(values)
            for values in (
                state.qpos,
                state.qvel,
                state.act,
                state.ctrl,
                state.mocap_pos,
                state.mocap_quat,
            )
        )

    @property
    def scene_snapshots(self) -> tuple[SceneSnapshotInfo, ...]:
        """Return transient state samples independently of model keys and recorded takes."""
        return self._scene_snapshot_info

    def _clear_scene_snapshots(self) -> None:
        self._scene_snapshots.clear()
        self._scene_snapshot_info = ()
        self._scene_snapshot_bytes = 0

    def _snapshot_command(self, command) -> CommandResult:
        if isinstance(command, cmd.CaptureSceneSnapshot):
            # submit() has fenced the simulation owner before copying this state.
            state = self._adapter.capture_state()
            if state is None:
                return CommandResult.bad("Physics backend could not capture a scene snapshot")
            size = self._physics_state_bytes(state)
            if (
                len(self._scene_snapshots) >= 1024
                or self._scene_snapshot_bytes + size > 256 * 1024 * 1024
            ):
                return CommandResult.bad("Scene snapshot limit reached; remove unused snapshots")
            self._scene_snapshot_serial += 1
            snapshot_id = self._scene_snapshot_serial
            info = SceneSnapshotInfo(
                snapshot_id, command.name.strip() or f"Snapshot {snapshot_id}", float(state.time)
            )
            # Do not retain backend-owned buffers, even for custom adapters.
            frame = _StateTakeFrame(self._step_counter, deepcopy(state))
            self._scene_snapshots[snapshot_id] = _SceneSnapshot(
                info, frame, self._adapter.structure_revision
            )
            self._scene_snapshot_bytes += size
            self._scene_snapshot_info = tuple(item.info for item in self._scene_snapshots.values())
            return CommandResult.good("Captured scene snapshot", snapshot_id)
        snapshot = self._scene_snapshots.get(command.snapshot_id)
        if snapshot is None:
            return CommandResult.bad("Scene snapshot is no longer available")
        if isinstance(command, cmd.RemoveSceneSnapshot):
            self._scene_snapshot_bytes -= self._physics_state_bytes(snapshot.frame.state)
            del self._scene_snapshots[command.snapshot_id]
            self._scene_snapshot_info = tuple(item.info for item in self._scene_snapshots.values())
            return CommandResult.good("Removed scene snapshot")
        if self._state_take_recording or self._state_take_playing or not self._paused:
            return CommandResult.bad("Pause simulation and recording before restoring a snapshot")
        if snapshot.structure_revision != self._adapter.structure_revision:
            return CommandResult.bad("Scene structure changed since this snapshot was captured")
        result = self.restore_physics_state(deepcopy(snapshot.frame.state))
        if result.ok:
            self._step_counter = snapshot.frame.step
        return result

    def _clear_state_take(self) -> None:
        self._state_take.clear()
        self._state_take_times.clear()
        self._state_take_offsets.clear()
        self._state_take_loop = None
        self._state_take_cursor = -1
        self._state_take_recording = False
        self._state_take_playing = False
        self._state_take_elapsed = 0.0
        self._state_take_signature = None
        self._state_take_bytes = 0
        self._state_take_append_error = ""
        self._state_take_limit_reached = False

    @staticmethod
    def _physics_state_bytes(state: PhysicsState) -> int:
        return 8 + sum(
            np.asarray(values).nbytes
            for values in (
                state.qpos,
                state.qvel,
                state.act,
                state.ctrl,
                state.mocap_pos,
                state.mocap_quat,
            )
        )

    def _clear_frame_history(self) -> None:
        self._frame_history.clear()
        self._frame_history_cursor = -1
        self._frame_history_bytes = 0
        self._frame_history_signature = None
        self._frame_history_dirty = False

    def _append_frame_history(self, state: PhysicsState | None = None) -> bool:
        """Append one displayed simulation state and discard an abandoned future."""

        if state is None:
            state = self._adapter.capture_state()
        if state is None:
            return False
        signature = self._physics_state_signature(state)
        if self._frame_history_signature not in (None, signature):
            self._clear_frame_history()
        self._frame_history_signature = signature
        if self._frame_history_cursor + 1 < len(self._frame_history):
            future = self._frame_history[self._frame_history_cursor + 1 :]
            self._frame_history_bytes -= sum(
                self._physics_state_bytes(frame.state) for frame in future
            )
            del self._frame_history[self._frame_history_cursor + 1 :]
        self._frame_history.append(_StateTakeFrame(self._step_counter, state))
        self._frame_history_cursor = len(self._frame_history) - 1
        self._frame_history_bytes += self._physics_state_bytes(state)
        if len(self._frame_history) > 2 and (
            len(self._frame_history) > FRAME_HISTORY_LIMIT
            or self._frame_history_bytes > FRAME_HISTORY_BYTE_LIMIT
        ):
            target_count = max(2, int(FRAME_HISTORY_LIMIT * FRAME_HISTORY_TRIM_RATIO))
            target_bytes = int(FRAME_HISTORY_BYTE_LIMIT * FRAME_HISTORY_TRIM_RATIO)
            remove_count = 0
            remove_bytes = 0
            while len(self._frame_history) - remove_count > 2 and (
                len(self._frame_history) - remove_count > target_count
                or self._frame_history_bytes - remove_bytes > target_bytes
            ):
                remove_bytes += self._physics_state_bytes(self._frame_history[remove_count].state)
                remove_count += 1
            if remove_count:
                del self._frame_history[:remove_count]
                self._frame_history_bytes -= remove_bytes
                self._frame_history_cursor -= remove_count
        self._frame_history_dirty = False
        return True

    def _restore_previous_frame(self) -> bool:
        if not self._frame_history:
            return False
        index = (
            self._frame_history_cursor
            if self._frame_history_dirty
            else self._frame_history_cursor - 1
        )
        if not 0 <= index < len(self._frame_history):
            return False
        frame = self._frame_history[index]
        if not self._adapter.restore_state(frame.state):
            return False
        self._frame_history_cursor = index
        self._frame_history_dirty = False
        self._step_counter = frame.step
        self._pending_steps = 0
        self._sim_time_credit = 0.0
        self._perturb = PerturbState()
        self._active_keyframe = -1
        self._state_take_playing = False
        self._state_take_cursor = -1
        return True

    def _append_state_take_frame(self, state: PhysicsState | None = None) -> bool:
        if state is None:
            state = self._adapter.capture_state()
        if state is None:
            self._state_take_append_error = "Physics backend could not capture the current state"
            self._state_take_limit_reached = False
            return False
        if self._state_take and self._state_take[-1].step == self._step_counter:
            return True
        signature = self._physics_state_signature(state)
        if self._state_take_signature is not None and signature != self._state_take_signature:
            self._state_take_append_error = (
                "State-take recording stopped because the scene state changed"
            )
            self._state_take_limit_reached = False
            return False
        frame_bytes = self._physics_state_bytes(state)
        if (
            len(self._state_take) >= STATE_TAKE_FRAME_LIMIT
            or self._state_take_bytes + frame_bytes > STATE_TAKE_BYTE_LIMIT
        ):
            self._state_take_append_error = (
                "State-take recording stopped after reaching its recording limit"
            )
            self._state_take_limit_reached = True
            return False
        self._state_take_signature = signature
        self._state_take.append(_StateTakeFrame(self._step_counter, state))
        self._state_take_times.append(float(state.time))
        offset = 0.0
        if self._state_take_offsets:
            duration = self._state_take_times[-1] - self._state_take_times[-2]
            if not np.isfinite(duration) or duration <= 1e-9:
                duration = max(self._adapter.timestep(), 1.0 / 60.0)
            offset = self._state_take_offsets[-1] + duration
        self._state_take_offsets.append(offset)
        self._state_take_cursor = len(self._state_take) - 1
        self._state_take_bytes += frame_bytes
        self._state_take_append_error = ""
        self._state_take_limit_reached = False
        return True

    def _restore_state_take_frame(self, frame_index: int) -> bool:
        index = int(frame_index)
        if not 0 <= index < len(self._state_take):
            return False
        frame = self._state_take[index]
        if not self._adapter.restore_state(frame.state):
            return False
        self._state_take_cursor = index
        self._step_counter = frame.step
        self._pending_steps = 0
        self._sim_time_credit = 0.0
        self._perturb = PerturbState()
        self._active_keyframe = -1
        return True

    def _advance_state_take(self, wall_dt: float | None) -> None:
        if not self._state_take_playing or not self._state_take:
            return
        dt = self._adapter.timestep() if wall_dt is None else max(0.0, float(wall_dt))
        offsets = self._state_take_offsets
        loop = self._state_take_loop if self._state_take_use_loop else None
        first, last = loop or (0, len(self._state_take) - 1)
        cursor = self._state_take_cursor
        position = (
            offsets[cursor] + self._state_take_elapsed
            if first <= cursor <= last
            else offsets[first]
        ) + dt * self._speed
        if loop is not None:
            # Include the last selected frame for one recorded interval, then
            # wrap excess time in one operation even after a long display stall.
            tail = min(last + 1, len(offsets) - 1)
            interval = offsets[tail] - offsets[tail - 1]
            duration = offsets[last] - offsets[first] + interval
            if position + 1e-12 >= offsets[first] + duration:
                position = offsets[first] + max(0.0, position - offsets[first]) % duration
                if offsets[first] + duration - position < 1e-12:
                    position = offsets[first]
        index = min(last, max(first, bisect.bisect_right(offsets, position + 1e-12) - 1))
        self._state_take_elapsed = max(0.0, position - offsets[index])
        # Only the final displayed sample needs a physics restore/forward pass.
        if index != cursor and not self._restore_state_take_frame(index):
            self._state_take_playing = False
            self._publish_message(
                "Recorded take is incompatible with the current scene",
                level="error",
                duration=10.0,
            )
            return
        if loop is None and index >= last:
            self._state_take_playing = False
            self._state_take_elapsed = 0.0

    def restore_physics_state(
        self, state: PhysicsState, *, active_keyframe: int = -1
    ) -> CommandResult:
        """Restore a complete physics state while the session is paused."""
        if not self._paused:
            return CommandResult.bad("physics is running; pause to restore a scene snapshot")
        if not self._adapter.restore_state(state):
            return CommandResult.bad("scene snapshot state is incompatible with this model")
        keyframe_id = int(active_keyframe)
        self._active_keyframe = (
            keyframe_id
            if keyframe_id == -1
            or any(keyframe.keyframe_id == keyframe_id for keyframe in self._keyframes)
            else -1
        )
        self._pending_steps = 0
        self._sim_time_credit = 0.0
        self._perturb = PerturbState()
        self._state_take_recording = False
        self._state_take_playing = False
        self._state_take_cursor = -1
        self._frame_history_dirty = True
        return CommandResult.good("Scene state restored")

    def set_threaded_physics(self, enabled: bool) -> bool:
        """Choose concurrent real-time stepping; explicit ticks remain synchronous."""
        if not enabled:
            if self._simulation_driver is not None:
                self._step_counter += self._simulation_driver.suspend()
                self._simulation_driver.close()
                self._simulation_driver = None
            return False
        if self._simulation_driver is None and not self._adapter.caps.external_clock:
            factory = getattr(self._adapter, "create_simulation_driver", None)
            self._simulation_driver = factory() if factory is not None else None
        return self._simulation_driver is not None

    def _resume_physics(self, *, reset_clock: bool = True) -> None:
        if (
            self._simulation_driver is not None
            and not self._paused
            and not self._state_take_playing
        ):
            try:
                self._simulation_driver.start(self._speed, reset_clock=reset_clock)
            except Exception as exc:
                self._physics_failed(exc)

    def _physics_failed(self, error: Exception) -> None:
        self.set_threaded_physics(False)
        self._paused = True
        self._state_take_recording = False
        self._adapter.set_paused(True)
        self.report_message(f"Physics worker stopped: {error}", level="error")

    def tick(self, needs: FrameNeeds, wall_dt: float | None = None) -> SceneFrame:
        """Advance simulation time and obtain one composed dynamic frame.

        Args:
            needs: Optional dynamic arrays required by current consumers.
            wall_dt: Elapsed wall time used for real-time simulation scheduling.
        """
        step_before = self._step_counter
        concurrent = self._simulation_driver is not None and wall_dt is not None
        if self._simulation_driver is not None:
            try:
                self._step_counter += self._simulation_driver.poll()
                if concurrent and not self._paused and not self._state_take_playing:
                    self._resume_physics()
                else:
                    self._step_counter += self._simulation_driver.suspend()
            except Exception as exc:
                self._physics_failed(exc)
        history_enabled = bool(
            self._adapter.caps.simulation
            and self._adapter.caps.state_snapshots
            and not self._state_take_recording
            and not self._state_take_playing
        )
        if history_enabled and not self._frame_history:
            self._append_frame_history()
        frame_step_before = int(self._frame.step)
        frame_time_before = float(self._frame.time)
        if self._state_take_playing:
            self._advance_state_take(wall_dt)
        elif not self._paused and not self._adapter.caps.external_clock and not concurrent:
            timestep = self._adapter.timestep()
            if wall_dt is not None and timestep > 0.0:
                self._sim_time_credit += float(wall_dt) * self._speed
                n = int(self._sim_time_credit / timestep + 1e-9)
                self._sim_time_credit -= n * timestep
            else:
                n = max(1, round(self._speed))
            if n:
                self._adapter.step(n)
                self._step_counter += n
        elif self._pending_steps > 0:
            count = self._pending_steps
            # A failed external step can have an uncertain outcome; consume the
            # request before dispatch so another render tick cannot retry it.
            self._pending_steps = 0
            self._adapter.step(count)
            self._step_counter += count

        prepare_frame = getattr(self._adapter, "prepare_frame", None)
        if prepare_frame is not None:
            prepare_frame(needs)
        if self._adapter.structure_revision != self._adapter_revision:
            self._refresh_structure()

        self._frame = self._adapter.frame(needs)
        if self._model_edit_preview is not None and self._model_edit_preview.geometry is not None:
            self._model_edit_preview.geometry._base_frame = None
        self._sync_equality_state()
        self._compose_lights()
        self._compose_cameras()
        if self._adapter.caps.external_clock:
            self._paused = bool(self._frame.paused)
        else:
            self._frame.paused = self._paused
            self._frame.step = self._step_counter
            self._frame.physics_hz = (
                self._physics_rate.update(self._step_counter, self._frame.time)
                if self._adapter.caps.simulation
                else None
            )
        if (
            self._state_take_recording
            and (not self._state_take or self._state_take[-1].step != self._step_counter)
            and not self._append_state_take_frame()
        ):
            self._state_take_recording = False
            message = self._state_take_append_error or (
                "State-take recording stopped because the scene state changed"
            )
            self._publish_message(
                message,
                level="warning" if self._state_take_limit_reached else "error",
                duration=10.0,
            )
        externally_advanced = self._adapter.caps.external_clock and (
            int(self._frame.step) != frame_step_before
            or abs(float(self._frame.time) - frame_time_before) > 1e-12
        )
        if history_enabled and (
            self._step_counter != step_before or externally_advanced or self._frame_history_dirty
        ):
            self._append_frame_history()
        return self.frame
