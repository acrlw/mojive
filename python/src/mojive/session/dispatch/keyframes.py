"""Session commands: keyframes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import (
    KeyframeProperties,
)
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


from ..state import PerturbState


def load_keyframe(self: Session, c: cmd.LoadKeyframe) -> CommandResult:
    caps = self._adapter.caps
    if not caps.keyframes:
        return CommandResult.bad(f"{caps.name} does not expose keyframes")
    if not self._paused:
        return CommandResult.bad("physics is running; pause to load a keyframe")
    i = int(c.keyframe_id)
    slot = self._keyframe_slot(i)
    if slot < 0:
        return CommandResult.bad(f"keyframe {i} is unavailable")
    if not self._adapter.load_keyframe(i):
        return CommandResult.bad(f"failed to load keyframe {i}")
    self._step_counter = 0
    self._sim_time_credit = 0.0
    self._pending_steps = 0
    self._perturb = PerturbState()
    self._active_keyframe = i
    self._state_take_playing = False
    self._state_take_cursor = -1
    self._frame_history_dirty = True
    return CommandResult.good(f"loaded {self._keyframes[slot].name}")


def add_model_keyframe(self: Session, c: cmd.AddModelKeyframe) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support keyframe authoring")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before adding a keyframe")
    name = str(c.name).strip()
    if not name:
        return CommandResult.bad("Keyframe name cannot be empty")
    try:
        keyframe_id = self._adapter.add_model_keyframe(c.model_id, name)
    except Exception as exc:
        return CommandResult.bad(f"Keyframe could not be added: {exc}")
    if keyframe_id < 0:
        return CommandResult.bad(f"Keyframe {name} could not be added")
    self._refresh_structure()
    return CommandResult.good(f"Added keyframe {name}", keyframe_id)


def set_model_keyframe(self: Session, c: cmd.SetModelKeyframe) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support keyframe authoring")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing a keyframe")
    current = self._adapter.keyframe_properties(c.keyframe_id)
    if current is None or current.model_id != int(c.model_id):
        return CommandResult.bad(f"Keyframe {c.keyframe_id} is unavailable")
    name = str(c.name).strip()
    try:
        time_value = float(c.time)
        arrays = tuple(
            np.asarray(values, np.float64).reshape(-1)
            for values in (
                c.qpos,
                c.qvel,
                c.act,
                c.ctrl,
                c.mocap_position,
                c.mocap_quaternion,
            )
        )
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Keyframe values have invalid value types")
    if not name:
        return CommandResult.bad("Keyframe name cannot be empty")
    if not np.isfinite(time_value) or any(not np.all(np.isfinite(values)) for values in arrays):
        return CommandResult.bad("Keyframe values must be finite")
    expected = tuple(
        len(values)
        for values in (
            current.qpos,
            current.qvel,
            current.act,
            current.ctrl,
            current.mocap_position,
            current.mocap_quaternion,
        )
    )
    if tuple(len(values) for values in arrays) != expected:
        return CommandResult.bad("Keyframe array lengths must match the model")
    properties = KeyframeProperties(
        keyframe_id=int(c.keyframe_id),
        model_id=int(c.model_id),
        name=name,
        time=time_value,
        qpos=tuple(float(value) for value in arrays[0]),
        qvel=tuple(float(value) for value in arrays[1]),
        act=tuple(float(value) for value in arrays[2]),
        ctrl=tuple(float(value) for value in arrays[3]),
        mocap_position=tuple(float(value) for value in arrays[4]),
        mocap_quaternion=tuple(float(value) for value in arrays[5]),
    )
    try:
        changed = self._adapter.set_keyframe_properties(properties)
    except Exception as exc:
        return CommandResult.bad(f"Keyframe could not be applied: {exc}")
    if not changed:
        return CommandResult.bad("Keyframe could not be edited")
    self._refresh_structure()
    return CommandResult.good(f"Updated keyframe {name}")


def remove_model_keyframe(self: Session, c: cmd.RemoveModelKeyframe) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support keyframe authoring")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before removing a keyframe")
    if self._adapter.keyframe_properties(c.keyframe_id) is None:
        return CommandResult.bad(f"Keyframe {c.keyframe_id} is unavailable")
    try:
        changed = self._adapter.remove_model_keyframe(c.keyframe_id)
    except Exception as exc:
        return CommandResult.bad(f"Keyframe could not be removed: {exc}")
    if not changed:
        return CommandResult.bad("Keyframe could not be removed")
    self._active_keyframe = -1
    self._refresh_structure()
    return CommandResult.good("Removed keyframe")
