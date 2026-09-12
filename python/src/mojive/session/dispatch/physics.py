"""Session commands: physics."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mojive import commands as cmd
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


from ..state import PerturbState


def set_equality_enabled(self: Session, c: cmd.SetEqualityEnabled) -> CommandResult:
    caps = self._adapter.caps
    if not caps.equality_constraints:
        return CommandResult.bad(f"{caps.name} does not expose equality constraints")
    i = int(c.constraint_id)
    slot = self._equality_slot(i)
    if slot < 0:
        return CommandResult.bad(f"equality constraint {i} is unavailable")
    if not self._adapter.set_equality_enabled(i, c.enabled):
        return CommandResult.bad(f"equality constraint {i} update failed")
    self._equality_constraints[slot] = replace(
        self._equality_constraints[slot], enabled=bool(c.enabled)
    )
    return CommandResult.good("")


def set_ctrl_vector(self: Session, c: cmd.SetCtrlVector) -> CommandResult:
    ok = self._adapter.set_ctrl_vector(c.values)
    if ok:
        self._frame_history_dirty = True
    return CommandResult.good("") if ok else CommandResult.bad("Control vector update failed")


def set_ctrl(self: Session, c: cmd.SetCtrl) -> CommandResult:
    ok = self._adapter.set_ctrl(c.index, c.value)
    if ok:
        self._frame_history_dirty = True
    return CommandResult.good("") if ok else CommandResult.bad(f"Actuator {c.index} update failed")


def perturb(self: Session, c: cmd.Perturb) -> CommandResult:
    caps = self._adapter.caps
    if not caps.perturb:
        return CommandResult.bad(f"{caps.name} does not support perturbation")
    ok = self._adapter.apply_perturb(c.node_id, c.target_position, c.target_rotation, c.mode)
    if ok:
        self._frame_history_dirty = True
    return CommandResult.good("") if ok else CommandResult.bad("Perturbation failed")


def clear_perturb(self: Session, c: cmd.ClearPerturb) -> CommandResult:
    self._adapter.clear_perturb()
    self._perturb = PerturbState()
    return CommandResult.good("")
