"""Local clip commands remain independent of physics and training state."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, ReplayControl, RolloutSync
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


def set_replay_playback(self: Session, c: cmd.SetReplayPlayback) -> CommandResult:
    try:
        cast(ReplayControl, self._adapter).set_replay_playback(
            paused=c.paused, speed=c.speed, loop=c.loop
        )
    except ValueError as error:
        return CommandResult.bad(str(error))
    self._frame = self._adapter.frame(FrameNeeds())
    self._paused = self._frame.paused
    return CommandResult.good("")


def seek_replay(self: Session, c: cmd.SeekReplay) -> CommandResult:
    try:
        cast(ReplayControl, self._adapter).seek_replay(c.frame)
    except ValueError as error:
        return CommandResult.bad(str(error))
    self._frame = self._adapter.frame(FrameNeeds())
    self._paused = self._frame.paused
    return CommandResult.good("")


def sync_rollout(self: Session, c: cmd.SyncRollout) -> CommandResult:
    try:
        cast(RolloutSync, self._adapter).sync_rollout(c.world_ids, c.frame_count)
    except ValueError as error:
        return CommandResult.bad(str(error))
    return CommandResult.good("Rollout download requested")
