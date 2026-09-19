"""Manual background rollout synchronization feeding an independent local clip."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import AdapterCaps, FrameNeeds, RolloutSyncInfo
from .joint_replay import JointReplayAdapter


class RolloutReplayAdapter(JointReplayAdapter):
    """Download only requested windows; pause on complete, atomically installed clips."""

    caps = AdapterCaps(
        name="rollout_replay",
        external_clock=True,
        clock_control=False,
        features=(("world.selection", 1), ("replay.control", 1), ("replay.sync", 1)),
    )

    def __init__(
        self,
        url: str,
        worlds: int | None = None,
        *,
        world_ids=None,
        max_worlds=64,
        window_frames=128,
        paused=True,
    ):
        from mojive.remote.rollout import RolloutClient

        self._client = RolloutClient(url)
        info = self._client.info()
        self._max_frames = min(512, info["max_frames"])
        if type(window_frames) is not int or not 2 <= window_frames <= self._max_frames:
            raise ValueError(f"Window frames must be between 2 and {self._max_frames}")
        if (
            (worlds is not None and (type(worlds) is not int or worlds < 1))
            or type(max_worlds) is not int
            or max_worlds < 1
        ):
            raise ValueError("World count and preview limit must be positive integers")
        limit = min(max_worlds, info["max_worlds"])
        available = info.get("world_ids", range(info["total_worlds"]))
        count = min(4, limit) if worlds is None else worlds
        ids = (
            tuple(available[: min(count, len(available))])
            if world_ids is None
            else tuple(world_ids)
        )
        if (
            not 1 <= len(ids) <= limit
            or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < info["total_worlds"] for i in ids)
        ):
            raise ValueError("Invalid initial world selection or preview limit")
        window = self._client.fetch(ids, window_frames)
        if window is None or window.metadata["model_sha256"] != info["model_sha256"]:
            raise ValueError("Rollout model changed while opening the source")
        temporary = tempfile.TemporaryDirectory(prefix="mojive-rollout-")
        self._future = None
        try:
            model = Path(temporary.name) / "model.mjb"
            model.write_bytes(self._client.model(info["model_sha256"]))
            self._initialize(
                window.metadata,
                window.qpos,
                model,
                worlds=worlds,
                world_ids=ids,
                max_worlds=limit,
                realtime=True,
                paused=paused,
            )
        finally:
            temporary.cleanup()
        self._window_frames = window_frames
        self._rollout_revision = window.metadata["revision"]
        self._start_step = window.metadata["start_step"]
        self._received_bytes = window.received_bytes
        self._sync_error = ""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mojive-rollout")
        self._closed = False

    def rollout_sync_info(self) -> RolloutSyncInfo:
        """Read status without polling the server or installing pending data."""
        return RolloutSyncInfo(
            self._future is not None,
            self._rollout_revision,
            self._start_step,
            self._window_frames,
            self._max_frames,
            self._received_bytes,
            self._sync_error,
        )

    def sync_rollout(self, world_ids=None, frame_count=None) -> None:
        """Request exactly one window; preserve current playback until it is complete."""
        if self._closed:
            raise ValueError("Rollout replay is closed")
        if self._future is not None:
            raise ValueError("A rollout sync is already in progress")
        ids = self._validate_ids(self._world_ids if world_ids is None else world_ids)
        frames = self._window_frames if frame_count is None else frame_count
        if type(frames) is not int or not 2 <= frames <= self._max_frames:
            raise ValueError(f"Window frames must be between 2 and {self._max_frames}")
        after = (
            self._rollout_revision
            if ids == self._available_ids and frames == self._window_frames
            else ""
        )
        self._sync_error = ""
        self._requested_frames = frames
        self._future = self._executor.submit(self._client.fetch, ids, frames, after=after)

    def set_world_selection(self, world_ids) -> bool:
        """Keep pending remote selection from overwriting a newer local selection."""
        if self._future is not None:
            raise ValueError("Wait for the pending rollout sync before changing worlds")
        return super().set_world_selection(world_ids)

    def prepare_frame(self, needs: FrameNeeds) -> None:
        """Install complete downloads on the owner thread, before structure inspection."""
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        try:
            window = future.result()
            if window is not None:
                self._install_window(window)
            self._window_frames = self._requested_frames
        except Exception as error:
            # Retain the complete previous clip and the concrete failure for UI/RPC.
            self._sync_error = f"{type(error).__name__}: {error}"

    def _install_window(self, window):
        metadata, qpos = window.metadata, window.qpos
        self._validate_recording(metadata, qpos)
        for key in (
            "model_sha256",
            "mujoco_version",
            "joint_names",
            "total_worlds",
        ):
            if metadata[key] != self.metadata[key]:
                raise ValueError(
                    f"Rollout {key} changed; reopen the source to load a different model"
                )
        if float(metadata.get("display_spacing", 7.0)) != self._spacing:
            raise ValueError("Rollout display spacing changed; reopen the source")
        if qpos.shape[2] != self.model.nq:
            raise ValueError("Rollout joint layout does not match model")
        ids = self._validate_ids(metadata["world_ids"])
        columns = {world: column for column, world in enumerate(ids)}
        changed = ids != self._world_ids
        if changed:
            worlds, positions, rotations, sample = self._make_worlds(ids)
        else:
            worlds, positions, rotations, sample = (
                self.worlds,
                self.positions,
                self.rotations,
                self._sample,
            )
        self._evaluate(
            worlds, ids, sample, positions, rotations, 0, 0.0, qpos=qpos, columns=columns
        )
        self.metadata, self.qpos, self.hz = metadata, qpos, float(metadata["hz"])
        self._available_ids, self._columns, self._world_ids = ids, columns, ids
        self.worlds, self.positions, self.rotations, self._sample = (
            worlds,
            positions,
            rotations,
            sample,
        )
        self._rollout_revision, self._start_step = metadata["revision"], metadata["start_step"]
        self._received_bytes = window.received_bytes
        self._index, self._seconds, self._playhead = 0, 0.0, 0.0
        self._last_clock = None
        self._paused, self._error = True, ""
        worlds.frame(FrameNeeds()).paused = True
        if changed:
            self._revision += 1

    def release(self) -> None:
        """Discard pending results and stop the optional, deadline-bounded fetch worker."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._future = None
