"""Bounded local playback of compact, model-bound qpos recordings."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .base import AdapterCaps, FrameNeeds, ReplayInfo, SceneAdapterBase, WorldSelectionInfo
from .worlds import WorldInstances


class JointReplayAdapter(SceneAdapterBase):
    """Evaluate only selected worlds from a memory-mapped joint recording.

    The version-1 diagnostic archive contains manifest.json, model.mjb, qpos.npy
    and origins.npy. MJB requires the recorded MuJoCo version. Displayed worlds
    use a compact grid; IDs and local qpos retain their recorded meaning. Local
    playback has its own pause/seek clock, independent of a training producer.
    """

    caps = AdapterCaps(
        name="joint_replay",
        external_clock=True,
        clock_control=False,
        features=(("world.selection", 1), ("replay.control", 1)),
    )

    def __init__(
        self,
        archive: Path,
        worlds: int | None = None,
        *,
        world_ids: tuple[int, ...] | None = None,
        max_worlds: int = 64,
        realtime: bool = True,
        paused: bool = False,
    ) -> None:
        archive = Path(archive)
        metadata = json.loads((archive / "manifest.json").read_text())
        qpos = np.load(archive / "qpos.npy", mmap_mode="r", allow_pickle=False)
        origins = np.load(archive / "origins.npy", mmap_mode="r", allow_pickle=False)
        if qpos.ndim != 3 or origins.shape != (qpos.shape[1], 3):
            raise ValueError("Invalid recording shape or origins")
        self._initialize(
            metadata,
            qpos,
            archive / "model.mjb",
            worlds=worlds,
            world_ids=world_ids,
            max_worlds=max_worlds,
            realtime=realtime,
            paused=paused,
        )

    def _initialize(
        self, metadata, qpos, model_path, *, worlds, world_ids, max_worlds, realtime, paused
    ):
        import mujoco

        from .mujoco import MuJoCoAdapter

        self._validate_recording(metadata, qpos)
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise ValueError("Recording model checksum mismatch")
        if metadata["mujoco_version"] != mujoco.__version__:
            raise ValueError("This MJB archive requires the recorded MuJoCo version")
        if (
            (worlds is not None and (type(worlds) is not int or worlds < 1))
            or type(max_worlds) is not int
            or max_worlds < 1
        ):
            raise ValueError("World count and preview limit must be positive integers")
        self.metadata, self.qpos = metadata, qpos
        self.hz = float(metadata["hz"])
        self._total_worlds = int(metadata.get("total_worlds", qpos.shape[1]))
        self._available_ids = tuple(metadata.get("world_ids", range(qpos.shape[1])))
        self._columns = {world: column for column, world in enumerate(self._available_ids)}
        self._max_worlds = min(max_worlds, self._total_worlds)
        initial_count = min(
            min(4, self._max_worlds) if worlds is None else worlds, len(self._available_ids)
        )
        if world_ids is None and initial_count > self._max_worlds:
            raise ValueError(f"Choose between 1 and {self._max_worlds} worlds")
        self._world_ids: tuple[int, ...] = ()
        self._revision = 0
        self._index = 0
        self._seconds = 0.0
        self._realtime = realtime
        self._paused = paused
        self._speed = 1.0
        self._loop = True
        self._playhead = 0.0
        self._last_clock: float | None = None
        self._error = ""
        self._spacing = float(metadata.get("display_spacing", 7.0))
        adapter = MuJoCoAdapter(external_clock=True)
        try:
            adapter.load_model(mujoco.MjModel.from_binary_path(str(model_path)))
            self.model = adapter.model
            if qpos.shape[2] != self.model.nq or metadata["joint_names"] != [
                self.model.joint(i).name for i in range(self.model.njnt)
            ]:
                raise ValueError("Recording joint layout does not match model")
            self.data = mujoco.MjData(self.model)
            self._template = adapter.scene_source()
            self._template_frame = adapter.frame(FrameNeeds())
        finally:
            adapter.release()
        self.set_world_selection(
            self._available_ids[:initial_count] if world_ids is None else world_ids
        )

    @staticmethod
    def _validate_recording(metadata, qpos):
        if metadata["version"] != 1:
            raise ValueError("Unsupported diagnostic archive version")
        hz = float(metadata["hz"])
        spacing = float(metadata.get("display_spacing", 7.0))
        if (
            qpos.dtype != np.float32
            or metadata["dtype"] != "float32"
            or list(qpos.shape) != metadata["shape"]
            or qpos.ndim != 3
            or min(qpos.shape) < 1
            or qpos.shape[0] < 2
            or not math.isfinite(hz)
            or hz <= 0
            or not math.isfinite(spacing)
            or spacing <= 0
        ):
            raise ValueError("Invalid recording shape, dtype, or cadence")
        total = metadata.get("total_worlds", qpos.shape[1])
        ids = metadata.get("world_ids", range(qpos.shape[1]))
        if (
            type(total) is not int
            or total < 1
            or len(ids) != qpos.shape[1]
            or any(type(i) is not int or not 0 <= i < total for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Invalid recorded world identities")

    @property
    def structure_revision(self) -> int:
        """Return the revision of the displayed world subset."""
        return self._revision

    def world_selection(self) -> WorldSelectionInfo:
        """Return original IDs and the explicit preview budget."""
        return WorldSelectionInfo(self._total_worlds, self._world_ids, self._max_worlds)

    def _validate_ids(self, world_ids):
        ids = tuple(world_ids)
        if not 1 <= len(ids) <= self._max_worlds:
            raise ValueError(f"Choose between 1 and {self._max_worlds} worlds")
        if any(type(i) is not int or not 0 <= i < self._total_worlds for i in ids):
            raise ValueError(f"World IDs must be integers from 0 to {self._total_worlds - 1}")
        if len(set(ids)) != len(ids):
            raise ValueError("World IDs must be unique")
        return ids

    def _make_worlds(self, ids):
        side = math.ceil(math.sqrt(len(ids)))
        offsets = np.zeros((len(ids), 3), np.float32)
        offsets[:, 0] = np.arange(len(ids)) % side
        offsets[:, 1] = np.arange(len(ids)) // side
        offsets -= (offsets.min(axis=0) + offsets.max(axis=0)) * 0.5
        offsets *= self._spacing
        worlds = WorldInstances(self._template, self._template_frame, offsets, world_ids=ids)
        n = len(worlds.pose_indices)
        return (
            worlds,
            np.empty((len(ids), n, 3), np.float32),
            np.empty((len(ids), n, 3, 3), np.float32),
            np.empty((len(ids), self.model.nq), np.float32),
        )

    def set_world_selection(self, world_ids: tuple[int, ...]) -> bool:
        """Atomically replace the subset, evaluating only the requested worlds."""
        ids = self._validate_ids(world_ids)
        if any(i not in self._columns for i in ids):
            raise ValueError(
                "Worlds are not in the local clip; sync a rollout containing these IDs"
            )
        if ids == self._world_ids:
            return False
        candidate, positions, rotations, sample = self._make_worlds(ids)
        self._evaluate(candidate, ids, sample, positions, rotations, self._index, self._seconds)
        self.worlds, self.positions, self.rotations, self._sample = (
            candidate,
            positions,
            rotations,
            sample,
        )
        self._world_ids = ids
        self._revision += 1
        return True

    def _evaluate(
        self, worlds, ids, sample, positions, rotations, index, seconds, *, qpos=None, columns=None
    ):
        import mujoco

        qpos = self.qpos if qpos is None else qpos
        columns = self._columns if columns is None else columns
        recorded = qpos[index % len(qpos)]
        # Gather one selected sample, never the full trajectory or full world batch.
        for row, world_id in enumerate(ids):
            sample[row] = recorded[columns[world_id]]
        if not np.isfinite(sample).all() or not math.isfinite(seconds):
            raise ValueError("Recording contains non-finite qpos or time")
        slots = worlds.pose_indices
        for row, values in enumerate(sample):
            self.data.qpos[:] = values
            mujoco.mj_kinematics(self.model, self.data)
            np.take(self.data.geom_xpos, slots, axis=0, out=positions[row])
            np.take(self.data.geom_xmat.reshape(-1, 3, 3), slots, axis=0, out=rotations[row])
        worlds.set_poses(positions, rotations, time=seconds)
        result = worlds.frame(FrameNeeds())
        result.step, result.paused = index, self._paused
        return result

    def update(self, frame: int, seconds: float):
        """Evaluate a caller-owned sample; realtime=False keeps that caller's clock."""
        result = self._evaluate(
            self.worlds,
            self._world_ids,
            self._sample,
            self.positions,
            self.rotations,
            frame,
            seconds,
        )
        self._index, self._seconds = frame, seconds
        self._error = ""
        return result

    def replay_info(self) -> ReplayInfo:
        """Read the clip clock without advancing, decoding or making network requests."""
        return ReplayInfo(
            self._index % len(self.qpos),
            len(self.qpos),
            self.hz,
            self._paused,
            self._speed,
            self._loop,
            self._error,
        )

    def set_replay_playback(self, *, paused=None, speed=None, loop=None) -> None:
        """Patch playback settings without discontinuities when speed or pause changes."""
        if (paused is not None and type(paused) is not bool) or (
            loop is not None and type(loop) is not bool
        ):
            raise ValueError("Paused and loop must be booleans")
        if speed is not None and (
            isinstance(speed, bool) or not math.isfinite(speed) or not 0.05 <= speed <= 8
        ):
            raise ValueError("Replay speed must be between 0.05 and 8")
        self.frame(FrameNeeds())
        if speed is not None:
            self._speed = float(speed)
        if loop is not None:
            self._loop = loop
        if paused is not None:
            self._paused = paused
            if not paused:
                self._realtime = True
                if not self._loop and self._index >= len(self.qpos) - 1:
                    self.seek_replay(0)
                    self._paused = False
        self._last_clock = None
        self.worlds.frame(FrameNeeds()).paused = self._paused

    def seek_replay(self, frame: int) -> None:
        """Pause on a valid frame; invalid or corrupt frames leave playback intact."""
        if type(frame) is not int or not 0 <= frame < len(self.qpos):
            raise ValueError(f"Frame must be between 0 and {len(self.qpos) - 1}")
        if frame != self._index:
            self.update(frame, frame / self.hz)
        self._paused = True
        self._error = ""
        self._playhead = float(frame)
        self._last_clock = None
        self.worlds.frame(FrameNeeds()).paused = True

    def frame(self, needs: FrameNeeds):
        """Reuse poses while paused/between samples; never build a catch-up queue."""
        if self._realtime and not self._paused:
            now = time.perf_counter()
            if self._last_clock is not None:
                self._playhead += max(0.0, now - self._last_clock) * self.hz * self._speed
            self._last_clock = now
            if self._loop:
                self._playhead %= len(self.qpos)
            elif self._playhead >= len(self.qpos) - 1:
                self._playhead = float(len(self.qpos) - 1)
                self._paused = True
            index = int(self._playhead)
            if index != self._index:
                try:
                    self.update(index, index / self.hz)
                except ValueError as error:
                    self._error = str(error)
                    self._paused = True
                    self._playhead = float(self._index)
        result = self.worlds.frame(needs)
        result.paused = self._paused
        return result

    def scene_source(self):
        """Return resources shared by only the displayed instances."""
        return self.worlds.scene_source()

    def nodes(self):
        """Return hierarchy entries labelled with original world IDs."""
        return self.worlds.nodes()

    def camera_hint(self):
        """Suggest initial framing for the displayed subset."""
        return self.worlds.camera_hint()
