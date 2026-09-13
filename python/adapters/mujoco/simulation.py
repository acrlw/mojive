"""MuJoCo stepping with exclusive state ownership and pooled display snapshots."""

from __future__ import annotations

import threading
import time

import mujoco

from mojive.session.simulation import Snapshot, SnapshotPool


class MuJoCoSimulation:
    """Keep native physics work separate from the adapter's displayed MjData.

    The model is read-only while running. Session fences this worker before
    commands transfer authoritative data back to the adapter for editing.
    """

    def __init__(self, adapter):
        self._adapter = adapter
        self._condition = threading.Condition()
        self._thread = None
        self._running = False
        self._busy = False
        self._closed = False
        self._error = None
        self._model = self._data = None
        self._slots = []
        self._pool = None
        self._held = None
        self._displaying = False
        self._completed = self._observed = 0
        self._seconds_per_step = 0.0
        self._suspended_time = 0.0
        self._speed = 1.0

    def start(self, speed: float, *, reset_clock: bool = True) -> None:
        """Start real-time stepping after binding an immutable initial snapshot."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Simulation driver is closed")
            if self._error is not None:
                raise RuntimeError("Physics worker failed") from self._error
            if self._running:
                return
            model, data = self._adapter.model, self._adapter.data
            if model is not self._model or data is not self._data:
                self._model, self._data = model, data
                self._slots = [mujoco.MjData(model) for _ in range(3)]
                self._seconds_per_step = 0.0
                reset_clock = True
            self._pool = SnapshotPool(latest=True)
            slot = self._pool.reserve()
            mujoco.mj_copyData(self._slots[slot], model, data)
            self._adapter.use_data(self._slots[slot])
            self._held = slot
            self._displaying = True
            if reset_clock or data.time != self._suspended_time or speed != self._speed:
                self._epoch = time.perf_counter()
                self._epoch_step = self._completed
            self._speed = speed
            self._running = True
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="mojive-physics", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()

    def poll(self) -> int:
        """Bind the newest completed state without waiting for a physics batch."""
        if self._error is not None:
            raise RuntimeError("Physics worker failed") from self._error
        if not self._running or self._pool is None:
            return 0
        snapshot = self._pool.acquire(wait=False)
        if snapshot is None:
            return 0
        self._adapter.use_data(self._slots[snapshot.slot])
        self._pool.release(self._held)
        self._held = snapshot.slot
        steps = snapshot.step - self._observed
        self._observed = snapshot.step
        return steps

    def suspend(self) -> int:
        """Fence native work and transfer current authoritative state to edits."""
        with self._condition:
            self._running = False
            self._condition.notify_all()
            self._condition.wait_for(lambda: not self._busy)
            if self._displaying:
                self._adapter.use_data(self._data)
                self._displaying = False
                self._suspended_time = float(self._data.time)
            steps = self._completed - self._observed
            self._observed = self._completed
            return steps

    def close(self) -> None:
        """Join the physics owner before model or graphics resources are released."""
        if self._closed:
            return
        self.suspend()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
        self._slots.clear()
        self._model = self._data = self._pool = None

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._running or self._closed)
                if self._closed:
                    return
                # Aim for at most 8 ms between publications, subject to the
                # model timestep. Bound batches by count and measured native
                # cost; a single expensive step is not preemptible.
                timestep = float(self._model.opt.timestep)
                count = max(1, min(8, int(0.008 * self._speed / timestep)))
                if self._seconds_per_step > 0:
                    count = min(count, max(1, int(0.008 / self._seconds_per_step)))
                else:
                    count = 1
                due = (
                    self._epoch
                    + (self._completed - self._epoch_step + count) * timestep / self._speed
                )
                delay = due - time.perf_counter()
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    continue
                self._busy = True
            try:
                started = time.perf_counter()
                mujoco.mj_step(self._model, self._data, nstep=count)
                elapsed = (time.perf_counter() - started) / count
                # React immediately to a costly step; relax the estimate slowly
                # so new contacts do not leave edits waiting on oversized batches.
                self._seconds_per_step = max(elapsed, 0.9 * self._seconds_per_step + 0.1 * elapsed)
                self._completed += count
                slot = self._pool.reserve()
                mujoco.mj_copyData(self._slots[slot], self._model, self._data)
                self._pool.publish(Snapshot(slot, self._completed, time.perf_counter_ns()))
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._running = False
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
