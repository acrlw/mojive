"""Non-blocking GPU render-pass timing for the wgpu backend."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from queue import SimpleQueue
from threading import Thread
from typing import Any

import wgpu

TimestampWriter = Callable[[str], dict[str, Any] | None]

_QUERY_COUNT = 64
_READBACK_COUNT = 3


def default_device() -> wgpu.GPUDevice:
    """Return the shared device with timestamp queries when the adapter supports them."""

    with suppress(RuntimeError):
        wgpu.utils.preconfigure_default_device(
            "forge-viewer GPU timing", preferred_features={wgpu.FeatureName.timestamp_query}
        )
        # Another consumer may have created the shared device first. Timing
        # remains optional and WgpuTiming reports the actual device capability.
    return wgpu.utils.get_default_device()


class WgpuTiming:
    """Collect pass timestamps without stalling the render loop."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self.active = wgpu.FeatureName.timestamp_query in device.features
        self.gpu_ms: dict[str, float] = {}
        self._query_set: wgpu.GPUQuerySet | None = None
        self._resolve_buffer: wgpu.GPUBuffer | None = None
        self._readbacks: list[wgpu.GPUBuffer] = []
        self._busy: list[bool] = []
        self._jobs: SimpleQueue[Any] = SimpleQueue()
        self._worker: Thread | None = None
        self._frame_slot: int | None = None
        self._entries: list[tuple[str, int, int]] = []
        self._cursor = 0
        self._released = False
        if not self.active:
            return
        size = _QUERY_COUNT * 8
        self._query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=_QUERY_COUNT)
        self._resolve_buffer = device.create_buffer(
            size=size, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC
        )
        self._readbacks = [
            device.create_buffer(
                size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
            )
            for _ in range(_READBACK_COUNT)
        ]
        self._busy = [False] * len(self._readbacks)
        self._worker = Thread(target=self._readback_loop, name="wgpu-timing", daemon=True)
        self._worker.start()

    def begin_frame(self) -> None:
        self._frame_slot = None
        self._entries = []
        self._cursor = 0
        if not self.active:
            return
        self._frame_slot = next((i for i, busy in enumerate(self._busy) if not busy), None)
        if self._frame_slot is not None:
            self._busy[self._frame_slot] = True

    def timestamp_writes(self, name: str) -> dict[str, Any] | None:
        if self._frame_slot is None or self._cursor + 2 > _QUERY_COUNT:
            return None
        begin, end = self._cursor, self._cursor + 1
        self._cursor += 2
        self._entries.append((str(name), begin, end))
        return {
            "query_set": self._query_set,
            "beginning_of_pass_write_index": begin,
            "end_of_pass_write_index": end,
        }

    def resolve(self, encoder: wgpu.GPUCommandEncoder):
        slot = self._frame_slot
        self._frame_slot = None
        if slot is None:
            return None
        if not self._entries:
            self._busy[slot] = False
            return None
        entries = tuple(self._entries)
        count = self._cursor
        size = count * 8
        encoder.resolve_query_set(self._query_set, 0, count, self._resolve_buffer, 0)
        encoder.copy_buffer_to_buffer(self._resolve_buffer, 0, self._readbacks[slot], 0, size)
        return slot, entries, count

    def submitted(self, pending) -> None:
        if pending is None:
            return
        slot, entries, count = pending
        buffer = self._readbacks[slot]
        promise = buffer.map_async(wgpu.MapMode.READ)
        self._jobs.put((slot, entries, count, promise))

    def _readback_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            slot, entries, count, promise = job
            buffer = self._readbacks[slot]
            try:
                promise.sync_wait()
                values = buffer.read_mapped(size=count * 8).cast("Q").tolist()
                totals: dict[str, float] = {}
                for name, begin, end in entries:
                    totals[name] = totals.get(name, 0.0) + (values[end] - values[begin]) / 1e6
                if not self._released:
                    self.gpu_ms = totals
            except Exception:
                pass
            finally:
                if buffer.map_state == wgpu.BufferMapState.mapped:
                    buffer.unmap()
                self._busy[slot] = False

    def release(self) -> None:
        if not self.active or self._released:
            return
        self._released = True
        self._jobs.put(None)
        self._worker.join()
        for buffer in self._readbacks:
            if buffer.map_state == wgpu.BufferMapState.mapped:
                buffer.unmap()
            buffer.destroy()
        self._readbacks.clear()
        self._resolve_buffer.destroy()
        self._query_set.destroy()
