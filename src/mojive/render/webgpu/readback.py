"""Bounded asynchronous texture readback for the WebGPU backend."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import SimpleQueue
from threading import Condition, Lock, Thread
from typing import Any

import numpy as np
import wgpu


def aligned_row_bytes(width: int, bytes_per_pixel: int) -> int:
    """Return WebGPU's 256-byte-aligned texture-copy row stride."""

    raw = int(width) * int(bytes_per_pixel)
    return (raw + 255) // 256 * 256


def decode_rows(
    data: Any,
    *,
    width: int,
    height: int,
    row_bytes: int,
    dtype: np.dtype,
    storage_channels: int,
    output_channels: int,
    flip: bool,
    out: np.ndarray | None,
) -> np.ndarray:
    """Remove row padding and convert a mapped texture copy into its result array."""

    dtype = np.dtype(dtype)
    pixel_bytes = dtype.itemsize * int(storage_channels)
    raw = np.frombuffer(data, np.uint8, count=int(height) * int(row_bytes)).reshape(
        int(height), int(row_bytes)
    )
    image = (
        raw[:, : int(width) * pixel_bytes]
        .view(dtype)
        .reshape(int(height), int(width), int(storage_channels))
    )
    if output_channels == 1:
        image = image[..., 0]
    elif output_channels != storage_channels:
        image = image[..., :output_channels]
    # WebGPU rows are top-first; flip=True follows the public top-first convention.
    if not flip:
        image = image[::-1]
    if out is not None:
        np.copyto(out, image, casting="unsafe")
        return out
    return np.array(image, copy=True, order="C")


@dataclass
class _Slot:
    buffer: wgpu.GPUBuffer | None = None
    capacity: int = 0
    busy: bool = False


@dataclass(frozen=True)
class _Job:
    slot: int
    promise: Any
    future: Future[np.ndarray]
    size: int
    width: int
    height: int
    row_bytes: int
    dtype: np.dtype
    storage_channels: int
    output_channels: int
    flip: bool
    out: np.ndarray | None


class WgpuReadbackQueue:
    """Copy textures through a bounded staging ring without blocking the render thread.

    A queue submission is ordered immediately after the render submission. Mapping,
    row unpacking, channel conversion, and destination-array copying happen on one
    worker. When all slots are occupied, enqueueing applies bounded backpressure
    instead of allocating an unbounded number of GPU buffers.
    """

    def __init__(self, device: wgpu.GPUDevice, slots: int = 3) -> None:
        if slots < 1:
            raise ValueError("readback queue requires at least one slot")
        self._device = device
        self._slots = [_Slot() for _ in range(int(slots))]
        self._condition = Condition()
        self._lifecycle = Lock()
        self._jobs: SimpleQueue[_Job | None] = SimpleQueue()
        self._closed = False
        self._worker = Thread(target=self._run, name="wgpu-readback", daemon=True)
        self._worker.start()

    def enqueue(
        self,
        texture: wgpu.GPUTexture,
        *,
        width: int,
        height: int,
        dtype: np.dtype,
        storage_channels: int,
        output_channels: int | None = None,
        flip: bool = True,
        out: np.ndarray | None = None,
    ) -> Future[np.ndarray]:
        """Schedule one texture copy and return a future for its owned CPU result."""

        with self._lifecycle:
            return self._enqueue(
                texture,
                width=width,
                height=height,
                dtype=dtype,
                storage_channels=storage_channels,
                output_channels=output_channels,
                flip=flip,
                out=out,
            )

    def _enqueue(
        self,
        texture: wgpu.GPUTexture,
        *,
        width: int,
        height: int,
        dtype: np.dtype,
        storage_channels: int,
        output_channels: int | None,
        flip: bool,
        out: np.ndarray | None,
    ) -> Future[np.ndarray]:
        width, height = int(width), int(height)
        dtype = np.dtype(dtype)
        storage_channels = int(storage_channels)
        output_channels = int(storage_channels if output_channels is None else output_channels)
        if width < 1 or height < 1:
            raise ValueError("readback dimensions must be positive")
        if not 1 <= output_channels <= storage_channels:
            raise ValueError("output channels must be between one and the stored channel count")
        shape = (height, width) if output_channels == 1 else (height, width, output_channels)
        if out is not None and out.shape != shape:
            raise ValueError(f"Expected destination with shape {shape}, got {out.shape}")
        row_bytes = aligned_row_bytes(width, dtype.itemsize * int(storage_channels))
        size = row_bytes * height
        slot_index = self._reserve_slot(size)
        slot = self._slots[slot_index]
        assert slot.buffer is not None
        future: Future[np.ndarray] = Future()
        try:
            encoder = self._device.create_command_encoder()
            encoder.copy_texture_to_buffer(
                {"texture": texture, "origin": (0, 0, 0)},
                {
                    "buffer": slot.buffer,
                    "offset": 0,
                    "bytes_per_row": row_bytes,
                    "rows_per_image": height,
                },
                (width, height, 1),
            )
            self._device.queue.submit([encoder.finish()])
            promise = slot.buffer.map_async(wgpu.MapMode.READ, size=size)
        except Exception:
            self._release_slot(slot_index)
            raise
        self._jobs.put(
            _Job(
                slot_index,
                promise,
                future,
                size,
                width,
                height,
                row_bytes,
                dtype,
                storage_channels,
                output_channels,
                bool(flip),
                out,
            )
        )
        return future

    def _reserve_slot(self, size: int) -> int:
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("readback queue is closed")
                index = next((i for i, slot in enumerate(self._slots) if not slot.busy), None)
                if index is not None:
                    break
                self._condition.wait()
            slot = self._slots[index]
            slot.busy = True
            try:
                if slot.buffer is None or slot.capacity < size:
                    if slot.buffer is not None:
                        slot.buffer.destroy()
                    slot.buffer = self._device.create_buffer(
                        size=size,
                        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
                    )
                    slot.capacity = size
            except Exception:
                slot.busy = False
                self._condition.notify_all()
                raise
            return index

    def _release_slot(self, index: int) -> None:
        with self._condition:
            self._slots[index].busy = False
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            slot = self._slots[job.slot]
            assert slot.buffer is not None
            running = job.future.set_running_or_notify_cancel()
            result = None
            error = None
            try:
                job.promise.sync_wait()
                if running:
                    mapped = slot.buffer.read_mapped(size=job.size, copy=True)
                    result = decode_rows(
                        mapped,
                        width=job.width,
                        height=job.height,
                        row_bytes=job.row_bytes,
                        dtype=job.dtype,
                        storage_channels=job.storage_channels,
                        output_channels=job.output_channels,
                        flip=job.flip,
                        out=job.out,
                    )
            except Exception as exc:
                error = exc
            finally:
                if slot.buffer.map_state == wgpu.BufferMapState.mapped:
                    slot.buffer.unmap()
                self._release_slot(job.slot)
            # Release GPU ownership before completing the public future. User
            # callbacks may immediately submit another capture.
            if running:
                if error is None:
                    job.future.set_result(result)
                else:
                    job.future.set_exception(error)

    def drain(self) -> None:
        """Wait until every queued mapping has released its staging slot."""

        with self._condition:
            self._condition.wait_for(lambda: not any(slot.busy for slot in self._slots))

    def release(self) -> None:
        """Complete queued reads, stop the worker, and destroy staging buffers."""

        with self._lifecycle:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
            # Lifecycle serialization guarantees no producer can append a job
            # after this sentinel.
            self._jobs.put(None)
        self._worker.join()
        for slot in self._slots:
            if slot.buffer is not None:
                if slot.buffer.map_state == wgpu.BufferMapState.mapped:
                    slot.buffer.unmap()
                slot.buffer.destroy()
                slot.buffer = None
            slot.capacity = 0
            slot.busy = False
