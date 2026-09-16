"""Bounded frame ownership between interactive capture and synchronous video encoding."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .recording import VideoRecorder


class BufferedVideoRecorder:
    """Encode on one worker using two reusable RGB buffers, without dropping frames.

    Only the producer calls append/close. Full buffers apply backpressure; close
    drains accepted frames and reports worker failures. Offline callers can keep
    using VideoRecorder directly. Memory is bounded by two input-sized RGB frames.
    """

    def __init__(self, recorder: VideoRecorder, *, timeout: float = 30.0) -> None:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Encoder timeout must be finite and positive")
        self.recorder = recorder
        self.size = recorder.size
        self._free: queue.Queue = queue.Queue(maxsize=2)
        self._pending: queue.Queue = queue.Queue()
        self._failure: Exception | None = None
        self._failed = threading.Event()
        self._closed = False
        self._joined = False
        self._finished = threading.Event()
        self.started = threading.Event()
        self.written_frames = 0
        self._busy_since = 0.0
        self._timeout = float(timeout)
        self._timeout_error: RuntimeError | None = None
        try:
            for _ in range(2):
                self._free.put(np.empty((self.size[1], self.size[0], 3), np.uint8))
            self._worker = threading.Thread(target=self._encode, name="mojive-video-encoder")
            self._worker.start()
        except BaseException as exc:
            try:
                recorder.close()
            except Exception as failure:
                exc.add_note(f"Video finalization: {failure}")
            raise

    def _check_failure(self) -> None:
        if self._timeout_error is not None:
            raise self._timeout_error
        if self._failed.is_set():
            notes = "; ".join(getattr(self._failure, "__notes__", ()))
            detail = f"{self._failure}; {notes}" if notes else str(self._failure)
            raise RuntimeError(f"Video encoder failed: {detail}") from self._failure

    def check(self) -> None:
        """Report worker failures, including a stalled writer while capture is paused."""
        self._interrupt_stalled_encoder()
        self._check_failure()

    def _interrupt_stalled_encoder(self) -> None:
        since = self._busy_since
        if since and time.monotonic() - since >= self._timeout and self._timeout_error is None:
            self._timeout_error = RuntimeError(
                f"Video encoder made no progress for {self._timeout:g} seconds"
            )
            self.recorder.abort()

    def append(self, frame: np.ndarray) -> None:
        """Copy one frame into owned storage; wait only when both buffers are occupied."""
        if self._closed:
            raise RuntimeError("Cannot append video frames after close()")
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[:2] != (self.size[1], self.size[0]) or image.shape[2] < 3:
            raise ValueError(
                f"video frames must be {self.size[0]}×{self.size[1]} RGB, got {image.shape}"
            )
        while True:
            self.check()
            try:
                buffer = self._free.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        try:
            self._check_failure()
            np.copyto(buffer, image[..., :3], casting="unsafe")
        except BaseException:
            self._free.put(buffer)
            raise
        self._pending.put(buffer)

    def _encode(self) -> None:
        try:
            while (frame := self._pending.get()) is not None:
                try:
                    self._busy_since = time.monotonic()
                    self.recorder.append(frame)
                    self.written_frames += 1
                    self.started.set()
                finally:
                    self._busy_since = 0.0
                    self._free.put(frame)
        except Exception as exc:
            self._failure = exc
            self._failed.set()
        finally:
            try:
                self._busy_since = time.monotonic()
                self.recorder.close()
            except Exception as exc:
                if self._failure is None:
                    self._failure = exc
                else:
                    self._failure.add_note(f"Video finalization: {exc}")
                self._failed.set()
            finally:
                self._busy_since = 0.0
                self._finished.set()
                self.started.set()

    def begin_close(self) -> None:
        """Stop accepting frames and finalize on the worker without waiting."""
        if not self._closed:
            self._closed = True
            self._pending.put(None)

    def poll_close(self) -> bool:
        """Return whether finalization finished; close() retrieves its result."""
        self._interrupt_stalled_encoder()
        return self._finished.is_set()

    def close(self) -> None:
        """Drain accepted frames, join the worker, and surface any encoding failure."""
        if self._joined:
            return
        self.begin_close()
        while not self._finished.wait(0.05):
            self._interrupt_stalled_encoder()
            if self._timeout_error is not None:
                self._worker.join(timeout=1.0)
                if self._worker.is_alive():
                    raise RuntimeError("Video encoder did not exit after abort")
        self._worker.join()
        self._joined = True
        self._check_failure()
