"""Interactive encoding preserves ownership, ordering, bounded memory and failure evidence."""

import threading

import numpy as np
import pytest

from mojive.capture.video_queue import BufferedVideoRecorder


class Recorder:
    size = (8, 6)

    def __init__(self):
        self.values = []
        self.buffers = set()
        self.closed = 0

    def append(self, image):
        self.values.append(int(image[0, 0, 0]))
        self.buffers.add(id(image))

    def close(self):
        self.closed += 1


def test_encoder_owns_two_reusable_buffers_and_drains_frames_in_order():
    recorder = Recorder()
    writer = BufferedVideoRecorder(recorder)
    image = np.zeros((6, 8, 4), np.uint8)
    try:
        for index in range(100):
            image[:] = index
            writer.append(image)
            image[:] = 255
    finally:
        writer.close()
    assert recorder.values == list(range(100))
    assert len(recorder.buffers) <= 2
    assert recorder.closed == 1
    assert not writer._worker.is_alive()
    writer.close()
    assert recorder.closed == 1
    with pytest.raises(RuntimeError, match="after close"):
        writer.append(image)


def test_full_queue_applies_backpressure_without_dropping_or_overwriting():
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()

    class SlowRecorder(Recorder):
        def append(self, image):
            entered.set()
            assert release.wait(5)
            super().append(image)

    recorder = SlowRecorder()
    writer = BufferedVideoRecorder(recorder)
    image = np.full((6, 8, 3), 1, np.uint8)
    producer = None
    try:
        writer.append(image)
        assert entered.wait(2)
        image[:] = 2
        writer.append(image)
        image[:] = 3

        def submit():
            writer.append(image)
            submitted.set()

        producer = threading.Thread(target=submit)
        producer.start()
        assert not submitted.wait(0.05)
        release.set()
        assert submitted.wait(2)
    finally:
        release.set()
        if producer is not None:
            producer.join(2)
        writer.close()
    assert recorder.values == [1, 2, 3]


def test_worker_failure_unblocks_producer_and_is_reported_on_close():
    failed = threading.Event()

    class BrokenRecorder(Recorder):
        def append(self, image):
            failed.set()
            raise ValueError("broken pipe fixture")

    recorder = BrokenRecorder()
    writer = BufferedVideoRecorder(recorder)
    image = np.zeros((6, 8, 3), np.uint8)
    writer.append(image)
    assert failed.wait(2)
    assert writer._failed.wait(2)
    with pytest.raises(RuntimeError, match="broken pipe fixture"):
        writer.append(image)
    with pytest.raises(RuntimeError, match="broken pipe fixture"):
        writer.close()
    assert not writer._worker.is_alive()
    assert recorder.closed == 1
    writer.close()


def test_finalization_error_is_reported_after_successful_frame_submission():
    class BrokenFinalizer(Recorder):
        def close(self):
            raise RuntimeError("encoder finalization fixture")

    writer = BufferedVideoRecorder(BrokenFinalizer())
    writer.append(np.zeros((6, 8, 3), np.uint8))
    with pytest.raises(RuntimeError, match="encoder finalization fixture"):
        writer.close()
    assert not writer._worker.is_alive()


@pytest.mark.parametrize("shape", [(6, 8), (8, 6, 3), (6, 8, 2)])
def test_bad_frame_does_not_consume_a_buffer(shape):
    recorder = Recorder()
    writer = BufferedVideoRecorder(recorder)
    try:
        with pytest.raises(ValueError, match="video frames"):
            writer.append(np.zeros(shape, np.uint8))
        writer.append(np.zeros((6, 8, 3), np.uint8))
    finally:
        writer.close()
    assert recorder.values == [0]


def test_begin_close_returns_while_worker_finishes_and_close_drains():
    entered, release = threading.Event(), threading.Event()

    class Finalizer(Recorder):
        def close(self):
            entered.set()
            assert release.wait(2)
            super().close()

    recorder = Finalizer()
    writer = BufferedVideoRecorder(recorder)
    writer.append(np.zeros((6, 8, 3), np.uint8))
    try:
        writer.begin_close()
        assert entered.wait(1)
        assert not writer.poll_close()
        with pytest.raises(RuntimeError, match="after close"):
            writer.append(np.zeros((6, 8, 3), np.uint8))
    finally:
        release.set()
        writer.close()
    assert writer.poll_close() and recorder.closed == 1


@pytest.mark.parametrize("stall", ["append", "close"])
def test_stalled_encoder_is_aborted_and_worker_is_joined(stall):
    entered, release = threading.Event(), threading.Event()

    class Stalled(Recorder):
        def append(self, image):
            if stall == "append":
                entered.set()
                assert release.wait(2)
            super().append(image)

        def close(self):
            if stall == "close":
                entered.set()
                assert release.wait(2)
            super().close()

        def abort(self):
            release.set()

    writer = BufferedVideoRecorder(Stalled(), timeout=0.05)
    writer.append(np.zeros((6, 8, 3), np.uint8))
    writer.begin_close()
    assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError, match="no progress"):
            writer.close()
    finally:
        release.set()
        writer._worker.join(2)
    assert not writer._worker.is_alive()
