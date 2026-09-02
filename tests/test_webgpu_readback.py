from __future__ import annotations

import numpy as np

from mojive.render.webgpu.readback import (
    WgpuSyncReadback,
    aligned_row_bytes,
    decode_packed_rgb,
    decode_rows,
    rgba_to_rgb,
)


class _FakeBuffer:
    def __init__(self, size: int) -> None:
        self.data = bytearray(size)
        self.map_state = "unmapped"
        self.destroyed = False

    def map_sync(self, _mode, *, size: int) -> None:
        assert size <= len(self.data)
        self.map_state = "mapped"

    def read_mapped(self, *, size: int, copy: bool):
        assert self.map_state == "mapped"
        assert not copy
        return memoryview(self.data)[:size]

    def unmap(self) -> None:
        self.map_state = "unmapped"

    def destroy(self) -> None:
        self.destroyed = True


class _FakeEncoder:
    def finish(self):
        return self


class _FakeQueue:
    def __init__(self) -> None:
        self.submissions = 0

    def submit(self, commands) -> None:
        assert len(commands) == 1
        self.submissions += 1


class _FakeDevice:
    def __init__(self) -> None:
        self.queue = _FakeQueue()
        self.buffers: list[_FakeBuffer] = []

    def create_buffer(self, *, size: int, usage) -> _FakeBuffer:
        del usage
        buffer = _FakeBuffer(size)
        self.buffers.append(buffer)
        return buffer

    def create_command_encoder(self) -> _FakeEncoder:
        return _FakeEncoder()


def test_rgba_to_rgb_is_exact_and_returns_writable_storage() -> None:
    rgba = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]],
        np.uint8,
    )

    rgb = rgba_to_rgb(rgba)

    assert rgb.flags.c_contiguous
    assert rgb.flags.writeable
    assert np.array_equal(rgb, rgba[..., :3])


def test_rgba_to_rgb_writes_the_requested_destination() -> None:
    rgba = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
    storage = np.empty((3, 8, 3), np.float32)
    out = storage[:, ::2]

    result = rgba_to_rgb(rgba, out)

    assert result is out
    assert np.array_equal(out, rgba[..., :3])


def test_decode_packed_rgb_preserves_casting_strides_and_orientation() -> None:
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.concatenate((pixels.reshape(-1), np.array([201, 202], np.uint8)))
    storage = np.empty((2, 6, 3), np.float32)
    out = storage[:, ::2]

    result = decode_packed_rgb(
        padded,
        width=3,
        height=2,
        flip=False,
        out=out,
    )

    assert result is out
    assert np.array_equal(out, pixels[::-1])


def test_decode_rows_removes_padding_channels_and_preserves_orientation() -> None:
    width, height = 3, 2
    row_bytes = aligned_row_bytes(width, 4)
    storage = np.zeros((height, row_bytes), np.uint8)
    pixels = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ],
        np.uint8,
    )
    storage[:, : width * 4] = pixels.reshape(height, width * 4)

    image = decode_rows(
        storage,
        width=width,
        height=height,
        row_bytes=row_bytes,
        dtype=np.dtype(np.uint8),
        storage_channels=4,
        output_channels=3,
        flip=True,
        out=None,
    )

    assert image.flags.c_contiguous
    assert np.array_equal(image, pixels[..., :3])

    flipped = decode_rows(
        storage,
        width=width,
        height=height,
        row_bytes=row_bytes,
        dtype=np.dtype(np.uint8),
        storage_channels=4,
        output_channels=3,
        flip=False,
        out=None,
    )

    assert flipped.flags.c_contiguous
    assert np.array_equal(flipped, pixels[::-1, ..., :3])


def test_decode_rows_supports_casting_and_strided_destinations() -> None:
    width, height = 2, 2
    row_bytes = aligned_row_bytes(width, 4)
    storage = np.zeros((height, row_bytes), np.uint8)
    values = np.array([[1.5, 2.5], [3.5, 4.5]], np.float32)
    storage[:, : width * 4] = values.view(np.uint8).reshape(height, width * 4)
    destination_storage = np.zeros((height, width * 2), np.float64)
    destination = destination_storage[:, ::2]

    result = decode_rows(
        storage,
        width=width,
        height=height,
        row_bytes=row_bytes,
        dtype=np.dtype(np.float32),
        storage_channels=1,
        output_channels=1,
        flip=False,
        out=destination,
    )

    assert result is destination
    assert np.array_equal(destination, values[::-1])


def test_sync_readback_reuses_and_grows_one_staging_buffer() -> None:
    device = _FakeDevice()
    readback = WgpuSyncReadback(device)  # type: ignore[arg-type]

    def transfer(payload: bytes):
        def encode(_encoder, destination) -> None:
            destination.data[: len(payload)] = payload

        return encode

    def decode(data):
        return np.frombuffer(data, np.uint8).copy()

    assert np.array_equal(
        readback.read_copy(4, transfer(b"abcd"), decode), np.frombuffer(b"abcd", np.uint8)
    )
    first = device.buffers[0]
    assert np.array_equal(
        readback.read_copy(4, transfer(b"efgh"), decode), np.frombuffer(b"efgh", np.uint8)
    )
    assert device.buffers == [first]

    assert np.array_equal(
        readback.read_copy(8, transfer(b"ijklmnop"), decode),
        np.frombuffer(b"ijklmnop", np.uint8),
    )
    assert len(device.buffers) == 2
    assert first.destroyed
    assert all(buffer.map_state == "unmapped" for buffer in device.buffers)
    assert device.queue.submissions == 3

    latest = device.buffers[-1]
    readback.release()
    assert latest.destroyed


def test_sync_readback_unmaps_when_decode_raises() -> None:
    device = _FakeDevice()
    readback = WgpuSyncReadback(device)  # type: ignore[arg-type]

    def fail(_data):
        raise RuntimeError("decode failed")

    with np.testing.assert_raises_regex(RuntimeError, "decode failed"):
        readback.read_copy(4, lambda _encoder, _destination: None, fail)

    assert device.buffers[0].map_state == "unmapped"
