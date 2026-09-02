from __future__ import annotations

import numpy as np

from mojive.render.webgpu.readback import (
    aligned_row_bytes,
    decode_packed_rgb,
    decode_rows,
    rgba_to_rgb,
)


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
