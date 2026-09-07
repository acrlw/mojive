"""In-memory image encoding retains exact array content and metadata."""

import numpy as np
import pytest

from mojive.capture import decode_image, encode_image


@pytest.mark.parametrize("encoding", ["raw", "npy", "png"])
def test_rgb_capture_roundtrip(encoding):
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[::-1]
    payload = encode_image(image, encoding)
    restored = decode_image(payload)
    np.testing.assert_array_equal(restored, image)
    assert restored.flags.owndata
    assert payload["orientation"] == "top_left"


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<i4", "<u4"])
@pytest.mark.parametrize("encoding", ["raw", "npy"])
def test_numeric_capture_preserves_dtype_and_byte_order(dtype, encoding):
    image = np.arange(40).astype(dtype).reshape(4, 5, 2)
    restored = decode_image(encode_image(image, encoding))
    assert restored.dtype == image.dtype
    np.testing.assert_array_equal(restored, image)


def test_png_rejects_metric_depth():
    with pytest.raises(ValueError, match="RGB"):
        encode_image(np.ones((3, 4), np.float32), "png")
