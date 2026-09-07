"""Shared image ownership, view lifetime, validation, and independent processes."""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from mojive import SharedImage


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.uint32, np.int32])
def test_shared_views_reuse_pixels_and_outlive_closed_handles(dtype):
    owner = SharedImage((7, 11), dtype)
    descriptor = owner.descriptor
    retained = owner.array
    with SharedImage.attach(descriptor) as other:
        other.array[:] = 7
        np.testing.assert_array_equal(retained, 7)
        retained[2, 3] = 9
        assert other.array[2, 3] == 9
    owner.close()
    np.testing.assert_array_equal(retained[2, 3], 9)
    with pytest.raises(RuntimeError, match="closed"):
        _ = owner.array
    if os.name != "nt":
        with pytest.raises(FileNotFoundError):
            SharedImage.attach(descriptor)
    owner.close()


@pytest.mark.parametrize(
    "shape,dtype", [((0, 3), np.uint8), ((3,), np.uint8), ((2, 3), object), ((2, 3), np.float64)]
)
def test_invalid_image_formats_are_rejected(shape, dtype):
    with pytest.raises(ValueError):
        SharedImage(shape, dtype)


def test_mapping_rejects_an_oversized_descriptor_without_touching_pixels():
    with SharedImage((2, 3)) as owner:
        owner.array.fill(41)
        with pytest.raises(ValueError, match="smaller"):
            SharedImage.attach({**owner.descriptor, "shape": [1000, 1000]})
        np.testing.assert_array_equal(owner.array, 41)


def test_pickle_transfers_a_descriptor_without_pixels_or_ownership():
    with SharedImage((480, 640, 3)) as owner:
        encoded = pickle.dumps(owner)
        assert len(encoded) < 512
        with pickle.loads(encoded) as attached:
            attached.array.fill(19)
        np.testing.assert_array_equal(owner.array, 19)
        with SharedImage.attach(owner.descriptor) as again:
            np.testing.assert_array_equal(again.array, 19)


@pytest.mark.integration
def test_independent_process_exit_does_not_unlink_owner_memory():
    with SharedImage((7, 11), np.float32) as owner:
        script = """
import json, sys
from mojive import SharedImage
with SharedImage.attach(json.loads(sys.argv[1])) as image:
    image.array.fill(17)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps(owner.descriptor)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        assert "resource_tracker" not in result.stderr
        np.testing.assert_array_equal(owner.array, 17)
        with SharedImage.attach(owner.descriptor) as again:
            np.testing.assert_array_equal(again.array, 17)
