"""CPU-only texture format tests for the optional WebGPU backend."""

from __future__ import annotations

import pytest

wgpu = pytest.importorskip("wgpu")

import numpy as np  # noqa: E402

from mojive.render.webgpu.textures import TextureStore, _mip_chain  # noqa: E402


@pytest.mark.parametrize("components", (3, 4))
def test_linear_rgb_textures_are_not_decoded_as_srgb(components):
    store = TextureStore.__new__(TextureStore)

    texture_format, bytes_per_pixel, linearize = store._format_for(components, False)

    assert texture_format == wgpu.TextureFormat.rgba8unorm
    assert bytes_per_pixel == 4
    assert not linearize


@pytest.mark.parametrize("components", (3, 4))
def test_srgb_rgb_textures_use_hardware_srgb_decode(components):
    store = TextureStore.__new__(TextureStore)

    texture_format, bytes_per_pixel, linearize = store._format_for(components, True)

    assert texture_format == wgpu.TextureFormat.rgba8unorm_srgb
    assert bytes_per_pixel == 4
    assert not linearize


def test_non_power_of_two_mip_chain_reaches_one_texel():
    pixels = np.full((1, 300, 300, 4), 73, np.uint8)

    levels = _mip_chain(pixels)

    assert [level.shape[1:3] for level in levels] == [
        (300, 300),
        (150, 150),
        (75, 75),
        (37, 37),
        (18, 18),
        (9, 9),
        (4, 4),
        (2, 2),
        (1, 1),
    ]
    assert all(np.all(level == 73) for level in levels)


def test_odd_mip_reduction_includes_last_row_and_column():
    pixels = np.zeros((1, 3, 5, 1), np.uint8)
    pixels[:, -1, :, :] = 120
    pixels[:, :, -1, :] = 240

    levels = _mip_chain(pixels)

    assert levels[1].shape == (1, 1, 2, 1)
    assert np.all(levels[1] > 0)
    assert levels[-1].shape == (1, 1, 1, 1)
