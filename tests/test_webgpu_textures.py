"""CPU-only texture format tests for the optional WebGPU backend."""

from __future__ import annotations

import pytest

wgpu = pytest.importorskip("wgpu")

from forge_viewer.render.webgpu.textures import TextureStore  # noqa: E402


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
