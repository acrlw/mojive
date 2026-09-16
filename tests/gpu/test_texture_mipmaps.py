"""Actual WebGPU uploads retain color, alpha, orientation and complete mip chains."""

import numpy as np
import pytest

from mojive.render.texture import mip_chain, srgb_to_linear_u8
from mojive.types import TextureData, TextureType

pytestmark = pytest.mark.gpu
wgpu = pytest.importorskip("wgpu")


@pytest.mark.parametrize(
    "shape", [(1, 1, 8), (1, 8, 1), (1, 16, 32), (6, 8, 8), (1, 1, 1), (1, 3, 5), (6, 3, 3)]
)
@pytest.mark.parametrize("components", [1, 2, 3, 4])
@pytest.mark.parametrize("srgb", [False, True])
def test_wgpu_texture_mips_match_reference_without_reuploading_power_of_two_levels(
    shape, components, srgb, monkeypatch
):
    from mojive.render.webgpu import textures
    from mojive.render.webgpu.timing import default_device

    device = default_device()
    store = textures.TextureStore(device)
    pixels = np.random.default_rng(71).integers(0, 256, (*shape, components), np.uint8)
    # Opposite alpha and RGB gradients expose accidental alpha-weighted filtering.
    if components == 4:
        pixels[..., 3] = 255 - pixels[..., 0]
    reference = pixels
    if components == 3:
        reference = np.concatenate([pixels, np.full((*shape, 1), 255, np.uint8)], axis=-1)
    if srgb and components < 3:
        reference = srgb_to_linear_u8(reference)
    expected = mip_chain(reference, srgb=srgb and components >= 3)
    _, h, w = shape
    gpu_mips = not (w & (w - 1) or h & (h - 1))
    if gpu_mips:

        def forbidden(*args, **kwargs):
            pytest.fail("Power-of-two upload regenerated its mip chain on the CPU")

        monkeypatch.setattr(textures, "_mip_chain", forbidden)
    created = []
    create_texture = device.create_texture

    def readable_texture(**kwargs):
        kwargs["usage"] |= wgpu.TextureUsage.COPY_SRC
        texture = create_texture(**kwargs)
        created.append(texture)
        return texture

    monkeypatch.setattr(device, "create_texture", readable_texture)
    uploaded = []
    write = store._write_payload

    def record_write(texture, pixels, bpp, level):
        uploaded.append(level)
        write(texture, pixels, bpp, level)

    monkeypatch.setattr(store, "_write_payload", record_write)
    data = TextureData(
        "pattern",
        TextureType.CUBE if shape[0] == 6 else TextureType.TWO_D,
        pixels if shape[0] == 6 else pixels[0],
        srgb=srgb,
    )
    try:
        store.sync({data.name: data})
        assert uploaded == ([0] if gpu_mips else list(range(len(expected))))
        store.sync({data.name: data})
        assert len(created) == 1, "Unchanged source must retain its GPU texture"
        for level, reference in enumerate(expected):
            layers, h, w, bpp = reference.shape
            payload = device.queue.read_texture(
                {"texture": created[0], "mip_level": level},
                {"bytes_per_row": w * bpp, "rows_per_image": h},
                (w, h, layers),
            )
            actual = np.frombuffer(payload, np.uint8).reshape(reference.shape)
            if not gpu_mips or level == 0:
                np.testing.assert_array_equal(actual, reference)
            else:
                np.testing.assert_allclose(actual, reference, atol=2)
    finally:
        store.release()
        for texture in created:
            texture.destroy()


def test_wgpu_mips_filter_light_energy_and_alpha_independently():
    from mojive.render.webgpu.mipmaps import MipmapGenerator
    from mojive.render.webgpu.timing import default_device

    device = default_device()
    texture = device.create_texture(
        size=(2, 1, 1),
        format="rgba8unorm-srgb",
        mip_level_count=2,
        usage=(
            wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.COPY_SRC
        ),
    )
    try:
        device.queue.write_texture(
            {"texture": texture},
            bytes([0, 0, 0, 0, 255, 255, 255, 255]),
            {"bytes_per_row": 8},
            (2, 1, 1),
        )
        MipmapGenerator(device).generate(texture)
        actual = device.queue.read_texture(
            {"texture": texture, "mip_level": 1}, {"bytes_per_row": 4}, (1, 1, 1)
        )
        np.testing.assert_allclose(np.frombuffer(actual, np.uint8), [188, 188, 188, 128], atol=1)
    finally:
        texture.destroy()
