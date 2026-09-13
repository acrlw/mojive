"""Backend-neutral CPU texture mip generation."""

import numpy as np


def srgb_to_linear_u8(pixels: np.ndarray) -> np.ndarray:
    """Decode RGB channels to the shared eight-bit linear fallback, preserving alpha."""
    out = pixels.astype(np.float32) / 255.0
    rgb = out[..., :3]
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    out[..., :3] = linear
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def box_reduce_axis(pixels: np.ndarray, axis: int) -> np.ndarray:
    """Area-average one image axis to the next legal WebGPU mip extent."""

    size = pixels.shape[axis]
    target = max(1, size // 2)
    if size == target:
        return pixels
    moved = np.moveaxis(pixels, axis, 0)
    if size == target * 2:
        return np.moveaxis(moved.reshape(target, 2, *moved.shape[1:]).mean(axis=1), 0, axis)

    # Odd, non-power-of-two dimensions cannot be reshaped into 2x blocks.
    # Use exact source-pixel coverage so the final row/column is included
    # instead of truncating it or prematurely ending the mip chain.
    edges = np.linspace(0.0, float(size), target + 1, dtype=np.float32)
    source_lo = np.arange(size, dtype=np.float32)
    weights = np.maximum(
        0.0,
        np.minimum(edges[1:, None], source_lo[None, :] + 1.0)
        - np.maximum(edges[:-1, None], source_lo[None, :]),
    )
    weights /= float(size) / float(target)
    reduced = np.tensordot(weights, moved, axes=((1,), (0,)))
    return np.moveaxis(reduced, 0, axis)


def mip_chain(pixels: np.ndarray, *, srgb: bool = False) -> list[np.ndarray]:
    """Complete box-filtered mip chain for a (layers, h, w, comps) u8 array."""

    levels = [np.ascontiguousarray(pixels)]
    while levels[-1].shape[1] > 1 or levels[-1].shape[2] > 1:
        level = levels[-1].astype(np.float32)
        if srgb:
            rgb = level[..., :3] / 255
            level[..., :3] = (
                np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4) * 255
            )
        level = box_reduce_axis(level, 1)
        level = box_reduce_axis(level, 2)
        if srgb:
            rgb = level[..., :3] / 255
            level[..., :3] = (
                np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055) * 255
            )
        levels.append(np.ascontiguousarray(np.clip(level + 0.5, 0.0, 255.0).astype(np.uint8)))
    return levels
