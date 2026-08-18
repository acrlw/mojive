"""Color-space conversion, tone mapping, and output transforms."""

from __future__ import annotations

import numpy as np

GAMMA = 2.2

KNEE = 0.8


EXPOSURE = 1.0


AMBIENT_GAIN = 1.0


def srgb_to_linear(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 0.0) ** (1.0 / 2.4) - 0.055)


def gamma_encode(x):
    return np.maximum(np.asarray(x, np.float64), 0.0) ** (1.0 / GAMMA)


def gamma_decode(x):
    return np.maximum(np.asarray(x, np.float64), 0.0) ** GAMMA


def ambient_linear(ambient, gain: float | None = None):
    a = np.asarray(ambient, np.float64)
    k = AMBIENT_GAIN if gain is None else float(gain)
    return srgb_to_linear(np.clip(k * a, 0.0, 1.0))


def softroll(excess, headroom):
    excess = np.maximum(np.asarray(excess, np.float64), 0.0)
    return headroom * excess / (excess + headroom)


def tonemap(rgb, knee: float = KNEE):
    rgb = np.asarray(rgb, np.float64)
    peak = rgb.max(axis=-1)
    headroom = 1.0 - knee
    mapped = knee + softroll(peak - knee, headroom)

    scale = np.where(peak > knee, mapped / np.maximum(peak, 1e-12), 1.0)
    return rgb * scale[..., None]


def finish(rgb, exposure: float = EXPOSURE, tonemap_on: bool = True):
    out = np.asarray(rgb, np.float64) * float(exposure)
    out = tonemap(out) if tonemap_on else np.clip(out, 0.0, 1.0)
    return gamma_encode(np.clip(out, 0.0, 1.0))


def to_u8(x) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, np.float64) * 255.0), 0, 255).astype(np.uint8)


def shade_flat(albedo_linear, ambient_display, exposure: float = EXPOSURE) -> np.ndarray:
    albedo = np.asarray(albedo_linear, np.float64)
    return to_u8(finish(ambient_linear(ambient_display) * albedo, exposure))
