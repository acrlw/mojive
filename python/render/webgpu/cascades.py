"""Cascaded shadow-map layout and camera fitting for wgpu."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import math3d as M
from .targets import orthographic_wgpu

ATLAS_SIZE = 4096

ATLAS_COLS = 2
ATLAS_ROWS = 2
TILE_PIXELS = ATLAS_SIZE // ATLAS_COLS

SLOT_COUNT = ATLAS_COLS * ATLAS_ROWS
CASCADE_COUNT = 3

RADIUS_DIVISORS: tuple[float, ...] = (9.0, 3.0, 1.0)


PCF_RADIUS = 1


DEFAULT_SHADOW_CLIP = 1.0


def slot_pixels(slot: int) -> tuple[int, int, int, int]:
    if not 0 <= slot < SLOT_COUNT:
        raise ValueError(f"cascade slot {slot} exceeds atlas capacity {SLOT_COUNT}")
    col, row = slot % ATLAS_COLS, slot // ATLAS_COLS
    return col * TILE_PIXELS, row * TILE_PIXELS, TILE_PIXELS, TILE_PIXELS


def slot_uv(slot: int, pcf_radius: int = PCF_RADIUS) -> np.ndarray:
    x, y, w, h = slot_pixels(slot)
    m = float(pcf_radius) + 0.5
    return np.array(
        [
            (x + m) / ATLAS_SIZE,
            (y + m) / ATLAS_SIZE,
            (x + w - m) / ATLAS_SIZE,
            (y + h - m) / ATLAS_SIZE,
        ],
        np.float32,
    )


def light_basis(direction) -> np.ndarray:
    f = M.normalize(direction).astype(np.float64)
    if float(np.dot(f, f)) < 0.5:
        f = np.array([0.0, 0.0, -1.0])

    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([s, u, -f], np.float32)


def cascade_radii(
    scene_extent: float,
    shadow_clip: float = DEFAULT_SHADOW_CLIP,
    radius_divisors: tuple[float, ...] = RADIUS_DIVISORS,
) -> np.ndarray:
    r = max(float(scene_extent), 1e-6) * max(float(shadow_clip), 1e-6)
    return np.array([r / d for d in radius_divisors], np.float32)


def snap_to_texel(center, basis: np.ndarray, texel: float) -> np.ndarray:
    c = basis.astype(np.float64) @ np.asarray(center, np.float64)
    c = np.floor(c / texel) * texel
    return (basis.astype(np.float64).T @ c).astype(np.float32)


@dataclass
class CascadeSet:
    matrices: np.ndarray = field(
        default_factory=lambda: np.zeros((CASCADE_COUNT, 4, 4), np.float32)
    )

    splits: np.ndarray = field(default_factory=lambda: np.zeros(CASCADE_COUNT, np.float32))

    texel_world: np.ndarray = field(default_factory=lambda: np.zeros(CASCADE_COUNT, np.float32))

    tile_uv: np.ndarray = field(default_factory=lambda: np.zeros((CASCADE_COUNT, 4), np.float32))

    centers: np.ndarray = field(default_factory=lambda: np.zeros((CASCADE_COUNT, 3), np.float32))

    basis: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    slots: tuple[int, ...] = tuple(range(CASCADE_COUNT))
    count: int = 0


def build_cascades(
    light_direction,
    focus,
    scene_extent: float,
    scene_center=None,
    shadow_clip: float = DEFAULT_SHADOW_CLIP,
    slots: tuple[int, ...] = tuple(range(CASCADE_COUNT)),
    tile_pixels: int = TILE_PIXELS,
    pcf_radius: int = PCF_RADIUS,
    radius_divisors: tuple[float, ...] = RADIUS_DIVISORS,
    into: CascadeSet | None = None,
) -> CascadeSet:
    out = into if into is not None else CascadeSet()
    basis = light_basis(light_direction)
    radii = cascade_radii(scene_extent, shadow_clip, radius_divisors)
    focus = np.asarray(focus, np.float64).reshape(3)
    center_w = (
        np.asarray(scene_center, np.float64).reshape(3) if scene_center is not None else focus
    )
    n = min(len(slots), CASCADE_COUNT)

    out.basis = basis
    out.slots = tuple(int(s) for s in slots[:n])
    out.count = n

    for i in range(n):
        r = float(radii[i])
        texel = 2.0 * r / max(int(tile_pixels), 1)
        center = snap_to_texel(focus, basis, texel)

        half = float(scene_extent) + float(np.linalg.norm(center.astype(np.float64) - center_w)) + r
        half = max(half, 1e-4)

        eye = center.astype(np.float64) - M.normalize(light_direction).astype(np.float64) * half
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = basis
        view[:3, 3] = -(basis.astype(np.float64) @ eye)
        # WebGPU clip z in [0, 1]; the square box (height 2r, aspect 1) matches
        # opengl's ortho_box(-r, r, -r, r, 0, 2*half) apart from the z row.
        proj = orthographic_wgpu(2.0 * r, 1.0, 0.0, 2.0 * half).astype(np.float64)

        out.matrices[i] = (proj @ view).astype(np.float32)
        out.splits[i] = r
        out.texel_world[i] = texel
        out.tile_uv[i] = slot_uv(out.slots[i], pcf_radius)
        out.centers[i] = center

    for i in range(n, CASCADE_COUNT):
        out.matrices[i] = 0.0
        out.splits[i] = 0.0
        out.texel_world[i] = 0.0
        out.tile_uv[i] = 0.0
        out.centers[i] = 0.0
    return out
