"""Compare wgpu spike outputs against the forge EGL reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "spike"


def main() -> None:
    rgb_w = np.load(OUT / "wgpu_rgb.npy")[..., :3].astype(np.int16)
    rgb_r = np.load(OUT / "ref_rgb.npy").astype(np.int16)
    dep_w = np.load(OUT / "wgpu_depth.npy")
    dep_r = np.load(OUT / "ref_depth.npy")
    ids_w = np.load(OUT / "wgpu_ids.npy")
    ids_r = np.load(OUT / "ref_ids.npy")

    mask_w = ids_w > 0
    mask_r = ids_r > 0
    inter = mask_w & mask_r
    union = mask_w | mask_r
    iou = inter.sum() / max(union.sum(), 1)
    id_match = (ids_w[union] == ids_r[union]).mean() if union.any() else float("nan")

    # Depth: ignore the plane bucket (forge renders it analytically, spike skips it)
    valid = inter & (dep_r > 0.05)
    d_err = np.abs(dep_w[valid] - dep_r[valid])
    p95 = np.percentile(d_err, 95) if d_err.size else float("nan")

    mae = np.abs(rgb_w[inter] - rgb_r[inter]).mean() if inter.any() else float("nan")

    print(f"silhouette IoU:      {iou:.4f}")
    print(f"id match on union:   {id_match:.4f}")
    print(f"depth p95 |err| (m): {p95:.4f}")
    print(f"rgb MAE on overlap:  {mae:.1f} / 255  (lighting models differ; sanity only)")

    side = np.concatenate([rgb_w.astype(np.uint8), rgb_r.astype(np.uint8)], axis=1)
    Image.fromarray(side).save(OUT / "compare_rgb.png")
    diff = np.zeros((*ids_w.shape, 3), np.uint8)
    diff[mask_w & ~mask_r] = (255, 0, 0)   # wgpu only
    diff[mask_r & ~mask_w] = (0, 0, 255)   # ref only
    diff[inter] = (40, 40, 40)
    Image.fromarray(diff).save(OUT / "compare_mask.png")
    print(f"artifacts in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
