"""Spike reference: render the same scene/camera through forge (moderngl/EGL).

Requires the EGL vendor override on this machine. Saves ref_{rgb,depth,ids}
alongside the wgpu outputs for compare.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge_viewer.adapters.base import FrameNeeds
from forge_viewer.renderer import _StandaloneContext, _metric_depth
from forge_viewer.render.forge.backend import ForgeBackend

from spike_offscreen import HEIGHT, OUT, WIDTH, build_scene


def main() -> None:
    adapter, source, scene = build_scene()
    scene.object_id = np.arange(1, scene.count + 1, dtype=np.uint32)
    camera = scene.camera

    context = _StandaloneContext("egl")
    with context.current():
        backend = ForgeBackend(context.gl_context, WIDTH, HEIGHT, samples=0)
        backend.set_background((0.13, 0.14, 0.16, 1.0))
        backend.set_scene(source)
        backend.set_camera(camera)
        backend.update(adapter.frame(FrameNeeds(poses=True)))
        # Match the spike's per-instance id encoding; render() re-uploads instances.
        backend._scene.object_id = scene.object_id
        backend.render()
        rgb = np.ascontiguousarray(backend.target.read_color(flip=True)[..., :3])
        depth = _metric_depth(backend.target.read_depth(flip=True), camera)
        ids = backend.target.read_ids(flip=True)
        backend.release()
    context.close()
    adapter.release()

    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUT / "ref_rgb.png")
    Image.fromarray((np.clip(depth / 4.0, 0, 1) * 255).astype(np.uint8)).save(OUT / "ref_depth.png")
    Image.fromarray((ids % 253).astype(np.uint8)).save(OUT / "ref_ids.png")
    np.save(OUT / "ref_rgb.npy", rgb)
    np.save(OUT / "ref_depth.npy", depth)
    np.save(OUT / "ref_ids.npy", ids)
    covered = ids > 0
    print(f"ref coverage: {covered.mean():.3%}, ids: {sorted(np.unique(ids).tolist())}")
    if covered.any():
        print(f"ref depth range on geometry: {depth[covered].min():.2f}..{depth[covered].max():.2f} m")


if __name__ == "__main__":
    main()
