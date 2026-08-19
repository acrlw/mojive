"""Spike A: render parity_scene.xml through wgpu-py, offscreen, and save outputs.

Usage: .venv/bin/python spike/wgpu/spike_offscreen.py
Writes output/spike/wgpu_{rgb,depth,ids}.png / .npy plus a stats line.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge_viewer.adapters.base import FrameNeeds
from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.render.builder import SceneSourceBuilder
from forge_viewer.render.mesh import all_builtin
from forge_viewer.types import CameraView

from backend import WgpuSceneBackend

WIDTH, HEIGHT = 640, 480
OUT = ROOT / "output" / "spike"


def build_scene():
    adapter = MuJoCoAdapter()
    adapter.load(ROOT / "assets" / "parity_scene.xml")
    source = adapter.scene_source()
    frame = adapter.frame(FrameNeeds(poses=True))
    camera = CameraView(
        eye=np.array([2.0, -2.4, 1.6], np.float32),
        target=np.array([0.0, 0.0, 0.3], np.float32),
        aspect=WIDTH / HEIGHT,
    )
    builder = SceneSourceBuilder()
    builder.set_source(source, camera)
    scene = builder.update(frame, camera)
    return adapter, source, scene


def main() -> None:
    adapter, source, scene = build_scene()
    print(f"instances={scene.count} buckets={scene.bucket_count()} meshes={len(source.meshes)}")
    # The raw adapter source carries no object ids (they are assigned by the
    # session layer); encode per-instance ids to validate the uint MRT path.
    scene.object_id = np.arange(1, scene.count + 1, dtype=np.uint32)

    backend = WgpuSceneBackend(WIDTH, HEIGHT)
    print(f"adapter: {backend.adapter_name}")
    backend.set_meshes({**all_builtin(), **source.meshes})

    t0 = time.perf_counter()
    backend.draw_scene(scene)
    rgb = backend.read_color()
    t1 = time.perf_counter()
    print(f"first frame + readback: {(t1 - t0) * 1e3:.1f} ms")

    depth = backend.read_linear_depth()
    ids = backend.read_ids()

    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OUT / "wgpu_rgb.png")
    depth_vis = np.clip(depth / 4.0, 0, 1)
    Image.fromarray((depth_vis * 255).astype(np.uint8)).save(OUT / "wgpu_depth.png")
    ids_vis = (ids % 253).astype(np.uint8)
    Image.fromarray(ids_vis).save(OUT / "wgpu_ids.png")
    np.save(OUT / "wgpu_rgb.npy", rgb)
    np.save(OUT / "wgpu_depth.npy", depth)
    np.save(OUT / "wgpu_ids.npy", ids)

    covered = ids > 0
    print(f"coverage: {covered.mean():.3%} of pixels, {len(np.unique(ids))} unique ids")
    print(f"depth range on geometry: {depth[covered].min():.2f}..{depth[covered].max():.2f} m")
    adapter.release()
    backend.release()


if __name__ == "__main__":
    main()
