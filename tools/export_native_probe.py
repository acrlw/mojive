"""Export a fixed MuJoCo trajectory for the experimental native renderer probe.

The private benchmark stream is deliberately not a new public scene file format.
It reuses Mojive's source builder, mesh tessellation, identities, and camera math.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import mujoco
import numpy as np

from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco_adapter import MuJoCoAdapter
from mojive.render.builder import SceneSourceBuilder
from mojive.render.mesh import builtin_mesh
from mojive.types import CameraView


def main() -> None:
    """Write meshes, identity metadata, and row-major frame matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("output/native-probe/humanoids100.mjvp")
    )
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.frames <= 1000:
        parser.error("frames must be between 1 and 1000")
    adapter = MuJoCoAdapter()
    adapter.load(args.model)
    source = adapter.scene_source()
    camera = CameraView(
        eye=np.array([18, -22, 18], np.float32),
        target=np.array([0, 0, 0.5], np.float32),
        up=np.array([0, 0, 1], np.float32),
        near=0.1,
        far=150,
        aspect=16 / 9,
    )
    builder = SceneSourceBuilder()
    builder.set_source(source, camera)
    scene = builder.update(adapter.frame(FrameNeeds()), camera)
    keys = list(dict.fromkeys(key for key, _ in scene.bucket_keys))
    meshes = [source.meshes[key] if key in source.meshes else builtin_mesh(key) for key in keys]
    mesh_indices = {key: i for i, key in enumerate(keys)}
    slots = np.array([mesh_indices[scene.bucket_keys[int(b)][0]] for b in scene.bucket], np.uint32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"MJVPROB1")
        stream.write(struct.pack("<III", len(meshes), scene.count, args.frames))
        stream.write(camera.view_matrix().astype("<f4").tobytes())
        stream.write(camera.proj_matrix().astype("<f4").tobytes())
        stream.write(struct.pack("<f", camera.far))
        for mesh in meshes:
            stream.write(struct.pack("<II", len(mesh.positions), len(mesh.indices)))
            stream.write(np.column_stack((mesh.positions, mesh.normals)).astype("<f4").tobytes())
            stream.write(mesh.indices.astype("<u4").tobytes())
        for i in range(scene.count):
            stream.write(
                struct.pack(
                    "<IIii4f",
                    int(slots[i]),
                    int(scene.object_id[i]),
                    *map(int, scene.segmentation[i]),
                    *map(float, scene.colors[i]),
                )
            )
        for _ in range(args.frames):
            scene = builder.update(adapter.frame(FrameNeeds()), camera)
            stream.write(scene.transforms.astype("<f4").tobytes())
            mujoco.mj_step(adapter.model, adapter.data)
    metadata = {
        "model": str(args.model.resolve()),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "moving_bodies": adapter.model.nbody - 1,
        "degrees_of_freedom": adapter.model.nv,
        "geoms": adapter.model.ngeom,
        "render_instances": scene.count,
        "meshes": len(meshes),
        "frames": args.frames,
        "timestep": adapter.model.opt.timestep,
        "camera_eye": camera.eye.tolist(),
        "camera_target": camera.target.tolist(),
        "scope": "Fixed trajectory; native renderer excludes production shadows, reflections, and textures",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
