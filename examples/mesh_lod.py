"""Render shared meshes with optional, background-prepared display LOD."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import CameraView, Light, LightSet, Scene, SceneRenderer


def create_scene(side: int = 16) -> Scene:
    scene = Scene(lights=LightSet(lights=(Light(direction=np.array([0.3, 0.5, -1])),)))
    scene.plane(size=(side, side, 0.01), color=(0.22, 0.24, 0.28, 1))
    for y in range(side):
        for x in range(side):
            scene.sphere(
                name=f"sphere_{x}_{y}",
                position=((x - (side - 1) / 2) * 1.5, (y - (side - 1) / 2) * 1.5, 0.5),
                size=(0.5, 0.5, 0.5),
                color=(0.85, 0.25 + 0.4 * y / side, 0.12, 1),
            )
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-lod", action="store_true", help="Explicitly enable native mesh LOD")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("output/examples/mesh-lod"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    scene = create_scene()
    camera = CameraView(eye=np.array([22, -32, 24]), target=np.array([0, 0, 0.5]), far=150)
    with SceneRenderer(
        scene.source, renderer="bgfx", width=960, height=640, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        # No simplification or LOD worker starts unless the caller opts in.
        if args.mesh_lod:
            renderer.set_flag("mesh_lod", True)
        for _ in range(args.frames):
            image = renderer.render()
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.output / "rgb.png")
        # Disabling restores original geometry and frees levels when no other
        # enabled scene uses them. Exact sensor exports should keep LOD off.
        renderer.set_flag("mesh_lod", False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
