"""Render an authored scene to RGB, metric depth, and object-ID images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import CameraView, RenderProduct, Scene, SceneRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("opengl", "wgpu"), default=None)
    parser.add_argument("--output", type=Path, default=Path("output/scene-renderer"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    scene.plane(size=(3, 3, 0.01), color=(0.6, 0.65, 0.7, 1))
    scene.box(name="box", position=(0, -0.65, 0.5), color=(0.1, 0.5, 0.85, 1))
    scene.sphere(name="sphere", position=(0, 0.65, 0.5), color=(0.95, 0.35, 0.12, 1))
    camera = CameraView(eye=np.array([4, -4, 3]), target=np.array([0, 0, 0.4]))
    with SceneRenderer(scene.source, width=640, height=480, renderer=args.renderer) as renderer:
        renderer.update(scene.frame, camera=camera)
        Image.fromarray(renderer.render()).save(args.output / "rgb.png")
        depth = renderer.render(product=RenderProduct.METRIC_DEPTH)
        ids = renderer.render(product=RenderProduct.OBJECT_ID)
    np.save(args.output / "depth.npy", depth)
    np.save(args.output / "object-id.npy", ids)
    # Picking coverage may include MSAA edge pixels whose depth is background.
    visible = (ids != 0) & np.isfinite(depth) & (depth < camera.far * (1.0 - 1e-5))
    depth_image = np.zeros(ids.shape, np.uint8)
    if visible.any():
        low, high = depth[visible].min(), depth[visible].max()
        depth_image[visible] = 255 - (200 * (depth[visible] - low) / max(high - low, 1e-6)).astype(
            np.uint8
        )
    Image.fromarray(depth_image).save(args.output / "depth.png")
    palette = np.array([[0, 0, 0], [110, 120, 135], [45, 150, 235], [245, 120, 40]], np.uint8)
    Image.fromarray(palette[ids % len(palette)]).save(args.output / "object-id.png")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
