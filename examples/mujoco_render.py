"""Render MuJoCo images through the Forge Renderer API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/examples"))
    parser.add_argument("--backend", choices=("forge", "wgpu"), default="forge")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["FORGE_VIEWER_BACKEND"] = args.backend
    from forge_viewer import Renderer

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    args.output.mkdir(parents=True, exist_ok=True)

    with Renderer(model, height=args.height, width=args.width) as renderer:
        renderer.update_scene(data, camera=-1)
        rgb = renderer.render()
        Image.fromarray(rgb).save(args.output / "rgb.png")

        renderer.enable_depth_rendering()
        depth = renderer.render()
        np.save(args.output / "depth.npy", depth)

        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()
        np.save(args.output / "segmentation.npy", segmentation)

    print(args.output.resolve())


if __name__ == "__main__":
    main()
