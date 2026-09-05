"""Render every named MuJoCo camera through one Mojive Renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse the model, renderer backend, and destination directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/examples/cameras"))
    parser.add_argument(
        "--renderer", "--backend", dest="backend", choices=("opengl", "wgpu"), default="opengl"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def camera_names(model: mujoco.MjModel) -> list[str]:
    """Return non-empty names for all fixed cameras in a model."""
    names = []
    for camera_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        names.append(name or f"camera-{camera_id}")
    return names


def main() -> None:
    """Render the free camera and each named fixed camera to PNG."""
    args = parse_args()
    from mojive import Renderer

    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    args.output.mkdir(parents=True, exist_ok=True)
    cameras: list[int | str] = [-1, *camera_names(model)]
    with Renderer(model, height=args.height, width=args.width, renderer=args.backend) as renderer:
        for camera in cameras:
            renderer.update_scene(data, camera=camera)
            image = renderer.render()
            stem = "free" if camera == -1 else str(camera).replace("/", "-")
            Image.fromarray(image).save(args.output / f"{stem}.png")
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
