"""Record a streamed MuJoCo rollout, optionally annotating RGB frames with Pillow."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from mojive import Renderer, VideoRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/examples/rollout.mp4"))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", help="Fixed model camera name; omit for the free view")
    parser.add_argument("--backend", choices=("opengl", "wgpu"))
    parser.add_argument("--pixel-format", choices=("yuv420p", "yuv444p"), default="yuv420p")
    parser.add_argument("--label", default="", help="Add a label and simulation timestamp to RGB")
    args = parser.parse_args()
    if min(args.width, args.height, args.frames) <= 0:
        parser.error("width, height, and frames must be positive")
    if not math.isfinite(args.fps) or args.fps <= 0.0:
        parser.error("fps must be finite and positive")
    return args


def annotate(rgb: np.ndarray, label: str, simulation_time: float) -> np.ndarray:
    """Annotate a copy of RGB; never draw into depth or segmentation arrays."""
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    text = f"{label}  |  t = {simulation_time:.3f} s"
    x0, y0, x1, y1 = draw.textbbox((12, 10), text)
    draw.rectangle((x0 - 5, y0 - 5, x1 + 5, y1 + 5), fill=(24, 28, 32))
    draw.text((12, 10), text, fill=(235, 238, 240))
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    if args.backend:
        os.environ["MOJIVE_BACKEND"] = args.backend
    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = float(data.time)
    with (
        Renderer(model, width=args.width, height=args.height) as renderer,
        VideoRecorder(
            args.output, (args.width, args.height), fps=args.fps, pixel_format=args.pixel_format
        ) as video,
    ):
        for index in range(args.frames):
            # Sample at video FPS, keeping the model's physical timestep unchanged.
            # Use the first physics state at or after each requested sample time.
            target_time = start + index / args.fps
            while data.time + model.opt.timestep * 1e-6 < target_time:
                # Apply policy controls here before each physics step, if needed.
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=args.camera if args.camera is not None else -1)
            rgb = renderer.render()
            if args.label:
                rgb = annotate(rgb, args.label, float(data.time))
            video.append(rgb)
    print(
        f"Saved {video.frames} frames at {video.fps:g} fps, {video.encoded_size}: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
