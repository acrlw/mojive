"""Capture MuJoCo convex-hull visualization references."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..assets import resolve
from ..render.backend import RenderFlag
from ..types import CameraView
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo mesh and convex-hull views")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/mujoco-convex-hull"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    with OffscreenHarness(resolve("convex_hull"), args.width, args.height) as harness:
        harness.backend.set_camera(
            CameraView(
                eye=np.array([3.6, -4.8, 4.2], np.float32),
                target=np.zeros(3, np.float32),
                up=np.array([0.0, 0.0, 1.0], np.float32),
                near=0.05,
                far=50.0,
            )
        )
        harness.warmup(4)
        for enabled, name in ((False, "mesh.png"), (True, "convex-hull.png")):
            harness.backend.set_flag(RenderFlag.CONVEXHULL, enabled)
            harness.step_and_render(0)
            harness.save_png(args.output / name)

    for name in ("mesh.png", "convex-hull.png"):
        print((args.output / name).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
