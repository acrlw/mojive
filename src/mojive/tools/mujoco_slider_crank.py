"""Capture MuJoCo slider-crank visualization references."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.base import FrameNeeds
from ..assets import resolve
from ..render.backend import RenderFlag
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the MuJoCo slider-crank overlay")
    parser.add_argument("-o", "--out", default="output/mujoco-slider-crank.png")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    output = Path(args.out)
    with OffscreenHarness(resolve("slider_crank"), args.width, args.height) as harness:
        harness.needs = FrameNeeds(poses=True, actuator=True, diagnostics=True)
        harness.backend.set_flag(RenderFlag.ACTUATOR, True)
        harness.step_and_render(0)
        harness.save_png(output)

    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
