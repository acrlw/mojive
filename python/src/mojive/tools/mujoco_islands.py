"""Capture MuJoCo constraint-island visualization references."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.base import FrameNeeds
from ..assets import resolve
from ..render.backend import RenderFlag
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo constraint-island colors")
    parser.add_argument("-o", "--out", default="output/mujoco-islands")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)
    output = Path(args.out)

    with OffscreenHarness(resolve("joint_types"), args.width, args.height) as harness:
        camera = next(item for item in harness.adapter.cameras() if item.name == "joints")
        harness.backend.set_camera(harness.adapter.camera_view(camera.camera_id))
        harness.needs = FrameNeeds(
            poses=True,
            contacts=True,
            tendons=True,
            deformables=True,
            islands=True,
        )
        harness.backend.set_flag(RenderFlag.CONTACTPOINT, True)
        harness.step_and_render(1)
        harness.save_png(output / "materials.png")

        harness.backend.set_flag(RenderFlag.ISLAND, True)
        harness.step_and_render(0)
        harness.save_png(output / "islands.png")

    for path in sorted(output.glob("*.png")):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
