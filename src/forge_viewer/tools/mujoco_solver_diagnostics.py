from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.base import FrameNeeds
from ..assets import resolve
from ..render.backend import RenderFlag
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo solver diagnostic overlays")
    parser.add_argument("-o", "--out", default="output/mujoco-solver-diagnostics")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)
    output = Path(args.out)

    with OffscreenHarness(resolve("solver_diagnostics"), args.width, args.height) as harness:
        camera = next(item for item in harness.adapter.cameras() if item.name == "overview")
        harness.backend.set_camera(harness.adapter.camera_view(camera.camera_id))
        harness.needs = FrameNeeds(poses=True, contacts=True, diagnostics=True)
        harness.backend.set_flag(RenderFlag.CONTACTPOINT, True)
        harness.backend.set_flag(RenderFlag.CONTACTFORCE, True)
        harness.backend.set_flag(RenderFlag.CONTACTSPLIT, True)
        harness.adapter.data.qvel[-6] = 1.5
        harness.step_and_render(60)
        harness.save_png(output / "contact-split.png")

    with OffscreenHarness(resolve("solver_diagnostics"), args.width, args.height) as harness:
        harness.adapter.set_visual_group("geom", 1, False)
        harness.source = harness.adapter.scene_source()
        harness.backend.set_scene(harness.source)
        camera = next(item for item in harness.adapter.cameras() if item.name == "overview")
        harness.backend.set_camera(harness.adapter.camera_view(camera.camera_id))
        harness.needs = FrameNeeds(poses=True, diagnostics=True)
        harness.backend.set_flag(RenderFlag.AUTOCONNECT, True)
        harness.step_and_render(0)
        harness.save_png(output / "autoconnect.png")

    for path in sorted(output.glob("*.png")):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
