from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.base import FrameNeeds
from ..assets import resolve
from ..render.backend import RenderFlag
from ._harness import OffscreenHarness


def _show_all_volumes(adapter) -> None:
    adapter.model.vis.global_.bvactive = 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo bounding-volume hierarchies")
    parser.add_argument("-o", "--out", default="output/mujoco-bvh")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)
    output = Path(args.out)

    with OffscreenHarness(resolve("joint_types"), args.width, args.height) as harness:
        camera = next(item for item in harness.adapter.cameras() if item.name == "joints")
        harness.backend.set_camera(harness.adapter.camera_view(camera.camera_id))
        harness.needs = FrameNeeds(poses=True, diagnostics=True, bvh=True)
        harness.backend.set_flag(RenderFlag.BODYBVH, True)
        harness.backend.set_bvh_depth(1)
        harness.step_and_render(0)
        harness.save_png(output / "body.png")

    with OffscreenHarness(
        resolve("dense_mesh"),
        args.width,
        args.height,
        configure_adapter=_show_all_volumes,
    ) as harness:
        camera = next(item for item in harness.adapter.cameras() if item.name == "dense")
        harness.backend.set_camera(harness.adapter.camera_view(camera.camera_id))
        harness.needs = FrameNeeds(poses=True, diagnostics=True, bvh=True)
        harness.backend.set_flag(RenderFlag.MESHBVH, True)
        harness.backend.set_bvh_depth(2)
        harness.step_and_render(0)
        harness.save_png(output / "mesh.png")

    with OffscreenHarness(resolve("deformables"), args.width, args.height) as harness:
        harness.needs = FrameNeeds(poses=True, deformables=True, diagnostics=True, bvh=True)
        harness.backend.set_flag(RenderFlag.MESHBVH, True)
        harness.backend.set_bvh_depth(2)
        harness.step_and_render(1)
        harness.save_png(output / "flex.png")

    for path in sorted(output.glob("*.png")):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
