"""Capture MuJoCo labels, frames, and deformable overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.base import FrameNeeds
from ..assets import resolve
from ..render.backend import FrameMode, LabelMode, RenderFlag
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo debug overlay references")
    parser.add_argument("-o", "--out", default="output/mujoco-overlays")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with OffscreenHarness(resolve("deformables"), args.width, args.height) as harness:
        harness.needs = FrameNeeds(
            poses=True,
            contacts=True,
            tendons=True,
            actuator=True,
            deformables=True,
            diagnostics=True,
        )
        harness.backend.set_flag(RenderFlag.FLEXEDGE, True)
        harness.backend.set_flag(RenderFlag.FLEXVERT, False)
        harness.step_and_render(8)
        harness.save_png(out / "flex-edges.png")

        harness.backend.set_flag(RenderFlag.FLEXVERT, True)
        harness.step_and_render(0)
        harness.save_png(out / "flex-vertices.png")

        harness.backend.set_flag(RenderFlag.FLEXVERT, False)
        harness.backend.set_flag(RenderFlag.FLEXEDGE, False)
        harness.backend.set_label_mode(LabelMode.GEOM)
        harness.step_and_render(0)
        harness.save_png(out / "geom-labels.png")

        harness.backend.set_label_mode(LabelMode.NONE)
        harness.backend.set_frame_mode(FrameMode.GEOM)
        harness.step_and_render(0)
        harness.save_png(out / "geom-frames.png")

    for path in sorted(out.glob("*.png")):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
