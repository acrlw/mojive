from __future__ import annotations

import argparse
from pathlib import Path

from ..assets import resolve
from ..render.backend import RenderFlag
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture standard and additive transparency")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/additive"))
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    with OffscreenHarness(resolve("transparency"), args.width, args.height) as harness:
        harness.warmup(4)
        for enabled, name in ((False, "standard.png"), (True, "additive.png")):
            harness.backend.set_flag(RenderFlag.ADDITIVE, enabled)
            harness.step_and_render(0)
            harness.save_png(args.output / name)

    for name in ("standard.png", "additive.png"):
        print((args.output / name).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
