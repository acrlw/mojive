from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..assets import resolve
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo scenes with 16 and 24 lights")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/mujoco-many-lights"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    with OffscreenHarness(resolve("many_lights"), args.width, args.height) as harness:
        full = harness.source.lights
        captures: dict[str, np.ndarray] = {}
        for count, name in ((16, "first-16.png"), (len(full.lights), "all-24.png")):
            for index, light in enumerate(full.lights[16:], 16):
                harness.adapter.set_light(index, replace(light, active=index < count))
            harness.step_and_render(0)
            harness.save_png(args.output / name)
            captures[name] = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)

    delta = np.abs(captures["all-24.png"] - captures["first-16.png"])
    print(f"lights: {len(full.lights)}")
    print(f"changed pixels: {np.count_nonzero(np.max(delta, axis=2)):,}")
    for name in captures:
        print((args.output / name).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
