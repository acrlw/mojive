"""Capture material, transparency, tendon, deformable, and dense-scene references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..adapters.base import FrameNeeds
from ..assets import resolve
from ..render.backend import RenderFlag
from ._harness import OffscreenHarness
from .golden import compare, side_by_side

CASES = (
    ("parity_texture", "texuniform-boxes.png", FrameNeeds(poses=True), ()),
    ("transparency", "transparent-stack.png", FrameNeeds(poses=True), ()),
    (
        "mujoco_visuals",
        "tendon-heightfield.png",
        FrameNeeds(poses=True, tendons=True, actuator=True),
        (RenderFlag.TENDON, RenderFlag.ACTUATOR),
    ),
    (
        "deformables",
        "deformables.png",
        FrameNeeds(poses=True, deformables=True),
        (RenderFlag.FLEXFACE, RenderFlag.SKIN),
    ),
    ("dense_mesh", "dense-model.png", FrameNeeds(poses=True), ()),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/material-parity"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--golden-dir", type=Path, default=Path("tests/golden/material-parity"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for asset, filename, needs, flags in CASES:
        with OffscreenHarness(resolve(asset), args.width, args.height) as harness:
            harness.needs = needs
            for flag in flags:
                harness.backend.set_flag(flag, True)
            harness.warmup(4)
            harness.step_and_render(4)
            image_path = args.output / filename
            harness.save_png(image_path)
            stats = harness.stats()
            image = np.asarray(Image.open(image_path).convert("RGB"))
            reference_path = args.golden_dir / filename
            reference = (
                np.asarray(Image.open(reference_path).convert("RGB"))
                if reference_path.exists()
                else None
            )
            if args.accept:
                side_by_side(reference, image, args.output / f"{Path(filename).stem}.review.png")
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image, "RGB").save(reference_path)
                passed, note = True, "baseline updated"
            elif reference is None:
                passed, note = False, "missing baseline"
            else:
                passed, note = compare(reference, image)
                if not passed:
                    side_by_side(reference, image, args.output / f"{Path(filename).stem}.diff.png")
            if not passed:
                failures.append(asset)
            report[asset] = {
                "image": filename,
                "draw_calls": stats.draw_calls,
                "instances": stats.instances,
                "triangles": stats.triangles,
                "buckets": stats.buckets,
                "golden": note,
                "passed": passed,
            }

    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for path in sorted(args.output.iterdir()):
        print(path.resolve())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
