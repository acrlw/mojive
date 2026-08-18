"""Capture inverse-kinematics acceptance scenes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..adapters.base import FrameNeeds, IkOptions
from ..assets import resolve
from ._harness import OffscreenHarness

SCENARIOS = (
    ("arm", "ik_arm", "hand", np.array([1.2, 0.8, 0.3])),
    ("quadruped-leg", "ik_quadruped_leg", "foot", np.array([0.55, 0.35, 0.5])),
    ("human-chain", "ik_human_chain", "toe", np.array([0.35, 0.12, 0.1])),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture MuJoCo inverse-kinematics scenes")
    parser.add_argument("-o", "--out", type=Path, default=Path("output/mujoco-ik"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, asset, site_name, target in SCENARIOS:
        with OffscreenHarness(resolve(asset), args.width, args.height) as harness:
            harness.needs = FrameNeeds(poses=True)
            harness.step_and_render(0)
            harness.save_png(args.out / f"{name}-before.png")
            model = harness.adapter.model
            import mujoco

            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            node = next(item for item in harness.adapter.nodes() if item.site_index == site)
            result = harness.adapter.solve_ik(
                node.node_id,
                target,
                np.eye(3),
                IkOptions(max_iterations=128, tolerance=2e-4),
            )
            harness.step_and_render(0)
            harness.save_png(args.out / f"{name}-after.png")
            report[name] = asdict(result)
            if not result.converged:
                raise RuntimeError(f"{name} IK failed: {result}")

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
