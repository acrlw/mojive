"""Run caller-owned physics and an independently scheduled passive window."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from mojive import launch_passive
from mojive.assets import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="joint_types")
    parser.add_argument("--physics-hz", type=float, default=1000)
    parser.add_argument("--display-fps", type=float, default=60)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--renderer", choices=("opengl", "wgpu"))
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("output/passive-viewer"))
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            args.physics_hz,
            args.display_fps,
            args.seconds,
        )
    ):
        parser.error("frequencies and duration must be finite and positive")

    model = mujoco.MjModel.from_xml_path(str(resolve(args.model)))
    model.opt.timestep = 1.0 / args.physics_hz
    data = mujoco.MjData(model)
    args.output.mkdir(parents=True, exist_ok=True)
    gaps = []
    steps = 0
    with launch_passive(
        model,
        data,
        max_fps=args.display_fps,
        renderer=args.renderer,
        width=args.width,
        height=args.height,
        show_window=not args.hidden,
    ) as viewer:
        before = viewer.stats
        started = previous = due = time.perf_counter()
        while viewer.is_running() and time.perf_counter() - started < args.seconds:
            with viewer.lock():
                # Replace this vector assignment with policy(observation).
                data.ctrl[:] = 0.15 * math.sin(data.time * 2.0)
                mujoco.mj_step(model, data)
                steps += 1
                viewer.sync(step=steps)
            now = time.perf_counter()
            gaps.append(now - previous)
            previous = now
            due += model.opt.timestep
            time.sleep(max(0.0, due - time.perf_counter()))
        elapsed = time.perf_counter() - started
        after = viewer.stats
        image = viewer.capture_array()
        Image.fromarray(image).save(args.output / "scene.png")
        window = viewer.capture_array(surface="window")
        Image.fromarray(window).save(args.output / "window.png")
        # Rendering continues even while the caller deliberately stops publishing.
        time.sleep(0.2)
        idle = viewer.stats
        expected_time = steps * model.opt.timestep
        if not math.isclose(data.time, expected_time, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError("The passive viewer changed the caller's physics clock")
        if idle["rendered_frames"] <= after["rendered_frames"]:
            raise RuntimeError("Display stopped while the physics owner was idle")
    if viewer.is_running():
        raise RuntimeError("The passive viewer did not close")
    report = {
        "model": str(resolve(args.model)),
        "renderer": args.renderer or "environment default",
        "requested_physics_hz": args.physics_hz,
        "requested_display_fps": args.display_fps,
        "physics_steps": steps,
        "elapsed_seconds": elapsed,
        "physics_hz": steps / elapsed,
        "display_fps": (after["rendered_frames"] - before["rendered_frames"]) / elapsed,
        "status_physics_hz": after["physics_hz"],
        "physics_gap_p99_ms": float(np.percentile(gaps, 99) * 1000),
        "simulation_time": float(data.time),
        "expected_time": expected_time,
        "display_progress_while_idle": idle["rendered_frames"] - after["rendered_frames"],
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "closed": not viewer.is_running(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
