"""Capture and time a limited hinge while its ring fades through an edge-on view.

Run separately from other GPU workloads. Timings measure application frame work
and CPU overlay drawing, not display scanout or physical input latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build
from mojive.config import LayoutConfig, ViewerConfig
from mojive.interaction.gizmo import rotation_ring_alpha
from mojive.scene.assets import resolve
from mojive.types import CameraView


def _summary(samples):
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": float(max(samples)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="joint_gizmo")
    parser.add_argument("--link", default="01_revolute")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--distance", type=float, default=0.6)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/joint-gizmo-profile"))
    args = parser.parse_args(argv)
    if args.frames < 2 or args.distance <= 0:
        parser.error("frames must be at least two and distance must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with build(
        resolve(args.asset),
        paused=True,
        vsync=False,
        show_window=args.visible,
        width=1400,
        height=900,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        node = next(node for node in viewer.session.nodes if node.name == args.link)
        assert viewer.session.submit(cmd.Select(node.object_id))
        viewer.set_gizmo_mode("rotate")
        for _ in range(12):
            viewer.sync()
        gizmo = viewer.app.gizmo
        if gizmo._joint_range is None or gizmo._joint_range.joint_type != "hinge":
            raise ValueError("Select a link with one limited hinge joint")
        origin = gizmo._frame.position.copy()
        axis = gizmo._frame.rotation[:, 2].copy()
        radial = gizmo._frame.rotation[:, 0].copy()
        up = np.cross(axis, radial)

        def camera(facing):
            direction = radial * np.sqrt(1 - facing * facing) + axis * facing
            return CameraView(
                eye=origin + direction * args.distance,
                target=origin.copy(),
                up=up,
                near=0.001,
                far=100,
            )

        overlay_ms = []
        draw_overlay = gizmo.draw_overlay

        def timed_overlay(*args, **kwargs):
            start = time.perf_counter_ns()
            result = draw_overlay(*args, **kwargs)
            overlay_ms.append((time.perf_counter_ns() - start) / 1e6)
            return result

        gizmo.draw_overlay = timed_overlay
        rows = []
        for facing in (0.5, 0.18, 0.12, 0.09, 0.06, -0.12):
            for moving in (False, True):
                viewer.set_camera(camera(facing))
                for _ in range(6):
                    viewer.sync()
                overlay_ms.clear()
                frame_ms = []
                for index in range(args.frames):
                    # Unique projections exercise cache misses during an orbit.
                    if moving:
                        viewer.set_camera(camera(facing + 0.004 * index / args.frames - 0.002))
                    start = time.perf_counter_ns()
                    viewer.sync()
                    frame_ms.append((time.perf_counter_ns() - start) / 1e6)
                if len(overlay_ms) != args.frames:
                    raise RuntimeError("The selected joint overlay was not drawn every frame")
                row = {
                    "facing": facing,
                    "moving": moving,
                    "alpha": rotation_ring_alpha(camera(facing), origin, axis),
                    "frame": _summary(frame_ms),
                    "overlay": _summary(overlay_ms),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            viewer.set_camera(camera(facing))
            for _ in range(3):
                viewer.sync()
            Image.fromarray(viewer.capture_array(surface="window")).save(
                args.output / f"facing-{facing:.2f}.png"
            )
        report = {
            "asset": str(resolve(args.asset)),
            "link": node.name,
            "backend": viewer.backend.caps.name,
            "visible": args.visible,
            "vsync": False,
            "window_pixels": list(viewer.window.size_pixels),
            "frames_per_case": args.frames,
            "cases": rows,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
