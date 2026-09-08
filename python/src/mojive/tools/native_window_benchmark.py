"""Measure real-window submission cadence and same-frame camera publication.

These CPU timings are not physical input-to-photon latency or displayed FPS.
Run without other GPU workloads; OS composition and refresh policy remain active.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mojive.composition import build
from mojive.config import LayoutConfig, ViewerConfig


def run(asset, output, backends, seconds, repeats, width, height):
    from mojive.ui import window as window_module

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in backends:
        with build(
            asset,
            renderer=backend,
            paused=True,
            width=width,
            height=height,
            vsync=True,
            show_window=True,
            config=ViewerConfig(layout=LayoutConfig(persistence=False)),
        ) as viewer:
            glfw = window_module.glfw
            glfw.focus_window(viewer.window._window)
            publish = viewer.backend.set_camera
            published = None

            def record_camera(camera, sink=publish):
                nonlocal published
                published = camera
                return sink(camera)

            viewer.backend.set_camera = record_camera
            for enabled in (True, False):
                viewer.window.set_vsync(enabled)
                for _ in range(45):
                    viewer.sync()
                for repeat in range(repeats):
                    durations = []
                    deadline = time.perf_counter() + seconds
                    while time.perf_counter() < deadline:
                        published = None
                        start = time.perf_counter()
                        viewer.app.camera.pan(
                            0.25 if len(durations) % 2 == 0 else -0.25,
                            0,
                            viewer.backend.target.height,
                        )
                        viewer.sync()
                        durations.append(time.perf_counter() - start)
                        if published is None or not np.allclose(
                            published.view_matrix(),
                            viewer.app.camera.view().view_matrix(),
                        ):
                            raise AssertionError(
                                "Camera changes must reach rendering in the same frame"
                            )
                    row = {
                        "backend": backend,
                        "vsync": enabled,
                        "repeat": repeat,
                        "window_points": list(viewer.window.size_points),
                        "viewport_pixels": [
                            viewer.backend.target.width,
                            viewer.backend.target.height,
                        ],
                        "frames": len(durations),
                        "submission_fps": len(durations) / sum(durations),
                        "frame_ms": {
                            "median": float(np.median(durations) * 1000),
                            "p95": float(np.percentile(durations, 95) * 1000),
                            "maximum": float(max(durations) * 1000),
                        },
                        "same_frame_camera": True,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            Image.fromarray(viewer.capture_array(surface="window")).save(output / f"{backend}.png")
    result = {
        "scope": "CPU submission cadence, not physical input-to-photon latency or displayed FPS",
        "rows": rows,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=Path("assets/test_scene.xml"))
    parser.add_argument("--output", type=Path, default=Path("output/native-window-benchmark"))
    parser.add_argument(
        "--backends", nargs="+", choices=("bgfx", "opengl"), default=["bgfx", "opengl"]
    )
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats <= 0:
        parser.error("Duration and repeat count must be positive")
    run(args.asset, args.output, args.backends, args.seconds, args.repeats, args.width, args.height)


if __name__ == "__main__":
    main()
