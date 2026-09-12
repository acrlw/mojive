"""Measure source preparation through first GPU-ready window after queued model loads.

Each model/backend runs in a fresh process; repeated loads use the same Viewer.
OS file caches are not purged. Readback is an upper bound on GPU readiness, not
physical scanout latency. Run without other GPU or performance workloads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODELS = (
    "anybotics_anymal_c",
    "anybotics_anymal_b",
    "boston_dynamics_spot",
    "unitree_go2",
    "unitree_g1",
    "unitree_h1",
    "franka_emika_panda",
    "franka_fr3",
    "universal_robots_ur5e",
    "google_robot",
)


def worker(args):
    import numpy as np
    from PIL import Image

    from mojive.application.composition import build
    from mojive.config import LayoutConfig, ViewerConfig
    from mojive.scene.assets import resolve

    result = {"backend": args.worker, "model": args.model, "loads": []}
    asset = args.root / args.model / "scene.xml"
    startup = time.perf_counter()
    with build(
        resolve("empty"),
        renderer=args.worker,
        paused=True,
        vsync=False,
        width=960,
        height=640,
        show_window=True,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        for _ in range(20):
            viewer.sync()
        viewer.window.read_frame()
        result["empty_viewer_ready_s"] = time.perf_counter() - startup
        for repeat in range(args.repeats):
            if repeat:
                assert viewer.app.load_model(resolve("empty")).ok
                for _ in range(3):
                    viewer.sync()
                viewer.window.read_frame()
            milestones = {}
            finish = viewer.app._finish_model_load_frame

            def record_completion(milestones=milestones, finish=finish):
                completion = viewer.app._model_load_completion
                if completion is not None:
                    milestones.update(
                        source_s=completion.prepared - completion.started,
                        resources_s=completion.resources_ready - completion.prepared,
                    )
                finish()

            viewer.app._finish_model_load_frame = record_completion
            start = time.perf_counter()
            viewer.app._queue_model_load("load", asset)
            frames = []
            while True:
                tick = time.perf_counter()
                viewer.sync()
                frames.append(time.perf_counter() - tick)
                if not viewer.app._model_load_queue and viewer.app._model_load_future is None:
                    break
                if time.perf_counter() - start > 120:
                    raise TimeoutError(f"Model load stalled: {asset}")
            submitted = time.perf_counter()
            pixels = viewer.window.read_frame()[::-1, :, :3]
            ready = time.perf_counter()
            viewer.app._finish_model_load_frame = finish
            assert viewer.session.asset_path == asset.resolve(), viewer.app._model_load_error
            assert viewer.backend.stats.instances == viewer.session.source.instance_count > 0
            row = {
                "repeat": repeat,
                "cache_scope": "first process load" if repeat == 0 else "same-process reload",
                **milestones,
                "first_frame_submitted_s": submitted - start,
                "first_frame_readable_s": ready - start,
                "readiness_wait_s": ready - submitted,
                "longest_ui_frame_s": max(frames),
                "loading_frames": len(frames),
                "instances": viewer.session.source.instance_count,
                "viewport_pixels": [viewer.backend.target.width, viewer.backend.target.height],
            }
            if not np.isfinite(pixels).all() or pixels.std() < 5:
                raise AssertionError("Empty or invalid first window frame")
            if repeat == 0:
                Image.fromarray(pixels).save(args.output.with_suffix(".png"))
            result["loads"].append(row)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("output/native-load-benchmark"))
    parser.add_argument("--worker", choices=("opengl", "bgfx"), help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeat count must be positive")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, model in enumerate(args.models):
        if not (args.root / model / "scene.xml").is_file():
            parser.error(f"Missing model: {model}/scene.xml")
        for backend in ("opengl", "bgfx") if index % 2 == 0 else ("bgfx", "opengl"):
            output = args.output / f"{model}-{backend}.json"
            command = [
                sys.executable,
                "-m",
                __name__ if __name__ != "__main__" else "mojive.tools.native_load_benchmark",
                "--worker",
                backend,
                "--root",
                str(args.root),
                "--model",
                model,
                "--repeats",
                str(args.repeats),
                "--output",
                str(output),
            ]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600
                )
            row = json.loads(output.read_text())
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "report.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
