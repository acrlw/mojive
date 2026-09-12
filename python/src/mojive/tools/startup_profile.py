"""Measure editor startup in fresh processes, including the first presented frame."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _measure(args) -> None:
    import cProfile
    import functools
    import pstats
    import shutil

    launch = float(os.environ["MOJIVE_PROFILE_LAUNCH"])
    events = []
    frames = []
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()

    def event(name):
        events.append({"name": name, "elapsed_s": time.perf_counter() - launch})

    from mojive import cli
    from mojive.application import composition
    from mojive.session import Session
    from mojive.ui.app import ViewerApp
    from mojive.ui.window import Window, layout_settings_path

    event("imports_ready")
    if args.dynamic_icons:
        from mojive.ui import icons

        icons._production_icon_preset = lambda _name: None

    # Exercise the user's layout without writing it during the measurement.
    layout = layout_settings_path()
    probe_layout = args.output.with_suffix(".ini")
    if layout.is_file():
        shutil.copyfile(layout, probe_layout)
    composition._viewer_layout_path = lambda *_a, **_kw: str(probe_layout)

    def trace(owner, method, label):
        original = getattr(owner, method)

        @functools.wraps(original)
        def measured(*a, **kw):
            result = original(*a, **kw)
            event(label)
            return result

        setattr(owner, method, measured)

    trace(Window, "_load_fonts", "fonts_ready")
    trace(Window, "show", "window_shown")
    trace(Session, "__init__", "session_ready")
    trace(ViewerApp, "__init__", "app_ready")
    original_frame = ViewerApp.frame

    def frame(app):
        start = time.perf_counter()
        original_frame(app)
        frames.append(time.perf_counter() - start)
        if len(frames) == 1 and profiler:
            profiler.disable()

    ViewerApp.frame = frame
    original_run = composition.Viewer.run

    def run(viewer):
        trace(viewer.window, "end_frame", "frame_presented")
        original_run(viewer, max_frames=4)
        # Readback and image encoding are outside every startup timing interval.
        from PIL import Image

        Image.fromarray(viewer.window.read_frame()[::-1]).save(args.output.with_suffix(".png"))

    composition.Viewer.run = run
    try:
        result = cli.main(["editor", args.asset])
        if result:
            raise RuntimeError(f"Editor exited with status {result}")
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.output.with_suffix(".prof")))
            with args.output.with_suffix(".profile.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(60)
        args.output.write_text(json.dumps({"events": events, "frames_s": frames}, indent=2) + "\n")


def _trial(args, backend: str, dynamic: bool, repeat: int) -> dict:
    mode = "dynamic" if dynamic else "presets"
    path = args.output / f"{backend}-{mode}-{repeat}.json"
    command = [
        sys.executable,
        "-m",
        "mojive.tools.startup_profile",
        "--worker",
        "--asset",
        args.asset,
        "--output",
        str(path),
    ]
    if dynamic:
        command.append("--dynamic-icons")
    if args.profile:
        command.append("--profile")
    env = dict(
        os.environ,
        MOJIVE_RENDERER=backend,
        MOJIVE_PROFILE_LAUNCH=str(time.perf_counter()),
    )
    with path.with_suffix(".log").open("w") as log:
        subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=True,
        )
    events = json.loads(path.read_text())["events"]
    present = next(e["elapsed_s"] for e in events if e["name"] == "frame_presented")
    shown = next(e["elapsed_s"] for e in events if e["name"] == "window_shown")
    row = {
        "backend": backend,
        "mode": mode,
        "repeat": repeat,
        "launch_s": present,
        "shown_s": present - shown,
    }

    return row


def main(argv: list[str] | None = None) -> int:
    """Run serial startup trials and save timings, logs, and window captures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", nargs="+", choices=("opengl", "wgpu", "bgfx"), default=["opengl"]
    )
    parser.add_argument("--asset", default="joint_gizmo")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("output/startup-profile"))
    parser.add_argument(
        "--profile", action="store_true", help="Collect cProfile; adds timing overhead"
    )
    parser.add_argument(
        "--compare-icons", action="store_true", help="Compare presets with dynamic icon fitting"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dynamic-icons", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        _measure(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        backends = args.backend if repeat % 2 == 0 else args.backend[::-1]
        modes = [False, True] if args.compare_icons else [False]
        if repeat % 2:
            modes.reverse()
        for backend in backends:
            for dynamic in modes:
                row = _trial(args, backend, dynamic, repeat)
                rows.append(row)
                print(json.dumps(row), flush=True)
    medians = []
    for backend, mode in sorted({(row["backend"], row["mode"]) for row in rows}):
        selected = [row for row in rows if row["backend"] == backend and row["mode"] == mode]
        medians.append(
            {
                "backend": backend,
                "mode": mode,
                "launch_s": statistics.median(row["launch_s"] for row in selected),
                "shown_s": statistics.median(row["shown_s"] for row in selected),
            }
        )
    report = {"asset": args.asset, "profiled": args.profile, "runs": rows, "medians": medians}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(medians, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
