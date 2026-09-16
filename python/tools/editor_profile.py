"""Profile editor frames and document edits without mixing timers with CPU attribution."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import platform
import pstats
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mojive import commands as cmd
from mojive.app.composition import build
from mojive.scene.assets import resolve


def _summary(values):
    return {
        key: float(np.percentile(values, percentile))
        for key, percentile in (("median", 50), ("p95", 95), ("p99", 99), ("max", 100))
    }


def _stages(viewer):
    return (
        (viewer.session, "tick", "session.tick"),
        (viewer.app, "_sync_structure", "source_publication"),
        (viewer.backend, "set_scene", "backend.set_scene"),
        (viewer.backend, "update", "backend.update"),
        (viewer.backend, "render", "backend.render"),
        (viewer.app.scene_entities, "publish", "helpers"),
        (viewer.panels, "draw", "panels"),
        (viewer.window, "begin_frame", "window.begin"),
        (viewer.app, "_present_frame", "presentation"),
    )


def _run(args):
    report = {
        "platform": platform.platform(),
        "asset": args.asset,
        "frames": args.frames,
        "scope": (
            "Uninstrumented wall/CPU samples, inclusive stage timers and cProfile run separately. "
            "Nested stage times must not be added. sync return is not display scanout latency. "
            "OpenGL drain waits for GPU completion; other backends include color readback."
        ),
        "cases": {},
    }
    with build(
        resolve(args.asset),
        renderer=args.renderer,
        paused=True,
        vsync=False,
        show_window=False,
        width=args.width,
        height=args.height,
    ) as viewer:
        for _ in range(24):
            viewer.sync()
        body = next(
            (node for node in viewer.session.nodes if node.posable and node.object_id), None
        )
        geom = next((node for node in viewer.session.nodes if node.geom_index >= 0), None)
        if body is None or geom is None:
            raise ValueError("Editor profiling requires a posable body and an editable geometry")
        report.update(
            renderer=viewer.backend.caps.name,
            window_points=viewer.window.size_points,
            pixel_scale=viewer.window.pixel_scale,
            instances=viewer.session.source.instance_count,
            nodes=len(viewer.session.nodes),
            gpu_timing_supported=viewer.backend.caps.gpu_timing,
        )

        def drain():
            if viewer.backend.caps.name == "opengl":
                viewer.backend.ctx.finish()
            else:
                viewer.backend.target.read_color()

        def execute(mode, index):
            result = None
            if mode == "orbit":
                viewer.app.camera.orbit(0.2, 0.05)
            elif mode == "visibility":
                result = viewer.session.submit(cmd.SetVisible(body.node_id, bool(index % 2)))
            elif mode == "color":
                result = viewer.session.submit(
                    cmd.SetGeometryColor(geom.node_id, ((index % 10) / 10, 0.4, 0.8, 1))
                )
            if result is not None and not result.ok:
                raise RuntimeError(result.message)
            viewer.sync()

        for mode in ("idle", "orbit", "visibility", "color"):
            count = args.frames if mode in ("idle", "orbit") else max(12, args.frames // 3)
            for index in range(5):
                execute(mode, index)
            drain()
            wall, cpu, waits = [], [], []
            for index in range(count):
                start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
                execute(mode, index)
                cpu.append((time.thread_time_ns() - cpu_start) / 1e6)
                wall.append((time.perf_counter_ns() - start) / 1e6)
            for index in range(10):
                execute(mode, index)
                start = time.perf_counter_ns()
                drain()
                waits.append((time.perf_counter_ns() - start) / 1e6)
            row = {
                "sync_wall_ms": _summary(wall),
                "main_thread_cpu_ms": _summary(cpu),
                "post_sync_gpu_wait_or_readback_ms": _summary(waits),
                "gpu_pass_ms": dict(viewer.backend.stats.gpu_ms),
            }
            stages = {}
            with ExitStack() as stack:
                for owner, name, label in _stages(viewer):
                    function = getattr(owner, name)

                    def measured(
                        *values, _function=function, _label=label, _stages=stages, **kwargs
                    ):
                        start = time.perf_counter_ns()
                        try:
                            return _function(*values, **kwargs)
                        finally:
                            _stages.setdefault(_label, []).append(
                                (time.perf_counter_ns() - start) / 1e6
                            )

                    stack.enter_context(patch.object(owner, name, measured))
                for index in range(min(30, count)):
                    execute(mode, index)
            row["inclusive_stages_ms"] = {key: _summary(value) for key, value in stages.items()}
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                for index in range(min(30, count)):
                    execute(mode, index)
            finally:
                profiler.disable()
            profiler.dump_stats(str(args.output / f"{mode}.prof"))
            with (args.output / f"{mode}.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(60)
            report["cases"][mode] = row
        viewer.capture(args.output / "window.png", surface="window")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="joint_gizmo")
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/editor-profile"))
    args = parser.parse_args(argv)
    if args.frames < 12 or min(args.width, args.height) < 320:
        parser.error("frames must be at least 12 and window dimensions at least 320")
    args.output.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {"MOJIVE_SETTINGS": str(args.output / "settings.json")}):
        report = _run(args)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
