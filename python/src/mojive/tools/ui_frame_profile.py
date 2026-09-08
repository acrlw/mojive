"""Profile production viewport chrome and quantify its incremental frame cost."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from .. import commands as cmd
from ..assets import resolve
from ..composition import build
from ..gizmo import GizmoMode, project

_CAPSULE_METHODS = (
    "_draw_playback_widget",
    "_draw_tool_column_widget",
    "_draw_context_hint_widget",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output"))
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--profile-frames", type=int, default=180)
    parser.add_argument("--max-capsule-ms", type=float, default=0.75)
    parser.add_argument("--asset", default="gizmo", help="Asset name or model path to profile")
    parser.add_argument("--running", action="store_true", help="Include live simulation ticks")
    parser.add_argument(
        "--expand-hierarchy", action="store_true", help="Profile expanded hierarchy rows"
    )
    parser.add_argument(
        "--hover-gizmo", action="store_true", help="Sweep the pointer over the selected gizmo"
    )
    parser.add_argument(
        "--gizmo-mode", choices=[mode.value for mode in GizmoMode], default="translate"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build(
        resolve(args.asset),
        paused=not args.running,
        vsync=False,
        width=1600,
        height=1000,
        show_window=False,
    )
    try:
        selected = next(
            node
            for node in viewer.session.nodes
            if node.posable and (args.gizmo_mode != "dimensions" or node.geom_index >= 0)
        )
        viewer.session.submit(cmd.Select(selected.object_id))
        viewer.app.gizmo.set_mode(args.gizmo_mode)
        for _ in range(max(1, args.warmup)):
            viewer.sync()
        if args.expand_hierarchy:
            hierarchy = viewer.panels.get("Hierarchy")
            hierarchy._open_state.update(
                {node.node_id: True for node in viewer.session.nodes if node.children}
            )

        cursor_step = 0
        origin, _rotation = viewer.app._node_pose(selected)
        center = project(viewer.app._camera_view(), (origin,), viewer.app._viewport_rect)[0, :2]

        # Match the native gallery's queued pointer events without moving the OS cursor.
        def sync():
            nonlocal cursor_step
            if args.hover_gizmo:
                angle = cursor_step * 0.31
                radius = 35.0 + 45.0 * (cursor_step % 17) / 16.0
                io = imgui.get_io()
                io.add_mouse_pos_event(
                    *(center + radius * np.array((np.cos(angle), np.sin(angle))))
                )
                cursor_step += 1
            viewer.sync()

        variants = {
            "full": (),
            "without_playback": (_CAPSULE_METHODS[0],),
            "without_tools": (_CAPSULE_METHODS[1],),
            "without_hint": (_CAPSULE_METHODS[2],),
            "without_capsules": _CAPSULE_METHODS,
        }
        samples: dict[str, list[float]] = {name: [] for name in variants}
        # Interleave variants so temperature and driver scheduling do not
        # systematically favor the first or last measurement.
        for round_index in range(3):
            order = tuple(variants) if round_index % 2 == 0 else tuple(reversed(variants))
            for name in order:
                with _disabled(viewer.app, variants[name]):
                    cursor_step = 0
                    for _ in range(4):
                        sync()
                    samples[name].extend(_sample(sync, max(1, args.frames // 3)))

        report = {name: _summary(values) for name, values in samples.items()}
        full = report["full"]["median_ms"]
        for name, method in (
            ("playback_cost_ms", "without_playback"),
            ("tools_cost_ms", "without_tools"),
            ("hint_cost_ms", "without_hint"),
            ("capsules_cost_ms", "without_capsules"),
        ):
            report[name] = max(0.0, full - report[method]["median_ms"])

        profile_path = args.output / "ui-frame-profile.prof"
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(max(1, args.profile_frames)):
            sync()
        profiler.disable()
        profiler.dump_stats(profile_path)
        text_path = args.output / "ui-frame-profile.txt"
        with text_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumtime").print_stats(50)

        report["profile"] = str(profile_path.resolve())
        report["scene"] = {
            "asset": args.asset,
            "running": args.running,
            "expanded_hierarchy": args.expand_hierarchy,
        }
        report["interaction"] = {"hover_gizmo": args.hover_gizmo, "gizmo_mode": args.gizmo_mode}
        report["interaction"]["hit_test_calls"] = sum(
            entry[1]
            for (path, _line, name), entry in stats.stats.items()
            if path == "gizmo.py" and name == "hit_test"
        )
        if args.hover_gizmo and not report["interaction"]["hit_test_calls"]:
            raise RuntimeError("The pointer sweep did not exercise gizmo hit testing")
        report_path = args.output / "ui-frame-profile.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(report_path.resolve())
        print(text_path.resolve())
        if report["capsules_cost_ms"] > args.max_capsule_ms:
            raise SystemExit(
                "viewport capsule overhead "
                f"{report['capsules_cost_ms']:.3f} ms exceeds "
                f"{args.max_capsule_ms:.3f} ms"
            )
    finally:
        viewer.release()
    return 0


def _sample(sync, frames: int) -> list[float]:
    values = []
    for _ in range(frames):
        start = time.perf_counter_ns()
        sync()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return values


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "frames": len(array),
        "median_ms": round(float(np.median(array)), 4),
        "p95_ms": round(float(np.percentile(array, 95)), 4),
        "mean_ms": round(float(np.mean(array)), 4),
    }


@contextmanager
def _disabled(app, names: tuple[str, ...]):
    originals = {name: getattr(app, name) for name in names}
    try:
        for name in names:
            setattr(app, name, lambda: None)
        yield
    finally:
        for name, method in originals.items():
            setattr(app, name, method)


if __name__ == "__main__":
    raise SystemExit(main())
