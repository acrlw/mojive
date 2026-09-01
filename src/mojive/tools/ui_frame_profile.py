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

from .. import commands as cmd
from ..assets import resolve
from ..composition import build

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
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build(
        resolve("gizmo"),
        paused=True,
        vsync=False,
        width=1600,
        height=1000,
        show_window=False,
    )
    try:
        selected = next(node for node in viewer.session.nodes if node.posable)
        viewer.session.submit(cmd.Select(selected.object_id))
        for _ in range(max(1, args.warmup)):
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
                    for _ in range(4):
                        viewer.sync()
                    samples[name].extend(_sample(viewer, max(1, args.frames // 3)))

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
            viewer.sync()
        profiler.disable()
        profiler.dump_stats(profile_path)
        text_path = args.output / "ui-frame-profile.txt"
        with text_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumtime").print_stats(50)

        report["profile"] = str(profile_path.resolve())
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


def _sample(viewer, frames: int) -> list[float]:
    values = []
    for _ in range(frames):
        start = time.perf_counter_ns()
        viewer.sync()
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
