"""Measure the native probe's rendering and readback costs on one fixed trajectory."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path


def main() -> None:
    """Run alternating mode orders and preserve each result and frame sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("output/native-build"))
    parser.add_argument("--scene", type=Path, default=Path("output/native-probe/humanoids100.mjvp"))
    parser.add_argument("--output", type=Path, default=Path("output/native-probe/benchmark"))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resolutions", default="1920x1080,2560x1440")
    parser.add_argument("--modes", default="none,pick,color,depth,segmentation")
    parser.add_argument("--fps-limit", type=float, default=0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("seconds and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    modes = args.modes.split(",")
    for repeat in range(args.repeats):
        for resolution in args.resolutions.split(","):
            width, height = map(int, resolution.split("x"))
            for mode in modes if repeat % 2 == 0 else modes[::-1]:
                path = args.output / f"{resolution}-{mode}-{repeat}.json"
                subprocess.run(
                    [
                        str(args.build / "mojive_native_benchmark"),
                        str(args.build / "shaders"),
                        str(args.scene),
                        str(path),
                        str(width),
                        str(height),
                        str(args.seconds),
                        mode,
                        str(args.fps_limit),
                    ],
                    check=True,
                    timeout=args.seconds + 120,
                )
                record = json.loads(path.read_text())
                record["repeat"] = repeat
                records.append(record)
                summary = {}
                for item in records:
                    key = f"{item['width']}x{item['height']}/{item['mode']}"
                    summary.setdefault(key, []).append(item)
                aggregates = {
                    key: {
                        metric: statistics.median(item[metric] for item in values)
                        for metric in (
                            "fps",
                            "frame_cpu_p50_ms",
                            "frame_cpu_p95_ms",
                            "frame_cpu_p99_ms",
                            "readback_p50_ms",
                            "readback_p95_ms",
                        )
                    }
                    for key, values in summary.items()
                }
                report = {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "scope": "Fixed trajectory, simplified shading; not a production editor comparison",
                    "seconds_per_run": args.seconds,
                    "repeats_requested": args.repeats,
                    "fps_limit": args.fps_limit,
                    "records": records,
                    "median_of_runs": aggregates,
                }
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
