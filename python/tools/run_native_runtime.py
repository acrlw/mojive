"""Compare native serial and parallel physics with live rendering and bounded readback."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import subprocess
from pathlib import Path


def main() -> None:
    """Alternate backends and execution order; keep raw runs and median metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("output/cpp-build"))
    parser.add_argument("--scene", type=Path, default=Path("output/native-probe/humanoids100.mjvp"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/native-probe/runtime"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be between 1 and 10")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    combinations = list(
        itertools.product(("pick", "record", "multi"), ("serial", "parallel"), ("bgfx", "sdl"))
    )
    for repeat in range(args.repeats):
        for mode, execution, backend in combinations if repeat % 2 == 0 else combinations[::-1]:
            output = args.output / f"{backend}-{execution}-{mode}-{repeat}.json"
            subprocess.run(
                [
                    str(args.build / "mojive_native_runtime"),
                    str(args.build / "shaders"),
                    str(args.scene),
                    str(args.model),
                    str(output),
                    backend,
                    execution,
                    mode,
                ],
                check=True,
                timeout=60,
            )
            record = json.loads(output.read_text())
            record["repeat"] = repeat
            records.append(record)
            groups = {}
            for row in records:
                key = f"{row['backend']}/{row['execution']}/{row['mode']}"
                groups.setdefault(key, []).append(row)
            medians = {
                key: {
                    metric: statistics.median(row[metric] for row in rows)
                    for metric in rows[0]
                    if isinstance(rows[0][metric], (float, int)) and metric != "repeat"
                }
                for key, rows in groups.items()
            }
            report = {
                "platform": platform.platform(),
                "records": records,
                "median_of_runs": medians,
                "scope": "Native MuJoCo C API, 100 humanoids, 1080p 4x MSAA, 120 FPS pacing; simplified shading without editor UI",
                "recording": "RGB/depth/segmentation every fourth rendered frame, picking every second frame with recording (30/60 Hz at 120 FPS); bounded queue waits when full and never drops outputs",
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
