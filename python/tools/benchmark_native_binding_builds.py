"""Measure wrapper rebuild costs separately from the binding support libraries."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Recompile each wrapper in alternating order, retaining the common kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("output/cpp-bindings-build"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/native-probe/bindings/build-cost.json")
    )
    args = parser.parse_args()
    sources = {
        "nanobind": Path("cpp/bindings/Nanobind.cpp"),
        "pybind": Path("cpp/bindings/Pybind.cpp"),
    }
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(5):
        for name in list(sources) if repeat % 2 == 0 else list(sources)[::-1]:
            source = sources[name]
            source.touch()
            start = time.perf_counter()
            result = subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(args.build),
                    "--target",
                    f"_mojive_{name}_probe",
                    "-j",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            elapsed = time.perf_counter() - start
            records.append({"binding": name, "repeat": repeat, "seconds": elapsed})
            (args.output.parent / f"build-{name}-{repeat}.log").write_text(
                result.stdout + result.stderr
            )
    report = {
        "scope": "Incremental wrapper compilation plus link; support libraries and C++ kernel already built; warm filesystem cache; no LTO or stripping",
        "records": records,
        "median_seconds": {
            name: statistics.median(r["seconds"] for r in records if r["binding"] == name)
            for name in sources
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["median_seconds"]))


if __name__ == "__main__":
    main()
