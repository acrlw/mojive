"""Compare public MuJoCo and Mojive Renderer API performance."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

RENDERERS = ("mujoco", "mojive-opengl", "mojive-wgpu")
WORKLOADS = ("primitives", "many_objects", "dense_mesh", "dynamic", "materials")
MODES = ("rgb", "depth", "segmentation")
QUICK_RESOLUTIONS = ((640, 480),)
FULL_RESOLUTIONS = ((640, 480), (1280, 720), (1920, 1080))


def _csv(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    if not selected:
        raise ValueError(f"At least one {label} is required")
    return selected


def _resolutions(value: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in value.split(","):
        width, separator, height = item.strip().lower().partition("x")
        if not separator or not width.isdigit() or not height.isdigit():
            raise ValueError(f"Invalid resolution: {item!r}; expected WIDTHxHEIGHT")
        result.append((int(width), int(height)))
    return tuple(result)


def _host() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _case_key(case: dict[str, object]) -> tuple[object, ...]:
    return (case["workload"], case["mode"], case["width"], case["height"], case["samples"])


def _add_relative_results(cases: list[dict[str, object]]) -> None:
    references = {
        _case_key(case): case
        for case in cases
        if case.get("status") == "ok" and case.get("renderer") == "mujoco"
    }
    for case in cases:
        if case.get("status") != "ok":
            continue
        reference = references.get(_case_key(case))
        if reference is None:
            case["relative_to_mujoco"] = None
            continue
        value = float(case["update_and_render"]["median_ms"])  # type: ignore[index]
        baseline = float(reference["update_and_render"]["median_ms"])  # type: ignore[index]
        case["relative_to_mujoco"] = value / max(baseline, 1e-9)


def _print_report(cases: list[dict[str, object]], output: Path) -> None:
    print(
        f"\n{'renderer':<18}{'workload':<15}{'output':<15}{'median ms':>11}{'p95 ms':>10}{'fps':>10}{'vs MuJoCo':>12}"
    )
    for case in cases:
        renderer = str(case.get("renderer", "?"))
        workload = str(case.get("workload", "?"))
        mode = str(case.get("mode", "?"))
        size = f"{case.get('width', '?')}x{case.get('height', '?')}"
        if case.get("status") != "ok":
            print(f"{renderer:<18}{workload:<15}{mode + ' ' + size:<15}{'FAILED':>11}")
            continue
        combined = case["update_and_render"]
        relative = case.get("relative_to_mujoco")
        ratio = "—" if relative is None else f"{float(relative):.2f}x"
        print(
            f"{renderer:<18}{workload:<15}{mode + ' ' + size:<15}"
            f"{float(combined['median_ms']):>11.3f}{float(combined['p95_ms']):>10.3f}"
            f"{float(case['median_fps']):>10.1f}{ratio:>12}"
        )
    print(f"\nReport: {output.resolve()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--workloads")
    parser.add_argument("--modes")
    parser.add_argument("--resolutions")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output/renderer-benchmark/report.json")
    )
    args = parser.parse_args(argv)

    try:
        renderers = _csv(args.renderers, RENDERERS, "renderer")
        default_workloads = WORKLOADS if args.preset == "full" else WORKLOADS[:3]
        workloads = (
            _csv(args.workloads, WORKLOADS, "workload") if args.workloads else default_workloads
        )
        default_modes = MODES if args.preset == "full" else MODES[:1]
        modes = _csv(args.modes, MODES, "mode") if args.modes else default_modes
        resolutions = (
            _resolutions(args.resolutions)
            if args.resolutions
            else (FULL_RESOLUTIONS if args.preset == "full" else QUICK_RESOLUTIONS)
        )
    except ValueError as exc:
        parser.error(str(exc))

    frames = args.frames if args.frames is not None else (200 if args.preset == "full" else 80)
    warmup = args.warmup if args.warmup is not None else (20 if args.preset == "full" else 10)
    cases: list[dict[str, object]] = []
    jobs = [
        (renderer, workload, mode, width, height)
        for workload in workloads
        for width, height in resolutions
        for mode in modes
        for renderer in renderers
    ]
    print(
        f"Renderer benchmark: {len(jobs)} isolated cases, {frames} measured + {warmup} warmup frames"
    )
    with tempfile.TemporaryDirectory(prefix="mojive-renderer-benchmark-") as directory:
        temp = Path(directory)
        for index, (renderer, workload, mode, width, height) in enumerate(jobs, 1):
            print(
                f"[{index:>3}/{len(jobs)}] {renderer} · {workload} · {mode} · {width}x{height}",
                flush=True,
            )
            case_path = temp / f"case-{index}.json"
            command = [
                sys.executable,
                "-m",
                "mojive.tools.renderer_benchmark_worker",
                "--renderer",
                renderer,
                "--workload",
                workload,
                "--mode",
                mode,
                "--width",
                str(width),
                "--height",
                str(height),
                "--samples",
                str(args.samples),
                "--frames",
                str(frames),
                "--warmup",
                str(warmup),
                "--output",
                str(case_path),
            ]
            environment = os.environ.copy()
            environment.pop("MOJIVE_BACKEND", None)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                cases.append(
                    {
                        "status": "failed",
                        "renderer": renderer,
                        "workload": workload,
                        "mode": mode,
                        "width": width,
                        "height": height,
                        "samples": args.samples,
                        "error": f"timed out after {exc.timeout} seconds",
                    }
                )
                continue
            process_ms = (time.perf_counter() - started) * 1000.0
            if completed.returncode != 0 or not case_path.exists():
                cases.append(
                    {
                        "status": "failed",
                        "renderer": renderer,
                        "workload": workload,
                        "mode": mode,
                        "width": width,
                        "height": height,
                        "samples": args.samples,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                    }
                )
                continue
            case = json.loads(case_path.read_text(encoding="utf-8"))
            case["process_elapsed_ms"] = process_ms
            cases.append(case)

    _add_relative_results(cases)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "method": {
            "api": "public Renderer update_scene() and render() wall-clock time",
            "isolation": "one subprocess per renderer/workload/output/resolution case",
            "vsync": "offscreen",
            "frames": frames,
            "warmup": warmup,
            "samples": args.samples,
            "preset": args.preset,
            "note": "Mojive internal GPU pass timers are intentionally excluded from cross-renderer ratios.",
            "output_buffers": (
                "RGB and depth reuse out arrays. Segmentation uses allocating render() for both "
                "implementations because MuJoCo 3.11 rejects its ID-pair shape as an out array."
            ),
        },
        "host": _host(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(cases, args.output)
    return 1 if any(case.get("status") != "ok" for case in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
