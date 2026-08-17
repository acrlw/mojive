from __future__ import annotations

import argparse
import statistics
import time

from ..assets import resolve
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description="Measure per-pass CPU and GPU time")
    ap.add_argument("scene", nargs="?", default="many_objects")
    ap.add_argument("-n", "--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--width", type=int, default=1904)
    ap.add_argument("--height", type=int, default=1850)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args(argv)

    path = resolve(args.scene)
    with OffscreenHarness(path, args.width, args.height, args.samples) as h:
        h.warmup(args.warmup)

        frame_ms: list[float] = []
        cpu: dict[str, list[float]] = {}
        gpu: dict[str, list[float]] = {}
        for _ in range(args.frames):
            t0 = time.perf_counter()
            h.step_and_render(1)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)
            s = h.backend.stats
            for k, v in s.cpu_ms.items():
                cpu.setdefault(k, []).append(v)
            for k, v in s.gpu_ms.items():
                gpu.setdefault(k, []).append(v)

        s = h.stats()
        print(f"\n{path.stem}   {args.width}×{args.height}   {args.samples}× MSAA   vsync=off")
        print(h.backend.describe())
        print(
            f"\n  draws {s.draw_calls} · instances {s.instances} · "
            f"triangles {s.triangles} · buckets {s.buckets}"
        )
        print(f"  median frame CPU  {statistics.median(frame_ms):7.3f} ms   ({args.frames} frames)")

        keys = list(dict.fromkeys([*cpu, *gpu]))
        if keys:
            print(f"\n  {'pass':<12}{'CPU ms':>10}{'GPU ms':>10}")
            for k in keys:
                c = statistics.median(cpu[k]) if cpu.get(k) else float("nan")
                g = statistics.median(gpu[k]) if gpu.get(k) else None
                gs = f"{g:10.3f}" if g is not None else f"{'—':>10}"
                print(f"  {k:<12}{c:10.3f}{gs}")
        if not gpu:
            print("\n  GPU timing queries are unavailable; see docs/PLATFORM.md.")
        else:
            zeros = [k for k in keys if gpu.get(k) and statistics.median(gpu[k]) == 0.0]
            if zeros and any(gpu.get(k) and statistics.median(gpu[k]) > 0 for k in keys):
                print(
                    f"\n  Zero GPU samples: {zeros}. Tile-based deferred rendering may charge "
                    "their work to another open query. Use aggregate GPU time on these devices; "
                    "see docs/PLATFORM.md."
                )
        print(
            "\n  CPU and GPU columns overlap in time; the larger value identifies the bottleneck."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
