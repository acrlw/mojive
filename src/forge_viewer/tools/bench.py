from __future__ import annotations

import argparse
import statistics
import time

from ..assets import resolve
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description="逐 pass 量 CPU 与 GPU 耗时")
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
        print(f"\n{path.stem}   {args.width}×{args.height}   {args.samples}× MSAA   vsync 关")
        print(h.backend.describe())
        print(
            f"\n  绘制次数 {s.draw_calls} · 实例 {s.instances} · "
            f"三角形 {s.triangles} · 桶 {s.buckets}"
        )
        print(f"  整帧 CPU 中位数  {statistics.median(frame_ms):7.3f} ms   ({args.frames} 帧)")

        keys = list(dict.fromkeys([*cpu, *gpu]))
        if keys:
            print(f"\n  {'pass':<12}{'CPU ms':>10}{'GPU ms':>10}")
            for k in keys:
                c = statistics.median(cpu[k]) if cpu.get(k) else float("nan")
                g = statistics.median(gpu[k]) if gpu.get(k) else None
                gs = f"{g:10.3f}" if g is not None else f"{'—':>10}"
                print(f"  {k:<12}{c:10.3f}{gs}")
        if not gpu:
            print("\n  （本机没有 GPU 计时查询，GPU 那一列为空——见 docs/PLATFORM.md）")
        else:
            zeros = [k for k in keys if gpu.get(k) and statistics.median(gpu[k]) == 0.0]
            if zeros and any(gpu.get(k) and statistics.median(gpu[k]) > 0 for k in keys):
                print(
                    f"\n  ⚠ 这几条 pass 的 GPU 计为 0：{zeros}。**不是它们不花时间**——"
                    "\n    分块延迟渲染（TBDR）把工作攒到 tile flush 才执行，谁的查询还开着就算给谁，"
                    "\n    于是它们的时间被并进了前面某一条。整帧合计仍然可信，逐条分摊不可信。"
                    "\n    见 docs/PLATFORM.md §3。"
                )
        print("\n  CPU 与 GPU 两列不该相加：一列是发命令，一列是执行。谁大谁是瓶颈。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
