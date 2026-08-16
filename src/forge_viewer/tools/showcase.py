from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..assets import resolve
from ..bridge import Occlusion, bridge
from ._harness import OffscreenHarness

OUT = Path("out/showcase")


def _add_debug_layers(harness: OffscreenHarness) -> str:

    notes = []
    br = bridge(harness.backend)
    axes = br.layer("showcase.axes", Occlusion.ALWAYS)
    rays = br.layer("showcase.rays", Occlusion.GHOST)
    boxes = br.layer("showcase.boxes", Occlusion.DEPTH)
    marks = br.layer("showcase.marks", Occlusion.ALWAYS)
    if axes is None or rays is None or boxes is None or marks is None:
        return "后端不支持 debug draw，四层标注全部跳过"

    for i, x in enumerate(np.linspace(-3.0, 3.0, 5)):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = (float(x), 3.4, 0.6)
        axes.frame(f"axis{i}", m, 0.35)
    for i, x in enumerate(np.linspace(-3.0, 3.0, 7)):
        rays.line(f"ray{i}", (float(x), 0.6, 0.0), (float(x), 0.6, 2.2), (0.9, 0.7, 0.2, 1.0), 2.0)
    for i, x in enumerate(np.linspace(-2.0, 2.0, 4)):
        m = np.eye(4, dtype=np.float32) * 0.25
        m[3, 3] = 1.0
        m[:3, 3] = (float(x), -2.0, 1.0)
        boxes.box(f"box{i}", m, (0.3, 0.6, 0.9, 0.45))
    marks.point("origin", (0.0, 0.0, 0.05), (1.0, 0.3, 0.3, 1.0), 9.0)
    notes.append("四层：axes(ALWAYS) rays(GHOST) boxes(DEPTH) marks(ALWAYS)")
    return "；".join(notes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="showcase", description="一屏看全部已落地的渲染能力")
    ap.add_argument("-o", "--out", default=str(OUT / "showcase.png"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--no-debug-draw", action="store_true")
    args = ap.parse_args(argv)

    path = resolve("showcase")
    with OffscreenHarness(path, args.width, args.height) as h:
        h.warmup(4)
        note = "（跳过）" if args.no_debug_draw else _add_debug_layers(h)
        h.step_and_render(20)
        out = Path(args.out)
        h.save_png(out)
        s = h.stats()
        print(f"\n{out.resolve()}")
        print(h.backend.describe())
        print(
            f"\n  绘制 {s.draw_calls} · 实例 {s.instances} · 三角形 {s.triangles} · 桶 {s.buckets}"
        )
        print(f"  debug draw：{note}")
        print("\n  它只出图、不判对错——门禁答不了「现在到底画成什么样」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
