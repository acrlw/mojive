"""Generate the project visual acceptance gallery."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ..assets import assets_dir, list_assets
from ._harness import OffscreenHarness

OUT = Path("out/gallery")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gallery", description="Render one image per scene")
    ap.add_argument("-o", "--out", default=str(OUT))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--steps", type=int, default=30, help="Simulation steps before capture")
    ap.add_argument("scenes", nargs="*", help="Scene names; defaults to all scenes")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.scenes or list_assets()
    rows: list[tuple[str, str]] = []

    for name in names:
        path = assets_dir() / name if (assets_dir() / name).exists() else None
        if path is None:
            from ..assets import resolve

            try:
                path = resolve(name)
            except FileNotFoundError as e:
                rows.append((name, f"not found: {e}"))
                continue
        try:
            with OffscreenHarness(path, args.width, args.height) as h:
                h.warmup(4)
                h.step_and_render(max(0, args.steps))
                png = out / f"{path.stem}.png"
                h.save_png(png)
                s = h.stats()
                rows.append(
                    (
                        path.stem,
                        f"draws {s.draw_calls:>4} · instances {s.instances:>5} · "
                        f"triangles {s.triangles:>7} · buckets {s.buckets:>3} · "
                        f"{s.frame_ms:6.3f} ms",
                    )
                )
        except Exception as e:
            rows.append((path.stem, f"failed: {type(e).__name__}: {e}"))
            traceback.print_exc(file=sys.stderr)

    width = max((len(r[0]) for r in rows), default=8)
    print(f"\n{out.resolve()}")
    for name, note in rows:
        print(f"  {name:<{width}}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
