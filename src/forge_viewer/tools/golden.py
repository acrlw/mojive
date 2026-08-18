"""Render, compare, and review golden images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..assets import resolve
from ._harness import OffscreenHarness

GOLDEN_DIR = Path("tests/golden")
OUT_DIR = Path("output/golden")


CASES: tuple[tuple[str, int], ...] = (
    ("test_scene", 40),
    ("pick_scene", 40),
    ("sunlight_shadow", 40),
    ("material_matrix", 40),
    ("transparency", 40),
    ("showcase", 40),
)
WIDTH, HEIGHT = 640, 480


TOL_P99 = 6
TOL_MAXFRAC = 0.002


def render_case(name: str, steps: int) -> np.ndarray:
    with OffscreenHarness(resolve(name), WIDTH, HEIGHT) as h:
        h.warmup(4)
        h.step_and_render(steps)
        return h.backend.target.read_color(flip=True)[..., :3].copy()


def compare(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape mismatch: {a.shape} vs {b.shape}"
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    p99 = float(np.percentile(d, 99))
    frac = float((d.max(axis=2) > 32).mean())
    ok = p99 <= TOL_P99 and frac <= TOL_MAXFRAC
    return ok, f"p99={p99:.1f} (≤{TOL_P99}) outliers={frac:.3%} (≤{TOL_MAXFRAC:.1%})"


def side_by_side(old: np.ndarray | None, new: np.ndarray, path: Path) -> None:
    from PIL import Image

    parts = (
        [new]
        if old is None
        else [old, new, np.abs(old.astype(np.int16) - new.astype(np.int16)).astype(np.uint8)]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(parts, axis=1), "RGB").save(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="golden", description="Golden-image regression")
    ap.add_argument("--accept", action="store_true", help="Accept current renders as baselines")
    ap.add_argument("--golden-dir", default=str(GOLDEN_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("cases", nargs="*", help="Case names; defaults to all cases")
    args = ap.parse_args(argv)

    gdir, odir = Path(args.golden_dir), Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)
    names = args.cases or [c[0] for c in CASES]
    steps = dict(CASES)

    from PIL import Image

    failures, news = [], []
    for name in names:
        try:
            img = render_case(name, steps.get(name, 40))
        except Exception as e:
            print(f"✗ {name:18} render failed: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(name)
            continue

        ref_path = gdir / f"{name}.png"
        old = np.asarray(Image.open(ref_path).convert("RGB")) if ref_path.exists() else None

        if args.accept:
            side_by_side(old, img, odir / f"{name}.review.png")
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img, "RGB").save(ref_path)
            print(f"· {name:18} baseline updated; review {odir / f'{name}.review.png'}")
            continue

        if old is None:
            news.append(name)
            Image.fromarray(img, "RGB").save(odir / f"{name}.new.png")
            print(f"? {name:18} missing baseline → {odir / f'{name}.new.png'}")
            continue

        ok, note = compare(old, img)
        print(f"{'✓' if ok else '✗'} {name:18} {note}")
        if not ok:
            failures.append(name)
            side_by_side(old, img, odir / f"{name}.diff.png")

    if args.accept:
        print(f"\nReview old | new | diff images before adding {gdir}/.")
        return 0
    if news:
        print(f"\n{len(news)} missing baselines: {news}. Review new images, then run --accept.")
    if failures:
        print(f"\n{len(failures)} mismatches: {failures}. Diff images: {odir}/*.diff.png")
        return 1
    print(f"\n{len(names) - len(news)} cases match their baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
