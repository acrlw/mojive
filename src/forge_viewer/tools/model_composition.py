"""Capture the runtime MJCF model-composition workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from ..assets import resolve
from ..composition import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture runtime model composition")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/model-composition"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build(resolve("gizmo"), paused=True, vsync=False, width=1400, height=900)
    try:
        result = viewer.app.add_model(resolve("test_scene.urdf"), (2.5, 0.0, 0.0))
        if not result.ok:
            raise RuntimeError(result.message)
        _capture(viewer, args.output / "combined.png")
        result = viewer.app.remove_model(result.entity_id)
        if not result.ok:
            raise RuntimeError(result.message)
        _capture(viewer, args.output / "removed.png")
    finally:
        viewer.release()

    for name in ("combined.png", "removed.png"):
        print((args.output / name).resolve())
    return 0


def _capture(viewer, path: Path) -> None:
    for _ in range(4):
        viewer.sync()
    Image.fromarray(viewer.window.read_frame()[::-1, :, :3], "RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
