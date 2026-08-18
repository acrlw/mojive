"""Capture MJCF and URDF model-loading references."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from ..assets import resolve
from ..composition import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture runtime MJCF and URDF loading")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/model-loading"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build(resolve("empty"), paused=True, vsync=False, width=1400, height=900)
    try:
        _capture(viewer, args.output / "empty.png")
        viewer.window._file_drag_active = True
        _capture(viewer, args.output / "drop-hover.png")
        viewer.window._on_file_drop(None, [str(resolve("test_scene.xml"))])
        _capture(viewer, args.output / "mjcf.png")
        for name, output_name in (("test_scene.urdf", "urdf.png"),):
            result = viewer.app.load_model(resolve(name))
            if not result.ok:
                raise RuntimeError(result.message)
            _capture(viewer, args.output / output_name)
    finally:
        viewer.release()

    for name in ("empty.png", "drop-hover.png", "mjcf.png", "urdf.png"):
        path = args.output / name
        print(path.resolve())
    return 0


def _capture(viewer, path: Path) -> None:
    for _ in range(4):
        viewer.sync()
    pixels = viewer.window.read_frame()[::-1, :, :3]
    Image.fromarray(pixels, "RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
