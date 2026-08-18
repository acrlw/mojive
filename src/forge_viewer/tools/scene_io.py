"""Exercise authored scene save, load, and capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from ..composition import build_scene
from ..demos import canvas_scene
from ..scene import Scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save and reload a Forge scene")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/scene-io"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    document = canvas_scene().save(args.output / "canvas.forge.json")
    viewer = build_scene(Scene.load(document), vsync=False, width=1400, height=900)
    image = args.output / "canvas.png"
    try:
        for _ in range(4):
            viewer.sync()
        pixels = viewer.window.read_frame()[::-1, :, :3]
        Image.fromarray(pixels, "RGB").save(image)
    finally:
        viewer.release()

    print(document.resolve())
    print(image.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
