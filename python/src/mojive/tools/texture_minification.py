"""Capture near and distant textured-plane views for minification acceptance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..composition import build_workspace, render_backend_name
from ..types import CameraView


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/texture-minification"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build_workspace(
        args.asset,
        paused=True,
        vsync=False,
        width=1600,
        height=1000,
        show_window=False,
    )
    try:
        for _ in range(8):
            viewer.sync()
        source = viewer.session.source
        center = np.asarray(source.scene_center, np.float32)
        extent = max(float(source.scene_extent), 1.0)
        original = viewer.app.camera.view()
        for name, offset in (
            ("near", (3.8, -3.8, 2.3)),
            ("far", (18.0, -18.0, 11.0)),
        ):
            viewer.app.camera.adopt(
                CameraView(
                    eye=center + np.asarray(offset, np.float32) * extent,
                    target=center,
                    up=np.array((0.0, 0.0, 1.0), np.float32),
                    fov_y=original.fov_y,
                    near=original.near,
                    far=max(original.far, extent * 200.0),
                )
            )
            for _ in range(6):
                viewer.sync()
            output = args.output / f"{render_backend_name()}-{name}.png"
            _capture_viewport(viewer, output)
            print(output.resolve())
    finally:
        viewer.release()
    return 0


def _capture_viewport(viewer, output: Path) -> None:
    pixels = viewer.window.read_frame()[::-1, :, :3]
    x, y, width, height = viewer.app._viewport_rect
    x, y, width, height = viewer.window.points_to_pixels((x, y, width, height))
    x0, y0 = max(0, round(x)), max(0, round(y))
    x1 = min(pixels.shape[1], round(x + width))
    y1 = min(pixels.shape[0], round(y + height))
    Image.fromarray(pixels[y0:y1, x0:x1], "RGB").save(output)


if __name__ == "__main__":
    raise SystemExit(main())
