"""Capture the anymal-c spotlight receiver for local-shadow precision acceptance."""

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
    parser.add_argument("-o", "--output", type=Path, default=Path("output/local-shadow-precision"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build_workspace(
        args.asset,
        paused=True,
        vsync=False,
        width=1600,
        height=1200,
        show_window=False,
    )
    try:
        for _ in range(8):
            viewer.sync()
        source = viewer.session.source
        center = np.asarray(source.scene_center, np.float32)
        extent = max(float(source.scene_extent), 1.0)
        viewer.app.camera.adopt(
            CameraView(
                eye=center + np.asarray((0.0, -0.09, 2.65), np.float32) * extent,
                target=center - np.asarray((0.0, 0.0, 0.11), np.float32) * extent,
                up=np.asarray((0.0, 1.0, 0.0), np.float32),
                fov_y=float(np.radians(45.0)),
                near=max(0.01, extent * 0.01),
                far=extent * 150.0,
            )
        )
        for _ in range(6):
            viewer.sync()
        output = args.output / f"{render_backend_name()}-spot-shadow.png"
        Image.fromarray(viewer.backend.target.read_color(flip=True)[..., :3], "RGB").save(output)
        print(output.resolve())
    finally:
        viewer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
