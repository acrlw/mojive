"""Capture default, visual, collision and combined geometry views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mojive import GeometryStyle, GeometryView, SceneRenderer
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.render.backend import RenderFlag
from mojive.scene.assets import resolve
from mojive.types import CameraView
from mojive.ui.scene_capture import SceneCapture


def capture(output: Path, backend: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    adapter = MuJoCoAdapter(resolve("geometry_views"))
    camera = CameraView(
        eye=np.array((2.8, -4.8, 3.5), np.float32),
        target=np.array((0, 0, 0.4), np.float32),
        up=np.array((0, 0, 1), np.float32),
        near=0.02,
        far=40,
    )
    try:
        source = adapter.scene_source()
        images = {}
        with SceneRenderer(source, width=960, height=640, renderer=backend, camera=camera) as view:
            view.update(adapter.frame(FrameNeeds(poses=True, deformables=True)))
            for mode in GeometryView:
                assert view.set_geometry_view(mode)
                images[mode.value] = view.render().copy()
                Image.fromarray(images[mode.value]).save(output / f"{mode.value}.png")
            view.set_geometry_view(GeometryView.DEFAULT)
            np.testing.assert_array_equal(view.render(), images["default"])
            view.set_geometry_view(GeometryView.BOTH)
            view.set_geometry_style(GeometryStyle((0.25, 0.5, 0.7), 0.9, 0.2))
            Image.fromarray(view.render()).save(output / "both-custom.png")
            view.set_flag(RenderFlag.INERTIA, True)
            view.set_flag(RenderFlag.SCLINERTIA, True)
            frame = adapter.frame(FrameNeeds(diagnostics=True))
            view.update(frame)
            capture = SceneCapture()
            try:
                session = SimpleNamespace(source=source, frame=frame, structure_generation=0)
                with view._current():
                    captured = capture.read(view._backend, session, camera.with_aspect(1.5))
                np.testing.assert_array_equal(captured, view.render())
                Image.fromarray(captured).save(output / "scene-scaled-inertia.png")
            finally:
                with view._current():
                    capture.release()
        changes = {
            mode: int(np.any(pixels != images["visual"], axis=2).sum())
            for mode, pixels in images.items()
        }
        assert changes["collision"] > 1000 and changes["both"] > 1000
        assert adapter.scene_source() is source
        result = {
            "backend": backend,
            "pixels_different_from_visual": changes,
            "model_time": adapter.data.time,
            "source_retained": True,
        }
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        adapter.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/geometry-views"))
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    args = parser.parse_args()
    print(json.dumps(capture(args.output, args.renderer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
