"""Draw retained world-space diagnostics in an interactive Mojive scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mojive import CameraView, Occlusion, Scene, build_scene
from mojive import commands as cmd
from mojive.config import LayoutConfig, ViewerConfig


def add_diagnostics(viewer) -> None:
    """Populate named debug layers owned by the viewer renderer."""
    draw = viewer.backend.debug
    if draw is None:
        raise RuntimeError("The selected renderer does not provide debug drawing")

    depth = draw.layer("example.depth", Occlusion.DEPTH)
    depth.line("ground-axis", (-2.0, 0.0, 0.02), (2.0, 0.0, 0.02), (0.9, 0.9, 0.9, 1.0), 2.0)
    depth.arrow("velocity", (0.0, 0.0, 0.5), (1.4, 0.6, 1.2), (0.2, 0.7, 1.0, 1.0), 3.0)
    depth.point("contact", (0.0, 0.0, 0.04), (1.0, 0.35, 0.15, 1.0), 7.0)

    depth.arrow_3d("target", (-0.6, 0, 1.5), (0.9, 0, 1.5), (0, 1, 0, 1))
    depth.arc_arrow_3d(
        "yaw",
        (0, 0, 0.9),
        (0, 0, 1),
        (1, 0, 0),
        -1.5 * np.pi,
        (1, 0, 0, 1),
        radius=0.65,
        shaft_radius=0.025,
    )

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = (0.0, 0.0, 0.5)
    depth.frame("body-frame", transform, axis_len=0.8)

    overlay = draw.layer("example.labels", Occlusion.ALWAYS)
    overlay.text(
        "label",
        (1.4, 0.6, 1.2),
        "velocity",
        color=(0.85, 0.93, 1.0, 1.0),
        offset_px=(10.0, -8.0),
    )
    for index, (name, radius, smoothing) in enumerate(
        (("g3", 1.5, 0.6), ("circular", 1.5, 0.0), ("sharp", 0.0, 0.0))
    ):
        y = 60.0 + index * 50.0
        overlay.arrow_2d(
            name,
            (40.0, y),
            (240.0, y),
            (0.85, 0.93, 1.0, 1.0),
            4.0,
            head_length_px=18.0,
            head_width_px=20.0,
            corner_radius_px=radius,
            smoothing=smoothing,
        )


def main() -> None:
    """Open the debug-draw example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, help="Save two views and exit")
    args = parser.parse_args()
    scene = Scene()
    scene.plane(name="floor", size=(4.0, 4.0, 0.04), color=(0.18, 0.21, 0.25, 1.0))
    scene.box(name="body", size=(0.4, 0.3, 0.5), position=(0.0, 0.0, 0.5))
    config = (
        ViewerConfig(layout=LayoutConfig(persistence=False, reset=True))
        if args.capture_dir
        else None
    )
    viewer = build_scene(
        scene, title="Mojive debug draw", show_window=args.capture_dir is None, config=config
    )
    try:
        add_diagnostics(viewer)
        # Resolve initial scene framing and the docked viewport before setting the camera.
        for _ in range(4):
            viewer.sync()
        body = next(node for node in viewer.session.nodes if node.name == "body")
        viewer.session.submit(cmd.Select(body.object_id))
        viewer.set_camera(CameraView(eye=(2.7, -3.8, 2.8), target=(0, 0, 0.8)))
        if args.capture_dir is None:
            viewer.run()
        else:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
            for name, eye in (("oblique", (2.7, -3.8, 2.8)), ("rear", (-2.7, 3.8, 2.0))):
                viewer.set_camera(CameraView(eye=eye, target=(0, 0, 0.8)))
                for _ in range(4):
                    viewer.sync()
                viewer.capture(args.capture_dir / f"{name}.png", surface="viewport")
    finally:
        viewer.release()


if __name__ == "__main__":
    main()
