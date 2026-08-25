"""Draw retained world-space diagnostics in an interactive Forge scene."""

from __future__ import annotations

import numpy as np

from forge_viewer import Occlusion, Scene, build_scene


def add_diagnostics(viewer) -> None:
    """Populate named debug layers owned by the viewer renderer."""
    draw = viewer.backend.debug
    if draw is None:
        raise RuntimeError("The selected renderer does not provide debug drawing")

    depth = draw.layer("example.depth", Occlusion.DEPTH)
    depth.line("ground-axis", (-2.0, 0.0, 0.02), (2.0, 0.0, 0.02), (0.9, 0.9, 0.9, 1.0), 2.0)
    depth.arrow("velocity", (0.0, 0.0, 0.5), (1.4, 0.6, 1.2), (0.2, 0.7, 1.0, 1.0), 3.0)
    depth.point("contact", (0.0, 0.0, 0.04), (1.0, 0.35, 0.15, 1.0), 7.0)

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


def main() -> None:
    """Open the debug-draw example."""
    scene = Scene()
    scene.plane(name="floor", size=(4.0, 4.0, 0.04), color=(0.18, 0.21, 0.25, 1.0))
    scene.box(name="body", size=(0.4, 0.3, 0.5), position=(0.0, 0.0, 0.5))
    viewer = build_scene(scene, title="forge debug draw")
    try:
        add_diagnostics(viewer)
        viewer.run()
    finally:
        viewer.release()


if __name__ == "__main__":
    main()
