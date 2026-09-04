"""Visualize retained 2D physics and geometry diagnostics in Mojive."""

from __future__ import annotations

from mojive import Scene, build_scene


def main() -> None:
    scene = Scene()
    viewer = build_scene(scene, title="Mojive 2D canvas")
    try:
        canvas = viewer.canvas2d
        grid = canvas.layer("grid", depth=-0.02)
        grid.grid("unit-grid", (-5.0, -3.0, 5.0, 3.0), 0.5)

        shapes = canvas.layer("shapes")
        shapes.circle("body-a", (-1.2, 0.2), 0.65, (0.35, 0.75, 1.0, 1.0), 3.0)
        shapes.rectangle("body-b", (0.3, -0.7), (1.8, 0.6), (1.0, 0.55, 0.25, 1.0), 3.0)
        shapes.polygon(
            "simplex",
            ((-0.2, 1.0), (1.0, 1.7), (1.8, 0.9)),
            (0.55, 0.95, 0.45, 1.0),
            3.0,
        )
        shapes.arrow("velocity", (-1.2, 0.2), (-0.1, 0.8), (1.0, 0.9, 0.25, 1.0), 3.0)
        shapes.point("contact", (-0.55, 0.55), (1.0, 0.25, 0.2, 1.0), 7.0)

        labels = canvas.layer("labels", depth=0.02)
        labels.text("velocity-label", (-0.1, 0.8), "velocity", offset_px=(8.0, -6.0))
        # Resolve the docked viewport before fitting its orthographic camera.
        viewer.sync()
        width, height = viewer.viewport_size
        viewer.set_camera(canvas.camera((-5.0, -3.0, 5.0, 3.0), aspect=width / height))
        viewer.run()
    finally:
        viewer.release()


if __name__ == "__main__":
    main()
