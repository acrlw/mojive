"""Build an interactive backend-neutral scene."""

from __future__ import annotations

import numpy as np

from mojive import CameraView, Light, LightType, Scene, build_scene


def create_scene() -> Scene:
    scene = Scene(
        camera=CameraView(
            eye=np.array((4.0, -5.0, 3.0), np.float32),
            target=np.array((0.0, 0.0, 0.6), np.float32),
        )
    )
    scene.plane(name="floor", size=(5.0, 5.0, 0.04), color=(0.18, 0.21, 0.25, 1.0))
    scene.box(name="crate", size=(0.5, 0.5, 0.5), position=(0.0, 0.0, 0.5))
    scene.sphere(
        name="marker",
        size=(0.3, 0.3, 0.3),
        position=(1.2, 0.0, 0.3),
        color=(0.2, 0.55, 0.95, 1.0),
    )
    scene.add_light(
        "sun",
        Light(
            type=LightType.DIRECTIONAL,
            direction=np.array((-0.5, 0.4, -1.0), np.float32),
        ),
    )
    return scene


def main() -> None:
    viewer = build_scene(create_scene(), title="Mojive programmatic scene")
    try:
        viewer.run()
    finally:
        viewer.release()


if __name__ == "__main__":
    main()
