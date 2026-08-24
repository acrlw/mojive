"""Capture camera and light editor helpers for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .. import commands as cmd
from ..adapters.base import NodeKind
from ..composition import build_scene
from ..scene import Scene
from ..types import CameraView, Light, LightKind, LightSet


def acceptance_scene() -> Scene:
    lights = (
        Light(
            kind=LightKind.SPOT,
            position=np.array((-2.0, -1.2, 2.8), np.float32),
            direction=np.array((1.4, 1.0, -1.8), np.float32),
            diffuse=np.array((0.90, 0.68, 0.42), np.float32),
            range=3.2,
            cutoff=28.0,
        ),
        Light(
            kind=LightKind.POINT,
            position=np.array((1.8, -0.2, 1.6), np.float32),
            diffuse=np.array((0.36, 0.58, 0.92), np.float32),
            range=1.35,
        ),
        Light(
            kind=LightKind.AREA,
            position=np.array((0.2, 2.0, 2.6), np.float32),
            direction=np.array((0.0, -1.0, -1.0), np.float32),
            diffuse=np.array((0.52, 0.82, 0.48), np.float32),
            range=3.0,
            area_radius=0.65,
        ),
    )
    scene = Scene(lights=LightSet(lights=lights, ambient=np.full(3, 0.12, np.float32)))
    scene.plane(
        name="ground",
        size=(4.0, 4.0, 0.04),
        position=(0.0, 0.0, -0.04),
        color=(0.23, 0.26, 0.30, 1.0),
    )
    scene.box(
        name="subject",
        size=(0.65, 0.65, 0.65),
        position=(0.0, 0.0, 0.65),
        color=(0.88, 0.42, 0.20, 1.0),
    )
    scene.sphere(
        name="reference",
        size=(0.45, 0.45, 0.45),
        position=(1.15, 0.9, 0.45),
        color=(0.22, 0.58, 0.90, 1.0),
    )
    scene.add_camera(
        "perspective camera",
        CameraView(
            eye=np.array((-2.3, -2.5, 2.4), np.float32),
            target=np.array((0.0, 0.0, 0.7), np.float32),
            near=0.15,
            far=4.5,
            aspect=16.0 / 9.0,
        ),
    )
    scene.add_camera(
        "orthographic camera",
        CameraView(
            eye=np.array((2.8, 1.8, 2.6), np.float32),
            target=np.array((0.0, 0.0, 0.6), np.float32),
            near=0.1,
            far=5.0,
            aspect=4.0 / 3.0,
            orthographic=True,
            ortho_height=2.2,
        ),
    )
    return scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/scene-entities"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for capture in args.output.glob("*.png"):
        capture.unlink()

    viewer = build_scene(acceptance_scene(), vsync=False, width=1600, height=1000)
    try:
        viewer.sync()
        editor_view = CameraView(
            eye=np.array((5.6, -7.2, 4.8), np.float32),
            target=np.array((0.0, 0.1, 0.9), np.float32),
            near=0.02,
            far=30.0,
            aspect=16.0 / 9.0,
        )
        viewer.app.camera.adopt(editor_view)
        viewer.app.camera.publish(viewer.app.camera_out)
        for _ in range(2):
            viewer.sync()
        entities = [
            node for node in viewer.session.nodes if node.kind in (NodeKind.CAMERA, NodeKind.LIGHT)
        ]
        for node in entities:
            viewer.session.submit(cmd.Select(node.object_id))
            viewer.app.gizmo.set_mode("translate")
            for _ in range(3):
                viewer.sync()
            stem = f"{node.kind.value}-{node.name.replace(' ', '-')}"
            _save(viewer, args.output / f"{stem}.png")
            if _supports_rotation(viewer, node):
                viewer.app.gizmo.set_mode("rotate")
                for _ in range(3):
                    viewer.sync()
                _save(viewer, args.output / f"{stem}-rotation.png")
            if node.kind is NodeKind.CAMERA:
                camera_id = viewer.session.cameras[node.camera_index].camera_id
                viewer.app.select_model_camera(camera_id)
                for _ in range(3):
                    viewer.sync()
                _save(viewer, args.output / f"{stem}-through.png")
                viewer.app.select_model_camera(-1)
                viewer.app.camera.adopt(editor_view)
                viewer.app.camera.publish(viewer.app.camera_out)
                for _ in range(2):
                    viewer.sync()
    finally:
        viewer.release()
    print(args.output.resolve())
    return 0


def _supports_rotation(viewer, node) -> bool:
    if node.kind is NodeKind.CAMERA:
        return True
    if node.kind is not NodeKind.LIGHT:
        return False
    light = viewer.session.source.lights.lights[node.light_index]
    return light.kind not in (LightKind.POINT, LightKind.IMAGE)


def _save(viewer, path: Path) -> None:
    pixels = viewer.window.read_frame()[::-1, :, :3]
    Image.fromarray(pixels, "RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
