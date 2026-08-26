"""Programmatic scenes used by interactive demos."""

from __future__ import annotations

import numpy as np

from .scene import Scene
from .types import CameraView, Light, LightSet, LightType


def canvas_scene() -> Scene:
    sun = Light(
        type=LightType.DIRECTIONAL,
        direction=np.array([-0.5, 0.4, -1.0], np.float32),
        diffuse=np.full(3, 0.75, np.float32),
        specular=np.full(3, 0.3, np.float32),
    )
    scene = Scene(lights=LightSet(lights=(sun,), ambient=np.full(3, 0.18, np.float32)))
    scene.plane(
        name="ground",
        size=(4.0, 4.0, 0.04),
        position=(0.0, 0.0, -0.04),
        color=(0.24, 0.27, 0.31, 1.0),
    )
    scene.box(
        name="crate",
        size=(0.45, 0.45, 0.45),
        position=(-0.7, 0.0, 0.45),
        color=(0.92, 0.42, 0.18, 1.0),
    )
    scene.sphere(
        name="ball",
        size=(0.42, 0.42, 0.42),
        position=(0.45, -0.3, 0.42),
        color=(0.25, 0.58, 0.92, 1.0),
    )
    scene.cylinder(
        name="column",
        size=(0.25, 0.25, 0.75),
        position=(0.85, 0.55, 0.75),
        color=(0.38, 0.78, 0.48, 1.0),
    )
    return scene


def lighting_scene() -> Scene:
    lights = (
        Light(
            type=LightType.SPOT,
            position=np.array([-3.0, -4.0, 5.5], np.float32),
            direction=np.array([3.0, 7.0, -5.0], np.float32),
            diffuse=np.array([0.65, 0.52, 0.42], np.float32),
            cutoff=42.0,
            exponent=3.0,
            range=12.0,
        ),
        Light(
            type=LightType.POINT,
            position=np.array([3.2, 1.5, 3.0], np.float32),
            diffuse=np.array([0.35, 0.48, 0.72], np.float32),
            attenuation=np.array([1.0, 0.03, 0.015], np.float32),
            range=10.0,
        ),
        Light(
            type=LightType.AREA,
            position=np.array([-1.0, 7.0, 4.0], np.float32),
            diffuse=np.array([0.48, 0.62, 0.44], np.float32),
            attenuation=np.array([1.0, 0.03, 0.01], np.float32),
            range=11.0,
            area_radius=0.65,
        ),
    )
    scene = Scene(
        camera=CameraView(
            eye=np.array([0.0, -11.0, 3.8], np.float32),
            target=np.array([0.0, 5.0, 0.9], np.float32),
            far=50.0,
        ),
        lights=LightSet(
            lights=lights,
            headlight=None,
            ambient=np.full(3, 0.08, np.float32),
            fog_color=np.array([0.24, 0.48, 0.76], np.float32),
            fog_start=7.0,
            fog_end=25.0,
            haze_color=np.array([0.90, 0.56, 0.22], np.float32),
            haze_density=0.065,
        ),
    )
    scene.plane(
        name="ground",
        size=(7.0, 18.0, 0.04),
        position=(0.0, 5.0, -0.04),
        color=(0.25, 0.27, 0.30, 1.0),
    )
    scene.box(
        name="spot-box",
        size=(0.7, 0.7, 0.9),
        position=(-1.7, -1.0, 0.9),
        color=(0.88, 0.38, 0.18, 1.0),
    )
    scene.sphere(
        name="point-ball",
        size=(0.7, 0.7, 0.7),
        position=(1.4, 2.0, 0.7),
        color=(0.20, 0.56, 0.92, 1.0),
    )
    scene.cylinder(
        name="area-column",
        size=(0.55, 0.55, 1.35),
        position=(-0.8, 5.5, 1.35),
        color=(0.35, 0.80, 0.45, 1.0),
    )
    scene.box(
        name="far-box",
        size=(0.55, 0.55, 1.4),
        position=(1.8, 10.5, 1.4),
        color=(0.72, 0.60, 0.30, 1.0),
    )
    for index, y in enumerate((-2.0, 2.0, 6.0, 10.0, 14.0)):
        scene.box(
            name=f"depth-{index}",
            size=(0.18, 0.18, 1.25),
            position=(3.2, y, 1.25),
            color=(0.92, 0.92, 0.92, 1.0),
        )
    scene.add_camera(
        "preview camera",
        CameraView(
            eye=np.array([4.5, -3.5, 3.2], np.float32),
            target=np.array([0.0, 2.0, 0.8], np.float32),
            near=0.1,
            far=16.0,
            aspect=16.0 / 9.0,
        ),
    )
    return scene
