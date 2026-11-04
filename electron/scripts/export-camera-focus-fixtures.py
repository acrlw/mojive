"""Independent native focus_bounds/focus_target samples for the Electron port."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mojive.config import CameraNavigationConfig
from mojive.types import CameraView
from mojive.ui.camera import OrbitCamera, elevated_focus_view_direction

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/camera-focus-native.json"


class Sink:
    def set_camera(self, view):
        self.view = view


def data(view):
    result = {
        "projection": "orthographic" if view.orthographic else "perspective",
        "position": np.asarray(view.eye).tolist(), "target": np.asarray(view.target).tolist(),
        "up": np.asarray(view.up).tolist(), "fov": float(np.degrees(view.fov_y)),
        "near": float(view.near), "far": float(view.far), "orthoHeight": float(view.ortho_height),
        "sensorWidth": 35, "focalLength": 35 / max(view.aspect, 1) / (2 * np.tan(view.fov_y / 2)),
        "resolution": [view.aspect * 1000, 1000],
    }
    if view.orthographic_blend is not None:
        result["projectionBlend"] = view.projection_blend()
    if view.uses_intrinsics():
        result["normalizedFocal"] = (view.focal_length / view.sensor_size).tolist()
        result["principalOffset"] = (view.principal_offset / view.sensor_size).tolist()
    return result


def scenario(name, start, center, half, **options):
    view_cube = options.pop("viewCube", False)
    camera = OrbitCamera()
    camera.configure_navigation(CameraNavigationConfig(
        focus_margin=options.get("margin", 1.15),
        focus_duration=options.get("duration", 0.36),
        focus_easing="smoothstep" if view_cube or options.get("frameScene") else options.get("easing", "smoothstep"),
        min_distance=options.get("minDistance", 0.001),
        max_distance=options.get("maxDistance", 1e6),
    ))
    camera.adopt(start, exact=True)
    sink = Sink()
    if "eyeDirection" in options:
        direction = options["eyeDirection"]
    else:
        direction = elevated_focus_view_direction(start.eye - center, camera.direction())
    if options.get("frameScene"):
        camera.frame_scene((np.asarray(center)-half,np.asarray(center)+half),sink,minimum_pitch=30)
    elif view_cube:
        axis = np.asarray(direction, np.float64)
        axis /= np.linalg.norm(axis)
        camera.look_from_target(
            float(np.degrees(np.arctan2(axis[1], axis[0]))),
            float(np.degrees(np.arcsin(axis[2]))),
            center, options["sphereRadius"], sink, margin=options["sphereMargin"],
        )
    elif "sphereRadius" in options:
        camera.focus_target(center, options["sphereRadius"], direction, sink, margin=options.get("sphereMargin", 1.5))
    else:
        camera.focus_bounds(center, half, sink, eye_direction=direction)
    steps = []
    duration = camera.navigation.focus_duration
    previous = 0
    for elapsed in sorted(set((0, duration * 0.05, duration * 0.15, duration * 0.3, duration * 0.5, duration * 0.8, duration, duration + 0.1))):
        camera.advance(elapsed - previous, sink)
        view = camera.view()
        steps.append({
            "elapsed": elapsed,
            "expected": {"view": data(view), "viewMatrix": view.view_matrix().tolist(), "projectionMatrix": view.proj_matrix().tolist()},
        })
        previous = elapsed
    return {
        "id": name, "start": data(start),
        "bounds": {"center": np.asarray(center).tolist(), "half": np.asarray(half).tolist()},
        "options": options, "steps": steps,
    }


def main():
    start = CameraView(eye=np.array((6.0, 0.0, 0.0)), target=np.zeros(3), aspect=1.5, near=0.02, far=200)
    center = np.array((2.0, -1.5, 1.0))
    half = np.array((0.6, 0.2, 0.35))
    rolled = replace(start, eye=np.array((5.0, -4.0, 3.0)), target=np.array((0.2, 0.4, 0.5)), up=np.array((0.3, 0.2, 1.0)), aspect=0.65)
    calibrated = replace(rolled, focal_length=np.array((0.9, 1.3)), sensor_size=np.array((0.65, 1.0)), principal_offset=np.array((0.12, -0.09)))
    cases = [
        scenario("off-center-low-elevation", start, center, half),
        scenario("rolled-calibrated-portrait", calibrated, center, half),
        scenario("orthographic-portrait", replace(rolled, orthographic=True, ortho_height=12), center, half),
        scenario("retain-orientation", calibrated, center, half, eyeDirection=None),
        scenario("view-cube-front", rolled, center, half, eyeDirection=[0, -1, 0], margin=1.2),
        scenario("exact-top", replace(start, eye=np.array((0.0, 0.0, 8.0)), up=np.array((1.0, 0.0, 0.0))), np.zeros(3), half),
        scenario("degenerate-bounds", start, np.array((6.0, 0.0, 0.0)), np.zeros(3)),
        scenario("limits", rolled, center, half * 10, minDistance=0.01, maxDistance=3),
        scenario("linear", start, center, half, duration=0.6, easing="linear"),
        scenario("smootherstep", rolled, center, half, duration=0.4, easing="smootherstep"),
        scenario("ease-out-cubic", rolled, center, half, duration=0.25, easing="ease_out_cubic"),
        scenario("instant", rolled, center, half, duration=0),
        scenario("joint-sphere-perspective", rolled, center, half, sphereRadius=0.3, eyeDirection=[0.7, -0.3, 0.8]),
        scenario("joint-sphere-orthographic", replace(rolled, orthographic=True, ortho_height=8), center, half, sphereRadius=0.8, eyeDirection=[0.2, 0.5, 0.5], margin=1.4),
        scenario("view-cube-sphere-quart", rolled, center, half, sphereRadius=float(np.linalg.norm(half)), eyeDirection=[0, -1, 0], sphereMargin=1.2, margin=1.15, duration=0.35, easing="ease_out_quart", viewCube=True),
        scenario("frame-scene", start, center, half, frameScene=True, duration=0.35),
        scenario("frame-scene-orthographic", replace(rolled,orthographic=True,ortho_height=8), center, half, frameScene=True, duration=0.35),
        scenario("frame-scene-empty", rolled, np.zeros(3), np.zeros(3), frameScene=True, duration=0.35),
    ]
    result = {
        "schema": 1,
        "source": {"generator": "electron/scripts/export-camera-focus-fixtures.py", "files": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in ("python/ui/camera.py", "python/math3d.py", "python/config.py")}},
        "tolerance": 2e-5, "cases": cases,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(cases)} native focus cases / {sum(len(case['steps']) for case in cases)} samples.")


if __name__ == "__main__":
    main()
