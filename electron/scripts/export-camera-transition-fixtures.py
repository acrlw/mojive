"""Export scene-camera handoffs directly from the native CameraViewTransition.

Run with .venv/bin/python electron/scripts/export-camera-transition-fixtures.py.
The JavaScript module is deliberately not involved in fixture generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mojive.types import CameraView
from mojive.ui.camera import CameraViewTransition


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/camera-transition-native.json"


def data(view: CameraView) -> dict:
    value = {
        "projection": "orthographic" if view.orthographic else "perspective",
        "position": view.eye.tolist(),
        "target": view.target.tolist(),
        "up": view.up.tolist(),
        "fov": float(np.degrees(view.fov_y)),
        "near": float(view.near),
        "far": float(view.far),
        "orthoHeight": float(view.ortho_height),
        "sensorWidth": 35,
        "focalLength": 35 / max(view.aspect, 1) / (2 * np.tan(view.fov_y / 2)),
        "resolution": [view.aspect * 1000, 1000],
    }
    if view.orthographic_blend is not None:
        value["projectionBlend"] = view.projection_blend()
    if view.uses_intrinsics():
        value["normalizedFocal"] = (view.focal_length / view.sensor_size).tolist()
        value["principalOffset"] = (view.principal_offset / view.sensor_size).tolist()
    return value


def output(view: CameraView) -> dict:
    result = {
        "view": data(view),
        "viewMatrix": view.view_matrix().tolist(),
        "projectionMatrix": view.proj_matrix().tolist(),
    }
    assert np.isfinite(result["viewMatrix"]).all()
    assert np.isfinite(result["projectionMatrix"]).all()
    return result


def scenario(name, start, target, *, moving=False, interruption=None):
    transition = CameraViewTransition(start)
    steps = []
    previous = 0.0
    for elapsed in (0, 0.014, 0.035, 0.07, 0.14, 0.21, 0.279, 0.28, 0.35):
        current_target = target
        if moving:
            offset = np.array((elapsed * 1.5, -elapsed * 0.4, elapsed * 0.2))
            current_target = replace(
                target,
                eye=target.eye + offset,
                target=target.target + offset * 0.3,
                fov_y=target.fov_y + np.deg2rad(elapsed * 12),
            )
        sampled = transition.advance(current_target, elapsed - previous)
        steps.append({"elapsed": elapsed, "target": data(current_target), "expected": output(sampled)})
        if elapsed >= transition.duration:
            assert sampled is current_target
        previous = elapsed
    result = {"id": name, "start": data(start), "duration": transition.duration, "steps": steps}
    if interruption is not None:
        result["interruption"] = interruption
    return result


def main():
    start = CameraView(
        eye=np.array((4.5, -6.0, 3.0)),
        target=np.array((0.25, -0.5, 0.75)),
        up=np.array((0.2, 0.3, 1.0)),
        fov_y=np.deg2rad(48),
        near=0.012,
        far=350,
        aspect=1.6,
        ortho_height=7,
    )
    target = CameraView(
        eye=np.array((-2.5, 3.5, 1.25)),
        target=np.array((1.5, 0.25, -0.5)),
        up=np.array((1.0, 0.3, 0.2)),
        fov_y=np.deg2rad(65),
        near=0.08,
        far=800,
        aspect=1.25,
        ortho_height=4,
    )
    calibrated = replace(
        target,
        focal_length=np.array((0.018, 0.020)),
        sensor_size=np.array((0.036, 0.024)),
        principal_offset=np.array((0.002, -0.001)),
    )
    opposite_start = replace(start, eye=np.array((0.0, -3.0, 1.0)), target=np.zeros(3), up=np.array((0.0, 0.0, 1.0)))
    opposite_target = replace(calibrated, eye=-opposite_start.eye, target=np.zeros(3), up=np.array((1.0, 0.0, 0.0)))
    close_target = replace(start, eye=start.eye + np.array((0.0001, -0.0002, 0.0003)), near=0.025, aspect=0.8)
    cases = [
        scenario("perspective-rolled", start, target),
        scenario("perspective-calibrated", start, calibrated),
        scenario("orthographic-rolled", replace(start, orthographic=True), replace(target, orthographic=True)),
        scenario("perspective-to-orthographic-opposing", opposite_start, replace(opposite_target, orthographic=True)),
        scenario("orthographic-to-perspective-opposing", replace(opposite_start, orthographic=True), opposite_target),
        scenario("nearly-parallel-nlerp", start, close_target),
        scenario("moving-target", start, calibrated, moving=True),
    ]
    interrupted_target = replace(calibrated, orthographic=True)
    interrupted_at = 0.093
    intermediate = CameraViewTransition(start).advance(interrupted_target, interrupted_at)
    cases.append(scenario(
        "interrupted-mid-projection",
        intermediate,
        opposite_start,
        interruption={"start": data(start), "target": data(interrupted_target), "elapsed": interrupted_at},
    ))
    result = {
        "schema": 1,
        "source": {
            "generator": "electron/scripts/export-camera-transition-fixtures.py",
            "operation": "CameraViewTransition(start).advance(current_target, delta_seconds)",
            "files": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                for path in ("python/ui/camera.py", "python/types.py", "python/math3d.py")
            },
        },
        "units": {"fov": "degrees", "elapsed": "seconds", "matrices": "row-major"},
        "tolerance": 2e-5,
        "cases": cases,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(cases)} native transitions / {sum(len(case['steps']) for case in cases)} samples.")


if __name__ == "__main__":
    main()
