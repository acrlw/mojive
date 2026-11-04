"""Export orbit references from Mojive's native camera implementation.

Run from the repository root with its Python environment:
    .venv/bin/python electron/scripts/export-camera-orbit-fixtures.py

The inputs are authored here; every expected pose and basis is obtained from
the production OrbitCamera and CameraView rather than a second implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mojive.types import CameraView
from mojive.ui.camera import ORBIT_DEG_PER_PIXEL, PITCH_LIMIT, OrbitCamera


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/camera-orbit-native.json"


def view_data(view: CameraView) -> dict:
    return {
        "eye": view.eye.tolist(),
        "target": view.target.tolist(),
        "up": view.up.tolist(),
        "fovY": float(np.degrees(view.fov_y)),
        "near": float(view.near),
        "far": float(view.far),
        "aspect": float(view.aspect),
        "orthographic": bool(view.orthographic),
        "orthoHeight": float(view.ortho_height),
    }


def camera_data(camera: OrbitCamera) -> dict:
    view = camera.view()
    matrix = view.view_matrix()
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6)
    return {
        **view_data(view),
        "screenUp": matrix[1, :3].tolist(),
        "yaw": float(camera.yaw),
        "pitch": float(camera.pitch),
        "distance": float(camera.distance),
    }


def reference(name: str, view: CameraView, dx: float, dy: float) -> dict:
    camera = OrbitCamera()
    camera.adopt(view, exact=True)
    adopted = camera_data(camera)
    np.testing.assert_array_equal(camera.view().eye, view.eye)
    np.testing.assert_array_equal(camera.view().up, view.up)
    camera.orbit(dx, dy)
    expected = camera_data(camera)
    # Orbit intentionally clears the exact view, including on a zero delta.
    np.testing.assert_array_equal(camera.view().up, [0, 0, 1])
    np.testing.assert_array_equal(camera.view().target, view.target)
    return {
        "id": name,
        "initial": view_data(view),
        "adopted": adopted,
        "gesture": {"dx": dx, "dy": dy},
        "expected": expected,
    }


def main() -> None:
    optics = dict(fov_y=np.deg2rad(53), near=0.003, far=700.0, aspect=1.7)
    target = np.array([0.5, -0.25, 1.25], np.float32)
    yaw, pitch, roll = np.radians([-135, 25, 30])
    backward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0])
    level_up = np.cross(backward, right)
    rolled_up = -np.sin(roll) * right + np.cos(roll) * level_up
    oblique = CameraView(
        eye=(target + backward * 5.75).astype(np.float32),
        target=target,
        up=rolled_up.astype(np.float32),
        **optics,
    )
    cases = [
        reference(f"oblique-roll30-{name}", oblique, dx, dy)
        for name, dx, dy in (
            ("zero", 0, 0),
            ("small", 1, -1),
            ("orbit", 40, -15),
            ("opposite", -80, 30),
            ("pitch-clamp", 125, 250),
        )
    ]
    for pole_sign, pole_name in ((1, "top"), (-1, "bottom")):
        for heading in (0, 45, 90, 180):
            angle = np.deg2rad(heading)
            view = CameraView(
                eye=target + np.array([0, 0, pole_sign * 5.75], np.float32),
                target=target,
                up=np.array([np.cos(angle), np.sin(angle), 0], np.float32),
                **optics,
            )
            for name, dx, dy in (
                ("zero", 0, 0),
                ("small-inward", 1, -pole_sign),
                ("inward", 40, -pole_sign * 20),
                ("outward-clamp", -30, pole_sign * 15),
            ):
                cases.append(reference(f"{pole_name}-up{heading}-{name}", view, dx, dy))
    result = {
        "schema": 1,
        "source": {
            "generator": "electron/scripts/export-camera-orbit-fixtures.py",
            "files": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                for path in ("python/ui/camera.py", "python/types.py", "python/math3d.py")
            },
            "operation": "OrbitCamera.adopt(CameraView(**initial), exact=True); OrbitCamera.orbit(dx, dy)",
        },
        "units": {"angles": "degrees", "fovY": "degrees", "gesture": "logical pixels"},
        "tolerance": 1e-5,
        "orbitDegreesPerPixel": ORBIT_DEG_PER_PIXEL,
        "pitchLimit": PITCH_LIMIT,
        "notes": [
            "Initial exact adoption retains the authored roll and the exact pole position.",
            "An orbit call clears exact adoption and uses world-Z up, including for zero deltas.",
            "screenUp is row 1 of the native CameraView.view_matrix(), not CameraView.up.",
            "Pole heading comes from authored screen-up before pitch is clamped inside +/-89.9 degrees.",
        ],
        "cases": cases,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(cases)} native camera orbit fixtures to {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
