"""Camera state and parameter updates independent of viewer gestures and animation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .control_errors import ControlError
from .control_schema import CAMERA_FIELDS, camera_value, json_value
from .types import CameraView


class CameraState:
    """Own an exact CameraView for independent capture settings."""

    def __init__(self, view: CameraView):
        self._view = view

    def view(self) -> CameraView:
        """Return the exact current camera, including roll and physical intrinsics."""
        return self._view

    def adopt(self, view: CameraView) -> None:
        """Replace this camera state without orbit-controller reconstruction."""
        self._view = view

    @property
    def yaw(self) -> float:
        """Return the orbit azimuth in degrees for legacy bookmark interoperability."""
        delta = self._view.eye - self._view.target
        return float(np.degrees(np.arctan2(delta[1], delta[0])))

    @property
    def pitch(self) -> float:
        """Return the orbit elevation in degrees for legacy bookmark interoperability."""
        delta = self._view.eye - self._view.target
        return float(np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]))))

    @property
    def distance(self) -> float:
        """Return the world-space eye-to-target distance."""
        return float(self._view.distance())


def update_camera(current: CameraView, params: dict, session) -> tuple[CameraView, int]:
    """Build a validated camera update without mutating existing state."""
    if "camera_id" in params:
        if len(params) != 1:
            raise ControlError(
                "invalid_params", "camera_id cannot be combined with free-camera fields"
            )
        view = session.camera_view(params["camera_id"])
        if view is None:
            raise ControlError("not_found", f"Camera {params['camera_id']} is unavailable")
        return view, params["camera_id"]
    source = params.get("source", -1)
    if source >= 0:
        view = session.camera_view(source)
        if view is None:
            raise ControlError("not_found", f"Camera {source} is unavailable")
        return view, source
    values = {name: json_value(getattr(current, name)) for name in CAMERA_FIELDS}
    values.update({name: value for name, value in params.items() if name in CAMERA_FIELDS})
    view = camera_value(values)
    if any(name in params for name in ("yaw", "pitch", "distance")):
        if "eye" in params:
            raise ControlError(
                "invalid_params", "eye cannot be combined with yaw, pitch, or distance"
            )
        orbit = CameraState(view)
        yaw = np.deg2rad(params.get("yaw", orbit.yaw))
        pitch = np.deg2rad(params.get("pitch", orbit.pitch))
        distance = params.get("distance", orbit.distance)
        direction = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        view = replace(view, eye=np.asarray(view.target + direction * distance, np.float32))
        view = camera_value({name: json_value(getattr(view, name)) for name in CAMERA_FIELDS})
    return view, -1
