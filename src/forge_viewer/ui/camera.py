from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .. import math3d
from ..commands import SetCamera
from ..types import CameraView

if TYPE_CHECKING:
    from collections.abc import Sequence


class CameraSink(Protocol):
    def set_camera(self, camera: CameraView) -> None: ...


@dataclass
class CameraOut:
    backend: CameraSink
    session: Any = None

    def set_camera(self, camera: CameraView) -> None:
        self.backend.set_camera(camera)
        if self.session is not None:
            self.session.submit(SetCamera(camera))


PITCH_LIMIT = 89.9
MIN_DISTANCE = 1e-3
MIN_NEAR = 1e-4
NEAR_DISTANCE_FRACTION = 0.0025

ORBIT_DEG_PER_PIXEL = 0.35
DOLLY_PER_STEP = 0.12
FLY_RATE = 1.6
FRAME_MARGIN = 1.15
FRAME_FAR_MARGIN = 32.0

FRAME_DURATION = 0.35


PRESETS: dict[str, tuple[float, float]] = {
    "front": (-90.0, 0.0),
    "back": (90.0, 0.0),
    "right": (0.0, 0.0),
    "left": (180.0, 0.0),
    "top": (-90.0, PITCH_LIMIT),
    "bottom": (-90.0, -PITCH_LIMIT),
    "iso": (-135.0, 30.0),
}


def camera_basis(view: CameraView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    forward = np.asarray(view.forward(), np.float64)
    right = np.cross(forward, np.asarray(view.up, np.float64))
    n = np.linalg.norm(right)
    right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    return right, up, forward


def _ease_out_quad(t: float) -> float:

    t = min(1.0, max(0.0, t))
    return t * (2.0 - t)


def _wrap_deg(a: float) -> float:

    return float((a + 180.0) % 360.0 - 180.0)


@dataclass
class _Anim:
    pivot: np.ndarray
    distance: float
    yaw: float
    pitch: float
    ortho_height: float
    elapsed: float = 0.0
    duration: float = FRAME_DURATION


class OrbitCamera:
    def __init__(
        self,
        pivot=None,
        distance: float = 4.0,
        yaw: float = -135.0,
        pitch: float = 25.0,
        fov_y_deg: float = 45.0,
        near: float = 0.02,
        far: float = 200.0,
        aspect: float = 1.0,
        orthographic: bool = False,
        ortho_height: float = 4.0,
    ) -> None:
        self._pivot = (
            np.zeros(3, np.float64)
            if pivot is None
            else np.asarray(pivot, np.float64).reshape(3).copy()
        )
        self._distance = float(distance)
        self._yaw = float(yaw)
        self._pitch = float(np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT))
        self._fov_y = float(np.radians(fov_y_deg))
        self.near = float(near)
        self.far = float(far)
        self._aspect = float(aspect)
        self._orthographic = bool(orthographic)
        self.ortho_height = float(ortho_height)
        self._anim: _Anim | None = None

        self._anim_start: _Anim | None = None
        self._dirty = True

        self._out: CameraSink | None = None

    def __repr__(self) -> str:
        return (
            f"OrbitCamera(pivot={self.pivot.tolist()}, distance={self._distance:.3f}, "
            f"yaw={self._yaw:.1f}°, pitch={self._pitch:.1f}°)"
        )

    @property
    def pivot(self) -> np.ndarray:

        return self._pivot

    @pivot.setter
    def pivot(self, value) -> None:
        self._pivot = np.asarray(value, np.float64).reshape(3).copy()
        self._touch()

    @property
    def yaw(self) -> float:

        return self._yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._yaw = float(value)
        self._stop_anim()
        self._touch()

    @property
    def pitch(self) -> float:

        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._pitch = float(np.clip(float(value), -PITCH_LIMIT, PITCH_LIMIT))
        self._stop_anim()
        self._touch()

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        self._distance = max(MIN_DISTANCE, float(value))
        self._stop_anim()
        self._touch()

    @property
    def aspect(self) -> float:
        return self._aspect

    @aspect.setter
    def aspect(self, value: float) -> None:
        self._aspect = float(value)
        self._touch()

    @property
    def fov_y(self) -> float:

        return self._fov_y

    @property
    def orthographic(self) -> bool:
        return self._orthographic

    @orthographic.setter
    def orthographic(self, value: bool) -> None:

        self.set_orthographic(bool(value))

    def attach(self, sink: CameraSink) -> None:

        self._out = sink

    def _require_out(self) -> CameraSink:

        if self._out is None:
            raise RuntimeError("Attach a camera sink before publishing")
        return self._out

    @property
    def fov_y_deg(self) -> float:

        return float(np.degrees(self.fov_y))

    @fov_y_deg.setter
    def fov_y_deg(self, value: float) -> None:
        fov = float(np.clip(np.radians(float(value)), np.radians(5.0), np.radians(150.0)))
        if abs(fov - self._fov_y) > 1e-9:
            self._fov_y = fov
            if self._orthographic:
                self.ortho_height = self.matched_ortho_height()
            self._touch()

    def frame_all(self, lo, hi) -> CameraView:

        return self.frame_scene((lo, hi), self._require_out())

    def set_preset(self, name: str) -> CameraView:

        yaw, pitch = PRESETS.get(name.lower(), (None, None))
        if yaw is None:
            raise ValueError(f"Unknown camera preset: {name}")
        return self.look_from(yaw, pitch, self._require_out())

    def adopt(self, view: CameraView) -> None:
        """Seed the free orbit camera from an arbitrary scene camera without publishing it."""
        eye = np.asarray(view.eye, np.float64)
        target = np.asarray(view.target, np.float64)
        delta = eye - target
        distance = max(float(np.linalg.norm(delta)), MIN_DISTANCE)
        direction = delta / distance
        self._stop_anim()
        self._pivot = target.copy()
        self._distance = distance
        self._yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
        self._pitch = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
        self._fov_y = float(view.fov_y)
        self.near = float(view.near)
        self.far = float(view.far)
        self._aspect = float(view.aspect)
        self._orthographic = bool(view.orthographic)
        self.ortho_height = float(view.ortho_height)
        self._touch()

    def direction(self) -> np.ndarray:

        yaw = np.radians(self._yaw)
        pitch = np.radians(self._pitch)
        cp = np.cos(pitch)
        return np.array([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)], np.float64)

    def eye(self) -> np.ndarray:
        return self.pivot + self.direction() * self.distance

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        return camera_basis(self.view())

    def view(self) -> CameraView:

        near = max(MIN_NEAR, min(float(self.near), self._distance * NEAR_DISTANCE_FRACTION))
        return CameraView(
            eye=self.eye().astype(np.float32),
            target=self.pivot.astype(np.float32),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            fov_y=float(self._fov_y),
            near=near,
            far=float(self.far),
            aspect=float(self._aspect),
            orthographic=bool(self._orthographic),
            ortho_height=float(self.ortho_height),
        )

    def publish(self, sink: CameraSink) -> CameraView:

        v = self.view()
        sink.set_camera(v)
        self._dirty = False
        return v

    def _touch(self) -> None:
        self._dirty = True

    def _stop_anim(self) -> None:

        self._anim = None
        self._anim_start = None

    @property
    def dirty(self) -> bool:
        return self._dirty

    def set_aspect(self, aspect: float) -> None:
        aspect = float(aspect)
        if abs(aspect - self._aspect) > 1e-9:
            self._aspect = aspect
            self._touch()

    def orbit(self, dx_px: float, dy_px: float) -> None:

        self._yaw -= float(dx_px) * ORBIT_DEG_PER_PIXEL
        self._pitch = float(
            np.clip(self._pitch + float(dy_px) * ORBIT_DEG_PER_PIXEL, -PITCH_LIMIT, PITCH_LIMIT)
        )
        self._stop_anim()
        self._touch()

    def pan(self, dx_px: float, dy_px: float, viewport_h_px: float) -> None:

        h = max(float(viewport_h_px), 1.0)
        world_per_px = self._image_height() / h
        right, up, _ = self.basis()
        self.pivot = self.pivot - right * (dx_px * world_per_px) + up * (dy_px * world_per_px)
        self._stop_anim()
        self._touch()

    def dolly(self, steps: float) -> None:

        self._distance = max(MIN_DISTANCE, self._distance * float(np.exp(-steps * DOLLY_PER_STEP)))
        if self._orthographic:
            self.ortho_height = self.matched_ortho_height()
        self._stop_anim()
        self._touch()

    def fly(self, dt: float, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> None:

        speed = self._distance * FLY_RATE * float(dt)
        if speed <= 0.0:
            return
        r, _, f = self.basis()
        world_up = np.array([0.0, 0.0, 1.0])
        self.pivot = self.pivot + (f * forward + r * right + world_up * up) * speed
        self._stop_anim()
        self._touch()

    def _image_height(self) -> float:

        if self._orthographic:
            return float(self.ortho_height)
        return 2.0 * self._distance * float(np.tan(self._fov_y * 0.5))

    def matched_ortho_height(self) -> float:
        return self.view().matched_ortho_height()

    def set_orthographic(self, on: bool) -> None:

        on = bool(on)
        if on == self._orthographic:
            return
        if on:
            self.ortho_height = self.matched_ortho_height()
        else:
            self._distance = max(
                MIN_DISTANCE, self.ortho_height * 0.5 / float(np.tan(self._fov_y * 0.5))
            )
        self._orthographic = on
        self._touch()

    def frame_scene(
        self,
        bounds: tuple[Sequence[float], Sequence[float]],
        sink: CameraSink,
        *,
        animate: bool = True,
    ) -> CameraView:

        lo = np.asarray(bounds[0], np.float64).reshape(3)
        hi = np.asarray(bounds[1], np.float64).reshape(3)
        center = (lo + hi) * 0.5
        radius = float(np.linalg.norm(hi - lo) * 0.5)
        if not np.isfinite(radius) or radius < 1e-6:
            radius = 0.5

        half_y = self._fov_y * 0.5
        half_x = float(np.arctan(np.tan(half_y) * max(self._aspect, 1e-3)))

        distance = radius / max(float(np.sin(min(half_y, half_x))), 1e-3) * FRAME_MARGIN

        near = max(1e-3, (distance - radius) * 0.25)
        far = (distance + radius) * FRAME_FAR_MARGIN
        goal = _Anim(
            pivot=center,
            distance=distance,
            yaw=self._yaw,
            pitch=self._pitch,
            ortho_height=2.0 * distance * float(np.tan(half_y)),
        )
        self.near, self.far = float(near), float(far)
        self._apply_goal(goal, animate=animate)
        return self.publish(sink)

    def look_from(self, yaw: float, pitch: float, sink: CameraSink, *, animate: bool = True):

        goal = _Anim(
            pivot=self.pivot.copy(),
            distance=self._distance,
            yaw=float(yaw),
            pitch=float(np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT)),
            ortho_height=self.ortho_height,
        )
        self._apply_goal(goal, animate=animate)
        return self.publish(sink)

    def _apply_goal(self, goal: _Anim, *, animate: bool) -> None:
        if animate:
            goal.yaw = self.yaw + _wrap_deg(goal.yaw - self.yaw)
            self._anim_start = _Anim(
                pivot=self.pivot.copy(),
                distance=float(self._distance),
                yaw=float(self._yaw),
                pitch=float(self._pitch),
                ortho_height=float(self.ortho_height),
            )
            self._anim = goal
        else:
            self._stop_anim()
            self.pivot = goal.pivot.copy()
            self._distance = goal.distance
            self._yaw = goal.yaw
            self._pitch = goal.pitch
            self.ortho_height = goal.ortho_height
        self._touch()

    @property
    def animating(self) -> bool:
        return self._anim is not None

    def advance(self, dt: float, sink: CameraSink) -> bool:

        anim = self._anim
        start = self._anim_start
        if anim is not None and start is not None:
            anim.elapsed += max(0.0, float(dt))
            t = _ease_out_quad(anim.elapsed / max(anim.duration, 1e-6))
            self.pivot = start.pivot + (anim.pivot - start.pivot) * t

            self._distance = float(
                np.exp(
                    np.log(max(start.distance, MIN_DISTANCE)) * (1.0 - t)
                    + np.log(max(anim.distance, MIN_DISTANCE)) * t
                )
            )
            self._yaw = start.yaw + (anim.yaw - start.yaw) * t
            self._pitch = start.pitch + (anim.pitch - start.pitch) * t
            self.ortho_height = start.ortho_height + (anim.ortho_height - start.ortho_height) * t
            self._touch()
            if anim.elapsed >= anim.duration:
                self._anim = None
                self._anim_start = None
        if self._dirty:
            self.publish(sink)
            return True
        return False


def unproject(view: CameraView, ndc_x: float, ndc_y: float) -> tuple[np.ndarray, np.ndarray]:

    return math3d.unproject_ray(ndc_x, ndc_y, view.view_matrix(), view.proj_matrix())


def ndc_from_viewport(
    x_px: float, y_px: float, rect: tuple[float, float, float, float]
) -> tuple[float, float]:

    rx, ry, rw, rh = rect
    nx = (float(x_px) - rx) / max(rw, 1.0) * 2.0 - 1.0
    ny = 1.0 - (float(y_px) - ry) / max(rh, 1.0) * 2.0
    return nx, ny
