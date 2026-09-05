"""Orbit camera controls, animation, and unprojection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .. import math3d
from ..commands import SetCamera
from ..math3d import camera_basis
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
FOCUS_DURATION = 0.36
PROJECTION_DURATION = 0.28
DEFAULT_YAW = 0.0
DEFAULT_PITCH = 0.0
ISO_PITCH = 30.0


PRESETS: dict[str, tuple[float, float]] = {
    "front": (-90.0, 0.0),
    "back": (90.0, 0.0),
    "right": (0.0, 0.0),
    "left": (180.0, 0.0),
    "top": (-90.0, PITCH_LIMIT),
    "bottom": (-90.0, -PITCH_LIMIT),
    "iso": (-135.0, ISO_PITCH),
}


def _ease_out_quad(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * (2.0 - t)


def _ease_out_quart(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 4


def _ease_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def _smoothstep(t: float) -> float:
    """Cubic ease-in/out with zero velocity at both projection endpoints."""

    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _wrap_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def _adopted_yaw(direction: np.ndarray, up) -> float:
    """Recover a stable orbit yaw, including at the world-Z poles."""

    horizontal = float(np.linalg.norm(direction[:2]))
    if horizontal > 1e-9:
        return float(np.degrees(np.arctan2(direction[1], direction[0])))

    # A world-Z orbit cannot use its regular up vector at an exact pole. Use
    # the source camera's screen-up direction to select the equivalent yaw
    # before pitch is moved just inside the representable orbit range.
    screen_up = np.asarray(up, np.float64).reshape(3)
    screen_up = screen_up - direction * float(np.dot(screen_up, direction))
    planar_length = float(np.linalg.norm(screen_up[:2]))
    if np.isfinite(planar_length) and planar_length > 1e-9:
        pole_sign = 1.0 if direction[2] >= 0.0 else -1.0
        heading = -pole_sign * screen_up[:2] / planar_length
        return float(np.degrees(np.arctan2(heading[1], heading[0])))
    return PRESETS["top"][0]


def closest_axis_view_direction(axis, eye_offset) -> np.ndarray:
    """Return the axis direction on the side nearest the current camera eye."""

    direction = _normalized_direction(axis, np.array((0.0, 0.0, 1.0), np.float64))
    offset = np.asarray(eye_offset, np.float64).reshape(3)
    return direction if float(np.dot(direction, offset)) >= 0.0 else -direction


def closest_perpendicular_view_direction(axis, eye_offset, fallback) -> np.ndarray:
    """Return the nearest stable view direction perpendicular to ``axis``."""

    normal = _normalized_direction(axis, np.array((0.0, 0.0, 1.0), np.float64))
    for candidate in (eye_offset, fallback):
        projected = np.asarray(candidate, np.float64).reshape(3)
        projected = projected - normal * float(np.dot(projected, normal))
        length = float(np.linalg.norm(projected))
        if np.isfinite(length) and length > 1e-9:
            return projected / length

    # Both supplied directions can be parallel to the joint axis.  Pick the
    # world basis least aligned with it so the fallback remains deterministic.
    basis = np.eye(3, dtype=np.float64)
    reference = basis[int(np.argmin(np.abs(basis @ normal)))]
    projected = reference - normal * float(np.dot(reference, normal))
    return projected / max(float(np.linalg.norm(projected)), 1e-9)


def oblique_axis_view_directions(
    axis,
    eye_offset,
    fallback,
    angle_degrees: float,
) -> tuple[np.ndarray, ...]:
    """Return nearby elevated views that keep a rotation axis readable.

    Preserve the current azimuth whenever it already gives the joint useful
    depth. Only clamp views that are nearly axial or edge-on, then offer small
    neighboring turns for the one-shot occlusion check. This avoids snapping
    every focus request to one of four global ISO-like quadrants.
    """

    minimum_angle = float(np.clip(angle_degrees, 0.0, 45.0))
    maximum_angle = 90.0 - minimum_angle
    current = _normalized_direction(eye_offset, fallback)
    elevated = elevated_focus_view_direction(current, fallback, minimum_angle)
    normal = _normalized_direction(axis, np.array((0.0, 0.0, 1.0), np.float64))
    current_pitch = float(np.degrees(np.arcsin(np.clip(elevated[2], -1.0, 1.0))))
    current_yaw = float(np.degrees(np.arctan2(elevated[1], elevated[0])))

    def readable(direction: np.ndarray) -> bool:
        axis_angle = float(
            np.degrees(np.arccos(np.clip(abs(np.dot(direction, normal)), -1.0, 1.0)))
        )
        return minimum_angle - 1e-6 <= axis_angle <= maximum_angle + 1e-6

    scored: list[tuple[float, np.ndarray]] = []
    if readable(elevated):
        scored.append((1.0 - float(np.dot(elevated, current)), elevated))

    # This bounded search runs only on a focus request. A five-degree lattice
    # is dense enough to find a small corrective turn while remaining much
    # cheaper than the raycasts performed for the chosen nearby candidates.
    pitch_values = [current_pitch]
    pitch_values.extend(
        value
        for value in np.arange(minimum_angle, PITCH_LIMIT + 2.5, 5.0)
        if abs(float(value) - current_pitch) > 1e-6
    )
    for pitch_degrees in pitch_values:
        pitch = float(np.deg2rad(np.clip(pitch_degrees, minimum_angle, PITCH_LIMIT)))
        horizontal = float(np.cos(pitch))
        height = float(np.sin(pitch))
        for yaw_offset in np.arange(0.0, 360.0, 5.0):
            yaw = np.deg2rad(current_yaw + float(yaw_offset))
            candidate = np.array(
                (horizontal * np.cos(yaw), horizontal * np.sin(yaw), height),
                np.float64,
            )
            if readable(candidate):
                scored.append((1.0 - float(np.dot(candidate, current)), candidate))

    if not scored:
        return (elevated,)
    scored.sort(key=lambda item: item[0])
    candidates: list[np.ndarray] = []
    minimum_separation = float(np.cos(np.deg2rad(10.0)))
    for _turn, candidate in scored:
        if any(float(np.dot(candidate, known)) > minimum_separation for known in candidates):
            continue
        candidates.append(candidate)
        if len(candidates) >= 9:
            break
    return tuple(candidates)


def elevated_focus_view_direction(
    eye_offset,
    fallback,
    minimum_elevation_degrees: float = ISO_PITCH,
) -> np.ndarray:
    """Preserve view azimuth while keeping a focus direction above the target."""

    direction = _normalized_direction(eye_offset, fallback)
    elevation = np.deg2rad(float(np.clip(minimum_elevation_degrees, 0.0, PITCH_LIMIT)))
    minimum_z = float(np.sin(elevation))
    if direction[2] >= minimum_z:
        return direction

    horizontal = direction.copy()
    horizontal[2] = 0.0
    length = float(np.linalg.norm(horizontal))
    if not np.isfinite(length) or length <= 1e-9:
        horizontal = np.asarray(fallback, np.float64).reshape(3).copy()
        horizontal[2] = 0.0
        length = float(np.linalg.norm(horizontal))
    if not np.isfinite(length) or length <= 1e-9:
        horizontal = np.array((1.0, 0.0, 0.0), np.float64)
        length = 1.0
    horizontal /= length
    return horizontal * np.cos(elevation) + np.array((0.0, 0.0, minimum_z), np.float64)


def _normalized_direction(value, fallback) -> np.ndarray:
    direction = np.asarray(value, np.float64).reshape(3)
    length = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or not np.isfinite(length) or length <= 1e-9:
        direction = np.asarray(fallback, np.float64).reshape(3)
        length = float(np.linalg.norm(direction))
    return direction / max(length, 1e-9)


@dataclass
class _Anim:
    pivot: np.ndarray
    distance: float
    yaw: float
    pitch: float
    ortho_height: float
    elapsed: float = 0.0
    duration: float = FRAME_DURATION


@dataclass
class ProjectionTransition:
    """Small reusable state machine for perspective/orthographic morphing."""

    value: float = 0.0
    start: float = 0.0
    target: float = 0.0
    elapsed: float = 0.0
    duration: float = PROJECTION_DURATION
    active: bool = False

    def snap(self, orthographic: bool) -> None:
        self.value = self.start = self.target = 1.0 if orthographic else 0.0
        self.elapsed = 0.0
        self.active = False

    def set(self, orthographic: bool, *, animate: bool) -> None:
        target = 1.0 if orthographic else 0.0
        if not animate:
            self.snap(orthographic)
            return
        if self.active and target == self.target:
            return
        if not self.active and target == self.value:
            return
        self.start = float(self.value)
        self.target = target
        self.elapsed = 0.0
        self.active = True

    def finish(self) -> None:
        if self.active:
            self.snap(self.target >= 0.5)

    def advance(self, dt: float) -> bool:
        if not self.active:
            return False
        self.elapsed += max(0.0, float(dt))
        t = min(1.0, self.elapsed / max(self.duration, 1e-6))
        eased = _smoothstep(t)
        self.value = self.start + (self.target - self.start) * eased
        if t >= 1.0:
            self.snap(self.target >= 0.5)
        return True


class OrbitCamera:
    def __init__(
        self,
        pivot=None,
        distance: float = 4.0,
        yaw: float = DEFAULT_YAW,
        pitch: float = DEFAULT_PITCH,
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
        self._projection = ProjectionTransition()
        self._projection.snap(self._orthographic)
        self._adaptive_near = True
        self._anim: _Anim | None = None

        self._anim_start: _Anim | None = None
        self._anim_easing: Callable[[float], float] = _ease_out_quad
        self._dirty = True

        self._out: CameraSink | None = None
        self._exact_view: CameraView | None = None

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
        self.set_aspect(value)

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

    def adopt(self, view: CameraView, *, exact: bool = False) -> None:
        """Seed the orbit camera; optionally retain the exact view until a gesture."""
        eye = np.asarray(view.eye, np.float64)
        target = np.asarray(view.target, np.float64)
        delta = eye - target
        distance = max(float(np.linalg.norm(delta)), MIN_DISTANCE)
        direction = delta / distance
        self._stop_anim()
        self._pivot = target.copy()
        self._distance = distance
        self._yaw = _adopted_yaw(direction, view.up)
        self._pitch = float(
            np.clip(
                np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))),
                -PITCH_LIMIT,
                PITCH_LIMIT,
            )
        )
        self._fov_y = float(view.fov_y)
        self.near = float(view.near)
        self.far = float(view.far)
        self._adaptive_near = False
        self._aspect = float(view.aspect)
        self._orthographic = bool(view.orthographic)
        self.ortho_height = float(view.ortho_height)
        self._projection.snap(self._orthographic)
        self._touch()
        self._exact_view = view if exact else None

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
        if self._exact_view is not None:
            return self._exact_view.with_aspect(self._aspect)
        near = max(MIN_NEAR, float(self.near))
        if self._adaptive_near:
            near = min(near, self._distance * NEAR_DISTANCE_FRACTION)
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
            orthographic_blend=(float(self._projection.value) if self._projection.active else None),
        )

    def publish(self, sink: CameraSink) -> CameraView:
        v = self.view()
        sink.set_camera(v)
        self._dirty = False
        return v

    def _touch(self) -> None:
        self._exact_view = None
        self._dirty = True

    def _stop_anim(self) -> None:
        self._anim = None
        self._anim_start = None
        self._anim_easing = _ease_out_quad
        self._projection.finish()

    @property
    def dirty(self) -> bool:
        return self._dirty

    def set_aspect(self, aspect: float) -> None:
        aspect = float(aspect)
        if abs(aspect - self._aspect) > 1e-9:
            self._aspect = aspect
            self._dirty = True

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

    def set_orthographic(self, on: bool, *, animate: bool = False) -> None:
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
        self._projection.set(on, animate=animate)
        self._touch()

    def frame_scene(
        self,
        bounds: tuple[Sequence[float], Sequence[float]],
        sink: CameraSink,
        *,
        animate: bool = True,
        clip: CameraView | None = None,
        minimum_pitch: float | None = None,
    ) -> CameraView:
        lo = np.asarray(bounds[0], np.float64).reshape(3)
        hi = np.asarray(bounds[1], np.float64).reshape(3)
        center = (lo + hi) * 0.5
        radius = float(np.linalg.norm(hi - lo) * 0.5)
        if not np.isfinite(radius) or radius < 1e-6:
            radius = 0.5
        distance, ortho_height = self._framing_distance(radius, FRAME_MARGIN)

        if clip is None:
            near = max(1e-3, (distance - radius) * 0.25)
            far = (distance + radius) * FRAME_FAR_MARGIN
            self._adaptive_near = True
        else:
            near = max(MIN_NEAR, float(clip.near))
            far = max(near, float(clip.far))
            self._adaptive_near = False
        goal = _Anim(
            pivot=center,
            distance=distance,
            yaw=self._yaw,
            pitch=(
                self._pitch
                if minimum_pitch is None
                else max(self._pitch, float(np.clip(minimum_pitch, -PITCH_LIMIT, PITCH_LIMIT)))
            ),
            ortho_height=ortho_height,
        )
        self.near, self.far = float(near), float(far)
        self._apply_goal(goal, animate=animate)
        return self.publish(sink)

    def look_from_target(
        self,
        yaw: float,
        pitch: float,
        center,
        radius: float,
        sink: CameraSink,
        *,
        margin: float = FRAME_MARGIN,
        animate: bool = True,
    ) -> CameraView:
        """Align to one direction while framing a selected target sphere."""

        target = np.asarray(center, np.float64).reshape(3)
        target_radius = max(float(radius), 1e-6)
        if not np.isfinite(target).all() or not np.isfinite(target_radius):
            return self.look_from(yaw, pitch, sink, animate=animate)
        distance, ortho_height = self._framing_distance(target_radius, margin)
        self.near = min(self.near, max(MIN_NEAR, (distance - target_radius) * 0.25))
        self.far = max(self.far, (distance + target_radius) * 2.0)
        goal = _Anim(
            pivot=target.copy(),
            distance=distance,
            yaw=float(yaw),
            pitch=float(np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT)),
            ortho_height=ortho_height,
        )
        self._apply_goal(goal, animate=animate, easing=_ease_out_quart)
        return self.publish(sink)

    def focus_target(
        self,
        center,
        radius: float,
        eye_direction,
        sink: CameraSink,
        *,
        margin: float = FRAME_MARGIN,
        animate: bool = True,
    ) -> CameraView:
        """Frame a target from a world direction with a gentle focus transition."""

        direction = _normalized_direction(eye_direction, self.direction())
        yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
        pitch = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
        target = np.asarray(center, np.float64).reshape(3)
        target_radius = max(float(radius), 1e-6)
        if not np.isfinite(target).all() or not np.isfinite(target_radius):
            return self.look_from(yaw, pitch, sink, animate=animate)
        distance, ortho_height = self._framing_distance(target_radius, margin)
        self.near = min(self.near, max(MIN_NEAR, (distance - target_radius) * 0.25))
        self.far = max(self.far, (distance + target_radius) * 2.0)
        goal = _Anim(
            pivot=target.copy(),
            distance=distance,
            yaw=yaw,
            pitch=float(np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT)),
            ortho_height=ortho_height,
            duration=FOCUS_DURATION,
        )
        self._apply_goal(goal, animate=animate, easing=_ease_out_cubic)
        return self.publish(sink)

    def _framing_distance(self, radius: float, margin: float) -> tuple[float, float]:
        half_y = self._fov_y * 0.5
        half_x = float(np.arctan(np.tan(half_y) * max(self._aspect, 1e-3)))
        padding = max(1.0, float(margin))
        distance = max(
            MIN_DISTANCE,
            float(radius) / max(float(np.sin(min(half_y, half_x))), 1e-3) * padding,
        )
        return distance, 2.0 * distance * float(np.tan(half_y))

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

    def _apply_goal(
        self,
        goal: _Anim,
        *,
        animate: bool,
        easing: Callable[[float], float] = _ease_out_quad,
    ) -> None:
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
            self._anim_easing = easing
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
        return self._anim is not None or self._projection.active

    def advance(self, dt: float, sink: CameraSink) -> bool:
        anim = self._anim
        start = self._anim_start
        if anim is not None and start is not None:
            anim.elapsed += max(0.0, float(dt))
            t = self._anim_easing(anim.elapsed / max(anim.duration, 1e-6))
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
                self._anim_easing = _ease_out_quad
        if self._projection.advance(dt):
            self._touch()
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
