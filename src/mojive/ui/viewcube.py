"""Orientation view gizmo layout and interaction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..types import CameraView
from .camera import PITCH_LIMIT, OrbitCamera, camera_basis
from .draw2d import Draw2D
from .theme import THEME

RADIUS_PT = 34.0

BALL_PT = 9.5
MARGIN_PT = 10.0
LINE_PT = 2.0


AXIS_NAMES = ("X", "Y", "Z")

TOP_YAW = -90.0


LABEL_FILL = (0.97, 0.97, 0.98, 1.0)

DARK_MIX = 0.72

DARK_BASE = (0.10, 0.11, 0.13)

HOVER_GAIN = 1.35
HOVER_LIFT = 0.10

BACK_FADE_START = 0.72
BACK_FADE_END = 0.96

DEFAULT_SELECTION_PADDING = 1.2
MIN_SELECTION_PADDING = 1.0
MAX_SELECTION_PADDING = 4.0


@dataclass(frozen=True)
class Ball:
    axis: int
    sign: float
    screen: tuple[float, float]
    radius: float
    depth: float

    alpha: float = 1.0

    @property
    def positive(self) -> bool:
        return self.sign > 0.0

    @property
    def label(self) -> str:
        return AXIS_NAMES[self.axis] if self.positive else f"-{AXIS_NAMES[self.axis]}"


def layout(
    cam: CameraView, center: tuple[float, float], radius_pt: float, ball_pt: float
) -> list[Ball]:
    right, up, forward = camera_basis(cam)
    return list(
        _cached_layout(
            tuple(float(value) for value in right),
            tuple(float(value) for value in up),
            tuple(float(value) for value in forward),
            tuple(float(value) for value in center),
            float(radius_pt),
            float(ball_pt),
        )
    )


@lru_cache(maxsize=256)
def _cached_layout(
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
    center: tuple[float, float],
    radius_pt: float,
    ball_pt: float,
) -> tuple[Ball, ...]:
    out: list[Ball] = []
    for axis in range(3):
        world = np.zeros(3)
        for sign in (1.0, -1.0):
            world[:] = 0.0
            world[axis] = sign
            depth = float(np.dot(world, forward))
            sx = center[0] + float(np.dot(world, right)) * radius_pt

            sy = center[1] - float(np.dot(world, up)) * radius_pt
            out.append(
                Ball(
                    axis=axis,
                    sign=sign,
                    screen=(sx, sy),
                    radius=ball_pt,
                    depth=depth,
                    alpha=_back_alpha(depth),
                )
            )
    out.sort(key=lambda b: -b.depth)
    return tuple(out)


def hit_test(balls: list[Ball], cursor: tuple[float, float]) -> Ball | None:
    best: Ball | None = None
    for b in balls:
        if b.alpha <= 0.1:
            continue
        dx = cursor[0] - b.screen[0]
        dy = cursor[1] - b.screen[1]
        if dx * dx + dy * dy <= b.radius * b.radius and (best is None or b.depth < best.depth):
            best = b
    return best


def widget_center(
    rect: tuple[float, float, float, float], style_scale: float
) -> tuple[float, float]:
    r = (RADIUS_PT + BALL_PT) * style_scale
    m = MARGIN_PT * style_scale
    return (rect[0] + rect[2] - m - r, rect[1] + m + r)


def yaw_pitch_for(axis: int, sign: float) -> tuple[float, float]:
    if axis == 2:
        return TOP_YAW, PITCH_LIMIT * (1.0 if sign > 0 else -1.0)
    yaw = 0.0 if axis == 0 else 90.0
    if sign < 0:
        yaw += 180.0
    return yaw, 0.0


class ViewCube:
    def __init__(self, selection_padding: float = DEFAULT_SELECTION_PADDING) -> None:
        self._balls: list[Ball] = []
        self._hover: Ball | None = None
        self._center: tuple[float, float] = (0.0, 0.0)
        self.selection_padding = selection_padding

    @property
    def balls(self) -> list[Ball]:
        return self._balls

    def update(
        self,
        cam: CameraView,
        rect: tuple[float, float, float, float],
        cursor,
        style_scale: float,
        *,
        enabled: bool = True,
    ) -> Ball | None:
        self._center = widget_center(rect, style_scale)
        self._balls = layout(cam, self._center, RADIUS_PT * style_scale, BALL_PT * style_scale)
        self._hover = hit_test(self._balls, cursor) if enabled else None
        return self._hover

    @property
    def hovered(self) -> Ball | None:
        return self._hover

    def drag(self, camera: OrbitCamera, dx: float, dy: float) -> None:
        camera.orbit(dx, dy)

    @property
    def selection_padding(self) -> float:
        return self._selection_padding

    @selection_padding.setter
    def selection_padding(self, value: float) -> None:
        parsed = float(value)
        if not np.isfinite(parsed):
            parsed = DEFAULT_SELECTION_PADDING
        self._selection_padding = float(
            np.clip(parsed, MIN_SELECTION_PADDING, MAX_SELECTION_PADDING)
        )

    def click(self, camera: OrbitCamera, ball: Ball, sink, *, focus=None) -> None:
        yaw, pitch = yaw_pitch_for(ball.axis, ball.sign)
        if focus is None:
            camera.look_from(yaw, pitch, sink)
            return
        center, radius = focus
        camera.look_from_target(
            yaw,
            pitch,
            center,
            radius,
            sink,
            margin=self.selection_padding,
        )

    def draw(self, overlay: Draw2D, style_scale: float = 1.0) -> None:
        if not self._balls:
            return

        if self._hover is not None:
            overlay.circle_filled(
                self._center,
                (RADIUS_PT + BALL_PT + 2.0) * style_scale,
                (0.0, 0.0, 0.0, 0.28),
                segments=32,
            )

        for b in self._balls:
            if b.alpha <= 0.0:
                continue
            hovered = b is self._hover
            rgb = _axis_rgb(b.axis)

            face = (
                (_lift(rgb) if hovered else rgb) if b.positive else (rgb if hovered else _dark(rgb))
            )
            color = (*face, b.alpha)
            if b.positive:
                outline = _lollipop_outline(self._center, b.screen, b.radius, LINE_PT * style_scale)
                overlay.fringed_concave_fill(outline, color)
            else:
                overlay.circle_filled(b.screen, b.radius, color, segments=24)
                overlay.circle(
                    b.screen,
                    b.radius,
                    (*rgb, b.alpha),
                    1.6 * style_scale,
                    segments=24,
                )

            label_alpha = _label_alpha(b, hovered)
            if label_alpha > 0.0:
                overlay.centered_label(
                    b.label, b.screen, (*LABEL_FILL[:3], label_alpha), b.radius * 1.5
                )


def _back_alpha(depth: float) -> float:
    return float(np.clip((BACK_FADE_END - depth) / (BACK_FADE_END - BACK_FADE_START), 0.0, 1.0))


@lru_cache(maxsize=256)
def _lollipop_outline(
    center: tuple[float, float],
    ball: tuple[float, float],
    radius: float,
    line_width: float,
    segments: int = 24,
) -> tuple[tuple[float, float], ...]:
    center_v = np.asarray(center, np.float64)
    ball_v = np.asarray(ball, np.float64)
    direction = ball_v - center_v
    distance = float(np.linalg.norm(direction))
    if distance <= radius:
        angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        return tuple(tuple(ball_v + radius * np.array((np.cos(a), np.sin(a)))) for a in angles)

    direction /= distance
    side = np.array((-direction[1], direction[0]))
    half = min(line_width * 0.5, radius * 0.5)
    angle = float(np.arcsin(half / radius))
    arc = np.linspace(-np.pi + angle, np.pi - angle, segments)
    points = [center_v - side * half]
    points.extend(ball_v + radius * (np.cos(a) * direction + np.sin(a) * side) for a in arc)
    points.append(center_v + side * half)
    return tuple(tuple(point) for point in points)


def _label_alpha(ball: Ball, hovered: bool) -> float:
    if ball.positive or hovered:
        return ball.alpha
    return 1.0 - _back_alpha(-ball.depth)


def _dark(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(  # type: ignore[return-value]
        c * (1.0 - DARK_MIX) + base * DARK_MIX for c, base in zip(rgb, DARK_BASE, strict=True)
    )


def _lift(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(min(1.0, c * HOVER_GAIN + HOVER_LIFT) for c in rgb)  # type: ignore[return-value]


def _axis_rgb(axis: int) -> tuple[float, float, float]:
    return tuple(float(c) for c in THEME.axis_color(axis)[:3])  # type: ignore[return-value]
