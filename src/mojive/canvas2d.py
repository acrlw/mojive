"""Retained 2D geometry-debugging canvas over Mojive's shared debug renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import math3d
from .render.debugdraw import NEVER, DebugDraw, Layer, Occlusion
from .types import CameraView


def _unit(value) -> np.ndarray:
    result = np.asarray(value, np.float32).reshape(3)
    length = float(np.linalg.norm(result))
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("canvas axes must be finite non-zero vectors")
    return result / length


@lru_cache(maxsize=32)
def _unit_circle(segments: int) -> np.ndarray:
    count = min(512, max(8, int(segments)))
    angle = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False, dtype=np.float32)
    return np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)


class CanvasLayer2D:
    """One retained, hideable layer expressed in canvas XY coordinates."""

    def __init__(self, canvas: Canvas2D, layer: Layer, depth: float) -> None:
        self._canvas = canvas
        self._layer = layer
        self.depth = float(depth)

    @property
    def name(self) -> str:
        return self._layer.name.removeprefix("canvas2d:")

    @property
    def visible(self) -> bool:
        return self._layer.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._layer.visible = bool(value)

    def world(self, point) -> np.ndarray:
        """Map one canvas point to the configured world plane."""

        return self._canvas.world(point, depth=self.depth)

    def _points(self, points) -> np.ndarray:
        value = np.asarray(points, np.float32).reshape(-1, 2)
        return (
            self._canvas.origin
            + value[:, :1] * self._canvas.x_axis
            + value[:, 1:] * self._canvas.y_axis
            + self.depth * self._canvas.normal
        )

    def line(self, ident: str, a, b, color, width_px: float = 1.5, duration: float = NEVER) -> None:
        self._layer.line(ident, self.world(a), self.world(b), color, width_px, duration)

    def lines(
        self, ident: str, points_a, points_b, color, width_px: float = 1.5, duration: float = NEVER
    ) -> None:
        self._layer.lines(
            ident,
            self._points(points_a),
            self._points(points_b),
            color,
            width_px,
            duration,
        )

    def polyline(
        self,
        ident: str,
        points,
        color,
        width_px: float = 1.5,
        *,
        closed: bool = False,
        duration: float = NEVER,
    ) -> None:
        self._layer.polyline(
            ident,
            self._points(points),
            color,
            width_px,
            closed=closed,
            duration=duration,
        )

    def polygon(
        self, ident: str, points, color, width_px: float = 1.5, duration: float = NEVER
    ) -> None:
        """Create or update a closed polygon outline."""

        self.polyline(ident, points, color, width_px, closed=True, duration=duration)

    def rectangle(
        self, ident: str, lo, hi, color, width_px: float = 1.5, duration: float = NEVER
    ) -> None:
        x0, y0 = np.asarray(lo, np.float32).reshape(2)
        x1, y1 = np.asarray(hi, np.float32).reshape(2)
        self.polygon(
            ident,
            ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            color,
            width_px,
            duration,
        )

    def circle(
        self,
        ident: str,
        center,
        radius: float,
        color,
        width_px: float = 1.5,
        *,
        segments: int = 48,
        duration: float = NEVER,
    ) -> None:
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0.0:
            self.erase(ident)
            return
        points = _unit_circle(segments) * radius + np.asarray(center, np.float32).reshape(2)
        self.polyline(ident, points, color, width_px, closed=True, duration=duration)

    def arrow(
        self, ident: str, a, b, color, width_px: float = 2.0, duration: float = NEVER
    ) -> None:
        self._layer.arrow(ident, self.world(a), self.world(b), color, width_px, duration)

    def point(
        self, ident: str, point, color, radius_px: float = 4.0, duration: float = NEVER
    ) -> None:
        self._layer.point(ident, self.world(point), color, radius_px, duration)

    def points(
        self, ident: str, points, color, radius_px: float = 4.0, duration: float = NEVER
    ) -> None:
        self._layer.points(ident, self._points(points), color, radius_px, duration)

    def text(
        self,
        ident: str,
        point,
        text: str,
        color=(1.0, 1.0, 1.0, 1.0),
        *,
        offset_px=(0.0, 0.0),
        align=(0.0, 0.5),
        duration: float = NEVER,
    ) -> None:
        self._layer.text(
            ident,
            self.world(point),
            text,
            color,
            offset_px,
            align,
            duration,
        )

    def grid(
        self,
        ident: str,
        bounds,
        spacing: float,
        color=(0.5, 0.5, 0.5, 0.35),
        width_px: float = 1.0,
        duration: float = NEVER,
    ) -> None:
        """Create or update a batched axis-aligned grid."""

        x0, y0, x1, y1 = (float(value) for value in bounds)
        spacing = float(spacing)
        if spacing <= 0.0 or not np.isfinite(spacing):
            raise ValueError("grid spacing must be finite and positive")
        xs = np.arange(math.ceil(x0 / spacing), math.floor(x1 / spacing) + 1) * spacing
        ys = np.arange(math.ceil(y0 / spacing), math.floor(y1 / spacing) + 1) * spacing
        starts = np.vstack(
            (
                np.column_stack((xs, np.full_like(xs, y0))),
                np.column_stack((np.full_like(ys, x0), ys)),
            )
        )
        ends = np.vstack(
            (
                np.column_stack((xs, np.full_like(xs, y1))),
                np.column_stack((np.full_like(ys, x1), ys)),
            )
        )
        self.lines(ident, starts, ends, color, width_px, duration)

    def erase(self, ident: str) -> None:
        self._layer.erase(ident)

    def clear(self) -> None:
        self._layer.clear()


@dataclass
class Canvas2D:
    """A plane-oriented 2D canvas backed by retained GPU debug primitives.

    The canvas is intended for physics and geometry diagnostics: stable IDs
    update allocations in place, layers can be hidden without discarding their
    contents, and line widths, points, and labels stay legible in screen pixels.
    """

    draw: DebugDraw
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    normal: np.ndarray

    def __init__(
        self,
        draw: DebugDraw,
        *,
        origin=(0.0, 0.0, 0.0),
        x_axis=(1.0, 0.0, 0.0),
        y_axis=(0.0, 1.0, 0.0),
    ) -> None:
        self.draw = draw
        self.origin = np.asarray(origin, np.float32).reshape(3).copy()
        self.x_axis = _unit(x_axis)
        y = _unit(y_axis)
        y -= self.x_axis * float(np.dot(self.x_axis, y))
        self.y_axis = _unit(y)
        self.normal = _unit(np.cross(self.x_axis, self.y_axis))
        self._layers: dict[str, CanvasLayer2D] = {}

    def world(self, point, *, depth: float = 0.0) -> np.ndarray:
        value = np.asarray(point, np.float32).reshape(2)
        return (
            self.origin
            + value[0] * self.x_axis
            + value[1] * self.y_axis
            + float(depth) * self.normal
        )

    def canvas_point(self, world) -> tuple[float, float]:
        """Project one world position onto the canvas basis."""

        relative = np.asarray(world, np.float32).reshape(3) - self.origin
        return float(np.dot(relative, self.x_axis)), float(np.dot(relative, self.y_axis))

    def screen_to_canvas(
        self,
        point,
        camera: CameraView,
        viewport: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        """Intersect a viewport pixel with the canvas plane for pointer interaction."""

        x, y, width, height = (float(value) for value in viewport)
        px, py = (float(value) for value in point)
        ndc_x = (px - x) / max(width, 1e-6) * 2.0 - 1.0
        ndc_y = 1.0 - (py - y) / max(height, 1e-6) * 2.0
        ray_origin, ray_direction = math3d.unproject_ray(
            ndc_x, ndc_y, camera.view_matrix(), camera.proj_matrix()
        )
        denominator = float(np.dot(ray_direction, self.normal))
        if abs(denominator) <= 1e-8:
            return None
        distance = float(np.dot(self.origin - ray_origin, self.normal) / denominator)
        if distance < 0.0:
            return None
        return self.canvas_point(ray_origin + ray_direction * distance)

    def canvas_to_screen(
        self,
        point,
        camera: CameraView,
        viewport: tuple[float, float, float, float],
        *,
        depth: float = 0.0,
    ) -> tuple[float, float] | None:
        """Project a canvas coordinate into viewport pixels."""

        world = self.world(point, depth=depth)
        clip = camera.proj_matrix() @ (camera.view_matrix() @ np.array((*world, 1.0)))
        if clip[3] <= 1e-8:
            return None
        ndc = clip[:2] / clip[3]
        x, y, width, height = (float(value) for value in viewport)
        return (
            x + (float(ndc[0]) * 0.5 + 0.5) * width,
            y + (0.5 - float(ndc[1]) * 0.5) * height,
        )

    def layer(
        self,
        name: str = "default",
        *,
        depth: float = 0.0,
        occlusion: Occlusion = Occlusion.ALWAYS,
    ) -> CanvasLayer2D:
        """Return a stable named layer; its initial depth and occlusion are retained."""

        key = str(name).strip()
        if not key:
            raise ValueError("canvas layer name must not be empty")
        existing = self._layers.get(key)
        if existing is None:
            existing = CanvasLayer2D(
                self,
                self.draw.layer(f"canvas2d:{key}", occlusion),
                depth,
            )
            self._layers[key] = existing
        return existing

    def layers(self) -> tuple[CanvasLayer2D, ...]:
        return tuple(self._layers.values())

    def camera(
        self,
        bounds,
        *,
        aspect: float = 1.0,
        padding: float = 0.05,
    ) -> CameraView:
        """Create a top-down orthographic camera that contains XY bounds."""

        x0, y0, x1, y1 = (float(value) for value in bounds)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("camera bounds must have positive width and height")
        aspect = max(float(aspect), 1e-6)
        padding = max(0.0, float(padding))
        width = (x1 - x0) * (1.0 + 2.0 * padding)
        height = (y1 - y0) * (1.0 + 2.0 * padding)
        ortho_height = max(height, width / aspect)
        center = self.world(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        distance = max(10.0, ortho_height * 2.0)
        return CameraView(
            eye=center + self.normal * distance,
            target=center,
            up=self.y_axis.copy(),
            near=max(0.001, distance - ortho_height * 2.0),
            far=distance + ortho_height * 2.0,
            aspect=aspect,
            orthographic=True,
            ortho_height=ortho_height,
        )

    def clear(self) -> None:
        for layer in self._layers.values():
            layer.clear()


__all__ = ["Canvas2D", "CanvasLayer2D"]
