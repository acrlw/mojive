"""Shared viewport surfaces and markers with local, reusable Draw2D geometry."""

from functools import lru_cache

from mojive.geometry2d.curves import CORNER_SMOOTHING, smooth_capsule_points
from mojive.ui.paint_protocol import Draw2D
from mojive.ui.theme import RGBA

from .model import CAPSULE_SMOOTHING


@lru_cache(maxsize=64)
def capsule_points(x, y, width, height, smoothing=CAPSULE_SMOOTHING):
    return smooth_capsule_points(x, y, width, height, smoothing)


def _paint_shape(draw, kind, origin, width, height, stroke, rounding, smoothing, fill, border):
    x, y = origin
    if kind == "capsule":
        points = capsule_points(x, y, width, height, smoothing)
        draw.convex_fill(points, fill)
        draw.polyline(points, border, stroke, closed=True)
    elif kind == "keycap":
        hi = x + width, y + height
        draw.rect_filled(origin, hi, fill, rounding=rounding, smoothing=smoothing)
        draw.rect(origin, hi, border, stroke, rounding=rounding, smoothing=smoothing)
    elif kind == "disc":
        draw.circle_filled(origin, width, fill)
    elif kind == "separator":
        draw.line(origin, (x + width, y + height), fill, stroke)
    else:
        raise ValueError(f"Unknown viewport shape: {kind!r}")


def draw_capsule_shell(
    draw: Draw2D,
    origin: tuple[float, float],
    width: float,
    height: float,
    fill: RGBA,
    border: RGBA,
    stroke: float,
    *,
    smoothing: float = CAPSULE_SMOOTHING,
) -> None:
    """Place the same capsule contour in production and adjustable design previews."""
    _paint_shape(draw, "capsule", origin, width, height, stroke, 0.0, smoothing, fill, border)


def draw_keycap_background(
    draw: Draw2D,
    origin: tuple[float, float],
    width: float,
    height: float,
    fill: RGBA,
    border: RGBA,
    stroke: float,
    rounding: float,
) -> None:
    smoothing = getattr(draw, "corner_smoothing", CORNER_SMOOTHING)
    _paint_shape(draw, "keycap", origin, width, height, stroke, rounding, smoothing, fill, border)


def draw_selection_disc(
    draw: Draw2D, center: tuple[float, float], radius: float, color: RGBA
) -> None:
    _paint_shape(draw, "disc", center, radius, 0.0, 0.0, 0.0, 0.0, color, color)


def draw_separator(
    draw: Draw2D,
    start: tuple[float, float],
    end: tuple[float, float],
    color: RGBA,
    width: float,
) -> None:
    _paint_shape(
        draw,
        "separator",
        start,
        end[0] - start[0],
        end[1] - start[1],
        width,
        0.0,
        0.0,
        color,
        color,
    )
