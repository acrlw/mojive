"""Cached native meshes for a diffuse underlay beneath a floating surface."""

from functools import lru_cache
from math import ceil, cos, erfc, pi, sin, sqrt

from imgui_bundle import imgui

# Logical points. A single diffuse layer, shared by all temporary surfaces.
_OFFSET = (0, 8)
_BLUR = 24
_SPREAD = -4


def _falloff_layers():
    # The Gaussian tail is convex outside the surface. A positive sum of linear
    # ramps approximates it without stepped opacity bands or a blur render pass.
    sigma = _BLUR / 2
    distances = tuple(sigma * step for step in (0, 0.4, 0.8, 1.4, 2.2, 3.2))
    opacity = (*(0.5 * erfc(d / (sqrt(2) * sigma)) for d in distances[:-1]), 0.0)
    slopes = (
        *(
            (opacity[i + 1] - opacity[i]) / (distances[i + 1] - distances[i])
            for i in range(len(distances) - 1)
        ),
        0.0,
    )
    return tuple((d, d * (slopes[i + 1] - slopes[i])) for i, d in enumerate(distances[1:]))


_LAYERS = _falloff_layers()


@lru_cache(maxsize=128)
def _shadow_meshes(width, height, radius, scale):
    """Retain local native vertices; moving a window does not rebuild its shadow."""
    spread = max(_SPREAD * scale, -min(width, height) / 2)
    radius = max(0, min(radius + spread, width / 2 + spread, height / 2 + spread))
    segments = max(8, ceil(pi / 2 * sqrt((radius + _LAYERS[-1][0] * scale) / 1.2)))

    def contour(distance):
        # Translate the contracted caster, rather than stretching its falloff.
        # The center remains filled beneath the native antialiased surface.
        dx, dy = _OFFSET[0] * scale, _OFFSET[1] * scale
        left, top = dx - spread - distance, dy - spread - distance
        right, bottom = dx + width + spread + distance, dy + height + spread + distance
        corner = radius + distance
        points = []
        for cx, cy, start in (
            (right - corner, top + corner, -pi / 2),
            (right - corner, bottom - corner, 0),
            (left + corner, bottom - corner, pi / 2),
            (left + corner, top + corner, pi),
        ):
            for step in range(segments + 1):
                angle = start + step * pi / (2 * segments)
                points.append(imgui.ImVec2(cx + cos(angle) * corner, cy + sin(angle) * corner))
        return points

    inner = contour(0)
    count = len(inner)
    indices = [index for i in range(1, count - 1) for index in (0, i, i + 1)]
    for i in range(count):
        j = (i + 1) % count
        indices.extend((i, j, count + j, i, count + j, count + i))
    opaque = [True] * count + [False] * count
    return tuple(
        imgui.DrawMesh(inner + contour(distance * scale), indices, opaque)
        for distance, _ in _LAYERS
    )


@lru_cache(maxsize=256)
def _shadow_colors(color):
    return tuple(
        (color & 0x00FFFFFF) | (round((color >> 24) * weight) << 24) for _, weight in _LAYERS
    )


def draw_surface_shadow(draw, origin, size, radius, scale, color):
    """Insert the falloff behind the owner's native fill, border and content."""
    meshes = _shadow_meshes(*size, radius, scale)
    colors = _shadow_colors(color)
    draw.push_clip_rect_full_screen()
    try:
        # Prepending reverses submission order; preserve the authored blend order.
        for index in range(len(meshes) - 1, -1, -1):
            meshes[index].draw_behind(draw, origin, colors[index])
    finally:
        draw.pop_clip_rect()
