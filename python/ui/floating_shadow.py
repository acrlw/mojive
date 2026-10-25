"""Cached native meshes for a diffuse shadow outside a floating surface."""

from functools import lru_cache
from math import ceil, cos, erfc, pi, sin, sqrt

from imgui_bundle import imgui


def _falloff_layers():
    # The Gaussian tail is convex outside the surface. A positive sum of linear
    # ramps approximates it without stepped opacity bands or a blur render pass.
    distances = (0, 4, 8, 14, 22, 32)
    opacity = (*(erfc(d / (sqrt(2) * 10)) for d in distances[:-1]), 0.0)
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
    radius = max(0, min(radius, width / 2, height / 2))
    segments = max(8, ceil(pi / 2 * sqrt((radius + 32 * scale) / 1.2)))

    def contour(distance):
        # The lower penumbra is wider; the inner edge still follows the native
        # surface exactly so no shadow is blended over its content or border.
        left, top = -distance, -distance * 0.65
        right, bottom = width + distance, height + distance * 1.35
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
    indices = []
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
    """Submit the cached falloff in the owning window's normal compositing order."""
    meshes = _shadow_meshes(*size, radius, scale)
    draw.push_clip_rect_full_screen()
    try:
        for mesh, ink in zip(meshes, _shadow_colors(color), strict=True):
            mesh.draw(draw, origin, ink)
    finally:
        draw.pop_clip_rect()
