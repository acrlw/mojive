"""Backend-neutral 2D overlay drawing.

Overlay features (gizmo, view cube, perturb hints, panel chrome) draw through
the ``Draw2D`` protocol — plain ``(x, y)`` points and float RGBA colors — so
feature code never touches an imgui draw list.  ``ImguiDraw2D`` is the imgui
adapter; every imgui idiom (u32 colors, ``ImVec2``, draw flags, the font
atlas) lives inside it, which keeps the overlay features testable without a
window and portable to a future non-imgui overlay renderer.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np


@lru_cache(maxsize=512)
def _cached_imgui_points(points: tuple[tuple[float, float], ...]):
    """Reuse immutable ImVec2 paths submitted by screen-space UI widgets."""

    from imgui_bundle import imgui

    return tuple(imgui.ImVec2(float(x), float(y)) for x, y in points)


def _anti_alias_fringe_outer(points) -> np.ndarray:
    """Return a one-pixel outward miter ring for either polygon winding."""

    outline = np.asarray(points, np.float64).reshape(-1, 2)
    if len(outline) < 3:
        return np.empty((0, 2), np.float64)
    edges = np.roll(outline, -1, axis=0) - outline
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-9):
        return np.empty((0, 2), np.float64)
    signed_area = 0.5 * float(
        np.sum(outline[:, 0] * np.roll(outline[:, 1], -1))
        - np.sum(outline[:, 1] * np.roll(outline[:, 0], -1))
    )
    if abs(signed_area) < 1e-9:
        return np.empty((0, 2), np.float64)
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    if signed_area < 0.0:
        normals *= -1.0
    miters = (np.roll(normals, 1, axis=0) + normals) * 0.5
    scale = np.minimum(1.0 / np.maximum(np.sum(miters * miters, axis=1), 1e-4), 100.0)
    return outline + miters * scale[:, None]


@lru_cache(maxsize=256)
def _open_polyline_ribbon(
    path: tuple[tuple[float, float], ...],
    width: float,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
]:
    """Build cached left/right edges and the complete boundary of an open stroke."""

    if len(path) < 2 or width <= 0.0:
        return (), (), ()
    points = tuple((float(x), float(y)) for x, y in path)
    directions = []
    for start, end in pairwise(points):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return (), (), ()
        directions.append((dx / length, dy / length))
    normals = tuple((-dy, dx) for dx, dy in directions)
    half_width = 0.5 * float(width)
    offsets = []
    for index in range(len(points)):
        if index == 0:
            offsets.append((normals[0][0] * half_width, normals[0][1] * half_width))
            continue
        if index == len(points) - 1:
            offsets.append((normals[-1][0] * half_width, normals[-1][1] * half_width))
            continue
        mx = normals[index - 1][0] + normals[index][0]
        my = normals[index - 1][1] + normals[index][1]
        miter_length = math.hypot(mx, my)
        if miter_length <= 1e-9:
            offsets.append((normals[index][0] * half_width, normals[index][1] * half_width))
            continue
        mx, my = mx / miter_length, my / miter_length
        projection = max(mx * normals[index][0] + my * normals[index][1], 0.5)
        miter_scale = half_width / projection
        offsets.append((mx * miter_scale, my * miter_scale))
    left = tuple((p[0] + o[0], p[1] + o[1]) for p, o in zip(points, offsets, strict=True))
    right = tuple((p[0] - o[0], p[1] - o[1]) for p, o in zip(points, offsets, strict=True))
    return left, right, left + tuple(reversed(right))


@runtime_checkable
class Draw2D(Protocol):
    """Immediate-mode 2D primitives; later calls paint over earlier ones."""

    def line(self, a, b, color, width: float) -> None: ...

    def polyline(self, points, color, width: float, *, closed: bool = False) -> None: ...

    def convex_fill(self, points, color) -> None: ...

    def triangle_fan_fill(self, points, color) -> None: ...

    def concave_fill(self, points, color) -> None: ...

    def fringed_concave_fill(self, points, color) -> None:
        """Concave fill with a 1 px alpha-gradient fringe instead of builtin AA."""
        ...

    def circle(
        self, center, radius: float, color, width: float = 1.0, *, segments: int = 0
    ) -> None: ...

    def circle_filled(self, center, radius: float, color, *, segments: int = 0) -> None: ...

    def rect(self, lo, hi, color, width: float = 1.0, *, rounding: float = 0.0) -> None: ...

    def rect_filled(self, lo, hi, color, *, rounding: float = 0.0) -> None: ...

    def text(self, pos, color, text: str) -> None: ...

    def text_size(self, text: str) -> tuple[float, float]: ...

    def centered_label(self, text: str, center, color, max_width: float) -> None:
        """Text centered on its ink box at ``center``, shrinking to fit ``max_width``."""
        ...


class ImguiDraw2D:
    """Draw2D over the current imgui window's draw list (or a given one)."""

    def __init__(self, draw_list=None) -> None:
        from imgui_bundle import imgui

        self._imgui = imgui
        self._dl = draw_list if draw_list is not None else imgui.get_window_draw_list()

    def _vec(self, p):
        imgui = self._imgui
        return imgui.ImVec2(float(p[0]), float(p[1]))

    def _vecs(self, points):
        if isinstance(points, tuple):
            try:
                return _cached_imgui_points(points)
            except (TypeError, ValueError):
                pass
        return [self._vec(point) for point in points]

    def _u32(self, color) -> int:
        imgui = self._imgui
        return imgui.color_convert_float4_to_u32(imgui.ImVec4(*(float(c) for c in color)))

    def line(self, a, b, color, width: float) -> None:
        self._dl.add_line(self._vec(a), self._vec(b), self._u32(color), float(width))

    def polyline(self, points, color, width: float, *, closed: bool = False) -> None:
        imgui = self._imgui
        flags = imgui.ImDrawFlags_.closed if closed else imgui.ImDrawFlags_.none
        self._dl.add_polyline(self._vecs(points), self._u32(color), float(width), flags.value)

    def convex_fill(self, points, color) -> None:
        self._dl.add_convex_poly_filled(self._vecs(points), self._u32(color))

    def triangle_fan_fill(self, points, color) -> None:
        vertices = np.asarray(points, np.float64).reshape(-1, 2)
        if len(vertices) < 3:
            return
        dl = self._dl
        base = dl._vtx_current_idx
        uv = self._imgui.get_io().fonts.tex_uv_white_pixel
        rgba = self._u32(color)
        dl.prim_reserve((len(vertices) - 2) * 3, len(vertices))
        for point in vertices:
            dl.prim_write_vtx(self._vec(point), uv, rgba)
        for index in range(1, len(vertices) - 1):
            dl.prim_write_idx(base)
            dl.prim_write_idx(base + index)
            dl.prim_write_idx(base + index + 1)
        self._write_anti_alias_fringe(vertices, rgba)

    def concave_fill(self, points, color) -> None:
        self._dl.add_concave_poly_filled(self._vecs(points), self._u32(color))

    def fringed_concave_fill(self, points, color) -> None:
        imgui = self._imgui
        dl = self._dl
        rgba = self._u32(color)
        outline = np.asarray(points, np.float64).reshape(-1, 2)
        aa_flag = imgui.ImDrawListFlags_.anti_aliased_fill.value
        flags = dl.flags
        dl.flags = flags & ~aa_flag
        dl.add_concave_poly_filled([self._vec(p) for p in outline], rgba)
        dl.flags = flags
        if not (flags & aa_flag):
            return

        self._write_anti_alias_fringe(outline, rgba)

    def _write_anti_alias_fringe(self, outline, rgba: int) -> None:
        """Emit one alpha-gradient ring around an already solid polygon fill."""

        imgui = self._imgui
        dl = self._dl
        if not (dl.flags & imgui.ImDrawListFlags_.anti_aliased_fill.value):
            return
        outline = np.asarray(outline, np.float64).reshape(-1, 2)
        outer = _anti_alias_fringe_outer(outline)
        if len(outer) != len(outline):
            return

        count = len(outline)
        base = dl._vtx_current_idx
        transparent = rgba & 0x00FFFFFF
        uv = imgui.get_io().fonts.tex_uv_white_pixel
        dl.prim_reserve(count * 6, count * 2)
        for inner, fringe in zip(outline, outer, strict=True):
            dl.prim_write_vtx(self._vec(inner), uv, rgba)
            dl.prim_write_vtx(self._vec(fringe), uv, transparent)
        for i in range(count):
            j = (i + 1) % count
            dl.prim_write_idx(base + i * 2)
            dl.prim_write_idx(base + j * 2)
            dl.prim_write_idx(base + j * 2 + 1)
            dl.prim_write_idx(base + i * 2)
            dl.prim_write_idx(base + j * 2 + 1)
            dl.prim_write_idx(base + i * 2 + 1)

    def circle(
        self, center, radius: float, color, width: float = 1.0, *, segments: int = 0
    ) -> None:
        self._dl.add_circle(
            self._vec(center), float(radius), self._u32(color), int(segments), float(width)
        )

    def circle_filled(self, center, radius: float, color, *, segments: int = 0) -> None:
        self._dl.add_circle_filled(
            self._vec(center), float(radius), self._u32(color), int(segments)
        )

    def rect(self, lo, hi, color, width: float = 1.0, *, rounding: float = 0.0) -> None:
        self._dl.add_rect(
            self._vec(lo), self._vec(hi), self._u32(color), float(rounding), float(width), 0
        )

    def rect_filled(self, lo, hi, color, *, rounding: float = 0.0) -> None:
        self._dl.add_rect_filled(self._vec(lo), self._vec(hi), self._u32(color), float(rounding))

    def text(self, pos, color, text: str) -> None:
        self._dl.add_text(self._vec(pos), self._u32(color), text)

    def text_size(self, text: str) -> tuple[float, float]:
        size = self._imgui.calc_text_size(text)
        return float(size.x), float(size.y)

    def centered_label(self, text: str, center, color, max_width: float) -> None:
        imgui = self._imgui
        font = imgui.get_font()
        size = imgui.get_font_size()
        box = ink_box(font, size, text)
        if box is None:
            return
        width = box[2] - box[0]
        if width > max_width > 0.0:
            size *= max_width / width
            box = ink_box(font, size, text) or box

        pen_x = round(float(center[0]) - (box[0] + box[2]) * 0.5)
        pen_y = round(float(center[1]) - (box[1] + box[3]) * 0.5)
        baked = font.get_font_baked(size)
        tex = imgui.get_io().fonts.tex_data.get_tex_ref()
        rgba = self._u32(color)
        for ch in text:
            g = baked.find_glyph(ord(ch))
            if g is None:
                continue
            if g.x1 > g.x0 and g.y1 > g.y0:
                self._dl.add_image(
                    tex,
                    self._vec((pen_x + g.x0, pen_y + g.y0)),
                    self._vec((pen_x + g.x1, pen_y + g.y1)),
                    self._vec((g.u0, g.v0)),
                    self._vec((g.u1, g.v1)),
                    rgba,
                )
            pen_x += g.advance_x


def ink_box(font, size: float, text: str):
    """Ink bounding box of ``text`` in pixels (the visual, not line, box)."""
    baked = font.get_font_baked(size)
    pen = 0.0
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for ch in text:
        g = baked.find_glyph(ord(ch))
        if g is not None and g.x1 > g.x0 and g.y1 > g.y0:
            x0 = min(x0, pen + g.x0)
            x1 = max(x1, pen + g.x1)
            y0 = min(y0, g.y0)
            y1 = max(y1, g.y1)
            pen += g.advance_x
        elif g is not None:
            pen += g.advance_x
    return None if x1 < x0 else (x0, y0, x1, y1)
