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
from typing import Protocol, runtime_checkable

import numpy as np

from ..curves2d import (
    CORNER_SMOOTHING,
    arrow_mesh,
    capped_polyline_points,
    smooth_rect_points,
)
from ..curves2d import polygon_fringe as _anti_alias_fringe_outer
from ..draglink2d import smooth_drag_link_mesh


@lru_cache(maxsize=512)
def _cached_imgui_points(points: tuple[tuple[float, float], ...]):
    """Reuse immutable ImVec2 paths submitted by screen-space UI widgets."""

    from imgui_bundle import imgui

    return tuple(imgui.ImVec2(float(x), float(y)) for x, y in points)


@lru_cache(maxsize=512)
def _clockwise_points(points: tuple[tuple[float, float], ...]):
    # ImGui's fill fringe requires clockwise screen winding, including mirrored glyphs.
    area = sum(
        a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1], strict=True)
    )
    return points if area >= 0.0 else points[::-1]


@lru_cache(maxsize=512)
def _cached_imgui_fill_points(points: tuple[tuple[float, float], ...]):
    return _cached_imgui_points(_clockwise_points(points))


def _fill_points(points):
    try:
        return _cached_imgui_fill_points(points)
    except TypeError:
        return _cached_imgui_fill_points(tuple((float(p[0]), float(p[1])) for p in points))


@lru_cache(maxsize=256)
def _cached_color(color):
    from imgui_bundle import imgui

    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*(float(c) for c in color)))


@lru_cache(maxsize=128)
def _triangle_fan_indices(count: int) -> tuple[int, ...]:
    return tuple(index for i in range(1, count - 1) for index in (0, i, i + 1))


@lru_cache(maxsize=512)
def _cached_fringe_points(points: tuple[tuple[float, float], ...], inside: bool = False):
    """Reuse both native vertex arrays and their outward miter construction."""
    fringe = _anti_alias_fringe_outer(points)
    if inside and len(fringe) == len(points):
        fringe = 2.0 * np.asarray(points) - fringe
    outer = tuple(map(tuple, fringe.tolist()))
    if len(outer) != len(points):
        return (), ()
    return _cached_imgui_points(points), _cached_imgui_points(outer)


@runtime_checkable
class Draw2D(Protocol):
    """Immediate-mode overlays in logical window coordinates, with float RGBA colors.

    Later calls paint over earlier ones. Geometry helpers own local paths;
    adapters own submission and antialiasing. See docs/how-to/ui-drawing.md.
    """

    def line(
        self, a, b, color, width: float, *, cap: str = "butt", smoothing: float | None = None
    ) -> None:
        """Stroke authored coordinates exactly, like a two-point polyline."""
        ...

    def arrow(
        self,
        a,
        b,
        color,
        width: float = 2.0,
        *,
        head_length: float = 7.0,
        head_width: float = 8.0,
        corner_radius: float = 1.0,
        join_radius: float | None = None,
        smoothing: float | None = None,
        round_tail: bool = False,
    ) -> None:
        """Draw one arrow outline; smoothing None uses the adapter's default.

        Radius zero is sharp; smoothing zero uses circular corners. Explicit
        dimensions use the same units as endpoints. The adapter controls AA.
        """
        ...

    def polyline(
        self,
        points,
        color,
        width: float,
        *,
        closed: bool = False,
        cap: str = "butt",
        smoothing: float | None = None,
    ) -> None: ...

    def convex_fill(self, points, color) -> None: ...

    def indexed_fill(
        self, points, indices, color, *, outline=(), hole=(), origin=None, direction=(1.0, 0.0)
    ) -> None:
        """Fill triangles and optional AA contours, optionally placing a local mesh.

        With origin set, direction is the unit local X axis in window coordinates.
        Placement is rigid so the antialias fringe retains its one-pixel width.
        """
        ...

    def triangle_fan_fill(self, points, color) -> None:
        """Fill a simple polygon visible from its first vertex, such as a sector."""
        ...

    def concave_fill(self, points, color) -> None: ...

    def fringed_concave_fill(self, points, color) -> None:
        """Concave fill with a 1 px alpha-gradient fringe instead of builtin AA."""
        ...

    def circle(
        self, center, radius: float, color, width: float = 1.0, *, segments: int = 0
    ) -> None: ...

    def circle_filled(self, center, radius: float, color, *, segments: int = 0) -> None: ...

    def rect(
        self,
        lo,
        hi,
        color,
        width: float = 1.0,
        *,
        rounding: float = 0.0,
        smoothing: float | None = None,
    ) -> None: ...

    def rect_filled(
        self, lo, hi, color, *, rounding: float = 0.0, smoothing: float | None = None
    ) -> None: ...

    def text(self, pos, color, text: str, *, pixel_snap: bool = True) -> None: ...

    def text_size(self, text: str) -> tuple[float, float]: ...

    def text_ink_bounds(self, text: str) -> tuple[float, float, float, float] | None:
        """Visible glyph bounds relative to the text pen position."""
        ...

    def centered_label(self, text: str, center, color, max_width: float) -> None:
        """Text centered on its ink box at ``center``, shrinking to fit ``max_width``."""
        ...


def text_line_y(draw: Draw2D, center_y: float) -> float:
    """Align a text row to one cap-height reference, independent of word contents."""

    ink = draw.text_ink_bounds("H")
    center = (ink[1] + ink[3]) * 0.5 if ink else draw.text_size("H")[1] * 0.5
    return center_y - center


class ImguiDraw2D:
    """Draw2D over the current imgui window's draw list (or a given one)."""

    def __init__(self, draw_list=None, *, corner_smoothing: float = CORNER_SMOOTHING) -> None:
        from imgui_bundle import imgui

        self._imgui = imgui
        self._dl = draw_list if draw_list is not None else imgui.get_window_draw_list()
        self.corner_smoothing = float(corner_smoothing)

    def with_corner_smoothing(self, value: float):
        """Share the draw list while selecting one element family's curve profile."""
        return ImguiDraw2D(self._dl, corner_smoothing=value)

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
        return _cached_color(color if isinstance(color, tuple) else tuple(color))

    def line(
        self, a, b, color, width: float, *, cap: str = "butt", smoothing: float | None = None
    ) -> None:
        # AddLine shifts its endpoints by half a pixel; AddPolyline preserves
        # the same authored coordinates as filled paths and other Draw2D ports.
        self.polyline((a, b), color, width, cap=cap, smoothing=smoothing)

    def arrow(
        self,
        a,
        b,
        color,
        width: float = 2.0,
        *,
        head_length: float = 7.0,
        head_width: float = 8.0,
        corner_radius: float = 1.0,
        join_radius: float | None = None,
        smoothing: float | None = None,
        round_tail: bool = False,
    ) -> None:
        ax, ay = map(float, a)
        bx, by = map(float, b)
        if not all(map(math.isfinite, (ax, ay, bx, by))):
            raise ValueError("arrow endpoints must each contain two finite coordinates")
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        vertices, indices, outline = arrow_mesh(
            length,
            width,
            head_length=head_length,
            head_width=head_width,
            corner_radius=corner_radius,
            join_radius=join_radius,
            smoothing=self.corner_smoothing if smoothing is None else smoothing,
            round_tail=round_tail,
        )
        if not indices:
            return
        self.indexed_fill(
            vertices,
            indices,
            color,
            outline=outline,
            origin=(ax, ay),
            direction=(dx / length, dy / length),
        )

    def polyline(
        self,
        points,
        color,
        width: float,
        *,
        closed: bool = False,
        cap: str = "butt",
        smoothing: float | None = None,
    ) -> None:
        if cap not in {"butt", "round", "round_start", "round_end"}:
            raise ValueError(f"unknown polyline cap: {cap!r}")
        if not closed and cap != "butt":
            outline = capped_polyline_points(
                points,
                width,
                round_start=cap in {"round", "round_start"},
                round_end=cap in {"round", "round_end"},
                smoothing=self.corner_smoothing if smoothing is None else smoothing,
            )
            if outline:
                self.fringed_concave_fill(outline, color)
            return
        imgui = self._imgui
        flags = imgui.ImDrawFlags_.closed if closed else imgui.ImDrawFlags_.none
        self._dl.add_polyline(self._vecs(points), self._u32(color), float(width), flags.value)

    def convex_fill(self, points, color) -> None:
        self._dl.add_convex_poly_filled(_fill_points(points), self._u32(color))

    def indexed_fill(
        self, points, indices, color, *, outline=(), hole=(), origin=None, direction=(1.0, 0.0)
    ) -> None:
        """Submit precomputed triangles and antialias only the external contours."""
        if not len(indices):
            return
        vertex_start = len(self._dl.vtx_buffer) if origin is not None else 0
        native = getattr(self._dl, "add_indexed_fill", None)
        rgba = self._u32(color)
        path = (
            points
            if isinstance(points, tuple)
            else tuple((float(point[0]), float(point[1])) for point in points)
        )
        vertices = self._vecs(path)
        if outline is points:
            outline = path
        if native is not None:
            native(vertices, indices, rgba)
        else:
            if len(indices) % 3 or any(i < 0 or i >= len(vertices) for i in indices):
                raise ValueError("triangle indices must be valid vertex indices grouped in threes")
            dl = self._dl
            dl.prim_reserve(len(indices), len(vertices))
            base = dl._vtx_current_idx
            uv = self._imgui.get_io().fonts.tex_uv_white_pixel
            for point in vertices:
                dl.prim_write_vtx(point, uv, rgba)
            for index in indices:
                dl.prim_write_idx(base + int(index))
        if len(outline):
            self._write_anti_alias_fringe(outline, rgba)
        if len(hole):
            self._write_anti_alias_fringe(hole, rgba, inside=True)
        if origin is not None:
            # Transform the submitted range, including AA, in one native pass.
            # Local vertex and fringe caches survive movement and rotation.
            self._imgui.internal.shade_verts_transform_pos(
                self._dl,
                vertex_start,
                len(self._dl.vtx_buffer),
                (0.0, 0.0),
                float(direction[0]),
                float(direction[1]),
                self._vec(origin),
            )

    def triangle_fan_fill(self, points, color) -> None:
        vertices = tuple((float(p[0]), float(p[1])) for p in points)
        if len(vertices) < 3:
            return
        self.indexed_fill(vertices, _triangle_fan_indices(len(vertices)), color, outline=vertices)

    def concave_fill(self, points, color) -> None:
        self._dl.add_concave_poly_filled(_fill_points(points), self._u32(color))

    def fringed_concave_fill(self, points, color) -> None:
        imgui = self._imgui
        dl = self._dl
        rgba = self._u32(color)
        try:
            outline = _clockwise_points(points)
        except TypeError:
            outline = _clockwise_points(tuple((float(p[0]), float(p[1])) for p in points))
        aa_flag = imgui.ImDrawListFlags_.anti_aliased_fill.value
        flags = dl.flags
        try:
            dl.flags = flags & ~aa_flag
            dl.add_concave_poly_filled(self._vecs(outline), rgba)
        finally:
            dl.flags = flags
        if not (flags & aa_flag):
            return

        self._write_anti_alias_fringe(outline, rgba)

    def _write_anti_alias_fringe(self, outline, rgba: int, *, inside: bool = False) -> None:
        """Emit one alpha-gradient ring around an already solid polygon fill."""

        imgui = self._imgui
        dl = self._dl
        if not (dl.flags & imgui.ImDrawListFlags_.anti_aliased_fill.value):
            return
        if not isinstance(outline, tuple):
            outline = tuple((float(point[0]), float(point[1])) for point in outline)
        inner_points, fringe_points = _cached_fringe_points(outline, inside)
        if not inner_points:
            return

        native_fringe = getattr(dl, "add_poly_fringe", None)
        if native_fringe is not None:
            native_fringe(inner_points, fringe_points, rgba)
            return

        count = len(outline)
        transparent = rgba & 0x00FFFFFF
        uv = imgui.get_io().fonts.tex_uv_white_pixel
        dl.prim_reserve(count * 6, count * 2)
        base = dl._vtx_current_idx
        for inner, fringe in zip(inner_points, fringe_points, strict=True):
            dl.prim_write_vtx(inner, uv, rgba)
            dl.prim_write_vtx(fringe, uv, transparent)
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

    def rect(
        self,
        lo,
        hi,
        color,
        width: float = 1.0,
        *,
        rounding: float = 0.0,
        smoothing: float | None = None,
    ) -> None:
        if rounding <= 0.0:
            self._dl.add_rect(self._vec(lo), self._vec(hi), self._u32(color), 0.0, float(width), 0)
            return
        points = smooth_rect_points(
            float(lo[0]),
            float(lo[1]),
            float(hi[0]),
            float(hi[1]),
            float(rounding),
            smoothing=self.corner_smoothing if smoothing is None else smoothing,
        )
        if points:
            self.polyline(points, color, width, closed=True)

    def rect_filled(
        self, lo, hi, color, *, rounding: float = 0.0, smoothing: float | None = None
    ) -> None:
        if rounding <= 0.0:
            self._dl.add_rect_filled(self._vec(lo), self._vec(hi), self._u32(color), 0.0)
            return
        points = smooth_rect_points(
            float(lo[0]),
            float(lo[1]),
            float(hi[0]),
            float(hi[1]),
            float(rounding),
            smoothing=self.corner_smoothing if smoothing is None else smoothing,
        )
        if points:
            self.convex_fill(points, color)

    def text(self, pos, color, text: str, *, pixel_snap: bool = True) -> None:
        if pixel_snap:
            self._dl.add_text(self._vec(pos), self._u32(color), text)
            return
        # Composite layouts share fractional coordinates with their icons.
        # Keep native text rendering, but do not truncate only the text origin.
        flags = self._dl.flags
        self._dl.flags |= self._imgui.ImDrawListFlags_.text_no_pixel_snap.value
        try:
            self._dl.add_text(self._vec(pos), self._u32(color), text)
        finally:
            self._dl.flags = flags

    def text_size(self, text: str) -> tuple[float, float]:
        size = self._imgui.calc_text_size(text)
        return float(size.x), float(size.y)

    def text_ink_bounds(self, text: str) -> tuple[float, float, float, float] | None:
        imgui = self._imgui
        return ink_box(imgui.get_font(), imgui.get_font_size(), text)

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


def draw_drag_link(
    overlay: Draw2D,
    start,
    end,
    core,
    edge,
    width: float,
    radius: float,
    edge_width: float,
    *,
    smoothing: float = CORNER_SMOOTHING,
) -> None:
    """Draw the shared implicit drag-link outline through a backend-neutral triangle mesh."""
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    distance = math.hypot(dx, dy)
    ux, uy = (dx / distance, dy / distance) if distance > 1e-9 else (1.0, 0.0)
    x, y = float(start[0]), float(start[1])

    for level, color in ((edge_width, edge), (0.0, core)):
        vertices, indices, outline, hole = smooth_drag_link_mesh(
            distance, radius, width, smoothing, level
        )
        overlay.indexed_fill(
            vertices, indices, color, outline=outline, hole=hole, origin=(x, y), direction=(ux, uy)
        )


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
