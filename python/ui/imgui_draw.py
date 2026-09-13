"""ImGui shape and font adapter for the neutral Draw2D protocol."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from mojive.geometry2d.curves import (
    CORNER_SMOOTHING,
    arc_ribbon_mesh,
    arrow_mesh,
    capped_polyline_points,
    smooth_rect_points,
)
from mojive.geometry2d.curves import polygon_fringe as _anti_alias_fringe_outer


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
def _cached_fringe_points(
    points: tuple[tuple[float, float], ...], inside: bool = False, width: float = 1.0
):
    """Reuse both native vertex arrays and their outward miter construction."""
    fringe = _anti_alias_fringe_outer(points)
    if len(fringe) == len(points):
        vertices = np.asarray(points)
        fringe = vertices + (fringe - vertices) * (-width if inside else width)
    outer = tuple(map(tuple, fringe.tolist()))
    if len(outer) != len(points):
        return (), ()
    return _cached_imgui_points(points), _cached_imgui_points(outer)


@lru_cache(maxsize=256)
def _concave_indices(points: tuple[tuple[float, float], ...]):
    """Triangulate immutable local paths once with ImGui's existing tessellator.

    Retain only index values, never an ImGui context or draw-list allocation.
    """
    from imgui_bundle import imgui

    scratch = imgui.ImDrawList(imgui.get_draw_list_shared_data())
    scratch._reset_for_new_frame()
    scratch.flags = 0
    scratch.add_concave_poly_filled(_cached_imgui_points(points), 0xFFFFFFFF)
    return tuple(scratch.idx_buffer)


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

    def _stroke_u32(self, color, width):
        # ImGui floors AA line geometry at one unit. Preserve the authored
        # subpixel coverage instead of silently making thin contours brighter.
        if (
            0 <= width < 1
            and self._dl.flags & self._imgui.ImDrawListFlags_.anti_aliased_lines.value
        ):
            color = (*tuple(color)[:3], color[3] * width)
        return self._u32(color)

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
            smoothing = self.corner_smoothing if smoothing is None else smoothing
            if smoothing == 0.0:
                vertices, indices = arc_ribbon_mesh(
                    points,
                    None,
                    None,
                    width,
                    round_start=cap in {"round", "round_start"},
                    round_end=cap in {"round", "round_end"},
                    smoothing=0.0,
                )
                self.indexed_fill(vertices, indices, color, outline=vertices)
                return
            outline = capped_polyline_points(
                points,
                width,
                round_start=cap in {"round", "round_start"},
                round_end=cap in {"round", "round_end"},
                smoothing=smoothing,
            )
            if outline:
                self.fringed_concave_fill(outline, color)
            return
        imgui = self._imgui
        flags = imgui.ImDrawFlags_.closed if closed else imgui.ImDrawFlags_.none
        draw_flags = self._dl.flags
        if width < 1.0:
            # Textured AA expands a clamped 1 u stroke to a 3 u strip. Its
            # inner edge folds over on subpixel corners, blending alpha twice.
            self._dl.flags &= ~imgui.ImDrawListFlags_.anti_aliased_lines_use_tex.value
        try:
            self._dl.add_polyline(
                self._vecs(points), self._stroke_u32(color, width), float(width), flags.value
            )
        finally:
            if width < 1.0:
                self._dl.flags = draw_flags

    def convex_fill(self, points, color) -> None:
        self._dl.add_convex_poly_filled(_fill_points(points), self._u32(color))

    def indexed_fill(
        self,
        points,
        indices,
        color,
        *,
        outline=(),
        hole=(),
        origin=None,
        direction=(1.0, 0.0),
        fringe_width=1.0,
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
            self._write_anti_alias_fringe(outline, rgba, width=fringe_width)
        if len(hole):
            self._write_anti_alias_fringe(hole, rgba, inside=True, width=fringe_width)
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

    def fringed_concave_fill(self, points, color, *, origin=None) -> None:
        if origin is not None:
            path = _clockwise_points(tuple(points))
            self.indexed_fill(path, _concave_indices(path), color, outline=path, origin=origin)
            return
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

    def _write_anti_alias_fringe(
        self, outline, rgba: int, *, inside: bool = False, width: float = 1.0
    ) -> None:
        """Emit one alpha-gradient ring around an already solid polygon fill."""

        imgui = self._imgui
        dl = self._dl
        if not (dl.flags & imgui.ImDrawListFlags_.anti_aliased_fill.value):
            return
        if not isinstance(outline, tuple):
            outline = tuple((float(point[0]), float(point[1])) for point in outline)
        inner_points, fringe_points = _cached_fringe_points(outline, inside, width)
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
            self._vec(center),
            float(radius),
            self._stroke_u32(color, width),
            int(segments),
            float(width),
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
            self._dl.add_rect(
                self._vec(lo), self._vec(hi), self._stroke_u32(color, width), 0.0, float(width), 0
            )
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

        # Labels move rigidly with their geometry, including fractional pixels.
        pen_x = float(center[0]) - (box[0] + box[2]) * 0.5
        pen_y = float(center[1]) - (box[1] + box[3]) * 0.5
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
