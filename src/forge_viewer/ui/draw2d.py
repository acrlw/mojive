"""Backend-neutral 2D overlay drawing.

Overlay features (gizmo, view cube, perturb hints, panel chrome) draw through
the ``Draw2D`` protocol — plain ``(x, y)`` points and float RGBA colors — so
feature code never touches an imgui draw list.  ``ImguiDraw2D`` is the imgui
adapter; every imgui idiom (u32 colors, ``ImVec2``, draw flags, the font
atlas) lives inside it, which keeps the overlay features testable without a
window and portable to a future non-imgui overlay renderer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


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

    def _u32(self, color) -> int:
        imgui = self._imgui
        return imgui.color_convert_float4_to_u32(imgui.ImVec4(*(float(c) for c in color)))

    def line(self, a, b, color, width: float) -> None:
        self._dl.add_line(self._vec(a), self._vec(b), self._u32(color), float(width))

    def polyline(self, points, color, width: float, *, closed: bool = False) -> None:
        imgui = self._imgui
        flags = imgui.ImDrawFlags_.closed if closed else imgui.ImDrawFlags_.none
        self._dl.add_polyline(
            [self._vec(p) for p in points], self._u32(color), float(width), flags.value
        )

    def convex_fill(self, points, color) -> None:
        self._dl.add_convex_poly_filled([self._vec(p) for p in points], self._u32(color))

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

    def concave_fill(self, points, color) -> None:
        self._dl.add_concave_poly_filled([self._vec(p) for p in points], self._u32(color))

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

        # Builtin AA fills concave polygons with visible seams, so emit the
        # anti-aliasing fringe by hand: one ring of alpha-gradient triangles.
        edges = np.roll(outline, -1, axis=0) - outline
        lengths = np.linalg.norm(edges, axis=1)
        normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
        miters = (np.roll(normals, 1, axis=0) + normals) * 0.5
        scale = np.minimum(1.0 / np.maximum(np.sum(miters * miters, axis=1), 1e-4), 100.0)
        outer = outline + miters * scale[:, None]

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
