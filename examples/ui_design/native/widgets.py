"""Reference controls: native hit testing and shared browser glyph contours."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from imgui_bundle import imgui

from .fonts import resolve_fonts

ROOT = Path(__file__).parent
GLYPHS = json.loads((ROOT / "icons.json").read_text())
TEXT = (230, 232, 235, 255)
MUTED = (169, 175, 183, 255)
DIM = (142, 152, 165, 255)
SAGE = (156, 191, 141, 255)
AMBER = (232, 176, 79, 255)
PANEL = (30, 33, 37, 255)
INPUT = (43, 47, 52, 255)
LINE = (255, 255, 255, 13)
# ImGuiFreeTypeLoaderFlags_NoHinting, from the bundled imgui_freetype.h.
# Sampling the outlines at 2x preserves small SF counters before GPU filtering.
FONT_NO_HINTING = 1
FONT_SAMPLING = 2


def font_config(em_ratio, face_index=0):
    config = imgui.ImFontConfig()
    config.extra_size_scale = em_ratio
    config.font_loader_flags = FONT_NO_HINTING
    config.rasterizer_density = FONT_SAMPLING
    # FreeType encodes one-based variable-font named instances in face-index bits 16–30.
    config.font_no = face_index
    return config


def color(c):
    return imgui.get_color_u32(imgui.ImVec4(*(v / 255 for v in c)))


@lru_cache(maxsize=512)
def icon_geometry(name, size):
    """Retain local vertices; moving a dock window only translates the submitted range."""
    scale = size / 24
    result = []
    for contour in GLYPHS[name]["contours"]:
        points = tuple(imgui.ImVec2(x * scale, y * scale) for x, y in contour["points"])
        paths = (
            tuple(
                tuple(imgui.ImVec2(x * scale, y * scale) for x, y in path)
                for path in contour["stroke_paths"]
            )
            if "stroke_paths" in contour
            else (points,)
        )
        result.append((contour, points, paths, contour["width"] * scale))
    return tuple(result)


class UI:
    def __init__(self, window, *, portable_fonts=False):
        self.window = window
        self.s = window.style_scale
        io = imgui.get_io()
        io.fonts.clear()
        self.font_sources = resolve_fonts(portable=portable_fonts)
        self.body, self.bold, self.mono = (
            io.fonts.add_font_from_file_ttf(
                source.path, size * self.s, font_config(em_ratio, source.index)
            )
            for (source, em_ratio), size in zip(self.font_sources, (12, 12, 11), strict=True)
        )
        io.font_default = self.body
        st = imgui.get_style()
        st.font_size_base = 12 * self.s
        st.window_rounding = 8 * self.s
        st.frame_rounding = 4.8 * self.s
        st.tab_rounding = 4.8 * self.s
        st.popup_rounding = 8 * self.s
        st.grab_rounding = 3 * self.s
        st.window_padding = (12 * self.s, 10 * self.s)
        st.frame_padding = (8 * self.s, 7 * self.s)
        st.item_spacing = (6 * self.s, 8 * self.s)
        st.item_inner_spacing = (6 * self.s, 4 * self.s)
        st.window_min_size = (230 * self.s, 120 * self.s)
        st.window_border_size = 1
        st.frame_border_size = 0
        st.popup_border_size = 1
        st.docking_separator_size = 2 * self.s
        st.tab_bar_border_size = 0
        st.tab_bar_overline_size = 0
        st.scrollbar_size = 7 * self.s
        st.scrollbar_rounding = 4 * self.s
        colors = {
            "text": TEXT,
            "text_disabled": DIM,
            "window_bg": PANEL,
            "child_bg": PANEL,
            "popup_bg": (37, 41, 46, 255),
            "frame_bg": INPUT,
            "frame_bg_hovered": (49, 54, 60, 255),
            "frame_bg_active": INPUT,
            "button": INPUT,
            "button_hovered": (54, 59, 65, 255),
            "button_active": (64, 71, 78, 255),
            "border": LINE,
            "separator": LINE,
            "header": (156, 191, 141, 24),
            "header_hovered": (255, 255, 255, 8),
            "header_active": (156, 191, 141, 32),
            "tab": (26, 29, 32, 255),
            "tab_hovered": (40, 44, 49, 255),
            "tab_selected": (30, 33, 37, 255),
            "tab_dimmed": (26, 29, 32, 255),
            "tab_dimmed_selected": (30, 33, 37, 255),
            "tab_selected_overline": (0, 0, 0, 0),
            "tab_dimmed_selected_overline": (0, 0, 0, 0),
            "title_bg": (26, 29, 32, 255),
            "title_bg_active": (26, 29, 32, 255),
            "check_mark": SAGE,
            "slider_grab": SAGE,
            "slider_grab_active": (177, 209, 161, 255),
            "docking_preview": (156, 191, 141, 90),
            "nav_cursor": SAGE,
            "scrollbar_bg": (0, 0, 0, 0),
            "scrollbar_grab": (69, 75, 82, 255),
        }
        for key, c in colors.items():
            st.set_color_(getattr(imgui.Col_, key), imgui.ImVec4(*(v / 255 for v in c)))
        self.hits = {}
        self.number_drafts = {}

    @property
    def draw(self):
        return imgui.get_window_draw_list()

    def text(self, x, y, text, c=TEXT, size=12, bold=False, mono=False, draw=None):
        font = self.mono if mono else self.bold if bold else self.body
        (draw or self.draw).add_text(font, size * self.s, (x, y), color(c), str(text))

    def measure(self, text, size=12):
        imgui.push_font(self.body, size * self.s)
        width = imgui.calc_text_size(text).x
        imgui.pop_font()
        return width

    def line(self, x, y, w):
        self.draw.add_line((x, y), (x + w, y), color(LINE), self.s)

    def rect(self, x, y, w, h, c, r=4.8):
        self.draw.add_rect_filled((x, y), (x + w, y + h), color(c), r * self.s)

    def icon(self, name, x, y, size=16, c=MUTED, draw=None):
        draw = draw or self.draw
        first_vertex = len(draw.vtx_buffer)
        for contour, pts, paths, thickness in icon_geometry(name, size * self.s):
            if len(pts) < 2:
                continue
            fill = contour["fill"]
            if fill != "none":
                fill_color = PANEL if fill.startswith("var") else c
                fill_color = (
                    *fill_color[:3],
                    round(fill_color[3] * contour["opacity"] * contour["fill_opacity"]),
                )
                draw.add_concave_poly_filled(pts, color(fill_color))
            if contour["stroke"] != "none":
                stroke_color = color(
                    (*c[:3], round(c[3] * contour["opacity"] * contour["stroke_opacity"]))
                )
                closed = contour["closed"] and "stroke_paths" not in contour
                for path in paths:
                    draw.add_polyline(
                        path, stroke_color, thickness, imgui.ImDrawFlags_.closed if closed else 0
                    )
                    if not closed:
                        for p in [path[0], path[-1]]:
                            draw.add_circle_filled(p, thickness / 2, stroke_color, 8)
        # Local tessellation avoids precision loss at large screen coordinates.
        imgui.internal.shade_verts_transform_pos(
            draw, first_vertex, len(draw.vtx_buffer), (0, 0), 1.0, 0.0, (x, y)
        )

    def tooltip(self, text):
        imgui.push_style_var(imgui.StyleVar_.window_padding, (9 * self.s, 7 * self.s))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, 4.8 * self.s)
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 260 * self.s)
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
        imgui.pop_style_var(3)

    def button(
        self,
        key,
        x,
        y,
        w=28,
        h=28,
        *,
        icon=None,
        label="",
        active=False,
        filled=False,
        enabled=True,
        tooltip="",
        text_color=None,
        icon_color=None,
        solid=False,
        icon_size=16,
    ):
        s = self.s
        w *= s
        h *= s
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_disabled(not enabled)
        hit = imgui.invisible_button(key, (w, h), imgui.ButtonFlags_.enable_nav)
        hovered = imgui.is_item_hovered()
        self.hits[key] = (x, y, w, h)
        if active:
            self.rect(x, y, w, h, SAGE if solid else (156, 191, 141, 30))
        elif hovered:
            self.rect(x, y, w, h, (255, 255, 255, 12))
        elif filled:
            self.rect(x, y, w, h, INPUT)
        c = text_color or (
            (30, 40, 27, 255) if active and solid else SAGE if active else TEXT if label else MUTED
        )
        # BeginDisabled already applies the shared ImGui disabled alpha to every painter.
        if icon:
            self.icon(
                icon,
                x + (8 * s if label else (w - icon_size * s) / 2),
                y + (h - icon_size * s) / 2,
                icon_size,
                icon_color or c,
            )
        if label:
            pad = 29 * s if icon else 10 * s
            self.text(x + pad, y + (h - 12 * s) / 2, label, c, 12)
        if tooltip and imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        ):
            self.tooltip(tooltip)
        imgui.end_disabled()
        return hit

    def input(self, key, x, y, w, value, *, axis=None):
        s = self.s
        if axis:
            self.rect(x, y, w, 26 * s, INPUT)
            c = {"X": (205, 140, 137, 150), "Y": (150, 194, 151, 150), "Z": (145, 172, 214, 150)}[
                axis
            ]
            self.text(x + 7 * s, y + 7 * s, axis, c, 10)
        imgui.set_cursor_screen_pos((x + (19 * s if axis else 0), y))
        imgui.set_next_item_width(max(25 * s, w - (19 * s if axis else 0)))
        imgui.push_font(self.mono, 11 * s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (5 * s, 7.5 * s))
        original, draft = self.number_drafts.get(key, (value, value))
        changed, draft = imgui.input_float(key, float(draft), 0, 0, "%.3g")
        active = imgui.is_item_active()
        finished = imgui.is_item_deactivated()
        if active:
            self.number_drafts[key] = (original, draft)
            changed = False
        elif finished and key in self.number_drafts:
            self.number_drafts.pop(key)
            changed = draft != original and not imgui.is_key_pressed(imgui.Key.escape)
        value = draft if changed else value
        imgui.pop_style_var()
        imgui.pop_font()
        self.hits[key] = (x, y, w, 26 * s)
        return changed, value

    def vector(self, key, x, y, w, label, values, unit=""):
        s = self.s
        self.text(x, y + 8 * s, label, MUTED, 11)
        if unit:
            self.text(x + 62 * s, y + 8 * s, unit, DIM, 10)
        stacked = w < 272 * s
        start = x if stacked else x + 82 * s
        field = (w - (0 if stacked else 82 * s) - 8 * s) / 3
        if stacked:
            y += 22 * s
        changed = False
        out = list(values)
        for n, a in enumerate("XYZ"):
            edit, out[n] = self.input(
                key + a, start + n * (field + 4 * s), y, field, out[n], axis=a
            )
            changed |= edit
        return changed, out

    def section(self, label, x, y, w, open=None):
        self.line(x, y, w)
        hit = False
        if open is not None:
            imgui.set_cursor_screen_pos((x, y + 4 * self.s))
            hit = imgui.invisible_button(
                "section-" + label, (w, 30 * self.s), imgui.ButtonFlags_.enable_nav
            )
            self.hits["section-" + label] = (x, y + 4 * self.s, w, 30 * self.s)
            self.icon("down" if open else "right", x + w - 14 * self.s, y + 12 * self.s, 12, DIM)
        self.text(x + 2 * self.s, y + 14 * self.s, label, MUTED, 10, bold=True)
        return hit

    def switch(self, key, x, y, on):
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        hit = imgui.invisible_button(key, (30 * s, 22 * s), imgui.ButtonFlags_.enable_nav)
        self.rect(x, y + 4 * s, 26 * s, 14 * s, (88, 108, 78, 255) if on else (55, 62, 68, 255), 7)
        self.draw.add_circle_filled(
            (x + (18 if on else 8) * s, y + 11 * s), 5 * s, color(SAGE if on else MUTED)
        )
        self.hits[key] = (x, y, 30 * s, 22 * s)
        return not on if hit else on
