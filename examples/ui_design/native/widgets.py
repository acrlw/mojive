"""Reference controls: native hit testing and shared browser glyph contours."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from imgui_bundle import imgui

from mojive.ui.compound_fields import borderless_numeric_input, draw_focus_frame
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.text_layout import fit_text
from mojive.ui.theme import THEME

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
        self.translate = lambda value: value
        io = imgui.get_io()
        io.fonts.clear()
        self.font_sources = resolve_fonts(portable=portable_fonts)
        fonts = []
        for (source, em_ratio), size in zip(self.font_sources, (14, 14, 13), strict=True):
            fonts.append(
                io.fonts.add_font_from_file_ttf(
                    source.path, size * self.s, font_config(em_ratio, source.index)
                )
            )
            cjk = window.font_report
            if cjk.cjk_path:
                merged = font_config(1.0, cjk.cjk_index)
                merged.merge_mode = True
                io.fonts.add_font_from_file_ttf(cjk.cjk_path, size * self.s, merged)
        self.body, self.bold, self.mono = fonts
        io.font_default = self.body
        st = imgui.get_style()
        st.font_size_base = 14 * self.s
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
        self.value_context = SimpleNamespace(
            style_scale=self.s,
            theme=THEME,
            tr=lambda value: self.translate(value),
            input_bindings=DEFAULT_INPUT_BINDINGS,
        )

    @property
    def draw(self):
        return imgui.get_window_draw_list()

    def text(self, x, y, text, c=TEXT, size=14, bold=False, mono=False, draw=None):
        font = self.mono if mono else self.bold if bold else self.body
        (draw or self.draw).add_text(
            font, size * self.s, (x, y), color(c), self.translate(str(text))
        )

    def measure(self, text, size=14):
        imgui.push_font(self.body, size * self.s)
        width = imgui.calc_text_size(self.translate(text)).x
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
        imgui.text_unformatted(self.translate(text))
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
        align="left",
    ):
        s = self.s
        w *= s
        h *= s
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_disabled(not enabled)
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(key, (w, h), imgui.ButtonFlags_.enable_nav)
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
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
        shown = ""
        translated_label = self.translate(label)
        label_width = 0
        if label:
            horizontal_padding = 37 if icon else 12 if align == "center" else 20
            imgui.push_font(self.body, 14 * s)
            shown = fit_text(
                ImguiDraw2D(),
                translated_label,
                max(0, w - horizontal_padding * s),
            )
            label_width = imgui.calc_text_size(shown).x
            imgui.pop_font()
        content_width = label_width + ((icon_size + 5) * s if icon and shown else 0)
        start = (
            x + max((8 if icon else 6) * s, (w - content_width) * 0.5)
            if align == "center"
            else x + 8 * s
        )
        # The native hit region and clipping share the same control rectangle.
        self.draw.push_clip_rect((x + s, y), (x + w - s, y + h), True)
        if icon:
            self.icon(
                icon,
                start if shown else x + (w - icon_size * s) / 2,
                y + (h - icon_size * s) / 2,
                icon_size,
                icon_color or c,
            )
        if shown:
            text_x = (
                start + (icon_size + 5) * s if icon else start + (0 if align == "center" else 2 * s)
            )
            self.text(text_x, y + (h - 14 * s) / 2, shown, c, 14)
        self.draw.pop_clip_rect()
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + w, y + h),
            rounding=4.8 * s,
            item_id=item_id,
        )
        hint = tooltip or (translated_label if shown != translated_label else "")
        if hint and imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        ):
            self.tooltip(hint)
        imgui.end_disabled()
        return hit

    def input(self, key, x, y, w, value, *, axis=None):
        s = self.s
        height = 28 * s
        badge = min(22 * s, max(0, w - s)) if axis else 0
        active = imgui.internal.get_active_id() == imgui.get_id(key)
        hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
            (x, y), (x + w, y + height)
        )
        self.rect(x, y, w, height, (49, 54, 60, 255) if active or hovered else INPUT)
        if axis:
            c = {"X": (205, 140, 137, 220), "Y": (150, 194, 151, 220), "Z": (145, 172, 214, 220)}[
                axis
            ]
            self.draw.push_clip_rect((x, y), (x + badge, y + height), True)
            self.rect(x + 3 * s, y + 7 * s, 2 * s, 14 * s, c, 1)
            self.text(x + 8 * s, y + 8 * s, axis, c, 12)
            self.draw.pop_clip_rect()
        imgui.set_cursor_screen_pos((x + badge, y))
        imgui.set_next_item_width(max(s, w - badge))
        imgui.push_font(self.mono, 13 * s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (5 * s, 7.5 * s))
        for slot in (imgui.Col_.frame_bg, imgui.Col_.frame_bg_hovered, imgui.Col_.frame_bg_active):
            imgui.push_style_color(slot, (0, 0, 0, 0))
        original, draft = self.number_drafts.get(key, (value, value))
        with borderless_numeric_input():
            changed, draft = imgui.input_float(key, float(draft), 0, 0, "%.3g")
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        item_id = imgui.get_item_id()
        active = imgui.is_item_active()
        finished = imgui.is_item_deactivated()
        if active:
            self.number_drafts[key] = (original, draft)
            changed = False
        elif finished and key in self.number_drafts:
            self.number_drafts.pop(key)
            changed = draft != original and not imgui.is_key_pressed(imgui.Key.escape)
        value = draft if changed else value
        imgui.pop_style_color(3)
        imgui.pop_style_var()
        imgui.pop_font()
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + w, hi.y),
            rounding=4.8 * s,
            item_id=item_id,
        )
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        self.hits[key + "-frame"] = (x, y, w, hi.y - y)
        return changed, value

    def vector(self, key, x, y, w, label, values, unit=""):
        s = self.s
        heading = self.translate(label) + (f" ({unit})" if unit else "")
        self.text(x, y + 7.5 * s, heading, MUTED, 13)
        stacked = w < 272 * s
        start = x if stacked else x + 82 * s
        field = (w - (0 if stacked else 82 * s) - 8 * s) / 3
        if stacked:
            y += 24 * s
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
        s = self.s
        item_id = None
        if open is not None:
            imgui.set_cursor_screen_pos((x, y + 4 * s))
            imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
            hit = imgui.invisible_button(
                "section-" + label, (w, 30 * s), imgui.ButtonFlags_.enable_nav
            )
            imgui.pop_style_color()
            item_id = imgui.get_item_id()
            self.hits["section-" + label] = (x, y + 4 * s, w, 30 * s)
            if imgui.is_item_hovered():
                self.rect(x, y + 4 * s, w, 30 * s, (255, 255, 255, 7))
            self.icon("down" if open else "right", x + w - 16 * s, y + 13 * s, 12, DIM)
        self.draw.push_clip_rect((x, y + 4 * s), (x + max(0, w - 22 * s), y + 34 * s), True)
        self.text(x + 2 * s, y + 12.5 * s, label, MUTED, 13, bold=True)
        self.draw.pop_clip_rect()
        if item_id is not None:
            draw_focus_frame(
                imgui.ImVec2(x, y + 4 * s),
                imgui.ImVec2(x + w, y + 34 * s),
                rounding=4.8 * s,
                item_id=item_id,
            )
        return hit

    def slider(self, key, x, y, w, value, lo, hi, format="%.2f"):
        """Reuse the production value rail and native numeric entry in one row."""
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(max(s, w))
        imgui.push_font(self.mono, 13 * s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (6 * s, 7.5 * s))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (8 * s, 0))
        if w >= 164 * s and lo < hi:
            edit = value_rail(
                self.value_context,
                "##" + key,
                value,
                (lo, hi),
                initial=None,
                fmt=format,
                show_reset=False,
            )
            changed, value = edit.changed, edit.value
        else:
            changed, value = imgui.drag_float(
                "##" + key + "-value",
                value,
                (hi - lo) / 200 if lo < hi else 0.01,
                lo,
                hi,
                format,
                imgui.SliderFlags_.always_clamp,
            )
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key + "-value"] = (a.x, a.y, b.x - a.x, b.y - a.y)
        self.hits[key] = (x, y, w, b.y - y)
        if a.x > x:
            self.hits[key + "-track"] = (x, y, a.x - x - 8 * s, b.y - y)
        else:
            self.hits.pop(key + "-track", None)
        imgui.pop_style_var(2)
        imgui.pop_font()
        return changed, value

    def switch(self, key, x, y, on):
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(key, (30 * s, 22 * s), imgui.ButtonFlags_.enable_nav)
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
        self.rect(
            x + 2 * s, y + 4 * s, 26 * s, 14 * s, (88, 108, 78, 255) if on else (55, 62, 68, 255), 7
        )
        self.draw.add_circle_filled(
            (x + (20 if on else 10) * s, y + 11 * s), 5 * s, color(SAGE if on else MUTED)
        )
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + 30 * s, y + 22 * s),
            rounding=7 * s,
            item_id=item_id,
        )
        self.hits[key] = (x, y, 30 * s, 22 * s)
        return not on if hit else on
