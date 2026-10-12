"""Reference controls: native hit testing and shared browser glyph contours."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from imgui_bundle import imgui

from mojive.text.sources import FontSource
from mojive.ui.compound_fields import (
    borderless_numeric_input,
    draw_focus_frame,
    draw_joined_field_frame,
)
from mojive.ui.controls import segmented_control
from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.panels.inspector.fields import _mix_color
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.severity_icons import severity_icon
from mojive.ui.text_layout import fit_text, text_line_y
from mojive.ui.theme import THEME

from .design import BODY_SIZE, CONTROL_RADIUS, DESIGN, MONO_SIZE, RAIL_STYLE
from .fonts import font_em_ratio, font_vertical_offset, resolve_cjk_faces, resolve_fonts

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
CONTROL_THEME = replace(
    THEME,
    text=tuple(channel / 255 for channel in TEXT),
    primary_bright=tuple(channel / 255 for channel in SAGE),
)
FORM_LABEL_COLUMN_WIDTH = DESIGN.label_width
FORM_LABEL_COLUMN_MINIMUM = DESIGN.label_minimum
FORM_LABEL_COLUMN_GAP = DESIGN.label_gap
SWITCH_WIDTH = 36
SWITCH_HEIGHT = DESIGN.control_height
# ImGuiFreeTypeLoaderFlags_NoHinting, from the bundled imgui_freetype.h.
# Sampling the outlines at 2x preserves small SF counters before GPU filtering.
FONT_NO_HINTING = 1
FONT_BOLD = 1 << 5
FONT_SAMPLING = 2


ICON_GRID = 24
ICON_STROKE = 1.75
IconRole = Literal["activity", "title", "toolbar", "subtle", "inline", "status"]


@dataclass(frozen=True, slots=True)
class IconRoleSpec:
    """Canonical canvas and default hit-slot sizes in logical points."""

    canvas_size: float
    slot_size: float


ICON_ROLES: dict[IconRole, IconRoleSpec] = {
    "activity": IconRoleSpec(24, 36),
    "title": IconRoleSpec(22, 30),
    "toolbar": IconRoleSpec(18, 28),
    "subtle": IconRoleSpec(16, 28),
    "inline": IconRoleSpec(BODY_SIZE, 28),
    "status": IconRoleSpec(12, 20),
}


def icon_role_canvas(role: IconRole) -> float:
    """Keep one family scale; a small authored mark remains small beside its peers."""
    return ICON_ROLES[role].canvas_size


@dataclass(frozen=True, slots=True)
class NumericEditState:
    """Native item signals for the caller's command transaction."""

    activated: bool = False
    active: bool = False
    deactivated: bool = False
    committed: bool = False
    canceled: bool = False
    reset: bool = False


@dataclass(frozen=True, slots=True)
class NumericGroupLayout:
    """Shared field rectangles and complete group height in screen coordinates."""

    fields: tuple[tuple[float, float, float, float], ...]
    columns: int
    height: float
    label_above: bool = False
    label_width: float = 0.0


def font_config(em_ratio, face_index=0, *, embolden=False, vertical_offset=0):
    config = imgui.ImFontConfig()
    config.extra_size_scale = em_ratio
    config.font_loader_flags = FONT_NO_HINTING | (FONT_BOLD if embolden else 0)
    config.rasterizer_density = FONT_SAMPLING
    # FreeType encodes one-based variable-font named instances in face-index bits 16–30.
    config.font_no = face_index
    config.glyph_offset = (0, vertical_offset)
    return config


def color(c):
    return imgui.get_color_u32(imgui.ImVec4(*(v / 255 for v in c)))


@lru_cache(maxsize=512)
def icon_geometry(name, size):
    """Retain local vertices; moving a dock window only translates the submitted range."""
    scale = size / ICON_GRID
    left, top, right, bottom = GLYPHS[name]["bounds"]
    dx = (ICON_GRID / 2 - (left + right) / 2) * scale
    dy = (ICON_GRID / 2 - (top + bottom) / 2) * scale
    result = []
    for contour in GLYPHS[name]["contours"]:
        points = tuple((x * scale + dx, y * scale + dy) for x, y in contour["points"])
        paths = (
            tuple(
                tuple((x * scale + dx, y * scale + dy) for x, y in path)
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
        report = window.font_report
        cjk = (
            FontSource(report.cjk_path, report.cjk_index, getattr(report, "cjk", "CJK"))
            if report.cjk_path
            else None
        )
        self.font_sources = resolve_fonts(portable=portable_fonts, cjk=cjk)
        regular, heading = resolve_cjk_faces(cjk)
        fonts = []
        for (source, em_ratio), size, fallback in zip(
            self.font_sources,
            (BODY_SIZE, BODY_SIZE, MONO_SIZE),
            (regular, heading, regular),
            strict=True,
        ):
            config = font_config(
                em_ratio, source.index, embolden=getattr(source, "embolden", False)
            )
            config.glyph_offset = (0, font_vertical_offset(source, size * self.s, config))
            fonts.append(
                io.fonts.add_font_from_file_ttf(
                    source.path,
                    size * self.s,
                    config,
                )
            )
            if fallback is not None and (source.path, source.index) != (
                fallback.path,
                fallback.index,
            ):
                merged = font_config(
                    font_em_ratio(fallback), fallback.index, embolden=fallback.embolden
                )
                merged.glyph_offset = (
                    0,
                    font_vertical_offset(
                        fallback,
                        size * self.s,
                        merged,
                        ascent=fonts[-1]
                        .get_font_baked(size * self.s, io.display_framebuffer_scale.x)
                        .ascent,
                    ),
                )
                merged.merge_mode = True
                io.fonts.add_font_from_file_ttf(fallback.path, size * self.s, merged)
        self.body, self.bold, self.mono = fonts
        io.font_default = self.body
        st = imgui.get_style()
        st.font_size_base = BODY_SIZE * self.s
        st.window_rounding = DESIGN.container_radius * self.s
        st.frame_rounding = CONTROL_RADIUS * self.s
        st.tab_rounding = CONTROL_RADIUS * self.s
        st.popup_rounding = 8 * self.s
        st.grab_rounding = DESIGN.grab_radius * self.s
        st.child_rounding = DESIGN.child_radius * self.s
        st.window_padding = (DESIGN.panel_padding_x * self.s, DESIGN.panel_padding_y * self.s)
        st.frame_padding = (DESIGN.frame_padding_x * self.s, DESIGN.frame_padding_y * self.s)
        st.item_spacing = (DESIGN.item_spacing_x * self.s, DESIGN.row_gap * self.s)
        st.item_inner_spacing = (DESIGN.item_spacing_x * self.s, DESIGN.numeric_gap * self.s)
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
            "separator_hovered": (91, 99, 108, 255),
            "separator_active": (121, 131, 142, 255),
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
        self.numeric_edits = {}
        self.value_context = SimpleNamespace(
            style_scale=self.s,
            rail_style=RAIL_STYLE,
            theme=THEME,
            tr=lambda value: self.translate(value),
            input_bindings=DEFAULT_INPUT_BINDINGS,
        )

    @property
    def draw(self):
        return imgui.get_window_draw_list()

    def text(
        self,
        x,
        y,
        text,
        c=TEXT,
        size=BODY_SIZE,
        bold=False,
        mono=False,
        draw=None,
        *,
        max_width=None,
        translate=True,
        native_line=False,
    ):
        """Center cap height in the requested line box, retaining a common text baseline."""
        font = self.mono if mono else self.bold if bold else self.body
        draw_list = draw or self.draw
        imgui.push_font(font, size * self.s)
        painter = ImguiDraw2D(draw_list)
        line_y = y if native_line else text_line_y(painter, y + size * self.s / 2)
        shown = self.translate(str(text)) if translate else str(text)
        if max_width is not None:
            shown = fit_text(painter, shown, max(0, max_width))
        painter.text(
            (x, line_y),
            (*[channel / 255 for channel in c[:3]], c[3] / 255 * imgui.get_style().alpha),
            shown,
            pixel_snap=native_line,
        )
        imgui.pop_font()

    def measure(self, text, size=BODY_SIZE, *, translate=True):
        imgui.push_font(self.body, size * self.s)
        width = imgui.calc_text_size(self.translate(text) if translate else text).x
        imgui.pop_font()
        return width

    def line(self, x, y, w):
        self.draw.add_line((x, y), (x + w, y), color(LINE), self.s)

    def rect(self, x, y, w, h, c, r=CONTROL_RADIUS):
        self.draw.add_rect_filled((x, y), (x + w, y + h), color(c), r * self.s)

    def icon(self, name, x, y, size=16, c=MUTED, draw=None):
        draw = draw or self.draw
        severity = {"info": "info", "warn": "warning", "err": "error"}.get(name)
        if severity is not None:
            severity_icon(
                ImguiDraw2D(draw),
                (x + size * self.s / 2, y + size * self.s / 2),
                size * self.s,
                severity,
                (*[channel / 255 for channel in c[:3]], c[3] / 255 * imgui.get_style().alpha),
            )
            return
        first_vertex = len(draw.vtx_buffer)
        # Filled strokes retain the canonical weight below one pixel as well.
        painter = ImguiIconDraw(
            ImguiDraw2D(draw), min(1.0, ICON_STROKE * size * self.s / (2 * ICON_GRID))
        )
        alpha = imgui.get_style().alpha
        for contour, pts, paths, thickness in icon_geometry(name, size * self.s):
            if len(pts) < 2:
                continue
            fill = contour["fill"]
            if fill != "none":
                fill_color = PANEL if fill.startswith("var") else c
                fill_color = (
                    *(channel / 255 for channel in fill_color[:3]),
                    fill_color[3] / 255 * contour["opacity"] * contour["fill_opacity"] * alpha,
                )
                painter.fringed_concave_fill(pts, fill_color)
            if contour["stroke"] != "none":
                stroke_color = (
                    *(channel / 255 for channel in c[:3]),
                    c[3] / 255 * contour["opacity"] * contour["stroke_opacity"] * alpha,
                )
                closed = contour["closed"] and "stroke_paths" not in contour
                for path in paths:
                    painter.polyline(
                        path, stroke_color, thickness, closed=closed, cap="round", smoothing=0.0
                    )
        # Local tessellation avoids precision loss at large screen coordinates.
        imgui.internal.shade_verts_transform_pos(
            draw, first_vertex, len(draw.vtx_buffer), (0, 0), 1.0, 0.0, (x, y)
        )

    def role_icon(
        self, name, x, y, w=None, h=None, c=MUTED, *, role: IconRole = "toolbar", draw=None
    ):
        """Center the master anchor in a physical slot without equalizing glyph bounds."""
        draw = draw or self.draw
        spec = ICON_ROLES[role]
        w = spec.slot_size * self.s if w is None else w
        h = spec.slot_size * self.s if h is None else h
        fit = min(1.0, max(0.0, min(w, h)) / (spec.canvas_size * self.s))
        if fit <= 0:
            return
        size = icon_role_canvas(role) * fit
        canvas = size * self.s
        left, top = x + (w - canvas) / 2, y + (h - canvas) / 2
        self.icon(name, left, top, size, c, draw)

    def tooltip(self, text, *, translate=True):
        imgui.push_style_var(imgui.StyleVar_.window_padding, (9 * self.s, 7 * self.s))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, CONTROL_RADIUS * self.s)
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 260 * self.s)
        imgui.text_unformatted(self.translate(text) if translate else text)
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
        icon_size=None,
        icon_role: IconRole | None = None,
        align="center",
        background=True,
        translate=True,
    ):
        s = self.s
        icon_role = icon_role or ("inline" if label else "toolbar")
        w *= s
        h *= s
        imgui.set_cursor_screen_pos((x, y))
        original_alpha = imgui.get_style().alpha
        imgui.begin_disabled(not enabled)
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(key, (w, h), imgui.ButtonFlags_.enable_nav)
        disabled = bool(imgui.get_item_flags() & imgui.ItemFlags_.disabled)
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
        hovered = imgui.is_item_hovered()
        held = imgui.is_item_active()
        self.hits[key] = (x, y, w, h)
        if background and held:
            self.rect(x, y, w, h, (130, 164, 116, 255) if active and solid else (156, 191, 141, 55))
        elif background and active:
            self.rect(x, y, w, h, SAGE if solid else (156, 191, 141, 42 if hovered else 30))
        elif background and hovered:
            self.rect(x, y, w, h, (255, 255, 255, 12))
        elif background and filled:
            self.rect(x, y, w, h, INPUT)
        c = text_color or (
            (30, 40, 27, 255) if active and solid else SAGE if active else TEXT if label else MUTED
        )
        shown = ""
        translated_label = self.translate(label) if translate else label
        label_width = 0
        icon_extent = ICON_ROLES[icon_role].canvas_size if icon_size is None else icon_size
        if label:
            horizontal_padding = icon_extent + 21 if icon else 12 if align == "center" else 20
            imgui.push_font(self.body, BODY_SIZE * s)
            shown = fit_text(
                ImguiDraw2D(),
                translated_label,
                max(0, w - horizontal_padding * s),
            )
            label_width = imgui.calc_text_size(shown).x
            imgui.pop_font()
        content_width = label_width + ((icon_extent + 5) * s if icon and shown else 0)
        start = (
            x + max((8 if icon else 6) * s, (w - content_width) * 0.5)
            if align == "center"
            else x + 8 * s
        )
        glyph_color = icon_color or c
        if disabled:
            # Opaque muted ink keeps intersecting strokes uniform when disabled.
            original_alpha = imgui.get_current_context().disabled_alpha_backup
            opacity = imgui.get_style().alpha / max(original_alpha, 1e-9)

            def muted_ink(tint):
                return (
                    *(
                        round(a * opacity + b * (1 - opacity))
                        for a, b in zip(tint[:3], PANEL[:3], strict=True)
                    ),
                    tint[3],
                )

            c, glyph_color = muted_ink(c), muted_ink(glyph_color)
            imgui.push_style_var(imgui.StyleVar_.alpha, original_alpha)
        # The native hit region and clipping share the same control rectangle.
        self.draw.push_clip_rect((x + s, y), (x + w - s, y + h), True)
        if icon:
            icon_x = start if shown else x + (w - icon_extent * s) / 2
            icon_y = y + (h - icon_extent * s) / 2
            if icon_size is None:
                self.role_icon(
                    icon,
                    icon_x,
                    icon_y,
                    icon_extent * s,
                    icon_extent * s,
                    glyph_color,
                    role=icon_role,
                )
            else:
                self.icon(icon, icon_x, icon_y, icon_size, glyph_color)
        if shown:
            text_x = (
                start + (icon_extent + 5) * s
                if icon
                else start + (0 if align == "center" else 2 * s)
            )
            self.text(text_x, y + (h - BODY_SIZE * s) / 2, shown, c, translate=False)
        self.draw.pop_clip_rect()
        if disabled:
            imgui.pop_style_var()
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + w, y + h),
            rounding=CONTROL_RADIUS * s,
            item_id=item_id,
        )
        hint = tooltip or (translated_label if shown != translated_label else "")
        if hint and imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        ):
            self.tooltip(hint, translate=translate)
        imgui.end_disabled()
        return hit

    def segmented(
        self,
        key,
        x,
        y,
        w,
        labels,
        selected,
        *,
        keys=None,
        icons=None,
        enabled=True,
        layout="equal",
        show_labels=True,
        tooltips=None,
    ):
        """Choose one value in the shared contiguous shell and retain native hit bounds."""
        if keys is not None and len(keys) != len(labels):
            raise ValueError("Each segment requires one hit key")
        if tooltips is not None and len(tooltips) != len(labels):
            raise ValueError("Each segment requires one tooltip")
        if not labels:
            self.hits[key] = (x, y, w * self.s, 0)
            return 0

        def record(index):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            item_key = keys[index] if keys is not None else f"{key}-{index}"
            self.hits[item_key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            if tooltips is not None and imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                self.tooltip(tooltips[index])

        imgui.push_font(self.body, BODY_SIZE * self.s)
        padding_y = max(0, (28 * self.s - imgui.get_font_size()) / 2)
        padding_x = (6 if layout == "content" else 8) * self.s
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (padding_x, padding_y))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, CONTROL_RADIUS * self.s)
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_disabled(not enabled)
        chosen = segmented_control(
            key,
            tuple(self.translate(label) for label in labels),
            selected,
            width=w * self.s,
            theme=CONTROL_THEME,
            icons=icons,
            icon_label_drawer=self._segment_label,
            on_item=record,
            layout=layout,
            show_labels=show_labels,
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        imgui.end_disabled()
        imgui.pop_style_var(2)
        imgui.pop_font()
        return chosen

    def _segment_label(self, draw, lo, hi, c, scale, icon, label):
        """Place native glyphs and captions on the shared segment's visual center."""
        alpha = imgui.get_style().alpha
        rgba = (
            *[round(channel * 255) for channel in c[:3]],
            round(c[3] * 255 / alpha) if alpha else 0,
        )
        role: IconRole = "inline" if label else "toolbar"
        icon_width = ICON_ROLES[role].canvas_size * self.s
        caption_width = imgui.calc_text_size(label).x if label else 0
        content_width = icon_width + (5 * self.s + caption_width if label else 0)
        x, center_y = lo[0] + (hi[0] - lo[0] - content_width) / 2, (lo[1] + hi[1]) / 2
        self.role_icon(icon, x, center_y - icon_width / 2, icon_width, icon_width, rgba, role=role)
        if label:
            self.text(
                x + icon_width + 5 * self.s,
                center_y - BODY_SIZE * self.s / 2,
                label,
                rgba,
                translate=False,
            )

    def input(
        self,
        key,
        x,
        y,
        w,
        value,
        *,
        axis=None,
        speed=0.01,
        lo=0.0,
        hi=0.0,
        format="%.3f",
        reset_value=0.0,
    ):
        """Draw a native drag field; callers own command transactions and cancellation."""
        s = self.s
        badge = min(18 * s, max(0, w * 0.35)) if axis else 0
        draw_list = self.draw
        splitter = None
        reset = False
        badge_active = badge_hovered = badge_activated = badge_deactivated = False
        badge_id = None
        imgui.push_font(self.mono, MONO_SIZE * s)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding, (8 * s, DESIGN.control_padding_y(MONO_SIZE) * s)
        )
        if axis:
            splitter = imgui.ImDrawListSplitter()
            splitter.split(draw_list, 2)
            splitter.set_current_channel(draw_list, 1)
            imgui.set_cursor_screen_pos((x, y))
            for slot in (
                imgui.Col_.button,
                imgui.Col_.button_hovered,
                imgui.Col_.button_active,
                imgui.Col_.nav_cursor,
            ):
                imgui.push_style_color(slot, (0, 0, 0, 0))
            reset = imgui.button("##" + key + "-reset", (badge, imgui.get_frame_height()))
            badge_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            badge_active = imgui.is_item_active()
            badge_activated = imgui.is_item_activated()
            badge_deactivated = imgui.is_item_deactivated()
            badge_lo, badge_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            badge_id = imgui.get_item_id()
            disabled = bool(imgui.get_item_flags() & imgui.ItemFlags_.disabled)
            imgui.pop_style_color(4)
            for slot in (
                imgui.Col_.frame_bg,
                imgui.Col_.frame_bg_hovered,
                imgui.Col_.frame_bg_active,
            ):
                imgui.push_style_color(slot, (0, 0, 0, 0))
        imgui.set_cursor_screen_pos((x + badge, y))
        imgui.set_next_item_width(max(s, w - badge))
        # This painter owns all visible text. A bare ID would add an ImGui label
        # after the field and enlarge both its item bounds and popup content.
        item_label = key if key.startswith("##") else "##" + key
        item_id = imgui.get_id(item_label)
        context, io = imgui.get_current_context(), imgui.get_io()
        hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
            (x + badge, y), (x + w, y + imgui.get_frame_height())
        )
        # Preserve native text on its activation frame. DragFloat still owns
        # activation and editing; these requests only select its text visibility.
        native_text = not bool(context.current_item_flags & imgui.ItemFlags_.disabled) and (
            imgui.internal.temp_input_is_active(item_id)
            or (
                hovered
                and (
                    (imgui.is_mouse_clicked(0) and io.key_ctrl)
                    or imgui.is_mouse_double_clicked(0)
                    or (
                        io.config_drag_click_to_input_text
                        and context.active_id == item_id
                        and imgui.is_mouse_released(0)
                    )
                )
            )
            or (
                context.nav_activate_id == item_id
                and bool(context.nav_activate_flags & imgui.internal.ActivateFlags_.prefer_input)
            )
        )
        if not native_text:
            imgui.push_style_color(imgui.Col_.text, (0, 0, 0, 0))
        draw_list.push_clip_rect((x + badge, y), (x + w, y + imgui.get_frame_height()), True)
        with borderless_numeric_input():
            changed, edited = imgui.drag_float(
                item_label,
                float(value),
                speed,
                lo,
                hi,
                format,
                imgui.SliderFlags_.always_clamp if lo < hi else 0,
            )
        draw_list.pop_clip_rect()
        if not native_text:
            imgui.pop_style_color()
        field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        item_id = imgui.get_item_id()
        text_editing = imgui.internal.temp_input_is_active(item_id)
        field_active = imgui.is_item_active()
        field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        active = field_active or badge_active
        deactivated = imgui.is_item_deactivated() or badge_deactivated
        state = NumericEditState(
            activated=imgui.is_item_activated() or badge_activated,
            active=active,
            deactivated=deactivated,
            committed=imgui.is_item_deactivated_after_edit() or reset,
            canceled=(active or deactivated) and imgui.is_key_pressed(imgui.Key.escape, False),
            reset=reset,
        )
        self.numeric_edits[key] = state
        if axis:
            imgui.pop_style_color(3)
            hovered = imgui.is_item_hovered()
            axis_index = {"X": 0, "R": 0, "Y": 1, "G": 1, "Z": 2, "B": 2}.get(axis)
            axis_color = (
                THEME.axis_color(axis_index) if axis_index is not None else THEME.text_disabled
            )
            field_color = (
                THEME.bg_frame_active
                if field_active
                else THEME.bg_frame_hovered
                if hovered
                else THEME.bg_frame
            )
            splitter.set_current_channel(draw_list, 0)
            draw_joined_field_frame(
                draw_list,
                badge_lo,
                badge_hi,
                field_lo,
                field_hi,
                badge_color=_mix_color(
                    THEME.bg_frame_active
                    if badge_active
                    else THEME.bg_frame_hovered
                    if badge_hovered and not disabled
                    else THEME.bg_frame,
                    axis_color,
                    0.88 if badge_active else 0.72 if badge_hovered and not disabled else 0.56,
                ),
                field_color=field_color,
                rounding=CONTROL_RADIUS * s,
                badge_opacity=imgui.get_style().alpha,
                field_opacity=imgui.get_style().alpha,
            )
            splitter.set_current_channel(draw_list, 1)
            draw_list.push_clip_rect((x, y), (x + badge, field_hi.y), True)
            self.text(x + (badge - self.measure(axis, 12)) / 2, y + 8 * s, axis, TEXT, 12)
            draw_list.pop_clip_rect()
            draw_focus_frame(
                badge_lo,
                badge_hi,
                rounding=CONTROL_RADIUS * s,
                corners=imgui.ImDrawFlags_.round_corners_left,
                item_id=badge_id,
            )
            self.hits[key + "-reset"] = (
                badge_lo.x,
                badge_lo.y,
                badge_hi.x - badge_lo.x,
                badge_hi.y - badge_lo.y,
            )
        current = reset_value if reset else edited if changed else value
        truncated = False
        if not native_text and not text_editing:
            shown_value = format % current
            fixed_number = format in ("%f", "%F") or (
                format.startswith("%.") and format.endswith(("f", "F")) and format[2:-1].isdigit()
            )
            if fixed_number and "." in shown_value:
                shown_value = shown_value.rstrip("0").rstrip(".")
            padding = min(8 * s, max(0, (field_hi.x - field_lo.x) / 2))
            text_left, text_right = field_lo.x + padding, field_hi.x - padding
            content_width = max(0, text_right - text_left)
            text_width = imgui.calc_text_size(self.translate(shown_value)).x
            truncated = text_width > content_width
            text_x = text_left + max(0, (content_width - text_width) / 2)
            draw_list.push_clip_rect((text_left, field_lo.y), (text_right, field_hi.y), True)
            self.text(
                text_x,
                field_lo.y + imgui.get_style().frame_padding.y,
                shown_value,
                TEXT,
                MONO_SIZE,
                mono=True,
                max_width=content_width,
                native_line=True,
            )
            draw_list.pop_clip_rect()
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + w, field_hi.y),
            rounding=CONTROL_RADIUS * s,
            item_id=item_id,
        )
        if splitter is not None:
            splitter.merge(draw_list)
        imgui.pop_style_var()
        imgui.pop_font()
        self.hits[key] = (
            field_lo.x,
            field_lo.y,
            field_hi.x - field_lo.x,
            field_hi.y - field_lo.y,
        )
        self.hits[key + "-frame"] = (x, y, w, field_hi.y - y)
        if field_hovered and truncated:
            self.tooltip(str(float(current)))
        if axis and badge_hovered:
            self.tooltip(
                self.translate("Read only")
                if disabled
                else f"{self.translate('Reset')}: {reset_value:g}"
            )
        return (True, reset_value) if reset else (changed, edited if changed else value)

    def numeric_minimum(self, *, axis=True, format=None, values=()):
        """Reserve the sign, common precision, padding and optional axis badge."""
        s = self.s
        samples = (
            tuple(format % value for value in (0.123456789, 123.456789, -0.123456789, *values))
            if format
            else ("0.0041472", "0.00288568", "-0.000123")
        )
        font_key = (id(self.mono), s, samples)
        measured = getattr(self, "_numeric_minimum", None)
        if measured is None or measured[0] != font_key:
            imgui.push_font(self.mono, MONO_SIZE * s)
            text_width = max(imgui.calc_text_size(value).x for value in samples)
            imgui.pop_font()
            measured = self._numeric_minimum = (font_key, text_width + 16 * s)
        return measured[1] + (18 * s if axis else 0)

    def numeric_layout(self, width, count, *, axis=True, column_options=None, minimum=None):
        """Keep common precision readable before placing equal-width numeric fields."""
        if not count:
            return NumericGroupLayout((), 0, 0)
        s = self.s
        minimum = self.numeric_minimum(axis=axis) if minimum is None else minimum
        gap = DESIGN.numeric_gap * s
        field_height = DESIGN.control_height * s
        row_stride = (DESIGN.control_height + DESIGN.row_gap) * s
        if column_options is None:
            column_options = (count, 2, 1) if count == 4 else (count, 1)
        column_options = tuple(min(count, columns) for columns in column_options)
        columns = next(
            (
                columns
                for columns in column_options
                if width >= columns * minimum + (columns - 1) * gap
            ),
            1,
        )
        field_width = (width - (columns - 1) * gap) / columns
        fields = tuple(
            (
                (index % columns) * (field_width + gap),
                (index // columns) * row_stride,
                field_width,
                field_height,
            )
            for index in range(count)
        )
        height = fields[-1][1] + field_height
        return NumericGroupLayout(fields, columns, height)

    def form_layout(
        self,
        width,
        count=1,
        *,
        axis=False,
        minimum=0,
        label_above=None,
        label_width=None,
        field_minimum=None,
    ):
        """Give controls their minimum width before reserving a label column."""
        s = self.s
        if label_width is None:
            label_width = min(FORM_LABEL_COLUMN_WIDTH * s, width * 0.28)
        field_minimum = self.numeric_minimum(axis=axis) if field_minimum is None else field_minimum
        required = count * max(minimum, field_minimum) + (count - 1) * DESIGN.numeric_gap * s
        label_width = min(label_width, max(0, width - required - FORM_LABEL_COLUMN_GAP * s))
        above = label_width < FORM_LABEL_COLUMN_MINIMUM * s if label_above is None else label_above
        heading_width = width if above else label_width
        field_x, field_y = (0, 24 * s) if above else (heading_width + FORM_LABEL_COLUMN_GAP * s, 0)
        layout = self.numeric_layout(width - field_x, count, axis=axis, minimum=field_minimum)
        return replace(
            layout,
            fields=tuple((x + field_x, y + field_y, w, h) for x, y, w, h in layout.fields),
            height=layout.height + field_y,
            label_above=above,
            label_width=heading_width,
        )

    def vector_layout(
        self,
        width,
        *,
        label_above=None,
        label_width=None,
        format=None,
        minimum=None,
        values=(),
        key=None,
    ):
        """Share the control-first label column and XYZ grid across each group."""
        cache = self.__dict__.setdefault("_vector_layouts", {})
        geometry = (width, self.s, id(self.mono), label_above, label_width, format)
        previous = cache.get(key) if key is not None else None
        # The active drag owns stable field rectangles until its gesture finishes.
        if previous is not None and previous[0] == geometry and imgui.is_any_item_active():
            return previous[1]
        layout = self.form_layout(
            width,
            3,
            axis=True,
            label_above=label_above,
            label_width=label_width,
            field_minimum=self.numeric_minimum(format=format, values=values)
            if minimum is None
            else minimum,
        )
        if key is not None:
            if len(cache) >= 128:
                cache.clear()
            cache[key] = (geometry, layout)
        return layout

    def vector(
        self,
        key,
        x,
        y,
        w,
        label,
        values,
        unit="",
        *,
        speed=0.01,
        lo=0.0,
        hi=0.0,
        format="%.3f",
        reset_values=(0.0, 0.0, 0.0),
        layout=None,
    ):
        s = self.s
        heading = self.translate(label) + (f" ({unit})" if unit else "")
        layout = (
            self.vector_layout(w, format=format, values=values, key=key)
            if layout is None
            else layout
        )
        self.text(
            x,
            y + (3 if layout.label_above else DESIGN.control_padding_y(BODY_SIZE)) * s,
            heading,
            MUTED,
            max_width=layout.label_width,
        )
        self.hits[key + "-label"] = (x, y, layout.label_width, 24 * s)
        if (
            self.measure(heading) > layout.label_width
            and imgui.is_window_hovered()
            and imgui.is_mouse_hovering_rect((x, y), (x + layout.label_width, y + 24 * s))
        ):
            self.tooltip(heading, translate=False)
        changed = False
        out = list(values)
        for n, a in enumerate("XYZ"):
            field_x, field_y, field_width, _ = layout.fields[n]
            edit, out[n] = self.input(
                key + a,
                x + field_x,
                y + field_y,
                field_width,
                out[n],
                axis=a,
                speed=speed,
                lo=lo,
                hi=hi,
                format=format,
                reset_value=reset_values[n],
            )
            changed |= edit
        states = [self.numeric_edits[key + axis] for axis in "XYZ"]
        self.numeric_edits[key] = NumericEditState(
            activated=any(state.activated for state in states),
            active=any(state.active for state in states),
            deactivated=any(state.deactivated for state in states),
            committed=any(state.committed for state in states),
            canceled=any(state.canceled for state in states),
            reset=any(state.reset for state in states),
        )
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
            self.role_icon(
                "down" if open else "right",
                x + w - 16 * s,
                y + 13 * s,
                12 * s,
                12 * s,
                DIM,
                role="status",
            )
        self.draw.push_clip_rect((x, y + 4 * s), (x + max(0, w - 22 * s), y + 34 * s), True)
        self.text(
            x + 2 * s,
            y + (4 + (30 - BODY_SIZE) / 2) * s,
            label,
            MUTED,
            BODY_SIZE,
            bold=True,
            max_width=max(0, w - 24 * s),
        )
        self.draw.pop_clip_rect()
        if item_id is not None:
            draw_focus_frame(
                imgui.ImVec2(x, y + 4 * s),
                imgui.ImVec2(x + w, y + 34 * s),
                rounding=CONTROL_RADIUS * s,
                item_id=item_id,
            )
        return hit

    def rail(self, key, x, y, w, value, bounds, **options):
        """One native slider, numeric entry and optional attached unit control."""
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_font(self.mono, MONO_SIZE * s)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (DESIGN.frame_padding_x * s, DESIGN.control_padding_y(MONO_SIZE) * s),
        )
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (DESIGN.row_gap * s, DESIGN.row_gap * s))
        for part in ("track", "value", "unit"):
            self.hits.pop(key + "-" + part, None)
        item_id = key if key.startswith("##") else "##" + key
        imgui.begin_group()
        imgui.set_next_item_width(max(s, w))
        try:
            edit = value_rail(
                self.value_context,
                item_id,
                value,
                bounds,
                hit_bounds=self.hits,
                unit_font=self.body,
                **options,
            )
        finally:
            imgui.end_group()
            imgui.pop_style_var(2)
            imgui.pop_font()
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        if item_id != key:
            for part in ("track", "value", "unit"):
                bounds = self.hits.get(item_id + "-" + part)
                if bounds is not None:
                    self.hits[key + "-" + part] = bounds
        self.hits[key] = (a.x, a.y, b.x - a.x, b.y - a.y)
        return edit

    def slider(self, key, x, y, w, value, lo, hi, format="%.2f"):
        edit = self.rail(
            key,
            x,
            y,
            w,
            value,
            (lo, hi) if lo < hi else None,
            initial=None,
            fmt=format,
            show_reset=False,
        )
        return edit.changed, edit.value

    def switch(self, key, x, y, on):
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(
            key, (SWITCH_WIDTH * s, SWITCH_HEIGHT * s), imgui.ButtonFlags_.enable_nav
        )
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
        hovered, held = imgui.is_item_hovered(), imgui.is_item_active()
        tint = (
            (105, 128, 93, 255)
            if on and hovered
            else (88, 108, 78, 255)
            if on
            else ((70, 78, 86, 255) if hovered else (55, 62, 68, 255))
        )
        self.rect(x + 2 * s, y + 5 * s, 32 * s, 18 * s, tint, 9)
        self.draw.add_circle_filled(
            (x + (25 if on else 11) * s, y + 14 * s),
            (6 if held else 7) * s,
            color(SAGE if on else MUTED),
            num_segments=32,
        )
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + SWITCH_WIDTH * s, y + SWITCH_HEIGHT * s),
            rounding=9 * s,
            item_id=item_id,
        )
        self.hits[key] = (x, y, SWITCH_WIDTH * s, SWITCH_HEIGHT * s)
        return not on if hit else on
