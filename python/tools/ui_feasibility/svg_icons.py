"""Interactive comparison of production painters and independently loaded SVG files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import ParseError

from imgui_bundle import imgui

from mojive.tools.svg_icons import (
    DEFAULT_DIRECTORY,
    ICON_NAMES,
    SvgIcon,
    draw_svg_icon,
    export_icons,
    load_icons,
)
from mojive.ui.icons import draw_icon
from mojive.ui.imgui_draw import ImguiDraw2D

from .fixtures import CONCEPT_THEME


@dataclass
class SvgStudy:
    directory: Path = DEFAULT_DIRECTORY
    glyph: str = "tool-rotate"
    icons: dict[str, SvgIcon] = field(default_factory=dict)
    initialized: bool = False
    error: str = ""
    light: bool = False
    guides: bool = False
    playing: bool = False

    def reload(self):
        try:
            icons = load_icons(self.directory)
        except (OSError, ValueError, KeyError, ParseError) as error:
            self.error = str(error)
        else:
            self.icons = icons
            self.error = ""
        self.initialized = True

    def export(self):
        try:
            export_icons(self.directory)
        except (OSError, ValueError) as error:
            self.error = str(error)
        else:
            self.reload()


def _note(text):
    imgui.push_style_color(imgui.Col_.text, CONCEPT_THEME.text_disabled)
    imgui.text_wrapped(text)
    imgui.pop_style_color()


def _specimen(draw, center, size, name, study, *, svg, color, accent=None):
    if svg:
        draw_svg_icon(draw, center, size, study.icons[name], color, accent_color=accent)
    else:
        draw_icon(draw, center, size, name, color, accent_color=accent)
    if study.guides:
        half = size / 2
        draw.rect(
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] + half),
            (0.78, 0.59, 0.31, 0.75),
            width=0.6,
        )


def draw_svg_page(study: SvgStudy, scale: float):
    if not study.initialized:
        study.reload()
    imgui.text("SVG asset comparison")
    imgui.text_wrapped(
        "Production painter -> outlined SVG file -> cached native mesh -> ImGui. "
        "Both columns use the same window, theme colors and logical sizes."
    )
    imgui.spacing()
    imgui.set_next_item_width(215 * scale)
    _, index = imgui.combo("Glyph", ICON_NAMES.index(study.glyph), list(ICON_NAMES))
    study.glyph = ICON_NAMES[index]
    _, study.light = imgui.checkbox("Light specimens", study.light)
    imgui.same_line()
    _, study.guides = imgui.checkbox("Canvas guides", study.guides)
    if imgui.button("Reload SVG files"):
        study.reload()
    imgui.same_line()
    if imgui.button("Export production assets"):
        study.export()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Replace the exported files with the current production geometry.")
    imgui.text_wrapped(str(study.directory.resolve()))
    if study.error:
        imgui.text_wrapped(f"SVG load failed: {study.error}")
        imgui.text_wrapped(
            "Export production assets to create the set. Reload preserves the "
            "last complete set if a file contains unsupported SVG syntax."
        )
    if not study.icons:
        return

    fg = (0.14, 0.17, 0.19, 1.0) if study.light else CONCEPT_THEME.text
    bg = (0.94, 0.95, 0.96, 1.0) if study.light else CONCEPT_THEME.bg_child
    accent = (0.29, 0.44, 0.23, 1.0) if study.light else CONCEPT_THEME.primary
    draw = ImguiDraw2D()
    imgui.spacing()
    if imgui.begin_table("svg-comparison", 3, imgui.TableFlags_.borders_inner_v.value):
        imgui.table_setup_column(
            "Logical size", imgui.TableColumnFlags_.width_fixed.value, 95 * scale
        )
        imgui.table_setup_column("Code / production")
        imgui.table_setup_column("SVG / loaded file")
        imgui.table_headers_row()
        for size in (14, 24, 56, 112):
            height = (size + 24) * scale
            imgui.table_next_row(0, height)
            imgui.table_next_column()
            imgui.text(f"{size} pt")
            for svg in (False, True):
                imgui.table_next_column()
                lo = imgui.get_cursor_screen_pos()
                width = imgui.get_content_region_avail().x
                draw.rect_filled(
                    (lo.x, lo.y), (lo.x + width, lo.y + height - 4 * scale), bg, rounding=4 * scale
                )
                center = (lo.x + width / 2, lo.y + (height - 4 * scale) / 2)
                _specimen(
                    draw, center, size * scale, study.glyph, study, svg=svg, color=fg, accent=accent
                )
                imgui.dummy((width, height - 4 * scale))
        imgui.end_table()

    imgui.spacing()
    imgui.text("SVG button interaction")
    _note("Click or focus and press Enter; the loaded SVG follows the button state.")
    glyph = "playback-pause" if study.playing else "playback-play"
    if imgui.button("##svg-playback", (34 * scale, 34 * scale)):
        study.playing = not study.playing
        glyph = "playback-pause" if study.playing else "playback-play"
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_svg_icon(
        draw,
        ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2),
        18 * scale,
        study.icons[glyph],
        CONCEPT_THEME.primary if study.playing else CONCEPT_THEME.text,
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip("Pause" if study.playing else "Play")
    imgui.same_line()
    imgui.text("Playing" if study.playing else "Paused")
    imgui.spacing()
    _note("Assets use M/L/Z fills and native strokes/circles; transforms are baked.")
    _note("Parsing occurs on load/reload. Meshes are cached by SVG contents and size.")
