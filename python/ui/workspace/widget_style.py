"""Native workspace palette, font-atlas setup and ImGui style installation."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.text.sources import FontSource
from mojive.ui.theme import THEME

from .design import BODY_SIZE, CONTROL_RADIUS, DESIGN, MONO_SIZE
from .fonts import font_em_ratio, font_vertical_offset, resolve_cjk_faces, resolve_fonts

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


def font_config(em_ratio, face_index=0, *, embolden=False, vertical_offset=0):
    config = imgui.ImFontConfig()
    config.extra_size_scale = em_ratio
    config.font_loader_flags = FONT_NO_HINTING | (FONT_BOLD if embolden else 0)
    config.rasterizer_density = FONT_SAMPLING
    # FreeType encodes one-based variable-font named instances in face-index bits 16–30.
    config.font_no = face_index
    config.glyph_offset = (0, vertical_offset)
    return config


def install_fonts(window, *, portable=False):
    """Replace the atlas once and return the sources and body, heading, numeric fonts."""
    scale = window.style_scale
    io = imgui.get_io()
    io.fonts.clear()
    report = window.font_report
    cjk = (
        FontSource(report.cjk_path, report.cjk_index, getattr(report, "cjk", "CJK"))
        if report.cjk_path
        else None
    )
    sources = resolve_fonts(portable=portable, cjk=cjk)
    regular, heading = resolve_cjk_faces(cjk)
    fonts = []
    for (source, em_ratio), size, fallback in zip(
        sources,
        (BODY_SIZE, BODY_SIZE, MONO_SIZE),
        (regular, heading, regular),
        strict=True,
    ):
        config = font_config(em_ratio, source.index, embolden=getattr(source, "embolden", False))
        config.glyph_offset = (0, font_vertical_offset(source, size * scale, config))
        fonts.append(
            io.fonts.add_font_from_file_ttf(
                source.path,
                size * scale,
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
                    size * scale,
                    merged,
                    ascent=fonts[-1]
                    .get_font_baked(size * scale, io.display_framebuffer_scale.x)
                    .ascent,
                ),
            )
            merged.merge_mode = True
            io.fonts.add_font_from_file_ttf(fallback.path, size * scale, merged)
    body, bold, mono = fonts
    io.font_default = body
    return sources, (body, bold, mono)


def install_style(scale):
    """Apply the workspace's shared palette and geometry to the current context."""
    st = imgui.get_style()
    st.font_size_base = BODY_SIZE * scale
    st.window_rounding = DESIGN.container_radius * scale
    st.frame_rounding = CONTROL_RADIUS * scale
    st.tab_rounding = CONTROL_RADIUS * scale
    st.popup_rounding = 8 * scale
    st.grab_rounding = DESIGN.grab_radius * scale
    st.child_rounding = DESIGN.child_radius * scale
    st.window_padding = (DESIGN.panel_padding_x * scale, DESIGN.panel_padding_y * scale)
    st.frame_padding = (DESIGN.frame_padding_x * scale, DESIGN.frame_padding_y * scale)
    st.item_spacing = (DESIGN.item_spacing_x * scale, DESIGN.row_gap * scale)
    st.item_inner_spacing = (DESIGN.item_spacing_x * scale, DESIGN.numeric_gap * scale)
    st.window_min_size = (230 * scale, 120 * scale)
    st.window_border_size = 1
    st.frame_border_size = 0
    st.popup_border_size = 1
    st.docking_separator_size = 2 * scale
    st.tab_bar_border_size = 0
    st.tab_bar_overline_size = 0
    st.scrollbar_size = 7 * scale
    st.scrollbar_rounding = 4 * scale
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
