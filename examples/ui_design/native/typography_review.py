"""Actual-size type specimens through the reference's real ImGui font configuration."""

from __future__ import annotations

import json

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.ui.workspace.widgets import DIM, MUTED, TEXT, UI


def _glyph_metrics(face, size, characters, scale):
    """Record actual advances and visible bounds in logical points."""
    baked = face.get_font_baked(size * scale)
    result = {"size": baked.size, "glyphs": {}}
    for char in characters:
        glyph = baked.find_glyph_no_fallback(ord(char))
        if glyph is not None:
            result["glyphs"][char] = {
                "advance_pt": glyph.advance_x / scale,
                "ink_width_pt": (glyph.x1 - glyph.x0) / scale,
                "ink_height_pt": (glyph.y1 - glyph.y0) / scale,
            }
    return result


def capture_typography(window, destination, *, portable_fonts=False):
    destination.mkdir(parents=True, exist_ok=True)
    ui = UI(window, portable_fonts=portable_fonts)
    s = ui.s
    sample = "World coordinates · 1 selected · Pose keys · 0123456789"
    legacy_config = imgui.ImFontConfig()
    source, em_ratio = ui.font_sources[0]
    legacy_config.extra_size_scale = em_ratio
    legacy_config.font_no = source.index
    legacy = imgui.get_io().fonts.add_font_from_file_ttf(source.path, 12 * s, legacy_config)
    glyphs = {}
    for _ in range(10):
        window.begin_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.begin("Typography", None, imgui.WindowFlags_.no_decoration)
        ui.text(24 * s, 24 * s, "Instrument 08 · native typography", size=17, bold=True)
        ui.text(
            24 * s,
            54 * s,
            "Actual logical sizes · "
            + " / ".join(source.label for source, _ratio in ui.font_sources),
            DIM,
            11,
        )
        ui.text(24 * s, 97 * s, "DrawList labels", MUTED)
        ui.text(680 * s, 97 * s, "Native ImGui text controls", MUTED)
        for n, size in enumerate((10, 11, 12, 14, 16, 18)):
            y = (135 + n * 63) * s
            ui.text(24 * s, y, str(size) + " pt", DIM, 11)
            ui.text(24 * s, y + 22 * s, sample, TEXT, size)
            imgui.push_font(ui.body, size * s)
            imgui.set_cursor_screen_pos((680 * s, y + 22 * s))
            imgui.text_unformatted(sample)
            baked = ui.body.get_font_baked(size * s)
            glyphs[str(size)] = {"density": baked.rasterizer_density, "size": baked.size}
            for ch in "Hxe@0123456789":
                g = baked.find_glyph(ord(ch))
                assert g.is_visible() and g.u1 > g.u0 and g.v1 > g.v0, (size, ch)
            imgui.pop_font()
        ui.line(24 * s, 530 * s, 1350 * s)
        ui.text(24 * s, 553 * s, "10 px · previous sampling", DIM, 11)
        ui.draw.add_text(
            legacy, 10 * s, (24 * s, 580 * s), imgui.get_color_u32((0.9, 0.91, 0.92, 1)), sample
        )
        ui.text(680 * s, 553 * s, "10 px · corrected sampling, same font and size", DIM, 11)
        ui.text(680 * s, 580 * s, sample, size=10)
        ui.text(24 * s, 629 * s, "TRANSFORM  /  MATERIAL LIBRARY", MUTED, 10, bold=True)
        ui.text(680 * s, 629 * s, "X  -0.25    Y  0.125    Z  1.000", MUTED, 11, mono=True)
        ui.text(24 * s, 681 * s, "Paused     G R  Tools     B  Frame     T  Timeline", DIM, 11)
        ui.text(680 * s, 681 * s, "World coordinates     Z up     Local scene preview", DIM, 11)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(destination / "specimen.png")
    bilingual = "Camera 相机 · Projection 投影 · Physics 物理 · Aa 0123456789"
    bilingual_glyphs = {}
    for _ in range(8):
        window.begin_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(imgui.get_io().display_size)
        imgui.begin("Bilingual typography", None, imgui.WindowFlags_.no_decoration)
        ui.text(24 * s, 24 * s, "Latin / Simplified Chinese", size=18, bold=True)
        ui.text(24 * s, 57 * s, "Matched family, weight and nominal em size", MUTED)
        for index, size in enumerate((13, 14, 16, 18)):
            y = (108 + index * 140) * s
            ui.text(24 * s, y, f"{size} pt · Body", DIM)
            ui.text(24 * s, y + 28 * s, bilingual, size=size)
            ui.text(24 * s, y + 67 * s, f"{size} pt · Heading", DIM)
            ui.text(24 * s, y + 95 * s, bilingual, size=size, bold=True)
            for role, face in (("body", ui.body), ("heading", ui.bold)):
                baked = face.get_font_baked(size * s)
                if window.font_report.cjk_path:
                    for char in "相机投影物理Aa09":
                        glyph = baked.find_glyph_no_fallback(ord(char))
                        assert glyph is not None and glyph.is_visible(), (role, size, char)
                bilingual_glyphs[f"{role}-{size}"] = _glyph_metrics(face, size, "相机H09", s)
        ui.text(24 * s, 712 * s, "Numeric · 0.02000 m · 200.000 m · 42.00 deg", size=13, mono=True)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(destination / "bilingual.png")
    io = imgui.get_io()
    report = {
        "loader": io.fonts.font_loader_name,
        "logical_viewport": list(io.display_size),
        "framebuffer_density": list(io.display_framebuffer_scale),
        "scale": s,
        "sampling": 2,
        "hinting": "NoHinting",
        "body_size_pt": 13,
        "numeric_size_pt": 13,
        "glyphs": glyphs,
        "bilingual_glyphs": bilingual_glyphs,
        "sources": [
            {
                "path": source.path,
                "face": source.index,
                "label": source.label,
                "em_ratio": ratio,
            }
            for source, ratio in ui.font_sources
        ],
        "method": "Real ImGui framebuffer; visual review at original resolution is required.",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    print("Typography specimen and glyph checks passed:", destination)
