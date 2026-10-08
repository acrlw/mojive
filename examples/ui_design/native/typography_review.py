"""Actual-size type specimens through the reference's real ImGui font configuration."""

from __future__ import annotations

import json

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from .widgets import DIM, MUTED, TEXT, UI


def capture_typography(window, destination, *, portable_fonts=False):
    destination.mkdir(parents=True, exist_ok=True)
    ui = UI(window, portable_fonts=portable_fonts)
    s = ui.s
    sample = "World coordinates · 1 selected · Pose keys · 0123456789"
    legacy_config = imgui.ImFontConfig()
    source, em_ratio = ui.font_sources[0]
    legacy_config.extra_size_scale = em_ratio
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
            ui.text(24 * s, y, str(size) + " px", DIM, 11)
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
        ui.text(24 * s, 681 * s, "Paused     W E R  Tools     B  Frame     T  Timeline", DIM, 11)
        ui.text(680 * s, 681 * s, "World coordinates     Z up     Local scene preview", DIM, 11)
        imgui.end()
        pixels = window.end_frame(readback=True)
    Image.fromarray(np.asarray(pixels)[::-1], "RGB").save(destination / "specimen.png")
    io = imgui.get_io()
    report = {
        "loader": io.fonts.font_loader_name,
        "logical_viewport": list(io.display_size),
        "framebuffer_density": list(io.display_framebuffer_scale),
        "scale": s,
        "sampling": 2,
        "hinting": "NoHinting",
        "glyphs": glyphs,
        "method": "Real ImGui framebuffer; visual review at original resolution is required.",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    print("Typography specimen and glyph checks passed:", destination)
