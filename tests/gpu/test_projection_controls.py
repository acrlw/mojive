"""Check composite centering against the native font renderer, including HiDPI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")


@pytest.mark.parametrize("scale", (1.0, 1.25, 1.5, 2.0))
@pytest.mark.parametrize("labels", (("persp", "ortho"), ("透视", "正交")))
def test_projection_pair_is_centered_without_text_origin_truncation(monkeypatch, scale, labels):
    from imgui_bundle import imgui

    from mojive.ui.icons import drawing
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.panels import segmented_control
    from mojive.ui.window import Window, WindowConfig
    from mojive.ui.workspace.widget_icons import COMPILED_ALIASES, GLYPHS, ICON_GRID

    window = Window(
        WindowConfig(
            title="Projection centering",
            width=600,
            height=140,
            ui_scale=scale,
            show_on_start=False,
            docking=False,
            ini_path="",
            vsync=False,
        )
    )
    original_text = ImguiDraw2D.text
    original_icon = drawing.draw_icon
    records, icon_records = [], []

    def record_icon(draw, center, size, name, color, **kwargs):
        start = len(draw._dl.vtx_buffer)
        original_icon(draw, center, size, name, color, **kwargs)
        vertices = list(draw._dl.vtx_buffer)[start:]
        # Measure the submitted contour, excluding transparent AA fringe vertices.
        alpha = max(vertex.col >> 24 for vertex in vertices)
        opaque = [vertex for vertex in vertices if vertex.col >> 24 == alpha]
        icon_records.append(
            (
                name,
                size,
                min(vertex.pos.x for vertex in opaque),
                max(vertex.pos.x for vertex in opaque),
            )
        )

    def record_text(draw, pos, color, text, **kwargs):
        start = len(draw._dl.vtx_buffer)
        flags = draw._dl.flags
        original_text(draw, pos, color, text, **kwargs)
        assert draw._dl.flags == flags
        if text in labels:
            vertices = list(draw._dl.vtx_buffer)[start:]
            records.append(
                (
                    imgui.get_item_rect_min(),
                    imgui.get_item_rect_max(),
                    min(v.pos.x for v in vertices),
                    min(v.pos.y for v in vertices),
                    max(v.pos.x for v in vertices),
                    draw.text_ink_bounds(text),
                    draw.text_ink_bounds("x" if text.isascii() else "田"),
                    pos,
                )
            )

    monkeypatch.setattr(ImguiDraw2D, "text", record_text)
    monkeypatch.setattr(drawing, "draw_icon", record_icon)
    try:
        for _ in range(3):
            records.clear()
            icon_records.clear()
            window.begin_frame()
            imgui.set_next_window_pos((0.0, 0.0))
            imgui.set_next_window_size((600.0, 140.0))
            imgui.begin("##projection", None, imgui.WindowFlags_.no_decoration)
            segmented_control("projection", labels, 0, width=560.0, icons=("persp", "ortho"))
            imgui.end()
            window.end_frame()

        assert len(records) == 2
        assert len(icon_records) == 2
        text_origins = []
        for index, (lo, hi, ink_left, ink_top, ink_right, ink, body, pos) in enumerate(records):
            # Native AddText must honor the same fractional origin as layout.
            assert ink_left == pytest.approx(pos[0] + ink[0], abs=1e-4)
            assert ink_top == pytest.approx(pos[1] + ink[1], abs=1e-4)
            body_center = ink_top - ink[1] + (body[1] + body[3]) * 0.5
            assert body_center == pytest.approx((lo.y + hi.y) * 0.5, abs=1e-4)
            glyph_scale = max(0.65, (hi.y - lo.y) / 24.0)
            name = ("panel-perspective", "panel-orthographic")[index]
            actual_name, size, icon_left, icon_right = icon_records[index]
            assert actual_name == name
            assert size == pytest.approx(16.0 * glyph_scale, abs=1e-4)
            bounds = GLYPHS[COMPILED_ALIASES[name]]["bounds"]
            glyph_width = (bounds[2] - bounds[0]) * 16.0 * glyph_scale / ICON_GRID
            assert icon_right - icon_left == pytest.approx(glyph_width, abs=1e-4)
            assert ink_left - icon_right == pytest.approx(7.0 * glyph_scale, abs=1e-4)
            assert (icon_left + ink_right) * 0.5 == pytest.approx((lo.x + hi.x) * 0.5, abs=1e-4)
            text_origins.append(pos[1])
        assert text_origins[0] == pytest.approx(text_origins[1])
    finally:
        window.close()
