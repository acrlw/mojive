"""Check composite centering against the native font renderer, including HiDPI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")


@pytest.mark.parametrize("scale", (1.0, 1.25, 1.5, 2.0))
@pytest.mark.parametrize("labels", (("persp", "ortho"), ("透视", "正交")))
def test_projection_pair_is_centered_without_text_origin_truncation(monkeypatch, scale, labels):
    from imgui_bundle import imgui

    from mojive.ui.draw2d import ImguiDraw2D
    from mojive.ui.panels import segmented_control
    from mojive.ui.window import Window, WindowConfig

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
    records = []

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
    try:
        for _ in range(3):
            records.clear()
            window.begin_frame()
            imgui.set_next_window_pos((0.0, 0.0))
            imgui.set_next_window_size((600.0, 140.0))
            imgui.begin("##projection", None, imgui.WindowFlags_.no_decoration)
            segmented_control("projection", labels, 0, width=560.0, icons=("persp", "ortho"))
            imgui.end()
            window.end_frame()

        assert len(records) == 2
        text_origins = []
        for lo, hi, ink_left, ink_top, ink_right, ink, body, pos in records:
            # Native AddText must honor the same fractional origin as layout.
            assert ink_left == pytest.approx(pos[0] + ink[0], abs=1e-4)
            assert ink_top == pytest.approx(pos[1] + ink[1], abs=1e-4)
            body_center = ink_top - ink[1] + (body[1] + body[3]) * 0.5
            assert body_center == pytest.approx((lo.y + hi.y) * 0.5, abs=1e-4)
            glyph_scale = max(0.65, (hi.y - lo.y) / 24.0)
            glyph_width = 10.4 * glyph_scale + max(1.0, 1.25 * glyph_scale)
            pair_left = ink_left - 7.0 * glyph_scale - glyph_width
            assert (pair_left + ink_right) * 0.5 == pytest.approx((lo.x + hi.x) * 0.5, abs=1e-4)
            text_origins.append(pos[1])
        assert text_origins[0] == pytest.approx(text_origins[1])
    finally:
        window.close()
