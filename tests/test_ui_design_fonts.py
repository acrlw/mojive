"""Native typography uses nominal em sizes with matching Latin and CJK faces."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.ui_design.native import fonts
from examples.ui_design.native.typography_review import _glyph_metrics
from examples.ui_design.native.widgets import UI, font_config
from fontTools.ttLib import TTFont
from imgui_bundle import imgui


def _cjk_source():
    try:
        sources = fonts.resolve_font_sources(allow_download=False)
    except FileNotFoundError:
        pytest.skip("CJK font is unavailable")
    if len(sources) < 2:
        pytest.skip("CJK font is unavailable")
    regular, _heading = fonts.resolve_cjk_faces(sources[1])
    if not Path(regular.path).name.startswith(("NotoSansCJK-", "NotoSansSC-", "SourceHanSansSC-")):
        pytest.skip("The installed CJK family has no declared nominal em conversion")
    return sources[1]


@pytest.mark.parametrize("portable", (False, True))
def test_proportional_face_em_ratio_matches_installed_font_metrics(portable):
    cjk = None if portable else _cjk_source()
    for source, ratio in fonts.resolve_fonts(cjk=cjk, portable=portable)[:2]:
        with TTFont(source.path, fontNumber=source.index & 0xFFFF) as face:
            expected = (face["hhea"].ascent - face["hhea"].descent) / face["head"].unitsPerEm
        assert ratio == pytest.approx(expected)


def test_numeric_font_keeps_its_compact_metrics():
    _source, ratio = fonts.resolve_fonts(portable=True)[2]
    assert ratio == 1.0


def test_existing_sf_named_instances_keep_their_single_conversion(monkeypatch):
    monkeypatch.setattr(
        fonts.Path, "is_file", lambda path: str(path).startswith("/System/Library/Fonts/")
    )
    resolved = fonts.resolve_fonts()
    assert [source.index for source, _ratio in resolved] == [4 << 16, 6 << 16, 2 << 16]
    assert [ratio for _source, ratio in resolved] == [2412 / 2048] * 3


@pytest.fixture(params=(1.0, 2.0, 2.25))
def typography(request):
    source = _cjk_source()
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    density = request.node.callspec.params.get("framebuffer", 1.0)
    io.display_framebuffer_scale = (density, density)
    window = SimpleNamespace(
        style_scale=request.param,
        font_report=SimpleNamespace(cjk_path=source.path, cjk_index=source.index),
    )
    ui = UI(window, portable_fonts=request.node.callspec.params.get("portable", False))

    def frame(paint):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Typography", flags=imgui.WindowFlags_.no_title_bar)
        result = paint()
        imgui.end()
        imgui.render()
        return result

    frame(lambda: None)
    try:
        yield ui, frame
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("role", ("body", "bold"))
def test_body_and_heading_cjk_advance_matches_declared_em_size(typography, portable, role):
    ui, frame = typography

    def paint():
        face = getattr(ui, role)
        imgui.push_font(face, 14 * ui.s)
        imgui.text_unformatted("Camera 相机 H")
        metrics = _glyph_metrics(face, 14, "相机H", ui.s)
        imgui.pop_font()
        return metrics

    metrics = frame(paint)
    assert metrics["size"] == round(14 * ui.s)
    for char in "相机":
        assert metrics["glyphs"][char]["advance_pt"] == pytest.approx(
            metrics["size"] / ui.s, abs=0.04
        )
        assert metrics["glyphs"][char]["ink_height_pt"] >= 12.5
    assert metrics["glyphs"]["H"]["ink_height_pt"] >= 9


def test_default_measure_matches_real_13pt_bilingual_text(typography):
    ui, frame = typography

    def paint():
        text = "Position 位置"
        measured = ui.measure(text)
        imgui.push_font(ui.body, 13 * ui.s)
        native_width = imgui.calc_text_size(text).x
        imgui.pop_font()
        first = len(ui.draw.vtx_buffer)
        ui.text(40 * ui.s, 50 * ui.s, text)
        vertices = [ui.draw.vtx_buffer[index] for index in range(first, len(ui.draw.vtx_buffer))]
        return measured, native_width, vertices

    measured, native_width, vertices = frame(paint)
    assert measured == native_width
    assert vertices


def test_numeric_digits_keep_the_legacy_13pt_face_geometry(typography):
    ui, frame = typography
    source, ratio = ui.font_sources[2]
    legacy = imgui.get_io().fonts.add_font_from_file_ttf(
        source.path, 13 * ui.s, font_config(ratio, source.index)
    )

    def paint():
        metrics = []
        for face in (ui.mono, legacy):
            imgui.push_font(face, 13 * ui.s)
            imgui.text_unformatted("0123456789")
            metrics.append(_glyph_metrics(face, 13, "0123456789", ui.s))
            imgui.pop_font()
        return metrics

    current, prior = frame(paint)
    assert current == prior


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("role", ("body", "bold", "mono"))
@pytest.mark.parametrize("framebuffer", (1.0, 2.0))
def test_native_face_cap_height_is_centered_in_its_line_box(
    typography, portable, role, framebuffer
):
    ui, frame = typography

    def paint():
        imgui.push_font(getattr(ui, role), 13 * ui.s)
        imgui.text_unformatted("H 位置 0123456789")
        baked = imgui.get_font_baked()
        glyph = baked.find_glyph_no_fallback(ord("H"))
        result = (glyph.y0 + glyph.y1) / 2, imgui.get_font_size() / 2
        imgui.pop_font()
        return result

    cap, line = frame(paint)
    assert cap == pytest.approx(line, abs=0.55)


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("role,text", (("body", "AgH09"), ("mono", "-0.0041472")))
@pytest.mark.parametrize("framebuffer", (1.0, 2.0))
def test_real_input_text_caret_uses_the_visible_font_line_and_character_advances(
    typography, portable, role, text, framebuffer
):
    ui, frame = typography
    io = imgui.get_io()
    io.config_input_text_cursor_blink = False
    cursor_color = (0.91, 0.12, 0.86, 1.0)
    current = text
    focus = True

    def paint():
        nonlocal current, focus
        imgui.push_font(getattr(ui, role), 13 * ui.s)
        imgui.push_style_color(imgui.Col_.input_text_cursor, cursor_color)
        imgui.set_cursor_screen_pos((40 * ui.s, 60 * ui.s))
        imgui.set_next_item_width(280 * ui.s)
        if focus:
            imgui.set_keyboard_focus_here()
            focus = False
        _changed, current = imgui.input_text("##caret-audit", current)
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        baked = imgui.get_font_baked()
        glyph = baked.find_glyph(ord("H"))
        padding = imgui.get_style().frame_padding
        cap_y = low.y + padding.y + (glyph.y0 + glyph.y1) / 2
        expected_x = low.x + padding.x + imgui.calc_text_size(current).x
        expected_y = low.y + padding.y + imgui.get_font_size() / 2 - 0.5
        encoded = imgui.get_color_u32(imgui.Col_.input_text_cursor)
        vertices = [
            (vertex.pos.x, vertex.pos.y) for vertex in ui.draw.vtx_buffer if vertex.col == encoded
        ]
        imgui.pop_style_color()
        imgui.pop_font()
        return low, high, cap_y, expected_x, expected_y, vertices

    frame(paint)
    frame(paint)
    io.add_key_event(imgui.Key.end, True)
    _low, _high, _cap_y, _expected_x, _expected_y, _vertices = frame(paint)
    io.add_key_event(imgui.Key.end, False)
    frame(paint)
    appended = "位置" if role == "body" else "7"
    io.add_input_characters_utf8(appended)
    frame(paint)
    low, high, cap_y, expected_x, expected_y, vertices = frame(paint)
    assert current == text + appended
    assert io.want_text_input and vertices
    left, right = min(point[0] for point in vertices), max(point[0] for point in vertices)
    top, bottom = min(point[1] for point in vertices), max(point[1] for point in vertices)
    assert (left + right) / 2 == pytest.approx(expected_x, abs=1)
    assert (top + bottom) / 2 == pytest.approx(expected_y, abs=1)
    assert (top + bottom) / 2 == pytest.approx(cap_y, abs=1.5)
    assert low.y <= top < bottom <= high.y
