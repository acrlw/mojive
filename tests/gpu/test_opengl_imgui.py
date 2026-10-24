"""Native ImGui renderer resources remain owned by their OpenGL window."""

import ctypes
import subprocess
import sys
from contextlib import ExitStack

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.ui import window as window_module
from mojive.ui.window import Window, WindowConfig

pytestmark = pytest.mark.gpu


def test_production_window_does_not_load_pyopengl_wrappers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mojive.ui.window import Window, WindowConfig; "
            "w = Window(WindowConfig(show_on_start=False, vsync=False, width=101, height=79)); "
            "w.begin_frame(); a = w.end_frame(readback=True); "
            "assert a.shape == (*w.size_pixels[::-1], 3); "
            "assert w.gl_version and w.renderer_name; "
            "w.close(); assert 'OpenGL.GL' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_current(window):
    assert imgui.get_current_context() == window._imgui_context
    assert ctypes.cast(window_module.glfw.get_current_context(), ctypes.c_void_p).value == (
        ctypes.cast(window.gl_context, ctypes.c_void_p).value
    )


def _text_frame(window, text, *, new_glyphs=False, before_submit=None):
    window.begin_frame()
    _assert_current(window)
    font = imgui.get_font()
    baked = font.get_font_baked(imgui.get_font_size())
    if new_glyphs:
        assert all(font.is_glyph_in_font(ord(char)) for char in text)
        assert all(not baked.is_glyph_loaded(ord(char)) for char in text)
    draw_list = imgui.get_background_draw_list()
    draw_list.add_text((20, 20), 0xFFFFFFFF, text)
    clips = [tuple(command.clip_rect) for command in draw_list.cmd_buffer if command.elem_count]
    if new_glyphs:
        assert all(baked.is_glyph_loaded(ord(char)) for char in text)
        assert any(
            texture.status
            in (imgui.ImTextureStatus.want_create, imgui.ImTextureStatus.want_updates)
            for texture in imgui.get_io().fonts.tex_list
        ), "First-use glyphs must request a texture upload after the peer renderer shuts down"
    if before_submit is not None:
        before_submit()
    pixels = window.end_frame(readback=True)[::-1].copy()
    _assert_current(window)
    # Post-submit design audits use logical clips, including on Retina displays.
    assert clips == [
        tuple(command.clip_rect) for command in draw_list.cmd_buffer if command.elem_count
    ]
    texture = imgui.get_io().fonts.tex_data
    assert texture.status == imgui.ImTextureStatus.ok and texture.get_tex_id() != 0
    assert pixels.shape == (*window.size_pixels[::-1], 3)
    np.testing.assert_allclose(
        pixels[-1, -1], np.array(window.config.clear_color[:3]) * 255, atol=1
    )
    # The frame contains only a clear color and text: bright pixels prove the
    # owning context sampled its font atlas, rather than merely clearing.
    assert np.count_nonzero(np.all(pixels > 200, axis=2)) > 20
    return pixels


def test_peer_shutdown_keeps_font_uploads_and_reinitialization_working(require_opengl, monkeypatch):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")

    def create(title, color):
        return Window(
            WindowConfig(
                title=title,
                width=360,
                height=160,
                docking=False,
                ini_path="",
                show_on_start=False,
                vsync=False,
                clear_color=(*color, 1.0),
            )
        )

    with ExitStack() as windows:
        first = windows.enter_context(create("ImGui context A", (0.2, 0.0, 0.0)))
        second = windows.enter_context(create("ImGui context B", (0.0, 0.0, 0.2)))
        assert first._imgui_context != second._imgui_context
        assert second.font_report.cjk_path, "The bundled CJK font is required by this test"
        for _ in range(3):
            baseline = _text_frame(first, "Context A")
            _text_frame(second, "Context B")

        def render_peer_before_submit():
            # Poison A's back buffer so a misplaced clear cannot pass because
            # the previous buffer happened to have the expected background.
            first._gl_surface.clear(*first.size_pixels, (0.0, 1.0, 0.0, 1.0))
            _text_frame(second, "Context B")

        interleaved = _text_frame(first, "Context A", before_submit=render_peer_before_submit)
        np.testing.assert_array_equal(interleaved, baseline)
        before = _text_frame(second, "Context B")

        # B is current when A closes. The compiled backend's loader is global,
        # but B's next frame must recreate it without losing B's resources.
        first.close()
        after = _text_frame(second, "\u5173\u8282", new_glyphs=True)
        assert not np.array_equal(before, after)
        np.testing.assert_array_equal(_text_frame(second, "\u5173\u8282"), after)
        second.close()

        # Reinitialize after both backend instances and GLFW windows shut down.
        third = windows.enter_context(create("ImGui context C", (0.0, 0.2, 0.0)))
        _text_frame(third, "\u7269\u7406", new_glyphs=True)
