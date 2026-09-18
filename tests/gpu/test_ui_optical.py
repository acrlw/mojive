"""Exercise the optical comparison with both UI renderers and responsive layouts."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.tools.ui_feasibility import ProbeState, optical
from mojive.tools.ui_feasibility.fixtures import _apply_concept_theme
from mojive.tools.ui_feasibility.workspace import _draw_workspace
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("scale", "width", "height"), ((1.0, 1120, 1100), (2.25, 2500, 2400), (1.0, 650, 2000))
)
def test_optical_offsets_are_local_interactive_and_survive_glyph_switch(
    backend_name, monkeypatch, scale, width, height
):
    rectangles, columns, copied = {}, [], []
    native_slider, native_button, draw_column = imgui.slider_float, imgui.button, optical._column
    native_checkbox = imgui.checkbox
    native_combo = imgui.combo

    def record(label):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rectangles[label] = (lo.x, lo.y, hi.x, hi.y)

    def slider(label, *args, **kwargs):
        result = native_slider(label, *args, **kwargs)
        record(label)
        return result

    def button(label, *args, **kwargs):
        result = native_button(label, *args, **kwargs)
        record(label)
        return result

    def checkbox(label, *args, **kwargs):
        result = native_checkbox(label, *args, **kwargs)
        record(label)
        return result

    def combo(label, *args, **kwargs):
        result = native_combo(label, *args, **kwargs)
        record(label)
        return result

    def column(ctx, study, candidate):
        origin = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail().x
        draw_column(ctx, study, candidate)
        # End at the last specimen, excluding spacing that can contain the next row's text ink.
        columns.append((origin.x, origin.y, origin.x + available, imgui.get_item_rect_max().y))

    monkeypatch.setattr(imgui, "slider_float", slider)
    monkeypatch.setattr(imgui, "button", button)
    monkeypatch.setattr(imgui, "checkbox", checkbox)
    monkeypatch.setattr(imgui, "combo", combo)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    monkeypatch.setattr(optical, "_column", column)
    config = WindowConfig(
        width=width,
        height=height,
        ui_scale=scale,
        vsync=False,
        docking=False,
        ini_path="",
        show_on_start=False,
    )
    with create_window(config, backend_name) as window:
        _apply_concept_theme(window.style_scale)
        state = ProbeState(page="Components", renderer=backend_name)
        state.components.tab = 1
        study = state.components.optical

        def frame():
            rectangles.clear()
            columns.clear()
            window.begin_frame()
            _draw_workspace(window, state)
            return window.end_frame(readback=True)[::-1].copy()

        def click(label, fraction=0.5):
            x0, y0, x1, y1 = rectangles[label]
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            imgui.get_io().add_mouse_pos_event(x0 + (x1 - x0) * fraction, (y0 + y1) / 2)
            frame()
            imgui.get_io().add_mouse_button_event(0, True)
            frame()
            imgui.get_io().add_mouse_button_event(0, False)
            frame()

        for _ in range(4):
            original = frame()
        study.offsets["helper-camera"] = (0.0, -0.81)
        tuned = frame()
        sx, sy = imgui.get_io().display_framebuffer_scale

        def crop(pixels, rect):
            x0, y0, x1, y1 = (round(v * s) for v, s in zip(rect, (sx, sy, sx, sy), strict=True))
            assert 0 <= x0 < x1 <= pixels.shape[1] and 0 <= y0 < y1 <= pixels.shape[0]
            return pixels[y0:y1, x0:x1]

        assert np.array_equal(crop(original, columns[0]), crop(tuned, columns[0]))
        assert not np.array_equal(crop(original, columns[1]), crop(tuned, columns[1]))
        output = Path("output/ui-optical")
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(tuned).save(output / f"{backend_name}-{scale}-{width}-candidate.png")
        click("##optical-x", 0.8)
        assert study.offset("helper-camera")[0] > 0.5
        before_blur = frame()
        offsets = study.offsets.copy()
        click("##optical-sigma", 0.9)
        assert study.sigma > 5.0
        after_blur = frame()
        assert not np.array_equal(crop(before_blur, columns[0]), crop(after_blur, columns[0]))
        assert study.offsets == offsets
        before_contour = frame()
        click("##optical-threshold", 0.75)
        assert study.threshold > 0.5
        # With guides hidden, threshold changes only measurements, not the image.
        after_contour = frame()
        assert np.array_equal(crop(before_contour, columns[0]), crop(after_contour, columns[0]))
        assert study.offsets == offsets
        study.sigma, study.threshold = 3.0, 0.25
        frame()
        click("Copy offsets")
        assert copied == [study.export()]
        camera_offset = study.offset("helper-camera")
        # All production glyphs must support the contexts and diagnostic drawing ports.
        study.guides = True
        for index in range(len(optical.SPECIMENS)):
            study.selected = index
            frame()
        study.selected = optical.SPECIMEN_INDICES["helper-light"]
        study.offsets["helper-light"] = (0.3, 0.2)
        frame()
        click("Reset selected")
        assert study.offset("helper-light") == (0.0, 0.0)
        assert study.offset("helper-camera") == camera_offset
        study.selected = optical.SPECIMEN_INDICES["helper-camera"]
        study.offsets["helper-camera"] = (0.0, -0.81)
        imgui.get_io().add_mouse_pos_event(-100, -100)
        Image.fromarray(frame()).save(output / f"{backend_name}-{scale}-{width}-guides.png")
        manual = frame()
        saved_offsets = study.offsets.copy()
        click("##optical-strength", 0.1)
        assert study.alignment_strength == 1.0
        click("Auto-align weighted centroid")
        assert study.auto_align
        aligned = frame()
        Image.fromarray(aligned).save(output / f"{backend_name}-{scale}-{width}-auto.png")
        assert np.array_equal(crop(manual, columns[0]), crop(aligned, columns[0]))
        assert not np.array_equal(crop(manual, columns[1]), crop(aligned, columns[1]))
        click("##optical-strength", 0.5)
        assert study.alignment_strength == pytest.approx(1.25, abs=0.02)
        scaled = frame()
        assert np.array_equal(crop(aligned, columns[0]), crop(scaled, columns[0]))
        assert not np.array_equal(crop(aligned, columns[1]), crop(scaled, columns[1]))
        Image.fromarray(scaled).save(output / f"{backend_name}-{scale}-{width}-strength.png")
        click("##optical-strength", 0.999)
        assert study.alignment_strength == 2.5
        Image.fromarray(frame()).save(output / f"{backend_name}-{scale}-{width}-strength-250.png")
        click("##optical-strength", 0.001)
        assert study.alignment_strength == 0.0
        assert study.effective_offset("helper-camera") == (0.0, 0.0)
        click("##optical-strength", 0.5)
        scaled_strength = study.alignment_strength
        click("##optical-y", 0.9)
        click("Reset selected")
        assert study.offsets == saved_offsets
        click("Next icon")
        assert study.selected == optical.SPECIMEN_INDICES["helper-light"]
        click("Previous icon")
        assert study.selected == optical.SPECIMEN_INDICES["helper-camera"]
        assert study.alignment_strength == scaled_strength
        click("Copy offsets")
        assert copied[-1] == study.export()
        study.alignment_strength = 1.0
        for index, (label, _) in enumerate(optical.SPECIMENS):
            study.selected = index
            pixels = frame()
            if width == 1120 and label in {"Reset", "Camera", "Light"}:
                Image.fromarray(pixels).save(output / f"{backend_name}-auto-{label.lower()}.png")
        study.selected = optical.SPECIMEN_INDICES["helper-camera"]
        frame()
        click("Auto-align weighted centroid")
        assert not study.auto_align
        assert study.offsets == saved_offsets
        assert np.array_equal(crop(manual, columns[1]), crop(frame(), columns[1]))
        if width == 1120:
            click("##optical-family")
            for key in (imgui.Key.home, imgui.Key.enter):
                for down in (True, False):
                    imgui.get_io().add_key_event(key, down)
                    frame()
            assert study.family == "Viewport tools"
            assert study.selected == optical.SPECIMEN_INDICES["tool-move"]
            study.selected = optical.SPECIMEN_INDICES["playback-previous"]
            frame()
            click("Link mirrored offsets")
            click("##optical-x", 0.65)
            click("##optical-y", 0.4)
            x, y = study.offset("playback-previous")
            assert study.offset("playback-next") == (-x, y)
            Image.fromarray(frame()).save(output / f"{backend_name}-linked-playback.png")
            study.auto_align = True
            for name in ("key-follow-locked", "status-mouse-left"):
                study.selected = optical.SPECIMEN_INDICES[name]
                Image.fromarray(frame()).save(output / f"{backend_name}-{name}.png")
