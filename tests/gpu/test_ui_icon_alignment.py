"""Exercise per-glyph offsets, automatic comparisons and cached dragging in the library."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.tools.ui_feasibility import ProbeState, icon_library
from mojive.tools.ui_feasibility.fixtures import _apply_concept_theme
from mojive.tools.ui_feasibility.icon_alignment import candidate_centroid
from mojive.tools.ui_feasibility.tuning import _icon_values_text
from mojive.tools.ui_feasibility.workspace import _draw_workspace
from mojive.ui.icons import ICON_FAMILIES, _icon_draw_commands
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("scale", (1.0, 2.25))
def test_library_offsets_toggle_export_and_drag_without_recomputing(
    backend_name, monkeypatch, scale
):
    rectangles, specimens, copied, timings = {}, {}, [], []
    native_slider, native_button, native_checkbox = imgui.slider_float, imgui.button, imgui.checkbox
    native_drag = imgui.drag_float
    native_begin_menu, native_menu_item = imgui.begin_menu, imgui.menu_item
    native_specimen = icon_library._draw_concept_icon_specimen

    def record(label):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rectangles.setdefault(label, []).append((lo.x, lo.y, hi.x, hi.y))

    def slider(label, *args, **kwargs):
        result = native_slider(label, *args, **kwargs)
        record(label)
        return result

    def drag_field(label, *args, **kwargs):
        result = native_drag(label, *args, **kwargs)
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

    def begin_menu(label, *args, **kwargs):
        result = native_begin_menu(label, *args, **kwargs)
        if label == "Probe":
            record("menu-Probe")
        return result

    def menu_item(label, *args, **kwargs):
        result = native_menu_item(label, *args, **kwargs)
        record(label)
        return result

    def specimen(draw, center, size, name, *args, **kwargs):
        if size == 56 * scale:
            specimens[name] = (
                center[0] - size * 0.6,
                center[1] - size * 0.6,
                center[0] + size * 0.6,
                center[1] + size * 0.6,
            )
        return native_specimen(draw, center, size, name, *args, **kwargs)

    monkeypatch.setattr(imgui, "slider_float", slider)
    monkeypatch.setattr(imgui, "drag_float", drag_field)
    monkeypatch.setattr(imgui, "button", button)
    monkeypatch.setattr(imgui, "checkbox", checkbox)
    monkeypatch.setattr(imgui, "begin_menu", begin_menu)
    monkeypatch.setattr(imgui, "menu_item", menu_item)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    monkeypatch.setattr(icon_library, "_draw_concept_icon_specimen", specimen)
    width, height = round(1800 * scale), round(1100 * scale)
    config = WindowConfig(
        width=width,
        height=height,
        ui_scale=scale,
        vsync=False,
        docking=False,
        ini_path="",
        show_on_start=False,
        samples=0,
    )
    output = Path("output/ui-icon-alignment")
    output.mkdir(parents=True, exist_ok=True)
    with create_window(config, backend_name) as window:
        _apply_concept_theme(window.style_scale)
        state = ProbeState(
            page="Geometry",
            geometry_tab="Icon library",
            renderer=backend_name,
            icon_library_tab="Scene helpers",
            show_icon_centroids=True,
        )

        def frame(readback=True):
            rectangles.clear()
            specimens.clear()
            window.begin_frame()
            start = perf_counter()
            _draw_workspace(window, state)
            timings.append((perf_counter() - start) * 1000)
            result = window.end_frame(readback=readback)
            return result[::-1].copy() if readback else None

        def click(label, index=0, fraction=0.5):
            x0, y0, x1, y1 = rectangles[label][index]
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            imgui.get_io().add_mouse_pos_event(x0 + (x1 - x0) * fraction, (y0 + y1) / 2)
            frame(False)
            for down in (True, False):
                imgui.get_io().add_mouse_button_event(0, down)
                frame(False)

        def save(pixels, name):
            Image.fromarray(pixels).save(output / f"{backend_name}-{scale:g}-{name}.png")

        def drag(label, distance):
            x0, y0, x1, y1 = rectangles[label][0]
            io = imgui.get_io()
            io.add_mouse_pos_event((x0 + x1) / 2, (y0 + y1) / 2)
            frame(False)
            io.add_mouse_button_event(0, True)
            frame(False)
            for delta in np.linspace(0, distance, 5):
                io.add_mouse_pos_event((x0 + x1) / 2 + delta, (y0 + y1) / 2)
                frame(False)
            io.add_mouse_button_event(0, False)
            frame(False)

        def enter_value(label, value):
            io = imgui.get_io()
            modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
            io.add_key_event(modifier, True)
            click(label)
            io.add_key_event(modifier, False)
            frame(False)
            io.add_input_characters_utf8(str(value))
            frame(False)
            io.add_key_event(imgui.Key.enter, True)
            frame(False)
            io.add_key_event(imgui.Key.enter, False)
            frame(False)

        def crop(pixels, name):
            sx, sy = imgui.get_io().display_framebuffer_scale
            x0, y0, x1, y1 = (
                round(v * s) for v, s in zip(specimens[name], (sx, sy, sx, sy), strict=True)
            )
            return pixels[y0:y1, x0:x1]

        for _ in range(4):
            original = frame()
        assert len(rectangles["##glyph-offset-x"]) == 2
        drag("##glyph-offset-x", 80)
        drag("##glyph-offset-y", -80)
        assert state.icon_manual_offset("helper-camera")[0] > 0.5
        assert state.icon_manual_offset("helper-camera")[1] < -0.5
        enter_value("##glyph-offset-x", 1.23)
        enter_value("##glyph-offset-y", -0.67)
        manual = frame()
        assert state.icon_manual_offset("helper-camera") == pytest.approx((1.23, -0.67))
        assert state.icon_manual_offset("helper-light") == (0, -0.65)
        assert not np.array_equal(crop(original, "helper-camera"), crop(manual, "helper-camera"))
        assert np.array_equal(crop(original, "helper-light"), crop(manual, "helper-light"))
        save(manual, "manual")
        saved_offsets = state.icon_offsets_by_glyph.copy()
        click("Auto align")
        assert state.icon_auto_align
        automatic = frame()
        assert not np.array_equal(crop(manual, "helper-camera"), crop(automatic, "helper-camera"))
        save(automatic, "auto")
        drag("##glyph-offset-x", 70)
        assert state.icon_offsets_by_glyph == saved_offsets
        click("##library-alignment-strength", fraction=0.999)
        assert state.icon_alignment_strength == 2.5
        click("Copy icon parameters")
        assert copied[-1] == _icon_values_text(state)
        click("Auto align")
        restored = frame()
        assert state.icon_offsets_by_glyph == saved_offsets
        assert np.array_equal(crop(manual, "helper-camera"), crop(restored, "helper-camera"))

        # Repeated drag updates translate cached vertices and the cached centroid only.
        misses = candidate_centroid.cache_info().misses
        builds = _icon_draw_commands.cache_info().misses
        x0, y0, x1, y1 = rectangles["##glyph-offset-y"][0]
        imgui.get_io().add_mouse_pos_event((x0 + x1) / 2, (y0 + y1) / 2)
        frame(False)
        imgui.get_io().add_mouse_button_event(0, True)
        frame(False)
        timings.clear()
        for fraction in np.linspace(0.2, 0.8, 30):
            imgui.get_io().add_mouse_pos_event(x0 + (x1 - x0) * fraction, (y0 + y1) / 2)
            frame(False)
        drag_times = timings.copy()
        imgui.get_io().add_mouse_button_event(0, False)
        frame(False)
        assert candidate_centroid.cache_info().misses == misses
        assert _icon_draw_commands.cache_info().misses == builds
        (output / f"{backend_name}-{scale:g}-drag.json").write_text(
            json.dumps(
                {
                    "ui_build_ms": {
                        "median": float(np.median(drag_times)),
                        "p95": float(np.percentile(drag_times, 95)),
                    },
                    "centroid_recomputations": 0,
                    "icon_geometry_rebuilds": 0,
                    "frames": len(drag_times),
                },
                indent=2,
            )
        )

        state.icon_library_tab = "Keyframe follow"
        frame(False)
        state.icon_library_tab = "Scene helpers"
        frame(False)
        state.icon_offsets_by_glyph["helper-light"] = (0.2, 0.3)
        click("Default")
        assert state.icon_manual_offset("helper-camera") == (0, -0.84)
        assert state.icon_manual_offset("helper-light") == (0.2, 0.3)

        # A paired edit is reflected on either side and survives opening the actual UI preview.
        state.icon_library_tab = "Viewport playback"
        state.icon_glyph = "playback-previous"
        frame(False)
        assert state.link_mirrored_icon_offsets
        click("Link mirrored offsets")
        assert not state.link_mirrored_icon_offsets
        click("Link mirrored offsets")
        assert state.link_mirrored_icon_offsets
        drag("##glyph-offset-x", 75)
        drag("##glyph-offset-y", -65)
        left = state.icon_manual_offset("playback-previous")
        assert state.icon_manual_offset("playback-next") == (-left[0], left[1])
        state.icon_glyph = "playback-next"
        frame(False)
        drag("##glyph-offset-x", -55)
        right = state.icon_manual_offset("playback-next")
        assert state.icon_manual_offset("playback-previous") == (-right[0], right[1])
        state.icon_glyph = None
        save(frame(), "linked-playback")

        state.page = "Redesign"
        state.preview_icon_library = True
        imgui.get_io().add_mouse_pos_event(-100, -100)
        for _ in range(3):
            applied = frame()
        saved_offsets = state.icon_offsets_by_glyph.copy()

        def viewport_crop(pixels):
            sx, sy = imgui.get_io().display_framebuffer_scale
            x0, y0, x1, y1 = (
                round(v * s)
                for v, s in zip(state.redesign.rects["viewport"], (sx, sy, sx, sy), strict=True)
            )
            return pixels[y0:y1, x0:x1]

        save(applied, "redesign-offsets-on")
        click("menu-Probe")
        save(frame(), "offset-menu")
        click("Apply icon X/Y offsets")
        assert not state.apply_icon_offsets and state.preview_icon_library
        imgui.get_io().add_mouse_pos_event(-100, -100)
        unshifted = frame()
        assert not np.array_equal(viewport_crop(applied), viewport_crop(unshifted))
        save(unshifted, "redesign-offsets-off")
        click("menu-Probe")
        click("Apply icon X/Y offsets")
        imgui.get_io().add_mouse_pos_event(-100, -100)
        restored = frame()
        assert state.apply_icon_offsets and state.icon_offsets_by_glyph == saved_offsets
        assert np.array_equal(viewport_crop(applied), viewport_crop(restored))
        state.page = "Geometry"

        if scale == 1:
            state.icon_auto_align = True
            for family, names in ICON_FAMILIES:
                state.icon_library_tab = family
                for _, name in names:
                    state.icon_glyph = name
                    frame(False)
                    assert len(rectangles["##glyph-offset-x"]) == 1
                    assert len(rectangles["##glyph-offset-y"]) == 1
            state.icon_glyph = None
            for family in (
                "Viewport playback",
                "Status & input",
                "Keyframe follow",
                "Capsules",
                "UI context",
            ):
                state.icon_library_tab = family
                frame(False)
                save(frame(), family.lower().replace(" ", "-"))
