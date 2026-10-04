"""Compare loaded SVG assets with production geometry on the real UI backend."""

import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.tools.svg_icons import ICON_NAMES, draw_svg_icon, export_icons, load_icons
from mojive.tools.ui_feasibility import ProbeState
from mojive.tools.ui_feasibility import svg_icons as study_ui
from mojive.tools.ui_feasibility.fixtures import _apply_concept_theme
from mojive.tools.ui_feasibility.workspace import _draw_workspace
from mojive.ui.icons import REVIEW_LOCKED_ICONS, draw_icon
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.severity_icons import severity_icon
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/svg-icons/validation"


@pytest.mark.parametrize("size", (14, 24, 56, 112))
@pytest.mark.parametrize("scale", (0.65, 1.0, 1.25, 1.5))
def test_svg_and_production_silhouettes_at_actual_sizes(backend_name, tmp_path, size, scale):
    export_icons(tmp_path)
    icons = load_icons(tmp_path)
    columns, cell = 6, math.ceil(max(80, size + 32) * scale)
    rows = (len(ICON_NAMES) + columns - 1) // columns
    width, height = columns * cell * 2, rows * cell
    config = WindowConfig(
        width=width,
        height=height,
        ui_scale=scale,
        show_on_start=False,
        vsync=False,
        docking=False,
        ini_path="",
        samples=0,
    )
    with create_window(config, backend_name) as window:
        regions = []
        for _ in range(3):
            window.begin_frame()
            draw = ImguiDraw2D(imgui.get_background_draw_list())
            regions.clear()
            for i, name in enumerate(ICON_NAMES):
                x, y = (i % columns) * cell * 2, (i // columns) * cell
                center = (x + cell / 2, y + cell / 2)
                if name in REVIEW_LOCKED_ICONS:
                    # Compare SVG against the actual Output source, not a second
                    # copy of the generic status painter that could drift with it.
                    severity_icon(
                        draw, center, size * scale, name.removeprefix("status-"), (1,) * 4
                    )
                else:
                    draw_icon(draw, center, size * scale, name, (1,) * 4)
                draw_svg_icon(
                    draw, (x + cell * 1.5, y + cell / 2), size * scale, icons[name], (1, 1, 1, 1)
                )
                regions.append((name, x, y))
            sx, sy = imgui.get_io().display_framebuffer_scale
            pixels = window.end_frame(readback=True)[::-1].copy()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(OUTPUT / f"{backend_name}-{size}-{scale}-pairs.png")
        report = []
        for name, x, y in regions:
            crops = [
                pixels[
                    round(y * sy) : round((y + cell) * sy),
                    round((x + col * cell) * sx) : round((x + (col + 1) * cell) * sx),
                ]
                for col in (0, 1)
            ]
            masks = [crop[..., :3].mean(axis=2) > 160 for crop in crops]
            union = np.count_nonzero(masks[0] | masks[1])
            intersection = np.count_nonzero(masks[0] & masks[1])
            overlap = intersection / max(1, union)
            # Rasterizers may cover border pixels differently; a translated,
            # stretched, filled-in or substituted silhouette must still fail.
            assert overlap > 0.75, (name, size, overlap)
            report.append(
                {
                    "icon": name,
                    "size": size,
                    "scale": scale,
                    "framebuffer_scale": [sx, sy],
                    "opaque_overlap": overlap,
                }
            )
        (OUTPUT / f"{backend_name}-{size}-{scale}.json").write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize("size", (14, 24, 56, 112))
def test_runtime_severity_matches_output_and_svg(backend_name, tmp_path, size):
    export_icons(tmp_path)
    icons = load_icons(tmp_path)
    cell, header = max(140, size + 40), 36
    config = WindowConfig(
        width=cell * 3,
        height=cell * 3 + header,
        ui_scale=1.0,
        show_on_start=False,
        vsync=False,
        docking=False,
        ini_path="",
        samples=0,
    )
    with create_window(config, backend_name) as window:
        for _ in range(3):
            window.begin_frame()
            draw = ImguiDraw2D(imgui.get_background_draw_list())
            for column, label in enumerate(("Output", "Runtime", "SVG file")):
                draw.text((column * cell + 12, 10), (1,) * 4, label)
            for row, kind in enumerate(("info", "warning", "error")):
                y = header + (row + 0.5) * cell
                severity_icon(draw, (cell / 2, y), size, kind, (1,) * 4)
                draw_icon(draw, (cell * 1.5, y), size, f"status-{kind}", (1,) * 4)
                draw_svg_icon(draw, (cell * 2.5, y), size, icons[f"status-{kind}"], (1,) * 4)
            sx, sy = imgui.get_io().display_framebuffer_scale
            pixels = window.end_frame(readback=True)[::-1].copy()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(OUTPUT / f"{backend_name}-severity-{size}.png")
        for row in range(3):
            masks = [
                pixels[
                    round((header + row * cell) * sy) : round((header + (row + 1) * cell) * sy),
                    round(column * cell * sx) : round((column + 1) * cell * sx),
                    :3,
                ].mean(axis=2)
                > 160
                for column in range(3)
            ]
            for candidate, threshold in zip(masks[1:], (0.99, 0.9), strict=True):
                overlap = np.count_nonzero(masks[0] & candidate) / max(
                    1, np.count_nonzero(masks[0] | candidate)
                )
                assert overlap > threshold, (row, size, overlap)


def test_svg_feasibility_buttons_and_file_reload(backend_name, tmp_path, monkeypatch):
    export_icons(tmp_path)
    state = ProbeState(page="Geometry", geometry_tab="SVG icons", renderer=backend_name)
    state.svg.directory = tmp_path
    config = WindowConfig(
        width=1100,
        height=1000,
        ui_scale=1.0,
        show_on_start=False,
        vsync=False,
        docking=False,
        ini_path="",
        samples=0,
    )
    buttons = {}
    button_ids = {}
    specimens = {}
    button = imgui.button
    specimen = study_ui._specimen

    def observe_specimen(draw, center, size, name, study, *, svg, **kwargs):
        specimens[svg, size] = center
        return specimen(draw, center, size, name, study, svg=svg, **kwargs)

    def observe(label, *args, **kwargs):
        result = button(label, *args, **kwargs)
        buttons[label] = (*imgui.get_item_rect_min(), *imgui.get_item_rect_max())
        button_ids[label] = imgui.get_id(label)
        return result

    monkeypatch.setattr(imgui, "button", observe)
    monkeypatch.setattr(study_ui, "_specimen", observe_specimen)
    with create_window(config, backend_name) as window:
        _apply_concept_theme(window.style_scale)

        def frame():
            window.begin_frame()
            _draw_workspace(window, state)
            return window.end_frame(readback=True)[::-1].copy()

        def click(label):
            x0, y0, x1, y1 = buttons[label]
            io = imgui.get_io()
            io.add_mouse_pos_event((x0 + x1) / 2, (y0 + y1) / 2)
            frame()
            io.add_mouse_button_event(0, True)
            frame()
            io.add_mouse_button_event(0, False)
            frame()

        for _ in range(3):
            frame()
        click("##svg-playback")
        assert state.svg.playing
        click("##svg-playback")
        assert not state.svg.playing

        imgui.get_io().add_mouse_pos_event(-100, -100)
        frame()
        for _ in range(20):
            imgui.get_io().add_key_event(imgui.Key.tab, True)
            frame()
            imgui.get_io().add_key_event(imgui.Key.tab, False)
            frame()
            if imgui.get_current_context().nav_id == button_ids["##svg-playback"]:
                break
        assert imgui.get_current_context().nav_id == button_ids["##svg-playback"]
        imgui.get_io().add_key_event(imgui.Key.enter, True)
        frame()
        imgui.get_io().add_key_event(imgui.Key.enter, False)
        frame()
        assert state.svg.playing

        state.svg.glyph = "playback-play"
        before = frame()
        centers = specimens.copy()
        file = tmp_path / "playback-play.svg"
        root = ET.fromstring(file.read_text())
        path = next(element for element in root if element.tag.endswith("}path"))
        path.set("d", "M 4 4 L 20 4 L 12 20 Z")
        file.write_text(ET.tostring(root, encoding="unicode"))
        click("Reload SVG files")
        after = frame()
        sx, sy = imgui.get_io().display_framebuffer_scale

        def crop(pixels, svg):
            x, y = centers[svg, 112]
            return pixels[
                round((y - 60) * sy) : round((y + 60) * sy),
                round((x - 60) * sx) : round((x + 60) * sx),
            ]

        assert np.array_equal(crop(before, False), crop(after, False))
        assert np.count_nonzero(crop(before, True) != crop(after, True)) > 500
        previous = state.svg.icons
        (tmp_path / "playback-play.svg").write_text("<svg>")
        click("Reload SVG files")
        assert state.svg.error
        assert state.svg.icons is previous
        click("Export production assets")
        assert not state.svg.error
        state.svg.light = True
        state.svg.glyph = "panel-search"
        frame()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame()).save(OUTPUT / f"{backend_name}-feasibility-search.png")
