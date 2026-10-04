"""The design workspace uses the selected backend and keeps UI drawing in ImGui."""

import json
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.tools.ui_feasibility import ProbeState, render, runtime
from mojive.tools.ui_feasibility.fixtures import _apply_concept_theme
from mojive.tools.ui_feasibility.workspace import _draw_workspace
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/canvas2d/ui-regression"
WINDOW_TYPES = {"opengl": "Window", "bgfx": "NativeWindow"}


@pytest.mark.parametrize("scale", (1.0, 2.25))
def test_wrapped_study_tabs_remain_visible_and_selectable(backend_name, monkeypatch, scale):
    from mojive.tools.ui_feasibility.layout import _wrapped_tabs

    tabs = tuple(
        (name, "study-tabs", str(i))
        for i, name in enumerate(
            (
                "Corners",
                "Playback",
                "Transform gizmos",
                "Joint helpers",
                "Diagnostics",
                "Workspaces",
            )
        )
    )
    rectangles = {}
    original = imgui.invisible_button

    def observe(label, *args, **kwargs):
        clicked = original(label, *args, **kwargs)
        if label.startswith("##study-tabs-"):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rectangles[label] = (lo.x, lo.y, hi.x, hi.y)
        return clicked

    monkeypatch.setattr(imgui, "invisible_button", observe)
    config = WindowConfig(
        width=500,
        height=650,
        ui_scale=scale,
        ini_path="",
        docking=False,
        show_on_start=False,
        vsync=False,
    )
    with create_window(config, backend_name) as window:
        active = tabs[0][0]

        def frame():
            nonlocal active
            window.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(imgui.get_io().display_size)
            imgui.begin("Study tabs", flags=imgui.WindowFlags_.no_decoration)
            active = _wrapped_tabs(tabs, active, imgui.get_content_region_avail().x)
            imgui.end()
            return window.end_frame(readback=True)

        for _ in range(3):
            frame()
        assert len(rectangles) == len(tabs)
        assert len({rect[1] for rect in rectangles.values()}) > 1
        width, height = imgui.get_io().display_size
        for x0, y0, x1, y1 in rectangles.values():
            assert 0 <= x0 < x1 <= width
            assert 0 <= y0 < y1 <= height
        x0, y0, x1, y1 = rectangles["##study-tabs-5-5"]
        imgui.get_io().add_mouse_pos_event((x0 + x1) / 2, (y0 + y1) / 2)
        frame()
        imgui.get_io().add_mouse_button_event(0, True)
        frame()
        imgui.get_io().add_mouse_button_event(0, False)
        pixels = frame()
        assert active == "Workspaces"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[::-1]).save(OUTPUT / f"{backend_name}-{scale}-wrapped-tabs.png")


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_feasibility_capture_uses_selected_backend(backend_name, monkeypatch, scale):
    monkeypatch.setenv("MOJIVE_RENDERER", backend_name)
    observed = []
    original = runtime._draw_workspace

    def observe(window, state):
        original(window, state)
        observed.append(
            {
                "window": type(window).__name__,
                "renderer": state.renderer,
            }
        )

    monkeypatch.setattr(runtime, "_draw_workspace", observe)
    output = OUTPUT / f"{backend_name}-{scale}-icons.png"
    render(
        output,
        1600,
        1000,
        interactive=False,
        initial_page="Geometry",
        initial_geometry_tab="Icon library",
        initial_icon_group="Viewport tools",
        initial_rotate_cap="round",
        ui_scale=scale,
        interactive_fps=30,
    )
    assert observed[-1]["window"] == WINDOW_TYPES[backend_name]
    assert observed[-1]["renderer"] == backend_name
    pixels = np.asarray(Image.open(output))
    assert pixels.std() > 15
    output.with_suffix(".json").write_text(json.dumps(observed, indent=2))


@pytest.mark.parametrize("glyph", ("playback-reset", "transport-loop"))
def test_focused_glyph_keeps_every_review_size_inside_hidpi_capture(
    backend_name, monkeypatch, glyph
):
    from mojive.tools.ui_feasibility import icon_library

    painted = []
    original = icon_library._draw_concept_icon_specimen

    def observe(draw, center, size, name, *args, **kwargs):
        painted.append((center, size, name, tuple(imgui.get_io().display_framebuffer_scale)))
        return original(draw, center, size, name, *args, **kwargs)

    monkeypatch.setattr(icon_library, "_draw_concept_icon_specimen", observe)
    output = OUTPUT / f"{backend_name}-{glyph}-2.5.png"
    render(
        output,
        1024,
        680,
        interactive=False,
        initial_page="Workspace",
        initial_geometry_tab="Playback",
        initial_icon_group="Overview",
        initial_rotate_cap="round",
        ui_scale=2.5,
        interactive_fps=30,
        renderer=backend_name,
        icon_glyph=glyph,
    )
    pixels = np.asarray(Image.open(output))
    height, width = pixels.shape[:2]
    assert {name for _center, _size, name, _scale in painted} == {glyph}
    assert {size / 2.5 for _center, size, _name, _scale in painted} == {14, 24, 56, 112}
    for (x, y), size, _name, (sx, sy) in painted[-4:]:
        x, y = x * sx, y * sy
        half_x, half_y = size * sx * 0.5, size * sy * 0.5
        assert 0 <= x - half_x < x + half_x <= width
        assert 0 <= y - half_y < y + half_y <= height
        region = pixels[int(y - half_y) : int(y + half_y), int(x - half_x) : int(x + half_x)]
        assert region.std() > 20


def test_live_tuning_reaches_redesign_and_production_panels(backend_name):
    config = WindowConfig(
        width=1600,
        height=1000,
        vsync=False,
        docking=False,
        ini_path="",
        show_on_start=False,
        ui_scale=1.0,
    )
    with create_window(config, backend_name) as window:
        _apply_concept_theme(window.style_scale)
        state = ProbeState(page="Redesign", renderer=backend_name, preview_icon_library=True)

        def capture():
            for _ in range(3):
                window.begin_frame()
                _draw_workspace(window, state)
                pixels = window.end_frame(readback=True)
            assert pixels is not None
            return pixels[::-1].copy()

        try:
            original = capture()
            state.set_icon_stroke_for_glyph("tool-rotate", 2.2)
            tuned = capture()
            # Compare the capsule itself; a preview-toggle caption must not be
            # sufficient to pass this check when the actual icon remains stale.
            sx, sy = imgui.get_io().display_framebuffer_scale
            left, top, right, bottom = (
                round(value * scale)
                for value, scale in zip(
                    state.redesign.rects["tools"], (sx, sy, sx, sy), strict=True
                )
            )
            assert (
                np.count_nonzero(original[top:bottom, left:right] != tuned[top:bottom, left:right])
                > 100
            )
            OUTPUT.mkdir(parents=True, exist_ok=True)
            Image.fromarray(tuned).save(OUTPUT / f"{backend_name}-redesign-tuned.png")
            state.page = "Geometry"
            state.geometry_tab = "Workspaces"
            pixels = capture()
            assert state.timeline_session is not None
            assert state.timeline_panel.toolbar.follow_mode_icon_drawer is not None
            Image.fromarray(pixels).save(OUTPUT / f"{backend_name}-keyframes.png")
        finally:
            if state.timeline_session is not None:
                state.timeline_session.release()


def test_preview_callbacks_remain_local_to_each_panel(backend_name, monkeypatch):
    from mojive.tools.ui_feasibility import panels as specimens
    from mojive.ui.keyframe_editor import controls
    from mojive.ui.panels import PanelContext, output
    from mojive.ui.panels.keyframes import KeyframesPanel

    production_command_icon = controls._draw_command_icon
    production_search = output.search_input
    preview_command_icon = specimens._draw_icon_library_command_icon
    preview_search_icon = specimens._draw_concept_control_icon
    commands, search_icons = set(), set()

    def command_icon(draw, center, kind, *args, **kwargs):
        # Check while the preview is painting, when a temporary global replacement
        # would also affect other panels or a reentrant draw.
        assert controls._draw_command_icon is production_command_icon
        commands.add(kind)
        return preview_command_icon(draw, center, kind, *args, **kwargs)

    def search_icon(draw, center, size, kind, *args, **kwargs):
        assert output.search_input is production_search
        search_icons.add(kind)
        return preview_search_icon(draw, center, size, kind, *args, **kwargs)

    monkeypatch.setattr(specimens, "_draw_icon_library_command_icon", command_icon)
    monkeypatch.setattr(specimens, "_draw_concept_control_icon", search_icon)
    state = ProbeState(renderer=backend_name, preview_icon_library=True)
    standard_timeline, standard_output = KeyframesPanel(), output.OutputPanel()
    config = WindowConfig(
        width=1600, height=1000, vsync=False, docking=False, ini_path="", show_on_start=False
    )
    with create_window(config, backend_name) as window:
        _apply_concept_theme(window.style_scale)

        def frame():
            window.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((800, 1000))
            imgui.begin("Design preview", flags=imgui.WindowFlags_.no_decoration)
            specimens._draw_keyframes(imgui.ImVec2(0, 440), window.style_scale, state)
            specimens._draw_output(imgui.ImVec2(0, 440), state, window.style_scale)
            imgui.end()
            imgui.set_next_window_pos((800, 0))
            imgui.set_next_window_size((800, 1000))
            imgui.begin("Production panels", flags=imgui.WindowFlags_.no_decoration)
            ctx = PanelContext(
                state.timeline_session,
                None,
                style_scale=window.style_scale,
                output=state.output_buffer,
            )
            imgui.begin_child("Timeline", (0, 440))
            standard_timeline.draw(ctx)
            imgui.end_child()
            imgui.begin_child("Output", (0, 440))
            standard_output.draw(ctx)
            imgui.end_child()
            imgui.end()
            return window.end_frame(readback=True)

        try:
            for _ in range(3):
                pixels = frame()
            assert {"record", "play", "add", "key-keyframe"} <= commands
            assert "search" in search_icons
            assert standard_timeline.toolbar.command_icon_drawer is production_command_icon
            assert standard_output.search_icon_drawer is None
            OUTPUT.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels[::-1]).save(
                OUTPUT / f"{backend_name}-panel-preview-isolation.png"
            )
            commands.clear()
            search_icons.clear()
            state.preview_icon_library = False
            frame()
            assert commands == search_icons == set()
            assert state.timeline_panel.toolbar.command_icon_drawer is production_command_icon
            assert state.output_panel.search_icon_drawer is None
        finally:
            if state.timeline_session is not None:
                state.timeline_session.release()
