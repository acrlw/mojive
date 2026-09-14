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
WINDOW_TYPES = {"opengl": "Window", "wgpu": "WgpuWindow", "bgfx": "NativeWindow"}


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
        painted.append((center, size, name))
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
    assert {name for _center, _size, name in painted} == {glyph}
    assert {size / 2.5 for _center, size, _name in painted} == {14, 24, 56, 112}
    for (x, y), size, _name in painted[-4:]:
        half = size * 0.5
        assert 0 <= x - half < x + half <= width
        assert 0 <= y - half < y + half <= height
        region = pixels[int(y - half) : int(y + half), int(x - half) : int(x + half)]
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
            assert state.timeline_panel.follow_mode_icon_drawer is not None
            Image.fromarray(pixels).save(OUTPUT / f"{backend_name}-keyframes.png")
        finally:
            if state.timeline_session is not None:
                state.timeline_session.release()
