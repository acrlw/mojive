"""Native popup edges composite once over their shadow, at real framebuffer density."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.ui.floating_shadow import _shadow_colors, _shadow_meshes, draw_surface_shadow
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu
OUTPUT = Path("output/popup-edges")


@pytest.mark.physics
@pytest.mark.parametrize("scale", (1, 1.25))
def test_content_focus_and_independent_settings_switches(
    backend_name, scale, monkeypatch, tmp_path
):
    from mojive.app.composition import build_workspace
    from mojive.assets import resolve
    from mojive.config import LayoutConfig, ViewerConfig
    from mojive.design.themes import select_theme
    from mojive.tools.native_ui_review import _NativeInput, _reveal
    from mojive.tools.native_workspace import capture

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1440,
            height=1000,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, workspace = viewer.app, viewer.app.workspace
        ui = workspace.ui
        app.set_language("zh_CN")
        app.design.observe_components = True
        output = OUTPUT / f"{backend_name}-settings-{scale:g}"
        output.mkdir(parents=True, exist_ok=True)
        inputs.sync(10)
        for theme in ("light", "dark"):
            select_theme(app.design, theme, persist=False)
            workspace.open["Settings"] = False
            inputs.sync()
            inputs.click("scene-filter-all")
            inputs.event("add_key_event", imgui.Key.tab, True)
            inputs.event("add_key_event", imgui.Key.tab, False)
            inputs.escape()
            capture(viewer, output / f"{theme}-scene-after-escape.png")
            workspace.settings.show_category("Recording")
            workspace.activate("Settings")
            inputs.sync(8)
            key = "##capture_clipboard"
            _reveal(inputs, key, "Reference settings content")
            record = next(r for r in reversed(ui.design.primitives.observed) if r["key"] == key)
            assert record["kind"] == "switch"
            x, y, w, h = record["labelRect"]
            before = app.recording_config.copy_to_clipboard
            inputs.move((x + min(w / 2, 16 * scale), y + h / 2))
            capture(viewer, output / f"{theme}-settings-label-hover.png")
            inputs.click_at(inputs.pointer)
            assert app.recording_config.copy_to_clipboard != before
            inputs.click_at(inputs.pointer)
            assert app.recording_config.copy_to_clipboard == before
            capture(viewer, output / f"{theme}-settings-recording.png")
            workspace.settings.show_category("Rendering")
            inputs.sync(5)
            checkboxes = [r for r in ui.design.primitives.observed if r["key"].startswith("##rf_")]
            assert checkboxes and all(r["kind"] == "checkbox" for r in checkboxes)
            capture(viewer, output / f"{theme}-settings-multiple-flags.png")


@pytest.mark.physics
@pytest.mark.parametrize("scale", (1, 1.25))
def test_viewport_popup_focus_switching_and_shared_chrome(
    backend_name, scale, monkeypatch, tmp_path
):
    from mojive.app.composition import build_workspace
    from mojive.assets import resolve
    from mojive.config import LayoutConfig, ViewerConfig
    from mojive.design.themes import select_theme
    from mojive.tools.native_ui_review import _NativeInput
    from mojive.tools.native_workspace import capture

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        width=1440,
        height=1000,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    output = OUTPUT / f"{backend_name}-focus-{scale:g}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        app, ui = viewer.app, viewer.app.workspace.ui
        app.set_language("zh_CN")
        with _NativeInput(viewer) as inputs:
            inputs.sync(10)
            for theme in ("light", "dark"):
                select_theme(app.design, theme, persist=False)
                inputs.sync()
                inputs.click("viewport-layers")
                assert "viewport-layers-popup" in ui.hits
                context = imgui.get_current_context()
                layer_window = context.open_popup_stack[0].window
                target = layer_window.get_id("layer-view_cube")
                for _ in range(12):
                    inputs.event("add_key_event", imgui.Key.tab, True)
                    inputs.event("add_key_event", imgui.Key.tab, False)
                    inputs.sync()
                    if context.nav_id == target:
                        break
                assert context.nav_id == target and context.nav_cursor_visible
                capture(viewer, output / f"{theme}-keyboard-focus.png")
                before = app.viewport_layers.view_cube
                inputs.event("add_key_event", imgui.Key.space, True)
                inputs.event("add_key_event", imgui.Key.space, False)
                inputs.sync()
                assert app.viewport_layers.view_cube != before
                rx, ry, rw, rh = ui.hits["layers-reset"]
                inputs.move((rx + rw / 2, ry + rh / 2))
                capture(viewer, output / f"{theme}-reset-hover.png")
                inputs.event("add_mouse_button_event", 0, True)
                capture(viewer, output / f"{theme}-reset-held.png")
                inputs.event("add_mouse_button_event", 0, False)
                inputs.sync()
                assert not context.nav_cursor_visible
                assert app.viewport_layers.view_cube
                shell = ui.hits["viewport-layers-popup"]
                assert rx - shell[0] > 0
                assert ry - shell[1] == pytest.approx(shell[0] + shell[2] - rx - rw, abs=1)
                border = layer_window.window_border_size
                radius = layer_window.window_rounding
                # Hover switches only within the viewport's own menu set.
                for trigger in ("projection-button", "shading", "viewport-layers"):
                    x, y, w, h = ui.hits[trigger]
                    inputs.move((x + w / 2, y + h / 2))
                    assert len(context.open_popup_stack) == 1
                    popup = context.open_popup_stack[0].window
                    assert popup.window_border_size == pytest.approx(border)
                    assert popup.window_rounding == pytest.approx(radius)
                    if trigger != "viewport-layers":
                        assert "viewport-layers-popup" not in ui.hits
                    capture(viewer, output / f"{theme}-{trigger}.png")
                inputs.click("viewport-layers")
                assert not context.open_popup_stack
                inputs.click("viewport-layers")
                assert "viewport-layers-popup" in ui.hits
                inputs.escape()
                assert not context.open_popup_stack
    finally:
        viewer.close()


@pytest.mark.parametrize("scale", (1, 1.25))
@pytest.mark.parametrize("theme", ("light", "dark"))
def test_native_shadow_matches_background_first_compositing(backend_name, scale, theme):
    config = WindowConfig(
        width=800,
        height=680,
        ui_scale=scale,
        show_on_start=False,
        vsync=False,
        docking=False,
        ini_path="",
        samples=0,
    )
    fill = (249, 251, 247, 255) if theme == "light" else (45, 53, 61, 255)
    ground = (220, 232, 220, 255) if theme == "light" else (62, 70, 82, 255)
    fill = imgui.color_convert_float4_to_u32(tuple(v / 255 for v in fill))
    ground = imgui.color_convert_float4_to_u32(tuple(v / 255 for v in ground))
    shadow = 0x40000000
    with create_window(config, backend_name) as window:
        regions = []
        for _ in range(3):
            window.begin_frame()
            regions.clear()
            imgui.get_background_draw_list().add_rect_filled((0, 0), (800, 680), ground)
            for row, radius in enumerate((8, 12, 24)):
                for border in (0, 1):
                    y = 45 + row * 210 + border * 100
                    size = (240, 64)
                    regions.append((y, size))
                    for column in (0, 1):
                        origin = (60 + 400 * column, y)
                        imgui.set_next_window_pos(origin)
                        imgui.set_next_window_size(size)
                        imgui.push_style_var(imgui.StyleVar_.window_rounding, radius * scale)
                        imgui.push_style_var(imgui.StyleVar_.window_border_size, border * scale)
                        imgui.push_style_color(imgui.Col_.window_bg, fill)
                        imgui.push_style_color(imgui.Col_.border, 0x30708090)
                        flags = (
                            imgui.WindowFlags_.no_title_bar
                            | imgui.WindowFlags_.no_resize
                            | imgui.WindowFlags_.no_saved_settings
                            | imgui.WindowFlags_.no_inputs
                        )
                        if column:
                            flags |= imgui.WindowFlags_.no_background
                        imgui.begin(f"Edge-{row}-{border}-{column}", None, flags)
                        draw = imgui.get_window_draw_list()
                        if column:
                            # Independent reference order: shadow, then native fill/border.
                            draw.push_clip_rect_full_screen()
                            for mesh, ink in zip(
                                _shadow_meshes(*size, radius * scale, scale),
                                _shadow_colors(shadow),
                                strict=True,
                            ):
                                mesh.draw(draw, origin, ink)
                            end = (origin[0] + size[0], origin[1] + size[1])
                            draw.add_rect_filled(origin, end, fill, radius * scale)
                            if border:
                                inset = border * scale / 2
                                offset = inset - 0.5
                                draw.add_rect(
                                    (origin[0] + offset, origin[1] + offset),
                                    (end[0] - offset, end[1] - offset),
                                    0x30708090,
                                    radius * scale - inset,
                                    border * scale,
                                )
                            draw.pop_clip_rect()
                        else:
                            draw_surface_shadow(draw, origin, size, radius * scale, scale, shadow)
                        imgui.end()
                        imgui.pop_style_color(2)
                        imgui.pop_style_var(2)
            sx, sy = imgui.get_io().display_framebuffer_scale
            pixels = window.end_frame(readback=True)[::-1].copy()
        output = OUTPUT / f"{backend_name}-{theme}-{scale:g}"
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output / "compositing.png")
        for y, (width, height) in regions:
            crops = [
                pixels[
                    round((y - 30) * sy) : round((y + height + 30) * sy),
                    round((40 + column * 400) * sx) : round((80 + width + column * 400) * sx),
                    :3,
                ]
                for column in (0, 1)
            ]
            error = np.abs(crops[0].astype(int) - crops[1].astype(int))
            # Native and reference order must agree to a single blend rounding step.
            assert error.max() <= 1, (y, error.max(), np.count_nonzero(error > 1))
