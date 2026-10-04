"""Keep the component experiment interactive and its style overrides local."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.tools.ui_feasibility import ProbeState, components
from mojive.tools.ui_feasibility.fixtures import _apply_concept_theme
from mojive.tools.ui_feasibility.workspace import _draw_workspace
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("scale", "width", "height"), ((1.0, 1120, 820), (2.25, 2500, 1600), (1.0, 650, 1250))
)
def test_component_study_edits_shared_values_without_leaking_style(
    backend_name, monkeypatch, scale, width, height
):
    rectangles = {}
    columns = []
    native_slider, native_button = imgui.slider_float, imgui.button
    draw_column = components._column

    def record(label):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rectangles.setdefault(label, []).append((lo.x, lo.y, hi.x, hi.y))

    def slider(label, *args, **kwargs):
        result = native_slider(label, *args, **kwargs)
        record(label)
        return result

    def button(label, *args, **kwargs):
        result = native_button(label, *args, **kwargs)
        record(label)
        return result

    def tracked_widget(native):
        def draw(label, *args, **kwargs):
            result = native(label, *args, **kwargs)
            record(label)
            return result

        return draw

    def column(ctx, study, candidate):
        origin = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail().x
        draw_column(ctx, study, candidate)
        columns.append((origin.x, origin.y, origin.x + available, imgui.get_cursor_screen_pos().y))

    monkeypatch.setattr(imgui, "slider_float", slider)
    monkeypatch.setattr(imgui, "button", button)
    for name in ("checkbox", "selectable", "slider_int", "collapsing_header"):
        monkeypatch.setattr(imgui, name, tracked_widget(getattr(imgui, name)))
    monkeypatch.setattr(components, "_column", column)
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

        def frame():
            rectangles.clear()
            columns.clear()
            window.begin_frame()
            style = imgui.get_style()
            before = (style.frame_rounding, style.child_rounding, *style.window_padding)
            _draw_workspace(window, state)
            assert (style.frame_rounding, style.child_rounding, *style.window_padding) == before
            return window.end_frame(readback=True)[::-1].copy()

        def click(rect, fraction=0.5):
            x0, y0, x1, y1 = rect
            imgui.get_io().add_mouse_pos_event(x0 + (x1 - x0) * fraction, (y0 + y1) * 0.5)
            frame()
            imgui.get_io().add_mouse_button_event(0, True)
            frame()
            imgui.get_io().add_mouse_button_event(0, False)
            frame()
            # Candidate edits become visible in the earlier reference column next frame.
            frame()

        for _ in range(4):
            original = frame()
        state.components.radius = 9.0
        state.components.camera_y = -1.5
        tuned = frame()
        sx, sy = imgui.get_io().display_framebuffer_scale

        def crop(pixels, rect):
            x0, y0, x1, y1 = (
                round(value * factor) for value, factor in zip(rect, (sx, sy, sx, sy), strict=True)
            )
            assert 0 <= x0 < x1 <= pixels.shape[1]
            assert 0 <= y0 < y1 <= pixels.shape[0]
            return pixels[y0:y1, x0:x1]

        # Local candidates must not move or repaint the production reference.
        assert np.array_equal(crop(original, columns[0]), crop(tuned, columns[0]))
        assert not np.array_equal(crop(original, columns[1]), crop(tuned, columns[1]))
        for rect in rectangles["##component-position"]:
            x0, y0, x1, y1 = rect
            assert 0 <= x0 < x1 <= width
            assert 0 <= y0 < y1 <= height
        click(rectangles["##component-position"][1], 0.85)
        assert state.components.position > 0.5
        click(rectangles["##component-position-restore"][0])
        assert state.components.position == 0.0
        click(rectangles["##component-projection-1"][1])
        assert state.components.projection == 1
        state.components.guides = False
        imgui.get_io().add_mouse_pos_event(-100, -100)
        pixels = frame()
        output = Path("output/ui-components") / f"{backend_name}-{scale}-{width}-interaction.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)
        click(rectangles["Reset experiment"][0])
        assert state.components.radius == components.ComponentStudy().radius
        assert state.components.position == 0.0
        for scene, action in (
            (1, "Reset exposure"),
            (2, "Clear filters"),
            (3, "Duplicate selection"),
        ):
            state.components.scene = scene
            state.components.exposure = 2.0
            state.components.search = "camera"
            # Native auto-sized children settle their height from the previous frame.
            for _ in range(3):
                frame()
            for rect in rectangles[action]:
                assert 0 <= rect[0] < rect[2] <= width
                assert 0 <= rect[1] < rect[3] <= height
            click(rectangles[action][1])
            if scene == 1:
                assert state.components.exposure == 1.0
            elif scene == 2:
                assert state.components.search == ""
            else:
                assert state.components.action == action
            imgui.get_io().add_mouse_pos_event(-100, -100)
            Image.fromarray(frame()).save(
                output.with_name(f"{backend_name}-{scale}-{width}-scene-{scene}.png")
            )

        def next_example(scene):
            click(rectangles["Next example"][0])
            assert state.components.scene == scene
            for _ in range(3):
                frame()

        def press(label, column=1, fraction=0.5):
            rect = rectangles[label][column]
            assert 0 <= rect[0] < rect[2] <= width
            assert 0 <= rect[1] < rect[3] <= height
            click(rect, fraction)

        def capture(scene):
            imgui.get_io().add_mouse_pos_event(-100, -100)
            Image.fromarray(frame()).save(
                output.with_name(f"{backend_name}-{scale}-{width}-scene-{scene}.png")
            )

        study = state.components
        next_example(4)
        press("Lock transform", 0)
        position = study.transform_position
        press("Reset transform")
        assert study.transform_position == position
        press("Lock transform")
        press("X##Position_0_0")
        assert study.transform_position[0] == 0.0
        assert study.transform_position[1:] == position[1:]
        press("Reset transform", 0)
        assert study.transform_position == study.transform_rotation == (0.0, 0.0, 0.0)
        assert study.transform_scale == (1.0, 1.0, 1.0)
        capture(4)

        next_example(5)
        press("##corner-material_roughness", fraction=0.85)
        assert study.material_roughness > 0.7
        press("Double sided")
        assert study.material_double_sided
        press("Surface options")
        assert len(rectangles["##corner-opacity"]) == 1
        press("Surface options")
        press("Reset material", 0)
        assert study.material_roughness == pytest.approx(
            components.ComponentStudy().material_roughness
        )
        assert not study.material_double_sided
        capture(5)

        next_example(6)
        press("Key light##corner-object-3")
        assert study.scene_selected == 3
        press("##corner-visible-3")
        assert study.scene_visible[3]
        press("Cameras##corner-object-filter-1")
        assert study.scene_filter == 1
        assert "Floor##corner-object-0" not in rectangles
        study.scene_search = "no matching object"
        for _ in range(3):
            frame()
        assert "Scene camera##corner-object-2" not in rectangles
        press("Show all objects")
        assert study.scene_search == "" and study.scene_filter == 0
        assert all(study.scene_visible)
        capture(6)

        next_example(7)
        press("##corner-joint-1", fraction=0.85)
        assert study.joint_values[1] > 0.5
        press("Lock joint editing", 0)
        values = study.joint_values.copy()
        press("Reset joint values")
        assert study.joint_values == values
        press("Lock joint editing")
        press("##corner-joint-1-restore")
        assert study.joint_values[1] == 0.0 and study.joint_values[0] == values[0]
        press("Reset joint values")
        assert study.joint_values == [0.0, 0.0, 0.0]
        capture(7)

        next_example(8)
        press("##corner-capture-quality", fraction=0.3)
        assert study.capture_quality == 90
        press("JPEG##corner-capture-format-1")
        assert study.capture_format == 1
        press("##corner-capture-quality", fraction=0.3)
        assert study.capture_quality < 50
        press("Transparent background")
        assert not study.capture_transparent
        press("Preview capture")
        assert study.capture_message.startswith("camera-study.jpg / 1920 x 1080")
        capture(8)
        message = study.capture_message
        study.capture_name = "  "
        frame()
        press("Preview capture")
        assert study.capture_message == message
        next_example(0)
        press("Previous example", 0)
        assert study.scene == 8 and study.capture_format == 1

        # Surface experiments belong to B; A remains an unchanged panel reference.
        # Switching modes must retain both sample edits and the saved card settings.
        sample_values = (study.transform_position, tuple(study.joint_values), study.capture_format)
        study.inset = 10.0
        for scene in range(len(components.CORNER_SCENES)):
            study.scene = scene
            # Delayed tooltips can cover A at HiDPI; compare unhovered surfaces.
            imgui.get_io().add_mouse_pos_event(-100, -100)
            for _ in range(3):
                flat = frame()
            reference_rect = columns[0]
            reference = crop(flat, reference_rect)
            assert "##component-inset" not in rectangles
            assert "Link card radius to inset" not in rectangles
            press("Card##corner-surface-mode-1", 0)
            assert study.card_background and study.inset == 10.0
            imgui.get_io().add_mouse_pos_event(-100, -100)
            for _ in range(3):
                card = frame()
            assert columns[0] == reference_rect
            assert np.array_equal(reference, crop(card, columns[0]))
            assert not np.array_equal(flat, card)
            Image.fromarray(card).save(
                output.with_name(f"{backend_name}-{scale}-{width}-card-{scene}.png")
            )
            press("Panel##corner-surface-mode-0", 0)
            assert not study.card_background and study.inset == 10.0
            imgui.get_io().add_mouse_pos_event(-100, -100)
            for _ in range(3):
                restored = frame()
            assert np.array_equal(reference, crop(restored, columns[0]))
            assert sample_values == (
                study.transform_position,
                tuple(study.joint_values),
                study.capture_format,
            )
