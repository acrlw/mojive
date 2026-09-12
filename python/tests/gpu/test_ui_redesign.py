"""Exercise layout candidates through hidden OpenGL windows and native ImGui input."""

from pathlib import Path

import numpy as np
import pytest
from design.tools import render_ui_feasibility as probe
from design.tools.ui_capsule_geometry import g3_capsule_spans, optical_end_padding, spacing_metrics
from design.tools.ui_redesign import _capsule, reset_glyph_path
from imgui_bundle import imgui
from PIL import Image

from mojive.ui.window import Window, WindowConfig

pytestmark = pytest.mark.gpu


class Rig:
    def __init__(self, monkeypatch, width=1400, height=820, scale=1.0, language="en"):
        monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
        self.window = Window(
            WindowConfig(
                width=width,
                height=height,
                docking=False,
                ini_path="",
                show_on_start=False,
                samples=0,
            )
        )
        probe._apply_concept_theme(self.window.style_scale)
        self.probe = probe.ProbeState(page="Redesign")
        self.state = self.probe.redesign
        self.state.language = language
        self.events = []
        drag = imgui.drag_float
        checkbox = imgui.checkbox

        def record_drag(label, *args, **kwargs):
            result = drag(label, *args, **kwargs)
            for index in range(3):
                if label == f"##redesign-control-{index}-value":
                    a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    self.state.rects[f"control-value-{index}"] = (a.x, a.y, b.x, b.y)
            preview = self.state.inspector
            if preview is not None:
                for name in ("position", "rotation"):
                    for axis in range(3):
                        if (
                            label
                            == f"##{preview.context.tr(name)}_{axis}_{preview.session.selected_node.node_id}"
                        ):
                            a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                            self.state.rects[f"{name}-{axis}"] = (a.x, a.y, b.x, b.y)
            return result

        def record_checkbox(label, *args, **kwargs):
            result = checkbox(label, *args, **kwargs)
            if label in ("Highlight G3 transitions", "Optical end spacing"):
                a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                self.state.rects[label] = (a.x, a.y, b.x, b.y)
            return result

        slider = imgui.slider_float

        def record_slider(label, *args, **kwargs):
            result = slider(label, *args, **kwargs)
            for index in range(3):
                if label == f"##redesign-control-{index}":
                    a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    self.state.rects[f"control-rail-{index}"] = (a.x, a.y, b.x, b.y)
            return result

        monkeypatch.setattr(imgui, "slider_float", record_slider)
        monkeypatch.setattr(imgui, "drag_float", record_drag)
        monkeypatch.setattr(imgui, "checkbox", record_checkbox)
        process = self.window._input.process_inputs

        def inputs():
            process()
            for method, args in self.events:
                getattr(imgui.get_io(), method)(*args)
            self.events.clear()

        monkeypatch.setattr(self.window._input, "process_inputs", inputs)
        for _ in range(4):
            self.frame()

    def frame(self):
        self.window.begin_frame()
        probe._draw_workspace(self.window, self.probe)
        self.pixels = self.window.end_frame(readback=True)

    def move(self, x, y):
        self.events.append(("add_mouse_pos_event", (x, y)))
        self.frame()

    def click(self, key, button=0):
        x0, y0, x1, y1 = self.state.rects[key]
        self.move((x0 + x1) / 2, (y0 + y1) / 2)
        for down in (True, False):
            self.events.append(("add_mouse_button_event", (button, down)))
            self.frame()
        self.frame()

    def key(self, key):
        for down in (True, False):
            self.events.append(("add_key_event", (key, down)))
            self.frame()

    def control_modifier(self, down):
        key = imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl
        self.events.append(("add_key_event", (key, down)))

    def double_click(self, key):
        self.click(key)
        self.click(key)

    def viewport(self):
        x0, y0, x1, y1 = self.state.rects["viewport"]
        self.move(x0 + (x1 - x0) * 0.65, y0 + (y1 - y0) * 0.65)

    def save(self, name):
        path = Path("output/ui-redesign") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.pixels[::-1]).save(path)

    def close(self):
        self.window.close()


@pytest.fixture
def rig(monkeypatch):
    value = Rig(monkeypatch)
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize("scale", (1.0, 1.5))
def test_keyframe_follow_icons_keep_native_selection_in_production_and_preview(monkeypatch, scale):
    width, height = probe._probe_window_size(1600, 1000, scale)
    rig = Rig(monkeypatch, width=width, height=height, scale=scale)
    original_button = imgui.button
    original_tooltip = imgui.set_item_tooltip
    hovered_tooltips = []
    style = imgui.get_style()
    style.hover_delay_normal = style.hover_delay_short = style.hover_stationary_delay = 0.0

    def record_button(label, *args, **kwargs):
        result = original_button(label, *args, **kwargs)
        item_id = label.partition("##")[2]
        if item_id.startswith("timeline-follow-"):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rig.state.rects[item_id] = (lo.x, lo.y, hi.x, hi.y)
        return result

    def record_tooltip(label):
        if label in probe.keyframes_panel_module.FOLLOW_MODE_TOOLTIPS and imgui.is_item_hovered():
            hovered_tooltips.append(label)
        original_tooltip(label)

    monkeypatch.setattr(imgui, "button", record_button)
    monkeypatch.setattr(imgui, "set_item_tooltip", record_tooltip)
    try:
        rig.probe.page = "Geometry"
        rig.probe.geometry_tab = "Workspaces"
        for preview in (False, True):
            rig.probe.preview_icon_library = preview
            for _ in range(4):
                rig.frame()
            bounds = [rig.state.rects[f"timeline-follow-{i}"] for i in range(3)]
            assert len({box[1] for box in bounds}) == 1
            assert all(x1 - x0 == pytest.approx(y1 - y0) for x0, y0, x1, y1 in bounds)
            for index in (0, 2, 1):
                rig.click(f"timeline-follow-{index}")
                assert rig.probe.timeline_panel._follow_mode == ("off", "page", "locked")[index]
                assert (
                    hovered_tooltips[-1] == probe.keyframes_panel_module.FOLLOW_MODE_TOOLTIPS[index]
                )
                assert any(
                    window.active and window.flags & imgui.WindowFlags_.tooltip.value
                    for window in imgui.get_current_context().windows
                )
        rig.probe.preview_icon_library = False
        rig.frame()
        assert rig.probe.timeline_panel.follow_mode_icon_drawer is probe.draw_icon_label
        assert rig.probe.timeline_panel._follow_mode == "page"
    finally:
        rig.close()


def test_transport_recording_and_tools_keep_independent_state(rig):
    rig.click("step")
    assert rig.state.frame == 25
    rig.click("previous")
    assert rig.state.frame == 24
    rig.click("rotate")
    rig.click("snap")
    rig.click("frame")
    assert (rig.state.tool, rig.state.snap_locked, rig.state.space) == ("rotate", True, "body")
    rig.viewport()
    rig.events.append(("add_key_event", (imgui.Key.mod_shift, True)))
    rig.frame()
    assert rig.state.snap_held and rig.state.snap_locked
    rig.events.append(("add_key_event", (imgui.Key.mod_shift, False)))
    rig.frame()
    assert not rig.state.snap_held and rig.state.snap_locked
    rig.click("menu")
    rig.click("record-video")
    assert rig.state.recording == "video"
    rig.save("recording-video.png")
    rig.click("record")
    assert not rig.state.recording
    rig.click("record")
    assert rig.state.recording == "video"
    rig.click("reset")
    assert rig.state.frame == 0 and not rig.state.playing and not rig.state.recording
    assert rig.state.tool == "rotate" and rig.state.snap_locked


def test_keyboard_scope_preserves_r_rotation_and_text_input(rig):
    rig.viewport()
    rig.key(imgui.Key.r)
    assert rig.state.tool == "rotate"
    rig.key(imgui.Key.e)
    assert rig.state.tool == "dimensions"
    rig.key(imgui.Key.right_arrow)
    assert rig.state.frame == 25
    rig.key(imgui.Key.left_arrow)
    assert rig.state.frame == 24
    rig.events.append(("add_mouse_button_event", (1, True)))
    rig.frame()
    rig.key(imgui.Key.r)
    assert rig.state.flying and rig.state.tool == "dimensions"
    rig.events.append(("add_mouse_button_event", (1, False)))
    rig.frame()
    rig.key(imgui.Key.r)
    assert not rig.state.flying and rig.state.tool == "rotate"
    rig.double_click("name")
    rig.frame()
    assert rig.state.renaming
    rig.viewport()
    rig.key(imgui.Key.e)
    assert rig.state.tool == "rotate"
    rig.events.append(("add_input_characters_utf8", ("torso_renamed",)))
    rig.frame()
    rig.key(imgui.Key.enter)
    assert rig.state.name == "torso_renamed" and not rig.state.renaming
    rig.double_click("name")
    rig.events.append(("add_input_characters_utf8", ("_canceled",)))
    rig.frame()
    rig.key(imgui.Key.escape)
    assert rig.state.name == "torso_renamed" and not rig.state.renaming


def test_name_double_click_blur_commit_and_blank_cancel(rig):
    rig.click("name")
    assert not rig.state.renaming
    rig.click("name")
    rig.frame()
    assert rig.state.renaming
    rig.events.append(("add_input_characters_utf8", ("frame_edited",)))
    rig.frame()
    rig.click("details")
    assert rig.state.name == "frame_edited" and not rig.state.renaming
    assert rig.state.inspector.session.selected_node.name == "frame_edited"
    rig.key(imgui.Key.escape)
    rig.double_click("name")
    rig.events.append(("add_input_characters_utf8", ("   ",)))
    rig.frame()
    rig.click("section-Control")
    assert rig.state.name == "frame_edited" and not rig.state.renaming
    rig.click("section-Inspector")
    rig.double_click("name")
    rig.events.append(("add_input_characters_utf8", ("frame_blur",)))
    rig.frame()
    rig.click("position-0")
    assert rig.state.name == "frame_blur" and not rig.state.renaming
    rig.double_click("name")
    rig.events.append(("add_input_characters_utf8", ("frame_focus_lost",)))
    rig.frame()
    rig.events.append(("add_focus_event", (False,)))
    rig.frame()
    rig.frame()
    assert rig.state.name == "frame_focus_lost" and not rig.state.renaming
    rig.events.append(("add_focus_event", (True,)))
    rig.frame()


def test_control_rail_exact_entry_zero_restore_and_readonly(rig):
    rig.click("section-Control")
    x0, y0, x1, y1 = rig.state.rects["control-rail-0"]
    rig.move(x0 + (x1 - x0) * 0.8, (y0 + y1) / 2)
    for down in (True, False):
        rig.events.append(("add_mouse_button_event", (0, down)))
        rig.frame()
    assert 0.8 < rig.state.controls[0] < 1.6
    rig.click("control-value-0", button=1)
    assert rig.state.controls[0] == pytest.approx(0.42)
    rig.click("readonly")
    before = rig.state.controls.copy()
    rig.click("control-value-1", button=1)
    assert rig.state.controls == before
    rig.click("readonly")
    rig.control_modifier(True)
    rig.frame()
    rig.click("control-value-2")
    rig.key(imgui.Key.a)
    rig.control_modifier(False)
    rig.frame()
    assert imgui.get_io().want_text_input
    rig.events.append(("add_input_characters_utf8", ("123.456",)))
    rig.frame()
    rig.key(imgui.Key.enter)
    assert rig.state.controls[2] == pytest.approx(123.456)
    rig.click("restore-all")
    rig.click("confirm-restore")
    assert rig.state.controls == pytest.approx([0.42, -1.1, 2.5])
    rig.save("control-verified.png")


def test_transform_uses_joined_fields_and_accepts_exact_values(rig):
    rig.click("section-Inspector")
    rig.control_modifier(True)
    rig.frame()
    rig.click("position-0")
    rig.key(imgui.Key.a)
    rig.control_modifier(False)
    rig.frame()
    assert imgui.get_io().want_text_input
    rig.events.append(("add_input_characters_utf8", ("0.125",)))
    rig.frame()
    rig.key(imgui.Key.enter)
    assert rig.state.position == pytest.approx([0.125, 0, 0.48])
    rig.save("transform-verified.png")


def test_existing_capsule_ink_matches_geometry_and_reset_stays_in_bound(rig, monkeypatch):
    redesign = False
    vertical = False

    def draw(_window, state):
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(900, 600))
        imgui.begin("##geometry-compatibility", None, imgui.WindowFlags_.no_decoration)
        if redesign:
            _capsule(state.redesign, (100, 100), 2, state, probe._circular_icon_button, vertical)
        elif vertical:
            probe._draw_tool_column(
                probe.ImguiDraw2D(imgui.get_window_draw_list()), (100, 100), 2, state
            )
        else:
            probe._draw_playback(
                probe.ImguiDraw2D(imgui.get_window_draw_list()), (100, 100), 2, state
            )
        imgui.end()

    monkeypatch.setattr(probe, "_draw_workspace", draw)
    for vertical in (False, True):
        for selected in (False, True):
            rig.probe.playing = rig.state.playing = selected
            rig.probe.active_tool = rig.state.tool = "move" if selected else "rotate"
            redesign = False
            rig.frame()
            baseline = rig.pixels[::-1].copy()
            redesign = True
            rig.frame()
            actual = rig.pixels[::-1]
            # First two buttons retain the accepted glyph sizes, alignment, circular
            # hover/selection backgrounds and capsule cap, without shortcut badges.
            # The redesign intentionally disables stepping while playing; compare
            # the selected Pause button separately from the disabled Previous glyph.
            region = (
                np.s_[96:286, 96:204] if vertical else np.s_[96:204, 192 if selected else 96 : 286]
            )
            np.testing.assert_array_equal(actual[region], baseline[region])
    for stroke in (1.0, probe.OVERLAY_GEOMETRY.tool_stroke, 2.2):
        for smoothing in (0.0, rig.probe.playback_smoothing, probe.CORNER_SMOOTHING, 1.0):
            path = np.asarray(reset_glyph_path(stroke, smoothing))
            assert np.isfinite(path).all()
            clearance = probe.OVERLAY_GEOMETRY.icon_radius - np.linalg.norm(path, axis=1).max()
            assert clearance == pytest.approx(1.2)


def test_icon_library_preview_substitutes_capsule_and_selected_tool(rig):
    rig.probe.active_tool = rig.state.tool = "rotate"
    rig.probe.preview_icon_library = False
    rig.frame()
    production = rig.pixels[::-1].copy()
    viewport = rig.state.rects["viewport"]
    cx = viewport[0] + (viewport[2] - viewport[0]) * 0.56
    cy = viewport[1] + (viewport[3] - viewport[1]) * 0.55

    rig.probe.preview_icon_library = True
    rig.frame()
    preview = rig.pixels[::-1]
    pixel_scale = preview.shape[1] / imgui.get_io().display_size.x
    tool_region = np.s_[
        round((cy - 36) * pixel_scale) : round((cy + 36) * pixel_scale),
        round((cx - 36) * pixel_scale) : round((cx + 36) * pixel_scale),
    ]
    x0, y0, x1, y1 = rig.state.rects["tools"]
    capsule_region = np.s_[
        round(y0 * pixel_scale) : round(y1 * pixel_scale),
        round(x0 * pixel_scale) : round(x1 * pixel_scale),
    ]

    assert not np.array_equal(production[tool_region], preview[tool_region])
    assert not np.array_equal(production[capsule_region], preview[capsule_region])


def test_geometry_g3_highlight_and_optical_spacing_toggles(monkeypatch):
    rig = Rig(monkeypatch, 1920, 1000)
    try:
        rig.probe.page = "Geometry"
        for _ in range(3):
            rig.frame()
        rig.move(0, 0)
        original = rig.pixels.copy()
        rig.save("geometry-playback.png")
        rig.click("Highlight G3 transitions")
        assert rig.probe.highlight_g3
        rig.move(0, 0)
        highlighted = rig.pixels.copy()
        assert not np.array_equal(original[::-1, 60:1450], highlighted[::-1, 60:1450])
        rig.save("geometry-g3-highlight.png")
        rig.click("Highlight G3 transitions")
        rig.move(0, 0)
        np.testing.assert_array_equal(original[::-1, 60:1450], rig.pixels[::-1, 60:1450])
        rig.click("Optical end spacing")
        assert not rig.probe.optical_capsule_spacing
        rig.move(0, 0)
        rig.save("geometry-spacing-original.png")
        assert (
            rig.probe.capsule_smoothing
            == rig.probe.playback_smoothing
            == rig.probe.tool_smoothing
            == 0.382
        )
        assert optical_end_padding(26, rig.probe.capsule_smoothing) == pytest.approx(
            26.46, abs=0.02
        )
        _, mean, _, side = spacing_metrics(26, 18, rig.probe.capsule_smoothing, True)
        assert mean == pytest.approx(side, abs=0.02)
        assert optical_end_padding(26, 0) == 26
        assert g3_capsule_spans(178, 52, 0) == ()
        assert len(g3_capsule_spans(178, 52, rig.probe.capsule_smoothing)) == 4
        assert len(g3_capsule_spans(178, 52, 1)) == 2
    finally:
        rig.close()


@pytest.mark.parametrize("scale,language", [(1, "en"), (1.5, "zh"), (2.5, "zh")])
def test_capsules_and_narrow_fields_fit_at_fractional_scales(monkeypatch, scale, language):
    rig = Rig(monkeypatch, round(1300 * scale), round(800 * scale), scale, language)
    try:
        for radial_step in (10, probe.OVERLAY_GEOMETRY.radial_step):
            rig.probe.overlay_radial_step = radial_step
            rig.frame()
            a, b = rig.state.rects["playback"], rig.state.rects["tools"]
            assert a[3] - a[1] == pytest.approx((20 + 4 * radial_step) * scale, abs=0.1)
            assert a[3] - a[1] == pytest.approx(b[2] - b[0], abs=0.1)
            assert a[2] - a[0] != pytest.approx(b[3] - b[1])
            for key in ("playback", "tools"):
                x0, y0, x1, y1 = rig.state.rects[key]
                vx0, vy0, vx1, vy1 = rig.state.rects["viewport"]
                assert vx0 <= x0 < x1 <= vx1 and vy0 <= y0 < y1 <= vy1
        rig.save(f"overview-{language}-{scale:g}x.png")
        if scale == 2.5:
            rig.probe.show_icon_bounds = rig.probe.show_state_circles = True
            rig.frame()
            rig.save("construction-zh-2.5x.png")
    finally:
        rig.close()
    narrow = Rig(monkeypatch, round(360 * scale), round(700 * scale), scale, language)
    try:
        for section in ("Inspector", "Control"):
            narrow.state.section = section
            for _ in range(3):
                narrow.frame()
            width = imgui.get_io().display_size.x
            for key, (x0, _, x1, _) in narrow.state.rects.items():
                assert x0 >= 0 and x1 <= width + 0.5, (key, x0, x1, width)
            narrow.save(f"{section.lower()}-narrow-{language}-{scale:g}x.png")
    finally:
        narrow.close()
