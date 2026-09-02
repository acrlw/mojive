"""Window scaling tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.ui import window as window_module
from mojive.ui.app import (
    JOINT_LIMIT_HOVER_GRACE_SECONDS,
    JOINT_LIMIT_LABEL_DELAY_SECONDS,
    MODEL_FILTERS,
    ViewerApp,
    _clipped_overlay_host_rect,
    _compact_status_for_selection,
    _fit_image_rect,
    _FrameRateDisplay,
    _JointLimitHoverState,
    _middle_elide_text,
    _prepare_modal,
    _simulation_timestep,
    _status_message_for_bar,
    _toggle_angle_input,
    _translated_file_filters,
    precise_input_status_hints,
)
from mojive.ui.window import (
    Window,
    _install_glfw_clipboard_callbacks,
    _is_dock_tab_nav_target,
    layout_scale,
    layout_settings_path,
    resolve_context_api,
    resolve_ui_scales,
)
from mojive.ui.window_wgpu import _scissor_rect_for_target


def test_glfw_clipboard_callbacks_use_the_process_wide_api_without_a_window() -> None:
    calls: list[tuple[str, object, object | None]] = []
    platform_io = SimpleNamespace()

    class GlfwApi:
        @staticmethod
        def get_clipboard_string(window):
            calls.append(("get", window, None))
            return b"Mojive"

        @staticmethod
        def set_clipboard_string(window, text):
            calls.append(("set", window, text))

    imgui_api = SimpleNamespace(get_platform_io=lambda: platform_io)
    _install_glfw_clipboard_callbacks(GlfwApi, imgui_api)

    assert platform_io.platform_get_clipboard_text_fn(None) == "Mojive"
    platform_io.platform_set_clipboard_text_fn(None, "viewer")
    assert calls == [("get", None, None), ("set", None, "viewer")]


def test_glfw_clipboard_callback_treats_an_empty_native_clipboard_as_text() -> None:
    platform_io = SimpleNamespace()
    glfw_api = SimpleNamespace(
        get_clipboard_string=lambda _window: None,
        set_clipboard_string=lambda _window, _text: None,
    )
    imgui_api = SimpleNamespace(get_platform_io=lambda: platform_io)

    _install_glfw_clipboard_callbacks(glfw_api, imgui_api)

    assert platform_io.platform_get_clipboard_text_fn(None) == ""


def test_only_a_docked_window_tab_owns_the_suppressed_nav_cursor() -> None:
    assert _is_dock_tab_nav_target(42, 42, True)
    assert not _is_dock_tab_nav_target(41, 42, True)
    assert not _is_dock_tab_nav_target(42, 42, False)
    assert not _is_dock_tab_nav_target(0, 0, True)


def test_joint_limit_hover_delays_label_and_tolerates_short_dropouts() -> None:
    state = _JointLimitHoverState()
    lower = (3, 7, "MIN")
    upper = (3, 7, "MAX")
    available = (lower, upper)

    assert state.update(lower, available, 10.0) is None
    assert (
        state.update(
            lower,
            available,
            10.0 + JOINT_LIMIT_LABEL_DELAY_SECONDS,
        )
        == lower
    )
    assert state.update(None, available, 10.0 + JOINT_LIMIT_LABEL_DELAY_SECONDS + 0.01) == lower
    assert (
        state.update(
            lower,
            available,
            10.0 + JOINT_LIMIT_LABEL_DELAY_SECONDS + JOINT_LIMIT_HOVER_GRACE_SECONDS,
        )
        == lower
    )
    assert state.update(upper, available, 11.0) is None


def test_joint_limit_hover_resets_after_a_real_exit_or_target_removal() -> None:
    state = _JointLimitHoverState()
    lower = (3, 7, "MIN")
    available = (lower,)

    state.update(lower, available, 10.0)
    assert (
        state.update(
            None,
            available,
            10.0 + JOINT_LIMIT_HOVER_GRACE_SECONDS + 0.01,
        )
        is None
    )
    assert state.key is None
    state.update(lower, available, 11.0)
    assert state.update(None, (), 11.01) is None
    assert state.key is None


@pytest.mark.parametrize(
    ("message", "selected", "expected"),
    (
        ("02_prismatic", "02_prismatic", ""),
        ("Selected 02_prismatic", "02_prismatic", ""),
        ("02_prismatic · +0.236 m", "02_prismatic", "+0.236 m"),
        ("Selection cleared", "no selection", ""),
        ("Saved viewport", "02_prismatic", "Saved viewport"),
    ),
)
def test_status_removes_selection_semantics_already_shown(message, selected, expected):
    assert _compact_status_for_selection(message, selected) == expected


@pytest.mark.parametrize(
    ("message", "level", "expected"),
    (
        ("Simulation resumed", "info", ""),
        ("Simulation paused", "info", ""),
        ("Stepped 1 frame(s)", "info", ""),
        ("Saved scene.xml", "info", "Saved scene.xml"),
        ("Viewport capture failed", "error", "Viewport capture failed"),
        ("02_prismatic · +0.236 m", "warning", "+0.236 m"),
    ),
)
def test_status_bar_keeps_only_actions_and_diagnostics(message, level, expected):
    assert _status_message_for_bar(message, "02_prismatic", level) == expected


def test_status_bar_uses_the_adapter_simulation_timestep() -> None:
    adapter = SimpleNamespace(timestep=lambda: 0.002)

    assert _simulation_timestep(adapter) == pytest.approx(0.002)
    assert _simulation_timestep(None, loading=True) == 0.0


def test_precise_input_status_hints_match_the_implemented_shortcuts() -> None:
    translate = {"Apply": "应用", "Cancel": "取消", "Switch angle unit": "切换角度单位"}.get

    linear = precise_input_status_hints(SimpleNamespace(unit="m"), translate)
    angular = precise_input_status_hints(SimpleNamespace(unit="°"), translate)

    assert [(hint.control, hint.label) for hint in linear] == [
        ("Enter", "应用"),
        ("Esc", "取消"),
    ]
    assert [(hint.control, hint.label) for hint in angular] == [
        ("Enter", "应用"),
        ("Esc", "取消"),
        ("U", "切换角度单位"),
    ]


@pytest.mark.parametrize(
    ("ui_scale", "framebuffer_scale", "expected"),
    (
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (2.0, 2.0, 1.0),
        (1.5, 1.0, 1.5),
    ),
)
def test_layout_scale_separates_desktop_and_framebuffer_scaling(
    ui_scale: float, framebuffer_scale: float, expected: float
) -> None:
    assert layout_scale(ui_scale, framebuffer_scale) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("content_scale", "framebuffer_scale", "override", "expected"),
    (
        (1.0, 1.0, None, (1.0, 1.0)),
        (2.0, 1.0, None, (2.0, 2.0)),
        (2.0, 2.0, None, (2.0, 1.0)),
        (2.0, 2.0, 2.0, (4.0, 2.0)),
        (2.0, 1.0, 2.0, (2.0, 2.0)),
    ),
)
def test_explicit_ui_scale_controls_layout_space(
    content_scale: float,
    framebuffer_scale: float,
    override: float | None,
    expected: tuple[float, float],
) -> None:
    assert resolve_ui_scales(content_scale, framebuffer_scale, override) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        ("auto", "native"),
        ("", "native"),
        ("native", "native"),
        ("glfw", "native"),
        ("egl", "egl"),
    ),
)
def test_resolve_context_api(requested: str, expected: str) -> None:
    assert resolve_context_api(requested) == expected


def test_resolve_context_api_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported MOJIVE_GL"):
        resolve_context_api("gles")


def test_status_frame_rate_is_smoothed_and_rate_limited() -> None:
    display = _FrameRateDisplay()

    first = display.update(1.0 / 30.0)
    for _ in range(6):
        assert display.update(1.0 / 30.0) == first
    changed = display.update(1.0 / 30.0)

    assert 30.0 < changed < 60.0
    previous = changed
    for _ in range(7):
        assert display.update(1.0 / 60.0) == previous


def test_dynamic_popover_title_elides_the_middle() -> None:
    value = "Rotate 01_revolute_y_with_a_long_joint_name"
    shown = _middle_elide_text(value, 18.0, len)

    assert len(shown) <= 18
    assert shown.startswith("Rotate")
    assert shown.endswith("name")
    assert "…" in shown


def test_viewport_recording_streams_and_finalizes_frames(monkeypatch) -> None:
    import mojive.recording as recording

    events = []

    class Recorder:
        def __init__(self, path, size, fps):
            self.path, self.size, self.fps = path, size, fps
            self.frames = 0
            self.closed = False

        def append(self, frame):
            assert frame.shape == (1, 2, 3)
            self.frames += 1

        def close(self):
            self.closed = True

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)
    target = SimpleNamespace(
        width=2,
        height=1,
        read_color=lambda flip: np.zeros((1, 2, 4), np.uint8),
    )
    app = ViewerApp.__new__(ViewerApp)
    app.backend = SimpleNamespace(target=target)
    app.localizer = SimpleNamespace(text=lambda value: value)
    app.session = SimpleNamespace(
        report_message=lambda message, level: events.append((message, level))
    )
    app._viewport_recorder = None
    app._viewport_recording_path = None
    app._viewport_record_elapsed = 0.0

    app._toggle_viewport_recording()
    recorder = app._viewport_recorder
    assert recorder is not None and recorder.size == (2, 1) and recorder.fps == 30.0
    app._record_viewport_frame(0.0)
    assert recorder.frames == 1
    app._toggle_viewport_recording()

    assert recorder.closed
    assert app._viewport_recorder is None
    assert events[-1][1] == "success"


def test_file_dialog_filters_translate_descriptions_without_touching_globs() -> None:
    translated = _translated_file_filters(MODEL_FILTERS, lambda value: f"zh:{value}")
    assert translated[::2] == [f"zh:{value}" for value in MODEL_FILTERS[::2]]
    assert translated[1::2] == MODEL_FILTERS[1::2]


def test_layout_settings_path_honors_exact_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "custom-layout.ini"
    monkeypatch.setenv("MOJIVE_IMGUI_INI", str(target))
    assert layout_settings_path() == target


def test_layout_settings_path_honors_config_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MOJIVE_IMGUI_INI", raising=False)
    monkeypatch.setenv("MOJIVE_CONFIG_DIR", str(tmp_path))
    assert layout_settings_path() == tmp_path / "imgui.ini"


def test_reset_layout_rebuilds_and_persists_immediately(monkeypatch, tmp_path) -> None:
    target = tmp_path / "nested" / "imgui.ini"
    events: list[object] = []
    fake_imgui = SimpleNamespace(
        load_ini_settings_from_memory=lambda value: events.append(("load", value)),
        save_ini_settings_to_disk=lambda value: events.append(("save", value)),
    )
    monkeypatch.setattr(window_module, "imgui", fake_imgui)

    window = Window.__new__(Window)
    window.config = SimpleNamespace(docking=True, ini_path=str(target))
    window.dockspace_id = 42
    window._ini_existed = True
    window._layout_done = True
    window._build_default_layout = lambda: events.append("build")

    window.reset_layout()

    assert events == [("load", ""), "build", ("save", str(target))]
    assert target.parent.is_dir()
    assert window._ini_existed is True


def test_viewport_image_is_aspect_fitted_while_render_target_resize_is_pending() -> None:
    assert _fit_image_rect((10.0, 20.0), (800.0, 800.0), (1600, 900)) == pytest.approx(
        (10.0, 195.0, 800.0, 450.0)
    )
    assert _fit_image_rect((10.0, 20.0), (800.0, 450.0), (1600, 900)) == pytest.approx(
        (10.0, 20.0, 800.0, 450.0)
    )


def test_wgpu_scissor_is_scaled_and_clamped_to_a_resized_target() -> None:
    scissor = _scissor_rect_for_target(
        (0.0, 40.0, 3840.0, 1978.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (3840, 1938),
        (1600, 1000),
    )
    assert scissor == (0, 20, 1600, 980)


def test_wgpu_scissor_discards_clips_outside_the_resized_target() -> None:
    assert (
        _scissor_rect_for_target(
            (4000.0, 2000.0, 4100.0, 2100.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (3840, 1938),
            (1600, 1000),
        )
        is None
    )


def test_precise_angle_input_toggles_units_without_changing_the_angle() -> None:
    radians, unit = _toggle_angle_input(180.0, "degrees")
    assert unit == "radians"
    assert radians == pytest.approx(np.pi)

    degrees, unit = _toggle_angle_input(radians, unit)
    assert unit == "degrees"
    assert degrees == pytest.approx(180.0)


def test_modal_layout_tracks_the_current_viewport_and_enforces_readable_width(
    monkeypatch,
) -> None:
    positions = []
    constraints = []
    center = imgui.ImVec2(700.0, 450.0)
    monkeypatch.setattr(
        imgui,
        "get_main_viewport",
        lambda: SimpleNamespace(get_center=lambda: center, work_size=imgui.ImVec2(800.0, 600.0)),
    )
    monkeypatch.setattr(imgui, "set_next_window_pos", lambda *args: positions.append(args))
    monkeypatch.setattr(
        imgui,
        "set_next_window_size_constraints",
        lambda *args: constraints.append(args),
    )

    _prepare_modal(440.0)

    position, condition, pivot = positions[0]
    assert (position.x, position.y) == pytest.approx((700.0, 450.0))
    assert condition == imgui.Cond_.always.value
    assert (pivot.x, pivot.y) == pytest.approx((0.5, 0.5))
    minimum, maximum = constraints[0]
    assert (minimum.x, minimum.y) == pytest.approx((440.0, 0.0))
    assert maximum.x == pytest.approx(440.0)
    assert maximum.y == pytest.approx(568.0)


def test_modal_layout_scales_authored_width_and_clamps_to_work_area(monkeypatch) -> None:
    constraints = []
    viewport = SimpleNamespace(
        get_center=lambda: imgui.ImVec2(400.0, 300.0),
        work_size=imgui.ImVec2(800.0, 600.0),
    )
    monkeypatch.setattr(imgui, "get_main_viewport", lambda: viewport)
    monkeypatch.setattr(imgui, "set_next_window_pos", lambda *_args: None)
    monkeypatch.setattr(
        imgui,
        "set_next_window_size_constraints",
        lambda *args: constraints.append(args),
    )

    _prepare_modal(360.0, 2.25)

    minimum, maximum = constraints[0]
    assert (minimum.x, minimum.y) == pytest.approx((728.0, 0.0))
    assert (maximum.x, maximum.y) == pytest.approx((728.0, 528.0))


def test_overlay_host_is_the_padded_content_intersection_with_the_viewport() -> None:
    viewport = (100.0, 50.0, 200.0, 120.0)

    assert _clipped_overlay_host_rect(
        viewport,
        (70.0, 60.0, 330.0, 120.0),
        4.0,
    ) == pytest.approx((100.0, 56.0, 200.0, 68.0))
    assert _clipped_overlay_host_rect(
        viewport,
        (120.0, 80.0, 180.0, 110.0),
        4.0,
    ) == pytest.approx((116.0, 76.0, 68.0, 38.0))
    assert (
        _clipped_overlay_host_rect(
            viewport,
            (10.0, 10.0, 20.0, 20.0),
            4.0,
        )
        is None
    )
