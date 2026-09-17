"""Window scaling tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import CaptureSurface, RecordingPhase
from mojive.config import ViewportOverlayConfig
from mojive.ui import window as window_module
from mojive.ui.app import (
    JOINT_LIMIT_HOVER_GRACE_SECONDS,
    JOINT_LIMIT_LABEL_DELAY_SECONDS,
    PRECISE_GIZMO_HINT_DELAY_SECONDS,
    ViewerApp,
    _clipped_overlay_host_rect,
    _fit_image_rect,
    _FrameRateDisplay,
    _GizmoHintHoverState,
    _JointLimitHoverState,
    _middle_elide_text,
    _model_filters,
    _prepare_modal,
    _simulation_timestep,
    _toggle_angle_input,
    _translated_file_filters,
    precise_input_status_hints,
)
from mojive.ui.clipping import scissor_rect_for_target
from mojive.ui.gestures import Claim, GestureRouter, InputState
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.viewport_widgets import DEFAULT_VIEWPORT_LABELS, ToolHint
from mojive.ui.window import (
    Window,
    _install_glfw_clipboard_callbacks,
    _is_dock_tab_nav_target,
    layout_scale,
    layout_settings_path,
    resolve_context_api,
    resolve_ui_scales,
)


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


def test_precise_gizmo_status_hint_requires_an_uninterrupted_hover_delay() -> None:
    state = _GizmoHintHoverState()

    assert not state.update(True, 10.0)
    assert not state.update(True, 10.0 + PRECISE_GIZMO_HINT_DELAY_SECONDS - 0.01)
    assert state.update(True, 10.0 + PRECISE_GIZMO_HINT_DELAY_SECONDS)
    assert not state.update(False, 10.0 + PRECISE_GIZMO_HINT_DELAY_SECONDS + 0.01)
    assert state.entered_at is None


def test_status_bar_uses_the_adapter_simulation_timestep() -> None:
    adapter = SimpleNamespace(timestep=lambda: 0.002)

    assert _simulation_timestep(adapter) == pytest.approx(0.002)
    assert _simulation_timestep(None, loading=True) == 0.0


@pytest.mark.parametrize(
    ("claim", "held", "using", "keyboard_using", "perturbing", "widget_active", "expected"),
    (
        (Claim.NONE, False, False, False, False, False, True),
        (Claim.CAMERA, True, False, False, False, False, True),
        (Claim.VIEW_CUBE, True, False, False, False, False, True),
        (Claim.UI, True, False, False, False, False, True),
        (Claim.UI, True, False, False, False, True, False),
        (Claim.OBJECT_GIZMO, True, False, False, False, False, False),
        (Claim.PERTURB, True, False, False, False, False, False),
        (Claim.NONE, False, True, False, False, False, False),
        (Claim.NONE, False, False, True, False, False, False),
        (Claim.NONE, False, False, False, True, False, False),
    ),
)
def test_clear_selection_distinguishes_navigation_from_target_edits(
    monkeypatch, claim, held, using, keyboard_using, perturbing, widget_active, expected
):
    app = ViewerApp.__new__(ViewerApp)
    app.session = SimpleNamespace(
        paused=True,
        state_take_playing=False,
        selected_node=object(),
        perturb=SimpleNamespace(active=perturbing),
    )
    app.router = SimpleNamespace(claim=claim, held=held)
    app.gizmo = SimpleNamespace(using=using, keyboard_using=keyboard_using)
    monkeypatch.setattr(imgui, "is_any_item_active", lambda: widget_active)
    monkeypatch.setattr(
        imgui, "get_current_context", lambda: SimpleNamespace(active_id_window=None)
    )

    assert app._selection_clear_enabled() is expected
    app.session.selected_node = None
    assert not app._selection_clear_enabled()
    app.session.selected_node = object()
    app.session.paused = False
    assert not app._selection_clear_enabled()
    app.session.paused = True
    app.session.state_take_playing = True
    assert not app._selection_clear_enabled()


@pytest.mark.parametrize("moving", (False, True))
def test_background_focus_capture_is_not_an_active_widget_edit(monkeypatch, moving):
    app = ViewerApp.__new__(ViewerApp)
    app.session = SimpleNamespace(
        paused=True,
        state_take_playing=False,
        selected_node=object(),
        perturb=SimpleNamespace(active=False),
    )
    app.router = SimpleNamespace(held=True, claim=Claim.CAMERA)
    app.gizmo = SimpleNamespace(using=False, keyboard_using=False)
    window = SimpleNamespace(move_id=123)
    context = SimpleNamespace(
        active_id_window=window,
        active_id=123,
        moving_window=window if moving else None,
    )
    monkeypatch.setattr(imgui, "is_any_item_active", lambda: True)
    monkeypatch.setattr(imgui, "get_current_context", lambda: context)

    assert app._selection_clear_enabled() is (not moving)


def _input_app(monkeypatch):
    app = ViewerApp.__new__(ViewerApp)
    for attribute in (
        "_pending_document_action",
        "_pending_pose_save",
        "_model_dialog",
        "_scene_dialog",
        "_resource_dialog",
        "_texture_dialog",
        "_geometry_resource_dialog",
        "_model_asset_dialog",
        "_resource_repair_dialog",
        "_precise_gizmo_edit",
    ):
        setattr(app, attribute, None)
    app._show_model_load_error = False
    app._open_resource_repair_popup = False
    app._open_rename_popup = False
    app._popup_owned_frame = False
    app._consume_scene_pointer_until_release = False
    app.router = GestureRouter()
    app.gizmo = SimpleNamespace(using=False)
    app.window = SimpleNamespace(style_scale=1.0)
    io = SimpleNamespace(
        want_capture_keyboard=True,
        want_text_input=False,
        mouse_down=[False] * 3,
        mouse_pos=SimpleNamespace(x=0.0, y=0.0),
    )
    monkeypatch.setattr(imgui, "get_io", lambda: io)
    monkeypatch.setattr(imgui, "get_current_context", lambda: None)
    monkeypatch.setattr(imgui, "is_popup_open", lambda *_args: False)
    return app, io


def test_scene_input_ignores_broad_imgui_keyboard_capture(monkeypatch):
    """A focused panel must not reserve policy keys unless it is editing text."""
    app, io = _input_app(monkeypatch)

    assert not app._scene_input_blocked()
    io.want_text_input = True
    assert app._scene_input_blocked()


@pytest.mark.parametrize("surface", ["overlay", "notice"])
@pytest.mark.parametrize(
    "press",
    [
        InputState(left=True, over_viewport=True),
        InputState(left=True, over_viewport=True, over_view_cube=True),
        InputState(left=True, over_viewport=True, ctrl=True, has_selection=True),
        InputState(left=True, over_viewport=True, gizmo_available=True, gizmo_hovered=True),
    ],
)
def test_scene_drag_retains_ownership_across_overlay_and_notice(monkeypatch, surface, press):
    app, io = _input_app(monkeypatch)
    app.viewport_overlays = ViewportOverlayConfig(movable=True)
    rect = (10.0, 20.0, 110.0, 60.0)
    if surface == "overlay":
        app._tool_widget_rect = rect
    else:
        app._status_notice_bounds = rect
    claim = app.router.update(press)
    io.mouse_down[0] = True
    io.mouse_pos.x, io.mouse_pos.y = 12.0, 40.0
    assert not app._scene_input_blocked()
    assert app.router.update(press) == claim
    # Explicit prompts still interrupt owned gestures.
    app._popup_owned_frame = True
    assert app._scene_input_blocked()
    app._popup_owned_frame = False
    app.router.abort()
    assert app._scene_input_blocked()


@pytest.mark.parametrize("panel", ("Viewport", "Joints", "Inspector"))
def test_selection_status_is_composed_with_panel_hints_but_yields_to_input(panel):
    app = ViewerApp.__new__(ViewerApp)
    app._status_panel = panel
    app._panel_status_hints = (ToolHint("mouse", "right", "Copy name", hint_id="panel.copy"),)
    app._precise_gizmo_edit = None
    app._gizmo_hint_hover = SimpleNamespace(visible=False)
    app._scene_input_blocked = lambda: False
    app._has_scene_content = lambda: True
    app._selection_clear_enabled = lambda: True
    app._context_tool_hint_variant = lambda: "ready"
    app._viewport_labels = DEFAULT_VIEWPORT_LABELS
    app.input_bindings = DEFAULT_INPUT_BINDINGS
    app.session = SimpleNamespace(
        can_step_back=False, adapter=SimpleNamespace(caps=SimpleNamespace(perturb=True))
    )
    app.interactions = SimpleNamespace(perturb=True)
    app.localizer = SimpleNamespace(text=lambda value: value)
    app.tool_hints = SimpleNamespace(resolve=lambda defaults, *, surface: defaults)

    hints = app._status_tool_hints(loading=False)
    assert hints[0].hint_id == "selection.clear"
    if panel != "Viewport":
        assert hints[1:] == app._panel_status_hints
    else:
        assert all(hint.hint_id != "gizmo.type_value" for hint in hints)
    app._status_panel = "Joints"
    app._gizmo_hint_hover.visible = True
    hints = app._status_tool_hints(loading=False)
    assert [hint.hint_id for hint in hints] == ["selection.clear", "gizmo.type_value"]
    app._gizmo_hint_hover.visible = False
    app._status_panel = panel
    assert app._status_tool_hints(loading=True) == ()
    app._scene_input_blocked = lambda: True
    assert app._status_tool_hints(loading=False) == ()
    app._precise_gizmo_edit = SimpleNamespace(unit="m")
    hints = app._status_tool_hints(loading=False)
    assert [(hint.control, hint.label) for hint in hints] == [
        ("Enter", "Apply"),
        ("Esc", "Cancel"),
    ]


def test_status_context_changes_only_on_click_and_ignores_transient_windows(monkeypatch):
    app = ViewerApp.__new__(ViewerApp)
    app._status_panel = "Viewport"
    app._panel_status_hints = ()
    app._consume_scene_pointer_until_release = False
    control, joints = ("copy-control",), ("copy-joint",)
    ctx = SimpleNamespace(
        status_hints_by_panel={"Control": control, "Joints": joints, "Inspector": ()}
    )
    pointer = {"window": "Control", "button": None}
    monkeypatch.setattr(imgui, "is_mouse_clicked", lambda button: pointer["button"] == button)
    monkeypatch.setattr(
        imgui,
        "get_current_context",
        lambda: SimpleNamespace(
            hovered_window=SimpleNamespace(root_window=SimpleNamespace(name=pointer["window"])),
            hovered_id=0,
        ),
    )

    app._update_status_context(ctx)
    assert app._status_panel == "Viewport"
    pointer["button"] = 0
    app._update_status_context(ctx)
    assert app._status_panel == "Control"
    assert app._panel_status_hints == control
    pointer.update(window="Joints", button=None)
    app._update_status_context(ctx)
    assert app._panel_status_hints == control
    pointer.update(window="关节###Joints", button=1)
    app._update_status_context(ctx)
    assert app._status_panel == "Joints"
    assert app._panel_status_hints == joints
    pointer.update(window="##Menu_00", button=0)
    app._update_status_context(ctx)
    assert app._panel_status_hints == joints
    pointer["window"] = "Inspector"
    app._update_status_context(ctx)
    assert app._status_panel == "Inspector"
    assert app._panel_status_hints == ()
    pointer["window"] = "视口###Viewport"
    app._consume_scene_pointer_until_release = True
    app._update_status_context(ctx)
    assert app._status_panel == "Inspector"
    app._consume_scene_pointer_until_release = False
    app._update_status_context(ctx)
    assert app._status_panel == "Viewport"
    pointer["window"] = "Control"
    app._update_status_context(ctx)
    pointer["button"] = None
    del ctx.status_hints_by_panel["Control"]
    app._update_status_context(ctx)
    assert app._status_panel == "Viewport"


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


def test_status_render_rate_can_drop_below_ten_fps() -> None:
    display = _FrameRateDisplay()
    for _ in range(20):
        display.update(0.5)
    assert display.value == pytest.approx(2.0, rel=1e-5)


def test_dynamic_popover_title_elides_the_middle() -> None:
    value = "Rotate 01_revolute_y_with_a_long_joint_name"
    shown = _middle_elide_text(value, 18.0, len)

    assert len(shown) <= 18
    assert shown.startswith("Rotate")
    assert shown.endswith("name")
    assert "…" in shown


def test_viewport_recording_streams_and_finalizes_frames(monkeypatch, capsys) -> None:
    import mojive.capture.recording as recording

    events = []

    class Recorder:
        def __init__(self, path, size, fps, **encoding):
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
    app.window = SimpleNamespace(points_to_pixels=lambda rect: rect)
    app._viewport_rect = (0, 0, 2, 1)
    app._scene_capture = SimpleNamespace(read=lambda *args, **kwargs: np.zeros((1, 2, 3), np.uint8))
    app._camera_view = lambda: None
    app.localizer = SimpleNamespace(text=lambda value: value)
    app.session = SimpleNamespace(
        report_message=lambda message, level, **kwargs: events.append((message, level))
    )
    app._viewport_recorder = None
    app._viewport_recording_path = None
    app._viewport_record_elapsed = 0.0

    app.start_recording(surface=CaptureSurface.SCENE, countdown=0)
    assert app._viewport_recorder is None
    app._record_viewport_frame(0.0)
    assert app._viewport_recorder.started.wait(2)
    recorder = app._viewport_recorder.recorder
    assert recorder is not None and recorder.size == (2, 1) and recorder.fps == 60.0
    assert recorder.frames == 1
    assert app.recording.phase is RecordingPhase.RECORDING
    assert app.pause_recording()
    app._record_viewport_frame(1.0)
    assert recorder.frames == 1
    assert app.resume_recording()
    app._record_viewport_frame(0.0)
    assert app.recording.frames == 2
    app._toggle_viewport_recording()
    assert app.recording.phase is RecordingPhase.FINALIZING
    app.stop_recording()
    assert recorder.frames == 2

    assert recorder.closed
    assert app._viewport_recorder is None
    assert events[-1][1] == "success"
    assert capsys.readouterr().out == f"Saved video to {recorder.path.resolve()}\n"


@pytest.mark.parametrize("duration,expected", ((5.0, 3.0), (8.0, 8.0), (10.0, 10.0), (None, None)))
def test_status_bridge_preserves_extended_and_persistent_message_lifetimes(
    monkeypatch, duration, expected
):
    from mojive.ui.messages import OutputBuffer

    monkeypatch.setattr("mojive.ui.messages.time.monotonic", lambda: 100.0)
    app = ViewerApp.__new__(ViewerApp)
    app._seen_message_revision = 0
    app.output = OutputBuffer()
    app.viewport_overlays = ViewportOverlayConfig(status_duration=3.0)
    app.session = SimpleNamespace(
        message_revision=1,
        last_message="Saved video to /tmp/video.mp4",
        last_message_level="success",
        last_message_duration=duration,
        last_message_copy_text="/tmp/video.mp4",
    )
    app._sync_session_status()
    app._sync_session_status()
    assert len(app.output.entries()) == 1
    message = app.output.entries()[0]
    assert message.copy_text == "/tmp/video.mp4"
    assert app.output.active_status(100.0 + (expected or 1000) - 0.01) == message
    if expected is not None:
        assert app.output.active_status(100.0 + expected) is None
    assert app.output.entries() == (message,)


def test_failed_video_finalization_never_prints_a_saved_receipt(tmp_path, capsys):
    app = ViewerApp.__new__(ViewerApp)
    app.localizer = SimpleNamespace(text=lambda value: value)
    messages = []
    app.session = SimpleNamespace(report_message=lambda text, **kwargs: messages.append(text))
    app.start_recording(tmp_path / "broken.mp4", countdown=0)

    def fail():
        raise RuntimeError("encoder finalization failed")

    app._viewport_recorder = SimpleNamespace(close=fail)
    app._viewport_recording_frames = 1
    with pytest.raises(RuntimeError, match="encoder finalization failed"):
        app.stop_recording(raise_on_error=True)
    assert not capsys.readouterr().out
    assert messages == ["Recording failed: encoder finalization failed"]
    assert not app.recording.active


def test_presented_capture_surface_flips_and_crops_viewport_pixels() -> None:
    # Native window readback follows OpenGL convention: its first row is the bottom row.
    bottom_up = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    app = ViewerApp.__new__(ViewerApp)
    app.window = SimpleNamespace(points_to_pixels=lambda value: value)
    app._viewport_rect = (1.0, 1.0, 3.0, 2.0)

    window = app._surface_image(CaptureSurface.WINDOW, bottom_up)
    viewport = app._surface_image(CaptureSurface.VIEWPORT, bottom_up)

    assert np.array_equal(window, bottom_up[::-1])
    assert np.array_equal(viewport, bottom_up[::-1][1:3, 1:4])


def test_file_dialog_filters_translate_descriptions_without_touching_globs() -> None:
    from mojive.adapters.base import AdapterCaps

    filters = _model_filters(AdapterCaps(asset_loading=True, model_formats=(".urdf",)))
    translated = _translated_file_filters(filters, lambda value: f"zh:{value}")
    assert translated[::2] == [f"zh:{value}" for value in filters[::2]]
    assert translated[1::2] == filters[1::2]


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


def test_application_layout_reset_restores_viewport_capsule_positions() -> None:
    events: list[object] = []
    app = ViewerApp.__new__(ViewerApp)
    app.window = SimpleNamespace(
        config=SimpleNamespace(ini_path="layout.ini"),
        reset_layout=lambda: events.append("layout"),
    )
    app.localizer = SimpleNamespace(set_preferences=lambda value: events.append(value))
    app.viewport_overlays = ViewportOverlayConfig(
        playback_scale=1.2,
        tool_scale=0.9,
        movable=True,
        playback_position=(0.2, 0.3),
        tool_position=(0.8, 0.7),
    )
    app._overlay_drag_kind = "tools"

    app.reset_layout()

    assert events[0] == "layout"
    assert app._overlay_drag_kind == ""
    assert app.viewport_overlays.playback_position is None
    assert app.viewport_overlays.tool_position is None
    assert app.viewport_overlays.playback_scale == pytest.approx(1.2)
    assert app.viewport_overlays.tool_scale == pytest.approx(1.2)
    assert events[1]["viewport_overlays"]["playback_position"] is None
    assert events[1]["viewport_overlays"]["tool_position"] is None


def test_retained_debug_primitives_count_as_scene_content() -> None:
    app = ViewerApp.__new__(ViewerApp)
    app.session = SimpleNamespace(
        source=SimpleNamespace(
            instance_count=0,
            lights=SimpleNamespace(lights=()),
            cameras=(),
        )
    )
    app.backend = SimpleNamespace(debug=SimpleNamespace(primitives=0))

    assert not app._has_scene_content()
    app.backend.debug.primitives = 3
    assert app._has_scene_content()


def test_viewport_image_is_aspect_fitted_while_render_target_resize_is_pending() -> None:
    assert _fit_image_rect((10.0, 20.0), (800.0, 800.0), (1600, 900)) == pytest.approx(
        (10.0, 195.0, 800.0, 450.0)
    )
    assert _fit_image_rect((10.0, 20.0), (800.0, 450.0), (1600, 900)) == pytest.approx(
        (10.0, 20.0, 800.0, 450.0)
    )


def test_scissor_is_scaled_and_clamped_to_a_resized_target() -> None:
    scissor = scissor_rect_for_target(
        (0.0, 40.0, 3840.0, 1978.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (3840, 1938),
        (1600, 1000),
    )
    assert scissor == (0, 20, 1600, 980)


def test_scissor_discards_clips_outside_the_resized_target() -> None:
    assert (
        scissor_rect_for_target(
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


def test_recording_countdown_uses_wall_time_and_waits_for_menu_dismissal(monkeypatch, tmp_path):
    import mojive.ui.app.capture as app_module
    from mojive import RecordingConfig

    now = [10.0]
    monkeypatch.setattr(app_module.time, "monotonic", lambda: now[0])
    app = ViewerApp.__new__(ViewerApp)
    app.recording_config = RecordingConfig(countdown=3)
    app.backend = SimpleNamespace()
    app._popup_owned_frame = False
    app._model_load_future = None
    app.localizer = SimpleNamespace(text=lambda value: value)
    app.session = SimpleNamespace(report_message=lambda *args, **kwargs: None)
    path = tmp_path / "countdown.mp4"
    assert app.start_recording(path) == path
    assert app.recording.phase is RecordingPhase.COUNTDOWN
    assert app.recording.countdown_remaining == 3
    assert app.recording.fps == 60
    assert app.recording.surface is CaptureSurface.VIEWPORT
    now[0] += 2.99
    app._advance_recording_countdown()
    assert app.recording.phase is RecordingPhase.COUNTDOWN
    now[0] += 1
    app._popup_owned_frame = True
    app._advance_recording_countdown()
    assert app.recording.phase is RecordingPhase.COUNTDOWN
    app._popup_owned_frame = False
    app._advance_recording_countdown()
    assert app.recording.phase is RecordingPhase.RECORDING
    assert app.recording.countdown_remaining == 0
    assert app._viewport_recorder is None
    assert not path.exists()
    assert app.stop_recording() is None
    assert app.start_recording(path, countdown=0) == path
    assert app.recording.phase is RecordingPhase.COUNTDOWN
    assert app.stop_recording() is None
    assert not path.exists()
    assert app.recording.phase is RecordingPhase.IDLE


@pytest.mark.parametrize("run_simulation", (False, True))
@pytest.mark.parametrize("clock_control", (False, True))
def test_video_start_action_follows_first_encoded_frame_and_never_repeats(
    monkeypatch, tmp_path, run_simulation, clock_control
):
    from mojive import RecordingConfig
    from mojive import commands as cmd
    from mojive.capture import recording

    events = []

    class Recorder:
        def __init__(self, path, size, fps, **encoding):
            self.size = size

        def append(self, frame):
            events.append("frame")

        def close(self):
            events.append("close")

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)
    app = ViewerApp.__new__(ViewerApp)
    app.recording_config = RecordingConfig(countdown=0, run_simulation=run_simulation)
    app._surface_image = lambda *args: np.zeros((2, 2, 3), np.uint8)
    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(clock_control=clock_control)),
        submit=lambda command: events.append(type(command).__name__) or cmd.CommandResult.good(""),
        report_message=lambda *args, **kwargs: None,
    )
    app.start_recording(tmp_path / "video.mp4")
    app._advance_recording_countdown()
    assert events == []
    app._finish_capture_and_recording(None, 0)
    assert app._viewport_recorder.started.wait(2)
    app._poll_recording()
    assert events == (["frame", "Play"] if run_simulation and clock_control else ["frame"])
    assert app.pause_recording()
    assert app.resume_recording()
    app._finish_capture_and_recording(None, 0)
    assert app.recording.frames == 2
    assert events.count("Play") == int(run_simulation and clock_control)
    app.stop_recording(report=False)
    assert events[-1] == "close"
    assert events.count("frame") == 2
    assert not any(name in events for name in ("Pause", "Reset", "PlayStateTake"))


@pytest.mark.parametrize("failure", ("cancel", "encoder", "simulation"))
def test_failed_or_canceled_video_start_does_not_run_an_unrecorded_action(
    monkeypatch, tmp_path, failure
):
    from mojive import RecordingConfig
    from mojive import commands as cmd
    from mojive.capture import recording

    events = []

    class Recorder:
        def __init__(self, path, size, fps, **encoding):
            self.size = size

        def append(self, frame):
            if failure == "encoder":
                raise RuntimeError("encoder unavailable")
            events.append("frame")

        def close(self):
            events.append("close")

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)
    app = ViewerApp.__new__(ViewerApp)
    app.recording_config = RecordingConfig(countdown=0, run_simulation=True)
    app._surface_image = lambda *args: np.zeros((2, 2, 3), np.uint8)
    app.localizer = SimpleNamespace(text=lambda value: value)
    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(clock_control=True)),
        submit=lambda command: (
            events.append(type(command).__name__) or cmd.CommandResult.bad("simulation unavailable")
        ),
        report_message=lambda message, **kwargs: events.append(message),
    )
    app.start_recording(tmp_path / "video.mp4")
    if failure == "cancel":
        app.stop_recording(report=False)
    else:
        app._advance_recording_countdown()
        app._finish_capture_and_recording(None, 0)
        recorder = app._viewport_recorder
        if recorder is not None:
            assert recorder.started.wait(2)
            app._poll_recording()
            if app.recording.active:
                assert recorder._finished.wait(2)
                app._poll_recording()
    assert not app.recording.active and not app._recording_run_simulation
    if failure == "simulation":
        assert events == ["frame", "Play", "close", "simulation unavailable"]
    else:
        assert "Play" not in events


@pytest.mark.parametrize("rate_control", ("quality", "bitrate"))
def test_encoding_settings_are_frozen_at_start_and_refreshed_for_the_next_video(
    monkeypatch, tmp_path, rate_control
):
    from dataclasses import replace

    from mojive import RecordingConfig
    from mojive.capture import recording

    encoded = []

    class Recorder:
        def __init__(self, path, size, fps, **encoding):
            self.size = size
            encoded.append(encoding)

        def append(self, frame):
            pass

        def close(self):
            pass

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)
    app = ViewerApp.__new__(ViewerApp)
    app.recording_config = RecordingConfig(
        countdown=0,
        rate_control=rate_control,
        crf=18,
        bitrate_mbps=6.5,
        encoder_preset="fast",
        pixel_format="yuv444p",
    )
    app._surface_image = lambda *args: np.zeros((2, 2, 3), np.uint8)
    app.session = SimpleNamespace(report_message=lambda *args, **kwargs: None)
    app.start_recording(tmp_path / "first.mp4")
    app.set_recording_config(
        replace(
            app.recording_config,
            crf=35,
            bitrate_mbps=1,
            encoder_preset="slow",
            pixel_format="yuv420p",
        ),
        persist=False,
    )
    app._advance_recording_countdown()
    app._finish_capture_and_recording(None, 0)
    assert encoded == [
        {
            "crf": 18 if rate_control == "quality" else None,
            "bitrate": 6_500_000 if rate_control == "bitrate" else None,
            "preset": "fast",
            "pixel_format": "yuv444p",
        }
    ]
    assert app.pause_recording() and app.resume_recording()
    app._finish_capture_and_recording(None, 0)
    assert len(encoded) == 1
    app.stop_recording(report=False)
    app.start_recording(tmp_path / "second.mp4")
    app._advance_recording_countdown()
    app._finish_capture_and_recording(None, 0)
    assert encoded[-1] == {
        "crf": 35 if rate_control == "quality" else None,
        "bitrate": 1_000_000 if rate_control == "bitrate" else None,
        "preset": "slow",
        "pixel_format": "yuv420p",
    }
    app.stop_recording(report=False)


def test_recording_finalization_keeps_ui_available_until_worker_finishes(tmp_path, capsys):
    import threading

    from mojive.capture.video_queue import BufferedVideoRecorder

    entered, release = threading.Event(), threading.Event()
    messages = []

    class Recorder:
        size = (2, 2)

        def close(self):
            entered.set()
            assert release.wait(2)

    app = ViewerApp.__new__(ViewerApp)
    app.localizer = SimpleNamespace(text=lambda value: value)
    app.session = SimpleNamespace(report_message=lambda message, **kw: messages.append(message))
    app._viewport_recorder = BufferedVideoRecorder(Recorder())
    app._viewport_recording_path = tmp_path / "video.mp4"
    app._viewport_recording_phase = RecordingPhase.PAUSED
    app._viewport_recording_frames = 1
    recorder = app._viewport_recorder
    try:
        path = app._request_recording_stop()
        assert entered.wait(1)
        app._poll_recording()
        assert app.recording.phase is RecordingPhase.FINALIZING
        assert app._request_recording_stop() == path
        assert not app.pause_recording() and not app.resume_recording()
        with pytest.raises(RuntimeError, match="already active"):
            app.start_recording(tmp_path / "second.mp4")
        assert not messages and not capsys.readouterr().out
    finally:
        release.set()
        assert recorder._finished.wait(2)
        app._poll_recording()
    assert not app.recording.active and not recorder._worker.is_alive()
    assert len(messages) == 1 and "Saved video" in capsys.readouterr().out


def test_paused_recording_reports_worker_failure_without_another_frame(tmp_path, capsys):
    from mojive.capture.video_queue import BufferedVideoRecorder

    class Broken:
        size = (2, 2)

        def append(self, image):
            raise RuntimeError("paused encoder failed")

        def close(self):
            pass

    app = ViewerApp.__new__(ViewerApp)
    app.localizer = SimpleNamespace(text=lambda value: value)
    messages = []
    app.session = SimpleNamespace(report_message=lambda message, **kw: messages.append(message))
    recorder = app._viewport_recorder = BufferedVideoRecorder(Broken())
    app._viewport_recording_path = tmp_path / "video.mp4"
    app._viewport_recording_phase = RecordingPhase.PAUSED
    app._viewport_recording_frames = 1
    recorder.append(np.zeros((2, 2, 3), np.uint8))
    assert recorder._finished.wait(2)
    app._poll_recording()
    app._poll_recording()
    assert not app.recording.active
    assert "paused encoder failed" in app.recording.error
    assert len(messages) == 1 and "failed" in messages[0]
    assert not capsys.readouterr().out


def test_first_video_write_does_not_block_ui_or_start_physics_before_ack(monkeypatch, tmp_path):
    import threading

    from mojive import RecordingConfig
    from mojive import commands as cmd
    from mojive.capture import recording

    entered, release = threading.Event(), threading.Event()
    actions = []

    class Recorder:
        def __init__(self, path, size, fps, **encoding):
            self.size = size

        def append(self, image):
            entered.set()
            assert release.wait(2)

        def close(self):
            pass

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)
    app = ViewerApp.__new__(ViewerApp)
    app.recording_config = RecordingConfig(countdown=0, run_simulation=True)
    app._surface_image = lambda *args: np.zeros((2, 2, 3), np.uint8)
    app.session = SimpleNamespace(
        adapter=SimpleNamespace(caps=SimpleNamespace(clock_control=True)),
        submit=lambda command: actions.append(type(command).__name__) or cmd.CommandResult.good(""),
        report_message=lambda *args, **kw: None,
    )
    app.start_recording(tmp_path / "video.mp4")
    app._advance_recording_countdown()
    try:
        app._finish_capture_and_recording(None, 0)
        assert entered.wait(1)
        app._poll_recording()
        assert not actions and app.recording.frames == 1
        assert app.pause_recording()
        release.set()
        assert app._viewport_recorder.started.wait(2)
        app._poll_recording()
        assert not actions
        assert app.resume_recording()
        app._poll_recording()
        app._poll_recording()
        assert actions == ["Play"]
    finally:
        release.set()
        app.stop_recording(report=False)
