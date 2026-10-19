"""Native startup installs one final atlas and constructs one panel registry."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import ViewerConfig
from mojive.adapters.static import StaticSceneAdapter
from mojive.config import PanelConfig
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.text import sources
from mojive.tools.startup_profile import (
    _startup_timings,
    _trace_app_frames,
    _trace_completion,
    _trace_window_frames,
)
from mojive.ui import fonts
from mojive.ui import window as window_module
from mojive.ui.app import ViewerApp
from mojive.ui.window import Window, WindowConfig
from mojive.ui.workspace import fonts as workspace_fonts
from mojive.ui.workspace.app import WorkspaceApp
from mojive.ui.workspace.widgets import UI


@pytest.fixture
def local_fonts(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "cache_dir", lambda: tmp_path / "font-cache")
    monkeypatch.setattr(
        sources.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("Native startup must not download fonts"),
    )
    monkeypatch.setattr(window_module, "imgui", imgui)


def _window(context, scale, *, default_fonts=False):
    window = Window.__new__(Window)
    window.config = WindowConfig(load_default_fonts=default_fonts)
    window._style_scale = scale
    window._imgui_context = context
    window._ini_existed = False
    window.apply_theme = lambda _theme: None
    window.set_layout_builder = lambda _builder: None
    return window


def _font_geometry(scale, portable, *, deferred):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (900 * scale, 400 * scale)
        io.display_framebuffer_scale = (2, 2)
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        window = _window(context, scale)
        if deferred:
            window._load_fonts(io)
            assert len(io.fonts.fonts) == 0
        else:
            window.font_report = fonts.load(imgui, io, size_pt=14 * scale, allow_download=False)
            assert len(io.fonts.fonts) == 1
        ui = UI(window, portable_fonts=portable)
        assert len(io.fonts.fonts) == 3
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Startup fonts", flags=imgui.WindowFlags_.no_title_bar)
        metrics = []
        for role in ("body", "bold", "mono"):
            face = getattr(ui, role)
            imgui.push_font(face, 14 * scale)
            imgui.text_unformatted("Camera 相机 120.5")
            baked = face.get_font_baked(14 * scale, 2)
            for char in "H相0":
                glyph = baked.find_glyph_no_fallback(ord(char))
                metrics.append(
                    (role, char, glyph.advance_x, glyph.x0, glyph.y0, glyph.x1, glyph.y1)
                )
            imgui.pop_font()
        geometry = tuple(
            (vertex.pos.x, vertex.pos.y, vertex.col)
            for vertex in imgui.get_window_draw_list().vtx_buffer
        )
        imgui.end()
        imgui.render()
        return ui.font_sources, metrics, geometry
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("scale", (1, 1.2, 1.25, 2.25))
@pytest.mark.parametrize("portable", (False, True))
def test_deferred_default_atlas_keeps_exact_native_font_sources_and_geometry(
    local_fonts, scale, portable
):
    assert _font_geometry(scale, portable, deferred=True) == _font_geometry(
        scale, portable, deferred=False
    )


def test_native_fonts_reuse_resolved_numeric_source_without_rediscovery(local_fonts, monkeypatch):
    context = imgui.create_context()
    try:
        window = _window(context, 1.2)
        window._load_fonts(imgui.get_io())
        report = window.font_report

        def rediscovered(*_args, **_kwargs):
            pytest.fail("The window already resolved the real primary font")

        monkeypatch.setattr(workspace_fonts, "resolve_font_sources", rediscovered)
        ui = UI(window, portable_fonts=True)
        source, _ratio = ui.font_sources[2]
        assert (source.path, source.index, source.label) == (
            report.mono_path,
            report.mono_index,
            report.mono,
        )
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "config",
    (
        None,
        ViewerConfig(),
        ViewerConfig.minimal("hierarchy", "inspector", "settings"),
        replace(ViewerConfig(), panels={"sensors": PanelConfig(enabled=False)}),
    ),
)
def test_workspace_constructs_each_builtin_once_with_final_native_visibility(
    local_fonts, tmp_path, monkeypatch, config
):
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.ui import panels
    from mojive.ui.workspace.panel_manager import WorkspacePanels

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "preferences.json"))
    calls = []
    create = panels._create_builtin_panel

    def counted(panel_id):
        calls.append(panel_id)
        return create(panel_id)

    monkeypatch.setattr(panels, "_create_builtin_panel", counted)
    context = imgui.create_context()
    app = None
    try:
        window = _window(context, 1.2)
        window._load_fonts(imgui.get_io())
        app = WorkspaceApp(
            Session(StaticSceneAdapter(Scene())),
            NullBackend(),
            window,
            config=config,
            design=DesignRuntime(SpecificationStore(tmp_path / "design.json")),
        )
        expected = config.builtin_panels if config else None
        assert calls == list(panels._BUILTIN_PANELS if expected is None else expected)
        assert isinstance(app.panels, WorkspacePanels)
        assert set(app.panels.initial_config) == set(config.panels if config else {})
        assert {panel.name for panel in app.panels if panel.open} == {"Scene", "Inspector"}
        if config and "sensors" in config.panels:
            assert not app.panels.get("sensors").enabled
    finally:
        if app is not None:
            app.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize("interface", ("native", "classic"))
def test_composition_only_defers_the_frontend_owned_atlas(tmp_path, monkeypatch, interface):
    from mojive.app.composition import build_scene
    from mojive.app.ui import window as window_factory
    from mojive.render.opengl import backend
    from mojive.ui.workspace import app as workspace

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "preferences.json"))
    configs = []

    def create(config, _renderer):
        configs.append(config)
        return SimpleNamespace(
            size_pixels=(640, 480),
            make_current=lambda: None,
            apply_theme=lambda _theme: None,
            close=lambda: None,
        )

    monkeypatch.setattr(window_factory, "create_window", create)
    monkeypatch.setattr(backend, "OpenGLBackend", lambda *_args: NullBackend())
    monkeypatch.setattr(workspace, "WorkspaceApp", ViewerApp)
    config = replace(ViewerConfig.minimal(), ui=interface)
    with build_scene(Scene(), config=config, renderer="opengl"):
        assert configs[0].load_default_fonts is (interface == "classic")
    assert WindowConfig().load_default_fonts


def test_startup_completion_marks_the_full_workspace_after_base_initialization():
    events = []

    class Base:
        def __init__(self):
            events.append("base initialized")

    class Workspace(Base):
        def __init__(self):
            super().__init__()
            events.append("native fonts and shell initialized")

    _trace_completion(Base, "__init__", events.append, "base_app_ready")
    _trace_completion(Workspace, "__init__", events.append, "app_ready")
    Workspace()
    assert events == [
        "base initialized",
        "base_app_ready",
        "native fonts and shell initialized",
        "app_ready",
    ]


def test_failed_initialization_does_not_publish_a_ready_marker():
    events = []
    target = SimpleNamespace(initialize=lambda: 1 / 0)
    _trace_completion(target, "initialize", events.append, "app_ready")
    with pytest.raises(ZeroDivisionError):
        target.initialize()
    assert not events


def _startup_root(**changes):
    values = {
        "active": True,
        "flags": 0,
        "collapsed": False,
        "dock_node": None,
        "tab_id": 10,
        "hidden": False,
        "hidden_frames_cannot_skip_items": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def reveal_window(monkeypatch):
    calls = []
    size = [640, 480]
    monkeypatch.setattr(window_module, "imgui", imgui)
    monkeypatch.setattr(
        window_module,
        "glfw",
        SimpleNamespace(
            get_framebuffer_size=lambda _handle: tuple(size),
            show_window=lambda _handle: calls.append("mapped"),
            hide_window=lambda _handle: calls.append("hidden"),
        ),
    )
    window = Window.__new__(Window)
    window._initialize_layout_state()
    window._shown = False
    window._window = object()
    window._imgui_context = SimpleNamespace(windows=[_startup_root()])
    return window, calls, size


@pytest.mark.parametrize(
    "changes",
    ({"hidden": True}, {"hidden_frames_cannot_skip_items": 1}),
)
def test_automatic_reveal_waits_for_measured_root_windows(reveal_window, changes):
    window, calls, _size = reveal_window
    root = _startup_root(**changes)
    window._imgui_context.windows = [root]
    window.request_show()
    assert not calls
    window._show_after_frame()
    assert not window.shown
    assert not calls
    root.hidden = False
    root.hidden_frames_cannot_skip_items = 0
    window._show_after_frame()
    window._show_after_frame()
    assert window.shown
    assert calls == ["mapped"]


@pytest.mark.parametrize(
    "changes",
    (
        {"active": False},
        {"collapsed": True},
        {"flags": imgui.WindowFlags_.child_window},
        {"flags": imgui.WindowFlags_.popup},
        {"flags": imgui.WindowFlags_.tooltip},
        {"dock_node": SimpleNamespace(selected_tab_id=20)},
    ),
)
def test_hidden_inactive_tabs_and_transient_windows_do_not_block_reveal(reveal_window, changes):
    window, calls, _size = reveal_window
    window._imgui_context.windows.append(
        _startup_root(hidden=True, hidden_frames_cannot_skip_items=1, **changes)
    )
    window.request_show()
    window._show_after_frame()
    assert window.shown
    assert calls == ["mapped"]


@pytest.mark.parametrize("selected", (0, 10))
def test_selected_or_unresolved_dock_tab_must_finish_its_hidden_frame(reveal_window, selected):
    window, calls, _size = reveal_window
    window._imgui_context.windows = [
        _startup_root(hidden=True, dock_node=SimpleNamespace(selected_tab_id=selected))
    ]
    window.request_show()
    window._show_after_frame()
    assert not window.shown
    assert not calls


@pytest.mark.parametrize(
    "pending", ("_layout_reset_pending", "_layout_rebuild_pending", "_default_tabs_frame")
)
def test_reveal_waits_for_pending_layout_requests(reveal_window, pending):
    window, calls, _size = reveal_window
    setattr(window, pending, 1)
    window.request_show()
    window._show_after_frame()
    assert not calls
    setattr(window, pending, 0)
    window._show_after_frame()
    assert window.shown


@pytest.mark.parametrize("size", ((0, 480), (640, 0)))
def test_zero_size_frame_does_not_reveal_a_window(reveal_window, size):
    window, calls, current_size = reveal_window
    current_size[:] = size
    window.request_show()
    window._show_after_frame()
    assert not window.shown
    assert not calls
    current_size[:] = (640, 480)
    window._show_after_frame()
    assert window.shown


def test_hiding_cancels_automatic_reveal_and_manual_show_remains_immediate(reveal_window):
    window, calls, _size = reveal_window
    window.request_show()
    window.hide()
    window._show_after_frame()
    assert not window.shown
    assert calls == ["hidden"]
    window._imgui_context.windows[0].hidden = True
    window.show()
    window._show_after_frame()
    assert window.shown
    assert calls == ["hidden", "mapped"]


def test_empty_frame_does_not_reveal_a_window(reveal_window):
    window, calls, _size = reveal_window
    window._imgui_context.windows.clear()
    window.request_show()
    window._show_after_frame()
    assert not window.shown and not calls


@pytest.mark.parametrize("has_frame", (False, True))
def test_new_native_surface_presents_retained_frame_without_an_app_tick(monkeypatch, has_frame):
    from mojive.ui.window_native import NativeWindow

    calls = []
    window = NativeWindow.__new__(NativeWindow)
    window._surface = None
    window._frame_target = object() if has_frame else None
    monkeypatch.setattr(Window, "show", lambda _self: calls.append("map"))
    monkeypatch.setattr(Window, "size_pixels", property(lambda _self: (640, 480)))

    def create_surface():
        calls.append("surface")
        window._surface = "surface"

    window._create_surface = create_surface
    window._present_packet = lambda size: ("vertices", "indices", "commands")
    window.runtime = SimpleNamespace(
        render_ui=lambda *args: calls.append(args),
        advance=lambda: calls.append("submit"),
    )
    window.show()
    expected = ["map", "surface"]
    if has_frame:
        expected += [(640, 480, "vertices", "indices", "commands", "surface"), "submit"]
    assert calls == expected
    calls.clear()
    window.show()
    assert calls == ["map"]


def test_frame_trace_records_hidden_completion_before_full_native_show(reveal_window):
    window, calls, _size = reveal_window
    root = window._imgui_context.windows[0]
    root.hidden = True
    native_show = window.show

    def show_retained_frame():
        native_show()
        calls.append("retained frame presented")

    window.show = show_retained_frame
    _trace_window_frames(window, calls.append)
    window.request_show()
    window._show_after_frame()
    assert calls == ["hidden_frame_completed"]
    root.hidden = False
    window._show_after_frame()
    window._show_after_frame()
    assert calls == [
        "hidden_frame_completed",
        "hidden_frame_completed",
        "mapped",
        "retained frame presented",
        "window_shown",
        "visible_frame_completed",
    ]


def test_app_frame_trace_measures_outer_workspace_scope_and_keeps_profiling_until_visible(
    monkeypatch,
):
    from mojive.tools import startup_profile

    elapsed = [0.0]
    profiler = SimpleNamespace(disable=lambda: disabled.append(True))
    disabled = []
    frames = []

    class Base:
        def frame(self):
            elapsed[0] += 0.010
            self.window.shown = True

    class Workspace(Base):
        def __init__(self):
            self.window = SimpleNamespace(shown=False)

        def frame(self):
            elapsed[0] += 0.005
            super().frame()

    monkeypatch.setattr(startup_profile.time, "perf_counter", lambda: elapsed[0])
    app = Workspace()
    _trace_app_frames(app, frames, profiler)
    app.frame()
    assert frames == [
        {"duration_s": pytest.approx(0.015), "shown_at_start": False, "shown_at_end": True}
    ]
    assert not disabled
    app.frame()
    assert frames[1] == {
        "duration_s": pytest.approx(0.015),
        "shown_at_start": True,
        "shown_at_end": True,
    }
    assert disabled == [True]


def test_failed_frame_does_not_publish_frame_duration_or_disable_startup_profile():
    disabled = []
    frames = []

    def fail():
        raise RuntimeError("Render failed")

    app = SimpleNamespace(window=SimpleNamespace(shown=False), frame=fail)
    _trace_app_frames(app, frames, SimpleNamespace(disable=lambda: disabled.append(True)))
    with pytest.raises(RuntimeError, match="Render failed"):
        app.frame()
    assert not frames
    assert not disabled


def _startup_trace():
    return {
        "events": [
            {"name": "hidden_frame_completed", "elapsed_s": 0.9},
            {"name": "hidden_frame_completed", "elapsed_s": 1.0},
            {"name": "window_shown", "elapsed_s": 1.1},
            {"name": "visible_frame_completed", "elapsed_s": 1.12},
            {"name": "visible_frame_completed", "elapsed_s": 1.15},
        ],
        "frames": [
            {"duration_s": 0.8, "shown_at_start": False, "shown_at_end": False},
            {"duration_s": 0.1, "shown_at_start": False, "shown_at_end": True},
            {"duration_s": 0.02, "shown_at_start": True, "shown_at_end": True},
            {"duration_s": 0.03, "shown_at_start": True, "shown_at_end": True},
        ],
    }


def test_startup_timings_separate_hidden_work_reveal_and_visible_frame_cost():
    result = _startup_timings(_startup_trace())
    assert result == pytest.approx(
        {
            "launch_to_first_hidden_frame_s": 0.9,
            "launch_to_reveal_s": 1.1,
            "launch_to_first_visible_frame_s": 1.12,
            "reveal_to_first_visible_frame_s": 0.02,
            "post_reveal_frame_max_s": 0.03,
            "launch_s": 1.12,
            "shown_s": 0.02,
        }
    )


@pytest.mark.parametrize("missing", ("window_shown", "visible_frame_completed"))
def test_startup_report_rejects_hidden_only_or_unrevealed_completion(missing):
    trace = _startup_trace()
    trace["events"] = [event for event in trace["events"] if event["name"] != missing]
    with pytest.raises(RuntimeError, match="visible frame"):
        _startup_timings(trace)


def test_startup_report_rejects_a_visible_marker_before_reveal():
    trace = _startup_trace()
    trace["events"][3]["elapsed_s"] = 1.05
    with pytest.raises(RuntimeError, match="visible frame"):
        _startup_timings(trace)


def test_startup_report_requires_a_full_measured_frame_after_reveal():
    trace = _startup_trace()
    trace["frames"] = trace["frames"][:2]
    with pytest.raises(RuntimeError, match="full frame"):
        _startup_timings(trace)
