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
from mojive.tools.startup_profile import _trace_completion
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
