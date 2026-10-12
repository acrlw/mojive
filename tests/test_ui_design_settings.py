"""Native Settings uses real ImGui input and the workbench's panel state."""

from itertools import product
from types import SimpleNamespace

import pytest
from examples.ui_design.native.appearance_review import _check_controls
from examples.ui_design.native.docking_study import BOTTOM_PANELS, LEFT_PANELS, PANELS, Study
from examples.ui_design.native.localization import reference_text
from examples.ui_design.native.scene_preview import ReferencePanels
from examples.ui_design.native.settings_panel import CATEGORIES, ReferenceSettings
from examples.ui_design.native.widgets import UI
from imgui_bundle import imgui

from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Localizer
from mojive.ui.panels import PanelContext


@pytest.fixture(params=tuple(product((1.0, 2.25), (420, 820), ("en", "zh_CN"))))
def settings(request, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scale, width, language = request.param
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (width * scale, 620 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    from mojive.text.sources import resolve_font_sources

    report = SimpleNamespace(cjk_path=None)
    try:
        sources = resolve_font_sources(allow_download=False)
    except FileNotFoundError:
        sources = ()
    if len(sources) > 1:
        report.cjk_path, report.cjk_index = sources[1].path, sources[1].index
    elif language == "zh_CN":
        pytest.skip("CJK font is unavailable")
    ui = UI(SimpleNamespace(style_scale=scale, font_report=report))
    localizer = Localizer(preferences={"language": language})
    ui.translate = lambda value: reference_text(localizer, value)
    style = imgui.get_style()
    style.font_size_base = 14 * scale
    style.window_padding = (12 * scale, 10 * scale)
    style.item_spacing = (6 * scale, 8 * scale)
    session = Session(StaticSceneAdapter(Scene()))
    app = ViewerApp(session, NullBackend())
    study = Study.__new__(Study)
    study.s, study.ui = scale, ui
    study.open = dict.fromkeys((*PANELS, "Inspector", "Timeline"), False)
    study.open.update(Scene=True, Inspector=True, Settings=True)
    study.left_active = study.active = "Scene"
    study.bottom_active = "Timeline"
    study.bottom_focus = study.focus_next = None
    study.timeline_layout_pending = False
    panels = ReferencePanels()
    panels.study = study
    study.preview = SimpleNamespace(panels=panels)
    panel = study.settings = ReferenceSettings(study)
    ctx = PanelContext(
        session,
        app.backend,
        camera=app.camera,
        set_camera_navigation=app.set_camera_navigation,
        gizmo=app.gizmo,
        view_cube=app.view_cube,
        perturb=app.perturb,
        scene_entities=app.scene_entities,
        interactions=app.interactions,
        selection_style=app.selection_style,
        input_bindings=app.input_bindings,
        viewport_overlays=app.viewport_overlays,
        recording_config=app.recording_config,
        set_recording_config=app.set_recording_config,
        panels=panels,
        language=language,
        translate=ui.translate,
        style_scale=scale,
    )

    def frame(count=1):
        for _ in range(count):
            ui.hits.clear()
            for item in panels:
                item.open = study.open[item.name]
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Settings", flags=imgui.WindowFlags_.no_title_bar)
            panel.draw(ctx)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def click(key):
        x, y, w, h = ui.hits[key]
        event("add_mouse_pos_event", x + w / 2, y + h / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(2)

    study.frame, study.click, study.ctx = frame, click, ctx
    frame(3)
    try:
        yield study
    finally:
        session.release()
        imgui.destroy_context(context)


def test_settings_header_navigation_and_form_controls_fit(settings, monkeypatch):
    painted = []
    text = ImguiDraw2D.text

    def observe(draw, position, color, value, **kwargs):
        painted.append(value)
        return text(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)
    for category in CATEGORIES:
        painted.clear()
        settings.click("settings-" + category)
        for name in CATEGORIES:
            assert settings.ctx.tr(name) in painted, ("Settings category was truncated", name)
        assert settings.settings._category == category
        window = imgui.internal.find_window_by_name("Settings")
        owner = (window.pos.x, window.pos.y, window.size.x, window.size.y)
        controls = settings.ui.hits
        header = {key: controls[key] for key in ("settings-title", "settings-close")}
        navigation = {"settings-" + name: controls["settings-" + name] for name in CATEGORIES}
        _check_controls(header | navigation, owner, tolerance=settings.s)
        assert window.scroll_max.y == 0, "Settings must scroll its page, not its header"
        if category == "General":
            workspace = {
                key: controls[key]
                for key in ("settings-sidebar", "settings-bottom", "settings-inspector")
            }
            _check_controls(header | navigation | workspace, owner, tolerance=settings.s)
    settings.click("settings-close")
    assert not settings.open["Settings"] and not settings.settings.open


def test_settings_choices_preserve_exclusive_sidebar_and_shared_bottom_tabs(settings):
    def choose(key, name):
        settings.click(key)
        settings.click(key + "-option-" + name)

    for name in ("Control", "Assets", "Scene"):
        choose("settings-sidebar", name)
        assert settings.left_active == name
        assert [panel for panel in LEFT_PANELS if settings.open[panel]] == [name]
    choose("settings-sidebar", "none")
    assert not any(settings.open[name] for name in LEFT_PANELS)
    choose("settings-sidebar", "Scene")
    for name in ("Output", "Timeline", "Statistics"):
        choose("settings-bottom", name)
        assert settings.bottom_active == name
        assert all(settings.open[panel] for panel in BOTTOM_PANELS)
        assert settings.left_active == "Scene" and settings.open["Scene"]
    choose("settings-bottom", "none")
    assert not settings.bottom_open and settings.timeline_layout_pending
    settings.click("settings-inspector")
    assert not settings.open["Inspector"]
    settings.click("settings-inspector")
    assert settings.open["Inspector"]


def test_settings_optional_pages_finish_without_extending_empty_children(settings):
    ctx = settings.ctx
    for name in (
        "camera",
        "set_camera_navigation",
        "gizmo",
        "view_cube",
        "perturb",
        "scene_entities",
        "interactions",
        "selection_style",
        "recording_config",
    ):
        setattr(ctx, name, None)
    for category in ("Camera", "Interaction", "Recording"):
        settings.click("settings-" + category)
        assert settings.settings._category == category
        assert imgui.get_draw_data().total_vtx_count > 0
