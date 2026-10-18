"""Native Settings uses real ImGui input and the workbench's panel state."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
from examples.ui_design.native.appearance_review import _check_controls
from examples.ui_design.native.docking_study import BOTTOM_PANELS, LEFT_PANELS, PANELS, Study
from examples.ui_design.native.localization import reference_text
from examples.ui_design.native.scene_preview import ReferencePanels
from imgui_bundle import imgui

from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.panels.settings_pages.catalog import settings_category_label
from mojive.ui.workspace.settings_panel import CATEGORIES, WorkspaceSettings
from mojive.ui.workspace.widgets import UI


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
    study.window_position, study.window_size = (0, 0), None
    study.open = dict.fromkeys((*PANELS, "Inspector", "Timeline"), False)
    study.open.update(Scene=True, Inspector=True, Settings=True)
    study.left_active = study.active = "Scene"
    study.bottom_active = "Timeline"
    study.bottom_focus = study.focus_next = None
    study.timeline_layout_pending = False
    panels = ReferencePanels()
    panels.study = study
    study.preview = SimpleNamespace(panels=panels)
    panel = study.settings = WorkspaceSettings(study)
    selectable = imgui.selectable

    def record_choice(label, *args, **kwargs):
        result = selectable(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits["settings-choice-" + label] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return result

    monkeypatch.setattr(imgui, "selectable", record_choice)
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
            imgui.set_next_window_pos(study.window_position)
            imgui.set_next_window_size(study.window_size or io.display_size)
            imgui.begin("Settings", flags=imgui.WindowFlags_.no_title_bar)
            panel.draw(ctx)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def reveal(key):
        if imgui.get_current_context().open_popup_stack:
            return
        if not key.startswith("##") and key not in (
            "settings-sidebar",
            "settings-bottom",
            "settings-inspector",
        ):
            return
        page = next(
            (
                window
                for window in imgui.get_current_context().windows
                if window.active and "Reference settings content" in window.name
            ),
            None,
        )
        if page is None:
            return
        _x, y, _w, h = ui.hits[key]
        clip = page.inner_clip_rect
        offset = min(0, y - clip.min.y) + max(0, y + h - clip.max.y)
        if offset:
            imgui.internal.set_scroll_y(page, page.scroll.y + offset)
            frame(3)

    def click(key):
        reveal(key)
        x, y, w, h = ui.hits[key]
        event("add_mouse_pos_event", x + w / 2, y + h / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(2)

    study.frame, study.click, study.reveal, study.ctx = frame, click, reveal, ctx
    frame(3)
    try:
        yield study
    finally:
        session.release()
        imgui.destroy_context(context)


def test_settings_navigation_paints_distinct_category_symbols(settings, monkeypatch):
    from examples.ui_design.native.guide_panels import SAMPLES

    from mojive.ui.workspace import widgets

    painted, draw_icon = [], widgets.draw_icon

    def observe(draw, name, *args, **kwargs):
        window = imgui.get_current_context().current_window.name
        first = len(draw.vtx_buffer)
        draw_icon(draw, name, *args, **kwargs)
        assert len(draw.vtx_buffer) > first
        painted.append((window, name))

    monkeypatch.setattr(widgets, "draw_icon", observe)
    settings.settings._category = "Camera"
    settings.frame()
    navigation = [name for window, name in painted if "Reference settings navigation" in window]
    assert navigation == ["general", "camera", "mouse", "shading", "video", "mujoco-visuals"]
    assert [name for window, name in painted if window == "Settings"].count("settings") == 1
    assert [icon for _label, icon in SAMPLES["Settings navigation"]] == navigation


def test_color_format_growth_keeps_actions_inside_the_viewport(settings, tmp_path):
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore

    ui, scale, io = settings.ui, settings.s, imgui.get_io()
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    authored = deepcopy(runtime.spec)
    authored["parts"]["rgba"] = {"columns": 1}
    # An authored large picker forces upward relocation when channels expand;
    # the compact default can fit at this anchor without moving at all.
    authored["parts"]["color_editor"] = {"popupWidth": 360, "pickerHeight": 260}
    runtime.update_spec(authored)
    ui.design = DesignPresentation(ui, runtime)
    settings.window_position, settings.window_size = (
        (310 * scale, 90 * scale),
        (820 * scale, 620 * scale),
    )
    io.display_size = (1440 * scale, 800 * scale)
    settings.settings._category = "MuJoCo Visuals"
    settings.frame(3)
    settings.click("##collision_color-swatch")
    stack = imgui.get_current_context().open_popup_stack
    window = stack[len(stack) - 1].window
    original_top, original_height = window.pos.y, window.size.y
    settings.click("##collision_color-format-2")
    assert window.size.y > original_height
    assert window.pos.y < original_top
    assert window.pos.y + window.size.y <= io.display_size.y - 8 * scale + 1
    for action in ("cancel", "done"):
        x, y, width, height = ui.hits["##collision_color-" + action]
        clip = window.inner_clip_rect
        assert clip.min.x <= x < x + width <= clip.max.x + 1
        assert clip.min.y <= y < y + height <= clip.max.y + 1
    settings.click("##collision_color-cancel")
    assert not imgui.get_current_context().open_popup_stack


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
            assert settings.ctx.tr(settings_category_label(name)) in painted, (
                "Settings category was truncated",
                name,
            )
        assert settings.settings._category == category
        window = imgui.internal.find_window_by_name("Settings")
        owner = (window.pos.x, window.pos.y, window.size.x, window.size.y)
        controls = settings.ui.hits
        header = {key: controls[key] for key in ("settings-title", "settings-close")}
        navigation = {"settings-" + name: controls["settings-" + name] for name in CATEGORIES}
        _check_controls(header | navigation, owner, tolerance=settings.s)
        assert window.scroll_max.y == 0, "Settings must scroll its page, not its header"
        if category == "General":
            settings.reveal("settings-inspector")
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
        settings.click("settings-choice-" + settings.ctx.tr(name))

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


def test_settings_sections_use_spacing_and_only_the_window_header_has_a_rule(settings, monkeypatch):
    ui = settings.ui
    line, section = ui.line, ui.section
    dividers, groups, header_dividers = [], [], []

    def is_page():
        return "Reference settings content" in imgui.get_current_context().current_window.name

    def record_line(x, y, width):
        if is_page():
            dividers.append(y)
        else:
            header_dividers.append(y)
        return line(x, y, width)

    def record_section(label, x, y, width, *args, **kwargs):
        if is_page():
            groups.append((y, kwargs.get("divider", True)))
        return section(label, x, y, width, *args, **kwargs)

    monkeypatch.setattr(ui, "line", record_line)
    monkeypatch.setattr(ui, "section", record_section)
    for category in CATEGORIES:
        settings.settings._category = category
        dividers.clear()
        groups.clear()
        header_dividers.clear()
        settings.frame()
        assert len(header_dividers) == 1, category
        assert not dividers, category
        assert all(not divider for _y, divider in groups), category
        if category == "Camera":
            assert len(groups) >= 2


def test_settings_navigation_uses_its_font_and_icon_sizes_without_changing_page_text(
    settings, monkeypatch, tmp_path
):
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore
    from mojive.ui.workspace import widgets

    ui = settings.ui
    ui.prepare_fonts("system", ("portable",))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)
    baseline = deepcopy(runtime.spec)
    baseline["parts"].update(
        {
            "button": {"fontSize": 14, "textColor": "#e04040"},
            "settings_page": {"fontSize": 14, "textColor": "#a0c0e0"},
        }
    )
    runtime.update_spec(baseline)
    settings.settings._category = "Camera"
    settings.frame(2)
    initial_height = ui.hits["settings-Camera"][3]
    authored = deepcopy(runtime.spec)
    authored["parts"]["settings_nav"] = {
        "fontSize": 22,
        "fontFamily": "portable",
        "textColor": "#20a060",
        "iconSize": 24,
        "iconColor": "#8060e0",
        "paddingX": 12,
        "marginLeft": 4,
        "marginRight": 6,
    }
    runtime.update_spec(authored)
    text, draw_icon = ImguiDraw2D.text, widgets.draw_icon
    captions, icons = [], []

    def record_text(draw, position, color, value, **kwargs):
        captions.append((value, imgui.get_font().font_id, imgui.get_font_size(), color))
        return text(draw, position, color, value, **kwargs)

    def record_icon(draw, name, x, y, size, tint, scale):
        icons.append((name, size, tint))
        return draw_icon(draw, name, x, y, size, tint, scale)

    monkeypatch.setattr(ImguiDraw2D, "text", record_text)
    monkeypatch.setattr(widgets, "draw_icon", record_icon)
    settings.frame(2)
    font = ui.resolve_font(ui.component_style("settings_nav")).font_id
    for category in CATEGORIES:
        shown = settings.ctx.tr(settings_category_label(category))
        assert any(
            value == shown
            and chosen == font
            and size == pytest.approx(22 * settings.s, abs=0.5)
            and color[:3] == pytest.approx((32 / 255, 160 / 255, 96 / 255))
            for value, chosen, size, color in captions
        ), (
            "Navigation must show the complete category in its authored style",
            category,
            font,
            [caption for caption in captions if caption[0] == shown],
            ui.hits["settings-" + category],
        )
    assert ui.hits["settings-Camera"][3] > initial_height
    assert any(size == 24 and tint[:3] == (128, 96, 224) for _name, size, tint in icons)
    assert any(
        value == settings.ctx.tr("Focus")
        and size != 22 * settings.s
        and color[:3] != pytest.approx((32 / 255, 160 / 255, 96 / 255))
        for value, _font, size, color in captions
    )
    if imgui.get_io().display_size.x / settings.s >= 560:
        full_width = ui.hits["settings-General"][2]
        authored["parts"]["settings_layout"] = {"navWidth": 128}
        runtime.update_spec(authored)
        settings.frame(2)
        assert ui.hits["settings-General"][2] >= full_width - settings.s


def test_navigation_width_moves_the_page_without_changing_other_groups(settings, tmp_path):
    from mojive.design.native.presentation import DesignPresentation
    from mojive.design.runtime import DesignRuntime
    from mojive.design.store import SpecificationStore

    ui = settings.ui
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    settings.frame(2)

    def geometry():
        return {row["key"]: row for row in runtime.observed}

    initial = geometry()
    for width in (160, 260, 0):
        spec = deepcopy(runtime.spec)
        spec["parts"]["settings_nav"] = {"navWidth": width}
        runtime.update_spec(spec)
        runtime.observed.clear()
        settings.frame(2)
        rows = geometry()
        nav = rows["settings-navigation"]["outer"]
        if width:
            assert nav[2] == pytest.approx(width * settings.s)
        else:
            assert nav == initial["settings-navigation"]["outer"]
        assert rows["settings-layout"]["windowRect"] == initial["settings-layout"]["windowRect"]
        page = rows["settings-page"]["outer"]
        if imgui.get_io().display_size.x / settings.s >= 560:
            assert page[0] == pytest.approx(nav[0] + nav[2] + 12 * settings.s)
        assert page[0] + page[2] <= imgui.get_io().display_size.x
        assert runtime.spec["parts"].get("settings_page") == spec["parts"].get("settings_page")


@pytest.mark.parametrize("presentation", (False, True))
def test_integer_quality_entry_accumulates_small_drags_and_cancels(
    settings, tmp_path, presentation
):
    if presentation:
        from mojive.design.native.presentation import DesignPresentation
        from mojive.design.runtime import DesignRuntime
        from mojive.design.store import SpecificationStore

        settings.ui.design = DesignPresentation(
            settings.ui, DesignRuntime(SpecificationStore(tmp_path / "design.json"))
        )
        settings.frame(2)
    ctx = settings.ctx
    setter = ctx.set_recording_config
    app = setter.__self__

    def update(config, *, persist=True):
        setter(config, persist=persist)
        ctx.recording_config = app.recording_config

    ctx.set_recording_config = update
    settings.click("settings-Recording")
    original = ctx.recording_config.crf
    saved = app.preferences.get("recording")
    key = "##recording_crf-value"
    x, y, width, height = settings.ui.hits[key]
    content = next(
        window
        for window in imgui.get_current_context().windows
        if window.active and "Reference settings content" in window.name
    )
    if y + height > content.clip_rect.max.y:
        imgui.internal.set_scroll_y(
            content, content.scroll.y + y + height - content.clip_rect.max.y + 8 * settings.s
        )
        settings.frame()
        x, y, width, height = settings.ui.hits[key]
    io = imgui.get_io()
    start = x + width / 2, y + height / 2
    io.add_mouse_pos_event(*start)
    settings.frame()
    io.add_mouse_button_event(0, True)
    settings.frame()
    for step in range(1, 4):
        io.add_mouse_pos_event(start[0] + step * 40 * settings.s, start[1])
        settings.frame()
    assert ctx.recording_config.crf != original
    assert isinstance(ctx.recording_config.crf, int)
    io.add_key_event(imgui.Key.escape, True)
    settings.frame()
    assert ctx.recording_config.crf == original
    io.add_key_event(imgui.Key.escape, False)
    io.add_mouse_button_event(0, False)
    settings.frame()
    assert ctx.recording_config.crf == original
    assert app.preferences.get("recording") == saved
