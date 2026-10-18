"""Camera bookmark actions and their compact production presentation."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.scene.state import list_named_snapshots
from mojive.ui.camera import OrbitCamera
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels.camera import CameraPanel
from mojive.ui.workspace.camera_bookmarks import _bookmark_content
from mojive.ui.workspace.camera_panel import _view_directions
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.widgets import UI


@pytest.fixture
def bookmarks(tmp_path):
    reports = []
    camera = OrbitCamera(yaw=27, pitch=14)
    panel = CameraPanel()
    panel.bookmark_directory = tmp_path / "camera-bookmarks"
    ctx = SimpleNamespace(
        camera=camera,
        model_camera_id=-1,
        model_camera_view=None,
        select_model_camera=None,
        tr=str,
        language="en",
        report=lambda message, **options: reports.append((message, options)),
    )
    return panel, ctx, reports


def test_bookmarks_save_unique_names_restore_and_explicitly_update(bookmarks):
    panel, ctx, _ = bookmarks
    panel._bookmark_name = "Assembly"
    assert panel.save_bookmark(ctx, ctx.camera)
    panel._bookmark_name = "Assembly"
    ctx.camera.yaw = 62
    assert panel.save_bookmark(ctx, ctx.camera)
    assert list_named_snapshots(panel.bookmark_directory) == ["Assembly", "Assembly-2"]
    assert panel.load_bookmark(ctx, ctx.camera, "Assembly")
    assert ctx.camera.yaw == pytest.approx(27)
    ctx.camera.yaw = 85
    assert panel.save_bookmark(ctx, ctx.camera, replace_name="Assembly")
    ctx.camera.yaw = 0
    assert panel.load_bookmark(ctx, ctx.camera, "Assembly")
    assert ctx.camera.yaw == pytest.approx(85)
    assert panel.delete_bookmark(ctx, "Assembly-2")
    assert list_named_snapshots(panel.bookmark_directory) == ["Assembly"]


def test_chinese_bookmark_names_survive_save_duplicate_restore_and_delete(bookmarks):
    panel, ctx, _ = bookmarks
    for _ in range(2):
        panel._bookmark_name = "前视图"
        assert panel.save_bookmark(ctx, ctx.camera)
    assert list_named_snapshots(panel.bookmark_directory) == ["前视图", "前视图-2"]
    ctx.camera.yaw = 90
    assert panel.load_bookmark(ctx, ctx.camera, "前视图")
    assert ctx.camera.yaw == pytest.approx(27)
    assert panel.delete_bookmark(ctx, "前视图")
    assert list_named_snapshots(panel.bookmark_directory) == ["前视图-2"]


@pytest.mark.parametrize("yaw,displayed", ((-495, -135), (630, -90), (180, -180)))
def test_yaw_field_displays_canonical_angle_without_changing_camera_winding(
    bookmarks, monkeypatch, yaw, displayed
):
    import math

    panel, ctx, _ = bookmarks
    ctx.camera.yaw = yaw
    painted = {}
    monkeypatch.setattr(panel, "_section", lambda *args: True)
    monkeypatch.setattr(panel, "_begin_properties", lambda *args: True)
    monkeypatch.setattr(panel, "_end_properties", lambda *args: None)
    monkeypatch.setattr(panel, "_property_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(panel, "_projection", lambda *args: None)

    def rail(ctx, key, value, bounds, **options):
        painted[key] = (value, bounds)
        return SimpleNamespace(changed=False)

    monkeypatch.setattr(panel, "_rail", rail)
    panel._params(ctx, ctx.camera)
    value, (low, high) = painted["##camera-yaw"]
    assert value == pytest.approx(math.radians(displayed))
    assert low <= value <= high
    assert ctx.camera.yaw == yaw


def test_blank_name_saves_a_distinct_default_without_losing_existing_bookmarks(bookmarks):
    panel, ctx, _ = bookmarks
    for _ in range(2):
        assert panel.save_bookmark(ctx, ctx.camera)
    assert list_named_snapshots(panel.bookmark_directory) == ["view-1", "view-2"]
    assert panel._bookmark_name == ""


def test_bookmark_storage_failure_preserves_name_and_reports_error(bookmarks):
    panel, ctx, reports = bookmarks
    panel.bookmark_directory.write_text("not a directory")
    panel._bookmark_name = "Assembly"
    assert not panel.save_bookmark(ctx, ctx.camera)
    assert panel._bookmark_name == "Assembly"
    assert panel._bookmark_error
    assert reports[-1][1]["level"] == "error"


@pytest.mark.parametrize("width", (160, 240, 380))
@pytest.mark.parametrize("scale", (1.0, 2.0))
@pytest.mark.parametrize("designed", (False, True))
def test_native_bookmark_rows_fit_and_restore_with_real_mouse_input(
    bookmarks, width, scale, designed, tmp_path
):
    panel, ctx, _ = bookmarks
    panel._bookmark_name = "Assembly view with a deliberately long name"
    assert panel.save_bookmark(ctx, ctx.camera)
    name = list_named_snapshots(panel.bookmark_directory)[0]
    ctx.camera.yaw = 70
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size, io.delta_time = (500 * scale, 350 * scale), 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        if designed:
            ui = UI(
                SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
                portable_fonts=True,
            )
            runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
            ui.design = DesignPresentation(ui, runtime)
        else:
            ui = UI.__new__(UI)
            ui.s, ui.hits, ui.translate = scale, {}, str
            ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
        workspace = SimpleNamespace(ui=ui, s=scale)
        workspace.menus = Menus(workspace, ())

        def frame():
            ui.hits.clear()
            imgui.new_frame()
            imgui.push_style_var(imgui.StyleVar_.window_min_size, (1, 1))
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((width * scale, 330 * scale))
            imgui.begin("Bookmarks", flags=imgui.WindowFlags_.no_title_bar)
            with group_scope("Cameras"):
                _bookmark_content(workspace, panel, ctx)
            imgui.end()
            imgui.pop_style_var()
            imgui.render()

        frame()
        frame()
        for key, (x, y, w, h) in ui.hits.items():
            if key.startswith("camera-bookmark"):
                assert 0 <= x < x + w <= width * scale
                assert 0 <= y < y + h <= 330 * scale
        key = "camera-bookmark-open/" + name
        x, y, w, h = ui.hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()
        assert ctx.camera.yaw == pytest.approx(27)
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (120, 300))
@pytest.mark.parametrize("language", (Language.ENGLISH, Language.SIMPLIFIED_CHINESE))
def test_camera_directions_include_iso_and_call_the_existing_preset(width, language):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size, io.delta_time = (420, 420), 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        ui = UI.__new__(UI)
        localizer = Localizer(language)
        ui.s, ui.hits, ui.translate = 1, {}, localizer.text
        ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
        presets = []
        app = SimpleNamespace(
            camera=SimpleNamespace(set_preset=presets.append),
            _frame_scene=lambda **kwargs: presets.append("frame"),
        )
        workspace = SimpleNamespace(
            ui=ui, s=1, app=app, _panel_context=SimpleNamespace(language=language)
        )

        def frame():
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((420, 420))
            imgui.begin("Directions")
            _view_directions(workspace, 20, 40, width)
            imgui.end()
            imgui.render()

        frame()
        frame()
        front, back = ui.hits["camera-Front"], ui.hits["camera-Back"]
        assert (front[0] == back[0]) == (width == 120)
        for name in ("Front", "Back", "Left", "Right", "Top", "Bottom", "ISO", "frame"):
            key = "camera-" + name
            x, y, w, h = ui.hits[key]
            io.add_mouse_pos_event(x + w / 2, y + h / 2)
            frame()
            io.add_mouse_button_event(0, True)
            frame()
            io.add_mouse_button_event(0, False)
            frame()
            assert presets[-1] == name.lower()
    finally:
        imgui.destroy_context(context)
