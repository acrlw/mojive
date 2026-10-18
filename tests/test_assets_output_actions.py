"""Production import and output actions retain real routing and empty-state recovery."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.messages import OutputBuffer
from mojive.ui.panels.assets import AssetsPanel
from mojive.ui.theme import THEME
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.output_panel import WorkspaceOutput
from mojive.ui.workspace.property_style import NativePropertyPresentation, property_controls
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=((1, "en"), (2, "en"), (1, "zh_CN"), (2, "zh_CN")))
def surface(request, tmp_path, monkeypatch):
    scale, language = request.param
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (1000 * scale, 900 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json")))
    ui.translate = Localizer(Language(language)).text
    workspace = SimpleNamespace(ui=ui, s=scale, collapse_bottom=lambda: None)
    workspace.menus = Menus(workspace, ())
    ctx = SimpleNamespace(
        tr=ui.translate,
        style_scale=scale,
        language=language,
        theme=THEME,
        input_bindings=None,
        input_claim=None,
    )
    painted, copied, enabled = [], [], {}
    draw_text, wrapped, button, local_item = (
        ImguiDraw2D.text,
        imgui.text_wrapped,
        ui.button,
        workspace.menus.local_item,
    )

    def text(draw, position, color, value, **options):
        painted.append(value)
        return draw_text(draw, position, color, value, **options)

    def wrap(value):
        painted.append(value)
        return wrapped(value)

    def record_button(key, *args, **options):
        enabled[key] = options.get("enabled", True)
        return button(key, *args, **options)

    def record_menu(label, *args, **options):
        enabled[options["key"]] = options.get("enabled", True)
        return local_item(label, *args, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    monkeypatch.setattr(imgui, "text_wrapped", wrap)
    monkeypatch.setattr(ui, "button", record_button)
    monkeypatch.setattr(workspace.menus, "local_item", record_menu)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    state = SimpleNamespace(
        ui=ui, ctx=ctx, workspace=workspace, text=painted, copied=copied, enabled=enabled
    )

    def frame():
        ui.hits.clear()
        painted.clear()
        enabled.clear()
        ui.design.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((state.width * scale, 650 * scale))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (1, 1))
        imgui.begin(state.group, flags=imgui.WindowFlags_.no_title_bar)
        try:
            with group_scope(state.group), property_controls(ui):
                state.draw(ctx)
        finally:
            imgui.end()
            imgui.pop_style_var()
            imgui.render()

    def click(key):
        x, y, w, h = ui.hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        io.add_mouse_button_event(0, False)
        frame()
        for _ in range(3):
            frame()

    state.frame, state.click = frame, click
    try:
        yield state
    finally:
        imgui.destroy_context(context)


def test_import_menu_fits_narrow_column_and_routes_original_asset_options(surface):
    view, calls = surface, []
    panel = AssetsPanel()
    panel.presentation = NativePropertyPresentation(view.ui, prefix="assets")
    panel._model_id = 42
    panel._hfield_import_size = np.array((2, 3, 4, 0.5))
    panel._texture_import_type = "cube"
    view.ctx.session = SimpleNamespace(
        paused=True, adapter=SimpleNamespace(caps=SimpleNamespace(model_assets=True))
    )
    view.ctx.request_model_asset_import = lambda *args: calls.append(("asset", args))
    view.ctx.request_texture_import = lambda *args: calls.append(("texture", args))
    view.group, view.width, view.draw = "Assets", 220, panel._import_controls
    for _ in range(3):
        view.frame()
    assert "##texture-import-type" not in view.ui.hits
    for index in range(3):
        view.click("asset-import")
        for label in ("Import Mesh...", "Import Height Field...", "Import Texture..."):
            assert view.ctx.tr(label) in view.text
        assert all("…" not in text[:-1] for text in view.text)
        view.click(f"asset-import/choice/{index}")
    assert calls == [
        ("asset", (42, "mesh", ())),
        ("asset", (42, "hfield", (("size", "2 3 4 0.5"),))),
        ("texture", (42, -1, "cube")),
    ]
    view.click("section-" + view.ctx.tr("Import options"))
    assert "##texture-import-type" in view.ui.hits
    view.click("section-" + view.ctx.tr("Height-field import size"))
    assert "##hfield-import-size-0" in view.ui.hits
    np.testing.assert_array_equal(panel._hfield_import_size, (2, 3, 4, 0.5))


def test_import_menu_disables_missing_requests_and_running_model_edits(surface):
    view, calls = surface, []
    panel = AssetsPanel()
    panel.presentation = NativePropertyPresentation(view.ui)
    view.ctx.session = SimpleNamespace(
        paused=True, adapter=SimpleNamespace(caps=SimpleNamespace(model_assets=True))
    )
    view.ctx.request_model_asset_import = None
    view.ctx.request_texture_import = lambda *args: calls.append(args)
    view.group, view.width, view.draw = "Assets", 220, panel._import_controls
    for _ in range(3):
        view.frame()
    view.click("asset-import")
    view.click("asset-import/choice/0")
    assert not calls
    view.click("asset-import/choice/2")
    assert len(calls) == 1
    view.ctx.session.paused = False
    view.frame()
    view.click("asset-import")
    assert "asset-import/choice/2" not in view.ui.hits
    assert len(calls) == 1


@pytest.mark.parametrize("width", (180, 700))
def test_output_empty_and_filtered_states_have_real_clear_copy_and_reset_actions(surface, width):
    view = surface
    panel = WorkspaceOutput(view.workspace)
    output = view.ctx.output = OutputBuffer()
    view.group, view.width, view.draw = "Output", width, panel.draw
    # An empty buffer remains an empty-history state even with saved filters.
    panel._filter_text = "missing"
    panel._levels = {"error"}
    for _ in range(3):
        view.frame()
    assert view.ctx.tr("No messages") in view.text
    assert "output-reset-filters" not in view.ui.hits

    def actions():
        if width < 300:
            view.click("output-more")
            return "output-menu-copy", "output-menu-clear"
        return "output-copy", "output-clear"

    copy_key, clear_key = actions()
    assert not view.enabled[copy_key] and not view.enabled[clear_key]
    view.click(clear_key)
    if width < 300:
        imgui.get_io().add_key_event(imgui.Key.escape, True)
        view.frame()
        imgui.get_io().add_key_event(imgui.Key.escape, False)
        view.frame()
    first = output.write("Mesh loaded", timestamp="10:00:00")
    output.write("Invalid joint", level="error", timestamp="10:00:01")
    original = output.entries()
    panel._selected_sequences = {first.sequence}
    view.frame()
    assert view.ctx.tr("No matching messages") in view.text
    assert "output-reset-filters" in view.ui.hits
    reset = view.ui.hits["output-reset-filters"]
    assert reset[0] + reset[2] <= width * view.ui.s
    assert view.ctx.tr("Clear filters") in view.text
    if width == 700:
        assert reset[2] < width * view.ui.s / 2
    copy_key, clear_key = actions()
    assert not view.enabled[copy_key] and view.enabled[clear_key]
    view.click(copy_key)
    assert view.copied == []
    if width < 300:
        imgui.get_io().add_key_event(imgui.Key.escape, True)
        view.frame()
        imgui.get_io().add_key_event(imgui.Key.escape, False)
        view.frame()
    view.click("output-reset-filters")
    assert panel._filter_text == "" and panel._levels == {"info", "warning", "error"}
    assert output.entries() == original
    assert panel._selected_sequences == {first.sequence}
    assert "output-empty-state" not in view.ui.hits
    copy_key, clear_key = actions()
    assert view.enabled[copy_key] and view.enabled[clear_key]
    view.click(copy_key)
    assert view.copied == ["Mesh loaded"]
    if width < 300:
        copy_key, clear_key = actions()
    view.click(clear_key)
    assert not output.entries()
    assert view.ctx.tr("No messages") in view.text


def test_output_reset_action_measures_local_font_and_padding_instead_of_panel_width(surface):
    view = surface
    panel = WorkspaceOutput(view.workspace)
    view.ctx.output = OutputBuffer()
    view.ctx.output.write("message")
    panel._filter_text = "unmatched"
    view.group, view.width, view.draw = "Output", 700, panel.draw
    for _ in range(3):
        view.frame()
    original = view.ui.hits["output-reset-filters"][2]
    view.width = 180
    view.frame()
    assert view.ui.hits["output-reset-filters"][2] == pytest.approx(original)
    spec = deepcopy(view.ui.design.runtime.spec)
    spec["parts"]["button"] = {
        "fontSize": 20,
        "paddingLeft": 18,
        "paddingRight": 20,
    }
    view.ui.design.runtime.update_spec(spec)
    view.width = 700
    view.frame()
    expanded = view.ui.hits["output-reset-filters"][2]
    assert expanded > original + 20 * view.ui.s
    assert expanded < 350 * view.ui.s
    assert view.ctx.tr("Clear filters") in view.text
