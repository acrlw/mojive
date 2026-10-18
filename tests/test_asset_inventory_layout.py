"""Asset inventory columns retain gutters inside the compact property-form style."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.adapters.base import ModelAssetInfo
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels.assets import AssetsPanel
from mojive.ui.workspace.property_style import NativePropertyPresentation, property_controls
from mojive.ui.workspace.widgets import UI


@pytest.mark.parametrize("scale", (1, 2))
@pytest.mark.parametrize("gap", (6, 14))
def test_inventory_separates_type_from_reference_count_without_changing_form_padding(
    scale, gap, monkeypatch
):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (500 * scale, 700 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    localizer = Localizer(Language.SIMPLIFIED_CHINESE)
    ui.translate = localizer.text
    panel = AssetsPanel()
    panel.presentation = NativePropertyPresentation(ui)
    panel._model_id = 0
    panel._assets = (ModelAssetInfo(0, "material", "surface", 0, references=("box", "floor")),)
    ctx = SimpleNamespace(
        style_scale=scale,
        tr=localizer.text,
        session=SimpleNamespace(scene_models=(SimpleNamespace(model_id=0, name="scene"),)),
    )
    monkeypatch.setattr(panel, "_refresh", lambda _ctx: None)
    monkeypatch.setattr(panel, "_import_controls", lambda _ctx: None)
    spans = {}
    original = imgui.text_disabled

    def text_disabled(value):
        if value in ("material", "2"):
            x = imgui.get_cursor_screen_pos().x
            spans[value] = (x, x + imgui.calc_text_size(value).x)
        return original(value)

    monkeypatch.setattr(imgui, "text_disabled", text_disabled)
    try:
        for _ in range(4):
            spans.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((300 * scale, 650 * scale))
            imgui.begin("Assets")
            with property_controls(ui):
                padding = imgui.get_style().cell_padding
                original_padding = (padding.x, padding.y)
                assert original_padding[0] == 0
                imgui.push_style_var(imgui.StyleVar_.item_spacing, (gap * scale, 8 * scale))
                panel.draw(ctx)
                imgui.pop_style_var()
                restored = imgui.get_style().cell_padding
                assert (restored.x, restored.y) == original_padding
            imgui.end()
            imgui.render()
        assert spans["2"][0] - spans["material"][1] >= gap * scale - 1
        assert panel._selected is None
    finally:
        imgui.destroy_context(context)
