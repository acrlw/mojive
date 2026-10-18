"""Explicit component inks remain literal when a global palette token changes."""

from copy import deepcopy
from types import SimpleNamespace

from imgui_bundle import imgui

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore
from mojive.ui.design_palette import PALETTE, rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import THEME
from mojive.ui.workspace import widgets
from mojive.ui.workspace.widgets import UI


def test_component_colors_outrank_global_remapping_even_when_they_match_palette_defaults(
    tmp_path, monkeypatch
):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800, 600)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    window = SimpleNamespace(
        style_scale=1.0,
        font_report=SimpleNamespace(cjk_path=None),
        _imgui_context=context,
        apply_theme=lambda theme: None,
    )
    ui = UI(window)
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    spec = deepcopy(DEFAULT)
    spec["tokens"].update(text_color="#ff0000", muted_color="#00ff00")
    spec["parts"]["button"] = {"textColor": PALETTE["text"], "iconColor": PALETTE["secondary"]}
    spec["parts"]["tab"] = {"textColor": "inherit", "iconColor": "inherit"}
    runtime.update_spec(spec)
    runtime.apply_theme(SimpleNamespace(window=window, app=SimpleNamespace(theme=THEME), s=1.0))
    ui.design = DesignPresentation(ui, runtime)
    text_inks, icon_inks = [], []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        text_inks.append(color)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    monkeypatch.setattr(
        widgets, "draw_icon", lambda draw, name, x, y, size, tint, scale: icon_inks.append(tint)
    )
    try:
        imgui.new_frame()
        imgui.begin("Appearance")
        ui.button("explicit", 30, 40, 220, 32, icon="camera", label="Explicit")
        ui.button("tab-global", 30, 90, 220, 32, icon="camera", label="Global")
        ui.button("unscoped", 30, 140, 32, 32, icon="camera")
        imgui.end()
        imgui.render()
        assert text_inks == [tuple(channel / 255 for channel in rgba8("text")), (1, 0, 0, 1)]
        assert icon_inks == [rgba8("secondary"), (255, 0, 0, 255), (0, 255, 0, 255)]
    finally:
        imgui.destroy_context(context)
