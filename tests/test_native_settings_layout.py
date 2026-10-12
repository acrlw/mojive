"""Native settings keep checkbox geometry across section and page boundaries."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.render.backend import NullBackend, RenderFlag
from mojive.ui.panels.settings_pages import rendering, visuals
from mojive.ui.theme import THEME
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.property_style import property_controls
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.widgets import UI


@pytest.mark.parametrize("scale", (1, 2, 2.25))
def test_visual_group_and_flag_checkboxes_keep_the_same_size(scale, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * scale, 600 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s, ui.translate = scale, str
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.hits, ui.numeric_edits = {}, {}
    panel = WorkspaceSettings(SimpleNamespace(ui=ui, s=scale))
    group = SimpleNamespace(category="geom", visible=(True,) * 6)
    ctx = SimpleNamespace(
        session=SimpleNamespace(visual_groups=lambda: (group,)),
        backend=NullBackend(),
        tr=str,
        style_scale=scale,
        theme=THEME,
    )
    bounds = {}
    checkbox = rendering.themed_checkbox

    def capture(key, value, theme):
        result = checkbox(key, value, theme)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        bounds[key] = (hi.x - lo.x, hi.y - lo.y)
        return result

    monkeypatch.setattr(rendering, "themed_checkbox", capture)
    monkeypatch.setattr(visuals, "themed_checkbox", capture)
    try:
        for _ in range(3):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Native settings", flags=imgui.WindowFlags_.no_title_bar)
            with property_controls(ui):
                panel._visual_groups(ctx)
                panel._flag_table(ctx, "flags", (RenderFlag.SHADOW,))
                if panel._begin_properties("numeric after checkboxes"):
                    panel._property("Depth")
                    panel._drag_int("##bvh_depth", 3, 1, 0, 64)
                    imgui.end_table()
            imgui.end()
            imgui.render()
        assert len(bounds) == 7
        for size in bounds.values():
            assert size == pytest.approx((DESIGN.checkbox_size * scale,) * 2, abs=1)
        assert ui.hits["##bvh_depth"][3] == pytest.approx(DESIGN.control_height * scale, abs=1)
    finally:
        imgui.destroy_context(context)
