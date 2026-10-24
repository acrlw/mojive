"""Registered panels retain canonical icons in the real Window submenu."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.adapters.static import StaticSceneAdapter
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import _BUILTIN_PANELS, Panel
from mojive.ui.workspace.constants import ICONS, PANEL_ICONS, PANELS
from mojive.ui.workspace.localization import workspace_text
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.panel_manager import WorkspacePanels
from mojive.ui.workspace.visibility import PanelVisibility
from mojive.ui.workspace.widget_icons import GLYPHS
from mojive.ui.workspace.widgets import UI


def test_every_builtin_panel_has_one_available_canonical_icon():
    panels = WorkspacePanels()
    assert set(PANEL_ICONS) == set(_BUILTIN_PANELS)
    assert set(PANEL_ICONS.values()) <= GLYPHS.keys()
    assert tuple(PANEL_ICONS[panels.get(name).id] for name in PANELS) == ICONS
    assert {panel.id for panel in panels} == set(PANEL_ICONS)
    assert {name: PANEL_ICONS[name] for name in ("plot", "help", "info")} == {
        "plot": "waveform",
        "help": "help",
        "info": "info",
    }


@pytest.mark.parametrize("scale", (1, 1.2, 1.25, 2.25))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_window_panels_paints_icons_for_all_registered_rows_and_keeps_columns_aligned(
    scale, language, monkeypatch
):
    context = imgui.create_context()
    session = Session(StaticSceneAdapter(Scene()))
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (1200 * scale, 1000 * scale)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        ui = UI(
            SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
            portable_fonts=True,
        )
        localizer = Localizer(Language(language))
        ui.translate = lambda text: workspace_text(localizer, text)
        panels = WorkspacePanels()
        workspace = SimpleNamespace(
            ui=ui,
            s=scale,
            app=SimpleNamespace(session=session, entity=None, panels=panels),
            open=PanelVisibility(panels),
            saved=False,
            show_viewport_tab=False,
        )
        menus = Menus(workspace, tuple(zip(PANELS, ICONS, strict=True)))
        painted = []
        original = ui.icon

        def paint_icon(name, *args, **kwargs):
            painted.append(name)
            return original(name, *args, **kwargs)

        monkeypatch.setattr(ui, "icon", paint_icon)
        menus.preview(("Window", "Panels"))
        for _ in range(4):
            painted.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Panel menu host")
            menus.begin_frame()
            menus.popup("Window", (20 * scale, 20 * scale, 100 * scale, 30 * scale))
            imgui.end()
            imgui.render()
        rows = [row for row in menus.items("Panels") if row]
        actual = menus.observed["Window/Panels"]["rows"]
        assert len(rows) == len(actual) == len(_BUILTIN_PANELS.keys() - {"settings", "info"})
        for item, record in zip(rows, actual, strict=True):
            panel_id = (
                "keyframes"
                if item.action == "timeline"
                else panels.get(item.action.partition(":")[2]).id
            )
            assert item.icon == PANEL_ICONS[panel_id]
            assert record["label"] == item.label
            assert record["visible"] and not record["clippedLabel"]
            assert record["iconSlotRect"][2] > 0
            assert item.icon in painted
        assert len({row["iconSlotRect"][0] for row in actual}) == 1
        assert len({row["label_rect"][0] for row in actual}) == 1

        class CustomPanel(Panel):
            id, name = "custom", "Custom diagnostics"

        panels.register(CustomPanel())
        custom = next(row for row in menus.items("Panels") if row and row.label == CustomPanel.name)
        assert custom.icon == "inspect"
        assert custom.action == "extra-panel:custom"
    finally:
        session.release()
        imgui.destroy_context(context)
