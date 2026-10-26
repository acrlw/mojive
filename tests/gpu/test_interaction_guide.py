"""Interaction Guide presents current gestures through shared status indicators."""

from copy import deepcopy
from pathlib import Path

import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.themes import select_theme
from mojive.tools.native_ui_review import _content_window, _NativeInput, _records
from mojive.tools.native_workspace import capture
from mojive.ui.imgui_draw import ink_box
from mojive.ui.input_bindings import InputAction
from mojive.ui.workspace.interaction_guide import interaction_rows
from mojive.ui.workspace.menu_geometry import native_epsilon

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/interaction-guide-fix"


@pytest.mark.parametrize(
    "scale,language,theme,width",
    (
        (1, "en", "light", 500),
        (1.25, "zh_CN", "dark", 500),
        (1.25, "en", "dark", 340),
        (2, "en", "dark", 500),
    ),
)
def test_interaction_guide_reuses_status_indicators_and_scrolls_complete_content(
    backend_name, monkeypatch, tmp_path, scale, language, theme, width
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(tmp_path / "design.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    target = OUTPUT / f"{backend_name}-{scale:g}-{language}-{width}"
    target.mkdir(parents=True, exist_ok=True)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app = viewer.app
        app.set_language(language)
        app.set_input_binding(InputAction.FRAME_SCENE, "h")
        spec = deepcopy(app.design.spec)
        spec["parts"]["dialog"] = {"width": width}
        spec["parts"]["status_hints_badge"] = {"fontFamily": "mono", "fontSize": 12}
        app.design.update_spec(spec)
        select_theme(app.design, theme, persist=False)
        app.design.observe_components = True
        inputs.sync(8)
        app.workspace.menus.dialog_pending = "Interaction guide"
        inputs.sync(8)
        content = _content_window("Guide content")
        rows = [row for row in _records(viewer) if row["kind"] == "interaction_hint"]
        assert len(rows) == len(interaction_rows(app.input_bindings))
        assert any(item["text"] == "H" for item in inputs.emitted)
        if width < 400:
            assert any(row["stacked"] for row in rows)
        else:
            ui = app.workspace.ui
            events = [item for item in inputs.emitted if item["windowId"] == content.id_]
            fonts = {font.font_id: font for font in imgui.get_io().fonts.fonts}

            def text_record(value):
                return next(item for item in events if item["text"] == value)

            def cap_center(item):
                ink = ink_box(fonts[item["fontId"]], item["fontPixels"], "H")
                return item["position"][1] + (ink[1] + ink[3]) / 2

            caption = text_record(ui.translate("Orbit"))
            note = text_record(ui.translate("Drag an empty area"))
            pan_note = text_record(ui.translate("Drag"))
            precise_note = text_record(ui.translate("Double-click a handle"))
            key = text_record(app.input_bindings.label(InputAction.FLY_FORWARD))
            assert cap_center(caption) == pytest.approx(cap_center(note), abs=0.01 * scale)
            assert key["fontPixels"] == caption["fontPixels"]
            assert cap_center(key) == pytest.approx(rows[3]["centerY"], abs=0.01 * scale)
            assert note["position"][0] == pytest.approx(pan_note["position"][0])
            assert note["position"][0] == pytest.approx(precise_note["position"][0])
            assert spec["parts"]["status_hints_badge"]["fontSize"] == 12
            first_slots = [rows[index]["fields"][0] for index in (0, 1, 2, 3, 4, 5, 6)]
            centers = [rect[0] + rows[0]["inputSlot"] / 2 for rect in first_slots]
            assert centers == pytest.approx([centers[0]] * len(centers))
        for row in rows:
            for x, _y, w, _h in row["fields"]:
                epsilon = native_epsilon(
                    x, w, content.inner_clip_rect.min.x, content.inner_clip_rect.max.x
                )
                assert content.inner_clip_rect.min.x - epsilon <= x
                assert x + w <= content.inner_clip_rect.max.x + epsilon
        assert content.scroll_max.y > 0
        popup = imgui.internal.find_window_by_name("##workspace-dialog")

        def save(name):
            path = target / f"{name}.png"
            capture(viewer, path)
            with Image.open(path) as image:
                density = image.width / imgui.get_io().display_size.x
                bounds = (*popup.pos, popup.pos.x + popup.size.x, popup.pos.y + popup.size.y)
                image.crop(tuple(round(value * density) for value in bounds)).save(path)

        save("top")
        imgui.internal.set_scroll_y(content, content.scroll_max.y)
        inputs.sync(8)
        save("bottom")
        inputs.click("done-guide")
        assert not imgui.get_current_context().open_popup_stack
