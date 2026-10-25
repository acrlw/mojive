"""Exercise shared filters, navigation and About in hidden production windows."""

import json
from pathlib import Path

import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.tools.native_ui_review import _NativeInput, _records
from mojive.tools.native_workspace import capture

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
OUTPUT = Path(__file__).resolve().parents[2] / "output/control-consistency-2026-10-24"


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_filters_and_modal_keep_native_focus_and_localized_copy(
    backend_name, monkeypatch, tmp_path, scale
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    native_text = []
    for method in ("text_unformatted", "text_wrapped", "text"):
        original = getattr(imgui, method)

        def record_text(value, *args, _original=original, **kwargs):
            native_text.append(value)
            return _original(value, *args, **kwargs)

        monkeypatch.setattr(imgui, method, record_text)
    viewer = build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        width=1600,
        height=1100,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    target = OUTPUT / f"{backend_name}-{scale:g}"
    target.mkdir(parents=True, exist_ok=True)
    try:
        viewer.app.set_language("zh_CN")
        viewer.app.design.observe_components = True
        workspace = viewer.app.workspace
        workspace.search_visible = True
        workspace.expanded["IDENTITY"] = True
        node = next(n for n in viewer.session.nodes if n.name == "01_revolute_anchor")
        assert viewer.session.submit(cmd.SelectNode(node.node_id)).ok
        with _NativeInput(viewer) as inputs:
            inputs.sync(10)
            capture(viewer, target / "scene.png")
            scene = next(r for r in _records(viewer) if r["key"] == "##search")
            inputs.click("##search")
            inputs.event("add_input_characters_utf8", "revolute")
            inputs.sync()
            assert workspace.query == "revolute"
            inputs.click("##search-clear")
            inputs.sync()
            assert workspace.query == ""
            # Tab changes focus through ImGui; no OS events or desktop activation.
            inputs.event("add_key_event", imgui.Key.tab, True)
            inputs.event("add_key_event", imgui.Key.tab, False)
            capture(viewer, target / "search-focus.png")
            workspace.activate("Assets")
            inputs.sync()
            capture(viewer, target / "assets.png")
            (target / "asset-keys.json").write_text(json.dumps(list(workspace.ui.hits), indent=2))
            # Expand the localized section through its production hit region.
            inputs.click("section-" + viewer.app.localizer.text("Model assets"))
            inputs.sync()
            capture(viewer, target / "assets-inventory.png")
            asset = next(r for r in _records(viewer) if r["key"] == "##asset-filter")
            assert asset["iconSize"] == scene["iconSize"]
            assert asset["hit"][3] == scene["hit"][3]
            inputs.click("##asset-type")
            inputs.sync()
            choices = [r for r in _records(viewer) if r["key"].startswith("##asset-type/choice/")]
            assert len(choices) > 1
            assert all(not r["clippedLabel"] for r in choices)
            capture(viewer, target / "asset-type-popup.png")
            inputs.click(choices[-1]["key"])
            assert viewer.app.panels.get("assets")._asset_type != "all"
            inputs.click("##asset-type")
            inputs.click("##asset-type/choice/0")
            assert viewer.app.panels.get("assets")._asset_type == "all"
            workspace.activate("Output")
            output = viewer.app.panels.get("output")
            output.show_search, output._focus_search = True, True
            inputs.sync()
            filter_record = next(r for r in _records(viewer) if r["key"] == "##output-filter")
            assert filter_record["iconSize"] == scene["iconSize"]
            assert filter_record["hit"][3] == scene["hit"][3]
            inputs.click("##output-filter")
            inputs.event("add_input_characters_utf8", "joint")
            inputs.sync()
            assert output._filter_text == "joint"
            inputs.event("add_key_event", imgui.Key.tab, True)
            inputs.event("add_key_event", imgui.Key.tab, False)
            inputs.event("add_key_event", imgui.Key.mod_shift, True)
            inputs.event("add_key_event", imgui.Key.tab, True)
            inputs.event("add_key_event", imgui.Key.tab, False)
            inputs.event("add_key_event", imgui.Key.mod_shift, False)
            capture(viewer, target / "output-focus.png")
            inputs.click("##output-filter-clear")
            assert output._filter_text == ""
            inputs.hover("tool-scale")
            assert "尺寸" in native_text
            tooltip = next(r for r in _records(viewer) if r.get("kind") == "tooltip")
            assert tooltip["nativeRadius"] == pytest.approx(4 * scale)
            capture(viewer, target / "dimension-tooltip.png")
            inputs.move((0, 0))
            workspace.activate("Control")
            inputs.sync()
            capture(viewer, target / "control.png")
            workspace.menus.dispatch("dialog:About Mojive")
            inputs.sync()
            assert imgui.is_popup_open("##workspace-dialog", imgui.PopupFlags_.any_popup_id)
            capture(viewer, target / "about.png")
            assert "面向多后端的三维场景编辑与仿真工具。" in native_text
            assert any(text.startswith("提交: ") for text in native_text)
            inputs.escape()
            assert not imgui.is_popup_open("##workspace-dialog", imgui.PopupFlags_.any_popup_id)
            inputs.click("toggle-left")
            assert not workspace.open["Control"]
            inputs.click("toggle-left")
            assert workspace.open["Control"]
            inputs.click("toggle-inspector")
            assert not workspace.open["Inspector"]
            inputs.click("toggle-inspector")
            assert workspace.open["Inspector"]
            (target / "geometry.json").write_text(json.dumps(_records(viewer), indent=2))
    finally:
        viewer.release()
