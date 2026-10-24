"""Production mode captions and menu groups stay readable without decorative icons."""

import json
from pathlib import Path

import pytest

from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.render.geometry import GeometryView, geometry_view
from mojive.tools.native_ui_review import _NativeInput, _records
from mojive.tools.native_workspace import capture

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
OUTPUT = Path("output/editor-tail-and-menu-2026-10-24/menus")


@pytest.mark.parametrize("scale", (1.0, 1.25))
def test_geometry_captions_and_semantic_menu_groups(backend_name, monkeypatch, tmp_path, scale):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        width=1600,
        height=1000,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    output = OUTPUT / f"{backend_name}-{scale:g}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        viewer.app.design.observe_components = True
        workspace = viewer.app.workspace
        with _NativeInput(viewer) as inputs:
            for language in ("zh_CN", "en"):
                viewer.app.set_language(language)
                workspace.activate("Scene")
                inputs.sync(10)
                records = [r for r in _records(viewer) if r["key"].startswith("scene-geometry-")]
                choices = [r for r in records if r.get("label")]
                assert len(choices) == 4
                assert all(r["label"] for r in choices)
                assert len({r["fontSize"] for r in choices}) == 1
                for view in (GeometryView.COLLISION, GeometryView.DEFAULT):
                    inputs.click("scene-geometry-" + view.value)
                    assert geometry_view(viewer.backend) is view
                inputs.move((800, 650))
                capture(viewer, output / f"{language}-scene.png")
                for menu in ("Simulate", "Window", "Help", "File", "Add"):
                    inputs.click("menu-" + menu)
                    inputs.sync()
                    capture(viewer, output / f"{language}-{menu.lower()}.png")
                    rows = workspace.menus.observed[menu]["rows"]
                    (output / f"{language}-{menu.lower()}.json").write_text(
                        json.dumps(rows, indent=2, ensure_ascii=False)
                    )
                    if menu == "Simulate":
                        speed = next(r for r in rows if r.get("label") == "Playback speed")
                        play = next(r for r in rows if r.get("label") == "Play simulation")
                        assert speed["iconSlotRect"][2] == 0
                        assert play["iconSlotRect"][2] > 0
                        measured = {
                            r["label"]: r
                            for r in _records(viewer)
                            if "labelRect" in r and r.get("label")
                        }
                        assert (
                            measured["Playback speed"]["labelRect"][0]
                            < measured["Play simulation"]["labelRect"][0]
                        )
                        inputs.click("menu-item-Simulate/Playback speed")
                        inputs.sync()
                        capture(viewer, output / f"{language}-speed.png")
                        inputs.click("menu-item-Simulate/Playback speed/0.5× speed")
                        assert workspace.speed == 0.5
                        workspace.speed = 1
                    inputs.escape()
                    inputs.move((0, 0))
    finally:
        viewer.release()
