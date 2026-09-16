"""Feature selection controls construction, imports, frame needs and later activation."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from mojive import ViewerConfig
from mojive.adapters.base import FrameNeeds, ModelAssetInfo, NodeType, SceneNode
from mojive.adapters.static import StaticSceneAdapter
from mojive.config import PanelConfig
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.panels import PanelContext, PanelManager


@pytest.mark.parametrize("config", [None, ViewerConfig()])
def test_default_viewer_exposes_full_tools_without_opt_in(tmp_path, monkeypatch, config):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    app = ViewerApp(Session(StaticSceneAdapter(Scene())), NullBackend(), None, config=config)
    try:
        assert not app.panels.unloaded_builtins()
        assert all(panel.enabled for panel in app.panels)
        for name in ("inspector", "control", "joints", "camera", "keyframes", "hierarchy"):
            assert app.panels.get(name).open
        assert app.interactions.gizmo and app.interactions.perturb
        assert app.interactions.playback_shortcuts and app.interactions.panel_shortcuts
        assert app.viewport_layers.helpers and app.viewport_layers.gizmos
        assert app.viewport_layers.viewport_ui and app.selection_style.gizmo
    finally:
        app.release()


def test_minimal_viewer_does_not_import_or_construct_unused_panels():
    # Use a fresh interpreter: imports from other tests must not mask eager loading.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from mojive import ViewerConfig
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.app import ViewerApp
app = ViewerApp(Session(StaticSceneAdapter(Scene())), NullBackend(), None,
                config=ViewerConfig.minimal("inspector", "control", "joints", "camera"))
assert {p.id for p in app.panels} == {"inspector", "control", "joints", "camera"}
assert "mojive.ui.panels.keyframes" not in sys.modules
assert "mojive.ui.keyframe_editor.controller" not in sys.modules
assert "mojive.ui.panels.assets" not in sys.modules
assert app.panels.open("keyframes")
assert "mojive.ui.keyframe_editor.controller" in sys.modules
app.release()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_empty_panel_set_requests_no_diagnostics_and_loads_only_on_activation():
    manager = PanelManager(builtin_ids=(), config={"sensors": PanelConfig(enabled=False)})
    assert list(manager) == []
    assert manager.frame_needs() == FrameNeeds.none()
    assert manager.get("keyframes") is None
    assert not manager.close("keyframes")
    assert manager.open("Keyframes")
    assert len(list(manager)) == 1
    assert manager.open("keyframes") and len(list(manager)) == 1
    assert not manager.open("sensors")
    assert not manager.state("sensors").enabled
    assert manager.enable("sensors") and manager.open("sensors")
    assert manager.frame_needs().sensors
    assert not PanelManager(panels=[]).open("keyframes")


@pytest.mark.parametrize("ids", [("inspector", "inspector"), ("does-not-exist",)])
def test_builtin_selection_rejects_invalid_ids(ids):
    with pytest.raises(ValueError):
        PanelManager(builtin_ids=ids)


def test_minimal_preset_preserves_navigation_and_debug_layers(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    config = ViewerConfig.minimal()
    session = Session(StaticSceneAdapter(Scene()))
    app = ViewerApp(session, NullBackend(), None, config=config)
    try:
        assert app.interactions.camera.orbit and app.interactions.selection.pick
        assert not app.interactions.gizmo and not app.interactions.perturb
        assert not app.viewport_layers.helpers and not app.viewport_layers.viewport_ui
        assert app.viewport_layers.debug_2d and app.viewport_layers.debug_3d
        assert not config.debug_server and not config.layout.persistence
        assert list(app.panels) == []
    finally:
        app.release()


@pytest.mark.parametrize("node_type", [NodeType.GEOM, NodeType.SITE])
@pytest.mark.parametrize("same_model", [False, True])
def test_asset_material_assignment_uses_selected_source_node(monkeypatch, node_type, same_model):
    from mojive import commands as cmd
    from mojive.ui.panels import assets

    node = SceneNode(
        31, "surface", node_type, model_id=2 if same_model else 9, source_editable=True
    )
    item = ModelAssetInfo(2, "material", "paint", 0, runtime_index=7)
    commands = []
    ctx = PanelContext(SimpleNamespace(selected_node=node, submit=commands.append), NullBackend())
    ctx.submit = commands.append
    monkeypatch.setattr(assets.imgui, "button", lambda *args: True)
    monkeypatch.setattr(assets.imgui, "text_disabled", lambda *args: None)
    assets.AssetsPanel._assignment_control(ctx, item)
    assert commands == ([cmd.SetGeometryMaterial(31, 7)] if same_model else [])


def test_minimal_composition_does_not_start_debug_socket(tmp_path, monkeypatch):
    from mojive.app.composition import build_scene
    from mojive.app.ui import window as window_factory
    from mojive.remote.bridge import DebugBridge
    from mojive.render.opengl import backend as backend_module

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    window = SimpleNamespace(
        size_pixels=(640, 480),
        make_current=lambda: None,
        apply_theme=lambda theme: None,
        close=lambda: None,
    )
    monkeypatch.setattr(window_factory, "create_window", lambda *args: window)
    monkeypatch.setattr(backend_module, "OpenGLBackend", lambda *args: NullBackend())

    def unexpected(*args):
        raise AssertionError("Minimal composition must not open a debug socket")

    monkeypatch.setattr(DebugBridge, "serve", unexpected)
    with build_scene(Scene(), config=ViewerConfig.minimal(), renderer="opengl") as viewer:
        assert viewer.bridge.socket is None
        assert list(viewer.panels) == []
