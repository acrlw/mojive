"""Settings previews, commit errors and callback contracts at the UI boundary."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.panels import PanelContext, settings
from mojive.ui.preferences import Preferences


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    session = Session(StaticSceneAdapter(Scene()))
    value = ViewerApp(session, NullBackend())
    yield value
    session.release()


def test_programmatic_save_failure_remains_an_error_while_ui_reports_unsaved_preview(
    app, monkeypatch
):
    original = app.camera.navigation
    requested = replace(original, zoom_speed=original.zoom_speed + 1)
    app.preferences.update({"camera_navigation": asdict(original)})
    original_bytes = app.preferences.path.read_bytes()

    def reject_replace(self, target):
        raise OSError("disk unavailable")

    with monkeypatch.context() as failing:
        failing.setattr(Path, "replace", reject_replace)
        # RPC and Python callers must still observe a failed persistence request.
        with pytest.raises(OSError, match="disk unavailable"):
            app.set_camera_navigation(requested)
        assert app.camera.navigation == requested
        assert app.preferences.path.read_bytes() == original_bytes
        ctx = PanelContext(app.session, app.backend)
        ctx.apply_setting(app.set_camera_navigation, requested)
        assert "could not be saved" in ctx.status and "disk unavailable" in ctx.status
        assert app.session.last_message == ctx.status
        assert app.camera.navigation == requested
        assert app.preferences.get("camera_navigation") == asdict(original)
    # A later successful commit persists exactly the accepted runtime value.
    app.set_camera_navigation(app.camera.navigation)
    assert Preferences.load(app.preferences.path).get("camera_navigation") == asdict(requested)


def test_setting_boundary_does_not_mask_validation_or_programming_errors(app):
    ctx = PanelContext(app.session, app.backend)
    with pytest.raises(ValueError, match="unknown viewport capsule"):
        ctx.apply_setting(app.set_viewport_capsule_scale, "missing", 2)
    with pytest.raises(TypeError):
        ctx.apply_setting(app.set_camera_navigation, object())


def test_view_padding_drag_previews_then_commits_once_and_idle_preserves_precision(
    app, monkeypatch
):
    panel = settings.SettingsPanel()
    phase = SimpleNamespace(changed=False, value=1.0, committed=False)
    saved = []
    save = app.preferences._save

    def record_save(values):
        saved.append(dict(values))
        save(values)

    monkeypatch.setattr(app.preferences, "_save", record_save)
    monkeypatch.setattr(panel, "_group_heading", lambda *args: None)
    monkeypatch.setattr(panel, "_begin_properties", lambda *args: True)
    monkeypatch.setattr(panel, "_property", lambda *args: None)
    monkeypatch.setattr(
        settings,
        "imgui",
        SimpleNamespace(
            drag_float=lambda identifier, value, *args: (
                phase.changed,
                phase.value if phase.changed else float(np.float32(value)),
            ),
            is_item_deactivated_after_edit=lambda: phase.committed,
            is_item_hovered=lambda: False,
            end_table=lambda: None,
        ),
    )
    ctx = PanelContext(
        app.session,
        app.backend,
        view_cube=app.view_cube,
        set_view_selection_padding=app.set_view_selection_padding,
    )
    original = app.view_cube.selection_padding
    panel._view_settings(ctx)
    assert saved == [] and app.view_cube.selection_padding == original
    phase.changed, phase.value = True, 1.234567890123
    panel._view_settings(ctx)
    assert saved == [] and app.view_cube.selection_padding == phase.value
    phase.changed, phase.committed = False, True
    panel._view_settings(ctx)
    assert len(saved) == 1
    assert saved[0]["view_selection_padding"] == phase.value
    phase.committed = False
    panel._view_settings(ctx)
    assert len(saved) == 1 and app.view_cube.selection_padding == phase.value


def test_failed_view_padding_commit_finishes_the_ui_table_and_keeps_preview(app, monkeypatch):
    panel = settings.SettingsPanel()
    completed = []
    monkeypatch.setattr(panel, "_group_heading", lambda *args: None)
    monkeypatch.setattr(panel, "_begin_properties", lambda *args: True)
    monkeypatch.setattr(panel, "_number_setting", lambda *args, **kwargs: (True, 1.7, True))
    monkeypatch.setattr(settings.imgui, "end_table", lambda: completed.append("table"))
    monkeypatch.setattr(
        app.preferences, "_save", lambda values: (_ for _ in ()).throw(OSError("full"))
    )
    ctx = PanelContext(
        app.session,
        app.backend,
        view_cube=app.view_cube,
        set_view_selection_padding=app.set_view_selection_padding,
    )
    panel._view_settings(ctx)
    assert completed == ["table"]
    assert app.view_cube.selection_padding == 1.7
    assert "could not be saved: full" in ctx.status
    assert app.preferences.get("view_selection_padding") is None
