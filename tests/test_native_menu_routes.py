"""Native menus retain help, configured shortcuts and lazy panel activation."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.adapters.base import AdapterCaps
from mojive.capture import RecordingPhase
from mojive.config import PanelConfig
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction
from mojive.ui.panels.output import OutputPanel
from mojive.ui.workspace.constants import ICONS, PANELS
from mojive.ui.workspace.menu_actions import dispatch
from mojive.ui.workspace.menu_catalog import items
from mojive.ui.workspace.output_panel import WorkspaceOutput
from mojive.ui.workspace.panel_manager import WorkspacePanels
from mojive.ui.workspace.property_style import NativePropertyPresentation
from mojive.ui.workspace.sensors_panel import WorkspaceSensors
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.shell import WorkspaceShell
from mojive.ui.workspace.statistics import WorkspaceStatistics
from mojive.ui.workspace.visibility import PanelVisibility


@pytest.fixture
def menu():
    context = imgui.create_context()
    panels = WorkspacePanels(builtin_ids=("hierarchy",))
    workspace = WorkspaceShell.__new__(WorkspaceShell)
    workspace.app = SimpleNamespace(
        entity=None,
        panels=panels,
        input_bindings=DEFAULT_INPUT_BINDINGS,
        tool="select",
        gizmos=True,
        camera=SimpleNamespace(orthographic=False),
        recording=SimpleNamespace(phase=RecordingPhase.IDLE, active=False),
        session=SimpleNamespace(
            adapter=SimpleNamespace(caps=AdapterCaps()),
            paused=True,
            state_take_playing=False,
            state_take_recording=False,
        ),
    )
    workspace.ui = SimpleNamespace()
    workspace.open = PanelVisibility(panels)
    workspace.settings = WorkspaceSettings(workspace)
    workspace.left_active = "Scene"
    workspace.bottom_active = "Timeline"
    panels.bind_workspace(workspace)
    try:
        yield SimpleNamespace(workspace=workspace, panels=tuple(zip(PANELS, ICONS, strict=True)))
    finally:
        imgui.destroy_context(context)


def test_help_documentation_routes_to_existing_project_readme(menu, monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    row = next(row for row in items(menu, "Help") if row.label == "Documentation")
    dispatch(menu, row.action)
    assert opened == ["https://github.com/acrlw/mojive#readme"]


def test_shortcut_labels_follow_remapping_and_omit_unbound_actions(menu):
    app = menu.workspace.app
    tool_rows = {row.label: row for row in items(menu, "Transform tool")}
    assert {name: row.shortcut for name, row in tool_rows.items()} == {
        "Select": "V",
        "Move": "G",
        "Rotate": "R",
        "Scale": "",
    }
    app.input_bindings = (
        DEFAULT_INPUT_BINDINGS.remap(InputAction.GIZMO_TRANSLATE, "h")
        .remap(InputAction.GIZMO_ROTATE, None)
        .remap(InputAction.GIZMO_DIMENSIONS, "j")
        .remap(InputAction.FRAME_SCENE, "k")
        .remap(InputAction.TOGGLE_PAUSE, "p")
    )
    assert {row.label: row.shortcut for row in items(menu, "Transform tool")} == {
        "Select": "V",
        "Move": "H",
        "Rotate": "",
        "Scale": "J",
    }
    for name in ("View", "Camera"):
        assert (
            next(row.shortcut for row in items(menu, name) if row and row.action == "frame") == "K"
        )
    assert (
        next(row.shortcut for row in items(menu, "Simulate") if row and row.action == "play") == "P"
    )


def test_lazy_panels_are_discoverable_without_eager_construction(menu):
    panels = menu.workspace.app.panels
    rows = {row.label: row for row in items(menu, "Panels") if row}
    assert set(rows) == (set(PANELS) - {"Settings"}) | {
        "Inspector",
        "Timeline",
        "Plot",
        "Help",
        "Info",
    }
    assert all(row.enabled for row in rows.values())
    assert [panel.id for panel in panels] == ["hierarchy"]
    dispatch(menu, rows["Control"].action)
    assert panels.get("control").open
    assert menu.workspace.left_active == "Control"
    assert not panels.get("hierarchy").open
    dispatch(menu, rows["Timeline"].action)
    assert panels.get("keyframes").open
    assert menu.workspace.bottom_active == "Timeline"
    dispatch(menu, rows["Plot"].action)
    assert panels.get("plot").open
    assert menu.workspace.bottom_active == "Plot"
    assert isinstance(panels.get("plot").presentation, NativePropertyPresentation)


@pytest.mark.parametrize(
    "panel_id, panel_type",
    [("output", WorkspaceOutput), ("stats", WorkspaceStatistics), ("sensors", WorkspaceSensors)],
)
def test_late_panel_loading_keeps_native_presenter_and_single_instance(menu, panel_id, panel_type):
    panels = menu.workspace.app.panels
    panel = panels.load(panel_id)
    assert isinstance(panel, panel_type)
    assert panel is panels.get(panel_id)
    assert panels.open(panel_id)
    assert panels.close(panel_id)
    assert panels.open(panel_id)
    assert panels.get(panel_id) is panel
    assert sum(existing.id == panel_id for existing in panels) == 1


@pytest.mark.parametrize("first_name", (0, 1, 2))
@pytest.mark.parametrize(
    "names",
    [
        ("camera", "Camera", "Cameras"),
        ("hierarchy", "Hierarchy", "Scene"),
        ("joints", "Joints", "Pose"),
        ("keyframes", "Keyframes", "Timeline"),
        ("stats", "Stats", "Statistics"),
    ],
)
def test_lazy_panels_preserve_legacy_titles_through_close_and_reopen(menu, names, first_name):
    workspace = menu.workspace
    panels = WorkspacePanels(builtin_ids=())
    workspace.app.panels = panels
    workspace.open = PanelVisibility(panels)
    panels.bind_workspace(workspace)
    assert panels.open(names[first_name])
    panel = panels.get(names[0])
    for name in names:
        assert panels.get(name) is panel
        assert panels.close(name)
        assert not panel.open
        assert panels.open(name)
        assert panel.open
        assert panels.load(name) is panel
    assert list(panels) == [panel]


def test_about_loads_info_and_settings_uses_shell_instance(menu):
    panels = menu.workspace.app.panels
    row = next(row for row in items(menu, "Help") if row.label == "About Mojive")
    dispatch(menu, row.action)
    assert panels.get("info").open
    assert isinstance(panels.get("info").presentation, NativePropertyPresentation)
    dispatch(menu, "panel:Settings")
    assert panels.get("settings") is menu.workspace.settings
    assert menu.workspace.open["Settings"]


def test_explicit_custom_panel_registration_retains_its_implementation(menu):
    class CustomOutput(OutputPanel):
        pass

    panels = menu.workspace.app.panels
    custom = CustomOutput()
    custom.open = False
    panels.register(custom)
    dispatch(menu, "toggle:Output")
    assert panels.get("output") is custom
    assert custom.open


def test_disabled_lazy_panels_stay_unavailable(menu):
    workspace = menu.workspace
    panels = WorkspacePanels(
        builtin_ids=(),
        config={name: PanelConfig(enabled=False) for name in ("sensors", "inspector", "help")},
    )
    workspace.app.panels = panels
    workspace.open = PanelVisibility(panels)
    panels.bind_workspace(workspace)
    rows = {row.label: row for row in items(menu, "Panels") if row}
    assert not rows["Sensors"].enabled
    assert not rows["Inspector"].enabled
    assert "Help" not in rows
    dispatch(menu, "toggle:Sensors")
    assert not workspace.open["Sensors"]
    assert not panels.get("sensors").enabled
