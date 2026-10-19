"""Native actuator rows distinguish source metadata, missing values and write capability."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import ActuatorInfo, NodeType
from mojive.design.groups import group_scope
from mojive.design.native.geometry import part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.render.backend import NullBackend
from mojive.ui.panels import PanelContext
from mojive.ui.panels.control import ControlPanel
from mojive.ui.workspace import panel_host
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.scene_panels import control_panel
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 2.25))
def controls(request, monkeypatch, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (360 * request.param, 850 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    ui.design = DesignPresentation(ui, DesignRuntime(SpecificationStore(tmp_path / "design.json")))
    target = SimpleNamespace(node_id=10, name="hinge_axis", joint_index=0, type=NodeType.JOINT)
    actuators = (
        ActuatorInfo(0, "motor", (-0.2, 0.5), True, ctrl_count=2, target_node_id=10),
        ActuatorInfo(1, "position", (0.0, 0.0), False, ctrl_address=2, unit="rad"),
    )
    calls = []
    session = SimpleNamespace(
        actuators=actuators,
        nodes=(target,),
        node=lambda identifier: target if identifier == 10 else None,
        adapter=SimpleNamespace(
            caps=SimpleNamespace(write_ctrl=True, external_clock=False, state_snapshots=True)
        ),
        frame=SimpleNamespace(ctrl=np.array([0.1, 0.2, 0.3])),
        selection_revision=0,
        structure_generation=0,
        equality_constraints=(),
        submit=lambda command: calls.append(command) or SimpleNamespace(ok=True),
    )
    ctx = PanelContext(session, NullBackend(), style_scale=ui.s)
    panel = ControlPanel()
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        _panel_context=ctx,
        app=SimpleNamespace(session=session, panels=SimpleNamespace(get=lambda _name: panel)),
        activate=lambda name: calls.append(name),
        control_search_visible=False,
        header_actions={"Control": {"reset"}},
        open={"Control": True},
    )
    workspace.menus = Menus(workspace, ())
    workspace.header_action = lambda name, key: panel_host.header_action(workspace, name, key)
    state = SimpleNamespace(
        ui=ui,
        session=session,
        calls=calls,
        panel=panel,
        rails=[],
        text=[],
        fonts={},
        force_edit=False,
        header=False,
        workspace=workspace,
    )
    original_rail, original_text = ui.rail, ui.component_text
    original_paint_text = ui.text

    def rail(key, x, y, width, value, bounds, **options):
        state.rails.append((key, value, bounds, options))
        result = original_rail(key, x, y, width, value, bounds, **options)
        return SimpleNamespace(changed=True, value=value + 0.1) if state.force_edit else result

    def text(part, rect, value, *args, **options):
        state.text.append(str(value))
        return original_text(part, rect, value, *args, **options)

    def paint_text(x, y, value, color, size, **options):
        state.fonts[str(value)] = size
        return original_paint_text(x, y, value, color, size, **options)

    monkeypatch.setattr(ui, "rail", rail)
    monkeypatch.setattr(ui, "component_text", text)
    monkeypatch.setattr(ui, "text", paint_text)

    def paint():
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Control", flags=imgui.WindowFlags_.no_title_bar)
        state.rails.clear()
        state.text.clear()
        state.fonts.clear()
        ui.hits.clear()
        with group_scope("Control"), part_scope("panel_body"):
            top = 20 * ui.s
            if state.header:
                top = panel_host.panel_header(workspace, "Control", 20 * ui.s, top, 280 * ui.s)
            control_panel(workspace, 20 * ui.s, top, 280 * ui.s)
        imgui.end()
        imgui.render()

    def click(key):
        x, y, width, height = ui.hits[key]
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        paint()
        io.add_mouse_button_event(0, True)
        paint()
        io.add_mouse_button_event(0, False)
        paint()

    state.paint, state.click = paint, click
    state.io = io
    try:
        for _ in range(3):
            paint()
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("values", (None, (), (0.12,), (0.12, np.nan, 0.34)))
def test_unavailable_control_components_never_paint_a_fabricated_numeric_value(controls, values):
    state = controls
    state.session.frame.ctrl = None if values is None else np.array(values)
    state.paint()
    available = (
        [] if values is None else [i for i, value in enumerate(values) if np.isfinite(value)]
    )
    assert [key for key, *_ in state.rails] == ["##ctrl-" + str(i) for i in available]
    assert state.text.count("Control value unavailable") == 3 - len(available)
    state.click("reset-actuators")
    assert state.calls == []
    state.click("open-pose")
    assert state.calls == ["Pose"]


def test_target_and_range_metadata_use_declared_units_and_fit_before_values(controls):
    state = controls
    assert "hinge_axis · -0.2 … 0.5" in state.text
    assert "Target not specified · Unbounded" in state.text
    assert state.rails[0][3]["unit"] == ""
    assert state.rails[2][3]["unit"] == "rad"
    for address in range(3):
        hx, hy, hw, hh = state.ui.hits["control-target-" + str(address)]
        mx, my, mw, mh = state.ui.hits["control-meta-" + str(address)]
        vx, vy, vw, _vh = state.ui.hits["##ctrl-" + str(address)]
        assert my >= hy + hh
        assert vy > my + mh
        assert hx <= mx and mx + mw <= hx + hw
        assert hx <= vx and vx + vw <= hx + hw
    reset, pose = (state.ui.hits[key] for key in ("reset-actuators", "open-pose"))
    assert reset[1] == pose[1]
    assert reset[0] + reset[2] < pose[0]


@pytest.mark.parametrize("part", ("control_metadata", "control_panel_body"))
def test_metadata_defaults_to_secondary_type_and_honors_local_font_edits(controls, part):
    state = controls
    detail = "hinge_axis · -0.2 … 0.5"
    assert state.fonts[detail] == 12
    runtime = state.ui.design.runtime
    spec = deepcopy(runtime.spec)
    spec["parts"][part] = {"fontSize": 19}
    runtime.update_spec(spec)
    state.paint()
    assert state.fonts[detail] == 19


@pytest.mark.parametrize("external", (False, True))
def test_read_only_source_explains_why_and_blocks_commands(controls, external):
    state = controls
    state.session.adapter.caps.write_ctrl = False
    state.session.adapter.caps.external_clock = external
    state.force_edit = True
    state.paint()
    assert state.calls == []
    reason = (
        "Read only · external control" if external else "Read only · control writes unavailable"
    )
    assert reason in state.text
    assert len(state.rails) == 3
    state.click("reset-actuators")
    assert state.calls == []


@pytest.mark.parametrize("limits", ((2.0, 1.0), (np.nan, 1.0), (0.0, np.inf)))
def test_invalid_declared_limits_are_not_presented_as_writable_unbounded_controls(controls, limits):
    state = controls
    state.session.actuators[0].ctrl_range = limits
    state.force_edit = True
    state.paint()
    assert "hinge_axis · Invalid control range" in state.text
    assert all(row[2] is None for row in state.rails[:2])
    assert len(state.calls) == 1
    assert isinstance(state.calls[0], cmd.SetCtrl)
    assert state.calls[0].index == 2


def test_writable_rows_keep_the_existing_command_address_boundary(controls):
    state = controls
    state.force_edit = True
    state.paint()
    assert [command.index for command in state.calls] == [0, 1, 2]
    assert all(isinstance(command, cmd.SetCtrl) for command in state.calls)
    state.force_edit = False
    state.calls.clear()
    state.click("reset-actuators")
    assert len(state.calls) == 1 and isinstance(state.calls[0], cmd.SetCtrlVector)
    assert np.array_equal(state.calls[0].values, np.zeros(3))


def _control_menu(state, key):
    state.click("header-Control-more")
    state.paint()
    state.click("header-menu-Control-" + key)
    state.paint()


def test_native_control_menu_restores_sorting_without_readdressing_controls(controls):
    state = controls
    state.header = True
    state.session.actuators[0].name = "Zulu"
    state.session.actuators[1].name = "Alpha"
    state.session.structure_generation += 1
    state.paint()
    initial = [row[0] for row in state.rails]
    assert initial == ["##ctrl-0", "##ctrl-1", "##ctrl-2"]
    _control_menu(state, "sort")
    assert [row[0] for row in state.rails] == ["##ctrl-2", "##ctrl-0", "##ctrl-1"]
    assert panel_host.header_action_active(state.workspace, "Control", "sort")
    state.force_edit = True
    state.paint()
    assert [command.index for command in state.calls] == [2, 0, 1]
    assert all(isinstance(command, cmd.SetCtrl) for command in state.calls)
    state.force_edit = False
    _control_menu(state, "sort")
    assert [row[0] for row in state.rails] == initial


def test_native_control_search_filters_actual_rows_and_clears_when_hidden(controls):
    state = controls
    state.header = True
    state.paint()
    assert "##control-search" not in state.ui.hits
    _control_menu(state, "search")
    state.click("##control-search")
    state.io.add_input_characters_utf8("MoToR")
    state.paint()
    assert state.panel._search == "MoToR"
    assert [row[0] for row in state.rails] == ["##ctrl-0", "##ctrl-1"]
    state.io.add_input_characters_utf8(" missing")
    state.paint()
    assert not state.rails
    assert "No matching actuators" in state.fonts
    _control_menu(state, "search")
    assert not state.panel._search and "##control-search" not in state.ui.hits
    assert [row[0] for row in state.rails] == ["##ctrl-0", "##ctrl-1", "##ctrl-2"]


def test_native_control_copy_entries_use_current_vectors_and_snapshot_only_on_request(
    controls, monkeypatch
):
    state, copied, captures = controls, [], []
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)

    def capture():
        captures.append(True)
        return SimpleNamespace(act=np.array([0.75, -0.25]))

    state.session.adapter.capture_state = capture
    state.header = True
    state.paint()
    _control_menu(state, "copy-ctrl")
    assert copied == ["[0.1, 0.2, 0.3]"] and not captures
    state.session.frame.ctrl = np.array([1.0, 2.0, 3.0])
    _control_menu(state, "copy-ctrl")
    assert copied[-1] == "[1.0, 2.0, 3.0]" and not captures
    _control_menu(state, "copy-act")
    assert copied[-1] == "[0.75, -0.25]" and captures == [True]
    for _ in range(4):
        state.paint()
    assert captures == [True]
    state.session.adapter.caps.state_snapshots = False
    _control_menu(state, "copy-act")
    assert captures == [True] and len(copied) == 3
    state.session.frame.ctrl = None
    _control_menu(state, "copy-ctrl")
    assert len(copied) == 3


def test_actuator_rows_reuse_lookup_until_structure_or_filter_changes(controls):
    state = controls
    panel, ctx = state.panel, state.workspace._panel_context
    rows, targets = panel.actuator_rows(ctx), panel._joint_nodes
    state.paint()
    assert panel.actuator_rows(ctx) is rows and panel._joint_nodes is targets
    panel._search = "position"
    assert [(a.ctrl_address, component) for a, component in panel.actuator_rows(ctx)] == [(2, 0)]
    panel._search = ""
    state.session.actuators = (state.session.actuators[1],)
    state.session.nodes = ()
    state.session.structure_generation += 1
    state.paint()
    assert [row[0] for row in state.rails] == ["##ctrl-2"]
    assert not panel._joint_nodes
