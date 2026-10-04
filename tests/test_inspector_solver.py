"""Inspector solver and geometry controls preserve precision and complete draft transactions."""

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import (
    GeometryAdvancedProperties,
    GeometryProperties,
    JointAdvancedProperties,
)
from mojive.ui.panels import PanelContext
from mojive.ui.panels.inspector import InspectorPanel, fields, geometry, physics
from mojive.ui.theme import THEME

REFERENCE = (0.020123456789, 1.23456789123)
IMPEDANCE = (0.80123456789, 0.95123456789, 0.00123456789, 0.5123456789, 2.123456789)


class _Inputs:
    def __init__(self):
        self.edits = {}
        self.buttons = set()
        self.button_states = {}
        self.disabled = 0
        self.calls = []
        self.rows = []
        self.messages = []

    def edit(self, label, value, *args):
        self.calls.append((label, value, args, bool(self.disabled)))
        changed = not self.disabled and label in self.edits
        value = self.edits.pop(label) if changed else value
        # ImGui float widgets round-trip even idle components through C++ float.
        if isinstance(value, (tuple, list, np.ndarray)):
            value = np.asarray(value, np.float32).tolist()
        elif isinstance(value, (float, np.floating)):
            value = float(np.float32(value))
        return changed, value

    def button(self, label):
        self.button_states[label] = not bool(self.disabled)
        if self.disabled or label not in self.buttons:
            return False
        self.buttons.remove(label)
        return True

    def begin_disabled(self):
        self.disabled += 1

    def end_disabled(self):
        self.disabled -= 1
        assert self.disabled >= 0


@pytest.fixture
def inputs(monkeypatch):
    values = _Inputs()
    for module in (fields, physics, geometry):
        monkeypatch.setattr(
            module, "_property_control_row", lambda _ctx, label, **_kw: values.rows.append(label)
        )
    for module in (physics, geometry):
        monkeypatch.setattr(module, "_begin_property_table", lambda _name: True)
    monkeypatch.setattr(fields, "button_width", lambda label: len(label) * 8.0)
    monkeypatch.setattr(imgui, "get_content_region_avail", lambda: SimpleNamespace(x=300))
    monkeypatch.setattr(
        imgui, "get_style", lambda: SimpleNamespace(item_spacing=SimpleNamespace(x=4))
    )
    monkeypatch.setattr(physics, "_property_section", lambda *_args: True)
    monkeypatch.setattr(
        geometry,
        "_property_vector_row",
        lambda _ctx, _node, _label, _name, value, **_kw: (False, np.asarray(value, np.float32)),
    )
    for name in (
        "drag_float",
        "drag_float2",
        "drag_float3",
        "drag_int",
        "input_int",
        "combo",
        "checkbox",
    ):
        monkeypatch.setattr(imgui, name, values.edit)
    for name in ("end_table", "set_item_tooltip", "same_line", "set_clipboard_text"):
        monkeypatch.setattr(imgui, name, lambda *_args: None)
    monkeypatch.setattr(imgui, "text_colored", lambda _color, text: values.messages.append(text))
    monkeypatch.setattr(imgui, "text_disabled", values.messages.append)
    monkeypatch.setattr(imgui, "collapsing_header", lambda *_args: True)
    monkeypatch.setattr(imgui, "is_any_item_active", lambda: False)
    monkeypatch.setattr(imgui, "button", values.button)
    monkeypatch.setattr(imgui, "begin_disabled", values.begin_disabled)
    monkeypatch.setattr(imgui, "end_disabled", values.end_disabled)
    yield values
    assert values.disabled == 0


@pytest.mark.parametrize(
    "name,prefix", (("joint_limit", "limit "), ("joint_friction", "friction "), ("contact", ""))
)
def test_solver_rows_keep_widget_contract_and_exact_idle_values(inputs, name, prefix):
    result = fields._property_solver_rows(
        SimpleNamespace(tr=lambda text: text), name, REFERENCE, IMPEDANCE, label_prefix=prefix
    )
    assert result == (False, REFERENCE, IMPEDANCE)
    assert inputs.rows == [
        prefix + "solver reference",
        prefix + "impedance min / max / width",
        prefix + "impedance midpoint / power",
    ]
    assert [(call[0], call[2]) for call in inputs.calls] == [
        (f"##{name}_solver_reference", (0.001, -1000000.0, 1000000.0, "%.5g")),
        (f"##{name}_impedance_first", (0.001, 0.0, 1.0, "%.5g")),
        (f"##{name}_impedance_shape", (0.01, 0.0, 1000.0, "%.4g")),
    ]


@pytest.mark.parametrize(
    "row,index", (("solver_reference", 0), ("impedance_first", 1), ("impedance_shape", 1))
)
def test_solver_edit_retains_other_components_at_authored_precision(inputs, row, index):
    original = (
        REFERENCE
        if row == "solver_reference"
        else IMPEDANCE[:3]
        if row == "impedance_first"
        else IMPEDANCE[3:]
    )
    edited = list(original)
    edited[index] = 0.25
    inputs.edits[f"##contact_{row}"] = edited
    changed, reference, impedance = fields._property_solver_rows(
        SimpleNamespace(tr=lambda text: text), "contact", REFERENCE, IMPEDANCE
    )
    assert changed
    expected_reference = list(REFERENCE)
    expected_impedance = list(IMPEDANCE)
    if row == "solver_reference":
        expected_reference[index] = 0.25
    else:
        expected_impedance[index + (3 if row == "impedance_shape" else 0)] = 0.25
    assert reference == tuple(expected_reference)
    assert impedance == tuple(expected_impedance)


def _joint_properties():
    return JointAdvancedProperties(
        0,
        2,
        0.123456789123,
        0.234567891234,
        0.345678912345,
        0.456789123456,
        0.00123456789,
        REFERENCE,
        IMPEDANCE,
        REFERENCE,
        IMPEDANCE,
        "auto",
        (-1.123456789, 2.123456789),
        False,
    )


def test_idle_joint_draw_does_not_create_draft_edits_or_submit(inputs):
    properties = _joint_properties()
    submitted = []
    session = SimpleNamespace(
        joint_advanced_properties=lambda _joint: properties,
        structure_generation=0,
        adapter=SimpleNamespace(caps=SimpleNamespace(model_properties=True, simulation=True)),
        paused=True,
    )
    ctx = SimpleNamespace(
        session=session, tr=lambda text: text, theme=THEME, submit=submitted.append
    )
    panel = InspectorPanel()
    joint = SimpleNamespace(joint_id=0, type="hinge")
    for _ in range(3):
        inputs.buttons.add("Apply##joint-advanced")
        panel._joint_advanced_properties(ctx, joint)
        assert panel._joint_advanced_edit is properties
    assert submitted == []


@pytest.mark.parametrize("editable", (False, True))
def test_geometry_contact_edit_preserves_unedited_payload_values(inputs, editable):
    properties = GeometryProperties(
        7,
        (0.7123456789, 0.0123456789, 0.00123456789),
        1,
        1,
        3,
        0,
        0.00123456789,
        0.00234567891,
        0.5123456789,
        REFERENCE,
        IMPEDANCE,
        0.0123456789,
        (0.123456789,) * 6,
    )
    submitted = []
    session = SimpleNamespace(
        geometry_properties=lambda _node: properties,
        node=lambda _node: SimpleNamespace(source_editable=True),
        adapter=SimpleNamespace(caps=SimpleNamespace(model_properties=True, simulation=True)),
        paused=editable,
    )
    ctx = SimpleNamespace(
        session=session, tr=lambda text: text, theme=THEME, submit=submitted.append
    )
    panel = InspectorPanel()
    panel._geometry_contact_properties(ctx, 7)
    assert submitted == []
    inputs.edits["##contact_priority"] = 3
    panel._geometry_contact_properties(ctx, 7)
    if not editable:
        assert submitted == []
        assert all(call[3] for call in inputs.calls)
        return
    assert len(submitted) == 1
    command = submitted[0]
    for name in properties.__dataclass_fields__:
        assert getattr(command, name) == (
            3 if name == "contact_priority" else getattr(properties, name)
        )


@pytest.fixture
def joint_editor(tmp_path):
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.workspace import WorkspaceAdapter
    from mojive.session import Session

    path = tmp_path / "joint.xml"
    path.write_text("""<mujoco><compiler angle="radian" autolimits="true"/><worldbody><body>
      <joint name="hinge" type="hinge" range="-1 1" armature="0.123456789123"
        ref="0.345678912345" springref="0.456789123456"
        solreflimit="0.020123456789 1.23456789123"
        solimplimit="0.80123456789 0.95123456789 0.00123456789 0.5123456789 2.123456789"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body></worldbody></mujoco>""")
    adapter = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(adapter)
    assert session.submit(cmd.Pause()).ok
    return InspectorPanel(), PanelContext(session, None), session.joints[0], adapter.primary


@pytest.mark.physics
def test_joint_multi_field_draft_compiles_once_and_undo_restores_it(
    inputs, joint_editor, monkeypatch
):
    panel, ctx, joint, adapter = joint_editor
    original = ctx.session.joint_advanced_properties(joint.joint_id)
    compiled = []
    compile_model = adapter._compile_composed_model

    def counted_compile():
        compiled.append(True)
        return compile_model()

    monkeypatch.setattr(adapter, "_compile_composed_model", counted_compile)
    inputs.edits.update(
        {
            "##joint_armature": 0.5,
            "##joint_reference": 45.0,
            "##joint_limit_impedance_shape": (0.75, original.limit_solver_impedance[4]),
        }
    )
    panel._joint_advanced_properties(ctx, joint)
    assert ctx.session.joint_advanced_properties(joint.joint_id) == original
    assert compiled == []
    inputs.buttons.add("Apply##joint-advanced")
    panel._joint_advanced_properties(ctx, joint)
    assert len(compiled) == 1
    updated = ctx.session.joint_advanced_properties(joint.joint_id)
    assert updated.armature == 0.5
    assert updated.reference == pytest.approx(np.pi / 4)
    assert updated.limit_solver_impedance == (
        *original.limit_solver_impedance[:3],
        0.75,
        original.limit_solver_impedance[4],
    )
    assert updated.spring_reference == original.spring_reference
    assert ctx.session.submit(cmd.Undo()).ok
    restored = ctx.session.joint_advanced_properties(joint.joint_id)
    assert restored == original
    panel._joint_advanced_properties(ctx, joint)
    assert panel._joint_advanced_edit == original


@pytest.mark.physics
def test_joint_failed_apply_keeps_draft_and_revert_discards_it(inputs, joint_editor, monkeypatch):
    panel, ctx, joint, _adapter = joint_editor
    original = ctx.session.joint_advanced_properties(joint.joint_id)
    inputs.edits["##joint_armature"] = 0.75
    panel._joint_advanced_properties(ctx, joint)
    submitted = []

    def fail(command):
        submitted.append(command)
        return cmd.CommandResult.bad("The model compiler rejected this draft")

    monkeypatch.setattr(ctx, "submit", fail)
    inputs.buttons.add("Apply##joint-advanced")
    panel._joint_advanced_properties(ctx, joint)
    assert len(submitted) == 1
    assert submitted[0].armature == 0.75
    assert ctx.session.joint_advanced_properties(joint.joint_id) == original
    assert panel._joint_advanced_edit.armature == 0.75
    assert "The model compiler rejected this draft" in inputs.messages
    panel._joint_advanced_properties(ctx, joint)
    assert panel._joint_advanced_edit.armature == 0.75
    inputs.buttons.add("Revert##joint-advanced")
    panel._joint_advanced_properties(ctx, joint)
    assert panel._joint_advanced_edit == original
    assert panel._joint_advanced_error == ""
    assert len(submitted) == 1


@pytest.mark.physics
def test_running_joint_keeps_draft_but_disables_edits_and_apply(inputs, joint_editor):
    panel, ctx, joint, _adapter = joint_editor
    inputs.edits["##joint_armature"] = 0.75
    panel._joint_advanced_properties(ctx, joint)
    draft = panel._joint_advanced_edit
    assert ctx.session.submit(cmd.Play()).ok
    inputs.edits["##joint_armature"] = 1.0
    inputs.buttons.add("Apply##joint-advanced")
    inputs.calls.clear()
    panel._joint_advanced_properties(ctx, joint)
    assert panel._joint_advanced_edit == draft
    assert all(call[3] for call in inputs.calls)
    assert ctx.session.joint_advanced_properties(joint.joint_id).armature != 0.75


@pytest.mark.parametrize("index", (None, 0, 1, 2, 3, 4))
def test_geometry_fluid_rows_preserve_all_unedited_components(inputs, index):
    coefficients = (
        0.1234567890123,
        0.2345678901234,
        0.3456789012345,
        0.4567890123456,
        0.5678901234567,
    )
    original = GeometryAdvancedProperties(
        7, 0, "density", 1.0, 1000.123456789, "volume", True, coefficients
    )
    expected = list(coefficients)
    if index is not None:
        expected[index] = 0.25
        inputs.edits["##geometry_fluid_first" if index < 3 else "##geometry_fluid_last"] = (
            expected[:3] if index < 3 else expected[3:]
        )
    ctx = SimpleNamespace(tr=lambda text: text)
    edited = geometry._geometry_fluid_rows(ctx, original)
    assert edited.fluid_coefficients == tuple(expected)
    assert edited.density == original.density
    if index is None:
        assert edited is original


@pytest.fixture
def geometry_editor(tmp_path):
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.workspace import WorkspaceAdapter
    from mojive.session import Session

    path = tmp_path / "geometry.xml"
    path.write_text("""<mujoco><worldbody><body><joint name="hinge"/>
      <geom name="shape" type="box" size=".1 .1 .1" density="1000.123456789"
        fluidshape="ellipsoid" fluidcoef="0.1234567890123 0.2345678901234 0.3456789012345 0.4567890123456 0.5678901234567"/>
    </body></worldbody></mujoco>""")
    adapter = WorkspaceAdapter(MuJoCoAdapter(path))
    session = Session(adapter)
    assert session.submit(cmd.Pause()).ok
    node = next(node for node in session.nodes if node.name == "shape")
    try:
        yield InspectorPanel(), PanelContext(session, None), node.node_id, adapter.primary
    finally:
        session.release()


@pytest.mark.physics
@pytest.mark.parametrize("resource_type", ("mesh", "hfield"))
def test_shape_without_resources_can_revert_without_compiling(
    inputs, geometry_editor, resource_type
):
    panel, ctx, node_id, _adapter = geometry_editor
    original = ctx.session.geometry_shape_properties(node_id)
    inputs.edits["##geometry_type"] = 7 if resource_type == "mesh" else 1
    panel._geometry_shape_properties(ctx, node_id)
    assert panel._geometry_shape_edit.type == resource_type
    assert panel._geometry_shape_edit.resource_name == ""
    assert not inputs.button_states["Apply##changes-0"]
    assert inputs.button_states["Revert##changes-1"]
    assert ctx.session.geometry_shape_properties(node_id) == original
    assert not ctx.session.can_undo
    inputs.buttons.add("Revert##changes-1")
    panel._geometry_shape_properties(ctx, node_id)
    assert panel._geometry_shape_edit == original
    assert ctx.session.geometry_shape_properties(node_id) == original
    assert not ctx.session.can_undo


@pytest.mark.physics
@pytest.mark.parametrize("kind", ("shape", "advanced"))
def test_geometry_draft_applies_once_and_undo_restores_source(
    inputs, geometry_editor, monkeypatch, kind
):
    panel, ctx, node_id, adapter = geometry_editor
    draw = getattr(panel, f"_geometry_{kind}_properties")
    query = getattr(ctx.session, f"geometry_{kind}_properties")
    original = query(node_id)
    compiled = []
    compile_model = adapter._compile_composed_model

    def record_compile():
        compiled.append(True)
        return compile_model()

    monkeypatch.setattr(adapter, "_compile_composed_model", record_compile)
    if kind == "shape":
        inputs.edits["##geometry_type"] = 2
    else:
        inputs.edits.update(
            {
                "##geometry_visual_group": 2,
                "##geometry_fluid_first": (0.25, *original.fluid_coefficients[1:3]),
            }
        )
    draw(ctx, node_id)
    assert query(node_id) == original and compiled == []
    inputs.buttons.add("Apply##changes-0")
    draw(ctx, node_id)
    assert len(compiled) == 1
    updated = query(node_id)
    if kind == "shape":
        assert updated.type == "sphere"
    else:
        assert updated.visual_group == 2
        assert updated.density == original.density
        assert updated.fluid_coefficients == (0.25, *original.fluid_coefficients[1:])
    assert ctx.session.submit(cmd.Undo()).ok
    assert query(node_id) == original
    assert not ctx.session.can_undo


@pytest.mark.physics
@pytest.mark.parametrize("kind", ("shape", "advanced"))
def test_failed_geometry_apply_keeps_draft_and_running_revert_is_local(
    inputs, geometry_editor, monkeypatch, kind
):
    panel, ctx, node_id, adapter = geometry_editor
    draw = getattr(panel, f"_geometry_{kind}_properties")
    query = getattr(ctx.session, f"geometry_{kind}_properties")
    original = query(node_id)
    inputs.edits["##geometry_type" if kind == "shape" else "##geometry_visual_group"] = 2
    draw(ctx, node_id)
    draft = getattr(panel, f"_geometry_{kind}_edit")
    method = "set_geometry_shape" if kind == "shape" else "set_geometry_advanced_properties"
    with monkeypatch.context() as failing:
        failing.setattr(adapter, method, lambda *args: False)
        inputs.buttons.add("Apply##changes-0")
        draw(ctx, node_id)
    assert query(node_id) == original
    assert getattr(panel, f"_geometry_{kind}_edit") == draft
    assert getattr(panel, f"_geometry_{kind}_error")
    assert not ctx.session.can_undo
    assert ctx.session.submit(cmd.Play()).ok
    draw(ctx, node_id)
    assert getattr(panel, f"_geometry_{kind}_edit") == draft
    assert not inputs.button_states["Apply##changes-0"]
    assert inputs.button_states["Revert##changes-1"]
    inputs.buttons.add("Revert##changes-1")
    draw(ctx, node_id)
    assert getattr(panel, f"_geometry_{kind}_edit") == original
    assert not getattr(panel, f"_geometry_{kind}_error")
    assert query(node_id) == original
    assert not ctx.session.can_undo
