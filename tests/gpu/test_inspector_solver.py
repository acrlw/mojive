"""Edit the production Inspector solver rows in a compact native window."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.app.ui.window import create_window
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.tools.ui_runtime import _click, _item_rect
from mojive.ui.localization import Localizer, parse_language
from mojive.ui.panels import PanelContext
from mojive.ui.panels.inspector import InspectorPanel, fields
from mojive.ui.window import WindowConfig

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.mark.parametrize("kind", ("joint", "contact"))
@pytest.mark.parametrize("language,scale", (("en", 1.0), ("zh_CN", 2.25)))
def test_inspector_solver_edit_apply_and_undo(backend_name, monkeypatch, kind, language, scale):
    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause()).ok
    panel = InspectorPanel()
    joint = next(item for item in session.joints if item.name == "hinge_limited")
    node = next(
        item for item in session.nodes if item.type is NodeType.GEOM and item.name == "hinge_arm"
    )
    original = (
        session.joint_advanced_properties(joint.joint_id)
        if kind == "joint"
        else session.geometry_properties(node.node_id)
    )
    localizer = Localizer(parse_language(language))
    submitted, compilations, labels = [], [], []
    submit, compile_model, property_row = (
        session.submit,
        adapter._compile_composed_model,
        fields.property_row,
    )

    def record_submit(command):
        submitted.append(command)
        return submit(command)

    def record_compile():
        compilations.append(True)
        return compile_model()

    def record_row(label, **kwargs):
        labels.append(label)
        return property_row(label, **kwargs)

    monkeypatch.setattr(session, "submit", record_submit)
    monkeypatch.setattr(adapter, "_compile_composed_model", record_compile)
    monkeypatch.setattr(fields, "property_row", record_row)
    output = Path("output/inspector/solver") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    config = WindowConfig(
        width=round(780 * scale),
        height=round(720 * scale),
        ui_scale=scale,
        vsync=False,
        docking=False,
        ini_path="",
        show_on_start=False,
    )
    try:
        with create_window(config, backend_name) as window:
            ctx = PanelContext(
                session,
                None,
                translate=localizer.text,
                style_scale=window.style_scale,
                painter=window.painter,
            )

            def frame(*, capture=False):
                window.begin_frame()
                labels.clear()
                imgui.set_next_window_pos((0, 0))
                imgui.set_next_window_size(imgui.get_io().display_size)
                imgui.begin("Inspector solver", flags=imgui.WindowFlags_.no_decoration)
                imgui.text(
                    localizer.text(
                        "advanced joint properties" if kind == "joint" else "contact properties"
                    )
                )
                imgui.separator()
                imgui.set_next_item_open(True, imgui.Cond_.always)
                if kind == "joint":
                    panel._joint_advanced_properties(ctx, joint)
                else:
                    panel._geometry_contact_properties(ctx, node.node_id)
                panel.finish_frame(ctx)
                imgui.end()
                pixels = window.end_frame(readback=capture)
                return None if pixels is None else pixels[::-1].copy()

            ui = SimpleNamespace(sync=frame)

            def capture(stage):
                imgui.get_io().add_mouse_pos_event(-100, -100)
                Image.fromarray(frame(capture=True)).save(
                    output / f"{kind}-{language}-{scale:g}-{stage}.png"
                )

            def control(function, label):
                lo, hi = _item_rect(ui, function, label)
                size = imgui.get_io().display_size
                assert 0 <= lo[0] < hi[0] <= size.x
                assert 0 <= lo[1] < hi[1] <= size.y
                return lo, hi

            for _ in range(4):
                frame()
            assert submitted == []
            prefixes = ("joint_limit", "joint_friction") if kind == "joint" else ("contact",)
            for prefix in prefixes:
                for function, suffix in (
                    ("drag_float2", "solver_reference"),
                    ("drag_float3", "impedance_first"),
                    ("drag_float2", "impedance_shape"),
                ):
                    control(function, f"##{prefix}_{suffix}")
                label_prefix = prefix.removeprefix("joint_") + " " if kind == "joint" else ""
                for label in (
                    "solver reference",
                    "impedance min / max / width",
                    "impedance midpoint / power",
                ):
                    assert localizer.text(label_prefix + label) in labels
            assert submitted == []
            capture("idle")

            # Drag only the first reference component through the real ImGui input path.
            lo, hi = control("drag_float2", f"##{prefixes[0]}_solver_reference")
            x, y = lo[0] + (hi[0] - lo[0]) * 0.25, (lo[1] + hi[1]) * 0.5
            io = imgui.get_io()
            io.add_mouse_pos_event(x, y)
            frame()
            io.add_mouse_button_event(0, True)
            frame()
            io.add_mouse_pos_event(x + 48 * scale, y)
            frame()
            io.add_mouse_button_event(0, False)
            frame()
            frame()
            capture("edited")
            if kind == "joint":
                assert submitted == []
                assert session.joint_advanced_properties(joint.joint_id) == original
                lo, hi = control("button", f"{localizer.text('Apply')}##joint-advanced")
                _click(ui, ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5))
                assert len(compilations) == 1
                changed = session.joint_advanced_properties(joint.joint_id)
                assert changed.limit_solver_reference[0] > original.limit_solver_reference[0]
                assert changed.limit_solver_reference[1:] == original.limit_solver_reference[1:]
                assert changed.limit_solver_impedance == original.limit_solver_impedance
                assert changed.friction_solver_reference == original.friction_solver_reference
                assert (
                    len([c for c in submitted if isinstance(c, cmd.SetJointAdvancedProperties)])
                    == 1
                )
            else:
                changed = session.geometry_properties(node.node_id)
                assert changed.solver_reference[0] > original.solver_reference[0]
                assert changed.solver_reference[1:] == original.solver_reference[1:]
                assert changed.solver_impedance == original.solver_impedance
                assert changed.friction == original.friction
                assert compilations == []
                assert len([c for c in submitted if isinstance(c, cmd.SetGeometryProperties)]) == 1
            capture("applied")
            assert session.submit(cmd.Undo()).ok
            restored = (
                session.joint_advanced_properties(joint.joint_id)
                if kind == "joint"
                else session.geometry_properties(node.node_id)
            )
            assert restored == original
            capture("undo")
    finally:
        session.release()


@pytest.mark.parametrize("language,scale", (("en", 1.0), ("zh_CN", 2.25)))
def test_geometry_missing_resource_revert_and_fluid_component_precision(
    tmp_path, backend_name, monkeypatch, language, scale
):
    path = tmp_path / "geometry.xml"
    path.write_text("""<mujoco><worldbody><body><joint/>
      <geom name="shape" type="box" size=".1 .1 .1" density="1000.123456789"
        fluidshape="ellipsoid" fluidcoef="0.1234567890123 0.2345678901234 0.3456789012345 0.4567890123456 0.5678901234567"/>
    </body></worldbody></mujoco>""")
    adapter = MuJoCoAdapter(path)
    session = Session(adapter)
    assert session.submit(cmd.Pause()).ok
    panel = InspectorPanel()
    node_id = next(node.node_id for node in session.nodes if node.name == "shape")
    original_shape = session.geometry_shape_properties(node_id)
    original_advanced = session.geometry_advanced_properties(node_id)
    submitted, compiled = [], []
    submit, compile_model = session.submit, adapter._compile_composed_model

    def record_submit(command):
        submitted.append(command)
        return submit(command)

    def record_compile():
        compiled.append(True)
        return compile_model()

    monkeypatch.setattr(session, "submit", record_submit)
    monkeypatch.setattr(adapter, "_compile_composed_model", record_compile)
    localizer = Localizer(parse_language(language))
    output = Path("output/inspector/geometry") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    config = WindowConfig(
        width=round(780 * scale),
        height=round(540 * scale),
        ui_scale=scale,
        vsync=False,
        docking=False,
        ini_path="",
        show_on_start=False,
    )
    try:
        with create_window(config, backend_name) as window:
            ctx = PanelContext(
                session,
                None,
                translate=localizer.text,
                style_scale=window.style_scale,
                painter=window.painter,
            )
            section = "shape"

            def frame(*, capture=False):
                window.begin_frame()
                imgui.set_next_window_pos((0, 0))
                imgui.set_next_window_size(imgui.get_io().display_size)
                imgui.begin("Inspector geometry", flags=imgui.WindowFlags_.no_decoration)
                imgui.set_next_item_open(True, imgui.Cond_.always)
                if section == "shape":
                    panel._geometry_shape_properties(ctx, node_id)
                else:
                    panel._geometry_advanced_properties(ctx, node_id)
                panel.finish_frame(ctx)
                imgui.end()
                pixels = window.end_frame(readback=capture)
                return None if pixels is None else pixels[::-1].copy()

            ui = SimpleNamespace(sync=frame)

            def click(function, label):
                lo, hi = _item_rect(ui, function, label)
                size = imgui.get_io().display_size
                assert 0 <= lo[0] < hi[0] <= size.x and 0 <= lo[1] < hi[1] <= size.y
                _click(ui, ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5))

            def key(value):
                imgui.get_io().add_key_event(value, True)
                frame()
                imgui.get_io().add_key_event(value, False)
                frame()

            def capture(stage):
                imgui.get_io().add_mouse_pos_event(-100, -100)
                Image.fromarray(frame(capture=True)).save(
                    output / f"{language}-{scale:g}-{stage}.png"
                )

            for _ in range(4):
                frame()
            assert submitted == []
            click("combo", "##geometry_type")
            key(imgui.Key.down_arrow)
            key(imgui.Key.enter)
            assert panel._geometry_shape_edit.type == "mesh"
            assert panel._geometry_shape_edit.resource_name == ""
            capture("missing-resource")
            click("button", f"{localizer.text('Apply')}##changes-0")
            assert submitted == [] and compiled == []
            click("button", f"{localizer.text('Revert')}##changes-1")
            assert panel._geometry_shape_edit == original_shape
            assert submitted == [] and compiled == []
            capture("reverted")

            section = "advanced"
            for _ in range(3):
                frame()
            assert submitted == []
            lo, hi = _item_rect(ui, "drag_float3", "##geometry_fluid_first")
            x, y = lo[0] + (hi[0] - lo[0]) / 6, (lo[1] + hi[1]) * 0.5
            io = imgui.get_io()
            io.add_mouse_pos_event(x, y)
            frame()
            io.add_mouse_button_event(0, True)
            frame()
            io.add_mouse_pos_event(x + 48 * scale, y)
            frame()
            io.add_mouse_button_event(0, False)
            frame()
            assert session.geometry_advanced_properties(node_id) == original_advanced
            assert submitted == [] and compiled == []
            assert (
                panel._geometry_advanced_edit.fluid_coefficients[0]
                != original_advanced.fluid_coefficients[0]
            )
            assert (
                panel._geometry_advanced_edit.fluid_coefficients[1:]
                == original_advanced.fluid_coefficients[1:]
            )
            click("button", f"{localizer.text('Apply')}##changes-0")
            assert len(compiled) == 1
            assert (
                len(
                    [
                        command
                        for command in submitted
                        if isinstance(command, cmd.SetGeometryAdvancedProperties)
                    ]
                )
                == 1
            )
            updated = session.geometry_advanced_properties(node_id)
            assert updated.fluid_coefficients[0] != original_advanced.fluid_coefficients[0]
            assert updated.fluid_coefficients[1:] == original_advanced.fluid_coefficients[1:]
            assert updated.density == original_advanced.density
            capture("fluid-applied")
            assert session.submit(cmd.Undo()).ok
            assert session.geometry_advanced_properties(node_id) == original_advanced
            assert not session.can_undo
            capture("fluid-undo")
    finally:
        session.release()
