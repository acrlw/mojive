"""Native Inspector component appearance and real input at the desktop window size."""

import numpy as np
import pytest
from examples.ui_design.native.appearance_review import _inspector_size
from imgui_bundle import imgui
from tests.gpu.test_ui_design_runtime import runtime as runtime

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.scene.queries import node_world_pose
from mojive.ui.imgui_draw import ImguiDraw2D

pytestmark = pytest.mark.gpu


def _center(rect):
    x, y, width, height = rect
    return x + width / 2, y + height / 2


def _child():
    return next(
        window
        for window in imgui.get_current_context().windows
        if "Inspector content" in str(window.name) and window.active
    )


def _visible_hit(study, key):
    for _ in range(20):
        assert key in study.ui.hits, (key, study.ui.hits)
        rect = study.ui.hits[key]
        child = _child()
        clip = child.inner_clip_rect
        x, y, width, height = rect
        if clip.min.y <= y and y + height <= clip.max.y:
            assert clip.min.x <= x < x + width <= clip.max.x + 1
            return rect
        direction = -1 if y + height > clip.max.y else 1
        study.event("add_mouse_pos_event", clip.max.x - 12 * study.s, (clip.min.y + clip.max.y) / 2)
        study.event("add_mouse_wheel_event", 0, direction)
        study.frame(3)
    raise AssertionError(f"Inspector field {key} could not be scrolled into view")


def _select(study, node):
    assert study.preview.session.submit(cmd.SelectNode(node.node_id)).ok
    study.activate("Inspector")
    study.inspector_tab = "Properties"
    study.frame(4)


def _drag(study, key, direction=1):
    point = _center(_visible_hit(study, key))
    study.event("add_mouse_pos_event", *point)
    study.event("add_mouse_button_event", 0, True)
    for offset in (8, 16, 24, 32):
        study.event("add_mouse_pos_event", point[0] + offset * study.s * direction, point[1])
    return point


def _release(study):
    study.event("add_mouse_button_event", 0, False)
    study.frame(3)


def test_light_and_body_share_native_headers_vectors_and_badges(runtime):
    study, session = runtime, runtime.preview.session
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    _select(study, light)
    _visible_hit(study, "section-LIGHT PROPERTIES")
    study.capture("inspector-native-light-properties")
    header_height = study.ui.hits["section-LIGHT PROPERTIES"][3]
    key = f"##light_position_0_{light.node_id}"
    _visible_hit(study, key)
    light_field = study.ui.hits[key + "-frame"]
    light_badge = study.ui.hits[key + "-reset"]
    light_fields = [
        study.ui.hits[f"##light_position_{index}_{light.node_id}-frame"] for index in range(3)
    ]
    assert light_field[3] == pytest.approx(28 * study.s, abs=1)
    assert 0 < light_badge[2] <= 18 * study.s
    study.capture("inspector-native-light-transform")

    body = next(node for node in session.nodes if node.type is NodeType.LINK and node.posable)
    _select(study, body)
    assert study.ui.hits["section-TRANSFORM"][3] == pytest.approx(header_height, abs=1)
    body_key = f"##{body.node_id}positionX"
    _visible_hit(study, body_key)
    body_field = study.ui.hits[body_key + "-frame"]
    body_badge = study.ui.hits[body_key + "-reset"]
    assert body_field[3] == pytest.approx(light_field[3], abs=1)
    assert body_badge[2] == pytest.approx(light_badge[2], abs=1)
    for index, axis in enumerate("XYZ"):
        rect = study.ui.hits[f"##{body.node_id}position{axis}-frame"]
        light_rect = light_fields[index]
        assert rect[2] == pytest.approx(body_field[2], abs=1)
        assert rect[0] - body_field[0] == pytest.approx(light_rect[0] - light_field[0], abs=1)
        assert rect[1] - body_field[1] == pytest.approx(light_rect[1] - light_field[1], abs=1)
    study.capture("inspector-native-body-transform")


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_camera_and_physics_numeric_rows_keep_clear_vertical_gaps(runtime, monkeypatch, language):
    study, session = runtime, runtime.preview.session
    components = {}
    for method in ("slider_float", "drag_float"):
        original = getattr(imgui, method)

        def observe(identifier, *args, _original=original, **kwargs):
            result = _original(identifier, *args, **kwargs)
            a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            components[identifier] = (a.x, a.y, b.x - a.x, b.y - a.y)
            return result

        monkeypatch.setattr(imgui, method, observe)
    study.preview.localizer.set_language(language, persist=False)
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    geometry = next(
        node
        for node in session.nodes
        if node.type is NodeType.GEOM and session.geometry_properties(node.node_id) is not None
    )
    for node, tab, keys, visible in (
        (
            camera,
            "Properties",
            tuple("##inspector-camera-" + name for name in ("vertical fov", "near", "far")),
            "##inspector-camera-far",
        ),
        (
            geometry,
            "Physics",
            (
                "##contact_margin",
                "##contact_gap",
                "##contact_solver_mix",
                "##contact_solver_reference-0",
                "##contact_impedance_first-0",
                "##contact_impedance_shape-0",
            ),
            "##contact_impedance_shape-0",
        ),
    ):
        _select(study, node)
        study.inspector_tab = tab
        study.frame(4)
        previous_bottom = None
        for key in keys:
            x, y, w, h = study.ui.hits[key]
            if node.type is NodeType.CAMERA:
                tx, ty, tw, th = components[key]
                fx, fy, fw, fh = components[key + "-value"]
                for a, b, width, height in ((tx, ty, tw, th), (fx, fy, fw, fh)):
                    assert height == pytest.approx(28 * study.s, abs=1), key
                    assert x <= a < a + width <= x + w + 1, key
                    assert y <= b < b + height <= y + h + 1, key
                assert min(tx + tw, fx + fw) <= max(tx, fx) or min(ty + th, fy + fh) <= max(
                    ty, fy
                ), key
                if fy > ty + 1:
                    assert fy - (ty + th) >= 8 * study.s - 1, key
                    assert h == pytest.approx(fy + fh - ty, abs=1), key
                else:
                    assert h == pytest.approx(fh, abs=1), key
            else:
                assert h == pytest.approx(28 * study.s, abs=1), key
            if previous_bottom is not None:
                assert y - previous_bottom >= 8 * study.s - 1, key
            previous_bottom = y + h
        _visible_hit(study, visible)
        study.capture(f"inspector-{node.type.value}-{language}-numeric-row-spacing")


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_physics_and_joint_fields_reflow_as_uniform_groups(runtime, monkeypatch, language):
    study, session = runtime, runtime.preview.session
    study.preview.localizer.set_language(language, persist=False)
    body = next(node for node in session.nodes if node.name == "hinge_body")
    joint = next(node for node in session.nodes if node.name == "hinge_limited")
    study.command("float")
    study.frame(4)
    painted = []
    text = ImguiDraw2D.text

    def observe(draw, position, color, value, **kwargs):
        painted.append(value)
        return text(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)
    for width in (284, 344, 704):
        _inspector_size(study, width, 820)
        _select(study, body)
        study.inspector_tab = "Physics"
        painted.clear()
        study.frame(4)
        if width >= 344:
            for tab in ("Properties", "Physics", "Material", "Joint"):
                assert study.ui.translate(tab) in painted, ("Inspector tab was truncated", tab)
        layouts = []
        for name in ("body_inertial_position", "body_inertial_rotation", "body_diagonal_inertia"):
            fields = [study.ui.hits[f"##{name}_{index}_{body.node_id}-frame"] for index in range(3)]
            label = study.ui.hits["vector-label-" + name]
            layouts.append([(x - fields[0][0], y - fields[0][1], w, h) for x, y, w, h in fields])
            above = fields[0][1] >= label[1] + label[3]
            assert above == (fields[0][0] <= label[0] + 1)
            assert len({round(field[2], 3) for field in fields}) == 1
        np.testing.assert_allclose(layouts[0], layouts[1], atol=1)
        np.testing.assert_allclose(layouts[0], layouts[2], atol=1)
        properties = session.body_properties(body.node_id)
        for value in properties.diagonal_inertia:
            assert f"{value:.6g}" in painted, ("Common inertia value was truncated", value, painted)
        assert session.selected_node == body
        study.capture(f"inspector-inertial-{width}-{language}")

        _select(study, joint)
        study.inspector_tab = "Joint"
        study.frame(4)
        if "##joint_armature" not in study.ui.hits:
            section = "section-" + study.ui.translate("advanced joint properties").upper()
            _visible_hit(study, section)
            study.click_hit(section)
            study.frame(3)
        starts = []
        for label, key in (
            ("damping", "##joint_damping"),
            ("stiffness", "##joint_stiffness"),
            ("armature", "##joint_armature"),
            ("friction loss", "##joint_friction_loss"),
            ("reference", "##joint_reference"),
            ("spring reference", "##joint_spring_reference"),
        ):
            field = study.ui.hits[key]
            label_rect = study.ui.hits["property-label-" + label]
            starts.append((field[0], field[2]))
            if field[0] > label_rect[0] + study.s:
                assert field[1] + field[3] / 2 == pytest.approx(
                    label_rect[1] + label_rect[3] / 2, abs=study.s
                )
            else:
                assert field[1] >= label_rect[1] + label_rect[3] + 4 * study.s
        np.testing.assert_allclose(starts, [starts[0]] * len(starts), atol=1)
        ranges = [study.ui.hits[f"##joint_range-{index}"] for index in range(2)]
        assert ranges[0][2] == pytest.approx(ranges[1][2], abs=1)
        label = study.ui.hits["property-label-range"]
        assert ranges[0][0] == pytest.approx(starts[0][0], abs=1)
        if ranges[0][0] <= label[0] + study.s:
            assert ranges[0][0] == pytest.approx(label[0], abs=1)
            assert ranges[0][1] >= label[1] + label[3] + 4 * study.s
        else:
            assert ranges[0][0] >= label[0] + label[2] + 12 * study.s - 1
            assert ranges[0][1] + ranges[0][3] / 2 == pytest.approx(
                label[1] + label[3] / 2, abs=study.s
            )
        _visible_hit(study, "##joint_armature")
        study.capture(f"inspector-joint-{width}-{language}")


def test_native_light_fields_preserve_live_edit_undo_escape_and_switch(runtime):
    study, session = runtime, runtime.preview.session
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    _select(study, light)
    initial_history = session.can_undo

    for kind in ("color", "position"):
        current = session.source.lights.lights[light.light_index]
        before = (current.diffuse if kind == "color" else current.position).copy()
        key = "##light_colorR" if kind == "color" else f"##light_position_0_{light.node_id}"
        _drag(study, key, -1 if kind == "color" else 1)
        current = session.source.lights.lights[light.light_index]
        assert not np.allclose(current.diffuse if kind == "color" else current.position, before)
        assert session.editing
        _release(study)
        assert not session.editing
        assert session.submit(cmd.Undo()).ok
        study.frame(3)
        current = session.source.lights.lights[light.light_index]
        np.testing.assert_allclose(current.diffuse if kind == "color" else current.position, before)
        assert session.can_undo is initial_history

    before = session.source.lights.lights[light.light_index].diffuse.copy()
    point = _drag(study, "##light_intensity")
    assert session.editing
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    study.event("add_mouse_pos_event", point[0] + 48 * study.s, point[1])
    _release(study)
    np.testing.assert_allclose(session.source.lights.lights[light.light_index].diffuse, before)
    assert not session.editing and session.can_undo is initial_history
    assert session.selected_node.node_id == light.node_id

    original_active = session.source.lights.lights[light.light_index].active
    _visible_hit(study, "##light_enabled")
    study.click_hit("##light_enabled")
    study.frame(3)
    assert session.source.lights.lights[light.light_index].active != original_active
    assert session.submit(cmd.Undo()).ok
    study.frame(3)
    assert session.source.lights.lights[light.light_index].active is original_active
    assert session.can_undo is initial_history
    study.capture("inspector-native-light-interactions")


def test_long_transform_text_preserves_value_and_single_undo(runtime):
    study, session = runtime, runtime.preview.session
    study.menus.dispatch("create:Box")
    study.frame(4)
    node = session.selected_node
    before = node_world_pose(session, node)[0].copy()
    key = f"##{node.node_id}positionX"
    point = _center(_visible_hit(study, key))
    modifier = (
        imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl
    )
    study.event("add_mouse_pos_event", *point)
    study.event("add_key_event", modifier, True)
    study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_button_event", 0, False)
    study.event("add_key_event", modifier, False)
    assert imgui.get_io().want_text_input
    study.event("add_input_characters_utf8", "123456.7890625")
    study.capture("inspector-long-transform-text-edit")
    study.event("add_key_event", imgui.Key.enter, True)
    study.event("add_key_event", imgui.Key.enter, False)
    study.frame(3)
    assert not session.editing
    assert node_world_pose(session, node)[0][0] == pytest.approx(123456.7890625)
    study.capture("inspector-long-transform-idle")
    assert session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(node_world_pose(session, node)[0], before)


def test_source_color_popup_keeps_one_undo_and_escape_restores_all_changes(runtime):
    study, session = runtime, runtime.preview.session
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    _select(study, light)
    before = session.source.lights.lights[light.light_index].diffuse.copy()
    initial_history = session.can_undo
    popup = f"##light_color-color-editor-{light.node_id}"
    for close in ("done", "escape"):
        _visible_hit(study, "##light_color-picker")
        study.click_hit("##light_color-picker")
        study.frame(3)
        assert study.inspector_presentation.color_edit_open
        for color in ("#336699", "#99AA66"):
            imgui.set_clipboard_text(color)
            study.click_hit(popup + "-paste")
            study.frame(3)
            assert session.editing
            assert session.can_undo is initial_history
        assert not np.allclose(session.source.lights.lights[light.light_index].diffuse, before)
        study.capture("inspector-source-color-popup-" + close)
        if close == "done":
            study.click_hit(popup + "-done")
            study.frame(3)
            assert session.submit(cmd.Undo()).ok
            study.frame(3)
        else:
            study.event("add_key_event", imgui.Key.escape, True)
            study.event("add_key_event", imgui.Key.escape, False)
            study.frame(3)
        assert not session.editing and not study.inspector_presentation.color_edit_open
        np.testing.assert_array_equal(
            session.source.lights.lights[light.light_index].diffuse, before
        )
        assert session.can_undo is initial_history
        assert session.selected_node.node_id == light.node_id
