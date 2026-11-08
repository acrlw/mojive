"""Native Inspector component appearance and real input at the desktop window size."""

import numpy as np
import pytest
from examples.ui_design.native.appearance_review import _inspector_size
from imgui_bundle import imgui
from tests.gpu.test_ui_design_runtime import runtime as runtime

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.scene.queries import node_world_pose

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
def test_camera_and_joint_fields_fit_narrow_inspectors(runtime, language):
    study, session = runtime, runtime.preview.session
    study.preview.localizer.set_language(language, persist=False)
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    joint = next(node for node in session.nodes if node.name == "hinge_limited")
    study.command("float")
    study.frame(4)
    for width in (284, 344, 704):
        _inspector_size(study, width, 820)
        for node, keys in (
            (
                camera,
                tuple(
                    "##inspector-camera-" + name + "-value"
                    for name in ("vertical fov", "near", "far")
                ),
            ),
            (joint, ("##joint_damping", "##joint_stiffness")),
        ):
            _select(study, node)
            previous = None
            for key in keys:
                field = _visible_hit(study, key)
                assert field[3] == pytest.approx(28 * study.s, abs=1)
                if node.type is NodeType.CAMERA:
                    anchor = study.ui.hits[f"camera-inspector-projection-{node.node_id}"]
                    left, right = anchor[0], anchor[0] + anchor[2]
                else:
                    lower, upper = (study.ui.hits[f"##joint_range-{i}"] for i in range(2))
                    left, right = lower[0], upper[0] + upper[2]
                assert field[0] == pytest.approx(left, abs=1)
                assert field[0] + field[2] == pytest.approx(right, abs=1)
                if previous is not None:
                    # Revealing the next row may scroll the common content window.
                    sibling = study.ui.hits[previous]
                    assert field[1] >= sibling[1] + sibling[3] + 4 * study.s - 1
                    assert field[0] == pytest.approx(sibling[0], abs=1)
                    assert field[2] == pytest.approx(sibling[2], abs=1)
                previous = key
            study.capture(f"inspector-{node.type.value}-{width}-{language}")


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_body_summary_retains_registered_joint_controls_without_advanced_fields(runtime, language):
    study, session = runtime, runtime.preview.session
    study.preview.localizer.set_language(language, persist=False)
    body = next(node for node in session.nodes if node.name == "hinge_body")
    _select(study, body)
    study.inspector_tab = "Physics"
    study.frame(4)
    assert "##body-derived-mass" in study.ui.hits
    assert "##body-inertia-mode" not in study.ui.hits
    assert "##body-gravity-compensation" not in study.ui.hits
    study.inspector_tab = "Joint"
    study.frame(4)
    joint = session.joints_for_body(body.body_index)[0]
    key = f"##pose-joint-{joint.joint_id}"
    value, rail = study.ui.hits[key + "-value"], study.ui.hits[key + "-track"]
    assert value[1] == pytest.approx(rail[1])
    assert rail[0] + rail[2] <= value[0]
    assert "##joint_armature" not in study.ui.hits
    study.capture("inspector-shared-joint-" + language)


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
