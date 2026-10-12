"""Native panel content, scaled layout and scene operations in a real viewer."""

from itertools import pairwise

import glfw
import numpy as np
import pytest
from examples.ui_design.native.docking_study import LEFT_PANELS
from examples.ui_design.native.settings_panel import CATEGORIES
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.render.selection import render_backend_name
from mojive.scene.queries import node_world_pose
from mojive.ui.imgui_draw import ImguiDraw2D
from tests.gpu.test_ui_design_gizmos import OUTPUT
from tests.gpu.test_ui_design_gizmos import reference as reference

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session")
def backend_name():
    return render_backend_name()


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_sensors_panel_uses_live_sensor_state_without_cursor_assert(reference):
    study = reference
    glfw.set_window_size(study.window._window, 1440, 900)
    study.activate("Sensors")
    study.frame(4)
    window = imgui.internal.find_window_by_name("Sensors")
    assert window.active and not window.hidden
    assert "header-Sensors-more" in study.ui.hits


def _center(rect):
    x, y, width, height = rect
    return x + width / 2, y + height / 2


def _click(study, key):
    study.click(_center(study.ui.hits[key]))
    study.frame(3)


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_every_panel_and_inspector_tab_draws_at_the_interactive_window_size(reference, language):
    study = reference
    glfw.set_window_size(study.window._window, 1440, 900)
    study.preview.localizer.set_language(language, persist=False)
    for name in LEFT_PANELS:
        study.activate(name)
        study.frame(4)
        window = imgui.internal.find_window_by_name(name)
        assert window.active and not window.hidden, name
        assert window.flags & imgui.WindowFlags_.no_title_bar, name
        assert window.scroll_max.x < 1, name
        assert "header-" + name + "-more" in study.ui.hits, name
    study.activate("Inspector")
    for tab in ("Properties", "Physics", "Material", "Joint"):
        _click(study, "tab-" + tab)
        window = imgui.internal.find_window_by_name("Inspector")
        assert window.active and not window.hidden
        assert window.flags & imgui.WindowFlags_.no_title_bar
        assert "inspector-close" in study.ui.hits
    study.preview.session.submit(cmd.SelectNode(-1))
    study.frame(3)
    assert "inspector-header" in study.ui.hits
    study.activate("Settings")
    for category in CATEGORIES:
        study.settings._category = category
        study.frame(3)
        window = imgui.internal.find_window_by_name("Settings")
        assert window.dock_id == 0
        assert window.flags & imgui.WindowFlags_.no_title_bar
        assert window.scroll_max.x < 1 and window.scroll_max.y < 1
        assert "settings-close" in study.ui.hits
    _click(study, "settings-close")
    assert not study.open["Settings"]
    logo = study.rects["application-logo"]
    assert logo[1] + logo[3] / 2 == pytest.approx(study.menus.menu_center_y, abs=0.5)
    reserved = study.ui.hits["toggle-inspector"][0]
    assert all(x + width < reserved for x, _y, width, _h in study.menus.menu_rects.values())


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_scene_and_pose_double_click_focus_the_real_camera(reference):
    study, app = reference, reference.preview
    glfw.set_window_size(study.window._window, 1440, 900)
    study.search_visible = True
    study.query = "free_body"
    study.frame(4)
    node = next(node for node in app.session.nodes if node.name == "free_body")
    before = app._camera_view().target.copy()
    point = _center(study.ui.hits["entity-" + str(node.node_id)])
    study.click(point)
    study.click(point)
    study.frame(12)
    assert app.session.selected_node.node_id == node.node_id
    assert not np.allclose(app._camera_view().target, before)
    study.activate("Pose")
    study.frame(2)
    study.navigation.joints._search = "hinge_limited"
    study.frame(4)
    joint = next(joint for joint in app.session.joints if joint.name == "hinge_limited")
    before = app._camera_view().target.copy()
    point = _center(study.ui.hits["pose-joint-" + str(joint.joint_id)])
    study.click(point)
    study.click(point)
    study.frame(12)
    assert app.session.selected_node.type is NodeType.JOINT
    assert app.session.selected_node.joint_index == joint.joint_id
    assert app.tool == "rotate"
    assert not np.allclose(app._camera_view().target, before)


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_control_double_click_focuses_actuator_target_and_projection_has_two_choices(reference):
    study, app = reference, reference.preview
    study.activate("Control")
    study.frame(4)
    actuator = next(item for item in app.session.actuators if item.joint >= 0)
    before = app._camera_view().target.copy()
    point = _center(study.ui.hits[f"control-target-{actuator.ctrl_address}"])
    study.click(point)
    study.click(point)
    study.frame(12)
    assert app.session.selected_node.joint_index == actuator.joint
    assert not np.allclose(app._camera_view().target, before)
    study.activate("Cameras")
    study.frame(4)
    assert "projection-0" in study.ui.hits and "projection-1" in study.ui.hits
    projection = study.ui.hits["projection"]
    assert study.ui.hits["camera-frame"][1] >= projection[1] + projection[3] + 12 * study.s
    _click(study, "projection-1")
    assert app.camera.orthographic
    _click(study, "projection-0")
    assert not app.camera.orthographic


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_control_heading_draws_distinct_hover_press_and_selection_feedback(reference):
    study, session = reference, reference.preview.session
    study.activate("Control")
    study.event("add_mouse_pos_event", -100, -100)
    baseline = study.frame(3)
    actuator = session.actuators[0]
    x, y, width, height = study.ui.hits[f"control-target-{actuator.ctrl_address}"]
    probe_x, probe_y = round(x + width - 6 * study.s), round(y + height / 2)
    point = (x + width / 2, y + height / 2)
    hovered = study.event("add_mouse_pos_event", *point)
    pressed = study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_button_event", 0, False)
    study.event("add_mouse_pos_event", -100, -100)
    selected = study.frame(3)
    samples = [
        pixels[probe_y, probe_x].astype(int) for pixels in (baseline, hovered, pressed, selected)
    ]
    assert all(np.max(np.abs(a - b)) >= 4 for a, b in pairwise(samples))
    panel = study.preview.panels.get("control")
    assert panel._selected_address == actuator.ctrl_address
    assert panel._selection_revision == session.selection_revision


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_layer_labels_align_left_and_switches_share_their_row_center(
    reference, monkeypatch, language
):
    study = reference
    study.preview.localizer.set_language(language, persist=False)
    painted = {}
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **kwargs):
        painted[value] = position
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    study.activate("Layers")
    study.frame(3)
    for name, caption in (
        ("viewport_ui", "Viewport UI"),
        ("gizmos", "Gizmos"),
        ("selection", "Selection"),
    ):
        label = study.ui.hits[f"layer-{name}-label"]
        switch = study.ui.hits[f"layer-{name}"]
        shown = study.ui.translate(caption)
        assert painted[shown][0] == pytest.approx(label[0] + 10 * study.s, abs=1)
        assert label[1] + label[3] / 2 == pytest.approx(switch[1] + switch[3] / 2, abs=1)
        assert switch[2:] == pytest.approx((36 * study.s, 28 * study.s), abs=1)


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_axis_badge_reset_is_a_single_undoable_scene_edit(reference):
    study, app = reference, reference.preview
    app.selected = "free_body"
    study.frame(3)
    node = app.session.selected_node
    original = node_world_pose(app.session, node)[0].copy()
    assert abs(original[0]) > 0.1
    _click(study, f"##{node.node_id}positionX-reset")
    edited = node_world_pose(app.session, node)[0]
    assert edited[0] == pytest.approx(0, abs=1e-6)
    np.testing.assert_allclose(edited[1:], original[1:])
    assert not app.session.editing
    assert app.session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(node_world_pose(app.session, node)[0], original, atol=1e-6)


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_inspector_draws_component_fields_for_every_scene_entity_type(reference):
    study, app = reference, reference.preview
    nodes = {node.type: node for node in app.session.nodes}
    for kind, node in nodes.items():
        assert app.session.submit(cmd.SelectNode(node.node_id)).ok
        study.inspector_tab = "Properties"
        pixels = study.frame(3)
        assert app.entity["node_id"] == node.node_id
        for tab in ("Physics", "Material", "Joint"):
            _click(study, "tab-" + tab)
        if study.s == 2.0:
            directory = OUTPUT / study.backend_name / "panels"
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(directory / ("inspector-" + kind.value + ".png"))


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_inspector_entity_header_can_undock_and_move_the_window(reference):
    study = reference
    window = imgui.internal.find_window_by_name("Inspector")
    assert window.dock_id != 0
    start = _center(study.ui.hits["inspector-header"])
    viewport = study.preview._viewport_rect
    end = (viewport[0] + viewport[2] * 0.6, viewport[1] + viewport[3] * 0.2)
    study.event("add_mouse_pos_event", *start)
    study.event("add_mouse_button_event", 0, True)
    for fraction in (0.15, 0.5, 1):
        study.event(
            "add_mouse_pos_event",
            start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction,
        )
    study.event("add_mouse_button_event", 0, False)
    study.frame(3)
    window = imgui.internal.find_window_by_name("Inspector")
    assert window.dock_id == 0
    assert window.active and not window.hidden
    assert window.flags & imgui.WindowFlags_.no_title_bar
    assert "inspector-close" in study.ui.hits


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_layers_panel_updates_shared_viewport_visibility(reference):
    study, app = reference, reference.preview
    study.activate("Layers")
    study.frame(3)
    for name in ("gizmos", "helpers", "selection"):
        value = getattr(app.viewport_layers, name)
        _click(study, "layer-" + name)
        assert getattr(app.viewport_layers, name) is not value
        _click(study, "layer-" + name + "-label")
        assert getattr(app.viewport_layers, name) is value


@pytest.mark.parametrize("reference", (1.0, 2.0), indirect=True)
def test_removing_a_pinned_entity_returns_inspector_to_the_current_selection(reference):
    study, app = reference, reference.preview
    camera = next(
        node for node in app.session.nodes if node.type is NodeType.CAMERA and node.model_id < 0
    )
    app.session.submit(cmd.SelectNode(camera.node_id))
    study.frame(3)
    _click(study, "pin")
    assert study.pinned == app.entity["identity"]
    assert app.session.submit(cmd.RemoveSceneEntity(camera.object_id)).ok
    app.selected = "free_body"
    study.frame(3)
    assert study.pinned is None
    assert app.entity["name"] == "free_body"
    assert "##" + app.entity["id"] + "positionX" in study.ui.hits
