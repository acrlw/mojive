"""Native workbench navigation and transform edits through real ImGui input."""

from __future__ import annotations

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.render.selection import render_backend_name
from mojive.scene.queries import node_geometry_indices, node_world_pose
from tests.gpu.test_ui_design_gizmos import reference as reference
from tests.gpu.test_ui_design_gizmos import wait_edits

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session")
def backend_name():
    return render_backend_name()


def _center(rect):
    x, y, width, height = rect
    return x + width / 2, y + height / 2


def _click(study, key):
    study.click(_center(study.ui.hits[key]))
    study.frame(3)


def _key(study, key):
    study.event("add_key_event", key, True)
    study.event("add_key_event", key, False)


def _drag(study, key, *, direction=1):
    start = _center(study.ui.hits[key])
    study.event("add_mouse_pos_event", *start)
    study.event("add_mouse_button_event", 0, True)
    for offset in (8, 16, 24, 32):
        study.event("add_mouse_pos_event", start[0] + offset * study.s * direction, start[1])


def _release(study):
    study.event("add_mouse_button_event", 0, False)
    study.frame(3)


def _number_text(study, key, text):
    modifier = (
        imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl
    )
    study.event("add_key_event", modifier, True)
    study.click(_center(study.ui.hits[key]))
    _key(study, imgui.Key.a)
    study.event("add_key_event", modifier, False)
    assert imgui.get_io().want_text_input, "Ctrl-click did not activate the numeric text editor"
    study.event("add_input_characters_utf8", text)


def _box(study):
    study.menus.dispatch("create:Box")
    study.frame(3)
    node = study.preview.session.selected_node
    assert node is not None and node.posable
    assert study.preview.entity["scalable"]
    return node


def _position(study, node_id):
    session = study.preview.session
    return node_world_pose(session, session.node(node_id))[0].copy()


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_left_rail_switches_one_panel_without_changing_selection(reference):
    study = reference
    selected = study.preview.session.selected_node.node_id
    _click(study, "rail-Control")
    assert study.left_active == "Control"
    assert study.open["Control"] and not study.open["Scene"]
    assert not study.open["Assets"] and not study.open["Pose"]
    assert "##ctrl-0-track" in study.ui.hits
    assert f"entity-{selected}" not in study.ui.hits
    assert study.preview.session.selected_node.node_id == selected
    _click(study, "rail-Scene")
    assert study.left_active == "Scene"
    assert study.open["Scene"] and not study.open["Control"]
    assert f"entity-{selected}" in study.ui.hits
    assert not any(key.startswith("##ctrl-") for key in study.ui.hits)
    assert study.preview.session.selected_node.node_id == selected


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_reset_layout_restores_single_sidebar_after_inspector_grouping(reference):
    study = reference
    study.pending = "group"
    study.frame(4)
    scene = imgui.internal.find_window_by_name("Scene")
    inspector = imgui.internal.find_window_by_name("Inspector")
    assert scene.dock_id == inspector.dock_id
    assert scene.dock_node.tab_bar is not None
    study.pending = "reset"
    study.frame(4)
    assert scene.dock_id != inspector.dock_id
    assert scene.dock_node.tab_bar is None
    assert "header-Scene-more" in study.ui.hits
    assert "undo-footer" not in study.ui.hits


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_sidebar_tabs_follow_actual_inspector_docking_and_restore(reference):
    study = reference
    _click(study, "rail-Control")
    control = imgui.internal.find_window_by_name("Control")
    scene = imgui.internal.find_window_by_name("Scene")
    inspector = imgui.internal.find_window_by_name("Inspector")
    imgui.internal.dock_builder_dock_window("Scene", study.right)
    imgui.internal.dock_builder_dock_window("Inspector", control.dock_id)
    imgui.internal.dock_builder_finish(study.window.dockspace_id)
    study.frame(4)
    assert scene.dock_id != control.dock_id == inspector.dock_id
    assert control.dock_node.tab_bar is not None
    study.pending = "save"
    study.frame(3)
    study.pending = "float"
    study.frame(4)
    assert inspector.dock_id == 0
    assert control.dock_node.tab_bar is None
    study.pending = "restore"
    study.frame(1)
    owner = imgui.get_current_context()
    other = imgui.create_context()
    try:
        study.frame(4)
    finally:
        imgui.destroy_context(other)
        imgui.set_current_context(owner)
    assert control.dock_id == inspector.dock_id != 0
    assert control.dock_node.tab_bar is not None
    study.open["Inspector"] = False
    study.frame(4)
    assert control.dock_node.tab_bar is None
    assert "header-Control-more" in study.ui.hits


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_scene_header_search_more_and_frame_keep_selection(reference):
    study, app = reference, reference.preview
    selected = app.session.selected_node.node_id
    entity_key = f"entity-{selected}"
    row_y = study.ui.hits[entity_key][1]
    assert not study.search_visible
    assert "undo-footer" not in study.ui.hits
    _click(study, "header-Scene-search")
    assert study.search_visible
    assert study.ui.hits[entity_key][1] > row_y + 20 * study.s
    assert app.session.selected_node.node_id == selected
    _click(study, "header-Scene-search")
    assert not study.search_visible
    assert study.ui.hits[entity_key][1] == pytest.approx(row_y, abs=study.s)
    assert app.session.selected_node.node_id == selected

    if "header-Scene-frame" in study.ui.hits:
        before = app._camera_view()
        eye, target = before.eye.copy(), before.target.copy()
        _click(study, "header-Scene-frame")
        study.frame(12)
        after = app._camera_view()
        assert not np.allclose(after.eye, eye) or not np.allclose(after.target, target)
        assert app.session.selected_node.node_id == selected

    _click(study, "header-Scene-more")
    scene = imgui.internal.find_window_by_name("Scene")
    popup_id = scene.get_id("Panel actions###Scene")
    popups = [
        popup
        for popup in imgui.get_current_context().open_popup_stack
        if popup.popup_id == popup_id
    ]
    assert len(popups) == 1, "More did not open the Scene actions popup"
    popup = popups[0].window
    assert popup is not None and popup.active and not popup.hidden
    assert popup.size.x > 0 and popup.size.y > 0
    assert app.session.selected_node.node_id == selected
    assert "undo-footer" not in study.ui.hits


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_output_timeline_statistics_share_bottom_tabs(reference):
    study = reference
    selected = study.preview.session.selected_node.node_id
    left = study.left_active
    _click(study, "rail-Output")
    assert study.bottom_open and study.open["Output"]
    _key(study, imgui.Key.t)
    study.frame(3)
    assert study.open["Timeline"]
    _click(study, "rail-Statistics")
    names = ("Output", "Timeline", "Statistics")
    windows = [imgui.internal.find_window_by_name(name) for name in names]
    assert all(window is not None and window.dock_id for window in windows)
    assert len({window.dock_id for window in windows}) == 1
    assert all(study.open[name] for name in names)
    for window in windows:
        bar = window.dock_node.tab_bar
        assert bar is not None, "Bottom panels need a visible native tab bar"
        tab = next(item for item in bar.tabs if item.id_ == window.tab_id)
        study.click(
            (
                bar.bar_rect.min.x + tab.offset - bar.scrolling_anim + tab.width * 0.4,
                (bar.bar_rect.min.y + bar.bar_rect.max.y) / 2,
            )
        )
        study.frame(3)
        assert window.dock_node.selected_tab_id == window.tab_id
        assert study.left_active == left
        assert study.preview.session.selected_node.node_id == selected
        if str(window.name).rsplit("###", 1)[-1] == "Output":
            assert study.preview._status_panel == "Output"
            assert study.preview._panel_status_hints


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_transform_drag_undo_exact_text_and_escape(reference):
    study, app = reference, reference.preview
    count = len(app.session.nodes)
    node = _box(study)
    key = f"##{node.node_id}positionX"
    original = _position(study, node.node_id)
    _drag(study, key)
    dragging = _position(study, node.node_id)
    assert dragging[0] > original[0] + 0.02
    np.testing.assert_allclose(dragging[1:], original[1:], atol=1e-5)
    _release(study)
    assert not app.session.editing
    assert app.session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(_position(study, node.node_id), original, atol=1e-5)

    _number_text(study, key, "0.125")
    _key(study, imgui.Key.enter)
    study.frame(3)
    expected = original.copy()
    expected[0] = 0.125
    np.testing.assert_allclose(_position(study, node.node_id), expected, atol=1e-5)
    assert not app.session.editing
    assert app.session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(_position(study, node.node_id), original, atol=1e-5)

    _number_text(study, key, "0.375")
    _key(study, imgui.Key.escape)
    study.frame(3)
    assert app.session.selected_node is not None
    assert app.session.selected_node.node_id == node.node_id
    np.testing.assert_allclose(_position(study, node.node_id), original, atol=1e-5)
    assert not app.session.editing
    assert app.session.submit(cmd.Undo()).ok
    assert len(app.session.nodes) == count, "Canceled text created an extra undo entry"


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_transform_scale_drag_uses_apply_and_single_undo(reference):
    study, app = reference, reference.preview
    node = _box(study)
    before = app.session.source.geom_size.copy()
    _drag(study, f"##{node.node_id}scaleX")
    _release(study)
    assert app.model_edits.active
    assert "apply-model-edits" in study.ui.hits
    assert "discard-model-edits" in study.ui.hits
    _click(study, "apply-model-edits")
    wait_edits(study)
    assert not app.model_edits.active
    assert not np.allclose(app.session.source.geom_size, before)
    assert app.session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(app.session.source.geom_size, before, atol=1e-6)


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_material_alpha_drag_preserves_rgb_and_undoes_the_whole_gesture(reference):
    study, app = reference, reference.preview
    node = _box(study)
    _click(study, "tab-Material")
    indices = node_geometry_indices(app.session, node)
    assert len(indices)
    before = app.session.source.geom_rgba.copy()
    assert np.all(before[indices, 3] > 0.5)
    _drag(study, f"##{node.node_id}colorA", direction=-1)
    dragging = app.session.source.geom_rgba.copy()
    assert np.all(dragging[indices, 3] < before[indices, 3] - 0.02)
    np.testing.assert_allclose(dragging[:, :3], before[:, :3], atol=1e-6)
    np.testing.assert_allclose(
        np.delete(dragging[:, 3], indices), np.delete(before[:, 3], indices), atol=1e-6
    )
    _release(study)
    assert not app.session.editing
    assert app.session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(app.session.source.geom_rgba, before, atol=1e-6)


@pytest.mark.parametrize("reference", (1.0, 2.25), indirect=True)
def test_scale_escape_cancels_the_pending_gesture_and_its_undo_entry(reference):
    study, app = reference, reference.preview
    count = len(app.session.nodes)
    node = _box(study)
    before = app.session.source.geom_size.copy()
    factors = np.array(app.session.scale_factors(node.node_id))
    _drag(study, f"##{node.node_id}scaleX")
    assert app.model_edits.active
    _key(study, imgui.Key.escape)
    _release(study)
    assert not app.model_edits.active
    assert not app.session.editing
    assert app.session.selected_node is not None
    assert app.session.selected_node.node_id == node.node_id
    assert "apply-model-edits" not in study.ui.hits
    np.testing.assert_allclose(app.session.source.geom_size, before, atol=1e-6)
    np.testing.assert_allclose(app.session.scale_factors(node.node_id), factors, atol=1e-6)
    assert app.session.submit(cmd.Undo()).ok
    assert len(app.session.nodes) == count, "Canceled scale created an extra undo entry"
