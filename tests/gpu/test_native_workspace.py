"""Production native UI: real input, command routing, panels and scale containment."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.app.composition import build_editor, build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, PanelConfig, ViewerConfig
from mojive.ui.workspace.app import WorkspaceApp

pytestmark = pytest.mark.gpu


@pytest.fixture(params=(1, 2))
def workspace(request, backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(request.param))
    viewer = build_workspace(
        resolve("joint_types"),
        renderer=backend_name,
        show_window=False,
        width=1600 if request.param == 1 else 2880,
        height=1000 if request.param == 1 else 1600,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    viewer.app.set_language("zh_CN")
    events = []
    process = viewer.window._input.process_inputs

    def inputs():
        process()
        io = imgui.get_io()
        for method, args in events:
            getattr(io, method)(*args)
        events.clear()
        io.delta_time = 1 / 60

    monkeypatch.setattr(viewer.window._input, "process_inputs", inputs)

    def frame(count=3):
        for _ in range(count):
            viewer.sync()

    def event(method, *args):
        events.append((method, args))
        frame(1)

    def click(key):
        rect = viewer.app.workspace.ui.hits[key]
        x, y, w, h = rect
        event("add_mouse_pos_event", x + w / 2, y + h / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame()

    viewer.frame, viewer.event, viewer.click = frame, event, click
    try:
        frame(8)
        yield viewer
    finally:
        viewer.release()


def test_default_production_panels_and_activity_toggle(workspace):
    viewer, shell = workspace, workspace.app.workspace
    assert isinstance(viewer.app, WorkspaceApp)
    assert viewer.panels.get("Hierarchy") is viewer.panels.get("Scene")
    assert viewer.panels.get("plot") is not None
    viewer.click("rail-Scene")
    assert not shell.open["Scene"]
    viewer.click("rail-Scene")
    assert shell.open["Scene"]
    for name in (
        "Pose",
        "Control",
        "Assets",
        "Cameras",
        "Sensors",
        "Layers",
        "Settings",
        "Scene",
        "Timeline",
        "Output",
        "Statistics",
    ):
        shell.activate(name)
        viewer.frame()
        assert shell.open[name]
        if name == "Settings":
            shell.open[name] = False
    pixels = np.asarray(viewer.window.read_frame())
    assert pixels.std() > 10


def test_scene_shift_range_and_ctrl_toggle_use_visible_row_order(workspace):
    viewer, shell = workspace, workspace.app.workspace
    navigation = shell.navigation
    ctx = viewer.app._panel_context()
    rows = navigation.scene_rows(ctx)
    visible = [row[0] for row in rows if "entity-" + str(row[0].node_id) in shell.ui.hits]
    first, last = visible[1], visible[4]
    viewer.click("entity-" + str(first.node_id))
    viewer.event("add_key_event", imgui.Key.mod_shift, True)
    viewer.click("entity-" + str(last.node_id))
    viewer.event("add_key_event", imgui.Key.mod_shift, False)
    row_ids = [row[0].node_id for row in rows]
    start, end = sorted((row_ids.index(first.node_id), row_ids.index(last.node_id)))
    assert navigation.hierarchy._batch_selected == set(row_ids[start : end + 1])
    viewer.event("add_key_event", imgui.Key.mod_ctrl, True)
    viewer.click("entity-" + str(last.node_id))
    viewer.event("add_key_event", imgui.Key.mod_ctrl, False)
    assert last.node_id not in navigation.hierarchy._batch_selected


def test_inspector_tabs_visibility_undo_and_file_dialog(workspace):
    viewer, shell = workspace, workspace.app.workspace
    node = next(n for n in viewer.session.nodes if n.name == "free_body")
    assert viewer.session.submit(cmd.SelectNode(node.node_id)).ok
    viewer.frame()
    for tab in ("Physics", "Material", "Joint", "Properties"):
        viewer.click("tab-" + tab)
        assert shell.inspector_tab == tab
    assert node.visible
    viewer.click("entity-visible")
    assert not node.visible
    viewer.click("entity-visible")
    viewer.frame()
    assert viewer.session.node(node.node_id).visible
    from mojive.scene.queries import node_world_pose

    position, rotation = node_world_pose(viewer.session, node)
    moved = position + np.array([0.2, 0.0, 0.0])
    assert viewer.session.submit(cmd.SetPose(node.node_id, moved, rotation)).ok
    shell.menus.dispatch("undo")
    viewer.frame()
    np.testing.assert_allclose(
        node_world_pose(viewer.session, viewer.session.node(node.node_id))[0], position
    )
    shell.menus.dispatch("file:save_as")
    viewer.frame()
    assert viewer.app._scene_dialog is not None


def test_settings_checkbox_and_timeline_stay_inside_windows(workspace):
    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Settings")
    viewer.frame()
    rect = shell.ui.hits["settings-inspector"]
    settings = imgui.internal.find_window_by_name("Settings")
    assert settings.pos.x <= rect[0] < rect[0] + rect[2] <= settings.pos.x + settings.size.x
    viewer.click("settings-inspector")
    assert not shell.open["Inspector"]
    viewer.click("settings-inspector")
    assert shell.open["Inspector"]
    shell.open["Settings"] = False
    shell.activate("Timeline")
    viewer.frame(8)
    timeline = imgui.internal.find_window_by_name("Timeline")
    viewport = imgui.internal.find_window_by_name("Viewport")
    assert timeline.pos.x == pytest.approx(viewport.pos.x, abs=1)
    assert timeline.size.x == pytest.approx(viewport.size.x, abs=1)


def test_timeline_flows_sequentially_and_all_lanes_fit(workspace):
    from mojive.ui.workspace.design import DESIGN

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Timeline")
    viewer.frame(8)
    hits = shell.ui.hits
    keys = (
        "timeline-first",
        "timeline-previous",
        "timeline-play-pause",
        "timeline-next",
        "timeline-last",
        "timeline-time",
        "timeline-record",
        "timeline-take",
        "timeline-snapshot",
        "timeline-video",
        "##timeline-range-start",
        "##timeline-range-end",
        "timeline-loop",
        "timeline-follow-0",
        "timeline-fit",
        "timeline-settings",
        "collapse-timeline",
    )
    window = imgui.internal.find_window_by_name("Timeline")
    previous = None
    for key in keys:
        x, y, w, h = hits[key]
        assert window.pos.x <= x < x + w <= window.pos.x + window.size.x + 1
        assert window.pos.y <= y < y + h <= window.pos.y + window.size.y + 1
        if previous is not None:
            px, py, pw, ph = previous
            assert y >= py - 1
            if abs(y - py) < 1:
                assert x >= px + pw + DESIGN.timeline_column_gap * shell.s - 1
            else:
                assert y >= py + ph + DESIGN.timeline_row_gap * shell.s - 1
        previous = (x, y, w, h)
    last_step, time = hits["timeline-last"], hits["timeline-time"]
    assert time[1] == pytest.approx(last_step[1], abs=1)
    assert time[0] - (last_step[0] + last_step[2]) == pytest.approx(
        DESIGN.timeline_column_gap * shell.s, abs=1
    )
    # Three complete lanes must fit at the default panel height, including at 2x.
    body = next(
        child
        for child in imgui.get_current_context().windows
        if child.active and "Timeline body" in child.name
    )
    for index in range(3):
        x, y, w, h = hits[f"timeline-lane-label-{index}"]
        assert body.inner_clip_rect.min.y - 1 <= y < y + h <= body.inner_clip_rect.max.y + 1


def test_selection_notices_are_quiet_and_real_results_and_receipts_remain(workspace, tmp_path):
    from mojive.ui.workspace.design import DESIGN

    viewer = workspace
    node = next(node for node in viewer.session.nodes if node.name == "hinge_body")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    viewer.frame()
    assert viewer.app._status_notice_bounds is None
    for level in ("info", "warning", "error"):
        viewer.session.report_message(node.name, level=level)
        viewer.frame()
        left, top, right, bottom = viewer.app._status_notice_bounds
        x, y, w, h = viewer.app._viewport_rect
        assert x <= left < right <= x + w
        assert y <= top < bottom <= y + h
    viewer.session.submit(cmd.Select(0))
    viewer.frame()
    assert viewer.app._status_notice_bounds is None
    saved = tmp_path / "scene.mojive.json"
    saved.write_text("{}")
    viewer.app.output.publish("Saved scene: " + str(saved), copy_text=str(saved), duration=None)
    viewer.frame()
    receipt = viewer.app._status_notice_bounds
    caption = viewer.app._status_path_bounds
    assert caption[0] - receipt[0] == pytest.approx(
        DESIGN.file_receipt_padding * viewer.window.style_scale, abs=1
    )


def test_empty_editor_and_minimal_panel_configuration(backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    config = replace(
        ViewerConfig.minimal("hierarchy"), panels={"hierarchy": PanelConfig(open=False)}
    )
    viewer = build_editor(renderer=backend_name, show_window=False, config=config)
    try:
        for _ in range(5):
            viewer.sync()
        assert isinstance(viewer.app, WorkspaceApp)
        assert not viewer.app.workspace.open["Scene"]
        assert not viewer.app.workspace.open["Inspector"]
        viewer.panels.open("hierarchy")
        viewer.sync()
        assert viewer.app.workspace.open["Scene"]
    finally:
        viewer.release()


def test_real_transform_drag_has_one_undo_and_restores_source(workspace):
    from mojive.scene.queries import node_world_pose

    viewer, shell = workspace, workspace.app.workspace
    node = next(n for n in viewer.session.nodes if n.name == "free_body")
    assert viewer.session.submit(cmd.SelectNode(node.node_id)).ok
    viewer.frame()
    original, _rotation = node_world_pose(viewer.session, node)
    key = f"##{node.node_id}positionX"
    x, y, w, h = shell.ui.hits[key]
    start = (x + w * 0.7, y + h / 2)
    viewer.event("add_mouse_pos_event", *start)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 6):
        viewer.event("add_mouse_pos_event", start[0] + step * 10, start[1])
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame()
    assert not viewer.session.editing
    moved, _ = node_world_pose(viewer.session, viewer.session.node(node.node_id))
    assert abs(moved[0] - original[0]) > 0.05
    shell.menus.dispatch("undo")
    viewer.frame()
    np.testing.assert_allclose(
        node_world_pose(viewer.session, viewer.session.node(node.node_id))[0], original
    )


def test_common_xyz_values_share_one_row_and_precision_reflows(workspace):
    from mojive.ui.workspace.design import DESIGN

    viewer, shell = workspace, workspace.app.workspace
    node = next(n for n in viewer.session.nodes if n.name == "free_body")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    shell.inspector_tab = "Properties"
    viewer.frame()
    body = next(
        window
        for window in imgui.get_current_context().windows
        if window.active and "Inspector content" in window.name
    )
    for component in ("position", "rotation", "scale"):
        fields = [shell.ui.hits[f"##{node.node_id}{component}{axis}-frame"] for axis in "XYZ"]
        assert fields[0][1] == pytest.approx(fields[1][1], abs=1)
        assert fields[0][1] == pytest.approx(fields[2][1], abs=1)
        for x, y, w, h in fields:
            assert body.inner_clip_rect.min.x - 1 <= x < x + w <= body.inner_clip_rect.max.x + 1
            assert body.inner_clip_rect.min.y - 1 <= y < y + h <= body.inner_clip_rect.max.y + 1
            assert h == pytest.approx(DESIGN.control_height * shell.s, abs=1)
        for first, second in pairwise(fields):
            assert second[0] - first[0] - first[2] == pytest.approx(
                DESIGN.numeric_gap * shell.s, abs=1
            )
    # Long precision reserves enough text width; narrow rows remain usable by stacking.
    assert shell.ui.vector_layout(296 * shell.s, format="%.7f").columns == 1
    assert (
        shell.ui.vector_layout(
            296 * shell.s, format="%.6g", values=(0.0533713, 0.0533713, 0.00288568)
        ).columns
        == 1
    )
    assert shell.ui.vector_layout(180 * shell.s, format="%.3g").columns == 1


def test_color_popup_rgba_fields_fit_and_cancel_without_crash(workspace):
    viewer, shell = workspace, workspace.app.workspace
    node = next(n for n in viewer.session.nodes if n.name == "free_box")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    shell.inspector_tab = "Material"
    viewer.frame()
    viewer.click("material-color")
    assert "color-editor-picker" in shell.ui.hits
    viewer.click("color-editor-format-2")
    px, py, pw, ph = shell.menus.observed["Color"]["rect"]
    fields = [
        rect
        for key, rect in shell.ui.hits.items()
        if key in {"color-editor" + axis for axis in "RGBA"}
    ]
    assert fields
    for x, y, w, h in fields:
        assert px <= x < x + w <= px + pw + 1
        assert py <= y < y + h <= py + ph + 1
    viewer.event("add_key_event", imgui.Key.escape, True)
    viewer.event("add_key_event", imgui.Key.escape, False)
    viewer.frame()
    assert not viewer.session.editing


def test_dock_float_restore_and_timeline_reopen(workspace):
    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Timeline")
    viewer.frame()
    shell.command("save")
    shell.command("float")
    viewer.frame()
    assert imgui.internal.find_window_by_name("Inspector").dock_id == 0
    shell.command("restore")
    viewer.frame()
    assert imgui.internal.find_window_by_name("Inspector").dock_id != 0
    shell.collapse_bottom()
    viewer.frame()
    shell.activate("Timeline")
    viewer.frame()
    assert shell.open["Timeline"]


def test_native_asset_focus_camera_tools_and_pinned_inspector_keep_production_state(workspace):
    viewer, shell = workspace, workspace.app.workspace
    model = viewer.session.scene_models[0]
    assets = viewer.session.model_assets(model.model_id)
    material = next(item for item in assets if item.type == "material")
    panel = viewer.panels.get("assets")
    panel.focus(model.model_id, "material", material.name)
    viewer.frame(8)
    assert shell.left_active == "Assets"
    assert not shell.open["Scene"]
    assert panel._selected_asset() is not None
    assert panel._assets == assets
    assert imgui.internal.find_window_by_name("Assets").active
    shell.activate("Cameras")
    viewer.frame(8)
    # The native shortcuts coexist with the registered camera parameter/bookmark tools.
    assert imgui.internal.find_window_by_name("Cameras").active
    viewer.click("camera-Top")
    viewer.app.camera.advance(1.0, viewer.backend)
    assert viewer.app.camera.pitch == pytest.approx(89.9)
    first = next(node for node in viewer.session.nodes if node.name == "free_body")
    second = next(node for node in viewer.session.nodes if node.name == "hinge_body")
    viewer.session.submit(cmd.SelectNode(first.node_id))
    viewer.frame()
    viewer.click("pin")
    viewer.session.submit(cmd.SelectNode(second.node_id))
    viewer.frame()
    assert {entry["node_id"] for entry in viewer.app.entities} == {
        first.node_id,
        second.node_id,
    }
    viewer.click("pin")
    viewer.frame()
    assert [entry["node_id"] for entry in viewer.app.entities] == [second.node_id]


def test_camera_sections_rails_units_and_native_spacing(workspace):
    from mojive.ui.workspace.design import DESIGN

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Cameras")
    viewer.frame(8)
    tr = viewer.app._panel_context().tr
    viewer.click("section-" + tr("VIEW DIRECTION"))
    hits = shell.ui.hits
    panel = viewer.panels.get("camera")
    assert panel.presentation is not None
    scale = shell.s
    window = imgui.internal.find_window_by_name("Cameras")
    content = next(
        child
        for child in imgui.get_current_context().windows
        if child.active
        and child.parent_window is not None
        and child.parent_window.id_ == window.id_
        and "Panel content" in child.name
    )
    previous = None
    for name in ("yaw", "pitch", "distance", "fov_y_deg", "far"):
        rail = hits["##camera-" + name + "-track"]
        value = hits["##camera-" + name + "-value"]
        unit = hits["##camera-" + name + "-unit"]
        for x, y, w, h in (rail, value, unit):
            assert (
                content.inner_clip_rect.min.x - 1 <= x < x + w <= content.inner_clip_rect.max.x + 1
            )
            assert (
                content.inner_clip_rect.min.y - 1 <= y < y + h <= content.inner_clip_rect.max.y + 1
            )
            assert h == pytest.approx(DESIGN.control_height * scale, abs=1)
        assert rail[1] == pytest.approx(value[1], abs=1)
        assert unit[1] == pytest.approx(value[1], abs=1)
        assert value[0] - rail[0] - rail[2] == pytest.approx(DESIGN.row_gap * scale, abs=1)
        assert unit[0] == pytest.approx(value[0] + value[2], abs=1)
        assert unit[2] == pytest.approx(DESIGN.rail_unit_width * scale, abs=1)
        assert value[2] == pytest.approx(DESIGN.rail_entry_width * scale, abs=1)
        assert rail[2] >= DESIGN.rail_minimum_track_width * scale
        if previous is not None and name in ("pitch", "distance"):
            heading = hits["camera-label-" + tr(name)]
            assert heading[1] - previous[1] - previous[3] == pytest.approx(
                (DESIGN.row_gap + 3) * scale, abs=1
            )
        previous = rail
    original = (viewer.app.camera.yaw, viewer.app.camera.pitch, viewer.app.camera.fov_y_deg)
    viewer.click("##camera-yaw-unit")
    assert not panel._angular_degrees
    assert (
        viewer.app.camera.yaw,
        viewer.app.camera.pitch,
        viewer.app.camera.fov_y_deg,
    ) == pytest.approx(original)
    viewer.click("##camera-fov_y_deg-unit")
    assert panel._angular_degrees
    assert (
        viewer.app.camera.yaw,
        viewer.app.camera.pitch,
        viewer.app.camera.fov_y_deg,
    ) == pytest.approx(original)
    x, y, w, h = hits["##camera-yaw-track"]
    viewer.event("add_mouse_pos_event", x + w * 0.75, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    assert viewer.app.camera.yaw > 0
    viewer.event("add_mouse_button_event", 1, True)
    viewer.event("add_mouse_button_event", 1, False)
    from mojive.ui.camera import DEFAULT_YAW

    assert viewer.app.camera.yaw == pytest.approx(DEFAULT_YAW)
    x, y, w, h = hits["##camera-yaw-value"]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 6):
        viewer.event("add_mouse_pos_event", x + w / 2 + step * 8 * scale, y + h / 2)
    viewer.event("add_mouse_button_event", 0, False)
    assert viewer.app.camera.yaw != pytest.approx(DEFAULT_YAW)
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 1, True)
    viewer.event("add_mouse_button_event", 1, False)
    assert viewer.app.camera.yaw == pytest.approx(DEFAULT_YAW)
    viewer.click("section-" + tr("View"))
    assert "##camera-yaw" not in hits
    assert "##camera-fov_y_deg" in hits
    viewer.click("section-" + tr("View"))
    assert "##camera-yaw" in hits


def _scroll_to_hit(viewer, key, panel_name):
    window = imgui.internal.find_window_by_name(panel_name)
    content = next(
        child
        for child in imgui.get_current_context().windows
        if child.active
        and child.parent_window is not None
        and child.parent_window.id_ == window.id_
        and "Panel content" in child.name
    )
    _x, y, _w, h = viewer.app.workspace.ui.hits[key]
    if y < content.inner_clip_rect.min.y or y + h > content.inner_clip_rect.max.y:
        target = (
            content.scroll.y + y - content.inner_clip_rect.min.y - 12 * viewer.window.style_scale
        )
        imgui.internal.set_scroll_y(content, target)
        viewer.frame()
    return content


def test_assets_native_sections_and_height_field_draft(workspace):
    from mojive.ui.workspace.design import DESIGN

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Assets")
    viewer.frame()
    tr = viewer.app._panel_context().tr
    key = "section-" + tr("Model assets")
    viewer.click(key)
    field_key = "section-" + tr("Height-field import size")
    _scroll_to_hit(viewer, field_key, "Assets")
    viewer.click(field_key)
    panel = viewer.panels.get("assets")
    original = panel._hfield_import_size.copy()
    control = "##hfield-import-size-0"
    content = _scroll_to_hit(viewer, control, "Assets")
    x, y, w, h = shell.ui.hits[control]
    assert h == pytest.approx(DESIGN.control_height * shell.s, abs=1)
    assert content.inner_clip_rect.min.x <= x < x + w <= content.inner_clip_rect.max.x + 1
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 5):
        viewer.event("add_mouse_pos_event", x + w / 2 + step * 8, y + h / 2)
    viewer.event("add_mouse_button_event", 0, False)
    assert panel._hfield_import_size[0] != original[0]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 1, True)
    viewer.event("add_mouse_button_event", 1, False)
    # This is an import draft; restore it without changing scene assets.
    panel._hfield_import_size[:] = original
    viewer.frame()
    _scroll_to_hit(viewer, field_key, "Assets")
    viewer.click(field_key)
    assert control not in shell.ui.hits


def test_settings_every_page_uses_native_fields_and_plot_toggles(workspace):
    from mojive.ui.workspace.design import DESIGN
    from mojive.ui.workspace.settings_panel import CATEGORIES

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Settings")
    viewer.frame()
    for category in CATEGORIES:
        viewer.click("settings-" + category)
        assert shell.settings._category == category
        fields = [
            rect
            for key, rect in shell.ui.hits.items()
            if key.startswith("##") and key.endswith("-frame")
        ]
        for _x, _y, w, h in fields:
            assert h == pytest.approx(DESIGN.control_height * shell.s, abs=1)
            assert w > 0
    shell.open["Settings"] = False
    viewer.panels.open_panel("Plot")
    viewer.frame()
    panel = viewer.panels.get("plot")
    assert panel.presentation is not None
    original = panel.show_angle
    viewer.click("plot-show_angle")
    assert panel.show_angle is not original
    viewer.click("plot-show_angle")
    assert panel.show_angle is original
    viewer.click("plot-source-1")
    assert panel.source == "sensor"
    viewer.click("plot-source-0")
    assert panel.source == "joint"


def test_control_layers_sensor_output_and_statistics_share_native_geometry(workspace):
    from mojive.tools.ui_runtime import _item_center
    from mojive.ui.workspace.design import DESIGN

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Control")
    viewer.frame()
    original = viewer.session.frame.ctrl.copy()
    x, y, w, h = shell.ui.hits["##ctrl-0-value"]
    assert h == pytest.approx(DESIGN.control_height * shell.s, abs=1)
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 5):
        viewer.event("add_mouse_pos_event", x + w / 2 + step * 8, y + h / 2)
    viewer.event("add_mouse_button_event", 0, False)
    assert viewer.session.frame.ctrl[0] != original[0]
    viewer.click("reset-actuators")
    np.testing.assert_allclose(viewer.session.frame.ctrl, 0)
    shell.activate("Layers")
    viewer.frame()
    original = viewer.app.viewport_layers.gizmos
    viewer.click("layer-gizmos")
    assert viewer.app.viewport_layers.gizmos is not original
    viewer.click("layer-gizmos")
    assert viewer.app.viewport_layers.gizmos is original
    for first, second in pairwise(("viewport_ui", "gizmos", "selection", "helpers")):
        a, b = shell.ui.hits["layer-" + first], shell.ui.hits["layer-" + second]
        assert b[1] - a[1] - a[3] == pytest.approx(DESIGN.row_gap * shell.s, abs=1)
    shell.activate("Sensors")
    viewer.frame()
    sensor = viewer.panels.get("sensors")
    assert sensor.presentation is not None
    if len(viewer.session.sensor_infos) > 1:
        x, y = _item_center(viewer, "combo", "##sensor")
        viewer.event("add_mouse_pos_event", x, y)
        viewer.event("add_mouse_button_event", 0, True)
        viewer.event("add_mouse_button_event", 0, False)
        viewer.event("add_key_event", imgui.Key.down_arrow, True)
        viewer.event("add_key_event", imgui.Key.down_arrow, False)
        viewer.event("add_key_event", imgui.Key.enter, True)
        viewer.event("add_key_event", imgui.Key.enter, False)
        assert sensor.sensor_index == 1
    shell.activate("Output")
    viewer.frame()
    output = viewer.panels.get("output")
    original = output.auto_scroll
    viewer.click("output-follow")
    assert output.auto_scroll is not original
    viewer.click("output-follow")
    assert output.auto_scroll is original
    viewer.click("output-search")
    assert output.show_search
    viewer.click("output-search")
    assert not output.show_search
    shell.activate("Statistics")
    viewer.frame()
    assert viewer.panels.get("stats")._frame_ms.filled > 0


@pytest.mark.parametrize(
    "asset,panel_name", (("constraints", "Control"), ("rangefinder", "Sensors"))
)
def test_populated_constraints_and_sensor_views_use_real_state(workspace, asset, panel_name):
    from mojive.tools.ui_runtime import _item_center

    viewer, shell = workspace, workspace.app.workspace
    assert viewer.session.submit(cmd.LoadAsset(resolve(asset))).ok
    viewer.frame(8)
    shell.activate(panel_name)
    viewer.frame()
    if panel_name == "Control":
        header = shell.ui.hits["section-" + viewer.app._panel_context().tr("actuators")]
        empty = shell.ui.hits["control-empty"]
        assert empty[1] >= header[1] + header[3]
        constraint = viewer.session.equality_constraints[0]
        original = constraint.enabled
        viewer.click(f"equality-{constraint.constraint_id}")
        assert viewer.session.equality_constraints[0].enabled is not original
        viewer.click(f"equality-{constraint.constraint_id}")
        assert viewer.session.equality_constraints[0].enabled is original
    else:
        assert len(viewer.session.sensor_infos) >= 2
        point = _item_center(viewer, "combo", "##sensor")
        viewer.event("add_mouse_pos_event", *point)
        viewer.event("add_mouse_button_event", 0, True)
        viewer.event("add_mouse_button_event", 0, False)
        viewer.event("add_key_event", imgui.Key.down_arrow, True)
        viewer.event("add_key_event", imgui.Key.down_arrow, False)
        viewer.event("add_key_event", imgui.Key.enter, True)
        viewer.event("add_key_event", imgui.Key.enter, False)
        assert viewer.panels.get("sensors").sensor_index == 1
        assert viewer.session.frame.sensors is not None


def test_native_settings_numeric_preview_reset_and_shared_color_popup(workspace):
    from mojive.config import CameraNavigationConfig

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Settings")
    viewer.frame()
    viewer.click("settings-Camera")
    x, y, w, h = shell.ui.hits["##camera_focus_duration"]
    original = viewer.app.camera.navigation.focus_duration
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 4):
        viewer.event("add_mouse_pos_event", x + w / 2 + step * 8, y + h / 2)
    viewer.event("add_mouse_button_event", 0, False)
    assert viewer.app.camera.navigation.focus_duration != original
    viewer.event("add_mouse_button_event", 1, True)
    viewer.event("add_mouse_button_event", 1, False)
    assert viewer.app.camera.navigation.focus_duration == pytest.approx(
        CameraNavigationConfig().focus_duration
    )
    viewer.click("settings-MuJoCo Visuals")
    original_color = viewer.backend.get_geometry_style().collision_color
    viewer.click("##collision_color-swatch")
    assert "##collision_color-picker" in shell.ui.hits
    viewer.click("##collision_color-format-2")
    assert "##collision_colorR" in shell.ui.hits
    x, y, w, h = shell.ui.hits["##collision_colorR"]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 4):
        viewer.event("add_mouse_pos_event", x + w / 2 - step * 10, y + h / 2)
    viewer.event("add_mouse_button_event", 0, False)
    assert viewer.backend.get_geometry_style().collision_color != original_color
    viewer.click("##collision_color-cancel")
    assert "##collision_color-picker" not in shell.ui.hits
    assert viewer.backend.get_geometry_style().collision_color == pytest.approx(original_color)
    viewer.click("##collision_color-swatch")
    viewer.click("##collision_color-format-2")
    x, y, w, h = shell.ui.hits["##collision_colorR"]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 4):
        viewer.event("add_mouse_pos_event", x + w / 2 - step * 10, y + h / 2)
    viewer.event("add_mouse_button_event", 0, False)
    current_color = viewer.backend.get_geometry_style().collision_color
    viewer.click("##collision_color-done")
    assert viewer.app.preferences.get("geometry_style")["collision_color"] == pytest.approx(
        current_color
    )


def test_camera_narrow_rails_stack_and_model_source_preserves_editor_camera(workspace):
    from mojive.ui.workspace.design import DESIGN

    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Cameras")
    viewer.frame(8)
    scale = shell.s
    window = imgui.internal.find_window_by_name("Cameras")
    imgui.internal.dock_context_queue_undock_window(imgui.get_current_context(), window)
    viewer.frame()
    imgui.get_style().window_min_size = (1, 1)
    imgui.set_window_size("Cameras", (180 * scale, 740 * scale))
    viewer.frame()
    tr = viewer.app._panel_context().tr
    viewer.click("section-" + tr("VIEW DIRECTION"))
    hits = shell.ui.hits
    for name in ("yaw", "pitch", "distance", "fov_y_deg", "far"):
        rail = hits["##camera-" + name + "-track"]
        value = hits["##camera-" + name + "-value"]
        unit = hits["##camera-" + name + "-unit"]
        assert value[1] - rail[1] - rail[3] == pytest.approx(DESIGN.row_gap * scale, abs=1)
        assert value[0] == pytest.approx(rail[0], abs=1)
        assert value[0] + value[2] + unit[2] == pytest.approx(rail[0] + rail[2], abs=1)
        assert window.pos.x <= rail[0] < rail[0] + rail[2] <= window.pos.x + window.size.x
    original = (viewer.app.camera.yaw, viewer.app.camera.pitch, viewer.app.camera.distance)
    model_camera = viewer.session.cameras[0]
    viewer.app.select_model_camera(model_camera.camera_id)
    viewer.frame()
    assert "##camera-yaw" not in hits
    assert "camera-editor" in hits
    viewer.click("camera-editor")
    assert viewer.app._model_camera_id == -1
    assert "##camera-yaw" in hits
    assert (
        viewer.app.camera.yaw,
        viewer.app.camera.pitch,
        viewer.app.camera.distance,
    ) == pytest.approx(original)


def test_shared_physics_draft_apply_undo_and_menu_use_native_controls(workspace):
    from mojive.tools.ui_runtime import _item_center, _open_main_menu

    viewer, shell = workspace, workspace.app.workspace
    node = next(n for n in viewer.session.nodes if n.name == "free_body")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    shell.inspector_tab = "Physics"
    viewer.frame()
    current = viewer.session.body_properties(node.node_id)
    x, y, w, h = shell.ui.hits["##body-gravity-compensation"]
    start = (x + w * 0.7, y + h / 2)
    viewer.event("add_mouse_pos_event", *start)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 6):
        viewer.event("add_mouse_pos_event", start[0] + step * 8, start[1])
    viewer.event("add_mouse_button_event", 0, False)
    panel = viewer.panels.get("inspector")
    assert panel._body_property_edit.gravity_compensation != current.gravity_compensation
    assert viewer.session.body_properties(node.node_id) == current
    point = _item_center(
        viewer, "button", viewer.app.localizer.text("Apply") + "##body-properties", reveal=True
    )
    viewer.event("add_mouse_pos_event", *point)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame(12)
    if viewer.app.model_edits.active:
        viewer.click("apply-model-edits")
        viewer.frame(24)
        assert not viewer.app.model_edits.active
        assert not viewer.app.model_edits.error
    assert (
        viewer.session.body_properties(node.node_id).gravity_compensation
        != current.gravity_compensation
    )
    shell.menus.dispatch("undo")
    viewer.frame(12)
    assert (
        viewer.session.body_properties(node.node_id).gravity_compensation
        == current.gravity_compensation
    )
    _open_main_menu(viewer, "Window")
    point = _item_center(viewer, "menu_item", "Physics Options...")
    viewer.event("add_mouse_pos_event", *point)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame()
    assert viewer.session.selected_node.type.value == "environment"
    assert shell.inspector_tab == "Physics"


def test_adopted_dimensions_share_drawing_and_input_geometry(workspace):
    viewer, shell = workspace, workspace.app.workspace
    scale = shell.s
    application = imgui.internal.find_window_by_name("Application")
    assert application.size.y == pytest.approx(46 * scale, abs=1)
    assert not any(key.startswith("workspace-") for key in shell.ui.hits)
    node = next(n for n in viewer.session.nodes if n.name == "free_body")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    shell.inspector_tab = "Physics"
    viewer.frame()
    assert shell.ui.hits["##body-gravity-compensation"][3] == pytest.approx(28 * scale, abs=1)
    layout = shell.ui.numeric_layout(600 * scale, 3)
    first, second = layout.fields[:2]
    assert second[0] - first[0] - first[2] == pytest.approx(4 * scale)
    wrapped = shell.ui.numeric_layout(100 * scale, 3)
    first, second = wrapped.fields[:2]
    assert second[1] - first[1] - first[3] == pytest.approx(8 * scale)
    shell.activate("Timeline")
    viewer.frame(8)
    track = shell.timeline_panel.last_layout
    assert track.row_height == pytest.approx(36 * scale)
    lane = shell.ui.hits["timeline-lane-label-0"]
    assert lane[0] == track.lo[0]
    assert lane[2] == track.channel_width
    assert lane[3] == track.row_height


@pytest.mark.parametrize(
    "category,field,config_name,attribute,part",
    (
        ("Camera", "##camera_focus_duration", "camera_navigation", "focus_duration", ""),
        ("Recording", "##recording_delay", "recording", "countdown", ""),
        ("Recording", "##recording_crf", "recording", "crf", "-value"),
        ("Recording", "##recording_crf", "recording", "crf", "-track"),
    ),
)
def test_native_settings_escape_restores_preview_without_saving(
    workspace, category, field, config_name, attribute, part
):
    viewer, shell = workspace, workspace.app.workspace
    shell.activate("Settings")
    viewer.frame()
    viewer.click("settings-" + category)

    def current():
        config = (
            viewer.app.camera.navigation if category == "Camera" else viewer.app.recording_config
        )
        return getattr(config, attribute)

    original = current()
    saved = viewer.app.preferences.get(config_name)
    x, y, w, h = shell.ui.hits[field + part]
    content = next(
        window
        for window in imgui.get_current_context().windows
        if window.active and "Reference settings content" in window.name
    )
    if y + h > content.clip_rect.max.y:
        imgui.internal.set_scroll_y(
            content, content.scroll.y + y + h - content.clip_rect.max.y + 8 * shell.s
        )
        viewer.frame()
        x, y, w, h = shell.ui.hits[field + part]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    for step in range(1, 4):
        viewer.event("add_mouse_pos_event", x + w / 2 + step * 40 * shell.s, y + h / 2)
    assert current() != original
    viewer.event("add_key_event", imgui.Key.escape, True)
    assert current() == original
    viewer.event("add_key_event", imgui.Key.escape, False)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame()
    assert current() == original
    assert viewer.app.preferences.get(config_name) == saved
