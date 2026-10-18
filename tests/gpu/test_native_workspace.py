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


def test_every_panel_has_owned_groups_and_local_header_edits(
    workspace, backend_name, monkeypatch, tmp_path
):
    import json
    from copy import deepcopy
    from pathlib import Path

    from PIL import Image

    from mojive.adapters.base import NodeType
    from mojive.design.groups import component_groups, group_part
    from mojive.tools.design_audit_scenes import write_audit_model, write_component_model
    from mojive.ui.workspace import inspector_style, property_style, scene_panels

    viewer, shell = workspace, workspace.app.workspace
    runtime = viewer.app.design
    runtime.observe_components = True
    result = viewer.session.submit(cmd.LoadAsset(write_audit_model(tmp_path / "panel-groups.xml")))
    assert result.ok, result.message
    viewer.frame(5)
    directory = Path("output/panel-groups") / backend_name / f"{shell.s:g}x"
    directory.mkdir(parents=True, exist_ok=True)
    seen, samples = set(), {}

    def records():
        return [*runtime.observed, *shell.ui.design.primitives.observed]

    def observe(owner):
        viewer.frame(4)
        assert not any(
            window.active and window.name == "Debug##Default"
            for window in imgui.get_current_context().windows
        )
        rows = records()
        parts = {part for row in rows for part in row["parts"]}
        parts.update(
            runtime.catalog[part]["controlGroup"]
            for part in tuple(parts)
            if "controlGroup" in runtime.catalog.get(part, {})
        )
        seen.update(parts)
        samples[owner] = sorted(
            set(samples.get(owner, ()))
            | {part for part in parts if runtime.catalog.get(part, {}).get("groupOwner") == owner}
        )
        return rows

    for owner in component_groups():
        if owner.startswith("Settings/"):
            shell.settings.show_category(owner.removeprefix("Settings/"))
            shell.activate("Settings")
            rows = observe(owner)
            assert group_part(owner, "settings_page") in samples[owner]
        else:
            if owner in ("Help", "Info"):
                viewer.panels.get(owner.lower()).open = True
            else:
                shell.activate(owner)
            if owner == "Inspector":
                viewer.app.selected = "hinge_body"
            rows = observe(owner)
            assert group_part(owner, "panel_body") in samples[owner]
        if "panel_header" in component_groups()[owner]["parts"]:
            part = group_part(owner, "panel_header")
            assert part in samples[owner]
            body_before = next(row["hit"] for row in rows if row["key"] == "panel-body-" + owner)
            spec = deepcopy(runtime.spec)
            spec["parts"][part] = {
                "fontSize": 20,
                "iconSize": 25,
                "columnGap": 7,
                "textColor": "#f4b860",
            }
            runtime.update_spec(spec)
            rows = observe(owner)
            actions = [
                row
                for row in rows
                if row["key"].startswith("header-" + owner + "-") and row.get("kind") == "button"
            ]
            if actions:
                assert all(row["fieldValues"]["iconSize"] == 25 for row in actions)
            assert any(row["key"] == "panel-body-" + owner for row in rows)
            assert body_before[2] > 0
            Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(
                directory / (owner.lower() + ".png")
            )
            spec["parts"].pop(part)
            runtime.update_spec(spec)
        if owner == "Settings" or owner.startswith("Settings/"):
            shell.open["Settings"] = False
        if owner == "Timeline":
            viewer.frame(4)
            viewer.click("timeline-settings")
            observe(owner)
            assert group_part(owner, "form_popup") in samples[owner]
            viewer.event("add_key_event", imgui.Key.escape, True)
            viewer.event("add_key_event", imgui.Key.escape, False)
        if owner in ("Help", "Info"):
            assert group_part(owner, "window_title") in samples[owner]
            viewer.panels.get(owner.lower()).open = False

    original_section = property_style.property_section

    def expanded_section(*args, **kwargs):
        imgui.set_next_item_open(True, imgui.Cond_.always)
        return original_section(*args, **kwargs)

    for module in (property_style, inspector_style, scene_panels):
        monkeypatch.setattr(module, "property_section", expanded_section)
    viewer.app.selected = None
    shell.activate("Inspector")
    rows = observe("Inspector")
    assert any("inspector_panel_body" in row["parts"] for row in rows)
    shell.activate("Assets")
    observe("Assets")
    shell.activate("Inspector")
    for name, tab in (("hinge_body", "Physics"), ("free_box", "Material")):
        viewer.app.selected = name
        shell.inspector_tab = tab
        observe("Inspector")
    viewer.backend.debug.layer("panel-group-coverage").line(
        "sample", (0, 0, 0), (1, 0, 0), (1, 1, 1, 1)
    )
    shell.activate("Layers")
    observe("Layers")
    model_ids = {model.model_id for model in viewer.session.scene_models}
    result = viewer.session.submit(
        cmd.AddSceneModel(
            write_component_model(tmp_path / "component-groups.xml"), position=(10, 0, 0)
        )
    )
    assert result.ok, result.message
    viewer.frame(5)
    added = [model for model in viewer.session.scene_models if model.model_id not in model_ids]
    assert len(added) == 1
    model_node = next(
        node
        for node in viewer.session.nodes
        if node.type is NodeType.MODEL and node.model_id == added[0].model_id
    )
    assert viewer.session.submit(cmd.SelectNode(model_node.node_id)).ok
    shell.activate("Inspector")
    shell.inspector_tab = "Properties"
    observe("Inspector")
    component = next(
        item for item in viewer.session.model_components(added[0].model_id, "tendon") if item.path
    )
    viewer.panels.get("inspector")._begin_component_edit(component)
    observe("Inspector")
    report = {
        "observed": samples,
        "unobservedGroups": sorted(
            part
            for part, definition in runtime.catalog.items()
            if "groupOwner" in definition and part not in seen
        ),
    }
    (directory / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    assert not report["unobservedGroups"]


def test_status_readouts_have_independent_divider_sides_and_unit_insets(workspace, backend_name):
    from copy import deepcopy
    from pathlib import Path

    from PIL import Image

    viewer, shell = workspace, workspace.app.workspace
    runtime = viewer.app.design
    runtime.observe_components = True
    viewer.app._status_metric_mode = "steps"
    viewer.frame()

    def columns():
        return {
            row["parts"][0]: row
            for row in runtime.observed
            if row["key"].startswith("application-status_") and "valueRect" in row
        }

    before = columns()
    assert set(before) == {
        "status_backend",
        "status_steps",
        "status_delta",
        "status_physics",
        "status_render",
    }
    for part in ("status_physics", "status_render"):
        row = before[part]
        assert row["hit"][0] + row["hit"][2] - row["padding"][2] == pytest.approx(
            row["unitRight"] + row["trailingReserve"]
        )
        assert row["unitX"] - row["valueTextRect"][0] - row["valueTextRect"][2] == (
            pytest.approx(row["unitGap"])
        )
    directory = Path("output/status-spacing") / backend_name / f"{shell.s:g}x"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(
        directory / "status-default.png"
    )
    spec = deepcopy(runtime.spec)
    spec["parts"].update(
        {
            "status_backend": {
                "fontFamily": "portable",
                "textColor": "#f4b860",
                "valueWidth": 72,
                "paddingRight": 3,
                "rightGap": 14,
            },
            "status_steps": {"valueWidth": 26, "rightGap": 4},
            "status_delta": {"valueWidth": 24, "rightGap": 5, "unitGap": 1},
            "status_physics": {"valueWidth": 16, "paddingRight": 2, "rightGap": 7},
            "status_render": {"valueWidth": 34, "leftGap": 11, "rightGap": 12},
        }
    )
    runtime.update_spec(spec)
    viewer.frame()
    after = columns()
    for name, values in spec["parts"].items():
        if name in after:
            assert after[name]["rightGap"] == pytest.approx(values["rightGap"] * shell.s)
            assert after[name]["valueRect"][2] == pytest.approx(values["valueWidth"] * shell.s)
    physics, render = after["status_physics"], after["status_render"]
    physics_right = physics["hit"][0] + physics["hit"][2]
    divider = physics_right + physics["rightGap"] + shell.s / 2
    assert physics_right - physics["unitRight"] - physics["trailingReserve"] == (
        pytest.approx(2 * shell.s)
    )
    assert render["hit"][0] - divider - shell.s / 2 == pytest.approx(11 * shell.s)
    assert any(
        row["hit"][0] + row["hit"][2] / 2 == pytest.approx(divider)
        for row in runtime.observed
        if row["parts"] == ("status_separator",)
    )
    assert after["status_render"]["unitX"] != before["status_render"]["unitX"]
    fixed = {name: (row["valueRect"], row["unitX"]) for name, row in after.items()}
    viewer.frame(8)
    assert {name: (row["valueRect"], row["unitX"]) for name, row in columns().items()} == fixed
    spec["parts"]["status_physics"]["rightGap"] = 17
    runtime.update_spec(spec)
    viewer.frame()
    changed = columns()
    assert changed["status_render"]["hit"] == render["hit"]
    assert changed["status_render"]["leftGap"] == render["leftGap"]
    assert changed["status_physics"]["unitRight"] == pytest.approx(
        physics["unitRight"] - 10 * shell.s
    )
    Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(
        directory / "status-custom.png"
    )


def test_native_titles_and_dock_tabs_have_separate_typography(workspace):
    from copy import deepcopy

    viewer, shell = workspace, workspace.app.workspace
    runtime = viewer.app.design
    runtime.observe_components = True
    shell.activate("Timeline")
    viewer.frame(5)
    baseline = next(row for row in runtime.observed if "dock_tabs" in row["parts"])
    before = tuple(baseline["hit"])
    spec = deepcopy(runtime.spec)
    spec["parts"].update(
        {
            "dock_tabs": {
                "fontSize": 20,
                "fontFamily": "portable",
                "paddingX": 14,
                "paddingY": 8,
                "gap": 9,
            },
            "help_window_title": {"fontSize": 22, "paddingY": 10},
        }
    )
    runtime.update_spec(spec)
    for panel in ("help", "info"):
        viewer.panels.get(panel).open = True
    viewer.frame(6)
    dock = next(row for row in runtime.observed if "dock_tabs" in row["parts"])
    assert dock["nativeFontSize"] == 20 * shell.s
    assert dock["hit"][3] > before[3]
    titles = {
        row["key"]: row["hit"] for row in runtime.observed if row["key"].startswith("window-title/")
    }
    assert titles["window-title/Help"][3] > titles["window-title/Info"][3]


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
    modifier = (
        imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl
    )
    viewer.event("add_key_event", modifier, True)
    viewer.click("entity-" + str(last.node_id))
    viewer.event("add_key_event", modifier, False)
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


def test_timeline_flows_sequentially_and_all_lanes_are_reachable(workspace):
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
    # Preserve tuned lane sizes; short panels scroll to the remaining lanes.
    body = next(
        child
        for child in imgui.get_current_context().windows
        if child.active and "Timeline body" in child.name
    )
    for index in range(3):
        x, y, w, h = shell.ui.hits[f"timeline-lane-label-{index}"]
        if y + h > body.inner_clip_rect.max.y + 1:
            assert body.scroll_max.y > 0
            imgui.internal.set_scroll_y(body, body.scroll_max.y)
            viewer.frame()
            x, y, w, h = shell.ui.hits[f"timeline-lane-label-{index}"]
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
    text_inset = viewer.app.workspace.ui.component_style("file_receipt").get("gap", 8)
    assert caption[0] - receipt[0] == pytest.approx(
        (DESIGN.file_receipt_padding + text_inset) * viewer.window.style_scale, abs=1
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
    assert shell.ui.vector_layout(296 * shell.s, format="%.12f").columns == 1
    layout = shell.ui.vector_layout(296 * shell.s, format="%.6g")
    minimum = shell.ui.numeric_minimum(format="%.6g")
    assert all(field[2] >= minimum for field in layout.fields)
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
        assert value[0] <= unit[0] < unit[0] + unit[2] <= value[0] + value[2]
        assert value[2] == pytest.approx(DESIGN.rail_entry_width * scale, abs=1)
        assert rail[2] >= DESIGN.rail_minimum_track_width * scale
        if previous is not None and name in ("pitch", "distance"):
            heading = hits["camera-label-" + tr(name)]
            assert heading[1] - previous[1] - previous[3] == pytest.approx(
                DESIGN.row_gap * scale, abs=1
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
    options_key = "section-" + tr("Import options")
    _scroll_to_hit(viewer, options_key, "Assets")
    viewer.click(options_key)
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
    viewer.click("plot-channels")
    viewer.click("plot-channels/choice/0")
    assert panel.show_angle is not original
    viewer.click("plot-channels")
    viewer.click("plot-channels/choice/0")
    assert panel.show_angle is original
    viewer.click("plot-source-1")
    assert panel.source == "sensor"
    viewer.click("plot-source-0")
    assert panel.source == "joint"


def test_control_layers_sensor_output_and_statistics_share_native_geometry(workspace):
    from mojive.tools.ui_runtime import _item_center
    from mojive.ui.layers import LAYER_SECTIONS
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
    for _title, controls in LAYER_SECTIONS:
        for (first, _), (second, _) in pairwise(controls):
            a, b = shell.ui.hits["layer-" + first], shell.ui.hits["layer-" + second]
            assert b[1] - a[1] - a[3] == pytest.approx(DESIGN.row_gap * shell.s, abs=1)
    for (_, previous), (_, following) in pairwise(LAYER_SECTIONS):
        a = shell.ui.hits["layer-" + previous[-1][0]]
        b = shell.ui.hits["layer-" + following[0][0]]
        assert b[1] - a[1] - a[3] > DESIGN.row_gap * shell.s
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
    stack = imgui.get_current_context().open_popup_stack
    popup = stack[len(stack) - 1].window
    viewport = imgui.get_main_viewport()
    assert popup.pos.y + popup.size.y <= viewport.work_pos.y + viewport.work_size.y
    for action in ("cancel", "done"):
        x, y, w, h = shell.ui.hits["##collision_color-" + action]
        clip = popup.inner_clip_rect
        assert clip.min.x <= x < x + w <= clip.max.x + 1
        assert clip.min.y <= y < y + h <= clip.max.y + 1
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
    viewer.click("##collision_color-original")
    assert viewer.backend.get_geometry_style().collision_color == pytest.approx(original_color)
    assert "##collision_color-picker" in shell.ui.hits
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
        assert value[0] + value[2] == pytest.approx(rail[0] + rail[2], abs=1)
        assert value[0] <= unit[0] < unit[0] + unit[2] <= value[0] + value[2]
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
