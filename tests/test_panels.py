from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge_viewer.adapters.base import (
    ModelAssetInfo,
    ModelComponentField,
    ModelComponentPathItem,
    NodeType,
    SceneNode,
)
from forge_viewer.types import MeshShape
from forge_viewer.ui.localization import Language, Localizer, parse_language
from forge_viewer.ui.messages import OutputBuffer
from forge_viewer.ui.panels import (
    Panel,
    PanelContext,
    PanelSet,
    button_row_layout,
    default_panels,
    horizontal_wheel_target,
    slider_gesture,
    sort_order_glyph,
    sort_order_tooltip,
    validate_panels,
)
from forge_viewer.ui.panels.assets import (
    AssetsPanel,
    filter_assets,
    height_field_preview_color,
    unique_asset_name,
)
from forge_viewer.ui.panels.control import filter_actuators, sort_actuators
from forge_viewer.ui.panels.hierarchy import (
    HierarchyPanel,
    disclosure_triangle,
    hierarchy_open_depth,
    hierarchy_shows_type_column,
)
from forge_viewer.ui.panels.inspector import (
    _compact_transform,
    _format_vector,
    _free_velocity,
    _geometry_dimensions,
    _geometry_size_from_dimensions,
    _matching_path_preset,
    _mix_color,
    _nearest_euler_degrees,
    _path_preset_label,
    _pose_editable,
    _unique_component_name,
    gizmo_refusal_reason,
)
from forge_viewer.ui.panels.joints import filter_joints, page_span, sort_joints
from forge_viewer.ui.panels.keyframes import (
    fitted_timeline_range,
    nearest_take_frame,
    neighboring_keyframe,
    nice_timeline_step,
    timeline_time_to_x,
    timeline_x_to_time,
    unique_keyframe_name,
    zoom_timeline_range,
)
from forge_viewer.ui.panels.output import filter_output_entries
from forge_viewer.ui.panels.plot import PlotPanel
from forge_viewer.ui.panels.settings import settings_category_matches
from forge_viewer.ui.panels.stats import StatsPanel, _scale_ceiling
from forge_viewer.ui.viewport_widgets import capsule_points, playback_size, tool_column_size
from forge_viewer.ui.window import ResizeLatch

EXPECTED_PANELS = {
    "Control",
    "Hierarchy",
    "Assets",
    "Inspector",
    "Joints",
    "Keyframes",
    "Camera",
    "Plot",
    "Stats",
    "Output",
    "Settings",
    "Sensors",
    "Help",
    "Info",
}


def test_hierarchy_disclosure_is_one_rigidly_rotated_antialiased_shape() -> None:
    closed = np.asarray(disclosure_triangle((13.0, 17.0), 5.0, opened=False))
    opened = np.asarray(disclosure_triangle((13.0, 17.0), 5.0, opened=True))
    center = np.array((13.0, 17.0))
    expected = np.column_stack((-(closed - center)[:, 1], (closed - center)[:, 0])) + center

    assert opened == pytest.approx(expected)
    assert np.linalg.norm(opened - np.roll(opened, -1, axis=0), axis=1) == pytest.approx(
        np.linalg.norm(closed - np.roll(closed, -1, axis=0), axis=1)
    )


def test_hierarchy_hides_type_column_before_it_overlaps_scaled_node_names() -> None:
    assert hierarchy_shows_type_column(300.0, 1.0)
    assert not hierarchy_shows_type_column(600.0, 4.0)
    assert hierarchy_shows_type_column(960.0, 4.0)


@pytest.fixture
def panels() -> PanelSet:
    return PanelSet()


def test_registered_panels(panels: PanelSet):

    assert {p.name for p in panels} == EXPECTED_PANELS


def test_settings_search_routes_queries_across_categories() -> None:
    assert settings_category_matches("Interaction", "selection padding")
    assert settings_category_matches("Rendering", "shadow casters")
    assert settings_category_matches("MuJoCo Visuals", "contact force")
    assert not settings_category_matches("General", "contact force")


def test_joint_and_actuator_search_matches_names_case_insensitively() -> None:
    joints = (
        SimpleNamespace(joint_id=0, name="LF_HFE"),
        SimpleNamespace(joint_id=1, name="RH_KFE"),
        SimpleNamespace(joint_id=2, name=""),
    )
    actuators = (
        SimpleNamespace(actuator_id=0, name="LF_HFE"),
        SimpleNamespace(actuator_id=1, name="wheel_motor"),
        SimpleNamespace(actuator_id=2, name=""),
    )

    assert filter_joints(joints, "hfe") == (joints[0],)
    assert filter_joints(joints, "JOINT2") == (joints[2],)
    assert filter_actuators(actuators, "MOTOR") == (actuators[1],)
    assert filter_actuators(actuators, "act2") == (actuators[2],)
    assert filter_joints(joints, "") == joints
    assert filter_actuators(actuators, "") == actuators


def test_joint_and_actuator_lists_toggle_between_state_and_name_order() -> None:
    joints = (
        SimpleNamespace(joint_id=8, name="z_joint", qpos_adr=0, qvel_adr=1),
        SimpleNamespace(joint_id=2, name="A_joint", qpos_adr=7, qvel_adr=6),
        SimpleNamespace(joint_id=4, name="m_joint", qpos_adr=3, qvel_adr=2),
    )
    actuators = (
        SimpleNamespace(actuator_id=8, name="z_motor", ctrl_address=0, act_address=3),
        SimpleNamespace(actuator_id=2, name="A_motor", ctrl_address=5, act_address=1),
        SimpleNamespace(actuator_id=4, name="m_motor", ctrl_address=2, act_address=2),
    )

    assert sort_joints(joints, by_name=False) == (joints[0], joints[2], joints[1])
    assert sort_joints(joints, by_name=True) == (joints[1], joints[2], joints[0])
    assert sort_actuators(actuators, by_name=False) == (actuators[0], actuators[2], actuators[1])
    assert sort_actuators(actuators, by_name=True) == (actuators[1], actuators[2], actuators[0])


def test_sort_button_uses_a_bounded_list_and_arrow_glyph() -> None:
    segments = sort_order_glyph((10.0, 20.0, 42.0, 52.0))

    assert len(segments) == 6
    assert all(
        10.0 <= coordinate <= 42.0
        for segment in segments
        for point in segment
        for coordinate in (point[0],)
    )
    assert all(
        20.0 <= coordinate <= 52.0
        for segment in segments
        for point in segment
        for coordinate in (point[1],)
    )


def test_sort_tooltip_names_only_the_active_order() -> None:
    assert sort_order_tooltip(False, "qpos / qvel") == "Order: qpos / qvel"
    assert sort_order_tooltip(True, "qpos / qvel") == "Order: Name"


def test_horizontal_wheel_maps_vertical_input_and_clamps_to_content() -> None:
    assert horizontal_wheel_target(20.0, 200.0, -1.0, step=48.0) == 68.0
    assert horizontal_wheel_target(20.0, 200.0, 1.0, step=48.0) == 0.0
    assert horizontal_wheel_target(190.0, 200.0, -1.0, step=48.0) == 200.0


def test_plot_sensor_deep_link_requests_sensor_frames() -> None:
    panel = PlotPanel()

    panel.focus_sensor(3, 7)

    assert panel.source == "sensor"
    assert (panel.sensor_index, panel.sensor_component) == (3, 7)
    needs = panel.frame_needs()
    assert needs.sensors
    assert not needs.qpos and not needs.qvel and not needs.contacts


def test_asset_panel_filters_cached_inventory_and_generates_unique_names():
    assets = (
        ModelAssetInfo(0, "hfield", "terrain", 0, "/tmp/terrain.png"),
        ModelAssetInfo(0, "hfield", "terrain2", 1, "/tmp/terrain2.png"),
        ModelAssetInfo(0, "mesh", "robot", 0, "/tmp/robot.obj"),
    )

    assert filter_assets(assets, "hfield", "terrain") == assets[:2]
    assert filter_assets(assets, "all", "robot.obj") == (assets[2],)
    assert unique_asset_name("terrain", assets, "hfield") == "terrain3"
    assert unique_asset_name("terrain", assets, "mesh") == "terrain"


def test_asset_panel_focus_opens_and_selects_the_requested_asset() -> None:
    panel = AssetsPanel()

    panel.focus(4, "mesh", "robot_shell")

    assert panel.open
    assert panel._model_id == 4
    assert panel._asset_type == "mesh"
    assert panel._selected == (4, "mesh", "robot_shell")


def test_height_field_preview_color_clamps_and_spans_the_palette():
    low = height_field_preview_color(-1.0)
    middle = height_field_preview_color(0.5)
    high = height_field_preview_color(2.0)

    assert low == height_field_preview_color(0.0)
    assert high == height_field_preview_color(1.0)
    assert low != middle != high
    assert low[3] == middle[3] == high[3] == 1.0


def test_panel_diagnostics_are_published_to_the_persistent_session_status():
    class Messages:
        def __init__(self):
            self.last_message = ""

        def report_message(self, message: str, **_options) -> None:
            self.last_message = message

    session = Messages()
    ctx = PanelContext(session=session, backend=SimpleNamespace())
    ctx.report("snapshot state is incompatible")

    assert ctx.status == "snapshot state is incompatible"
    assert session.last_message == "snapshot state is incompatible"


def test_output_buffer_separates_history_from_transient_status(monkeypatch):
    import forge_viewer.ui.messages as messages

    now = 100.0
    monkeypatch.setattr(messages.time, "monotonic", lambda: now)
    output = OutputBuffer(capacity=2)
    output.write("runtime detail", level="debug", timestamp="10:00:00")
    status = output.publish("saved scene", level="success", duration=5.0)

    assert status is not None
    assert [entry.text for entry in output.entries()] == ["runtime detail", "saved scene"]
    assert output.active_status(now + 4.9) == status
    assert output.active_status(now + 5.0) is None
    assert "[DEBUG] runtime detail" in output.copy_text()


def test_output_filter_combines_text_component_and_severity():
    output = OutputBuffer()
    output.write("[forge/window] cache detail", level="debug", timestamp="10:00:00")
    loading = output.write("[forge/ui] Loading robot model", level="info", timestamp="10:00:01")
    warning = output.write("[forge/window] Font fallback", level="warning", timestamp="10:00:02")
    error = output.write("[forge/ui] Model load failed", level="error", timestamp="10:00:03")
    entries = output.entries()

    assert filter_output_entries(entries, "forge/ui loading", 0) == (loading,)
    assert filter_output_entries(entries, "", 30) == (warning, error)
    assert filter_output_entries(entries, "forge/ui", 40) == (error,)
    assert "Font fallback" in output.copy_text(filter_output_entries(entries, "", 30))
    assert "cache detail" not in output.copy_text(filter_output_entries(entries, "", 30))


def test_default_workspace_panels_do_not_expose_accidental_close_buttons(panels: PanelSet):
    fixed = {"Control", "Hierarchy", "Inspector", "Joints", "Camera", "Output"}
    assert all(not panels.get(name).closable for name in fixed)


def test_large_editor_lists_are_bounded_and_large_hierarchies_start_closed():
    assert page_span(4096, 99, 128) == (31, 32, 3968, 4096)
    assert page_span(0, -1, 128) == (0, 1, 0, 0)
    assert hierarchy_open_depth(999) == 2
    assert hierarchy_open_depth(1000) == 1
    assert hierarchy_open_depth(2000) == 0


def test_hierarchy_batch_delete_collapses_selected_descendants_and_skips_scene_entities():
    panel = HierarchyPanel()
    nodes = (
        SceneNode(1, "body", NodeType.LINK, model_id=0),
        SceneNode(2, "child geom", NodeType.GEOM, parent=1, model_id=0),
        SceneNode(3, "sibling geom", NodeType.GEOM, model_id=0),
        SceneNode(4, "forge object", NodeType.LINK, model_id=-1),
    )
    panel._by_id = {node.node_id: node for node in nodes}
    panel._batch_selected = {1, 2, 3, 4}

    assert panel._batch_removable_roots() == (1, 3)


def test_settings_is_a_dockable_panel(panels: PanelSet):
    settings = panels.get("Settings")
    assert settings is not None
    assert not settings.modal
    assert not settings.standalone
    assert not settings.default_open
    assert settings.dock_with == "Camera"


def test_viewport_chrome_uses_exact_capsule_geometry_and_spacing():
    assert playback_size(1.0) == pytest.approx((116.0, 44.0))
    assert tool_column_size(1.0) == pytest.approx((44.0, 174.0))

    horizontal = np.asarray(capsule_points(10.0, 20.0, 116.0, 44.0))
    vertical = np.asarray(capsule_points(10.0, 20.0, 44.0, 174.0))
    assert horizontal.min(axis=0) == pytest.approx((10.0, 20.0))
    assert horizontal.max(axis=0) == pytest.approx((126.0, 64.0))
    assert vertical.min(axis=0) == pytest.approx((10.0, 20.0))
    assert vertical.max(axis=0) == pytest.approx((54.0, 194.0))


def test_language_preference_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("FORGE_VIEWER_SETTINGS", str(path))
    monkeypatch.delenv("FORGE_VIEWER_LANGUAGE", raising=False)

    localizer = Localizer.load()
    assert localizer.language is Language.ENGLISH
    localizer.set_preferences(
        {
            "remember_precise_input_choices": True,
            "precise_gizmo_angle_unit": "radians",
        }
    )
    localizer.set_language(Language.SIMPLIFIED_CHINESE)
    assert localizer.text("Settings") == "设置"

    restored = Localizer.load()
    assert restored.language is Language.SIMPLIFIED_CHINESE
    assert restored.text("Return to Editor Camera") == "返回编辑器相机"
    assert restored.preference("remember_precise_input_choices") is True
    assert restored.preference("precise_gizmo_angle_unit") == "radians"


def test_viewer_restores_precise_input_preferences(tmp_path, monkeypatch):
    from forge_viewer.ui.app import ViewerApp

    path = tmp_path / "settings.json"
    monkeypatch.setenv("FORGE_VIEWER_SETTINGS", str(path))
    Localizer.load().set_preferences(
        {
            "remember_precise_input_choices": False,
            "precise_gizmo_absolute": True,
            "precise_gizmo_angle_unit": "radians",
            "view_selection_padding": 1.8,
        }
    )

    app = ViewerApp(SimpleNamespace(), SimpleNamespace())

    assert not app.gizmo.remember_precise_input_choices
    assert app._precise_gizmo_preferred_absolute
    assert app._precise_gizmo_angle_unit == "radians"
    assert app.view_cube.selection_padding == pytest.approx(1.8)

    app.set_view_selection_padding(2.25)
    assert Localizer.load().preference("view_selection_padding") == pytest.approx(2.25)


def test_viewer_restores_and_persists_viewport_input_bindings(tmp_path, monkeypatch):
    from forge_viewer.ui.app import ViewerApp
    from forge_viewer.ui.input_bindings import InputAction

    path = tmp_path / "settings.json"
    monkeypatch.setenv("FORGE_VIEWER_SETTINGS", str(path))
    Localizer.load().set_preferences(
        {
            "input_bindings": {
                InputAction.FRAME_SCENE.value: "g",
            }
        }
    )

    app = ViewerApp(SimpleNamespace(), SimpleNamespace())

    assert app.input_bindings.key_id(InputAction.FRAME_SCENE) == "g"
    assert app.input_bindings.key_id(InputAction.GIZMO_TRANSLATE) == "f"
    app.set_input_binding(InputAction.SNAP, "x")

    saved = Localizer.load().preference("input_bindings")
    assert isinstance(saved, dict)
    assert saved[InputAction.SNAP.value] == "x"
    assert saved[InputAction.AXIS_X.value] == "shift"


def test_viewer_restores_and_persists_status_metric(tmp_path, monkeypatch):
    from forge_viewer.ui.app import ViewerApp

    path = tmp_path / "settings.json"
    monkeypatch.setenv("FORGE_VIEWER_SETTINGS", str(path))
    Localizer.load().set_preferences({"status_metric": "steps"})

    app = ViewerApp(SimpleNamespace(), SimpleNamespace())
    assert app._status_metric_mode == "steps"

    app._toggle_status_metric()
    assert app._status_metric_mode == "time"
    assert Localizer.load().preference("status_metric") == "time"

    app.set_language("zh_CN")
    assert app._viewport_labels.snap == "吸附"
    assert app._viewport_labels.type_value == "输入数值"


@pytest.mark.parametrize("value", ["zh_CN", "zh-CN", "zh_CN.UTF-8", "zh_CN:zh"])
def test_simplified_chinese_locale_variants(value):
    assert parse_language(value) is Language.SIMPLIFIED_CHINESE


def test_linux_language_environment_is_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_VIEWER_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("FORGE_VIEWER_LANGUAGE", "zh_CN.UTF-8")
    assert Localizer.load().language is Language.SIMPLIFIED_CHINESE


def test_every_panel_declares_a_shortcut(panels: PanelSet):

    for p in panels:
        assert isinstance(p.shortcut, str)
        assert isinstance(p.aliases, tuple)


def test_occasional_authoring_panels_start_closed_and_use_the_window_menu(panels: PanelSet):
    for name in ("Assets", "Keyframes", "Stats"):
        panel = panels.get(name)
        assert panel is not None and not panel.default_open


def test_shortcuts_are_unique(panels: PanelSet):

    keys: list[str] = []
    for p in panels:
        keys.extend(k for k in (p.shortcut, *p.aliases) if k)
    assert len(keys) == len(set(keys))


def test_validate_allows_a_keyless_closed_panel_reopened_from_the_window_menu():

    class Orphan(Panel):
        name = "Orphan"
        default_open = False
        shortcut = ""

    problems = validate_panels([*default_panels(), Orphan()])
    assert not problems
    assert PanelSet([*default_panels(), Orphan()]).get("Orphan") is not None


def test_validate_catches_a_duplicate_key():

    class Thief(Panel):
        name = "Thief"
        default_open = False
        shortcut = "F1"

    problems = validate_panels([*default_panels(), Thief()])
    assert any("F1" in p for p in problems), problems


def test_help_lists_every_panel_key(panels: PanelSet):

    table = panels.shortcut_table()
    assert {name for _key, name, _open in table} == EXPECTED_PANELS


def test_aggregate_needs_follow_the_plot_panel(panels: PanelSet):

    plot = panels.get("Plot")
    assert plot is not None and not plot.default_open

    plot.open = False
    assert panels.frame_needs().qvel is False

    plot.open = True
    assert panels.frame_needs().qvel is True


def test_needs_follow_the_series_toggle_not_just_the_panel(panels: PanelSet):

    plot = panels.get("Plot")
    plot.open = True
    plot.show_velocity = False
    plot.show_contact = False
    needs = panels.frame_needs()
    assert needs.qvel is False
    assert needs.contacts is False
    assert needs.qpos is True


def test_inspector_velocity_section_drives_qvel(panels: PanelSet):

    inspector = panels.get("Inspector")
    assert inspector.default_open is True
    assert inspector.show_transform is True
    assert inspector.show_velocity is False
    assert inspector.frame_needs().qvel is False

    inspector._transform_velocity = True
    assert inspector.frame_needs().qvel is True
    inspector.show_transform = False
    assert inspector.frame_needs().qvel is False
    inspector.show_velocity = True
    assert inspector.frame_needs().qvel is True


def test_transform_editability_matches_the_set_pose_contract():
    assert _pose_editable(write_pose=True, paused=True, posable=True)
    assert not _pose_editable(write_pose=False, paused=True, posable=True)
    assert not _pose_editable(write_pose=True, paused=False, posable=True)
    assert not _pose_editable(write_pose=True, paused=True, posable=False)


@pytest.mark.parametrize(
    ("shape", "size", "label", "dimensions"),
    (
        (MeshShape.PLANE, (2.0, 3.0, 1.0), "width / length", (4.0, 6.0)),
        (MeshShape.BOX, (1.0, 2.0, 3.0), "width / depth / height", (2.0, 4.0, 6.0)),
        (MeshShape.SPHERE, (0.5, 0.5, 0.5), "diameter", (1.0,)),
        (MeshShape.SPHERE, (0.5, 1.0, 1.5), "width / depth / height", (1.0, 2.0, 3.0)),
        (MeshShape.CYLINDER, (0.5, 0.5, 2.0), "diameter / height", (1.0, 4.0)),
        (
            MeshShape.CAPSULE_SHAFT,
            (0.5, 0.5, 2.0),
            "diameter / shaft length",
            (1.0, 4.0),
        ),
    ),
)
def test_geometry_dimension_editor_uses_full_user_facing_dimensions(shape, size, label, dimensions):
    editor = _geometry_dimensions(shape, size)
    assert editor is not None
    assert editor[0] == label
    assert editor[1] == pytest.approx(dimensions)
    assert _geometry_size_from_dimensions(shape, size, editor[1]) == pytest.approx(size)


def test_free_body_velocity_is_split_into_linear_and_angular_xyz():
    from types import SimpleNamespace

    joint = SimpleNamespace(body=3, type="free", dof=6, qvel_adr=2)
    linear, angular = _free_velocity([8, 9, 1, 2, 3, 4, 5, 6], [joint], 3)
    assert linear.tolist() == [1, 2, 3]
    assert angular.tolist() == [4, 5, 6]
    assert _free_velocity([1, 2, 3], [joint], 3) is None


def test_transform_clipboard_vector_is_plain_xyz():
    assert _format_vector((1.25, -2.0, 0.0)) == "1.25, -2, 0"


def test_keyframe_names_advance_without_exposing_raw_state_arrays():
    assert unique_keyframe_name(set()) == "key1"
    assert unique_keyframe_name({"key1", "key2"}) == "key3"


def test_keyframe_timeline_fits_isolated_and_distributed_snapshots():
    assert fitted_timeline_range((), 4.0) == pytest.approx((3.5, 4.5))
    assert fitted_timeline_range((10.0,)) == pytest.approx((9.5, 10.5))
    assert fitted_timeline_range((-2.0, 8.0)) == pytest.approx((-2.8, 8.8))


def test_keyframe_timeline_mapping_and_zoom_preserve_the_anchor():
    x = timeline_time_to_x(2.5, 0.0, 10.0, 100.0, 500.0)
    assert x == pytest.approx(200.0)
    assert timeline_x_to_time(x, 0.0, 10.0, 100.0, 500.0) == pytest.approx(2.5)

    start, end = zoom_timeline_range(0.0, 10.0, 2.5, 1.0)
    assert end - start < 10.0
    assert (2.5 - start) / (end - start) == pytest.approx(0.25)
    assert nice_timeline_step(10.0, 900.0) == pytest.approx(1.0)


def test_keyframe_timeline_navigation_uses_selection_then_playhead():
    markers = ((30, 3.0), (10, 1.0), (20, 2.0))
    assert neighboring_keyframe(markers, 20, 0.0, -1) == 10
    assert neighboring_keyframe(markers, 20, 0.0, 1) == 30
    assert neighboring_keyframe(markers, -1, 1.5, -1) == 10
    assert neighboring_keyframe(markers, -1, 1.5, 1) == 20
    assert nearest_take_frame((0.0, 0.2, 0.5), 0.31) == 1
    assert nearest_take_frame((0.0, 0.2, 0.5), 0.4) == 2
    assert nearest_take_frame((), 0.0) == -1


def test_transform_keeps_axis_groups_inline_until_the_panel_is_truly_narrow():
    assert _compact_transform(190.0, 1.0)
    assert not _compact_transform(240.0, 1.0)
    assert _compact_transform(400.0, 2.0)


def test_button_rows_wrap_without_clipping_items():
    widths = (156.0, 120.0, 120.0, 140.0)
    assert button_row_layout(widths, 400.0, 14.0) == (False, True, False, True)
    assert button_row_layout(widths, 700.0, 14.0) == (False, True, True, True)


def test_model_component_names_are_stable_and_unique():
    assert _unique_component_name("sensor", set()) == "sensor"
    assert _unique_component_name("sensor", {"sensor", "sensor2", "sensor4"}) == "sensor3"


def test_tuple_path_presets_disambiguate_and_follow_object_type():
    presets = (
        ModelComponentPathItem(
            "element",
            (
                ModelComponentField("objtype", "body", ("body", "geom")),
                ModelComponentField("objname", "world", ("world",)),
            ),
        ),
        ModelComponentPathItem(
            "element",
            (
                ModelComponentField("objtype", "geom", ("body", "geom")),
                ModelComponentField("objname", "floor", ("floor", "box")),
            ),
        ),
    )
    assert [_path_preset_label(preset) for preset in presets] == [
        "element · body",
        "element · geom",
    ]
    assert _matching_path_preset(presets, "element", [["objtype", "geom"]]) == presets[1]


def test_transform_axis_badges_mix_semantic_color_into_panel_surfaces():
    axis = (0.8, 0.4, 0.3, 1.0)
    background = (0.1, 0.1, 0.1, 1.0)
    muted = _mix_color(background, axis, 0.56)
    active = _mix_color(background, axis, 0.88)

    assert muted != axis
    assert muted != active
    assert all(m < a < source for m, a, source in zip(muted[:3], active[:3], axis[:3], strict=True))


@pytest.mark.parametrize("y", (91.0, -91.0, 271.0, 449.0))
def test_inspector_euler_y_stays_on_the_dragged_branch_past_gimbal_lock(y):
    from forge_viewer import math3d

    expected = np.array((12.0, y, -25.0))
    matrix = math3d.euler_xyz_to_mat3(np.radians(expected))
    reference = expected.copy()
    reference[1] -= 0.5
    actual = _nearest_euler_degrees(matrix, reference)

    assert actual == pytest.approx(expected, abs=1e-4)


def test_stats_scale_is_quantized_and_holds_after_a_spike():
    panel = StatsPanel()
    panel._update_scale(38.0)
    assert panel._scale_ms == 50.0
    assert _scale_ceiling(26.0) == 33.4

    for _ in range(120):
        panel._update_scale(16.0)
    assert panel._scale_ms == 50.0


def test_closed_panels_ask_for_nothing(panels: PanelSet):

    for p in panels:
        p.open = False
    needs = panels.frame_needs()
    assert not any(
        (needs.poses, needs.qpos, needs.qvel, needs.contacts, needs.actuator, needs.sensors)
    )


def test_app_frame_needs_preserves_consumer_requests():
    """Backend visualization choices must not erase panel or gizmo requirements."""

    from forge_viewer.adapters.base import FrameNeeds
    from forge_viewer.render.backend import FrameMode, LabelMode
    from forge_viewer.ui.app import ViewerApp

    requested = FrameNeeds(
        poses=False,
        contacts=True,
        tendons=True,
        actuator=True,
        deformables=True,
        diagnostics=True,
        islands=True,
        bvh=True,
    )
    app = ViewerApp.__new__(ViewerApp)
    app.panels = SimpleNamespace(frame_needs=lambda: requested)
    app.gizmo = SimpleNamespace(frame_needs=lambda _session: FrameNeeds.none())
    app.session = SimpleNamespace(source=SimpleNamespace(dynamic_meshes=()))
    app.backend = SimpleNamespace(
        get_flag=lambda _flag: False,
        get_label_mode=lambda: LabelMode.NONE,
        get_frame_mode=lambda: FrameMode.NONE,
    )

    needs = app.frame_needs()

    assert needs.contacts
    assert needs.tendons
    assert needs.actuator
    assert needs.deformables
    assert needs.diagnostics
    assert needs.islands
    assert needs.bvh


def test_gizmo_refusal_texts_are_verbatim():

    assert gizmo_refusal_reason(paused=False, posable=True) == (
        "Physics is running; pause to move things"
    )
    assert gizmo_refusal_reason(paused=True, posable=False) == (
        "This link is joint-driven; use its joint gizmo or the Joints panel"
    )
    assert gizmo_refusal_reason(paused=True, posable=True) is None


def test_gizmo_refusal_prefers_the_running_reason():

    assert gizmo_refusal_reason(paused=False, posable=False) == (
        "Physics is running; pause to move things"
    )


def test_slider_gestures():

    assert slider_gesture(True, right_clicked=True, double_clicked=False, shift=False) == "reset"
    assert slider_gesture(True, right_clicked=True, double_clicked=False, shift=True) == "expand"
    assert slider_gesture(True, right_clicked=False, double_clicked=True, shift=False) == "copy"
    assert slider_gesture(True, right_clicked=False, double_clicked=False, shift=False) is None

    assert slider_gesture(False, right_clicked=True, double_clicked=True, shift=True) is None


WARMUP_FRAMES = 30

SETTLE_SECONDS = 0.6

HUMAN_PAUSE = 0.35

FRAME_DT = 1.0 / 60.0


def _warm_up(latch: ResizeLatch, size=(800, 600), now: float = 0.0) -> None:

    for _ in range(WARMUP_FRAMES):
        latch.update(size, now)


def test_latch_constants_match_the_spec():

    latch = ResizeLatch()
    assert latch.settle_seconds == pytest.approx(SETTLE_SECONDS)
    assert latch.warmup_frames == WARMUP_FRAMES


def test_latch_rebuilds_immediately_during_startup():

    latch = ResizeLatch()
    for i in range(WARMUP_FRAMES):
        size = (100 + i, 100)
        assert latch.update(size, now=0.0) == size
    assert latch.rebuilds == WARMUP_FRAMES

    assert latch.update((900, 700), now=0.0) is None


def test_latch_commits_an_explicit_resize_transition_immediately():

    latch = ResizeLatch()
    _warm_up(latch)
    before = latch.rebuilds

    assert latch.update((1200, 700), now=0.0, immediate=True) == (1200, 700)
    assert latch.rebuilds == before + 1
    assert latch.update((1200, 700), now=0.1) is None


def test_latch_coalesces_changes_within_a_drag():

    latch = ResizeLatch()
    _warm_up(latch)
    before = latch.rebuilds

    t = 0.0
    last = (800, 600)
    while t < 0.3:
        last = (900 + int(t / 0.05) * 10, 700)
        latch.update(last, now=t)
        t += FRAME_DT

    committed = None
    while t < 0.3 + SETTLE_SECONDS * 2:
        out = latch.update(last, now=t)
        if out is not None:
            committed = out
        t += FRAME_DT

    assert latch.rebuilds - before == 1
    assert committed == last


def test_latch_survives_a_human_pause_between_drags():

    latch = ResizeLatch()
    _warm_up(latch)
    before = latch.rebuilds

    t = 0.0
    size = (900, 700)
    for i in range(4):
        size = (900 + i * 10, 700)

        stop = t + HUMAN_PAUSE
        while t < stop:
            latch.update(size, now=t)
            t += FRAME_DT

    assert latch.rebuilds - before == 0

    committed = None
    stop = t + SETTLE_SECONDS * 2
    while t < stop:
        out = latch.update(size, now=t)
        if out is not None:
            committed = out
        t += FRAME_DT
    assert committed == size
    assert latch.rebuilds - before == 1


def test_latch_ignores_a_size_that_comes_back():

    latch = ResizeLatch()
    _warm_up(latch, size=(800, 600))
    before = latch.rebuilds

    latch.update((1200, 600), now=0.0)
    latch.update((800, 600), now=0.1)
    t = 0.1
    while t < 1.2:
        t += 0.05
        latch.update((800, 600), now=t)
    assert latch.rebuilds == before


def test_latch_needs_no_gl():

    from forge_viewer.ui import window as win

    assert win.glfw is None and win.gl is None and win.imgui is None
