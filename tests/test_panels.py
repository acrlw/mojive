from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge_viewer.adapters.base import (
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
    slider_gesture,
    validate_panels,
)
from forge_viewer.ui.panels.camera import camera_snapshot, qpos_snapshot, reproduction_snapshot
from forge_viewer.ui.panels.hierarchy import HierarchyPanel, hierarchy_open_depth
from forge_viewer.ui.panels.inspector import (
    _compact_transform,
    _format_vector,
    _free_velocity,
    _geometry_dimensions,
    _geometry_size_from_dimensions,
    _lift_color,
    _matching_path_preset,
    _nearest_euler_degrees,
    _path_preset_label,
    _pose_editable,
    _unique_component_name,
    gizmo_refusal_reason,
)
from forge_viewer.ui.panels.joints import page_span
from forge_viewer.ui.panels.output import filter_output_entries
from forge_viewer.ui.panels.stats import StatsPanel, _scale_ceiling
from forge_viewer.ui.window import ResizeLatch

EXPECTED_PANELS = {
    "Control",
    "Hierarchy",
    "Inspector",
    "Joints",
    "Camera",
    "Plot",
    "Stats",
    "Output",
    "Settings",
    "Sensors",
    "Help",
    "Info",
}


@pytest.fixture
def panels() -> PanelSet:
    return PanelSet()


def test_registered_panels(panels: PanelSet):

    assert {p.name for p in panels} == EXPECTED_PANELS


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
    fixed = {"Control", "Hierarchy", "Inspector", "Joints", "Camera", "Stats", "Output"}
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


def test_settings_is_a_modal_dialog(panels: PanelSet):
    settings = panels.get("Settings")
    assert settings is not None
    assert settings.modal
    assert not settings.standalone
    assert not settings.default_open


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


def test_closed_by_default_panels_have_a_key(panels: PanelSet):

    for p in panels:
        if not p.default_open:
            assert p.shortcut


def test_shortcuts_are_unique(panels: PanelSet):

    keys: list[str] = []
    for p in panels:
        keys.extend(k for k in (p.shortcut, *p.aliases) if k)
    assert len(keys) == len(set(keys))


def test_validate_catches_a_keyless_closed_panel():

    class Orphan(Panel):
        name = "Orphan"
        default_open = False
        shortcut = ""

    problems = validate_panels([*default_panels(), Orphan()])
    assert any("Orphan" in p for p in problems), problems
    with pytest.raises(ValueError, match="Orphan"):
        PanelSet([*default_panels(), Orphan()])


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
    for key, _name, default_open in table:
        if not default_open:
            assert key


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


def test_qpos_clipboard_contains_the_complete_vector():
    assert qpos_snapshot(np.array([1.0, -2.5, 0.0])) == "qpos=[+1, -2.5, +0]"


def test_camera_snapshot_contains_exact_projection_values():
    from forge_viewer.types import CameraView

    text = camera_snapshot(CameraView(near=0.0125, far=320.0), (12.0, 24.0, 800.0, 600.0))
    assert "near=0.0125" in text
    assert "far=320" in text
    assert "viewport=(+12, +24, +800, +600)" in text

    intrinsic = CameraView(
        focal_length=np.array([0.05, 0.04], np.float32),
        sensor_size=np.array([0.036, 0.024], np.float32),
        principal_offset=np.array([0.003, -0.002], np.float32),
    )
    text = camera_snapshot(intrinsic, (0.0, 0.0, 800.0, 600.0))
    assert "focal_length=(+0.050" in text
    assert "sensor_size=(+0.035" in text
    assert "principal_offset=(+0.003" in text and ", -0.002" in text


def test_reproduction_snapshot_combines_scene_qpos_and_camera():
    from types import SimpleNamespace

    from forge_viewer.types import CameraView

    session = SimpleNamespace(
        asset_path=Path("/tmp/model.xml"),
        adapter=SimpleNamespace(caps=SimpleNamespace(name="mujoco")),
        frame=SimpleNamespace(time=1.25, step=625, qpos=np.array([0.5, -0.25])),
    )
    ctx = SimpleNamespace(
        session=session,
        backend=SimpleNamespace(caps=SimpleNamespace(name="forge")),
        viewport_rect=(0.0, 0.0, 800.0, 600.0),
    )

    text = reproduction_snapshot(ctx, CameraView())

    assert "asset=/tmp/model.xml" in text
    assert "physics_backend=mujoco" in text
    assert "render_backend=forge" in text
    assert "time=1.25" in text
    assert "step=625" in text
    assert "qpos=[+0.5, -0.25]" in text
    assert "forge-viewer camera" in text


def test_transform_switches_to_stacked_rows_before_columns_overlap():
    assert _compact_transform(360.0, 1.0)
    assert not _compact_transform(520.0, 1.0)
    assert _compact_transform(700.0, 2.0)


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


def test_transform_axis_buttons_have_distinct_hover_and_active_colors():
    color = (0.4, 0.6, 0.8, 1.0)
    hovered = _lift_color(color, 0.20)
    active = _lift_color(color, 0.32)
    assert color != hovered != active
    assert all(a > h > c for c, h, a in zip(color[:3], hovered[:3], active[:3], strict=True))


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
