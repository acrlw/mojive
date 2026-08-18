from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge_viewer.ui.panels import (
    Panel,
    PanelSet,
    default_panels,
    slider_gesture,
    validate_panels,
)
from forge_viewer.ui.panels.camera import camera_snapshot, qpos_snapshot, reproduction_snapshot
from forge_viewer.ui.panels.inspector import (
    _compact_transform,
    _format_vector,
    _free_velocity,
    _lift_color,
    _nearest_euler_degrees,
    _pose_editable,
    gizmo_refusal_reason,
)
from forge_viewer.ui.panels.stats import StatsPanel, _scale_ceiling
from forge_viewer.ui.window import ResizeLatch

EXPECTED_PANELS = {
    "Control",
    "Hierarchy",
    "Inspector",
    "Joints",
    "IK",
    "Camera",
    "Plot",
    "Stats",
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


def test_free_body_velocity_is_split_into_linear_and_angular_xyz():
    from types import SimpleNamespace

    joint = SimpleNamespace(body=3, kind="free", dof=6, qvel_adr=2)
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


def test_gizmo_refusal_texts_are_verbatim():

    assert gizmo_refusal_reason(paused=False, posable=True) == (
        "physics is running; pause to move things"
    )
    assert gizmo_refusal_reason(paused=True, posable=False) == (
        "this link is driven by joints; use the Joints panel"
    )
    assert gizmo_refusal_reason(paused=True, posable=True) is None
    assert (
        gizmo_refusal_reason(paused=True, posable=False, inverse_kinematics=True, ik_target=True)
        is None
    )


def test_gizmo_refusal_prefers_the_running_reason():

    assert gizmo_refusal_reason(paused=False, posable=False) == (
        "physics is running; pause to move things"
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
