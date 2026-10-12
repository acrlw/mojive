"""Native ball/free Pose fields with real HiDPI window input and history."""

import numpy as np
import pytest
from imgui_bundle import imgui
from tests.gpu.test_ui_design_inspector import _visible_hit
from tests.gpu.test_ui_design_runtime import runtime as runtime

from mojive import commands as cmd
from mojive import math3d

pytestmark = pytest.mark.gpu


def _pose_hit(study, key):
    parent = imgui.internal.find_window_by_name("Pose")
    for _ in range(20):
        child = next(
            window
            for window in imgui.get_current_context().windows
            if window.parent_window is not None
            and window.parent_window.id_ == parent.id_
            and "Panel content" in window.name
        )
        x, y, width, height = study.ui.hits[key]
        clip = child.inner_clip_rect
        if clip.min.y <= y and y + height <= clip.max.y:
            assert clip.min.x <= x < x + width <= clip.max.x + 1
            return x, y, width, height
        study.event("add_mouse_pos_event", clip.max.x - 4, (clip.min.y + clip.max.y) / 2)
        study.event("add_mouse_wheel_event", 0, -1 if y + height > clip.max.y else 1)
        study.frame(3)
    raise AssertionError(f"Pose field {key} could not be scrolled into view")


def _field(study, kind, group, axis, panel="Pose"):
    session = study.preview.session
    joint = next(joint for joint in session.joints if joint.type == kind)
    node = next(node for node in session.nodes if node.joint_index == joint.joint_id)
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    study.activate(panel)
    if panel == "Pose":
        study.frame(2)
        study.navigation.joints._search = joint.name
        study.navigation.pose_search_visible = True
    else:
        study.inspector_tab = "Joint"
    study.frame(4)
    key = f"##pose-joint-{joint.joint_id}-{group}{axis}"
    x, y, w, h = _pose_hit(study, key) if panel == "Pose" else _visible_hit(study, key)
    assert h == pytest.approx(28 * study.s, abs=1)
    return joint, (x + w / 2, y + h / 2)


def _text(study, point, value):
    study.event("add_mouse_pos_event", *point)
    study.event("add_key_event", imgui.Key.mod_ctrl, True)
    study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_button_event", 0, False)
    study.event("add_key_event", imgui.Key.mod_ctrl, False)
    assert imgui.get_io().want_text_input
    study.event("add_key_event", imgui.Key.mod_ctrl, True)
    study.event("add_key_event", imgui.Key.a, True)
    study.event("add_key_event", imgui.Key.a, False)
    study.event("add_key_event", imgui.Key.mod_ctrl, False)
    study.event("add_input_characters_utf8", str(value))
    study.event("add_key_event", imgui.Key.enter, True)
    study.event("add_key_event", imgui.Key.enter, False)
    study.frame(3)


@pytest.mark.parametrize(
    "kind,group,axis,value",
    (("ball", "rotation", "Y", 37), ("free", "rotation", "Z", 24), ("free", "position", "X", 0.42)),
)
@pytest.mark.parametrize("panel", ("Pose", "Inspector"))
def test_ball_and_free_fields_write_real_pose_with_one_undo(
    runtime, kind, group, axis, value, panel
):
    study, session = runtime, runtime.preview.session
    joint, point = _field(study, kind, group, axis, panel)
    before = session.frame.qpos.copy()
    offset = joint.qpos_adr + (3 if kind == "free" else 0)
    expected = np.degrees(
        math3d.mat3_to_euler_xyz(math3d.quat_to_mat3(before[offset : offset + 4]))
    )
    expected["XYZ".index(axis)] = value
    _text(study, point, value)
    after = session.frame.qpos.copy()
    assert not session.editing
    indices = (
        np.arange(joint.qpos_adr, joint.qpos_adr + 3)
        if group == "position"
        else np.arange(offset, offset + 4)
    )
    untouched = np.setdiff1d(np.arange(len(before)), indices)
    np.testing.assert_array_equal(after[untouched], before[untouched])
    if group == "position":
        assert after[joint.qpos_adr + "XYZ".index(axis)] == pytest.approx(value)
    else:
        assert np.linalg.norm(after[indices]) == pytest.approx(1, abs=1e-8)
        np.testing.assert_allclose(
            math3d.quat_to_mat3(after[indices]),
            math3d.euler_xyz_to_mat3(np.radians(expected)),
            atol=2e-6,
        )
    study.capture(f"pose-{kind}-{group}-{panel.lower()}-text")
    assert session.submit(cmd.Undo()).ok
    study.frame(3)
    np.testing.assert_allclose(session.frame.qpos, before, atol=1e-8)
    assert not session.can_undo
    assert session.submit(cmd.Redo()).ok
    study.frame(3)
    np.testing.assert_allclose(session.frame.qpos, after, atol=1e-8)


@pytest.mark.parametrize("kind,group", (("ball", "rotation"), ("free", "position")))
def test_ball_and_free_pose_drag_escape_restores_state(runtime, kind, group):
    study, session = runtime, runtime.preview.session
    _joint, point = _field(study, kind, group, "X")
    before = session.frame.qpos.copy()
    study.event("add_mouse_pos_event", *point)
    study.event("add_mouse_button_event", 0, True)
    for offset in (12, 24, 36):
        study.event("add_mouse_pos_event", point[0] + offset * study.s, point[1])
    assert session.editing
    assert not np.array_equal(session.frame.qpos, before)
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    study.event("add_mouse_button_event", 0, False)
    study.frame(3)
    np.testing.assert_allclose(session.frame.qpos, before, atol=1e-8)
    assert not session.editing and not session.can_undo
    study.capture(f"pose-{kind}-{group}-escape")
