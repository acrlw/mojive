"""Real generalized-position edits and CPU ImGui gestures in native Pose."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native.docking_study import Study
from examples.ui_design.native.pose_panel import NativePoseEditor, joint_pose_command
from examples.ui_design.native.widgets import UI
from imgui_bundle import imgui

from mojive import Scene, math3d
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.interaction.gizmo import (
    RING_RADIUS,
    SIZE_PT,
    GizmoHandle,
    GizmoMode,
    project,
    world_scale,
)
from mojive.render.backend import NullBackend
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.types import CameraView
from mojive.ui.gizmo import ObjectGizmo, _JointPrecisionProjection
from mojive.ui.panels import PanelContext
from mojive.ui.panels.joints import JointsPanel


@pytest.fixture
def session():
    state = Session(WorkspaceAdapter(MuJoCoAdapter(resolve("joint_types")), Scene()))
    assert state.submit(cmd.Pause()).ok
    state.tick(FrameNeeds(qpos=True, qvel=True), wall_dt=0)
    try:
        yield state
    finally:
        state.release()


@pytest.mark.physics
@pytest.mark.parametrize("kind", ("ball", "free"))
@pytest.mark.parametrize("angles", ((15, -23, 178), (12, 90, 44), (20, -90, 160), (0, 0, 721)))
def test_quaternion_edit_is_normalized_atomic_and_undoable(session, kind, angles):
    joint = next(joint for joint in session.joints if joint.type == kind)
    adapter = session.adapter.primary
    before, velocity = adapter.data.qpos.copy(), adapter.data.qvel.copy()
    command = joint_pose_command(joint, session.frame.qpos, rotation=angles)
    assert session.submit(command).ok
    after = adapter.data.qpos.copy()
    assert len(command.indices) == 4
    assert np.linalg.norm(after[command.indices]) == pytest.approx(1, abs=1e-12)
    assert math3d.quat_to_mat3(after[command.indices]) == pytest.approx(
        math3d.euler_xyz_to_mat3(np.radians(angles)), abs=2e-6
    )
    unchanged = np.setdiff1d(np.arange(len(before)), command.indices)
    assert after[unchanged] == pytest.approx(before[unchanged], abs=0)
    assert adapter.data.qvel == pytest.approx(velocity, abs=0)
    assert session.submit(cmd.Undo()).ok
    assert adapter.data.qpos == pytest.approx(before)
    assert not session.can_undo
    assert session.submit(cmd.Redo()).ok
    assert adapter.data.qpos == pytest.approx(after)


@pytest.mark.physics
def test_free_position_changes_world_pose_and_preserves_other_coordinates(session):
    joint = next(joint for joint in session.joints if joint.type == "free")
    adapter = session.adapter.primary
    before = adapter.data.qpos.copy()
    position = np.array((0.2, -0.4, 1.3))
    command = joint_pose_command(joint, session.frame.qpos, position=position)
    assert session.submit(command).ok
    assert adapter.data.xpos[joint.body] == pytest.approx(position)
    unchanged = np.setdiff1d(np.arange(len(before)), command.indices)
    assert adapter.data.qpos[unchanged] == pytest.approx(before[unchanged], abs=0)


@pytest.mark.physics
def test_ball_rotation_stays_relative_to_rotated_parent_and_body_frame(tmp_path):
    path = tmp_path / "relative.xml"
    path.write_text(
        '<mujoco><worldbody><body euler="20 30 40"><geom size=".1"/>'
        '<body name="ball_body" euler="10 15 25"><joint name="ball" type="ball"/>'
        '<geom size=".1"/></body></body></worldbody></mujoco>'
    )
    state = Session(MuJoCoAdapter(path))
    try:
        assert state.submit(cmd.Pause()).ok
        state.tick(FrameNeeds(qpos=True), wall_dt=0)
        joint = state.joints[0]
        angles = (27, -19, 51)
        assert state.submit(joint_pose_command(joint, state.frame.qpos, rotation=angles)).ok
        adapter = state.adapter
        parent = int(adapter.model.body_parentid[joint.body])
        expected = (
            adapter.data.xmat[parent].reshape(3, 3)
            @ math3d.quat_to_mat3(adapter.model.body_quat[joint.body])
            @ math3d.euler_xyz_to_mat3(np.radians(angles))
        )
        assert adapter.data.xmat[joint.body].reshape(3, 3) == pytest.approx(expected, abs=2e-6)
    finally:
        state.release()


@pytest.mark.physics
def test_invalid_batch_rolls_back_prior_pose_edit_and_leaves_no_history(session):
    joint = next(joint for joint in session.joints if joint.type == "ball")
    adapter = session.adapter.primary
    before = adapter.data.qpos.copy()
    assert session.submit(cmd.BeginEditTransaction("Pose")).ok
    assert session.submit(joint_pose_command(joint, session.frame.qpos, rotation=(10, 20, 30))).ok
    assert not session.submit(cmd.SetQposBatch(np.array((0, 0)), np.ones(2))).ok
    assert not session.submit(cmd.EndEditTransaction()).ok
    assert adapter.data.qpos == pytest.approx(before)
    assert not session.can_undo


@pytest.mark.physics
def test_invalid_pose_input_is_rejected_before_mutation(session):
    joint = next(joint for joint in session.joints if joint.type == "free")
    before = session.adapter.primary.data.qpos.copy()
    with pytest.raises(ValueError, match="finite"):
        joint_pose_command(joint, session.frame.qpos, rotation=(0, np.nan, 0))
    with pytest.raises(ValueError, match="finite"):
        joint_pose_command(joint, session.frame.qpos, position=(np.inf, 0, 0))
    assert session.adapter.primary.data.qpos == pytest.approx(before, abs=0)


@pytest.mark.physics
def test_history_capture_failure_prevents_the_pose_write(session, monkeypatch):
    joint = next(joint for joint in session.joints if joint.type == "ball")
    before = session.adapter.primary.data.qpos.copy()
    monkeypatch.setattr(session.adapter, "capture_edit_state", lambda: None)
    command = joint_pose_command(joint, session.frame.qpos, rotation=(10, 20, 30))
    with pytest.raises(RuntimeError, match="did not provide an edit state"):
        session.submit(command)
    assert session.adapter.primary.data.qpos == pytest.approx(before, abs=0)
    assert not session.can_undo


@pytest.fixture(params=(1.0, 2.0))
def editor(request, session, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (760 * request.param, 680 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s = request.param
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.translate = str
    ui.hits, ui.numeric_edits = {}, {}
    ctx = PanelContext(session, NullBackend(), style_scale=ui.s)
    ui.value_context = ctx
    study = Study.__new__(Study)
    study.ui, study.s, study._panel_context, study._numeric_edit = ui, ui.s, ctx, None
    study.preview = SimpleNamespace(
        session=session,
        model_edits=SimpleNamespace(_checkpoint=None),
        refresh_document=lambda: None,
    )
    panel = NativePoseEditor(study)
    state = SimpleNamespace(
        panel=panel,
        session=session,
        ui=ui,
        ctx=ctx,
        study=study,
        kind="ball",
        width=320,
        show_names=False,
        shared=False,
        shared_panel=JointsPanel(),
        duplicate=False,
    )
    drag_float = imgui.drag_float

    def record_drag(identifier, *args, **kwargs):
        result = drag_float(identifier, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits[identifier] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return result

    monkeypatch.setattr(imgui, "drag_float", record_drag)

    def select(ctx, joint, node, *, focused=False):
        assert ctx.submit(cmd.SelectNode(node.node_id)).ok

    def frame(count=1):
        for _ in range(count):
            session.tick(FrameNeeds(qpos=True, qvel=True), wall_dt=0)
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((state.width * ui.s, 660 * ui.s))
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.begin("Pose")
            pos, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
            joint = next(joint for joint in session.joints if joint.type == state.kind)
            state.joint = joint
            if state.shared:
                state.shared_panel._snapshot(ctx)
                node = next(node for node in session.nodes if node.joint_index == joint.joint_id)
                state.shared_panel._joint_row(ctx, joint, node)
                state.shared_panel.finish_frame(ctx)
            else:
                state.bottom = panel.draw(
                    ctx,
                    pos.x,
                    pos.y,
                    size.x,
                    joints=(joint,),
                    select_joint=select,
                    show_names=state.show_names,
                )
            state.primary_hits = dict(ui.hits)
            imgui.end()
            if state.duplicate:
                imgui.set_next_window_pos((400 * ui.s, 0))
                imgui.set_next_window_size((320 * ui.s, 660 * ui.s))
                imgui.begin("Inspector")
                pos, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
                panel.draw(
                    ctx,
                    pos.x,
                    pos.y,
                    size.x,
                    joints=(joint,),
                    select_joint=select,
                    show_names=False,
                )
                imgui.end()
            study.finish_numeric_frame(ctx)
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def point(key):
        x, y, w, h = (state.primary_hits if state.duplicate else ui.hits)[key]
        return x + w / 2, y + h / 2

    def text(key, value, *, commit=True):
        event("add_mouse_pos_event", *point(key))
        event("add_key_event", imgui.Key.mod_ctrl, True)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        event("add_key_event", imgui.Key.mod_ctrl, False)
        assert io.want_text_input
        event("add_key_event", imgui.Key.mod_ctrl, True)
        event("add_key_event", imgui.Key.a, True)
        event("add_key_event", imgui.Key.a, False)
        event("add_key_event", imgui.Key.mod_ctrl, False)
        event("add_input_characters_utf8", str(value))
        if commit:
            event("add_key_event", imgui.Key.enter, True)
            event("add_key_event", imgui.Key.enter, False)
            frame(2)

    state.frame, state.event, state.point, state.text = frame, event, point, text
    frame(3)
    try:
        yield state
    finally:
        if session.editing:
            session.submit(cmd.CancelEditTransaction())
        imgui.destroy_context(context)


@pytest.mark.physics
@pytest.mark.parametrize(
    "kind,name,axis,value",
    (("ball", "rotation", "Y", 38), ("free", "rotation", "Z", 24), ("free", "position", "X", 0.42)),
)
def test_actual_pose_text_input_writes_one_undo_step(editor, kind, name, axis, value):
    state = editor
    state.kind = kind
    state.frame(3)
    before = state.session.adapter.primary.data.qpos.copy()
    key = f"##pose-joint-{state.joint.joint_id}-{name}{axis}"
    state.text(key, value)
    after = state.session.adapter.primary.data.qpos.copy()
    assert not np.array_equal(before, after)
    if name == "position":
        assert after[state.joint.qpos_adr] == pytest.approx(value)
    else:
        offset = state.joint.qpos_adr + (3 if kind == "free" else 0)
        angles = [0, 0, 0]
        angles["XYZ".index(axis)] = value
        assert math3d.quat_to_mat3(after[offset : offset + 4]) == pytest.approx(
            math3d.euler_xyz_to_mat3(np.radians(angles)), abs=2e-6
        )
    assert not state.session.editing
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.can_undo
    assert state.session.submit(cmd.Redo()).ok
    assert state.session.adapter.primary.data.qpos == pytest.approx(after)


@pytest.mark.physics
@pytest.mark.parametrize("kind,value", (("hinge", 17), ("slide", 0.21)))
def test_existing_scalar_text_input_keeps_ids_and_one_undo_step(editor, kind, value):
    state = editor
    state.kind = kind
    state.frame(3)
    before = state.session.adapter.primary.data.qpos.copy()
    key = "##pose-" + ("hinge" if kind == "hinge" else "slide") + "-value"
    state.text(key, value)
    expected = np.radians(value) if kind == "hinge" else value
    assert state.session.adapter.primary.data.qpos[state.joint.qpos_adr] == pytest.approx(expected)
    assert not state.session.editing
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.can_undo


@pytest.mark.physics
def test_rotation_axis_reset_restores_model_pose_as_one_edit(editor):
    state = editor
    joint = state.joint
    assert state.session.submit(
        joint_pose_command(joint, state.session.frame.qpos, rotation=(35, 0, 0))
    ).ok
    state.frame(2)
    before = state.session.adapter.primary.data.qpos.copy()
    key = f"##pose-joint-{joint.joint_id}-rotationX-reset"
    state.event("add_mouse_pos_event", *state.point(key))
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    state.frame(2)
    after = state.session.adapter.primary.data.qpos
    assert after[joint.qpos_adr : joint.qpos_adr + 4] == pytest.approx((1, 0, 0, 0))
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)


@pytest.mark.physics
def test_pose_drag_cancel_restores_live_preview_and_has_no_undo(editor):
    state = editor
    before = state.session.adapter.primary.data.qpos.copy()
    key = f"##pose-joint-{state.joint.joint_id}-rotationX"
    x, y = state.point(key)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + 38 * state.ui.s, y)
    assert not np.array_equal(before, state.session.adapter.primary.data.qpos)
    state.event("add_key_event", imgui.Key.escape, True)
    state.event("add_key_event", imgui.Key.escape, False)
    state.event("add_mouse_button_event", 0, False)
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.editing
    assert not state.session.can_undo


def fail_first_checkpoint(session, monkeypatch, failure):
    if failure == "refusal":
        method = session._begin_edit
        calls = 0

        def reject_once(label):
            nonlocal calls
            calls += 1
            return (
                cmd.CommandResult.bad("Pose checkpoint unavailable")
                if calls == 1
                else method(label)
            )

        monkeypatch.setattr(session, "_begin_edit", reject_once)
        message = "Pose checkpoint unavailable"
    else:
        method = session.adapter.capture_edit_state
        calls = 0

        def capture_once():
            nonlocal calls
            calls += 1
            return None if calls == 1 else method()

        monkeypatch.setattr(session.adapter, "capture_edit_state", capture_once)
        message = "did not provide an edit state"
    return lambda: calls, message


@pytest.mark.physics
@pytest.mark.parametrize("failure", ("refusal", "capture"))
@pytest.mark.parametrize("shared", (False, True))
def test_failed_pose_checkpoint_blocks_the_entire_gesture_then_allows_retry(
    editor, monkeypatch, failure, shared
):
    state = editor
    if shared:
        state.kind, state.shared = "hinge", True
        state.frame(3)
    calls, message = fail_first_checkpoint(state.session, monkeypatch, failure)
    before = state.session.adapter.primary.data.qpos.copy()
    key = (
        f"##joint-qpos-{state.joint.qpos_adr}-value"
        if shared
        else f"##pose-joint-{state.joint.joint_id}-rotationX"
    )
    x, y = state.point(key)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    for delta in (15, 25, 35):
        state.event("add_mouse_pos_event", x + delta * state.ui.s, y)
    state.event("add_mouse_button_event", 0, False)
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.can_undo
    assert not state.session.editing
    assert calls() == 1
    assert message in state.session.last_message
    state.frame(24)
    x, y = state.point(key)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + 40 * state.ui.s, y)
    state.event("add_mouse_button_event", 0, False)
    assert not np.array_equal(before, state.session.adapter.primary.data.qpos)
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.can_undo


@pytest.mark.physics
def test_duplicate_pose_and_inspector_fields_do_not_finish_another_windows_gesture(editor):
    state = editor
    state.duplicate = True
    state.frame(3)
    before = state.session.adapter.primary.data.qpos.copy()
    key = f"##pose-joint-{state.joint.joint_id}-rotationX"
    x, y = state.point(key)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    for delta in (15, 25, 35):
        state.event("add_mouse_pos_event", x + delta * state.ui.s, y)
        assert state.session.editing
        assert not state.session.can_undo
    state.event("add_mouse_button_event", 0, False)
    assert not state.session.editing
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.can_undo


@pytest.mark.physics
@pytest.mark.parametrize("width", (180, 320, 704))
def test_free_fields_reflow_inside_the_row_without_overlap(editor, width):
    state = editor
    state.kind, state.width, state.show_names = "free", width, True
    state.frame(3)
    prefix = f"##pose-joint-{state.joint.joint_id}-"
    rows = [
        state.ui.hits[prefix + name + axis] for name in ("position", "rotation") for axis in "XYZ"
    ]
    rx, ry, rw, rh = state.ui.hits[f"pose-fields-{state.joint.joint_id}"]
    for x, y, w, h in rows:
        assert rx <= x < x + w <= rx + rw + 0.001
        assert ry <= y < y + h <= ry + rh + 0.001
    for index, a in enumerate(rows):
        for b in rows[index + 1 :]:
            assert (
                min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]) <= 0
                or min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]) <= 0
            )


@pytest.mark.physics
@pytest.mark.parametrize("mode", ("readonly", "running"))
def test_running_and_readonly_states_do_not_write_pose(editor, mode):
    state = editor
    adapter = state.session.adapter.primary
    if mode == "readonly":
        state.session.adapter.caps = replace(state.session.adapter.caps, write_qpos=False)
    else:
        assert state.session.submit(cmd.Play()).ok
    before = adapter.data.qpos.copy()
    key = f"##pose-joint-{state.joint.joint_id}-rotationX"
    state.frame(2)
    x, y = state.point(key)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + 40 * state.ui.s, y)
    state.event("add_mouse_button_event", 0, False)
    assert adapter.data.qpos == pytest.approx(before)
    assert not state.session.can_undo


@pytest.mark.physics
@pytest.mark.parametrize("cancel", (False, True))
def test_production_joint_rail_groups_live_changes_and_cancels(editor, cancel):
    state = editor
    state.kind, state.shared = "hinge", True
    state.frame(3)
    before = state.session.adapter.primary.data.qpos.copy()
    key = f"##joint-qpos-{state.joint.qpos_adr}-value"
    x, y = state.point(key)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_button_event", 0, True)
    for delta in (15, 25, 35):
        state.event("add_mouse_pos_event", x + delta * state.ui.s, y)
    assert not np.array_equal(before, state.session.adapter.primary.data.qpos)
    if cancel:
        state.event("add_key_event", imgui.Key.escape, True)
        state.event("add_key_event", imgui.Key.escape, False)
    state.event("add_mouse_button_event", 0, False)
    assert not state.session.editing
    if cancel:
        assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    else:
        assert state.session.submit(cmd.Undo()).ok
        assert state.session.adapter.primary.data.qpos == pytest.approx(before)
    assert not state.session.can_undo


@pytest.mark.physics
@pytest.mark.parametrize(
    "body_name,handle",
    (
        ("hinge_body", GizmoHandle.ROTATE_Z),
        ("slide_body", GizmoHandle.Z),
        ("ball_body", GizmoHandle.ROTATE_Z),
    ),
)
def test_joint_gizmo_multiple_drag_updates_are_one_undo(session, body_name, handle):
    node = next(node for node in session.nodes if node.name == body_name)
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    session.tick(FrameNeeds(qpos=True, joint_frames=True), wall_dt=0)
    before = session.adapter.primary.data.qpos.copy()
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    position, basis = gizmo._target_pose(session, node, target)
    camera = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=position.copy())
    rect = (0, 0, 800, 600)
    if target.mode is GizmoMode.ROTATE:
        scale = world_scale(camera, position, rect[3], SIZE_PT)

        def cursor(amount):
            point = (
                position
                + (basis[:, 0] * np.cos(amount) + basis[:, 1] * np.sin(amount))
                * scale
                * RING_RADIUS
            )
            return project(camera, (point,), rect)[0, :2]
    else:
        start = project(camera, (position + basis[:, 2] * 0.1,), rect)[0, :2]

        def cursor(amount):
            return start + gizmo._axis_screen * (amount / gizmo._world_per_pt)

    initial = cursor(0) if target.mode is GizmoMode.ROTATE else start
    assert gizmo._begin_handle(session, camera, rect, initial, handle)
    for amount in (0.08, 0.12, 0.18):
        assert gizmo._drag(session, camera, rect, cursor(amount), snap=False)
    after = session.adapter.primary.data.qpos.copy()
    assert not np.array_equal(before, after)
    gizmo._end(commit=True)
    assert not session.editing
    assert session.submit(cmd.Undo()).ok
    assert session.adapter.primary.data.qpos == pytest.approx(before)
    assert not session.can_undo
    assert session.submit(cmd.Redo()).ok
    assert session.adapter.primary.data.qpos == pytest.approx(after)


@pytest.mark.physics
@pytest.mark.parametrize("name", ("hinge_limited", "slide"))
def test_joint_precision_rail_multiple_updates_are_one_undo(session, name):
    joint = next(joint for joint in session.joints if joint.name == name)
    node = next(node for node in session.nodes if node.joint_index == joint.joint_id)
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    session.tick(FrameNeeds(qpos=True, joint_frames=True), wall_dt=0)
    before = session.adapter.primary.data.qpos.copy()
    gizmo = ObjectGizmo()
    gizmo._joint_precision = _JointPrecisionProjection(
        joint_id=joint.joint_id,
        qpos_adr=joint.qpos_adr,
        joint_type=joint.type,
        lower=joint.range[0],
        upper=joint.range[1],
        start=np.array((100.0, 300.0)),
        current=np.array((200.0, 300.0)),
        end=np.array((300.0, 300.0)),
        panel_rect=(90.0, 280.0, 310.0, 320.0),
        hit_rect=(90.0, 280.0, 310.0, 320.0),
        source=np.array((200.0, 350.0)),
    )
    camera = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=np.zeros(3))
    assert gizmo._begin_joint_precision(session, camera, (0, 0, 800, 600), (200, 300))
    for x in (220, 240, 260):
        assert gizmo._drag_joint_precision(session, (x, 300), snap=False)
    assert not np.array_equal(before, session.adapter.primary.data.qpos)
    gizmo._end(commit=True)
    assert session.submit(cmd.Undo()).ok
    assert session.adapter.primary.data.qpos == pytest.approx(before)
    assert not session.can_undo


@pytest.mark.physics
@pytest.mark.parametrize("failure", ("refusal", "capture"))
def test_joint_gizmo_checkpoint_failure_stops_drag_before_any_write(session, monkeypatch, failure):
    node = next(node for node in session.nodes if node.name == "ball_body")
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    session.tick(FrameNeeds(qpos=True, joint_frames=True), wall_dt=0)
    before = session.adapter.primary.data.qpos.copy()
    gizmo = ObjectGizmo()
    target, reason = gizmo._joint_target(session, node)
    assert target is not None, reason
    position, basis = gizmo._target_pose(session, node, target)
    camera = CameraView(eye=np.array((2.0, -4.0, 2.0)), target=position.copy())
    rect = (0, 0, 800, 600)
    scale = world_scale(camera, position, rect[3], SIZE_PT)

    def cursor(angle):
        point = (
            position
            + (basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)) * scale * RING_RADIUS
        )
        return project(camera, (point,), rect)[0, :2]

    calls, message = fail_first_checkpoint(session, monkeypatch, failure)
    assert not gizmo._begin_handle(session, camera, rect, cursor(0), GizmoHandle.ROTATE_Z)
    assert gizmo._active is GizmoHandle.NONE
    assert gizmo._active_joint is None
    assert session.adapter.primary.data.qpos == pytest.approx(before)
    assert not session.editing and not session.can_undo
    assert calls() == 1
    assert message in session.last_message
    assert gizmo._begin_handle(session, camera, rect, cursor(0), GizmoHandle.ROTATE_Z)
    assert gizmo._drag(session, camera, rect, cursor(0.2), snap=False)
    gizmo._end(commit=True)
    assert not np.array_equal(before, session.adapter.primary.data.qpos)
    assert session.submit(cmd.Undo()).ok
    assert session.adapter.primary.data.qpos == pytest.approx(before)
    assert not session.can_undo
