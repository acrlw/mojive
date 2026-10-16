"""Native pose rendering keeps offscreen controls out of the frame submission."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, JointInfo, NodeType, SceneNode
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.panels import PanelContext, PanelManager
from mojive.ui.workspace import transactions
from mojive.ui.workspace.navigation import Navigation, node_icon
from mojive.ui.workspace.pose_panel import NativePoseEditor
from mojive.ui.workspace.widgets import UI


def _joint(index, *, kind="hinge", address=None):
    return JointInfo(
        index,
        f"joint{index:04}",
        kind,
        True,
        (-1, 1),
        index if address is None else address,
        index,
        1,
    )


@pytest.fixture(params=(1, 2))
def pose(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    scale = request.param
    io.display_size = (360 * scale, 680 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI.__new__(UI)
    ui.s = scale
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.hits, ui.numeric_edits, ui.translate = {}, {}, str
    session = SimpleNamespace(
        document_id=1,
        structure_generation=1,
        nodes=[],
        joints=(),
        selected_node=None,
        frame=SimpleNamespace(qpos=np.empty(0)),
        source=SimpleNamespace(initial_qpos=None),
        paused=False,
        state_take_playing=False,
        adapter=SimpleNamespace(caps=SimpleNamespace(write_qpos=False, edit_history=False)),
    )
    ctx = PanelContext(session, NullBackend(), style_scale=scale)
    ui.value_context = ctx
    workspace = SimpleNamespace(ui=ui, s=scale)
    editor = NativePoseEditor(workspace)
    state = SimpleNamespace(
        session=session, editor=editor, ui=ui, scroll=None, width=360, workspace=workspace, ctx=ctx
    )

    def populate(kinds):
        coordinates, joints = [], []
        for index, kind in enumerate(kinds):
            joints.append(_joint(index, kind=kind, address=len(coordinates)))
            coordinates.extend(
                {"hinge": (0,), "slide": (0,), "ball": (1, 0, 0, 0), "free": (0, 0, 0, 1, 0, 0, 0)}[
                    kind
                ]
            )
        session.joints = tuple(joints)
        session.frame.qpos = np.asarray(coordinates, np.float64)
        session.source.initial_qpos = session.frame.qpos.copy()
        session.structure_generation += 1

    def frame(count=1):
        for _ in range(count):
            ctx.session = state.session
            if isinstance(state.session, Session):
                state.session.tick(FrameNeeds(qpos=True), wall_dt=0)
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((state.width * scale, 680 * scale))
            imgui.begin("Pose")
            if state.scroll is not None:
                imgui.set_scroll_y(state.scroll)
                state.scroll = None
            pos, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
            state.height = (
                editor.draw(
                    ctx,
                    pos.x,
                    pos.y,
                    size.x,
                    joints=state.session.joints,
                    select_joint=lambda *args, **kwargs: None,
                )
                - pos.y
            )
            state.scroll_max = imgui.get_scroll_max_y()
            imgui.end()
            if hasattr(workspace, "_numeric_edit"):
                transactions.finish_numeric_frame(workspace, ctx)
            imgui.render()
        return {key: rect for key, rect in ui.hits.items() if key.startswith("pose-fields-")}

    state.populate, state.frame = populate, frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


def test_thousand_joint_list_submits_visible_controls_and_scrolls_to_last_joint(pose):
    pose.populate(["hinge"] * 1000)
    top = pose.frame(3)
    assert "pose-fields-0" in top
    assert len(top) <= 12
    height = pose.height
    pose.scroll = 1e9
    bottom = pose.frame(3)
    assert "pose-fields-999" in bottom
    assert "pose-fields-0" not in bottom
    assert len(bottom) <= 12
    assert pose.height == pytest.approx(height)
    assert pose.scroll_max >= height - 680 * pose.ui.s


@pytest.mark.parametrize("width", (180, 360))
def test_mixed_joint_heights_preserve_complete_scroll_extent(pose, width):
    pose.width = width
    pose.populate(["hinge", "ball", "free", "slide"] * 30)
    pose.frame(3)
    height = pose.height
    pose.scroll = 1e9
    rows = pose.frame(3)
    assert "pose-fields-119" in rows
    assert len(rows) < 12
    assert pose.height == pytest.approx(height)
    bounds = sorted(rows.values(), key=lambda rect: rect[1])
    assert all(first[1] + first[3] <= second[1] + 0.001 for first, second in pairwise(bounds))


class JointAdapter(StaticSceneAdapter):
    def __init__(self):
        super().__init__(Scene())
        self.metadata = [_joint(42, address=1), _joint(7, address=0)]

    def joints(self):
        return self.metadata.copy()


def test_joint_lookup_uses_ids_and_replaces_metadata_on_refresh():
    adapter = JointAdapter()
    session = Session(adapter)
    try:
        assert session.joint(42) is session.joints[0]
        assert session.joint(7) is session.joints[1]
        assert session.joint(0) is None
        node = SceneNode(1, "joint", NodeType.JOINT, joint_index=42)
        assert node_icon(node, session) == "joint-hinge"
        adapter.metadata = [_joint(42, kind="slide"), _joint(80)]
        session._refresh_joint_metadata()
        assert session.joint(7) is None
        assert node_icon(node, session) == "joint-slide"
        assert session.joint(80) is adapter.metadata[1]
    finally:
        session.release()


def test_pose_browse_cache_updates_for_filter_sort_and_metadata_refresh():
    adapter = JointAdapter()
    session = Session(adapter)
    manager = PanelManager(builtin_ids=("hierarchy", "joints"))
    workspace = SimpleNamespace(app=SimpleNamespace(session=session, panels=manager))
    navigation = Navigation(workspace)
    ctx = PanelContext(session, NullBackend())
    try:
        first = navigation.pose_joints(ctx)
        assert [joint.joint_id for joint in first] == [7, 42]
        assert navigation.pose_joints(ctx) is first
        navigation.joints._search = "0042"
        assert [joint.joint_id for joint in navigation.pose_joints(ctx)] == [42]
        navigation.joints._search = ""
        adapter.metadata[0].name = "alpha"
        navigation.joints._sort_by_name = True
        assert [joint.joint_id for joint in navigation.pose_joints(ctx)] == [42, 7]
        adapter.metadata = [_joint(80), _joint(42, kind="slide")]
        session._refresh_joint_metadata()
        refreshed = navigation.pose_joints(ctx)
        assert [joint.joint_id for joint in refreshed] == [42, 80]
        assert refreshed[0] is session.joint(42)
        assert refreshed[0].type == "slide"
    finally:
        session.release()


@pytest.mark.physics
def test_active_pose_drag_survives_scroll_and_commits_one_undo(pose, tmp_path):
    from mojive.adapters.mujoco import MuJoCoAdapter

    path = tmp_path / "joints.xml"
    path.write_text(
        "<mujoco><worldbody>"
        + "".join(
            f'<body name="body{i}" pos="{i} 0 1"><joint name="joint{i}" '
            'type="hinge" range="-90 90"/><geom size=".1"/></body>'
            for i in range(30)
        )
        + "</worldbody></mujoco>"
    )
    session = Session(MuJoCoAdapter(path))
    assert session.submit(cmd.Pause()).ok
    pose.session = session
    workspace = pose.workspace
    workspace._numeric_edit = None
    workspace.app = SimpleNamespace(
        session=session,
        model_edits=SimpleNamespace(_checkpoint=None),
        refresh_document=lambda: None,
    )
    workspace._panel_context = pose.ctx
    workspace.begin_numeric_edit = lambda key, **kw: transactions.begin_numeric_edit(
        workspace, key, **kw
    )
    workspace.finish_numeric_edit = lambda key, state: transactions.finish_numeric_edit(
        workspace, key, state
    )
    io = imgui.get_io()
    try:
        pose.frame(3)
        before = session.frame.qpos.copy()
        x, y, width, height = pose.ui.hits["##pose-joint-0-track"]
        io.add_mouse_pos_event(x + width * 0.75, y + height / 2)
        pose.frame()
        io.add_mouse_button_event(0, True)
        pose.frame()
        assert session.editing
        assert workspace._numeric_edit is not None
        pose.scroll = 1e9
        rows = pose.frame(3)
        assert "pose-fields-0" in rows
        assert "pose-fields-29" in rows
        assert len(rows) <= 13
        assert session.editing
        io.add_mouse_button_event(0, False)
        pose.frame(2)
        assert not session.editing
        assert session.frame.qpos[0] != pytest.approx(before[0])
        assert session.submit(cmd.Undo()).ok
        assert session.adapter.data.qpos == pytest.approx(before)
        assert not session.can_undo
    finally:
        session.release()
