"""Native viewport transport controls seek real replay adapters without simulation."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.joint_replay import JointReplayAdapter
from mojive.session import Session
from mojive.tools.g1_replay import generate
from mojive.ui.workspace.shell import WorkspaceShell
from mojive.ui.workspace.widgets import UI

pytestmark = pytest.mark.physics


@pytest.fixture(params=(1.0, 2.0))
def replay_controls(request, tmp_path, monkeypatch):
    monkeypatch.setattr("mojive.tools.g1_worlds.JOINT_NAMES", ("hinge",))
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint name="floating_base_joint"/>'
        '<geom type="sphere" size="0.1"/><body pos="0 0 0.2">'
        '<joint name="hinge" axis="0 1 0"/><geom type="capsule" size="0.1 0.2"/>'
        "</body></body></worldbody></mujoco>"
    )
    archive = tmp_path / "archive"
    generate(model, archive, 1, 6, 30)
    session = Session(JointReplayAdapter(archive, worlds=1, realtime=False, paused=True))
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)))
    workspace = WorkspaceShell.__new__(WorkspaceShell)
    workspace.ui, workspace.s = ui, ui.s
    workspace.app = SimpleNamespace(session=session)
    workspace.menus = SimpleNamespace(popup=lambda *args: None)
    state = SimpleNamespace(session=session, ui=ui, width=700)

    def shell(x, y, width, height):
        ui.rect(x, y, width * ui.s, height * ui.s, (22, 25, 28, 230))

    def frame(count=1):
        for _ in range(count):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Replay viewport controls")
            workspace.transport(20 * ui.s, 40 * ui.s, state.width * ui.s, shell)
            imgui.end()
            imgui.render()

    def event(name, *args):
        getattr(io, name)(*args)
        frame()

    def click(key):
        x, y, width, height = ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(3)

    state.frame, state.click = frame, click
    try:
        frame(3)
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (300, 700))
def test_replay_transport_seeks_and_stops_at_clip_bounds(replay_controls, width):
    state = replay_controls
    state.width = width
    state.frame(3)
    assert not state.session.adapter.caps.simulation
    assert state.session.replay_info.frame_index == 0
    state.click("previous")
    assert state.session.replay_info.frame_index == 0
    state.click("next")
    assert state.session.replay_info.frame_index == 1
    state.click("previous")
    assert state.session.replay_info.frame_index == 0
    last = state.session.replay_info.frame_count - 1
    assert state.session.submit(cmd.SeekReplay(last)).ok
    state.frame(3)
    state.click("next")
    assert state.session.replay_info.frame_index == last
    state.click("previous")
    assert state.session.replay_info.frame_index == last - 1


def test_replay_previous_pauses_but_next_requires_paused_playback(replay_controls):
    state = replay_controls
    assert state.session.submit(cmd.SeekReplay(2)).ok
    assert state.session.submit(cmd.SetReplayPlayback(paused=False)).ok
    state.frame(3)
    before = state.session.replay_info.frame_index
    state.click("next")
    assert state.session.replay_info.frame_index == before
    assert not state.session.replay_info.paused
    state.click("previous")
    assert state.session.replay_info.frame_index == before - 1
    assert state.session.replay_info.paused
