"""Production scene navigation releases native state when a row fails."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.adapters.base import NodeType
from mojive.ui.theme import THEME
from mojive.ui.workspace.navigation import Navigation
from mojive.ui.workspace.widgets import UI


def test_row_failure_releases_clipper_before_window_and_context_close(monkeypatch):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (400, 300)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        navigation = Navigation.__new__(Navigation)
        navigation.workspace = SimpleNamespace(
            s=1,
            query="",
            filter="",
            ui=UI(SimpleNamespace(style_scale=1, font_report=SimpleNamespace(cjk_path=None))),
        )
        navigation._scene_filter_token = None
        navigation._scroll_scene_node = None
        monkeypatch.setattr(navigation, "_scene_controls", lambda ctx, x, y, w: y)
        monkeypatch.setattr(navigation, "_scene_row_geometry", lambda: (28, (), (0, 0, 0, 0)))
        monkeypatch.setattr(navigation, "scene_rows", lambda ctx: [(None, 0, True)])

        def fail_row(*args):
            raise RuntimeError("Row paint failed")

        monkeypatch.setattr(navigation, "_draw_scene_row", fail_row)
        imgui.new_frame()
        imgui.set_next_window_size((400, 300))
        imgui.begin("Scene navigation")
        font, size = imgui.get_font().font_id, imgui.get_font_size()
        with pytest.raises(RuntimeError, match="Row paint failed"):
            navigation.draw_scene(None, 20, 40, 300, 200)
        assert context.clipper_temp_data_stacked == 0
        assert context.current_window.name == "Scene navigation"
        assert (imgui.get_font().font_id, imgui.get_font_size()) == (font, size)
        imgui.end()
        imgui.render()
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize(
    "kind,count,names,types",
    (
        (NodeType.LINK, 1, True, True),
        (NodeType.LINK, 2, True, True),
        (NodeType.JOINT, 1, False, False),
        (None, 1, True, False),
    ),
)
def test_joint_identity_distinguishes_body_inspection_from_joint_inspection(
    monkeypatch, kind, count, names, types
):
    navigation = Navigation.__new__(Navigation)
    calls = []
    navigation.joints = SimpleNamespace(_initial_qpos=())
    navigation.pose_editor = SimpleNamespace(draw=lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(navigation, "pose_joints", lambda *args: tuple(range(count)))
    monkeypatch.setattr(navigation, "_pose_controls", lambda ctx, x, y, w: y)
    ctx = SimpleNamespace(
        theme=THEME,
        session=SimpleNamespace(node=lambda node_id: SimpleNamespace(type=kind)),
    )
    navigation.draw_pose(ctx, 0, 0, 300, node_id=None if kind is None else 7)
    assert calls[0]["show_names"] is names
    assert calls[0]["show_types"] is types
    assert calls[0]["collapsible_poses"] is (kind is None)
