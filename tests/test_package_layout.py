"""Published imports and command polymorphism survive implementation reorganization."""

import importlib

import pytest


@pytest.mark.parametrize(
    "legacy,implementation,names",
    [
        ("ui.draw2d", "ui.paint_protocol", ("Draw2D",)),
        ("ui.draw2d", "ui.imgui_draw", ("ImguiDraw2D", "ink_box")),
        ("ui.draw2d", "ui.text_layout", ("fit_text", "text_line_y")),
        ("ui.draw2d", "ui.drag_link", ("draw_drag_link",)),
        ("curves2d", "geometry2d.curves", ("arrow_points", "smooth_capsule_points")),
        ("draglink2d", "geometry2d.drag_link", ("drag_link_field", "smooth_drag_link_mesh")),
        ("canvas2d", "render.canvas", ("Canvas2D", "CanvasLayer2D")),
        ("backends", "app.backends", ("make_adapter", "available_backends")),
        ("composition", "app.composition", ("Viewer", "build", "build_scene")),
        ("renderer", "app.renderer", ("Renderer",)),
        ("scene_renderer", "render.offscreen", ("SceneRenderer",)),
        ("control_rpc", "control.rpc", ("RpcClient", "ControlService", "RpcError")),
        ("operations", "control.operations", ("OPERATIONS", "Operation")),
        ("scene_io", "scene.io", ("load_scene", "save_scene")),
        ("recording", "capture.recording", ("SnapshotWriter", "VideoRecorder")),
        ("shared_image", "capture.shared_image", ("SharedImage",)),
    ],
)
def test_published_imports_resolve_to_the_same_implementation(legacy, implementation, names):
    old = importlib.import_module(f"mojive.{legacy}")
    owner = importlib.import_module(f"mojive.{implementation}")
    for name in names:
        assert getattr(old, name) is getattr(owner, name)


def test_command_subclasses_keep_dispatch_and_unknown_commands_keep_failure():
    from mojive.adapters.toy import ToyPhysicsAdapter
    from mojive.commands import Command, SetSpeed
    from mojive.session import Session

    class CustomSpeed(SetSpeed):
        pass

    session = Session(ToyPhysicsAdapter())
    try:
        assert session.submit(CustomSpeed(2.5)).ok
        assert session.speed == 2.5
        result = session.submit(Command())
        assert not result.ok
        assert result.message == "Unknown command: Command"
        assert session.speed == 2.5
    finally:
        session.release()
