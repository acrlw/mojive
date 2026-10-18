"""Published imports and command polymorphism survive implementation reorganization."""

import ast
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "legacy,implementation,names",
    [
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


def test_internal_module_imports_resolve_including_deferred_imports():
    """Catch removed modules even when their imports only run during interaction."""
    source = Path(__file__).resolve().parents[1] / "python"
    missing = []
    for path in sorted(source.rglob("*.py")):
        package = ("mojive", *path.relative_to(source).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else ()
                modules = [".".join((*prefix, *node.module.split(".")))] if node.module else []
            else:
                continue
            for module in modules:
                if not module.startswith("mojive."):
                    continue
                target = source.joinpath(*module.split(".")[1:])
                if not target.with_suffix(".py").is_file() and not target.is_dir():
                    missing.append(f"{path.relative_to(source)}:{node.lineno}: {module}")
    assert not missing, "Unresolved internal imports:\n" + "\n".join(missing)


def test_command_subclasses_keep_dispatch_and_unknown_commands_keep_failure():
    from mojive.adapters.toy import ToyPhysicsAdapter
    from mojive.commands import Command, SetCtrlVector, SetSpeed
    from mojive.session import Session

    class CustomSpeed(SetSpeed):
        pass

    class MultipleSpeed(CustomSpeed, SetCtrlVector):
        pass

    session = Session(ToyPhysicsAdapter())
    try:
        assert session.submit(CustomSpeed(2.5)).ok
        assert session.speed == 2.5
        assert session.submit(MultipleSpeed(3.25)).ok
        assert session.speed == 3.25
        result = session.submit(Command())
        assert not result.ok
        assert result.message == "Unknown command: Command"
        assert session.speed == 3.25
    finally:
        session.release()


def test_command_subclasses_keep_supported_edits_and_undo_redo():
    from mojive import Scene
    from mojive import commands as cmd
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.session import Session

    class CustomObject(cmd.AddSceneObject):
        pass

    session = Session(StaticSceneAdapter(Scene()))
    try:
        result = session.submit(CustomObject("box", name="custom object"))
        assert result.ok, result.message
        assert session.node_by_object_id(result.entity_id).name == "custom object"
        assert session.source.instance_count == 1
        assert session.submit(cmd.Undo()).ok
        assert session.source.instance_count == 0
        assert not session.can_undo
        assert session.submit(cmd.Redo()).ok
        assert session.source.instance_count == 1
        assert session.node_by_object_id(result.entity_id).name == "custom object"
    finally:
        session.release()
