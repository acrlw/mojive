"""Static checks for the user-facing example programs."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import numpy as np
import pytest


def test_python_examples_are_import_safe_and_have_main_entry_points() -> None:
    examples = Path(__file__).parents[1] / "examples"
    scripts = sorted(examples.glob("*.py"))
    assert scripts
    for script in scripts:
        module = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        assert "main" in functions, script.name
        assert any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in module.body
        ), script.name


def test_video_annotation_leaves_source_rgb_untouched():
    pytest.importorskip("mujoco")
    example = Path(__file__).parents[1] / "examples/mujoco_video.py"
    annotate = runpy.run_path(str(example))["annotate"]
    rgb = np.full((60, 240, 3), 120, np.uint8)
    annotated = annotate(rgb, "Policy A", 1.234)
    assert np.all(rgb == 120)
    assert annotated.shape == rgb.shape
    assert not np.shares_memory(annotated, rgb)
    assert not np.array_equal(annotated, rgb)


@pytest.mark.integration
def test_remote_example_republishes_authoring_structure_and_survives_marker_removal(monkeypatch):
    from types import SimpleNamespace

    from mojive import commands as cmd

    example = Path(__file__).parents[1] / "examples/remote_publish.py"
    namespace = runpy.run_path(str(example))
    main = namespace["main"]
    events = []

    class Publisher:
        def __init__(self, *args):
            self.iteration = 0

        def publish_structure(self, structure):
            events.append(("structure", structure.source.instance_count))

        def pump_commands(self, callback):
            self.iteration += 1
            # Exercise the command callback provided by the running example.
            if self.iteration == 1:
                callback({"op": "remove_scene_entity", "object_id": 2})

        def publish_frame(self, frame):
            events.append(("frame", len(frame.geom_xpos)))
            if self.iteration == 2:
                raise StopIteration("end the example")

        def close(self):
            events.append(("closed", 0))

    # The transport envelope is handled separately; execute the authored operation
    # on the example's actual session when its publisher pumps a command.
    def remove(session, message):
        assert message["op"] == "remove_scene_entity"
        return session.submit(cmd.RemoveSceneEntity(message["object_id"]))

    monkeypatch.setitem(main.__globals__, "SnapshotPublisher", Publisher)
    monkeypatch.setitem(main.__globals__, "handle_session_command", remove)
    monkeypatch.setitem(
        main.__globals__, "parse_args", lambda: SimpleNamespace(host="localhost", port=0, hz=60)
    )
    monkeypatch.setattr(namespace["time"], "sleep", lambda *args: None)
    with pytest.raises(StopIteration, match="end the example"):
        main()
    assert events == [("structure", 2), ("structure", 1), ("frame", 1), ("frame", 1), ("closed", 0)]
