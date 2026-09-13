"""Attached Scale requests update the rendered scene and share authored history."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojive import Scene
from mojive import commands as cmd
from mojive.app.composition import build_scene
from mojive.capture import decode_image
from mojive.control.rpc import RpcClient, RpcError

pytestmark = pytest.mark.gpu


def test_attached_scale_bakes_and_undo_redo_reaches_the_renderer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    with build_scene(Scene(), width=640, height=480, vsync=False, show_window=False) as viewer:
        server = viewer.start_rpc(tmp_path / "scale.sock")

        def edit():
            with RpcClient(server.socket_path, timeout=10) as client:
                assert client.hello()["viewer_attached"]
                created = client.call("add_scene_object", {"shape": "box", "size": [0.2, 0.2, 0.2]})
                before = client.call("inspect_object", {"object_id": created["object_id"]})
                assert before["scalable"] and before["scale"] == [1, 1, 1]

                def pixels():
                    capture = client.call(
                        "capture",
                        {
                            "mode": "object_id",
                            "width": 128,
                            "height": 96,
                            "transport": "base64",
                        },
                    )
                    return np.count_nonzero(decode_image(capture) == created["object_id"])

                initial_pixels = pixels()
                assert initial_pixels > 0
                client.call(
                    "set_scale",
                    {
                        "node_id": before["node_id"],
                        "scale": [2, 2, 2],
                        "expected_document": before["document"],
                    },
                )
                assert pixels() > initial_pixels * 2
                for method, size in ((None, 0.4), ("undo", 0.2), ("redo", 0.4)):
                    if method:
                        client.call(method)
                    current = client.call("inspect_object", {"object_id": created["object_id"]})
                    assert current["scale"] == [1, 1, 1]
                    np.testing.assert_allclose(current["geometries"][0]["size"], [size] * 3)
                    if method == "undo":
                        assert pixels() == initial_pixels
                return created["object_id"]

        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(edit)
            deadline = time.monotonic() + 30
            while not future.done() and time.monotonic() < deadline:
                viewer.sync()
            object_id = future.result(timeout=1)
        viewer.sync()
        node = viewer.session.node_by_object_id(object_id)
        assert viewer.session.scale_factors(node.node_id) == (1, 1, 1)
        np.testing.assert_allclose(viewer.session.source.geom_size[-1], [0.4] * 3)


def test_attached_rpc_preserves_pending_scale_until_ui_applies(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    box = scene.box(size=(0.2, 0.3, 0.4))
    with build_scene(scene, width=640, height=480, vsync=False, show_window=False) as viewer:
        viewer.sync()
        node = viewer.session.node_by_object_id(box.object_id)
        draft = viewer.app.model_edits
        assert draft.stage(cmd.SetScale(node.node_id, [2, 2, 2])).ok
        server = viewer.start_rpc(tmp_path / "draft.sock")

        def inspect_and_try_edit():
            with RpcClient(server.socket_path, timeout=10) as client:
                before = client.call("inspect_object", {"object_id": box.object_id})
                assert before["scale"] == [2, 2, 2]
                for method, params in (
                    ("set_scale", {"node_id": node.node_id, "scale": [3, 3, 3]}),
                    ("undo", {}),
                ):
                    with pytest.raises(RpcError) as error:
                        client.call(method, params)
                    assert error.value.code == "pending_edits"
                assert client.call("inspect_object", {"object_id": box.object_id}) == before

        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(inspect_and_try_edit)
            deadline = time.monotonic() + 30
            while not future.done() and time.monotonic() < deadline:
                viewer.sync()
            future.result(timeout=1)
        assert draft.active and draft.compatible()
        np.testing.assert_allclose(scene.source.geom_size[0], [0.2, 0.3, 0.4])
        draft.applying = True
        result = viewer.session.apply_model_edits(draft.resolve_commands(viewer.session))
        assert result.ok, result.message
        draft.clear()
        viewer.sync()
        np.testing.assert_allclose(scene.source.geom_size[0], [0.4, 0.6, 0.8])
        assert viewer.session.scale_factors(node.node_id) == (1, 1, 1)
