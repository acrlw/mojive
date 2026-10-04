"""The official MCP client exercises the stdio bridge and the real RPC owner."""

import asyncio
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from mojive import Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.control.rpc import ControlServer, ControlService

mcp = pytest.importorskip(
    "mcp", reason="Install mojive[mcp] to validate the optional MCP transport"
)
StdioServerParameters = pytest.importorskip("mcp.client.stdio").StdioServerParameters

pytestmark = pytest.mark.integration


@contextmanager
def connection(service):
    with tempfile.TemporaryDirectory(prefix="fv-") as directory:
        server = ControlServer(Path(directory) / "control.sock", service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield StdioServerParameters(
                command=sys.executable,
                args=["-m", "mojive.control.mcp", "--socket", str(server.socket_path)],
                env=dict(os.environ),
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


async def value(client, method, **params):
    response = await client.call_tool(method, params)
    assert not response.is_error, response.content
    return response.structured_content["result"]


@pytest.mark.parametrize("protocol", ["auto", "legacy"])
def test_stdio_discover_edit_readback_stale_failure_and_recovery(protocol):
    scene = Scene()
    service = ControlService(StaticSceneAdapter(scene))
    try:
        with connection(service) as endpoint:

            async def exercise():
                async with mcp.Client(endpoint, mode=protocol) as client:
                    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
                    assert {
                        "hello",
                        "inspect_object",
                        "set_joint_properties",
                        "capture",
                        "undo",
                    } <= tools.keys()
                    assert tools["inspect_object"].annotations.read_only_hint
                    assert not tools["add_scene_object"].annotations.read_only_hint
                    assert (
                        "expected_document" in tools["add_scene_object"].input_schema["properties"]
                    )
                    assert not (await value(client, "hello"))["viewer_attached"]
                    original = await value(client, "get_scene")
                    created = await value(
                        client,
                        "add_scene_object",
                        shape="box",
                        name="created",
                        expected_document=original["document"],
                    )
                    inspected = await value(
                        client, "inspect_object", object_id=created["object_id"]
                    )
                    assert inspected["name"] == "created"
                    rejected = await client.call_tool(
                        "rename_scene_entity",
                        {
                            "object_id": created["object_id"],
                            "name": "wrong",
                            "expected_document": original["document"],
                        },
                    )
                    assert rejected.is_error
                    assert rejected.structured_content["error"]["code"] == "stale_document"
                    assert (
                        rejected.structured_content["error"]["details"]["actual"]
                        == inspected["document"]
                    )
                    await value(
                        client,
                        "rename_scene_entity",
                        object_id=created["object_id"],
                        name="updated",
                        expected_document=inspected["document"],
                    )
                    assert (await value(client, "inspect_object", object_id=created["object_id"]))[
                        "name"
                    ] == "updated"
                    await value(client, "undo")
                    assert (await value(client, "inspect_object", object_id=created["object_id"]))[
                        "name"
                    ] == "created"
                    invalid = await client.call_tool(
                        "set_geometry_color", {"node_id": 1, "rgba": [1, 0]}
                    )
                    assert (
                        invalid.is_error
                        and invalid.structured_content["error"]["code"] == "invalid_params"
                    )
                    unsupported = await client.call_tool(
                        "set_joint_properties",
                        {
                            "joint_id": 0,
                            "axis": [0, 0, 1],
                            "limited": False,
                            "range": [0, 0],
                            "damping": 0,
                            "stiffness": 0,
                        },
                    )
                    assert (
                        unsupported.is_error
                        and unsupported.structured_content["error"]["code"] == "unsupported"
                    )
                    assert (await value(client, "inspect_object", object_id=created["object_id"]))[
                        "name"
                    ] == "created"

            asyncio.run(exercise())
    finally:
        service.close()


@pytest.mark.gpu
def test_stdio_real_capture_edit_undo_matches_initial_pixels():
    import base64
    import io

    import numpy as np
    from PIL import Image

    scene = Scene()
    box = scene.box(size=(0.5, 0.5, 0.5), color=(0.1, 0.3, 0.8, 1))
    service = ControlService(StaticSceneAdapter(scene))
    try:
        with connection(service) as endpoint:

            async def exercise():
                async with mcp.Client(endpoint) as client:
                    await client.list_tools()

                    async def capture():
                        result = await client.call_tool(
                            "capture",
                            {"width": 160, "height": 120, "transport": "base64", "encoding": "png"},
                        )
                        assert not result.is_error, result.content
                        image = next(item for item in result.content if item.type == "image")
                        assert image.mime_type == "image/png"
                        payload = result.structured_content["result"]
                        assert "data" not in payload and payload["scope"] == "session_scene"
                        assert payload["image"]["mime_type"] == "image/png"
                        return np.asarray(Image.open(io.BytesIO(base64.b64decode(image.data))))

                    initial = await capture()
                    inspected = await value(client, "inspect_object", object_id=box.object_id)
                    await value(
                        client,
                        "set_geometry_color",
                        node_id=inspected["geometries"][0]["node_id"],
                        rgba=[0.9, 0.1, 0.1, 1],
                        expected_document=inspected["document"],
                    )
                    changed = await capture()
                    assert np.count_nonzero(initial != changed) > 100
                    await value(client, "undo")
                    restored = await capture()
                    np.testing.assert_array_equal(restored, initial)

            asyncio.run(exercise())
    finally:
        service.close()


def test_png_payload_is_carried_once_and_metadata_matches_tool_schema(monkeypatch):
    # This fixture checks transport conversion without opening a graphics context.
    # The GPU acceptance above checks that rendered pixels change and are restored.
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    service = ControlService(StaticSceneAdapter(Scene()))
    monkeypatch.setattr(
        service.application,
        "_capture",
        lambda _params: {
            "scope": "session_scene",
            "transport": "base64",
            "encoding": "png",
            "data": png,
            "shape": [1, 1, 3],
            "dtype": "uint8",
            "orientation": "top_left",
        },
    )
    try:
        with connection(service) as endpoint:

            async def exercise():
                from mojive.control.schema import Validator

                async with mcp.Client(endpoint) as client:
                    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
                    result = await client.call_tool(
                        "capture", {"transport": "base64", "encoding": "png"}
                    )
                    assert not result.is_error
                    Validator(tools["capture"].output_schema).validate(result.structured_content)
                    assert "data" not in result.structured_content["result"]
                    assert result.structured_content["result"]["image"] == {
                        "mime_type": "image/png"
                    }
                    assert [item.data for item in result.content if item.type == "image"] == [png]
                    assert all(
                        png not in item.text for item in result.content if item.type == "text"
                    )

            asyncio.run(exercise())
    finally:
        service.close()
