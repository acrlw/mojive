"""Real GPU capture through the local control service."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive import RenderProduct, Scene, SceneRenderer, SharedImage
from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.composition import build_scene
from mojive.control_rpc import ControlServer, ControlService, RpcClient
from mojive.control_schema import Validator
from mojive.operations import OPERATIONS
from mojive.types import CameraView

pytestmark = pytest.mark.gpu


@pytest.fixture
def standalone_capture_backend(monkeypatch):
    # Cocoa context creation must stay on the main thread. Standalone RPC
    # capture owns a graphics worker; attached-viewer tests exercise OpenGL.
    if sys.platform == "darwin":
        monkeypatch.setenv("MOJIVE_RENDERER", "wgpu")


@pytest.mark.parametrize("mode", ["rgb", "depth", "object_id", "segmentation"])
def test_shared_capture_matches_arrays_and_reuses_the_buffer(
    tmp_path, monkeypatch, mode, standalone_capture_backend
):
    from mojive.capture import decode_image

    scene = Scene()
    scene.box()
    service = ControlService(StaticSceneAdapter(scene))
    monkeypatch.chdir(tmp_path)
    try:
        params = {"mode": mode, "width": 64, "height": 48}
        reference = decode_image(service.dispatch("capture", {**params, "transport": "base64"}))
        with SharedImage(reference.shape, reference.dtype) as target:
            target.array.fill(0)
            for _ in range(2):
                payload = service.dispatch(
                    "capture",
                    {
                        **params,
                        "transport": "shared_memory",
                        "buffer": target.descriptor,
                    },
                )
                Validator(OPERATIONS["capture"].output_schema).validate(payload)
                assert "path" not in payload and "data" not in payload
                assert payload["buffer"] == target.descriptor
                np.testing.assert_array_equal(target.array, reference)
        assert not list(tmp_path.iterdir())
    finally:
        service.close()


@pytest.mark.parametrize(
    "mode,encoding", [("rgb", "raw"), ("rgb", "png"), ("depth", "npy"), ("segmentation", "raw")]
)
def test_in_memory_scene_capture_creates_no_files(
    tmp_path, monkeypatch, mode, encoding, standalone_capture_backend
):
    from mojive.capture import decode_image

    scene = Scene()
    scene.box()
    service = ControlService(StaticSceneAdapter(scene))
    monkeypatch.chdir(tmp_path)
    try:
        payload = service.dispatch(
            "capture",
            {
                "mode": mode,
                "width": 64,
                "height": 48,
                "transport": "base64",
                "encoding": encoding,
            },
        )
        Validator(OPERATIONS["capture"].output_schema).validate(payload)
        image = decode_image(payload)
        assert image.shape[:2] == (48, 64)
        assert "path" not in payload
        assert not list(tmp_path.iterdir())
    finally:
        service.close()


@pytest.mark.physics
def test_control_service_captures_rgb_depth_and_segmentation(tmp_path, standalone_capture_backend):
    from mojive.adapters.mujoco_adapter import MuJoCoAdapter

    asset = Path("assets/test_scene.xml").resolve()
    service = ControlService(MuJoCoAdapter(asset), asset)
    try:
        rgb = service.dispatch(
            "capture",
            {"mode": "rgb", "width": 128, "height": 96, "output": str(tmp_path / "rgb.png")},
        )
        depth = service.dispatch(
            "capture",
            {"mode": "depth", "width": 128, "height": 96, "output": str(tmp_path / "depth.npy")},
        )
        segmentation = service.dispatch(
            "capture",
            {
                "mode": "segmentation",
                "width": 128,
                "height": 96,
                "output": str(tmp_path / "segmentation.npy"),
            },
        )
    finally:
        service.close()

    assert Image.open(rgb["path"]).size == (128, 96)
    assert np.load(depth["path"]).shape == (96, 128)
    assert np.load(segmentation["path"]).shape == (96, 128, 2)


def test_rpc_scene_capture_follows_visibility_and_camera_across_clients(
    tmp_path, standalone_capture_backend
):
    scene = Scene()
    box = scene.box(color=(1, 0, 0, 1))
    view = CameraView(up=np.array([0.0, 0.5, 1.0], np.float32))
    camera_id = scene.add_camera("inspection", view)
    adapter = StaticSceneAdapter(scene)
    service = ControlService(adapter)
    server = ControlServer(tmp_path / "capture.sock", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    params = {"width": 96, "height": 72, "mode": "object_id", "output": str(tmp_path / "ids.npy")}
    try:
        with (
            RpcClient(server.socket_path, timeout=15) as first,
            RpcClient(server.socket_path, timeout=15) as second,
        ):
            hello = first.hello()
            assert hello["capture_scope"] == "session_scene"
            assert "object_id" in hello["capture_modes"]
            first.call("set_camera", {"camera_id": camera_id})
            result = first.call("capture", params)
            ids = np.load(result["path"])
            assert ids.dtype == np.uint32 and np.count_nonzero(ids == box.object_id) > 30
            assert result["orientation"] == "top_left"
            rgb_params = {**params, "mode": "rgb", "output": str(tmp_path / "color.png")}
            first.call("set_render_flag", {"name": "mjRND_DEPTH", "enabled": True})
            debug = first.call("capture", rgb_params)
            rgb = np.asarray(Image.open(debug["path"]))
            np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
            np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])
            first.call("set_render_flag", {"name": "depth", "enabled": False})
            color = first.call("capture", rgb_params)
            rgb = np.asarray(Image.open(color["path"]))
            assert rgb[36, 48, 0] > rgb[36, 48, 1] + 30
            first.call("set_visualization_flag", {"name": "transparent", "enabled": True})
            faded = first.call("capture", rgb_params)
            faded_rgb = np.asarray(Image.open(faded["path"]))
            assert faded_rgb[36, 48, 0] < rgb[36, 48, 0]
            assert np.all(service.session.source.geom_rgba[:, 3] == 1.0)
            first.call("set_visualization_flag", {"name": "transparent", "enabled": False})
            restored = first.call("capture", rgb_params)
            np.testing.assert_array_equal(np.asarray(Image.open(restored["path"])), rgb)

            # A selected camera keeps its full CameraView, including roll.
            with SceneRenderer(scene.source, width=96, height=72) as renderer:
                renderer.update(scene.frame, camera=view)
                np.testing.assert_array_equal(ids, renderer.render(product=RenderProduct.OBJECT_ID))

            node = first.call("inspect_object", {"object_id": box.object_id})
            second.call("set_visible", {"node_id": node["node_id"], "visible": False})
            hidden_result = second.call("capture", params)
            assert not np.load(hidden_result["path"]).any()
            assert hidden_result["structure_generation"] > result["structure_generation"]
            first.call("set_visible", {"node_id": node["node_id"], "visible": True})
            resized = first.call("capture", {**params, "width": 64, "height": 48})
            image = np.load(resized["path"])
            assert image.shape == (48, 64) and np.count_nonzero(image == box.object_id) > 20
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.close()


def test_attached_rpc_capture_uses_the_viewer_session_and_keeps_rendering(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    box = scene.box()
    with build_scene(scene, width=640, height=480, vsync=False, show_window=False) as viewer:
        server = viewer.start_rpc(tmp_path / "viewer.sock")

        def inspect():
            with RpcClient(server.socket_path, timeout=10) as client:
                assert client.hello()["viewer_attached"]
                node = client.call("inspect_object", {"object_id": box.object_id})
                params = {
                    "mode": "object_id",
                    "width": 96,
                    "height": 72,
                    "output": str(tmp_path / "viewer-ids.npy"),
                }
                visible = client.call("capture", params)
                assert np.any(np.load(visible["path"]) == box.object_id)
                client.call("set_visible", {"node_id": node["node_id"], "visible": False})
                hidden = client.call("capture", params)
                assert not np.load(hidden["path"]).any()
                return node["node_id"]

        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(inspect)
            deadline = time.monotonic() + 15
            while not future.done() and time.monotonic() < deadline:
                viewer.sync()
            node_id = future.result(timeout=1)
        assert not viewer.session.node(node_id).visible
        viewer.sync()
        assert viewer.backend.target.read_rgb().shape[-1] == 3


def test_viewport_capture_completes_after_present_and_keeps_capture_camera_separate(
    tmp_path, monkeypatch
):
    from mojive.capture import decode_image

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scene = Scene()
    scene.box(color=(1, 0, 0, 1))
    with build_scene(scene, width=640, height=480, vsync=False, show_window=False) as viewer:
        viewer.sync()
        server = viewer.start_rpc(tmp_path / "viewer.sock")

        def inspect():
            with RpcClient(server.socket_path, timeout=15) as client:

                def call(method, params=None):
                    result = client.call(method, params)
                    Validator(OPERATIONS[method].output_schema).validate(result)
                    return result

                call("get_viewer_settings")
                call("get_panels")
                capture_before = call("get_capture_settings")["camera"]
                view = {
                    "eye": [4, -3, 2],
                    "target": [0, 0, 0],
                    "up": [0, 0.5, 1],
                    "focal_length": [35, 35],
                    "sensor_size": [36, 24],
                }
                result = call("set_viewport_camera", view)
                assert result["up"] == view["up"]
                for surface in ("viewport", "window"):
                    capture = call(
                        "capture_viewport",
                        {"surface": surface, "output": str(tmp_path / f"{surface}.png")},
                    )
                    assert capture["scope"] == surface
                    pixels = np.asarray(Image.open(capture["path"]))
                    assert list(pixels.shape) == capture["shape"]
                    assert np.ptp(pixels) > 100
                    memory = call("capture_viewport", {"surface": surface, "transport": "base64"})
                    decoded = decode_image(memory)
                    assert decoded.shape == pixels.shape
                    assert np.ptp(decoded) > 100
                    assert "path" not in memory
                    with SharedImage(decoded.shape) as target:
                        metadata = client.capture_into(target, surface=surface)
                        Validator(OPERATIONS["capture_viewport"].output_schema).validate(metadata)
                        assert "data" not in metadata and "path" not in metadata
                        assert metadata["buffer"] == target.descriptor
                        assert np.ptp(target.array) > 100
                        # The status readout can change between frames; viewport content is stable.
                        if surface == "viewport":
                            np.testing.assert_array_equal(target.array, decoded)
                actual = call("get_viewport_camera")
                assert actual["up"] == view["up"]
                assert actual["focal_length"] == view["focal_length"]
                assert call("get_capture_settings")["camera"] == capture_before
                return actual

        with ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(inspect)
            deadline = time.monotonic() + 25
            while not future.done() and time.monotonic() < deadline:
                viewer.sync()
            actual = future.result(timeout=1)
        np.testing.assert_allclose(viewer.session.camera.eye, actual["eye"])
        width, height = viewer.window.size_pixels
        assert viewer.capture_array(surface="window").shape == (height, width, 3)


@pytest.mark.physics
def test_mujoco_capture_keeps_session_overrides_and_model_render_limits(
    tmp_path, monkeypatch, standalone_capture_backend
):
    import mujoco

    from mojive.adapters.mujoco_adapter import MuJoCoAdapter

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="32" offheight="24"/></visual>'
        '<worldbody><geom name="box" type="box" size=".5 .5 .5" rgba="1 0 0 1"/>'
        "</worldbody></mujoco>"
    )
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    service = ControlService(adapter)
    try:
        node = next(item for item in service.session.nodes if item.name == "box")
        # An adapter without color write-back still supports session-side edits.
        monkeypatch.setattr(adapter, "set_geometry_color", lambda *args: False)
        assert service.session.submit(cmd.SetGeometryColor(node.node_id, (0, 1, 0, 1))).ok
        result = service.dispatch(
            "capture",
            {
                "width": 96,
                "height": 72,
                "output": str(tmp_path / "override.png"),
            },
        )
        rgb = np.asarray(Image.open(result["path"]))
        assert rgb[36, 48, 1] > rgb[36, 48, 0] + 30
        assert model.vis.global_.offwidth == 32 and model.vis.global_.offheight == 24
        assert model.geom_rgba[0] == pytest.approx([1, 0, 0, 1])
        result = service.dispatch(
            "capture",
            {
                "mode": "segmentation",
                "width": 96,
                "height": 72,
                "output": str(tmp_path / "semantic.npy"),
            },
        )
        assert np.load(result["path"])[36, 48] == pytest.approx([0, int(mujoco.mjtObj.mjOBJ_GEOM)])
    finally:
        service.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Cocoa requires main-thread window creation")
def test_unsupported_worker_opengl_capture_returns_error_without_killing_service(
    tmp_path, monkeypatch
):
    from mojive.control_rpc import RpcError

    monkeypatch.setenv("MOJIVE_RENDERER", "opengl")
    monkeypatch.setenv("MOJIVE_GL", "glfw")
    scene = Scene()
    scene.box()
    service = ControlService(StaticSceneAdapter(scene))
    server = ControlServer(tmp_path / "unsupported.sock", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with RpcClient(server.socket_path) as client:
            with pytest.raises(RpcError, match="MOJIVE_RENDERER=wgpu"):
                client.call("capture", {"output": str(tmp_path / "unsupported.png")})
            assert client.hello()["service"] == "mojive.control"
            assert client.call("get_scene")["objects"]
            assert not (tmp_path / "unsupported.png").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.close()
