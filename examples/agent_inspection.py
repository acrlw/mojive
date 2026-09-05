"""Discover, edit, save, and visually verify a scene through one Mojive RPC connection."""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import CameraView, Scene
from mojive.adapters.static import StaticSceneAdapter
from mojive.control_rpc import ControlServer, ControlService, RpcClient, RpcError


def inspection_scene() -> Scene:
    scene = Scene()
    scene.plane(size=(3, 3, 0.01), color=(0.6, 0.65, 0.7, 1))
    scene.box(name="inspection-box", position=(0, -0.65, 0.5), color=(0.1, 0.5, 0.85, 1))
    scene.sphere(name="reference-sphere", position=(0, 0.65, 0.5), color=(0.95, 0.35, 0.12, 1))
    scene.add_camera(
        "inspection", CameraView(eye=np.array([4, -4, 3]), target=np.array([0, 0, 0.4]))
    )
    return scene


def exercise(socket_path: Path, output: Path) -> None:
    with RpcClient(socket_path, timeout=20) as client:
        capabilities = client.hello()
        schema = client.describe_operations(name="edit_scene")
        assert schema["operations"][0]["available"]
        metadata = client.call("get_scene")
        node = next(item for item in metadata["objects"] if item["name"] == "inspection-box")
        inspected = client.call("inspect_object", {"object_id": node["object_id"]})
        geometry = inspected["geometries"][0]
        camera = next(item for item in metadata["cameras"] if item["name"] == "inspection")
        client.call("set_capture_camera", {"camera_id": camera["camera_id"]})
        captures, counts = [], []
        for label, visible in (("visible", True), ("hidden", False), ("restored", True)):
            client.call(
                "set_visible",
                {
                    "node_id": node["node_id"],
                    "visible": visible,
                    "expected_document": metadata["document"],
                },
            )
            assert (
                client.call("inspect_object", {"object_id": node["object_id"]})["visible"]
                is visible
            )
            for mode in ("rgb", "object_id"):
                suffix = ".png" if mode == "rgb" else ".npy"
                result = client.call(
                    "capture",
                    {
                        "mode": mode,
                        "width": 640,
                        "height": 480,
                        "output": str(output / f"{label}-{mode}{suffix}"),
                    },
                )
                captures.append(result)
                if mode == "object_id":
                    counts.append(
                        int(np.count_nonzero(np.load(result["path"]) == node["object_id"]))
                    )
        assert counts[0] > 100 and counts[1] == 0 and counts[2] == counts[0]
        edited = client.call(
            "edit_scene",
            {
                "label": "Arrange inspection box",
                "expected_document": metadata["document"],
                "operations": [
                    {
                        "method": "rename_scene_entity",
                        "params": {"object_id": node["object_id"], "name": "cargo"},
                    },
                    {
                        "method": "set_pose",
                        "params": {
                            "node_id": node["node_id"],
                            "position": [0, -1, 0.8],
                            "rotation": np.eye(3).tolist(),
                        },
                    },
                    {
                        "method": "set_geometry_size",
                        "params": {"node_id": geometry["node_id"], "size": [0.55, 0.45, 0.65]},
                    },
                    {
                        "method": "set_geometry_color",
                        "params": {"node_id": geometry["node_id"], "rgba": [0.1, 0.8, 0.35, 1]},
                    },
                ],
            },
        )
        verified = client.call("inspect_object", {"object_id": node["object_id"]})
        np.testing.assert_allclose(verified["geometries"][0]["rgba"], [0.1, 0.8, 0.35, 1])
        np.testing.assert_allclose(verified["geometries"][0]["size"], [0.55, 0.45, 0.65])
        assert verified["position"] == [
            0,
            -1,
            float(np.float32(0.8)),
        ]
        client.call("undo")
        restored = client.call("inspect_object", {"object_id": node["object_id"]})
        np.testing.assert_allclose(restored["geometries"][0]["rgba"], geometry["rgba"])
        np.testing.assert_allclose(restored["geometries"][0]["size"], geometry["size"])
        assert (
            client.call("inspect_object", {"object_id": node["object_id"]})["name"]
            == "inspection-box"
        )
        client.call("redo")
        try:
            client.call(
                "edit_scene",
                {
                    "operations": [
                        {
                            "method": "rename_scene_entity",
                            "params": {"object_id": node["object_id"], "name": "temporary"},
                        },
                        {"method": "remove_scene_entity", "params": {"object_id": 999999}},
                    ]
                },
            )
        except RpcError as exc:
            assert exc.code == "command_failed"
        else:
            raise AssertionError("The failed edit must be rolled back")
        assert client.call("inspect_object", {"object_id": node["object_id"]})["name"] == "cargo"
        saved = output / "edited.mojive.json"
        client.call("save_scene", {"path": str(saved)})
        client.call("new_scene")
        client.call("open_scene", {"path": str(saved)})
        reopened = client.call("get_scene")
        assert reopened["document"]["id"] != metadata["document"]["id"]
        try:
            client.call(
                "remove_scene_entity",
                {"object_id": node["object_id"], "expected_document": edited["document"]},
            )
        except RpcError as exc:
            assert exc.code == "stale_document"
        else:
            raise AssertionError("Stale document IDs must be rejected")
        camera = next(item for item in reopened["cameras"] if item["name"] == "inspection")
        client.call("set_capture_camera", {"camera_id": camera["camera_id"]})
        captures.append(client.call("capture", {"output": str(output / "edited-rgb.png")}))
        if capabilities["viewer_attached"]:
            client.call("set_viewport_camera", {"camera_id": camera["camera_id"]})
            for surface in ("viewport", "window"):
                result = client.call(
                    "capture_viewport",
                    {"surface": surface, "output": str(output / f"edited-{surface}.png")},
                )
                assert list(np.asarray(Image.open(result["path"])).shape) == result["shape"]
                captures.append(result)
        report = {
            "capabilities": capabilities,
            "transaction_schema": schema,
            "object": node,
            "visible_pixels": counts,
            "reopened_document": reopened["document"],
            "captures": captures,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/agent-control"))
    parser.add_argument(
        "--viewer", action="store_true", help="Also verify the presented viewport and window"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mojive-") as directory:
        socket_path = Path(directory) / "control.sock"
        scene = inspection_scene()
        if args.viewer:
            from mojive.composition import build_scene

            with build_scene(
                scene, width=960, height=720, vsync=False, show_window=False
            ) as viewer:
                viewer.start_rpc(socket_path)
                with ThreadPoolExecutor(max_workers=1) as worker:
                    future = worker.submit(exercise, socket_path, args.output)
                    deadline = time.monotonic() + 90
                    while not future.done() and time.monotonic() < deadline:
                        viewer.sync()
                    future.result(timeout=1)
        else:
            service = ControlService(StaticSceneAdapter(scene))
            try:
                with ControlServer(socket_path, service) as server:
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        exercise(socket_path, args.output)
                    finally:
                        server.shutdown()
                        thread.join(timeout=2)
            finally:
                service.close()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
