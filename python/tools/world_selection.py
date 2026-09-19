"""Bounded CLI/RPC acceptance for local joint replay; at most four displayed worlds."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def check(archive: Path, output: Path) -> None:
    """Open the real CLI viewer, select original IDs, capture, and always shut down."""
    from mojive.control.rpc import RpcClient, RpcError

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mj-worlds-", dir="/tmp") as temporary:
        socket = Path(temporary) / "control.sock"
        cli = [sys.executable, "-m", "mojive.cli"]
        env = {
            **os.environ,
            "MOJIVE_SETTINGS": str(Path(temporary) / "settings.json"),
            "MOJIVE_LANGUAGE": "zh_CN",
        }
        with (output / "viewer.log").open("w") as log:
            viewer = subprocess.Popen(
                [
                    *cli,
                    "replay-joints",
                    str(archive),
                    "--worlds",
                    "2",
                    "--world-limit",
                    "4",
                    "--rpc-socket",
                    str(socket),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                with RpcClient(socket, timeout=10) as client:
                    deadline = time.monotonic() + 30
                    while not socket.exists():
                        if viewer.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(f"Viewer did not start; see {output / 'viewer.log'}")
                        time.sleep(0.05)
                    assert client.hello()["viewer_attached"]
                    scene = client.call("get_scene", {"include_objects": False})
                    for name in (
                        "get_world_selection",
                        "set_world_selection",
                        "capture_viewport",
                        "get_viewer_stats",
                        "set_viewport_geometry_view",
                        "list_objects",
                        "set_replay_playback",
                    ):
                        client.describe_operations(name=name)
                    initial = client.call("get_world_selection")
                    assert initial["max_worlds"] <= 4
                    last = initial["total_worlds"] - 1
                    ids = [last, 0] if last else [0]
                    client.call("set_world_selection", {"world_ids": ids})
                    assert client.call("get_world_selection")["world_ids"] == ids
                    objects = client.call("list_objects", {"type": "model"})
                    assert [item["name"] for item in objects] == [f"World {i}" for i in ids]
                    try:
                        client.call("set_world_selection", {"world_ids": [0] * 5})
                    except RpcError:
                        pass
                    else:
                        raise AssertionError("Invalid selection was accepted")
                    assert client.call("get_world_selection")["world_ids"] == ids
                    client.call(
                        "capture_viewport",
                        {
                            "surface": "window",
                            "output": str((output / "selected-worlds.png").resolve()),
                        },
                    )
                    report = {"initial": initial, "selected": ids, "geometry_views": []}
                    for view in ("collision", "both", "visual"):
                        started = time.perf_counter()
                        client.call("set_viewport_geometry_view", {"view": view})
                        report["geometry_views"].append(
                            {
                                "view": view,
                                "rpc_ms": (time.perf_counter() - started) * 1000,
                            }
                        )
                        client.call(
                            "capture_viewport",
                            {
                                "output": str((output / f"{view}.png").resolve()),
                            },
                        )
                    process = subprocess.run(
                        [
                            *cli,
                            "control",
                            "set_world_selection",
                            "--socket",
                            str(socket),
                            "--params",
                            json.dumps({"world_ids": [last]}),
                            "--json",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    assert json.loads(process.stdout)["ok"]
                    assert client.call("get_world_selection")["world_ids"] == [last]
                    assert (
                        client.call("get_scene", {"include_objects": False})["document"]
                        == scene["document"]
                    )
                    client.call("set_replay_playback", {"paused": False})
                    before = client.call("get_viewer_stats")
                    time.sleep(0.2)
                    after = client.call("get_viewer_stats")
                    assert after["scene_step"] > before["scene_step"]
                    assert after["presented_frames"] > before["presented_frames"]
                    report.update(cli_same_document=True, replay_advances=True)
                    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            finally:
                if viewer.poll() is None:
                    viewer.send_signal(signal.SIGINT)
                    try:
                        viewer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        viewer.kill()
                        viewer.wait(timeout=5)
    print(f"World selection verified; viewer closed. Results: {output}")


def main() -> None:
    """Run a small acceptance check against an existing diagnostic archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/world-selection"))
    args = parser.parse_args()
    check(args.archive, args.output)


if __name__ == "__main__":
    main()
