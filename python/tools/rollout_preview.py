"""Bounded HTTP/CLI/RPC acceptance using two selected worlds from a joint archive."""

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

import numpy as np


def check(archive: Path, output: Path) -> None:
    """Exercise manual windows, local playback and repeated sync, then close the viewer."""
    from mojive.control.rpc import RpcClient
    from mojive.remote.rollout import RolloutServer, RolloutStore

    output.mkdir(parents=True, exist_ok=True)
    store = RolloutStore.from_archive(archive, max_worlds=4)
    total = store.info()["total_worlds"]
    ids = [total - 1, 0] if total > 1 else [0]
    values = np.array(np.load(archive / "qpos.npy", mmap_mode="r")[-32:, ids, :])
    cli = [sys.executable, "-m", "mojive.cli"]
    report = {"total_worlds": total, "selected": ids, "syncs": [], "geometry_views": []}
    with (
        RolloutServer(store, port=0) as server,
        tempfile.TemporaryDirectory(prefix="mj-rollout-", dir="/tmp") as temporary,
    ):
        socket = Path(temporary) / "control.sock"
        env = {**os.environ, "MOJIVE_SETTINGS": str(Path(temporary) / "settings.json")}
        with (output / "viewer.log").open("w") as log:
            viewer = subprocess.Popen(
                [
                    *cli,
                    "replay-joints",
                    f"http://127.0.0.1:{server.address[1]}",
                    "--world-ids",
                    *map(str, ids),
                    "--world-limit",
                    "4",
                    "--window-frames",
                    "32",
                    "--rpc-socket",
                    str(socket),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while not socket.exists():
                    if viewer.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Viewer did not start; see {output / 'viewer.log'}")
                    time.sleep(0.05)
                with RpcClient(socket, timeout=10) as client:
                    assert client.hello()["viewer_attached"]
                    for name in (
                        "get_replay_info",
                        "set_replay_playback",
                        "seek_replay",
                        "get_rollout_sync",
                        "sync_rollout",
                        "get_world_selection",
                    ):
                        client.describe_operations(name=name)
                    assert client.call("get_replay_info")["paused"]
                    document = client.call("get_scene", {"include_objects": False})["document"]

                    def sync_done():
                        until = time.monotonic() + 5
                        while time.monotonic() < until:
                            info = client.call("get_rollout_sync")
                            if not info["pending"]:
                                assert not info["error"], info
                                return info
                            time.sleep(0.01)
                        raise AssertionError("Rollout download did not complete")

                    def rss():
                        return (
                            int(
                                subprocess.check_output(
                                    ["ps", "-o", "rss=", "-p", str(viewer.pid)], text=True
                                ).strip()
                            )
                            * 1024
                        )

                    initial_bytes = store.sent_pose_bytes
                    client.call("sync_rollout")
                    sync_done()
                    assert store.sent_pose_bytes == initial_bytes
                    requests = store.window_requests
                    client.call("set_replay_playback", {"paused": False})
                    before = client.call("get_viewer_stats")
                    time.sleep(0.5)
                    after = client.call("get_viewer_stats")
                    client.call("set_replay_playback", {"paused": True})
                    assert store.window_requests == requests
                    assert after["presented_frames"] > before["presented_frames"]
                    report["window_fps"] = (
                        after["presented_frames"] - before["presented_frames"]
                    ) / (after["sample_time"] - before["sample_time"])
                    process = subprocess.run(
                        [
                            *cli,
                            "control",
                            "seek_replay",
                            "--socket",
                            str(socket),
                            "--params",
                            '{"frame":5}',
                            "--json",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    assert json.loads(process.stdout)["ok"]
                    assert client.call("get_replay_info")["frame_index"] == 5
                    for view in ("visual", "collision", "both", "visual"):
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
                        report["geometry_views"][-1]["capture_complete_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                    report["rss_before_bytes"] = rss()
                    for iteration in range(16):
                        values[..., 0] += 0.005
                        old = client.call("get_rollout_sync")["revision"]
                        revision = store.publish(values, start_step=iteration * 32, world_ids=ids)
                        assert client.call("get_rollout_sync")["revision"] == old
                        start = time.perf_counter()
                        client.call("sync_rollout")
                        admitted = time.perf_counter()
                        assert sync_done()["revision"] == revision
                        assert client.call("get_replay_info")["paused"]
                        report["syncs"].append(
                            {
                                "request_ms": (admitted - start) * 1000,
                                "complete_ms": (time.perf_counter() - start) * 1000,
                                "rss_bytes": rss(),
                            }
                        )
                    report["rss_after_bytes"] = rss()
                    assert client.call("get_world_selection")["world_ids"] == ids
                    assert (
                        client.call("get_scene", {"include_objects": False})["document"] == document
                    )
                    client.call("set_panel", {"id": "output", "open": False})
                    client.call("set_panel", {"id": "keyframes", "open": True})
                    client.call(
                        "capture_viewport",
                        {
                            "surface": "window",
                            "output": str((output / "preview.png").resolve()),
                        },
                    )
                    report.update(
                        pose_bytes_per_window=values.nbytes,
                        total_pose_bytes=store.sent_pose_bytes,
                        window_requests=store.window_requests,
                        unchanged_sync_pose_bytes=0,
                        local_playback_network_requests=0,
                        cli_seek_verified=True,
                    )
                    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            finally:
                if viewer.poll() is None:
                    viewer.send_signal(signal.SIGINT)
                    try:
                        viewer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        viewer.kill()
                        viewer.wait(timeout=5)
    print(f"Rollout preview verified; viewer and server closed. Results: {output}")


def main() -> None:
    """Run a two-world acceptance check against an existing diagnostic archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/rollout-preview"))
    args = parser.parse_args()
    check(args.archive, args.output)


if __name__ == "__main__":
    main()
