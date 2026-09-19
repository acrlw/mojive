"""Generate compact G1 joint recordings and exercise local FK, TCP attach, and RPC.

This diagnostic uses synthetic kinematic motion, not physics or trained policies.
The archive is compact qpos; the local bridge currently sends expanded geometry
poses over Mojive's existing snapshot transport. It does not imply native qpos
support in that protocol or in .fvs files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from mojive.adapters.joint_replay import JointReplayAdapter


def generate(model: Path, archive: Path, worlds: int, frames: int, hz: float) -> None:
    import mujoco

    from .g1_worlds import JOINT_NAMES

    if worlds < 1 or frames < 2 or not math.isfinite(hz) or hz <= 0:
        raise ValueError("Worlds, frames, and sample rate must be positive; frames must be >= 2")
    m = mujoco.MjModel.from_xml_path(str(model))
    addresses = np.array([m.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=np.intp)
    root = m.joint("floating_base_joint").qposadr[0]
    archive.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(archive / "model.mjb"))
    positions = np.lib.format.open_memmap(
        archive / "qpos.npy", mode="w+", dtype=np.float32, shape=(frames, worlds, m.nq)
    )
    phases = np.random.default_rng(42).uniform(0, 2 * np.pi, worlds)
    for i in range(frames):
        angle = 2 * np.pi * i / frames + phases
        qpos = positions[i]
        qpos[:] = m.qpos0
        # Small synthetic joint motion keeps the test independent of downloaded clips.
        qpos[:, addresses] += 0.12 * np.sin(angle[:, None] + np.arange(len(addresses)))
        qpos[:, root] += 0.12 * np.sin(angle)
        qpos[:, root + 2] += 0.04 * np.sin(2 * angle)
        yaw = 0.15 * np.sin(angle)
        qpos[:, root + 3 : root + 7] = 0
        qpos[:, root + 3] = np.cos(yaw / 2)
        qpos[:, root + 6] = np.sin(yaw / 2)
    positions.flush()
    side = math.ceil(math.sqrt(worlds))
    offsets = np.zeros((worlds, 3), np.float32)
    offsets[:, 0] = (np.arange(worlds) % side - (side - 1) / 2) * 7
    offsets[:, 1] = (np.arange(worlds) // side - (side - 1) / 2) * 7
    np.save(archive / "origins.npy", offsets)
    metadata = {
        "version": 1,
        "motion": "synthetic kinematic motion; no physics integration",
        "mujoco_version": mujoco.__version__,
        "model_sha256": hashlib.sha256((archive / "model.mjb").read_bytes()).hexdigest(),
        "shape": list(positions.shape),
        "dtype": "float32",
        "display_spacing": 7.0,
        "hz": hz,
        "coordinates": "Z-up, MuJoCo qpos, root quaternion wxyz; origins for display only",
        "joint_names": [m.joint(i).name for i in range(m.njnt)],
        "qpos_bytes_per_frame": positions[0].nbytes,
    }
    (archive / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def rss_kib(pid: int) -> int:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]))


def serve(args) -> None:
    from mojive.remote import SnapshotPublisher, handle_session_command, snapshot_structure
    from mojive.session import Session

    from .g1_worlds import scene_camera

    replay = JointReplayAdapter(
        args.archive, args.worlds, max_worlds=max(64, args.worlds), realtime=False
    )
    session = Session(replay.worlds)
    publisher = SnapshotPublisher(port=args.port)
    samples, windows = [], []
    started = time.perf_counter()
    bucket = started
    try:
        publisher.publish_structure(
            replace(snapshot_structure(session), camera_hint=scene_camera(replay, args.camera))
        )
        (args.output / "server-ready.json").write_text(
            json.dumps({"started": started, "pid": os.getpid(), **replay.metadata})
        )
        previous = -1
        while True:
            now = time.perf_counter()
            elapsed = now - started
            index = int(elapsed * replay.hz)
            publisher.pump_commands(lambda message: handle_session_command(session, message))
            if index == previous:
                time.sleep(min(0.005, max(0, (index + 1) / replay.hz - elapsed)))
                continue
            frame = replay.update(index, elapsed)
            updated = time.perf_counter()
            publisher.publish_frame(frame)
            end = time.perf_counter()
            samples.append([(updated - now) * 1000, (end - updated) * 1000, index - previous - 1])
            previous = index
            if end - bucket >= 10:
                values = np.asarray(samples)
                row = {
                    "elapsed": end - started,
                    "published_hz": len(samples) / (end - bucket),
                    "fk_ms_p50": float(np.median(values[:, 0])),
                    "publish_ms_p50": float(np.median(values[:, 1])),
                    "skipped_samples": int(values[:, 2].sum()),
                    "rss_kib": rss_kib(os.getpid()),
                    "geometry_bytes_per_frame": frame.geom_xpos.nbytes + frame.geom_xmat.nbytes,
                }
                print(json.dumps(row), flush=True)
                windows.append(row)
                (args.output / "server-windows.json").write_text(json.dumps(windows, indent=2))
                samples.clear()
                bucket = end
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
        session.release()


def _stop(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def exercise(args) -> None:
    from mojive.control.rpc import RpcClient, RpcError

    args.output.mkdir(parents=True, exist_ok=True)
    ready = args.output / "server-ready.json"
    ready.unlink(missing_ok=True)
    server = viewer = None
    cli = [sys.executable, "-m", "mojive.cli"]
    with (
        tempfile.TemporaryDirectory(prefix="mojive-replay-", dir="/tmp") as temporary,
        (args.output / "server.log").open("w") as server_log,
        (args.output / "viewer.log").open("w") as viewer_log,
    ):
        socket_path = Path(temporary) / "rpc.sock"
        report = {"camera": args.camera, "mesh_lod": args.mesh_lod, "windows": []}
        try:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    __spec__.name,
                    "serve",
                    "--archive",
                    str(args.archive),
                    "--output",
                    str(args.output),
                    "--port",
                    str(args.port),
                    "--camera",
                    args.camera,
                    "--worlds",
                    str(args.worlds),
                ],
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 90
            while not ready.exists():
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Replay bridge did not start; inspect {args.output / 'server.log'}"
                    )
                time.sleep(0.05)
            metadata = json.loads(ready.read_text())
            report["archive"] = metadata
            options = ["--enable-render", "mesh_lod"] if args.mesh_lod else []
            if args.no_vsync:
                options.append("--no-vsync")
            viewer = subprocess.Popen(
                [
                    *cli,
                    "attach",
                    "--port",
                    str(args.port),
                    "--rpc-socket",
                    str(socket_path),
                    "--title",
                    "Mojive local qpos replay",
                    *options,
                ],
                env={
                    **os.environ,
                    "MOJIVE_RENDERER": args.renderer,
                    "MOJIVE_CONFIG_DIR": temporary,
                    "MOJIVE_SETTINGS": str(Path(temporary) / "settings.json"),
                },
                stdout=viewer_log,
                stderr=subprocess.STDOUT,
            )
            with RpcClient(socket_path, timeout=20) as client:
                deadline = time.monotonic() + 90
                while True:
                    try:
                        hello = client.hello()
                        break
                    except RpcError as exc:
                        if (
                            exc.code != "connection_failed"
                            or viewer.poll() is not None
                            or time.monotonic() >= deadline
                        ):
                            raise
                        time.sleep(0.1)
                assert hello["viewer_attached"]
                report["schema"] = client.describe_operations(name="get_viewer_stats")
                report["scene"] = client.call("get_scene", {"include_objects": False})
                report["first_world"] = client.call("list_objects", {"name": "World 0", "limit": 1})
                worlds = min(args.worlds, metadata["shape"][1])
                last_world = client.call(
                    "list_objects", {"type": "model", "offset": worlds - 1, "limit": 2}
                )
                assert len(last_world) == 1 and last_world[0]["name"] == f"World {worlds - 1}"
                # A separate CLI process must reach the same window and document.
                result = subprocess.run(
                    [
                        *cli,
                        "control",
                        "get_scene",
                        "--socket",
                        str(socket_path),
                        "--json",
                        "--params",
                        '{"include_objects":false}',
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                assert json.loads(result.stdout)["document"] == report["scene"]["document"]
                report["cli_same_document"] = True
                selected = report["first_world"][0]["object_id"]
                subprocess.run(
                    [
                        *cli,
                        "control",
                        "select_object",
                        "--socket",
                        str(socket_path),
                        "--params",
                        json.dumps({"object_id": selected}),
                        "--json",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                assert client.get_state(observations=False)["selected_object_id"] == selected
                client.call("select_object", {"object_id": 0})
                previous = client.call("get_viewer_stats")
                first = previous
                start = time.perf_counter()
                print(f"Viewer ready; RPC {socket_path}", flush=True)
                while viewer.poll() is None and (
                    args.seconds == 0 or time.perf_counter() - start < args.seconds
                ):
                    time.sleep(1)
                    before = time.perf_counter()
                    stats = client.call("get_viewer_stats")
                    row = {
                        **stats,
                        "elapsed": before - start,
                        "rpc_ms": (time.perf_counter() - before) * 1000,
                        "frame_age_ms": (
                            stats["sample_time"] - metadata["started"] - stats["scene_time"]
                        )
                        * 1000,
                        "rss_kib": rss_kib(viewer.pid),
                    }
                    row["window_fps"] = (
                        stats["presented_frames"] - previous["presented_frames"]
                    ) / (stats["sample_time"] - previous["sample_time"])
                    report["windows"].append(row)
                    previous = stats
                    if len(report["windows"]) % 10 == 0:
                        print(json.dumps(row), flush=True)
                if args.mode == "check":
                    if viewer.poll() is not None:
                        raise RuntimeError("Viewer closed before the timed acceptance completed")
                    assert previous["presented_frames"] > first["presented_frames"]
                    assert previous["scene_step"] > first["scene_step"]
                if viewer.poll() is None:
                    if args.geometry_switches:
                        client.describe_operations(name="set_viewport_geometry_view")
                        report["geometry_switches"] = []
                        for cycle in range(2):
                            for view in ("collision", "both", "visual"):
                                began = time.perf_counter()
                                client.call("set_viewport_geometry_view", {"view": view})
                                switched = time.perf_counter()
                                client.call(
                                    "capture_viewport",
                                    {
                                        "output": str(
                                            (args.output / f"{view}-{cycle}.png").resolve()
                                        )
                                    },
                                )
                                complete = time.perf_counter()
                                before = client.call("get_viewer_stats")
                                time.sleep(2)
                                after = client.call("get_viewer_stats")
                                report["geometry_switches"].append(
                                    {
                                        "view": view,
                                        "cycle": cycle,
                                        "switch_rpc_ms": (switched - began) * 1000,
                                        "first_image_ms": (complete - began) * 1000,
                                        "window_fps": (
                                            after["presented_frames"] - before["presented_frames"]
                                        )
                                        / (after["sample_time"] - before["sample_time"]),
                                    }
                                )
                        client.call("set_viewport_geometry_view", {"view": "default"})
                    for surface in ("viewport", "window"):
                        client.call(
                            "capture_viewport",
                            {
                                "surface": surface,
                                "output": str((args.output / f"{surface}.png").resolve()),
                            },
                        )
                    report["rpc_stats"] = client.call("get_rpc_stats")
        except KeyboardInterrupt:
            pass
        except RpcError:
            # Closing an interactive window may race the once-per-second sample.
            if args.mode != "play" or viewer is None:
                raise
            try:
                code = viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                raise
            if code != 0:
                raise
        finally:
            _stop(viewer)
            _stop(server)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "serve", "check", "play"))
    parser.add_argument("--archive", type=Path, default=Path("output/g1-replay/archive"))
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--worlds",
        type=int,
        default=64,
        help="Worlds to generate, or maximum worlds to display from an existing archive (default: 64)",
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--port", type=int, default=47670)
    parser.add_argument("--camera", choices=("detail", "overview"), default="detail")
    parser.add_argument("--renderer", choices=("bgfx", "opengl"), default="bgfx")
    parser.add_argument("--mesh-lod", action="store_true")
    parser.add_argument("--no-vsync", action="store_true")
    parser.add_argument("--geometry-switches", action="store_true")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("output/replay-integration/local"))
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = 0 if args.mode == "play" else 60
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if args.worlds < 1:
        parser.error("--worlds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "generate":
        if args.model is None:
            parser.error("generate requires --model")
        generate(args.model, args.archive, args.worlds, args.frames, args.hz)
    elif args.mode == "serve":
        serve(args)
    else:
        exercise(args)


if __name__ == "__main__":
    main()
