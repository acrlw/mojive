"""Reproduce independent G1 dance worlds and compare renderer parity before timing.

This is a kinematic monitoring/replay workload, not a physics or training
benchmark. Motion is pinned to Unitree's Apache-2.0 source. Only CSV is read;
no downloaded Python or pickle is executed. Worlds share the original Menagerie
meshes, use seeded independent phases, and have no collision interactions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from mojive.adapters.base import FrameNeeds
from mojive.adapters.worlds import WorldInstances

MOTION_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
MOTION_URL = f"https://raw.githubusercontent.com/unitreerobotics/unitree_rl_mjlab/{MOTION_COMMIT}/src/assets/motions/g1/dance1_subject2.csv"
MOTION_SHA256 = "1793edcd8345fa4736676c06008186d1a3ecaaf99df0c380218eaa4e0de100ec"
JOINT_NAMES = (
    *(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ),
)


def motion_file(path: Path, *, download: bool = False) -> np.ndarray:
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MOTION_URL, timeout=60) as response:
            data = response.read(2_000_000)
        if hashlib.sha256(data).hexdigest() != MOTION_SHA256:
            raise ValueError("Downloaded motion checksum mismatch")
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MOTION_SHA256:
        raise ValueError("Expected the pinned Unitree dance1_subject2.csv motion")
    values = np.loadtxt(path, delimiter=",")
    if values.shape != (3945, 36) or not np.isfinite(values).all():
        raise ValueError("Motion must contain 3945 finite 36-column frames")
    return values


class Dance:
    def __init__(self, model: Path, motion: np.ndarray, count: int, spacing: float, seed: int):
        import mujoco

        from mojive.application.backends import make_adapter

        if count < 1 or not np.isfinite(spacing) or spacing < 6.5:
            raise ValueError("World count must be positive and dance spacing at least 6.5 m")
        side = math.ceil(math.sqrt(count))
        offsets = np.zeros((count, 3), np.float32)
        offsets[:, 0] = (np.arange(count) % side - (side - 1) / 2) * spacing
        offsets[:, 1] = (np.arange(count) // side - (side - 1) / 2) * spacing
        adapter = make_adapter("mujoco", model)
        try:
            source = adapter.scene_source()
            self.worlds = WorldInstances(source, adapter.frame(FrameNeeds()), offsets)
            m, d = adapter.model, adapter.data
            addresses = np.array([m.joint(name).qposadr[0] for name in JOINT_NAMES])
            root = m.joint("floating_base_joint").qposadr[0]
            # Precompute 120 Hz rigid FK from the 30 Hz capture. Root quaternions
            # use normalized shortest-arc interpolation, not linear matrix blends.
            samples = (len(motion) - 1) * 4
            slots = self.worlds.pose_indices
            self.positions = np.empty((samples, len(slots), 3), np.float32)
            self.rotations = np.empty((samples, len(slots), 3, 3), np.float32)
            for i in range(samples):
                frame, fraction = divmod(i, 4)
                a, b = motion[frame], motion[frame + 1]
                blend = fraction / 4
                d.qpos[addresses] = a[7:] * (1 - blend) + b[7:] * blend
                d.qpos[root : root + 3] = a[:3] * (1 - blend) + b[:3] * blend
                qa, qb = a[[6, 3, 4, 5]], b[[6, 3, 4, 5]]  # CSV xyzw -> MuJoCo wxyz
                q = qa * (1 - blend) + qb * (blend if np.dot(qa, qb) >= 0 else -blend)
                d.qpos[root + 3 : root + 7] = q / np.linalg.norm(q)
                mujoco.mj_kinematics(m, d)
                self.positions[i] = d.geom_xpos[slots]
                self.rotations[i] = d.geom_xmat[slots].reshape(-1, 3, 3)
            lower = self.positions.min(axis=(0, 1))
            upper = self.positions.max(axis=(0, 1))
            display_source = self.worlds.scene_source()
            display_source.scene_center = offsets.mean(axis=0) + (lower + upper) * 0.5
            display_source.scene_extent = float(
                np.linalg.norm(np.ptp(offsets, axis=0) + upper - lower) + 2
            )
            self.phases = np.random.default_rng(seed).integers(0, samples, count)
            self._positions = np.empty((count, len(slots), 3), np.float32)
            self._rotations = np.empty((count, len(slots), 3, 3), np.float32)
            self.indices = np.empty(count, np.intp)
            self.update(0.0)
        finally:
            adapter.release()

    def update(self, seconds: float):
        np.add(self.phases, int(seconds * 120), out=self.indices)
        np.remainder(self.indices, len(self.positions), out=self.indices)
        np.take(self.positions, self.indices, axis=0, out=self._positions, mode="clip")
        np.take(self.rotations, self.indices, axis=0, out=self._rotations, mode="clip")
        self.worlds.set_poses(self._positions, self._rotations, time=seconds)
        return self.worlds.frame(FrameNeeds())


def scene_camera(dance, mode):
    """Keep overview throughput distinct from a close inspection of the same worlds."""
    from mojive import CameraView

    if mode == "overview":
        return dance.worlds.camera_hint()
    offsets = dance.worlds.offsets
    center = offsets[np.argmin(np.linalg.norm(offsets, axis=1))] + np.array([0, 0, 0.8])
    return CameraView(
        eye=center + np.array([3, -4, 2.5]),
        target=center,
        near=0.01,
        far=dance.worlds.camera_hint().far,
    )


def distribution(values):
    return {
        name: float(np.percentile(values, p))
        for name, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
    }


def prepare_lod(source, args):
    lod_errors = []
    if args.mesh_ratio != 1:
        from mojive.render.mesh_processing import simplify_mesh

        meshes = {}
        for key, mesh in source.meshes.items():
            lod = simplify_mesh(mesh, ratio=args.mesh_ratio, max_error=args.mesh_error)
            meshes[key] = lod.mesh
            lod_errors.append(lod.relative_error)
        source.meshes = meshes
    return lod_errors


def worker(args):
    from PIL import Image

    from mojive import RenderProduct, SceneRenderer

    started = time.perf_counter()
    clip = motion_file(args.motion)
    dance = Dance(args.model, clip, args.count, args.spacing, args.seed)
    source = dance.worlds.scene_source()
    lod_errors = prepare_lod(source, args)
    prepared = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "backend": args.worker,
        "worlds": args.count,
        "spacing_m": args.spacing,
        "motion_sha256": MOTION_SHA256,
        "seed": args.seed,
        "resolution": [args.width, args.height],
        "samples": args.samples,
        "instances": source.instance_count,
        "shared_meshes": len(source.meshes),
        "submitted_triangles_per_scene_pass": sum(
            source.meshes[k].triangle_count if k in source.meshes else 2 for k in source.geom_mesh
        ),
        "unique_mesh_bytes": sum(
            mesh.positions.nbytes + mesh.normals.nbytes + mesh.uvs.nbytes + mesh.indices.nbytes
            for mesh in source.meshes.values()
        ),
        "prepare_s": prepared - started,
        "mode": "independent kinematic replay",
        "camera": args.camera,
        "mesh_ratio_requested": args.mesh_ratio,
        "mesh_error_limit": args.mesh_error,
        "mesh_error_max": max(lod_errors, default=0),
    }
    with SceneRenderer(
        source,
        width=args.width,
        height=args.height,
        renderer=args.worker,
        samples=args.samples,
        camera=scene_camera(dance, args.camera),
    ) as renderer:
        uploaded = time.perf_counter()
        result["renderer_setup_s"] = uploaded - prepared
        renderer.update(dance.update(0))
        image = renderer.render()
        result["first_gpu_ready_s"] = time.perf_counter() - started
        Image.fromarray(image).save(args.output / "initial.png")
        if args.capture:
            for t in (0.0, 0.375, 2.25):
                from mojive import CameraView

                frame = dance.update(t)
                positions = frame.geom_xpos[: args.count * len(dance.worlds.moving_instances)]
                lower, upper = positions.min(axis=0), positions.max(axis=0)
                center = (lower + upper) * 0.5
                distance = max(2.8, float(np.linalg.norm(upper - lower)) * 1.2)
                camera = CameraView(
                    eye=center + np.array([0, -distance * 0.8, distance * 0.6]),
                    target=center,
                    far=max(50, distance * 4),
                )
                if args.camera == "detail":
                    camera = scene_camera(dance, args.camera)
                renderer.update(frame, camera=camera)
                captures = {
                    product.name: renderer.render(product=product) for product in RenderProduct
                }
                visible = np.unique(captures[RenderProduct.OBJECT_ID.name])
                if not np.any(visible > 0):
                    raise AssertionError("Camera must show at least one dance world")
                if args.camera == "overview" and len(visible[visible > 0]) != args.count:
                    raise AssertionError(f"Camera does not show all {args.count} worlds: {visible}")
                np.savez_compressed(args.output / f"capture-{t}.npz", **captures)
                Image.fromarray(captures[RenderProduct.COLOR.name]).save(
                    args.output / f"frame-{t}.png"
                )
        else:
            values = {"replay_ms": [], "update_ms": [], "render_readback_ms": [], "total_ms": []}
            out = np.empty_like(image)
            for i in range(args.warmup + args.frames):
                a = time.perf_counter()
                frame = dance.update(i / 120)
                b = time.perf_counter()
                renderer.update(frame)
                c = time.perf_counter()
                renderer.render(out=out)
                d = time.perf_counter()
                if i >= args.warmup:
                    for key, value in zip(
                        values,
                        ((b - a) * 1000, (c - b) * 1000, (d - c) * 1000, (d - a) * 1000),
                        strict=True,
                    ):
                        values[key].append(value)
            result["milliseconds"] = {key: distribution(value) for key, value in values.items()}
            result["fps_throughput"] = 1000 / float(np.mean(values["total_ms"]))
            Image.fromarray(out).save(args.output / "final.png")
            result["backend_draw_calls"] = renderer._backend.stats.draw_calls
            result["draw_call_scope"] = (
                "all native passes" if args.worker == "bgfx" else "OpenGL color submission buckets"
            )
            if args.worker == "bgfx":
                stats = renderer._backend.target.frame.statistics
                result["gpu_ms_last"] = stats.gpu_ms if stats.gpu_ms >= 0 else None
                result["instance_upload_bytes_last"] = stats.upload_bytes
                result["shadow_instances"] = stats.shadow_instances
                result["culled_shadow_instances"] = stats.culled_shadow_instances
                result["culled_instances"] = stats.culled_instances
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def parity(output, counts):
    from PIL import Image

    from mojive import RenderProduct
    from mojive.tools.native_parity import compare

    report = []
    for count in counts:
        for t in (0.0, 0.375, 2.25):
            captures = []
            for backend in ("opengl", "bgfx"):
                with np.load(
                    output / f"parity-{count}-{backend}" / f"capture-{t}.npz", allow_pickle=False
                ) as data:
                    captures.append({p: data[p.name] for p in RenderProduct})
            row = {"worlds": count, "time": t, **compare(*captures)}
            report.append(row)
            Image.fromarray(
                np.concatenate([v[RenderProduct.COLOR] for v in captures], axis=1)
            ).save(output / f"parity-{count}-{t}.png")
            assert (
                row["color_mean"] < 2
                and row["id_disagreement"] < 0.01
                and row["segmentation_disagreement"] < 0.01
            ), row
    (output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def show(args):
    from mojive import build_from_adapter

    dance = Dance(
        args.model,
        motion_file(args.motion, download=args.download),
        args.count,
        args.spacing,
        args.seed,
    )
    prepare_lod(dance.worlds.scene_source(), args)
    with build_from_adapter(
        dance.worlds,
        renderer=args.renderer,
        width=args.width,
        height=args.height,
        samples=args.samples,
        title=f"G1 dance · {args.count} independent worlds",
    ) as viewer:
        viewer.app.camera.adopt(scene_camera(dance, args.camera))
        started = time.perf_counter()
        while viewer.is_running():
            dance.update(time.perf_counter() - started)
            viewer.sync()
            if args.duration and time.perf_counter() - started >= args.duration:
                break
        if args.capture:
            args.output.mkdir(parents=True, exist_ok=True)
            viewer.capture(args.output / "viewer.png", surface="window")


def publish(args):
    from mojive.remote import SnapshotPublisher, snapshot_structure
    from mojive.session import Session

    dance = Dance(args.model, motion_file(args.motion), args.count, args.spacing, args.seed)
    prepare_lod(dance.worlds.scene_source(), args)
    session = Session(dance.worlds)
    publisher = SnapshotPublisher(port=args.port)
    timings = []
    stop = threading.Event()

    def wait_for_parent():
        sys.stdin.readline()
        stop.set()

    threading.Thread(target=wait_for_parent, daemon=True).start()
    try:
        publisher.publish_structure(snapshot_structure(session))
        started = time.perf_counter()
        frame = dance.update(0)
        frame.time = started
        publisher.publish_frame(frame)
        print(json.dumps({"ready": True}), flush=True)
        index = 0
        while not stop.is_set() and time.perf_counter() - started < args.duration + 60:
            now = time.perf_counter()
            frame = dance.update(now - started)
            frame.time = now  # Shared monotonic clock across local processes.
            before = time.perf_counter()
            publisher.publish_frame(frame)
            timings.append((time.perf_counter() - before) * 1000)
            index += 1
            time.sleep(max(0, started + index / 120 - time.perf_counter()))
        print(
            json.dumps(
                {
                    "published": index,
                    "elapsed_s": time.perf_counter() - started,
                    "publish_ms": distribution(timings),
                }
            ),
            flush=True,
        )
    finally:
        publisher.close()
        session.release()


def transport(args):
    from mojive.remote import RemoteSceneAdapter

    # A bounded localhost benchmark owns its server; never attach to another session.
    for port in range(49100, 50100, 2):
        with socket.socket() as first, socket.socket() as second:
            try:
                first.bind(("127.0.0.1", port))
                second.bind(("127.0.0.1", port + 1))
                break
            except OSError:
                continue
    else:
        raise RuntimeError("No free loopback port pair")
    command = [
        sys.executable,
        "-m",
        "mojive.tools.g1_worlds",
        "--publish",
        "--port",
        str(port),
        "--model",
        str(args.model),
        "--motion",
        str(args.motion),
        "--count",
        str(args.count),
        "--spacing",
        str(args.spacing),
        "--seed",
        str(args.seed),
        "--duration",
        str(args.duration or 8),
        "--mesh-ratio",
        str(args.mesh_ratio),
        "--mesh-error",
        str(args.mesh_error),
    ]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    adapter = renderer = None
    try:
        if child.stdout.readline().strip() != '{"ready": true}':
            raise RuntimeError("Publisher failed before readiness")
        connected = time.perf_counter()
        adapter = RemoteSceneAdapter(port=port, timeout=15)
        frame = adapter.frame(FrameNeeds())
        connect_ms = (time.perf_counter() - connected) * 1000
        render_times, completed_ages = [], []
        setup_ms = None
        image = None
        if args.monitor:
            from mojive import SceneRenderer

            ready = time.perf_counter()
            renderer = SceneRenderer(
                adapter.scene_source(),
                renderer=args.renderer,
                width=args.width,
                height=args.height,
                samples=args.samples,
                camera=adapter.camera_hint(),
            )
            renderer.update(frame)
            image = renderer.render()
            setup_ms = (time.perf_counter() - ready) * 1000
        first_step, previous_step = frame.step, frame.step
        ages, reads, received = [], [], 0
        started = time.perf_counter()
        while time.perf_counter() - started < (args.duration or 8):
            before = time.perf_counter()
            frame = adapter.frame(FrameNeeds())
            now = time.perf_counter()
            reads.append((now - before) * 1000)
            if frame.step != previous_step:
                assert frame.step > previous_step
                ages.append((now - frame.time) * 1000)
                received += 1
                previous_step = frame.step
            if renderer is not None:
                render_started = time.perf_counter()
                renderer.update(frame)
                image = renderer.render()
                done = time.perf_counter()
                render_times.append((done - render_started) * 1000)
                completed_ages.append((done - frame.time) * 1000)
            time.sleep(max(0, started + len(reads) / args.receive_hz - time.perf_counter()))
        assert received and frame.step > first_step
        result = {
            "worlds": args.count,
            "mode": "two-process TCP loopback"
            + (", completed GPU frames" if args.monitor else ", no rendering"),
            "receive_target_hz": args.receive_hz,
            "receive_observed_hz": len(reads) / (time.perf_counter() - started),
            "unique_frames": received,
            "coalesced_frames": frame.step - first_step - received,
            "snapshot_age_ms": distribution(ages),
            "read_ms": distribution(reads),
            "connect_structure_first_frame_ms": connect_ms,
            "pose_bytes": frame.geom_xpos.nbytes + frame.geom_xmat.nbytes,
        }
        if renderer is not None:
            from PIL import Image

            result.update(
                backend=args.renderer,
                render_ms=distribution(render_times),
                completed_frame_age_ms=distribution(completed_ages),
                renderer_first_frame_ms=setup_ms,
                mesh_ratio_requested=args.mesh_ratio,
                mesh_error_limit=args.mesh_error,
                render_frames=len(render_times),
            )
            renderer.close()
            renderer = None
            args.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(args.output / f"monitor-{args.count}-{args.renderer}.png")
        adapter.release()
        adapter = None
        child.stdin.write("stop\n")
        child.stdin.flush()
        result["publisher"] = json.loads(child.stdout.readline())
        assert child.wait(timeout=10) == 0
        args.output.mkdir(parents=True, exist_ok=True)
        name = (
            f"monitor-{args.count}-{args.renderer}" if args.monitor else f"transport-{args.count}"
        )
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        if adapter is not None:
            adapter.release()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        child.stdout.close()
        child.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--motion", type=Path, default=Path("output/g1-dance/source/dance1_subject2.csv")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/g1-worlds"))
    parser.add_argument("--counts", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--parity-counts", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument(
        "--camera",
        choices=("overview", "detail"),
        default="overview",
        help="Camera for local rendering and viewer; detail keeps all worlds loaded",
    )
    parser.add_argument("--spacing", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--mesh-ratio", type=float, default=1.0)
    parser.add_argument("--mesh-error", type=float, default=0.005)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--worker", choices=("opengl", "bgfx"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--renderer", choices=("opengl", "bgfx"), default="bgfx")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument(
        "--monitor", action="store_true", help="Render snapshots from a separate TCP publisher"
    )
    parser.add_argument("--receive-hz", type=float, default=30)
    parser.add_argument("--publish", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=49100, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.camera != "overview" and (args.monitor or args.publish or args.transport_only):
        parser.error("The detail camera applies to local rendering and viewer modes")
    if args.frames < 1 or args.warmup < 0 or args.duration < 0 or args.receive_hz <= 0:
        parser.error("Frame counts, duration, and receive rate must be valid")
    if args.view:
        show(args)
        return
    if args.publish:
        publish(args)
        return
    if args.transport_only or args.monitor:
        motion_file(args.motion, download=args.download)
        for count in args.counts:
            args.count = count
            transport(args)
        return
    if args.worker:
        worker(args)
        return
    motion_file(args.motion, download=args.download)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for capture, counts in ((True, args.parity_counts), (False, args.counts)):
        if not capture:
            parity(args.output, args.parity_counts)
            if args.parity_only:
                break
        for count in counts:
            for backend in ("opengl", "bgfx"):
                target = args.output / f"{'parity' if capture else 'bench'}-{count}-{backend}"
                command = [
                    sys.executable,
                    "-m",
                    "mojive.tools.g1_worlds",
                    "--worker",
                    backend,
                    "--model",
                    str(args.model),
                    "--motion",
                    str(args.motion),
                    "--count",
                    str(count),
                    "--output",
                    str(target),
                ]
                for key in (
                    "width",
                    "height",
                    "samples",
                    "spacing",
                    "seed",
                    "frames",
                    "warmup",
                    "mesh-ratio",
                    "mesh-error",
                    "camera",
                ):
                    command += [f"--{key}", str(getattr(args, key.replace("-", "_")))]
                if capture:
                    command.append("--capture")
                subprocess.run(command, check=True, timeout=1800)
                rows.append(json.loads((target / "report.json").read_text()))
                (args.output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
