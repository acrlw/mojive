"""Measure same-process physics/render overlap with the production Mojive viewer.

Production runs compare the serial and concurrent Session paths. A separate
fixed-topology experiment isolates snapshot-copy costs: paired runs preserve
every frame and physics step; realtime runs publish only the latest state.
All ImGui/GPU work remains on the main thread in both cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from mojive.simulation import Snapshot, SnapshotPool

MODES = ("serial", "serial_snapshot", "threaded")
WORKLOADS = ("joint_types", "deformables", "cloth_stress", "humanoids100")


class Trace:
    def __init__(self):
        self.events = []

    def call(self, name, function, *args, **kwargs):
        start = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            end = time.perf_counter_ns()
            self.events.append(
                (name, start, end, time.thread_time_ns() - cpu, threading.get_ident())
            )

    def durations(self, name):
        return [(end - start) / 1e6 for label, start, end, _, _ in self.events if label == name]


def interval_overlap(first, second):
    """Return wall-time overlap of two sorted, internally disjoint interval lists."""
    first, second = sorted(first), sorted(second)
    i = j = 0
    overlap = 0
    while i < len(first) and j < len(second):
        a, b = first[i], second[j]
        overlap += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return overlap


def summary(values):
    if not values:
        return None
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _xml(workload, humanoids_model=None):
    from mojive.assets import resolve

    if workload == "humanoids100":
        import mujoco

        if humanoids_model is None:
            raise ValueError(
                "humanoids100 requires --humanoids-model pointing to 100_humanoids.xml"
            )
        spec = mujoco.MjSpec.from_file(str(humanoids_model))
        spec.compile()
        return spec.to_xml()
    path = resolve("deformables" if workload == "cloth_stress" else workload)
    text = path.read_text()
    if workload == "cloth_stress":
        # A reproducible larger version of the repository's cloth, not a sleep
        # or synthetic Python workload standing in for physics computation.
        text = text.replace('count="7 7 1"', 'count="17 17 1"')
        text = text.replace('grid="6 0"', 'grid="16 0"')
    return text


def _state_digest(data):
    values = np.concatenate(([data.time], data.qpos, data.qvel, data.act))
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite physics state")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _gpu_drain(viewer, renderer):
    if renderer in {"bgfx", "wgpu"}:
        viewer.backend.target.read_color()
    else:
        viewer.backend.ctx.finish()


def run_case(args):
    import mujoco
    from imgui_bundle import imgui
    from PIL import Image

    from mojive.adapters.mujoco_adapter import MuJoCoAdapter
    from mojive.composition import build_from_adapter
    from mojive.config import LayoutConfig, ViewerConfig

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    xml = _xml(args.workload, args.humanoids_model)
    (output / "model.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.05
    mujoco.mj_forward(model, data)
    initial = mujoco.MjData(model)
    mujoco.mj_copyData(initial, model, data)
    slots = [mujoco.MjData(model) for _ in range(3)]
    pool = SnapshotPool(latest=args.schedule == "realtime")
    trace = Trace()
    stop = threading.Event()
    worker = None
    progress = [0]
    worker_cpu = [0]
    held = None

    class DisplayAdapter(MuJoCoAdapter):
        published_step = 0

        def frame(self, needs):
            result = super().frame(needs)
            result.step = self.published_step
            return result

    adapter = DisplayAdapter(external_clock=True)
    adapter.load_model(model, data)
    viewer = build_from_adapter(
        adapter,
        renderer=args.renderer,
        width=args.width,
        height=args.height,
        vsync=False,
        show_window=False,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    )
    try:
        for _ in range(args.warmup):
            viewer.sync()
        _gpu_drain(viewer, args.renderer)
        mujoco.mj_copyData(data, model, initial)
        yaw = viewer.app.camera.yaw
        frame_times, camera_latency, ages, lag_steps, displayed, gpu_ms = [], [], [], [], [], []
        frame_ends = []
        base = time.perf_counter_ns()
        main_cpu = time.thread_time_ns()
        deadline = base + int(args.seconds * 1e9)
        timestep = float(model.opt.timestep)
        batch = args.steps_per_frame

        def advance(count):
            trace.call("physics", mujoco.mj_step, model, data, nstep=count)
            progress[0] += count

        def publish():
            slot = trace.call("producer_wait", pool.reserve)
            if slot is None:
                return False
            trace.call("snapshot_copy", mujoco.mj_copyData, slots[slot], model, data)
            pool.publish(Snapshot(slot, progress[0], time.perf_counter_ns()))
            return True

        def physics_loop():
            cpu = time.thread_time_ns()
            try:
                for _ in range(args.frames if args.schedule == "paired" else 2**40):
                    if stop.is_set():
                        break
                    if args.schedule == "realtime":
                        due = base / 1e9 + (progress[0] + batch) * timestep
                        if due >= deadline / 1e9:
                            break
                        if stop.wait(max(0, due - time.perf_counter())):
                            break
                    advance(batch)
                    if not publish():
                        break
                pool.finish()
            except BaseException as exc:
                pool.finish(exc)
            finally:
                worker_cpu[0] = time.thread_time_ns() - cpu

        if args.mode == "threaded":
            worker = threading.Thread(target=physics_loop, name="mojive-physics-experiment")
            worker.start()

        frame_index = 0
        while (
            frame_index < args.frames
            if args.schedule == "paired"
            else time.perf_counter_ns() < deadline
        ):
            if args.schedule == "realtime":
                due = base / 1e9 + frame_index / args.target_fps
                time.sleep(max(0, due - time.perf_counter()))
                if time.perf_counter_ns() >= deadline:
                    break
            started = time.perf_counter_ns()
            viewer.app.camera.yaw = yaw + 8 * math.sin(frame_index * 0.05)
            if args.mode == "threaded":
                snapshot = trace.call("consumer_wait", pool.acquire, wait=args.schedule == "paired")
            else:
                count = batch
                if args.schedule == "realtime":
                    target = int((started - base) / 1e9 / timestep)
                    count = max(0, target - progress[0])
                if count:
                    advance(count)
                if args.mode == "serial_snapshot":
                    publish()
                    snapshot = pool.acquire()
                else:
                    snapshot = Snapshot(-1, progress[0], time.perf_counter_ns())
            if snapshot is not None:
                adapter.use_data(data if snapshot.slot == -1 else slots[snapshot.slot])
                if held is not None and held.slot >= 0:
                    pool.release(held.slot)
                held = snapshot
                adapter.published_step = snapshot.step
            # At startup, render an owned initial state rather than live worker data.
            elif held is None:
                adapter.use_data(initial)
            trace.call("viewer_frame", viewer.sync)
            end = time.perf_counter_ns()
            frame_times.append((end - started) / 1e6)
            camera_latency.append((end - started) / 1e6)
            frame_ends.append(end)
            displayed.append(held.step if held is not None else 0)
            if held is not None:
                ages.append((end - held.ready_ns) / 1e6)
            lag_steps.append(progress[0] - displayed[-1])
            values = viewer.backend.stats.gpu_ms
            if values:
                gpu_ms.append(sum(values.values()))
            frame_index += 1
        stop.set()
        pool.close()
        if worker is not None:
            worker.join(timeout=30)
            if worker.is_alive():
                raise RuntimeError("Physics worker did not stop")
            pool.acquire(wait=False)  # Surface a worker exception even after the last frame.
        _gpu_drain(viewer, args.renderer)
        end = time.perf_counter_ns()
        main_cpu = time.thread_time_ns() - main_cpu
        elapsed = (end - base) / 1e9
        physics_intervals = [(a, b) for n, a, b, _, _ in trace.events if n == "physics"]
        render_intervals = [(a, b) for n, a, b, _, _ in trace.events if n == "viewer_frame"]
        overlap = interval_overlap(physics_intervals, render_intervals)
        # Readback is untimed and precedes replay so the status bar is not
        # reporting the long pause spent validating the physics trajectory.
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "window.png")
        Image.fromarray(viewer.capture_array()).save(output / "scene.png")
        # Replay the same number of physics steps outside the measurement. No
        # render publication may alter integrator state, controls, or warmstarts.
        reference = mujoco.MjData(model)
        mujoco.mj_copyData(reference, model, initial)
        mujoco.mj_step(model, reference, nstep=progress[0])
        physics_digest = _state_digest(data)
        if physics_digest != _state_digest(reference):
            raise RuntimeError("Threaded execution changed the physics trajectory")
        if args.schedule == "paired" and displayed != [batch * (i + 1) for i in range(args.frames)]:
            raise RuntimeError("Paired run skipped or repeated a simulation state")
        report = {
            "mode": args.mode,
            "schedule": args.schedule,
            "workload": args.workload,
            "renderer": args.renderer,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "imgui_bundle": importlib.metadata.version("imgui-bundle"),
            "imgui": imgui.get_version(),
            "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "nv": model.nv,
            "ngeom": model.ngeom,
            "timestep": timestep,
            "requested_window": [args.width, args.height],
            "viewport_pixels": [viewer.backend.target.width, viewer.backend.target.height],
            "physics_batch": batch,
            "target_fps": args.target_fps if args.schedule == "realtime" else None,
            "elapsed_seconds": elapsed,
            "rendered_frames": frame_index,
            "render_fps": frame_index / elapsed,
            "physics_steps": progress[0],
            "physics_steps_per_second": progress[0] / elapsed,
            "simulation_seconds_per_wall_second": progress[0] * timestep / elapsed,
            "frame_ms": summary(frame_times),
            "present_interval_ms": summary([float(b - a) / 1e6 for a, b in pairwise(frame_ends)]),
            "camera_to_sync_return_ms": summary(camera_latency),
            "snapshot_age_at_sync_return_ms": summary(ages),
            "display_time_lag_ms": (
                summary(
                    [
                        ((end_ns - base) / 1e9 - step * timestep) * 1000
                        for end_ns, step in zip(frame_ends, displayed, strict=True)
                    ]
                )
                if args.schedule == "realtime"
                else None
            ),
            "state_lag_steps": summary(lag_steps),
            "unique_displayed_states": len(set(displayed)),
            "dropped_snapshots": pool.dropped,
            "cpu_core_equivalents": (main_cpu + worker_cpu[0]) / (end - base),
            "main_thread_cpu_ms": main_cpu / 1e6,
            "physics_thread_cpu_ms": worker_cpu[0] / 1e6,
            "physics_viewer_overlap_ms": overlap / 1e6,
            "physics_overlap_fraction": overlap / max(1, sum(b - a for a, b in physics_intervals)),
            "stages_ms": {
                name: summary(trace.durations(name))
                for name in (
                    "physics",
                    "snapshot_copy",
                    "producer_wait",
                    "consumer_wait",
                    "viewer_frame",
                )
            },
            "backend_gpu_ms": summary(gpu_ms),
            "physics_digest": physics_digest,
            "displayed_digest": _state_digest(adapter.data),
            "trajectory_matches_serial_replay": True,
            "window_capture": str((output / "window.png").resolve()),
            "scope": "Fixed-topology experiment using the production viewer; no live physics editing. Sync return is not photon latency; GPU timers exclude UI/presentation.",
        }
        events = [
            {
                "name": name,
                "ph": "X",
                "pid": 1,
                "tid": tid,
                "ts": (a - base) / 1000,
                "dur": (b - a) / 1000,
                "args": {"thread_cpu_us": cpu / 1000},
            }
            for name, a, b, cpu, tid in sorted(trace.events, key=lambda x: x[1])
        ]
        (output / "trace.json").write_text(json.dumps({"traceEvents": events}))
        (output / "frames.json").write_text(
            json.dumps(
                [
                    {"end_ms": (end_ns - base) / 1e6, "displayed_step": step, "work_ms": work}
                    for end_ns, step, work in zip(frame_ends, displayed, frame_times, strict=True)
                ]
            )
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        stop.set()
        pool.close()
        if worker is not None:
            worker.join(timeout=30)
        viewer.release()


def run_production_case(args):
    """Compare serial and concurrent stepping through the actual editor Session."""
    import mujoco
    from PIL import Image

    from mojive import commands as cmd
    from mojive.adapters.mujoco_adapter import MuJoCoAdapter
    from mojive.composition import build_from_adapter
    from mojive.config import LayoutConfig, ViewerConfig

    if args.mode == "serial_snapshot":
        raise ValueError("The production runtime supports serial and threaded modes")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    xml = _xml(args.workload, args.humanoids_model)
    (output / "model.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    viewer = build_from_adapter(
        adapter,
        renderer=args.renderer,
        paused=True,
        width=args.width,
        height=args.height,
        vsync=False,
        show_window=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), threaded_physics=args.mode == "threaded"
        ),
    )
    native_step = mujoco.mj_step
    trace = Trace()
    try:
        for _ in range(args.warmup):
            viewer.sync()
        # Allocate runtime buffers and warm the native path outside measurement.
        viewer.session.submit(cmd.Play())
        time.sleep(0.02)
        viewer.sync()
        viewer.session.submit(cmd.Pause())
        viewer.session.submit(cmd.Reset())
        viewer.session.submit(cmd.SetCtrlVector(np.full(model.nu, 0.05)))
        viewer.sync()
        _gpu_drain(viewer, args.renderer)
        initial = mujoco.MjData(model)
        mujoco.mj_copyData(initial, model, adapter.data)
        yaw = viewer.app.camera.yaw
        frames = []
        main_tid = threading.get_ident()
        main_cpu = time.thread_time_ns()

        def step(*arguments, **keywords):
            return trace.call("physics", native_step, *arguments, **keywords)

        mujoco.mj_step = step
        base = time.perf_counter_ns()
        viewer.app._last_time = base / 1e9
        viewer.session.submit(cmd.Play())
        while time.perf_counter_ns() < base + args.seconds * 1e9:
            due = base / 1e9 + len(frames) / args.target_fps
            time.sleep(max(0, due - time.perf_counter()))
            started = time.perf_counter_ns()
            if started >= base + args.seconds * 1e9:
                break
            viewer.app.camera.yaw = yaw + 8 * math.sin((started - base) / 1e9 * 3)
            trace.call("viewer_frame", viewer.sync)
            end = time.perf_counter_ns()
            frames.append(
                {
                    "end_ms": (end - base) / 1e6,
                    "work_ms": (end - started) / 1e6,
                    "displayed_step": viewer.session.frame.step,
                    "displayed_time": viewer.session.frame.time,
                }
            )
        viewer.session.submit(cmd.Pause())
        _gpu_drain(viewer, args.renderer)
        end = time.perf_counter_ns()
        main_cpu = time.thread_time_ns() - main_cpu
        mujoco.mj_step = native_step
        steps = viewer.session._step_counter
        elapsed = (end - base) / 1e9
        physics = [(a, b) for n, a, b, _, tid in trace.events if n == "physics" and tid != main_tid]
        render = [(a, b) for n, a, b, _, _ in trace.events if n == "viewer_frame"]
        overlap = interval_overlap(physics, render)
        worker_cpu = sum(cpu for _, _, _, cpu, tid in trace.events if tid != main_tid)
        work_times = [f["work_ms"] for f in frames]
        display_lag = [f["end_ms"] - f["displayed_time"] * 1000 for f in frames]
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "window.png")
        Image.fromarray(viewer.capture_array()).save(output / "scene.png")
        reference = mujoco.MjData(model)
        mujoco.mj_copyData(reference, model, initial)
        native_step(model, reference, nstep=steps)
        if _state_digest(reference) != _state_digest(adapter.data):
            raise RuntimeError("Production runtime changed the physics trajectory")
        report = {
            "production": True,
            "mode": args.mode,
            "schedule": "realtime",
            "workload": args.workload,
            "renderer": args.renderer,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "imgui_bundle": importlib.metadata.version("imgui-bundle"),
            "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "nbody": model.nbody,
            "nv": model.nv,
            "ngeom": model.ngeom,
            "nu": model.nu,
            "contacts_at_end": adapter.data.ncon,
            "timestep": model.opt.timestep,
            "snapshot_capacity_bytes": (
                3 * (adapter.data.nbuffer + adapter.data.narena) if args.mode == "threaded" else 0
            ),
            "viewport_pixels": [viewer.backend.target.width, viewer.backend.target.height],
            "elapsed_seconds": elapsed,
            "rendered_frames": len(frames),
            "render_fps": len(frames) / elapsed,
            "physics_steps": steps,
            "physics_steps_per_second": steps / elapsed,
            "simulation_seconds_per_wall_second": steps * model.opt.timestep / elapsed,
            "frame_ms": summary(work_times),
            "present_interval_ms": summary(
                [b["end_ms"] - a["end_ms"] for a, b in pairwise(frames)]
            ),
            "camera_to_sync_return_ms": summary(work_times),
            "display_time_lag_ms": summary(display_lag),
            "unique_displayed_states": len({f["displayed_step"] for f in frames}),
            "cpu_core_equivalents": (main_cpu + worker_cpu) / (end - base),
            "physics_viewer_overlap_ms": overlap / 1e6,
            "physics_overlap_fraction": overlap / max(1, sum(b - a for a, b in physics)),
            "stages_ms": {
                name: summary(trace.durations(name)) for name in ("physics", "viewer_frame")
            },
            "physics_digest": _state_digest(adapter.data),
            "trajectory_matches_serial_replay": True,
            "scope": "Production Session and Viewer; hidden window, VSync off. Sync return is not photon latency. CPU totals omit worker bookkeeping and driver threads.",
        }
        events = [
            {
                "name": n,
                "ph": "X",
                "pid": 1,
                "tid": tid,
                "ts": (a - base) / 1000,
                "dur": (b - a) / 1000,
            }
            for n, a, b, _, tid in sorted(trace.events, key=lambda event: event[1])
        ]
        (output / "trace.json").write_text(json.dumps({"traceEvents": events}))
        (output / "frames.json").write_text(json.dumps(frames))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        viewer.release()
        mujoco.mj_step = native_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, default="serial")
    parser.add_argument("--modes")
    parser.add_argument(
        "--production", action="store_true", help="Measure the default Session runtime"
    )
    parser.add_argument("--humanoids-model", type=Path, help="Path to MuJoCo's 100_humanoids.xml")
    parser.add_argument("--schedule", choices=("paired", "realtime"), default="paired")
    parser.add_argument("--workload", choices=WORKLOADS, default="joint_types")
    parser.add_argument("--workloads", default=",".join(WORKLOADS[:3]))
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--steps-per-frame", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--target-fps", type=float, default=120)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--output", type=Path, default=Path("output/physics-render-benchmark"))
    args = parser.parse_args(argv)
    if args.production:
        args.schedule = "realtime"
    if args.modes is None:
        args.modes = "serial,threaded" if args.production else ",".join(MODES)
    if any(
        not math.isfinite(x) or x <= 0
        for x in (
            args.frames,
            args.steps_per_frame,
            args.warmup,
            args.repeats,
            args.seconds,
            args.target_fps,
            args.width,
            args.height,
        )
    ):
        parser.error("Counts, dimensions, frequencies and durations must be positive")
    if args.worker:
        (run_production_case if args.production else run_case)(args)
        return 0
    modes, workloads = args.modes.split(","), args.workloads.split(",")
    if set(modes) - set(MODES) or set(workloads) - set(WORKLOADS):
        parser.error("Unknown mode or workload")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for workload in workloads:
        for repeat in range(args.repeats):
            for mode in modes if repeat % 2 == 0 else reversed(modes):
                directory = args.output / f"{workload}-{mode}-{repeat}"
                directory.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "mojive.tools.physics_render_benchmark",
                    "--worker",
                    "--mode",
                    mode,
                    "--workload",
                    workload,
                    "--renderer",
                    args.renderer,
                    "--schedule",
                    args.schedule,
                    "--output",
                    str(directory),
                ]
                if args.production:
                    command.append("--production")
                if args.humanoids_model is not None:
                    command.extend(["--humanoids-model", str(args.humanoids_model)])
                for name in (
                    "frames",
                    "steps_per_frame",
                    "warmup",
                    "seconds",
                    "target_fps",
                    "width",
                    "height",
                ):
                    command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
                with (directory / "worker.log").open("w") as log:
                    result = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        env=dict(os.environ, MOJIVE_RENDERER=args.renderer),
                    )
                if result.returncode:
                    raise RuntimeError(f"Benchmark failed: {directory / 'worker.log'}")
                report = json.loads((directory / "report.json").read_text())
                reports.append(report)
                print(
                    f"{workload:14} {mode:16} run={repeat} FPS={report['render_fps']:.1f} physics={report['physics_steps_per_second']:.0f}/s frame_p95={report['frame_ms']['p95']:.2f}ms overlap={report['physics_overlap_fraction']:.1%}",
                    flush=True,
                )
                (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    if args.schedule == "paired":
        for workload in workloads:
            rows = [r for r in reports if r["workload"] == workload]
            if len({r["physics_digest"] for r in rows}) != 1:
                raise RuntimeError("Modes did not produce identical final physics states")
            if len({r["displayed_digest"] for r in rows}) != 1:
                raise RuntimeError("Modes did not display identical final states")
    print((args.output / "report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
