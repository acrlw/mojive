"""Compare Viewer RPC pump policies using real sockets and injected keyboard events."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path


def _summary(values):
    import numpy as np

    if not values:
        raise RuntimeError("No benchmark samples collected")
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": max(values),
    }


def _trial(args, policy, repeat):
    from imgui_bundle import imgui
    from PIL import Image

    from mojive import InputClaim, Scene, build_scene
    from mojive.control.rpc import RpcClient, RpcLimits

    # Same server, workload and admission limits. Only the pump policy changes.
    limits = RpcLimits(
        max_connections=args.clients,
        requests_per_pump=64 if policy == "count_only" else 8,
        pump_budget_ms=60_000 if policy == "count_only" else 2.0,
    )
    start = threading.Event()
    stop = threading.Event()
    ready = [threading.Event() for _ in range(args.clients)]
    frame_ms, input_ms = [], []
    sent = None
    key_down = False

    def input_handler(context):
        if sent is not None and context.key_down("r") == key_down:
            input_ms.append((time.perf_counter() - sent) * 1000)
        return InputClaim(keys={"r"})

    scene = Scene()
    scene.box()
    with (
        tempfile.TemporaryDirectory(prefix="rpc-bench-") as temporary,
        build_scene(scene, width=args.width, height=args.height, vsync=False) as viewer,
    ):
        viewer.set_input_handler(input_handler)
        for _ in range(args.warmup):
            viewer.sync()
        server = viewer.start_rpc(Path(temporary) / "control.sock", limits=limits)

        def client_work(index):
            samples = []
            with RpcClient(server.socket_path, timeout=15) as client:
                client.hello()
                ready[index].set()
                if not start.wait(15):
                    raise TimeoutError("Benchmark start barrier timed out")
                for _ in range(args.requests):
                    if stop.is_set():
                        break
                    before = time.perf_counter()
                    client.call(args.method)
                    samples.append((time.perf_counter() - before) * 1000)
            return samples

        with ThreadPoolExecutor(max_workers=args.clients) as pool:
            futures = [pool.submit(client_work, index) for index in range(args.clients)]
            try:
                deadline = time.monotonic() + 20
                while not all(event.is_set() for event in ready):
                    for future in futures:
                        if future.done():
                            future.result()  # Preserve a failed connection's original error.
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Client connections did not become ready")
                    viewer.sync()
                begin = time.perf_counter()
                deadline = time.monotonic() + 120
                start.set()
                while not all(future.done() for future in futures):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("RPC workload did not drain")
                    key_down = not key_down
                    sent = time.perf_counter()
                    imgui.get_io().add_key_event(imgui.Key.r, key_down)
                    viewer.sync()
                    frame_ms.append((time.perf_counter() - sent) * 1000)
                elapsed = time.perf_counter() - begin
                request_ms = [sample for future in futures for sample in future.result()]
            finally:
                stop.set()
                start.set()
                # Closing transport unblocks pending clients if a trial fails.
                viewer.stop_rpc()
        stats = server.service.stats.snapshot()
        if len(input_ms) != len(frame_ms):
            raise RuntimeError("An injected keyboard event was not observed in its Viewer frame")
        sent = None
        imgui.get_io().add_key_event(imgui.Key.r, False)
        viewer.sync()
        stem = args.output / f"{args.backend}-{policy}-{repeat}"
        Image.fromarray(viewer.window.read_frame()[::-1]).save(stem.with_suffix(".png"))
        row = {
            "backend": args.backend,
            "policy": policy,
            "repeat": repeat,
            "device": viewer.backend.caps.renderer,
            "window_points": viewer.window.size_points,
            "framebuffer_pixels": viewer.window.size_pixels,
            "ui_scale": viewer.window.ui_scale,
            "limits": asdict(limits),
            "elapsed_s": elapsed,
            "requests_per_s": len(request_ms) / elapsed,
            "frame_ms": _summary(frame_ms),
            "input_dispatch_ms": _summary(input_ms),
            "request_ms": _summary(request_ms),
            "rpc_stats": stats,
            "samples": {
                "frame_ms": frame_ms,
                "input_dispatch_ms": input_ms,
                "request_ms": request_ms,
            },
        }
        stem.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in row.items() if key != "samples"}), flush=True
        )
        return {key: value for key, value in row.items() if key != "samples"}


def main(argv=None):
    """Run alternating policies serially and retain raw timing and capture evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--requests", type=int, default=30, help="Requests per client per trial")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--ui-scale", type=float, default=1.0)
    parser.add_argument(
        "--method",
        choices=("hello", "get_state", "describe_operations"),
        default="describe_operations",
    )
    parser.add_argument("--output", type=Path, default=Path("output/rpc-benchmark"))
    args = parser.parse_args(argv)
    if any(
        getattr(args, name) <= 0
        for name in ("clients", "requests", "repeats", "warmup", "width", "height")
    ):
        parser.error("Counts and dimensions must be positive")
    if args.clients > 128 or not 0 < args.ui_scale < float("inf"):
        parser.error("Use at most 128 clients and a finite positive UI scale")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(MOJIVE_RENDERER=args.backend, MOJIVE_UI_SCALE=str(args.ui_scale))
    os.environ["MOJIVE_SETTINGS"] = str(args.output.resolve() / "settings.json")
    rows = []
    for repeat in range(args.repeats):
        policies = (
            ("count_only", "time_and_count")
            if repeat % 2 == 0
            else ("time_and_count", "count_only")
        )
        for policy in policies:
            rows.append(_trial(args, policy, repeat))
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "configuration": {**vars(args), "output": str(args.output)},
        "measurement": "CPU wall time to sync return and injected ImGui key dispatch; not GPU completion or OS input latency. Same transport, comparing pump policies, not historical whole implementations. RPC stats include client hello warmup; timings retain their last 256 samples.",
        "runs": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
