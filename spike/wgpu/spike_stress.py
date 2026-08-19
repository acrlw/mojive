"""Spike B: wgpu-py robustness and readback-cost probes.

Each subcommand is meant to run in its own process (crash isolation):

  lifecycle   - create/destroy devices and full backend state, watch RSS
  readback    - render + read_texture timing at three resolutions
  threads     - concurrent read_texture from a worker while the main thread renders
  callbacks   - exercise the request_device callback path (issue #824 territory)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wgpu

from backend import WgpuSceneBackend
from spike_offscreen import build_scene


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def median(xs):
    return float(np.median(xs))


def cmd_lifecycle() -> None:
    # Full device re-request, the session-restart path.
    t0 = time.perf_counter()
    for i in range(50):
        device = wgpu.utils.get_default_device()
        del device
    print(f"device request x50: {time.perf_counter() - t0:.2f}s, rss={rss_mb():.0f} MB")

    # Backend (pipeline + targets + buffers) churn on a shared device.
    adapter, source, scene = build_scene()
    from forge_viewer.render.mesh import all_builtin

    meshes = {**all_builtin(), **source.meshes}
    r0 = rss_mb()
    t0 = time.perf_counter()
    for i in range(200):
        backend = WgpuSceneBackend(640, 480)
        backend.set_meshes(meshes)
        backend.draw_scene(scene)
        if i == 0:
            backend.read_color()  # force sync so destruction isn't deferred
        backend.release()
        del backend
    dt = time.perf_counter() - t0
    print(f"backend create/render/destroy x200: {dt:.2f}s ({dt / 200 * 1e3:.1f} ms/cycle)")
    print(f"rss before={r0:.0f} MB after={rss_mb():.0f} MB")
    adapter.release()


def cmd_readback() -> None:
    adapter, source, scene = build_scene()
    from forge_viewer.render.mesh import all_builtin

    meshes = {**all_builtin(), **source.meshes}
    for width, height in ((640, 480), (1280, 720), (1920, 1080)):
        scene.camera = scene.camera.with_aspect(width / height)
        backend = WgpuSceneBackend(width, height)
        backend.set_meshes(meshes)
        submit_ms, read_rgb_ms, read_all_ms, e2e_ms = [], [], [], []
        for i in range(110):
            t0 = time.perf_counter()
            backend.draw_scene(scene)
            t1 = time.perf_counter()
            rgb = backend.read_color()
            t2 = time.perf_counter()
            backend.read_linear_depth()
            backend.read_ids()
            t3 = time.perf_counter()
            if i >= 10:  # warmup
                submit_ms.append((t1 - t0) * 1e3)
                read_rgb_ms.append((t2 - t1) * 1e3)
                read_all_ms.append((t3 - t2) * 1e3)
                e2e_ms.append((t3 - t0) * 1e3)
        del rgb
        mpx = width * height / 1e6
        print(
            f"{width}x{height}: submit={median(submit_ms):.2f} ms, "
            f"read_rgb={median(read_rgb_ms):.2f} ms ({mpx * 4 / median(read_rgb_ms):.0f} GB/s), "
            f"read_depth+ids={median(read_all_ms):.2f} ms, e2e={median(e2e_ms):.2f} ms"
        )
        backend.release()
        del backend
    adapter.release()


def cmd_threads() -> None:
    adapter, source, scene = build_scene()
    from forge_viewer.render.mesh import all_builtin

    backend = WgpuSceneBackend(640, 480)
    backend.set_meshes({**all_builtin(), **source.meshes})
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                backend.read_color()
        except BaseException as exc:  # noqa: BLE001 - record anything, including native faults
            errors.append(exc)

    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    for i in range(200):
        backend.draw_scene(scene)
        backend.read_linear_depth()
    stop.set()
    worker.join(timeout=10)
    print(f"main-thread render x200 + concurrent reader: alive={not worker.is_alive()}")
    print(f"errors: {[type(e).__name__ + ': ' + str(e) for e in errors] or 'none'}")
    adapter.release()
    backend.release()


def cmd_callbacks() -> None:
    # The request path with explicit callbacks is where 0.32.0 had ABI trouble.
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    print(f"adapter ok: {adapter.info.device}")

    def on_device_lost(device, reason, message):
        print(f"device lost callback fired: {reason} {message}")

    device = adapter.request_device_sync()
    try:
        device.set_device_lost_callback(on_device_lost)
    except Exception as exc:
        print(f"set_device_lost_callback unsupported: {type(exc).__name__}: {exc}")
    # Uncaptured-error callback: push an intentional validation error through.
    device.push_error_scope("validation")
    device.create_buffer(size=-1, usage=wgpu.BufferUsage.VERTEX)  # invalid
    device.pop_error_scope()  # must not raise to be a usable diagnostics path
    print("error scope round-trip ok")


if __name__ == "__main__":
    {"lifecycle": cmd_lifecycle, "readback": cmd_readback, "threads": cmd_threads, "callbacks": cmd_callbacks}[
        sys.argv[1]
    ]()
