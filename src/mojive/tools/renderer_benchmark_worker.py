"""Run one isolated public Renderer API benchmark case."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def _document(width: int, height: int, samples: int, body: str, *, asset: str = "") -> str:
    return f"""
<mujoco model="renderer_benchmark">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002"/>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <quality offsamples="{samples}"/>
  </visual>
  {asset}
  <worldbody>
    <light pos="-3 -4 7" diffuse=".8 .8 .8"/>
    <camera name="benchmark" pos="0 -12 8" xyaxes="1 0 0 0 .555 .832"/>
    <geom type="plane" size="12 12 .1" rgba=".12 .16 .2 1"/>
    {body}
  </worldbody>
</mujoco>
"""


def _primitives(width: int, height: int, samples: int) -> str:
    geoms = []
    kinds = ("sphere", "box", "capsule", "cylinder")
    colors = ((0.15, 0.55, 0.9), (0.9, 0.3, 0.12), (0.2, 0.8, 0.35), (0.8, 0.65, 0.15))
    for index in range(16):
        x = (index % 4 - 1.5) * 1.35
        y = (index // 4 - 1.5) * 1.35
        kind = kinds[index % len(kinds)]
        r, g, b = colors[index % len(colors)]
        size = ".38" if kind == "sphere" else ".32 .32 .38"
        if kind in {"capsule", "cylinder"}:
            size = ".25 .45"
        geoms.append(
            f'<geom type="{kind}" pos="{x:.3f} {y:.3f} .55" size="{size}" rgba="{r} {g} {b} 1"/>'
        )
    return _document(width, height, samples, "\n".join(geoms))


def _many_objects(width: int, height: int, samples: int) -> str:
    bodies = []
    side = 16
    for index in range(side * side):
        x = (index % side - (side - 1) / 2) * 0.62
        y = (index // side - (side - 1) / 2) * 0.62
        kind = "sphere" if index % 2 else "box"
        size = ".22" if kind == "sphere" else ".2 .2 .2"
        hue = index % 5
        colors = (
            ".15 .55 .9 1",
            ".9 .3 .12 1",
            ".2 .8 .35 1",
            ".8 .65 .15 1",
            ".65 .3 .8 1",
        )
        bodies.append(
            f'<body pos="{x:.3f} {y:.3f} .3"><geom type="{kind}" size="{size}" '
            f'rgba="{colors[hue]}"/></body>'
        )
    return _document(width, height, samples, "\n".join(bodies))


def _dense_mesh(width: int, height: int, samples: int) -> str:
    side = 49
    vertices = []
    for row in range(side):
        y = (row / (side - 1) - 0.5) * 8.0
        for column in range(side):
            x = (column / (side - 1) - 0.5) * 8.0
            z = 0.18 * math.sin(x * 2.2) * math.cos(y * 1.7) + 0.2
            vertices.append(f"{x:.4f} {y:.4f} {z:.4f}")
    faces = []
    for row in range(side - 1):
        for column in range(side - 1):
            a = row * side + column
            b = a + 1
            c = a + side
            d = c + 1
            faces.extend((f"{a} {b} {d}", f"{a} {d} {c}"))
    asset = (
        '<asset><mesh name="terrain" vertex="'
        + " ".join(vertices)
        + '" face="'
        + " ".join(faces)
        + '"/></asset>'
    )
    body = '<geom type="mesh" mesh="terrain" rgba=".25 .62 .82 1"/>'
    return _document(width, height, samples, body, asset=asset)


def _dynamic(width: int, height: int, samples: int) -> str:
    return _dynamic_grid(width, height, samples, side=8, spacing=0.9, size=0.3)


def _dynamic_large(width: int, height: int, samples: int) -> str:
    return _dynamic_grid(width, height, samples, side=32, spacing=0.35, size=0.14)


def _dynamic_grid(
    width: int,
    height: int,
    samples: int,
    *,
    side: int,
    spacing: float,
    size: float,
) -> str:
    bodies = []
    for index in range(side * side):
        x = (index % side - (side - 1) / 2) * spacing
        y = (index // side - (side - 1) / 2) * spacing
        bodies.append(
            f'<body pos="{x:.3f} {y:.3f} .45"><freejoint/>'
            f'<geom type="box" size="{size} {size} {size}" rgba=".3 .7 .9 1"/></body>'
        )
    return _document(width, height, samples, "\n".join(bodies))


def _materials(width: int, height: int, samples: int) -> str:
    asset = """
<asset>
  <texture name="checker" type="2d" builtin="checker" rgb1=".1 .2 .35" rgb2=".8 .85 .9"
           width="256" height="256"/>
  <material name="checker" texture="checker" texrepeat="3 3" reflectance=".18"/>
  <material name="transparent" rgba=".25 .75 .95 .42"/>
</asset>
"""
    body = """
<geom type="box" pos="-1.2 0 .55" size=".75 .75 .55" material="checker"/>
<geom type="sphere" pos="1.2 0 .7" size=".7" rgba=".9 .35 .12 1"/>
<geom type="capsule" pos="0 1.5 .7" size=".35 .8" euler="0 90 0"
      material="transparent"/>
<geom type="cylinder" pos="0 -1.5 .6" size=".55 .6" rgba=".25 .8 .35 1"/>
"""
    return _document(width, height, samples, body, asset=asset)


_WORKLOADS = {
    "primitives": _primitives,
    "many_objects": _many_objects,
    "dense_mesh": _dense_mesh,
    "dynamic": _dynamic,
    "dynamic_large": _dynamic_large,
    "materials": _materials,
}


def _rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _distribution(values: list[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def _byte_distribution(values: list[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(samples, 95)),
        "mean": float(np.mean(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def _make_renderer(name: str, model, width: int, height: int):
    if name == "mujoco":
        import mujoco

        return mujoco.Renderer(model, width=width, height=height)
    from ..renderer import Renderer

    return Renderer(model, width=width, height=height)


def _output(mode: str, width: int, height: int) -> np.ndarray:
    if mode == "depth":
        return np.empty((height, width), dtype=np.float32)
    if mode == "segmentation":
        return np.empty((height, width, 2), dtype=np.int32)
    return np.empty((height, width, 3), dtype=np.uint8)


def _configure_mode(renderer, mode: str) -> None:
    if mode == "depth":
        renderer.enable_depth_rendering()
    elif mode == "segmentation":
        renderer.enable_segmentation_rendering()


def _render(renderer, mode: str, output: np.ndarray) -> np.ndarray:
    # MuJoCo 3.11 validates segmentation ``out`` as an RGB array even though
    # render() returns (height, width, 2) object/type pairs. Use the allocation
    # path for both implementations so this public-API comparison stays fair.
    if mode == "segmentation":
        return renderer.render()
    return renderer.render(out=output)


def _animate(mujoco, workload: str, model, data, base_qpos: np.ndarray, frame: int) -> None:
    if workload in {"dynamic", "dynamic_large"}:
        qpos = data.qpos.reshape(-1, 7)
        base = base_qpos.reshape(-1, 7)
        phase = np.arange(qpos.shape[0], dtype=np.float64) * 0.17 + frame * 0.035
        qpos[:, 2] = base[:, 2] + 0.12 * np.sin(phase)
    mujoco.mj_forward(model, data)


def _renderer_detail(renderer, name: str) -> str:
    if name == "mujoco":
        return "mujoco.Renderer"
    backend = getattr(renderer, "_backend", None)
    describe = getattr(backend, "describe", None)
    if callable(describe):
        return str(describe())
    return type(backend).__name__ if backend is not None else name


def run(args: argparse.Namespace) -> dict[str, object]:
    import mujoco

    source = _WORKLOADS[args.workload](args.width, args.height, args.samples)
    compile_start = time.perf_counter()
    model = mujoco.MjModel.from_xml_string(source)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    base_qpos = data.qpos.copy()

    rss_before = _rss_mb()
    construct_start = time.perf_counter()
    renderer = _make_renderer(args.renderer, model, args.width, args.height)
    constructor_ms = (time.perf_counter() - construct_start) * 1000.0
    try:
        detail = _renderer_detail(renderer, args.renderer)
        _configure_mode(renderer, args.mode)
        output = _output(args.mode, args.width, args.height)

        _animate(mujoco, args.workload, model, data, base_qpos, -1)
        first_start = time.perf_counter()
        renderer.update_scene(data, camera="benchmark")
        output = _render(renderer, args.mode, output)
        first_frame_ms = (time.perf_counter() - first_start) * 1000.0

        for frame in range(args.warmup):
            _animate(mujoco, args.workload, model, data, base_qpos, frame)
            renderer.update_scene(data, camera="benchmark")
            output = _render(renderer, args.mode, output)

        update_ms: list[float] = []
        render_ms: list[float] = []
        combined_ms: list[float] = []
        instance_upload_bytes: list[float] = []
        stream_upload_bytes: dict[str, list[float]] = {
            "pose": [],
            "visual": [],
            "identity": [],
        }
        checksum = 0
        for frame in range(args.frames):
            _animate(mujoco, args.workload, model, data, base_qpos, args.warmup + frame)
            start = time.perf_counter()
            renderer.update_scene(data, camera="benchmark")
            updated = time.perf_counter()
            output = _render(renderer, args.mode, output)
            finished = time.perf_counter()
            update_ms.append((updated - start) * 1000.0)
            render_ms.append((finished - updated) * 1000.0)
            combined_ms.append((finished - start) * 1000.0)
            backend = getattr(renderer, "_backend", None)
            instances = getattr(backend, "instances", None)
            if instances is not None:
                instance_upload_bytes.append(float(instances.uploaded_bytes))
                uploaded_streams = instances.uploaded_streams
                for stream, values in stream_upload_bytes.items():
                    values.append(float(uploaded_streams.get(stream, 0)))
            checksum ^= int(output.reshape(-1)[frame % output.size])
    finally:
        close_start = time.perf_counter()
        renderer.close()
        close_ms = (time.perf_counter() - close_start) * 1000.0

    peak_rss = _rss_mb()
    combined = _distribution(combined_ms)
    return {
        "status": "ok",
        "renderer": args.renderer,
        "renderer_detail": detail,
        "workload": args.workload,
        "mode": args.mode,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "frames": args.frames,
        "warmup": args.warmup,
        "model": {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "compile_ms": compile_ms,
        },
        "constructor_ms": constructor_ms,
        "first_frame_ms": first_frame_ms,
        "update_scene": _distribution(update_ms),
        "render": _distribution(render_ms),
        "update_and_render": combined,
        "median_fps": 1000.0 / max(combined["median_ms"], 1e-9),
        "instance_upload": (
            None
            if not instance_upload_bytes
            else {
                "bytes": _byte_distribution(instance_upload_bytes),
                "streams": {
                    stream: _byte_distribution(values)
                    for stream, values in stream_upload_bytes.items()
                },
            }
        ),
        "close_ms": close_ms,
        "rss_before_mb": rss_before,
        "peak_rss_mb": peak_rss,
        "peak_rss_delta_mb": (
            None if rss_before is None or peak_rss is None else max(0.0, peak_rss - rss_before)
        ),
        "checksum": checksum,
        "versions": {
            "python": platform.python_version(),
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "numpy": np.__version__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer", choices=("mujoco", "mojive-opengl", "mojive-wgpu"), required=True
    )
    parser.add_argument("--workload", choices=tuple(_WORKLOADS), required=True)
    parser.add_argument("--mode", choices=("rgb", "depth", "segmentation"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.renderer == "mojive-opengl":
        os.environ["MOJIVE_BACKEND"] = "opengl"
    elif args.renderer == "mojive-wgpu":
        os.environ["MOJIVE_BACKEND"] = "wgpu"
    else:
        os.environ.pop("MOJIVE_BACKEND", None)

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
