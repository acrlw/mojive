"""Compare optional bindings using shared C++ code and the 100-humanoid fixture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import platform
import statistics
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


def matrices_from_fixture(path: Path) -> np.ndarray:
    """Read only the first frame of the private native trajectory format."""
    with path.open("rb") as stream:
        if stream.read(8) != b"MJVPROB1":
            raise ValueError("Unknown native fixture")
        meshes, instances, _ = struct.unpack("<III", stream.read(12))
        stream.seek(132, 1)
        for _ in range(meshes):
            vertices, indices = struct.unpack("<II", stream.read(8))
            stream.seek(vertices * 24 + indices * 4, 1)
        stream.seek(instances * 32, 1)
        return np.frombuffer(stream.read(instances * 64), dtype="<f4").reshape(-1, 4, 4)


def measure(function, iterations: int) -> float:
    """Include the same Python dispatch loop for every binding."""
    for _ in range(100):
        function()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        return (time.perf_counter_ns() - start) / iterations
    finally:
        gc.enable()


def main() -> None:
    """Preserve independent run samples and neutral aggregate metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modules", type=Path, default=Path("output/native-bindings-build/bindings")
    )
    parser.add_argument("--scene", type=Path, default=Path("output/native-probe/humanoids100.mjvp"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("output/native-probe/bindings/report.json")
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.modules.resolve()))
    modules = {
        name: importlib.import_module(f"_mojive_{name}_probe") for name in ("nanobind", "pybind")
    }
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=10)
    matrices = matrices_from_fixture(args.scene)
    records = []
    for repeat in range(7):
        for name in list(modules) if repeat % 2 == 0 else list(modules)[::-1]:
            module = modules[name]
            batch = module.FrameBatch(len(matrices))
            state = module.OwnedState(data.qpos)
            expected = np.sum(matrices[:, :3, 3], dtype=np.float64)
            assert np.isclose(batch.pack(matrices), expected, rtol=1e-6)
            assert module.address(data.qpos) == data.qpos.ctypes.data
            metrics = {
                "scalar_call_ns": measure(lambda module=module: module.ping(10), 300000),
                "ndarray_borrow_ns": measure(
                    lambda module=module: module.address(data.qpos), 100000
                ),
                "owned_qpos_copy_ns": measure(
                    lambda module=module: module.OwnedState(data.qpos), 10000
                ),
                "pack_5101_instances_ns": measure(lambda batch=batch: batch.pack(matrices), 5000),
                "native_work_ns": measure(lambda state=state: state.work(10), 200),
            }
            records.append({"binding": name, "repeat": repeat, **metrics})
            print(name, repeat, metrics, flush=True)
    summary = {}
    for name in modules:
        samples = [item for item in records if item["binding"] == name]
        summary[name] = {
            metric: statistics.median(item[metric] for item in samples) for metric in metrics
        }
        summary[name]["module_bytes"] = Path(modules[name].__file__).stat().st_size
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "model": str(args.model),
        "nq": model.nq,
        "nv": model.nv,
        "instances": len(matrices),
        "packed_bytes": batch.bytes(),
        "fixture_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "scope": "Boundary cost and identical C++ kernels; not editor FPS or physics throughput",
        "records": records,
        "median_of_runs": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
