"""Measure model-composition editing costs and write a reproducible JSON baseline."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .. import commands as cmd
from ..adapters.mujoco_adapter import MuJoCoAdapter
from ..adapters.workspace import WorkspaceAdapter
from ..session import Session


def _model_xml(bodies: int) -> str:
    body_xml = "\n".join(
        f'''    <body name="body{index}" pos="{index % 16} {index // 16} 0.2">
      <joint name="joint{index}" type="hinge"/>
      <geom name="geom{index}" type="box" size="0.08 0.08 0.08"/>
    </body>'''
        for index in range(max(1, int(bodies)))
    )
    return f"""<mujoco model="editor_benchmark">
  <compiler autolimits="true"/>
  <worldbody>
{body_xml}
  </worldbody>
</mujoco>
"""


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(statistics.median(values)),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": float(max(values)),
    }


def run_benchmark(
    output: Path,
    *,
    models: int = 8,
    bodies: int = 64,
    iterations: int = 5,
) -> dict[str, object]:
    """Run representative document edits and return the serializable baseline."""
    output = Path(output)
    asset_dir = output.parent / "editor-performance-assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    model_path = asset_dir / f"model-{bodies}-bodies.xml"
    model_path.write_text(_model_xml(bodies), encoding="utf-8")

    primary = MuJoCoAdapter()
    primary.new_scene()
    session = Session(WorkspaceAdapter(primary))
    session.submit(cmd.Pause())
    add_ms: list[float] = []
    model_ids: list[int] = []
    spacing = float(min(16, max(1, int(bodies))) + 2)
    try:
        for index in range(max(1, int(models))):
            start = time.perf_counter()
            result = session.submit(
                cmd.AddSceneModel(model_path, (float(index) * spacing, 0.0, 0.0))
            )
            add_ms.append((time.perf_counter() - start) * 1000.0)
            if not result.ok:
                raise RuntimeError(result.message)
            model_ids.append(result.entity_id)

        target = model_ids[-1]
        target_x = float(len(model_ids) - 1) * spacing
        transform_ms: list[float] = []
        for index in range(max(1, int(iterations))):
            start = time.perf_counter()
            result = session.submit(
                cmd.SetSceneModelTransform(
                    target,
                    np.array((target_x + (index + 1) * 0.01, 0.0, 0.0), np.float32),
                    np.eye(3, dtype=np.float32),
                )
            )
            transform_ms.append((time.perf_counter() - start) * 1000.0)
            if not result.ok:
                raise RuntimeError(result.message)

        start = time.perf_counter()
        added = session.submit(cmd.AddModelComponent(target, "sensor", "jointpos", "angle"))
        component_add_ms = (time.perf_counter() - start) * 1000.0
        if not added.ok:
            raise RuntimeError(added.message)

        component_ms: list[float] = []
        for index in range(max(1, int(iterations))):
            component = session.model_components(target, "sensor")[0]
            fields = tuple(
                (field.name, f"{index * 0.01:g}" if field.name == "noise" else field.value)
                for field in component.fields
            )
            start = time.perf_counter()
            result = session.submit(
                cmd.UpdateModelComponent(
                    target,
                    "sensor",
                    component.component_id,
                    component.name,
                    fields,
                )
            )
            component_ms.append((time.perf_counter() - start) * 1000.0)
            if not result.ok:
                raise RuntimeError(result.message)

        report: dict[str, object] = {
            "schema": 1,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "workload": {
                "models": len(model_ids),
                "bodies_per_model": max(1, int(bodies)),
                "iterations": max(1, int(iterations)),
                "compiled_bodies": int(primary.model.nbody),
                "compiled_geometries": int(primary.model.ngeom),
            },
            "timing": {
                "add_model": _summary(add_ms),
                "commit_model_transform": _summary(transform_ms),
                "add_component_ms": float(component_add_ms),
                "update_component": _summary(component_ms),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        primary.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=int, default=8)
    parser.add_argument("--bodies", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/editor-performance.json"))
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.output,
        models=args.models,
        bodies=args.bodies,
        iterations=args.iterations,
    )
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
