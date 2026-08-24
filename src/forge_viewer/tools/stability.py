"""Exercise long-running frames and repeated large-model lifecycle operations."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import tracemalloc
from pathlib import Path

from ..adapters.base import FrameNeeds
from ..adapters.mujoco_adapter import MuJoCoAdapter
from .editor_performance import _model_xml


def _rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    resident_pages = int(statm.read_text(encoding="ascii").split()[1])
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def _memory_sample() -> dict[str, int | None]:
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    return {"python_bytes": int(current), "python_peak_bytes": int(peak), "rss_bytes": _rss_bytes()}


def _growth(after: dict[str, int | None], before: dict[str, int | None]) -> dict[str, int | None]:
    return {
        name: None
        if after[name] is None or before[name] is None
        else int(after[name] - before[name])
        for name in after
        if name != "python_peak_bytes"
    }


def run_stability(
    output: Path,
    *,
    frames: int = 10_000,
    cycles: int = 20,
    bodies: int = 256,
    max_growth_mb: float = 8.0,
) -> dict[str, object]:
    """Run the headless stability gate and return its serializable report."""
    output = Path(output)
    asset_dir = output.parent / "stability-assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    model_path = asset_dir / f"large-{bodies}-bodies.xml"
    model_path.write_text(_model_xml(bodies), encoding="utf-8")

    tracemalloc.start()
    adapter = MuJoCoAdapter(model_path)
    needs = FrameNeeds(poses=True, qpos=True, qvel=True, sensors=True, diagnostics=True)
    try:
        for _ in range(32):
            adapter.step()
            adapter.frame(needs)
        buffer_ids = (
            id(adapter._geom_xpos_buf),
            id(adapter._geom_xmat_buf),
            id(adapter._body_xpos_buf),
            id(adapter._body_xmat_buf),
            id(adapter._qpos_buf),
            id(adapter._qvel_buf),
        )
        half = max(1, int(frames) // 2)
        for _ in range(half):
            adapter.step()
            adapter.frame(needs)
        midpoint = _memory_sample()
        for _ in range(max(1, int(frames)) - half):
            adapter.step()
            adapter.frame(needs)
        frame_end = _memory_sample()
        buffers_reused = buffer_ids == (
            id(adapter._geom_xpos_buf),
            id(adapter._geom_xmat_buf),
            id(adapter._body_xpos_buf),
            id(adapter._body_xmat_buf),
            id(adapter._qpos_buf),
            id(adapter._qvel_buf),
        )

        adapter.new_scene()
        adapter.load(model_path)
        adapter.scene_source()
        adapter.frame(needs)
        lifecycle_start = _memory_sample()
        for _ in range(max(1, int(cycles))):
            adapter.new_scene()
            adapter.scene_source()
            adapter.frame(needs)
            adapter.load(model_path)
            adapter.scene_source()
            adapter.frame(needs)
        lifecycle_end = _memory_sample()

        frame_growth = _growth(frame_end, midpoint)
        lifecycle_growth = _growth(lifecycle_end, lifecycle_start)
        limit = int(float(max_growth_mb) * 1024 * 1024)
        checks = {
            "frame_buffers_reused": buffers_reused,
            "steady_python_growth_within_budget": int(frame_growth["python_bytes"] or 0) <= limit,
            "steady_rss_growth_within_budget": frame_growth["rss_bytes"] is None
            or int(frame_growth["rss_bytes"]) <= limit,
            "lifecycle_python_growth_within_budget": int(lifecycle_growth["python_bytes"] or 0)
            <= limit,
            "lifecycle_rss_growth_within_budget": lifecycle_growth["rss_bytes"] is None
            or int(lifecycle_growth["rss_bytes"]) <= limit,
        }
        report: dict[str, object] = {
            "schema": 1,
            "platform": platform.platform(),
            "workload": {
                "frames": max(1, int(frames)),
                "load_cycles": max(1, int(cycles)),
                "bodies": max(1, int(bodies)),
                "max_growth_mb": float(max_growth_mb),
            },
            "steady_frame_growth": frame_growth,
            "large_model_lifecycle_growth": lifecycle_growth,
            "checks": checks,
            "passed": all(checks.values()),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        adapter.release()
        tracemalloc.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--bodies", type=int, default=256)
    parser.add_argument("--max-growth-mb", type=float, default=8.0)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/stability.json"))
    args = parser.parse_args(argv)
    report = run_stability(
        args.output,
        frames=args.frames,
        cycles=args.cycles,
        bodies=args.bodies,
        max_growth_mb=args.max_growth_mb,
    )
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
