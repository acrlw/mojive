from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from forge_viewer.tools.editor_performance import run_benchmark


def test_editor_performance_baseline_covers_composition_and_structured_edits(
    tmp_path: Path,
) -> None:
    output = tmp_path / "editor-performance.json"
    report = run_benchmark(output, models=2, bodies=3, iterations=2)

    assert report["schema"] == 1
    assert report["workload"] == {
        "models": 2,
        "bodies_per_model": 3,
        "iterations": 2,
        "compiled_bodies": 7,
        "compiled_geometries": 6,
    }
    timing = report["timing"]
    assert set(timing) == {
        "add_model",
        "commit_model_transform",
        "add_component_ms",
        "update_component",
    }
    assert timing["commit_model_transform"]["median_ms"] > 0.0
    assert json.loads(output.read_text(encoding="utf-8"))["workload"]["models"] == 2
