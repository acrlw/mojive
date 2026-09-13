from __future__ import annotations

import pytest

from mojive.tools.renderer_benchmark import _add_relative_results, _resolutions
from mojive.tools.renderer_benchmark_worker import (
    _cache_summary,
    _counter_distribution,
    _distribution,
)


def test_renderer_benchmark_parses_resolution_matrix() -> None:
    assert _resolutions("640x480, 1280x720") == ((640, 480), (1280, 720))

    with pytest.raises(ValueError, match="Invalid resolution"):
        _resolutions("640")


def test_renderer_benchmark_adds_same_case_mujoco_ratio() -> None:
    cases = [
        {
            "status": "ok",
            "renderer": "mujoco",
            "workload": "primitives",
            "mode": "rgb",
            "width": 640,
            "height": 480,
            "samples": 4,
            "update_and_render": {"median_ms": 2.0},
        },
        {
            "status": "ok",
            "renderer": "mojive-opengl",
            "workload": "primitives",
            "mode": "rgb",
            "width": 640,
            "height": 480,
            "samples": 4,
            "update_and_render": {"median_ms": 1.5},
        },
    ]

    _add_relative_results(cases)

    assert cases[0]["relative_to_mujoco"] == 1.0
    assert cases[1]["relative_to_mujoco"] == 0.75


def test_renderer_benchmark_distribution_uses_median_and_p95() -> None:
    result = _distribution([1.0, 2.0, 3.0, 20.0])

    assert result["median_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(17.45)


def test_renderer_benchmark_summarizes_pass_cache_activity() -> None:
    result = _cache_summary(["rendered", "reused", "reused", "off"])

    assert result == {
        "rendered_frames": 1,
        "reused_frames": 2,
        "inactive_frames": 1,
        "reuse_ratio": pytest.approx(2 / 3),
    }


def test_renderer_benchmark_summarizes_integer_counters() -> None:
    assert _counter_distribution([7, 7, 11]) == {"median": 7.0, "min": 7, "max": 11}
