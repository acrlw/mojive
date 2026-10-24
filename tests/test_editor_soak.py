"""Long-session performance comparisons must retain equivalent workloads."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mojive.tools.editor_soak import FrameSamples, assert_released, compare, summarize


def checkpoint():
    return {
        "state": {"state_signature": "same", "ui_draw_counts": {"vertices": 100}},
        "idle": {"main_thread_cpu_ms": {"median": 2}, "sync_wall_ms": {"median": 4}},
    }


@pytest.mark.parametrize("field", ("state_signature", "ui_draw_counts"))
def test_mismatched_workload_does_not_produce_a_slowdown_ratio(field):
    before, after = checkpoint(), checkpoint()
    after["state"][field] = "different"
    result = compare(before, after)
    assert result["idle_cpu_median_ratio"] is None
    assert result["idle_wall_median_ratio"] is None


def test_matched_workloads_report_wall_and_cpu_separately():
    before, after = checkpoint(), deepcopy(checkpoint())
    after["idle"]["sync_wall_ms"]["median"] = 8
    assert compare(before, after) == {
        "same_state": True,
        "same_draw_counts": True,
        "idle_cpu_median_ratio": 1,
        "idle_wall_median_ratio": 2,
    }


def test_stalls_are_counted_per_frame_with_individual_evidence():
    values = {"wall": [3, 17, 34], "cpu": [2, 4, 5]}
    result = summarize(values)
    assert result["frames"] == 3
    assert result["over_16_67_ms"] == 2
    assert result["over_33_33_ms"] == 1
    assert result["wall_samples_ms"] == values["wall"]
    assert result["cpu_samples_ms"] == values["cpu"]


def test_frame_timer_calls_sync_once_and_separates_phases(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mojive.tools.editor_soak.time.perf_counter_ns",
        iter((0, 3_000_000, 4_000_000, 10_000_000)).__next__,
    )
    monkeypatch.setattr(
        "mojive.tools.editor_soak.time.thread_time_ns",
        iter((0, 1_000_000, 2_000_000, 4_000_000)).__next__,
    )
    samples = FrameSamples(SimpleNamespace(sync=lambda: calls.append("sync")))
    samples.sync()
    samples.phase = "drag"
    samples.sync()
    report = samples.drain()
    assert calls == ["sync", "sync"] and samples.frames == 2
    assert report["warmup"]["wall_samples_ms"] == [3]
    assert report["drag"]["cpu_samples_ms"] == [2]
    assert not samples.drain()


@pytest.mark.parametrize("owner", ("editing", "gizmo", "numeric"))
def test_completed_gesture_rejects_stale_input_ownership(owner):
    viewer = SimpleNamespace(
        session=SimpleNamespace(editing=owner == "editing"),
        app=SimpleNamespace(
            gizmo=SimpleNamespace(using=owner == "gizmo"),
            workspace=SimpleNamespace(_numeric_edit=object() if owner == "numeric" else None),
        ),
    )
    with pytest.raises(RuntimeError):
        assert_released(viewer)


def test_edit_latency_includes_the_command_before_its_frame(monkeypatch):
    events = []
    result = SimpleNamespace(ok=True)
    viewer = SimpleNamespace(
        session=SimpleNamespace(submit=lambda command: events.append("command") or result),
        sync=lambda: events.append("frame"),
    )
    monkeypatch.setattr(
        "mojive.tools.editor_soak.time.perf_counter_ns", iter((0, 9_000_000)).__next__
    )
    monkeypatch.setattr(
        "mojive.tools.editor_soak.time.thread_time_ns", iter((0, 7_000_000)).__next__
    )
    samples = FrameSamples(viewer)
    assert samples.edit(object()) is result
    row = samples.drain_edits()["object"]
    assert events == ["command", "frame"]
    assert row["operations"] == 1
    assert row["command_and_frame_wall_ms"]["median"] == 9
    assert row["main_thread_cpu_ms"]["median"] == 7
    assert not samples.drain_edits()
