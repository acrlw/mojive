"""Recovery comparisons require equivalent workloads and uninterrupted idle samples."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from mojive.tools import inspector_profile as profile


def _row():
    return {
        **dict.fromkeys(profile.STATE_FIELDS),
        "selected_node": 7,
        "hierarchy": {"open": [[1, True]], "scroll": [0, 0], "rows": [[1, 0, False]]},
        "field_hits": {"mass": [0, 0, 100, 30]},
        "main_thread_cpu_ms": {"median": 2.0},
        "ui_draw_counts": {"vertices": 100, "indices": 150, "commands": 3},
        "idle_windows": [{"main_thread_cpu_ms": {"median": 2.0}}],
        "idle_state_unchanged": True,
        "cache_bounds_ok": True,
        "fonts": 3,
        "font_cache": 3,
        "styles": 12,
        "windows": 6,
    }


def test_recovery_reports_each_idle_window_only_for_equivalent_state():
    before, after = _row(), _row()
    after["idle_windows"] = [{"main_thread_cpu_ms": {"median": value}} for value in (2.0, 2.5, 2.1)]
    after["main_thread_cpu_ms"]["median"] = 2.1
    after["styles"] += 2
    actual = profile._comparison(before, after)
    assert actual["same_state"] and actual["same_draw_counts"]
    assert actual["cpu_median_ratio"] == pytest.approx(1.05)
    assert actual["idle_cpu_median_ratios"] == pytest.approx([1, 1.25, 1.05])
    assert actual["count_changes"] == {"fonts": 0, "font_cache": 0, "styles": 2, "windows": 0}


@pytest.mark.parametrize(
    "changed", ("selected_node", "hierarchy", "field_hits", "inspector_scroll")
)
def test_recovery_does_not_label_different_scene_or_inspector_work_as_a_slowdown(changed):
    before, after = _row(), _row()
    after[changed] = "different"
    actual = profile._comparison(before, after)
    assert actual["changed_state_fields"] == [changed]
    assert not actual["same_state"]
    assert actual["cpu_median_ratio"] is None
    assert actual["idle_cpu_median_ratios"] is None


def test_recovery_rejects_an_idle_series_that_changed_midway():
    before, after = _row(), _row()
    after["idle_state_unchanged"] = False
    actual = profile._comparison(before, after)
    assert not actual["same_state"] and not actual["changed_state_fields"]
    assert actual["cpu_median_ratio"] is None
    assert actual["idle_cpu_median_ratios"] is None


def test_timed_window_measures_each_normal_sync_once(monkeypatch):
    calls = []
    wall = iter((0, 3_000_000, 4_000_000, 9_000_000))
    cpu = iter((0, 1_000_000, 2_000_000, 4_000_000))
    monkeypatch.setattr(profile.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(profile.time, "thread_time_ns", lambda: next(cpu))
    result = profile._timed_window(SimpleNamespace(sync=lambda: calls.append("frame")), 2)
    assert calls == ["frame", "frame"]
    assert result == ([3.0, 5.0], [1.0, 2.0])


@pytest.mark.parametrize("changed", (False, True))
def test_idle_windows_remain_contiguous_until_capture(monkeypatch, tmp_path, changed):
    events = []
    viewer = SimpleNamespace(capture_array=lambda **kwargs: events.append("capture") or "pixels")
    inputs = SimpleNamespace(
        viewer=viewer, pointer=(50, 50), sync=lambda count: events.append(("settle", count))
    )
    args = SimpleNamespace(frames=24, profile=False, output=tmp_path)
    original = {
        **_row(),
        "native_clicks": 300,
        "state_signature": "same",
        "cache_counts": {"vector_layouts": {"entries": 12, "limit": 128}},
    }

    def sample(actual_viewer, frames):
        assert actual_viewer is viewer and frames == 24
        events.append("timing")
        return [2.5, 3.0], [1.0, 1.5]

    def snapshot(actual_inputs):
        assert actual_inputs is inputs
        events.append("snapshot")
        row = deepcopy(original)
        if changed and events.count("snapshot") == 2:
            row["state_signature"] = "changed"
            row["cache_bounds_ok"] = False
        return row

    monkeypatch.setattr(profile, "_timed_window", sample)
    monkeypatch.setattr(profile, "_snapshot", snapshot)
    monkeypatch.setattr(
        profile.Image,
        "fromarray",
        lambda pixels: SimpleNamespace(save=lambda path: events.append("save")),
    )
    result = profile._phase(inputs, args, "idle", windows=3)
    assert events == [
        ("settle", 12),
        *(event for _ in range(3) for event in ("timing", "snapshot")),
        "capture",
        "save",
    ]
    assert inputs.pointer == (0, 0)
    assert len(result["idle_windows"]) == 3
    assert result["idle_state_unchanged"] is not changed
    assert result["cache_bounds_ok"] is not changed


def test_cache_report_includes_prepared_variants_and_explicit_bounds():
    variant = SimpleNamespace(_geometry_cache=dict.fromkeys(range(256)), _variants={})
    style = SimpleNamespace(_geometry_cache={}, _variants={"radius": variant})
    ui = SimpleNamespace(
        design=SimpleNamespace(
            primitives=SimpleNamespace(_styles={"button": style}, _style_requests={1: style})
        ),
        _vector_layouts={1: None},
        _numeric_layout_styles=(None, None, {"input": style}),
    )
    result = profile._cache_counts(ui)
    assert result["geometry_per_style_max"] == {"entries": 256, "limit": 256}
    assert result["variants_per_style_max"] == {"entries": 1, "limit": 128}
    assert result["vector_layouts"] == {"entries": 1, "limit": 128}


def test_reference_primes_selection_paths_and_restores_the_same_empty_tab(monkeypatch, tmp_path):
    events = []
    workspace = SimpleNamespace(
        ui=SimpleNamespace(translate=str, hits={}), inspector_tab="Properties"
    )
    viewer = SimpleNamespace(
        app=SimpleNamespace(
            workspace=workspace, set_language=lambda value: None, design=SimpleNamespace(spec={})
        ),
        backend=SimpleNamespace(caps=SimpleNamespace(name="opengl")),
        window=SimpleNamespace(
            style_scale=1.2, pixel_scale=2, size_points=(1600, 1000), size_pixels=(3200, 2000)
        ),
    )

    class Inputs:
        def __init__(self, viewer):
            self.workspace = viewer.app.workspace
            self.selected = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def sync(self, count=4):
            pass

        def top(self):
            pass

        def select(self, name):
            self.selected = name
            events.append(("select", name))

        def click(self, key):
            events.append(("click", key))
            if key.startswith("tab-"):
                self.workspace.inspector_tab = key[4:]

        def deselect(self):
            self.selected = None
            events.append(("deselect", None))

    def phase(inputs, args, label, *, windows):
        events.append(("phase", label))
        row = _row()
        row.update(selected_node=inputs.selected, inspector_tab=inputs.workspace.inspector_tab)
        if label in ("joint-expanded-after", "unselected-after"):
            assert windows == 3
        return row

    monkeypatch.setattr(profile, "build_workspace", lambda *args, **kwargs: nullcontext(viewer))
    monkeypatch.setattr(profile, "InspectorInput", Inputs)
    monkeypatch.setattr(profile, "_phase", phase)
    args = SimpleNamespace(
        frames=12,
        cycles=1,
        language="en",
        renderer="opengl",
        vsync=False,
        visible=False,
        output=tmp_path,
    )
    report = profile._run(args)
    assert events[:6] == [
        *(("select", name) for name in profile.CHURN_ENTITIES),
        ("click", "tab-Properties"),
        ("deselect", None),
        ("phase", "unselected-before"),
    ]
    assert events[-3:] == [
        ("click", "tab-Properties"),
        ("deselect", None),
        ("phase", "unselected-after"),
    ]
    assert all(comparison["same_state"] for comparison in report["recovery"].values())
