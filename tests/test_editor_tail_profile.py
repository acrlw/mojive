"""Tail attribution keeps nested timings and collection evidence unambiguous."""

import gc
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from mojive.tools import editor_tail_profile as profile


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(wall=0, cpu=0)
    monkeypatch.setattr(profile.time, "perf_counter_ns", lambda: state.wall)
    monkeypatch.setattr(profile.time, "thread_time_ns", lambda: state.cpu)
    return state


def make_timer(clock, stack):
    calls = []

    def submit(command):
        calls.append("command")
        clock.wall += 5_000_000
        clock.cpu += 3_000_000
        return SimpleNamespace(ok=True)

    def sync():
        calls.append("frame")
        clock.wall += 12_000_000
        clock.cpu += 8_000_000

    viewer = SimpleNamespace(
        sync=sync,
        session=SimpleNamespace(submit=submit, selected_node=SimpleNamespace(node_id=42)),
        app=SimpleNamespace(gizmo=SimpleNamespace(using=False)),
    )
    timer = profile.TailSamples(viewer, stack=stack, warmup_cycles=12)
    viewer.sync = timer.sync
    return timer, calls


def records(timer):
    return [profile.marshal.loads(row) for row in timer.events]


def test_explicit_edit_retains_separate_command_frame_and_total_latency(clock):
    with ExitStack() as stack:
        timer, calls = make_timer(clock, stack)
        timer.phase = "create-delete"
        timer.edit(SimpleNamespace(name="soak_13_0"))
    command, frame, edit = records(timer)
    assert calls == ["command", "frame"]
    assert [row["kind"] for row in records(timer)] == ["command", "frame", "edit"]
    assert [row["wall_ms"] for row in records(timer)] == [5, 12, 17]
    assert [row["cpu_ms"] for row in records(timer)] == [3, 8, 11]
    assert edit["command_ms"] == 5 and edit["frame_ms"] == 12
    assert edit["command_cpu_ms"] == 3 and edit["frame_cpu_ms"] == 8
    assert edit["observer_gap_ms"] == 0
    assert all(row["cycle"] == 1 and row["frame"] == 1 for row in records(timer))
    assert all(row["selected_node"] == 42 for row in records(timer))
    assert all(isinstance(row["stages"], tuple) for row in records(timer))
    assert command["command"] == frame["command"] == edit["command"] == "SimpleNamespace"
    assert timer.command is None
    assert timer.drain()["create-delete"]["sync_wall_ms"]["max"] == 12
    assert timer.drain_edits()["SimpleNamespace"]["command_and_frame_wall_ms"]["max"] == 17


def test_combined_edit_reports_observer_gap_without_subtracting_it(clock, monkeypatch):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        original = timer.finish

        def store(*args):
            original(*args)
            clock.wall += 2_000_000
            clock.cpu += 1_000_000

        monkeypatch.setattr(timer, "finish", store)
        timer.edit(SimpleNamespace(name="soak_13_0"))
    edit = records(timer)[-1]
    assert edit["wall_ms"] == 21 and edit["cpu_ms"] == 13
    assert edit["command_ms"] == 5 and edit["frame_ms"] == 12
    assert edit["command_cpu_ms"] == 3 and edit["frame_cpu_ms"] == 8
    assert edit["observer_gap_ms"] == 4


def test_completed_baseline_events_retain_only_untracked_scalar_metadata(clock):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        assert not gc.is_tracked(timer.event("frame"))
        timer.edit(SimpleNamespace(name="soak_13_0"))
        timer.phase = "drag"
        timer.sync()
    assert len(timer.events) == 4
    for event in records(timer):
        assert event["stages"] == ()
        assert all(type(key) is str for key in event)
        assert all(
            value is None
            or type(value) in (str, int, float, bool)
            or (type(value) is tuple and not value)
            for value in event.values()
        )
    # Baseline samples must never need a full collection to become untracked.
    # Otherwise the observer itself promotes retained records into later scans.
    assert all(not gc.is_tracked(event) for event in timer.events)


def test_retaining_completed_events_does_not_trigger_collections(clock):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        gc.collect()
        before = gc.get_stats()
        for _ in range(10000):
            timer.finish(timer.event("frame"), 0, 0, 0, 0)
        after = gc.get_stats()
    assert len(timer.events) == 10000
    assert [row["collections"] for row in after] == [row["collections"] for row in before]


def test_frame_events_distinguish_gizmo_press_hold_and_release(clock):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        original = timer.original
        states = iter((True, True, False))

        def sync():
            original()
            timer.viewer.app.gizmo.using = next(states)

        timer.original = sync
        for _ in range(3):
            timer.sync()
    assert [(row["gizmo_before"], row["gizmo_after"]) for row in records(timer)] == [
        (False, True),
        (True, True),
        (True, False),
    ]
    assert all(row["wall_ms"] == 12 for row in records(timer))


def test_nested_stage_timings_are_inclusive_and_preserve_call_counts(clock):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        timer._active_stages = {}

        def child():
            clock.wall += 4_000_000
            clock.cpu += 2_000_000

        wrapped_child = timer.stage(child, "child")

        def parent():
            wrapped_child()
            wrapped_child()
            clock.wall += 1_000_000
            clock.cpu += 1_000_000

        timer.stage(parent, "parent")()
        assert timer._active_stages == {
            "child": [2, 8_000_000, 4_000_000, 4_000_000],
            "parent": [1, 9_000_000, 5_000_000, 9_000_000],
        }
        event = timer.event("frame")
        event["stages"] = timer._active_stages
        timer.finish(event, 0, 0, clock.wall, clock.cpu)
        serialized = profile.serialize_events(timer, profile.Collections())[0]
        assert serialized["stages"]["child"] == {
            "calls": 2,
            "wall_ms": 8,
            "cpu_ms": 4,
            "max_call_ms": 4,
        }


def test_stage_wrapper_preserves_exception_and_records_failed_call(clock):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        timer._active_stages = {}

        def fail():
            clock.wall += 1_000_000
            raise ValueError("failure evidence")

        with pytest.raises(ValueError, match="failure evidence"):
            timer.stage(fail, "failed")()
        assert timer._active_stages["failed"] == [1, 1_000_000, 0, 1_000_000]


def test_panel_timings_keep_each_actual_owner_separate(clock):
    with ExitStack() as stack:
        timer, _ = make_timer(clock, stack)
        timer._active_stages = {}

        def draw(name):
            clock.wall += 2_000_000
            clock.cpu += 1_000_000

        timed = timer.stage(draw, lambda name: "panel." + name)
        timed("Scene")
        timed("Inspector")
        timed("Scene")
        assert timer._active_stages == {
            "panel.Scene": [2, 4_000_000, 2_000_000, 2_000_000],
            "panel.Inspector": [1, 2_000_000, 1_000_000, 2_000_000],
        }


def test_collection_correlates_only_with_overlapping_events(clock):
    collections = profile.Collections()
    clock.wall, clock.cpu = 2_000_000, 1_000_000
    collections.callback("start", {"generation": 2})
    clock.wall, clock.cpu = 9_000_000, 7_000_000
    collections.callback("stop", {"generation": 2, "collected": 15, "uncollectable": 0})
    timer = SimpleNamespace(
        origin_ns=0,
        events=[
            profile.marshal.dumps(
                {
                    "kind": "frame",
                    "start_ns": start,
                    "end_ns": end,
                    "wall_ms": (end - start) / 1e6,
                    "cpu_ms": 0,
                    "stages": {},
                }
            )
            for start, end in ((0, 10_000_000), (10_000_000, 20_000_000))
        ],
    )
    first, second = profile.serialize_events(timer, collections)
    assert first["gc"] == {"2": {"count": 1, "wall_ms": 7, "cpu_ms": 6, "collected": 15}}
    assert second["gc"] == {}
    assert not any(collections.starts)


def test_tail_summary_excludes_warmup_but_retains_raw_and_reports_off_cpu():
    events = [
        {
            "kind": "frame",
            "phase": "drag",
            "command": None,
            "cycle": cycle,
            "wall_ms": wall,
            "cpu_ms": cpu,
            "off_cpu_ms": wall - cpu,
        }
        for cycle, wall, cpu in ((0, 100, 100), (1, 10, 9), (2, 40, 12))
    ]
    result = profile.summarize_events(events, top=1)["frame/drag/none"]
    assert len(events) == 3
    assert result["count"] == 2
    assert result["wall_ms"]["max"] == 40 and result["cpu_ms"]["max"] == 12
    assert result["off_cpu_ms"]["max"] == 28
    assert result["over_33_33_ms"] == 1
    assert result["slowest"] == [events[-1]]
