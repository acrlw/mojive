"""Snapshot ownership, shutdown, and benchmark timing invariants."""

import threading

import pytest

from mojive.session.simulation import Snapshot, SnapshotPool
from mojive.tools.physics_render_benchmark import interval_overlap


def test_ordered_publication_keeps_the_readers_slot_until_release():
    pool = SnapshotPool()
    first, second, third = (pool.reserve() for _ in range(3))
    for step, slot in enumerate((first, second, third)):
        pool.publish(Snapshot(slot, step, 0))
    reading = pool.acquire()
    assert reading.slot == first
    assert pool.acquire().slot == second
    pool.release(second)
    # Even when the producer outruns rendering, it cannot reuse the frame
    # currently bound to the rendering adapter.
    assert pool.reserve() == second
    assert pool.acquire().slot == third
    pool.release(first)
    assert pool.reserve() == first


def test_latest_publication_drops_pending_frames_but_preserves_reader():
    pool = SnapshotPool(latest=True)
    reading = pool.reserve()
    pool.publish(Snapshot(reading, 1, 0))
    assert pool.acquire().slot == reading
    for step in range(2, 100):
        slot = pool.reserve()
        assert slot != reading
        pool.publish(Snapshot(slot, step, 0))
    assert pool.acquire().step == 99
    assert pool.dropped == 97


@pytest.mark.parametrize("blocked_operation", ["reserve", "acquire"])
def test_close_wakes_a_blocked_producer_or_consumer(blocked_operation):
    pool = SnapshotPool(2)
    if blocked_operation == "reserve":
        pool.reserve()
        pool.reserve()
    entered, returned = threading.Event(), threading.Event()
    result = []

    def wait():
        entered.set()
        result.append(getattr(pool, blocked_operation)())
        returned.set()

    worker = threading.Thread(target=wait, daemon=True)
    worker.start()
    assert entered.wait(timeout=1)
    pool.close()
    worker.join(timeout=1)
    assert returned.is_set()
    assert result == [None]


def test_worker_failure_is_reported_even_when_a_snapshot_is_ready():
    pool = SnapshotPool()
    pool.publish(Snapshot(pool.reserve(), 1, 0))
    cause = ValueError("invalid physics state")
    pool.finish(cause)
    with pytest.raises(RuntimeError, match="Physics worker failed") as error:
        pool.acquire(wait=False)
    assert error.value.__cause__ is cause


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([], [(0, 10)], 0),
        ([(0, 10)], [(10, 20)], 0),
        ([(0, 10)], [(2, 4), (6, 12)], 6),
        ([(6, 10), (0, 4)], [(2, 8)], 4),
    ],
)
def test_overlap_counts_intersections_without_double_counting(first, second, expected):
    assert interval_overlap(first, second) == expected
