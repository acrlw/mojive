"""Time-axis following preserves scale and user-selected screen positions."""

import pytest

from mojive.ui.panels.keyframes import follow_timeline_range, recorded_take_spans


def test_page_follow_leaves_visible_playhead_alone_and_places_overflow_near_start():
    assert follow_timeline_range(10, 20, 15, "page") == (10, 20)
    assert follow_timeline_range(10, 20, 20.1, "page") == pytest.approx((19.3, 29.3))
    assert follow_timeline_range(10, 20, 5, "page") == pytest.approx((4.2, 14.2))
    assert follow_timeline_range(10, 20, 1000, "off") == (10, 20)


@pytest.mark.parametrize("fraction", [0.1, 0.35, 0.8])
def test_locked_follow_keeps_playhead_at_same_relative_position_through_a_loop(fraction):
    start, end = 0, 20
    for playhead in (10, 10.1, 10.2, 5, 5.1):
        start, end = follow_timeline_range(start, end, playhead, "locked", fraction)
        assert end - start == pytest.approx(20)
        assert (playhead - start) / (end - start) == pytest.approx(fraction)


def test_dense_take_marks_merge_but_leave_visible_gaps_between_samples():
    assert list(recorded_take_spans((), 0, 1, 0, 100)) == []
    assert list(recorded_take_spans((0.1, 0.105, 0.11, 0.5), 0, 1, 0, 100)) == [
        (9, 12),
        (49, 51),
    ]

    class LongTake:
        reads = 0

        def __len__(self):
            return 100_000

        def __getitem__(self, index):
            self.reads += 1
            return index / 30

    take = LongTake()
    spans = list(recorded_take_spans(take, 0, 100_000 / 30, 0, 1000))
    assert spans == [(0, 1000)]
    assert take.reads < 1100
