"""Time-axis following preserves scale and user-selected screen positions."""

import pytest

from mojive.interaction.timeline import (
    TimelineMarkerIndex,
    decimated_marker_ids,
    follow_timeline_range,
    recorded_take_spans,
    timeline_time_to_x,
)


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


def test_marker_projection_only_retains_visible_markers_and_edge_hit_margin():
    index = TimelineMarkerIndex(tuple((i, float(i)) for i in range(100_000)))
    projection = index.project(50_000, 50_010, 100, 200, 11)
    assert set(projection.positions) == set(range(49_999, 50_012))
    assert projection.hit(100, 11) == 50_000
    assert projection.hit(89, 11) == 49_999
    assert projection.hit(70, 11) == -1
    assert index.project(50_000, 50_010, 100, 200, 11) is projection
    with pytest.raises(TypeError):
        projection.positions[50_000] = 0


def test_drag_preview_can_enter_and_leave_the_view_without_mutating_content():
    index = TimelineMarkerIndex(((1, 0), (2, 2), (3, 10)))
    initial = index.project(0, 3, 0, 300)
    assert initial.positions == {1: 0, 2: 200}
    entered = index.project(0, 3, 0, 300, moved=(3, 1))
    assert entered.positions == {1: 0, 2: 200, 3: 100}
    assert entered.hit(100, 5) == 3
    departed = index.project(0, 3, 0, 300, moved=(2, 5))
    assert departed.positions == {1: 0}
    assert index.project(0, 3, 0, 300).positions == initial.positions


def test_times_that_round_to_the_same_pixel_keep_identity_and_draw_order():
    projection = TimelineMarkerIndex(((9, 0), (8, 1e-20), (1, 2e-20))).project(0, 1e12, 100, 101)
    assert set(projection.positions.values()) == {100}
    assert projection.hit(100, 1) == 1
    assert projection.draw_ids(100, 101, 4, (1, 8)) == (9, 8, 1)


def test_index_matches_full_projection_hit_and_draw_decisions():
    import random

    rng = random.Random(73)
    markers = tuple((i, rng.randrange(100) / 10) for i in range(2000))
    index = TimelineMarkerIndex(markers)
    for start, end in ((0, 10), (3, 3.2), (9, 12)):
        full = {
            key: timeline_time_to_x(time, start, end, 100, 1000)
            for key, time in sorted(markers, key=lambda item: (item[1], item[0]))
        }
        projection = index.project(start, end, 100, 1000, 11)
        for x in range(100, 1001, 7):
            expected = min(
                ((abs(px - x), key) for key, px in full.items() if abs(px - x) <= 11),
                default=(float("inf"), -1),
            )[1]
            assert projection.hit(x, 11) == expected
        for priority in ((), (9, 200, 401), (1999, 1999)):
            assert projection.draw_ids(93, 1007, 10.5, priority) == decimated_marker_ids(
                tuple(full.items()), 93, 1007, 10.5, priority
            )


def test_keyframe_index_rebuilds_for_model_switch_and_content_revision():
    from types import SimpleNamespace

    from mojive.adapters.base import KeyframeInfo
    from mojive.ui.panels.keyframes import KeyframesPanel

    panel = KeyframesPanel()
    session = SimpleNamespace(
        keyframe_revision=0,
        keyframes=[KeyframeInfo(1, "one", 1, 0), KeyframeInfo(2, "two", 2, 1)],
    )
    session.model_keyframes = lambda model_id: tuple(
        key for key in session.keyframes if key.model_id == model_id
    )
    ctx = SimpleNamespace(session=session)
    panel._set_model(0)
    panel._keyframes(ctx)
    assert panel._marker_index.project(0, 3, 0, 300).positions == {1: 100}
    session.keyframes = [KeyframeInfo(1, "moved", 1.5, 0), session.keyframes[1]]
    session.keyframe_revision += 1
    panel._keyframes(ctx)
    assert panel._marker_index.project(0, 3, 0, 300).positions == {1: 150}
    panel._set_model(1)
    panel._keyframes(ctx)
    assert panel._marker_index.project(0, 3, 0, 300).positions == {2: 200}


def test_drag_projection_matches_full_projection_and_preserves_cached_base():
    import random

    rng = random.Random(384)
    markers = tuple((key, rng.randrange(150) * 0.1) for key in rng.sample(range(2000), 800))
    index = TimelineMarkerIndex(markers)
    ordered = sorted(markers, key=lambda item: (item[1], item[0]))
    base = index.project(3, 9, 50, 750, 12)
    for _ in range(100):
        key, time = rng.choice(markers)[0], rng.uniform(0, 15)
        projected = index.project(3, 9, 50, 750, 12, moved=(key, time))
        expected = {
            key_id: timeline_time_to_x(t, 3, 9, 50, 750)
            for key_id, t in ordered
            if 3 - 12 * 6 / 700 <= t <= 9 + 12 * 6 / 700
        }
        if 3 - 12 * 6 / 700 <= time <= 9 + 12 * 6 / 700:
            expected[key] = timeline_time_to_x(time, 3, 9, 50, 750)
        else:
            expected.pop(key, None)
        assert dict(projected.positions) == expected
        assert index.project(3, 9, 50, 750, 12) is base
        for priority in ((), (key,), (key, key, 5, 20)):
            for spacing in (0.1, 8, 23, 900):
                assert projected.draw_ids(50, 750, spacing, priority) == decimated_marker_ids(
                    tuple(expected.items()), 50, 750, spacing, priority
                )
        for x in (rng.uniform(40, 760), timeline_time_to_x(time, 3, 9, 50, 750)):
            nearest = min(
                ((abs(px - x), key_id) for key_id, px in expected.items()),
                default=(float("inf"), -1),
            )
            assert projected.hit(x, 12) == (nearest[1] if nearest[0] <= 12 else -1)


def test_indexed_selection_includes_equal_times_and_inclusive_endpoints():
    index = TimelineMarkerIndex(((8, 2), (2, 1), (3, 2), (9, 4)))
    assert index.ids_between(1, 2) == (2, 3, 8)
    assert index.ids_between(2.1, 3.9) == ()
    assert index.ids_between(3, 2) == ()
