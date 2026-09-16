"""Dense timelines bound work by screen width and keep exact input identities."""

from dataclasses import replace

import numpy as np
import pytest

from mojive.adapters.base import KeyframeInfo
from mojive.ui.timeline import (
    decimated_marker_ids,
    hit_keyframe,
    keyframe_range,
    neighboring_keyframe,
    neighboring_sorted_keyframe,
    timeline_time_to_x,
    visible_keyframe_positions,
)


@pytest.mark.parametrize("count", [0, 20, 1000, 100000])
@pytest.mark.parametrize("start,end", [(0, 1), (0, 1000), (10, 10.01)])
def test_indexed_marker_selection_preserves_bucket_contents(count, start, end):
    rng = np.random.default_rng(17)
    keys = tuple(
        KeyframeInfo(i, str(i), float(time))
        for i, time in enumerate(sorted(rng.uniform(0, 100, count)))
    )
    priority = keys[1:4]
    actual = visible_keyframe_positions(keys, start, end, 150, 950, 7, 10.5, priority)
    positions = {
        key.keyframe_id: timeline_time_to_x(key.time, start, end, 150, 950) for key in keys
    }
    expected = decimated_marker_ids(
        tuple(positions.items()), 143, 957, 10.5, [key.keyframe_id for key in priority]
    )
    assert actual.keys() == dict.fromkeys(expected).keys()
    for key_id, x in actual.items():
        assert x == pytest.approx(positions[key_id])


def test_dense_marker_query_is_bounded_and_hover_can_pick_an_undrawn_key():
    class Keys:
        reads = 0

        def __len__(self):
            return 100000

        def __getitem__(self, index):
            self.reads += 1
            return KeyframeInfo(index, str(index), index / 30)

    keys = Keys()
    visible = visible_keyframe_positions(keys, 0, 100000 / 30, 0, 1000, 7, 10.5)
    assert len(visible) < 110
    assert keys.reads < 3000
    assert 12345 not in visible
    assert hit_keyframe(keys, 12345 / 30, 0.01) == 12345
    assert keys.reads < 3100


def test_drag_preview_replaces_original_position_without_losing_identity():
    keys = tuple(KeyframeInfo(i, str(i), i / 30) for i in range(10000))
    dragged = replace(keys[7000], time=0.5)
    positions = visible_keyframe_positions(keys, 0, 1, 0, 1000, 7, 10.5, [dragged])
    assert positions[7000] == pytest.approx(500)
    dragged = replace(keys[0], time=1000)
    assert 0 not in visible_keyframe_positions(keys, 0, 1, 0, 1000, 7, 10.5, [dragged])


def test_marker_hit_preserves_duplicate_and_equal_distance_id_order():
    keys = (KeyframeInfo(9, "a", 0), KeyframeInfo(2, "b", 2), KeyframeInfo(5, "c", 2))
    assert hit_keyframe(keys, 1, 1) == 2
    assert hit_keyframe(keys, 2, 0) == 2
    assert hit_keyframe(keys, 1, 0.5) == -1
    assert hit_keyframe((), 1, 1) == -1


@pytest.mark.parametrize("selected_id", [-1, 2, 5, 9])
@pytest.mark.parametrize("direction", [-1, 1])
def test_sorted_navigation_preserves_neighbor_identity(selected_id, direction):
    keys = (KeyframeInfo(9, "a", 0), KeyframeInfo(2, "b", 2), KeyframeInfo(5, "c", 2))
    selected = next((key for key in keys if key.keyframe_id == selected_id), None)
    for time in (-1, 0, 1, 2, 3):
        expected = neighboring_keyframe(
            tuple((key.keyframe_id, key.time) for key in keys), selected_id, time, direction
        )
        assert neighboring_sorted_keyframe(keys, selected, time, direction).keyframe_id == expected
    assert neighboring_sorted_keyframe((), None, 0, direction) is None


def test_range_selection_keeps_coincident_keys_and_exact_boundaries():
    keys = (KeyframeInfo(9, "a", 0), KeyframeInfo(2, "b", 2), KeyframeInfo(5, "c", 2))
    assert list(keyframe_range(keys, 2, 2)) == [1, 2]
    assert list(keyframe_range(keys, 0.1, 1.9)) == []
