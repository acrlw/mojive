"""Timeline range, sample lookup and screen-space marker selection without a UI backend."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from operator import attrgetter

from mojive.adapters.base import KeyframeInfo
from mojive.interaction.timeline import (
    decimated_marker_ids as decimated_marker_ids,
)
from mojive.interaction.timeline import (
    fitted_timeline_range as fitted_timeline_range,
)
from mojive.interaction.timeline import (
    follow_timeline_range as follow_timeline_range,
)
from mojive.interaction.timeline import (
    nearest_take_frame as nearest_take_frame,
)
from mojive.interaction.timeline import (
    neighboring_keyframe as neighboring_keyframe,
)
from mojive.interaction.timeline import (
    nice_timeline_step as nice_timeline_step,
)
from mojive.interaction.timeline import (
    recorded_take_spans as recorded_take_spans,
)
from mojive.interaction.timeline import (
    timeline_channel_width as timeline_channel_width,
)
from mojive.interaction.timeline import (
    timeline_time_to_x as timeline_time_to_x,
)
from mojive.interaction.timeline import (
    timeline_x_to_time as timeline_x_to_time,
)
from mojive.interaction.timeline import (
    zoom_timeline_range as zoom_timeline_range,
)

_MIN_TIMELINE_SPAN = 1e-6
_key_time = attrgetter("time")


def visible_keyframe_positions(
    keys: Sequence[KeyframeInfo],
    start: float,
    end: float,
    lo: float,
    hi: float,
    radius: float,
    spacing: float,
    priority: Sequence[KeyframeInfo] = (),
) -> dict[int, float]:
    """Query sorted keys by pixel bucket instead of scanning offscreen or overlapping keys."""
    factor = (hi - lo) / max(end - start, _MIN_TIMELINE_SPAN)
    if factor <= 0:
        return {}
    first = bisect.bisect_left(keys, start - radius / factor, key=_key_time)
    last = bisect.bisect_right(keys, end + radius / factor, lo=first, key=_key_time)
    lower, upper = lo - radius, hi + radius
    spacing = max(1.0, spacing)
    capacity = max(1, int((upper - lower) // spacing) + 1)
    positions = {}
    if last - first <= capacity:
        for index in range(first, last):
            key = keys[index]
            positions[key.keyframe_id] = lo + (key.time - start) * factor
    else:
        priority_ids = {key.keyframe_id for key in priority}
        buckets = {}
        index = first
        while index < last:
            key = keys[index]
            x = lo + (key.time - start) * factor
            bucket = int((x - lower) // spacing)
            if key.keyframe_id in priority_ids:
                index += 1
                continue
            buckets.setdefault(bucket, (key.keyframe_id, x))
            boundary = start + ((bucket + 1) * spacing - radius) / factor
            index = bisect.bisect_left(keys, boundary, lo=index + 1, hi=last, key=_key_time)
        positions.update(buckets.values())
    for key in priority:
        # A dragging key may have moved outside its original bucket or visible range.
        positions.pop(key.keyframe_id, None)
        x = lo + (key.time - start) * factor
        if lower <= x <= upper:
            positions[key.keyframe_id] = x
    return positions


def hit_keyframe(keys: Sequence[KeyframeInfo], time: float, tolerance: float) -> int:
    """Pick the nearest sorted key, including keys omitted from dense marker drawing."""
    slot = bisect.bisect_left(keys, time, key=_key_time)
    candidates = []
    if slot < len(keys):
        candidates.append(keys[slot])
    if slot:
        # Coincident keys retain the same time/ID order as the unthinned marker list.
        index = bisect.bisect_left(keys, keys[slot - 1].time, hi=slot, key=_key_time)
        candidates.append(keys[index])
    key = min(candidates, key=lambda key: (abs(key.time - time), key.keyframe_id), default=None)
    return key.keyframe_id if key is not None and abs(key.time - time) <= tolerance else -1


def keyframe_range(keys: Sequence[KeyframeInfo], start: float, end: float) -> range:
    """Return sorted key indices within an inclusive time range."""
    first = bisect.bisect_left(keys, start, key=_key_time)
    return range(first, bisect.bisect_right(keys, end, lo=first, key=_key_time))


def neighboring_sorted_keyframe(
    keys: Sequence[KeyframeInfo], selected: KeyframeInfo | None, playhead: float, direction: int
) -> KeyframeInfo | None:
    """Navigate a cached time/ID ordering without rebuilding or sorting it."""
    if not keys:
        return None
    if selected is not None:
        slot = bisect.bisect_left(
            keys, (selected.time, selected.keyframe_id), key=lambda key: (key.time, key.keyframe_id)
        ) + (1 if direction > 0 else -1)
    elif direction > 0:
        slot = bisect.bisect_right(keys, playhead, key=_key_time)
    else:
        slot = bisect.bisect_left(keys, playhead, key=_key_time) - 1
    return keys[min(len(keys) - 1, max(0, slot))]
