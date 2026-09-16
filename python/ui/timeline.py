"""Timeline range, sample lookup and screen-space marker selection without a UI backend."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from operator import attrgetter

from mojive.adapters.base import KeyframeInfo

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


def fitted_timeline_range(times: tuple[float, ...], fallback: float = 0.0) -> tuple[float, float]:
    """Return a padded, finite range with enough context around isolated keys."""

    finite = tuple(float(value) for value in times if math.isfinite(value))
    if not finite:
        center = float(fallback) if math.isfinite(fallback) else 0.0
        return min(0.0, center), max(1.0, center + 0.5)
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-9:
        context = max(1.0, abs(lo) * 0.1)
        return max(0.0, lo - context * 0.5) if lo >= 0 else lo - context * 0.5, hi + context * 0.5
    padding = (hi - lo) * 0.08
    return max(0.0, lo - padding) if lo >= 0 else lo - padding, hi + padding


def nice_timeline_step(span: float, pixel_width: float, target_pixels: float = 90.0) -> float:
    """Choose a stable 1/2/5 ruler step for the visible time range."""

    if not math.isfinite(span) or not math.isfinite(pixel_width) or pixel_width <= 0.0:
        return 1.0
    raw = max(float(span), _MIN_TIMELINE_SPAN) * target_pixels / pixel_width
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10.0**exponent)
    nice = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0 if fraction <= 5.0 else 10.0
    return nice * (10.0**exponent)


def zoom_timeline_range(
    start: float, end: float, anchor: float, wheel: float
) -> tuple[float, float]:
    """Zoom around ``anchor`` while keeping it at the same screen position."""

    span = max(float(end) - float(start), _MIN_TIMELINE_SPAN)
    ratio = min(1.0, max(0.0, (float(anchor) - float(start)) / span))
    new_span = min(1e12, max(_MIN_TIMELINE_SPAN, span * math.exp(-float(wheel) * 0.18)))
    new_start = float(anchor) - ratio * new_span
    return new_start, new_start + new_span


def timeline_channel_width(available: float, scale: float) -> float:
    return min(150 * scale, available * 0.35)


def timeline_time_to_x(time: float, start: float, end: float, lo: float, hi: float) -> float:
    span = max(float(end) - float(start), _MIN_TIMELINE_SPAN)
    return float(lo) + (float(time) - float(start)) * (float(hi) - float(lo)) / span


def timeline_x_to_time(x: float, start: float, end: float, lo: float, hi: float) -> float:
    width = max(float(hi) - float(lo), 1e-9)
    return float(start) + (float(x) - float(lo)) * (float(end) - float(start)) / width


def follow_timeline_range(
    start: float, end: float, playhead: float, mode: str, locked_fraction: float = 0.15
) -> tuple[float, float]:
    """Page an escaped playhead forward, or hold its relative screen position."""
    span = max(end - start, _MIN_TIMELINE_SPAN)
    if mode == "locked":
        start = playhead - min(0.95, max(0.05, locked_fraction)) * span
    elif mode == "page" and not start + span * 0.01 <= playhead <= end - span * 0.01:
        start = playhead - span * 0.08
    else:
        return start, end
    return start, start + span


def neighboring_keyframe(
    markers: tuple[tuple[int, float], ...],
    selected_id: int,
    playhead: float,
    direction: int,
) -> int:
    """Return the adjacent marker ID, using the playhead when none is selected."""

    if not markers or direction == 0:
        return -1
    ordered = sorted(markers, key=lambda marker: (marker[1], marker[0]))
    for slot, marker in enumerate(ordered):
        if marker[0] == selected_id:
            adjacent = slot + (1 if direction > 0 else -1)
            return ordered[min(len(ordered) - 1, max(0, adjacent))][0]
    if direction > 0:
        return next((key_id for key_id, time in ordered if time > playhead), ordered[-1][0])
    return next((key_id for key_id, time in reversed(ordered) if time < playhead), ordered[0][0])


def nearest_take_frame(times: Sequence[float], time: float) -> int:
    """Return the nearest chronological take frame without scanning the full recording."""

    if not times:
        return -1
    slot = bisect.bisect_left(times, float(time))
    if slot <= 0:
        return 0
    if slot >= len(times):
        return len(times) - 1
    return slot - 1 if abs(times[slot - 1] - time) <= abs(times[slot] - time) else slot


def recorded_take_spans(times: Sequence[float], start: float, end: float, lo: float, hi: float):
    """Merge overlapping sample marks with work bounded by visible pixel columns."""
    first = bisect.bisect_left(times, start)
    last = bisect.bisect_right(times, end)
    if first == last:
        return
    factor = (hi - lo) / max(end - start, _MIN_TIMELINE_SPAN)
    if factor <= 0:
        return
    left = right = None
    index = first
    while index < last:
        x = lo + (times[index] - start) * factor
        # Marks in one pixel column overlap. Their first/last samples describe
        # the exact union; striding by sample count can erase isolated frames.
        boundary = start + (math.floor(x - lo) + 1) / factor
        following = bisect.bisect_left(times, boundary, lo=index + 1, hi=last)
        last_x = lo + (times[following - 1] - start) * factor
        mark_left, mark_right = max(lo, x - 1.0), min(hi, last_x + 1.0)
        if right is not None and mark_left > right:
            yield left, right
            left = None
        if left is None:
            left = mark_left
        right = mark_right
        index = following
    yield left, right


def decimated_marker_ids(
    markers: Sequence[tuple[int, float]],
    lo: float,
    hi: float,
    min_spacing: float,
    priority_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Bound overlapping marker draws while retaining interactive priority markers."""

    spacing = max(float(min_spacing), 1.0)
    visible = tuple((keyframe_id, x) for keyframe_id, x in markers if lo <= x <= hi)
    capacity = max(1, int(max(0.0, hi - lo) // spacing) + 1)
    if len(visible) <= capacity:
        return tuple(keyframe_id for keyframe_id, _x in visible)
    priority = set(priority_ids)
    buckets: dict[int, int] = {}
    priority_visible: list[int] = []
    for keyframe_id, x in visible:
        if keyframe_id in priority:
            priority_visible.append(keyframe_id)
            continue
        buckets.setdefault(int((x - lo) // spacing), keyframe_id)
    return (*buckets.values(), *dict.fromkeys(priority_visible))
