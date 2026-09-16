"""UI-independent timeline coordinates, marker queries, and bounded projection caches."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

MIN_TIMELINE_SPAN = 1e-6


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
    raw = max(float(span), MIN_TIMELINE_SPAN) * target_pixels / pixel_width
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10.0**exponent)
    nice = 1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0 if fraction <= 5.0 else 10.0
    return nice * (10.0**exponent)


def zoom_timeline_range(
    start: float, end: float, anchor: float, wheel: float
) -> tuple[float, float]:
    """Zoom around ``anchor`` while keeping it at the same screen position."""

    span = max(float(end) - float(start), MIN_TIMELINE_SPAN)
    ratio = min(1.0, max(0.0, (float(anchor) - float(start)) / span))
    new_span = min(1e12, max(MIN_TIMELINE_SPAN, span * math.exp(-float(wheel) * 0.18)))
    new_start = float(anchor) - ratio * new_span
    return new_start, new_start + new_span


def timeline_channel_width(available: float, scale: float) -> float:
    return min(150 * scale, available * 0.35)


def timeline_time_to_x(time: float, start: float, end: float, lo: float, hi: float) -> float:
    span = max(float(end) - float(start), MIN_TIMELINE_SPAN)
    return float(lo) + (float(time) - float(start)) * (float(hi) - float(lo)) / span


def timeline_x_to_time(x: float, start: float, end: float, lo: float, hi: float) -> float:
    width = max(float(hi) - float(lo), 1e-9)
    return float(start) + (float(x) - float(lo)) * (float(end) - float(start)) / width


def follow_timeline_range(
    start: float, end: float, playhead: float, mode: str, locked_fraction: float = 0.15
) -> tuple[float, float]:
    """Page an escaped playhead forward, or hold its relative screen position."""
    span = max(end - start, MIN_TIMELINE_SPAN)
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
    factor = (hi - lo) / max(end - start, MIN_TIMELINE_SPAN)
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


class MarkerProjection:
    """One view's marker geometry; selection and document state stay with their owners."""

    def __init__(self, positions: dict[int, float], *, chronological: bool = True, order=None):
        self.positions: Mapping[int, float] = MappingProxyType(positions)
        self._order = order if order is not None else {key: i for i, key in enumerate(positions)}
        self._chronological = chronological
        if chronological:
            self._xs = tuple(positions.values())
            self._ids = tuple(positions)
        else:
            ordered = sorted((x, key_id) for key_id, x in positions.items())
            self._xs = tuple(x for x, _ in ordered)
            self._ids = tuple(key_id for _, key_id in ordered)
        self._draw_key = None
        self._draw_ids: tuple[int, ...] = ()

    def hit(self, x: float, radius: float, *, excluded: int | None = None) -> int:
        """Find the nearest marker, breaking equal-distance ties by object identity."""
        slot = bisect.bisect_left(self._xs, x)
        candidates = []
        right = slot
        while right < len(self._xs):
            end = bisect.bisect_right(self._xs, self._xs[right], right + 1)
            ids = self._ids[right:end]
            key_id = min((key for key in ids if key != excluded), default=None)
            if key_id is not None:
                candidates.append((abs(self._xs[right] - x), key_id))
                break
            right = end
        while slot:
            previous = bisect.bisect_left(self._xs, self._xs[slot - 1])
            ids = self._ids[previous:slot]
            key_id = min((key for key in ids if key != excluded), default=None)
            if key_id is not None:
                candidates.append((abs(self._xs[previous] - x), key_id))
                break
            slot = previous
        distance, key_id = min(candidates, default=(math.inf, -1))
        return key_id if distance <= radius else -1

    def draw_ids(
        self, lo: float, hi: float, spacing: float, priority_ids: Sequence[int] = ()
    ) -> tuple[int, ...]:
        """Reuse unchanged decimation; priority markers remain fully interactive."""
        key = (lo, hi, spacing, tuple(priority_ids))
        if key != self._draw_key:
            if self._chronological:
                self._draw_ids = self._indexed_draw_ids(lo, hi, spacing, priority_ids)
            else:
                # A drag can reorder one marker. Preserve the existing preview draw order.
                self._draw_ids = decimated_marker_ids(
                    tuple(self.positions.items()), lo, hi, spacing, priority_ids
                )
            self._draw_key = key
        return self._draw_ids

    def _indexed_draw_ids(self, lo, hi, spacing, priority_ids):
        spacing = max(float(spacing), 1.0)
        first = bisect.bisect_left(self._xs, lo)
        last = bisect.bisect_right(self._xs, hi)
        capacity = max(1, int(max(0.0, hi - lo) // spacing) + 1)
        if last - first <= capacity:
            return self._ids[first:last]
        priority = set(priority_ids)
        priority_visible = sorted(
            (
                key_id
                for key_id in priority
                if key_id in self.positions and lo <= self.positions[key_id] <= hi
            ),
            key=self._order.__getitem__,
        )
        chosen = []
        index = first
        while index < last:
            if self._ids[index] in priority:
                index += 1
                continue
            chosen.append(self._ids[index])
            bucket = int((self._xs[index] - lo) // spacing)
            index = bisect.bisect_left(self._xs, lo + (bucket + 1) * spacing, index + 1, last)
            # Division and multiplication can round a bucket boundary differently.
            while index < last and int((self._xs[index] - lo) // spacing) == bucket:
                index = bisect.bisect_right(self._xs, self._xs[index], index + 1, last)
        return (*chosen, *priority_visible)


class _MovedPositions(Mapping[int, float]):
    """Overlay one preview position without copying the visible marker table."""

    def __init__(self, base: Mapping[int, float], key_id: int, x: float | None):
        self.base, self.key_id, self.x = base, key_id, x

    def __getitem__(self, key):
        if key != self.key_id:
            return self.base[key]
        if self.x is None:
            raise KeyError(key)
        return self.x

    def __len__(self):
        return len(self.base) - int(self.key_id in self.base) + int(self.x is not None)

    def __iter__(self):
        for key in self.base:
            if key != self.key_id or self.x is not None:
                yield key
        if self.x is not None and self.key_id not in self.base:
            yield self.key_id


class _DraggedProjection(MarkerProjection):
    """Reuse the fixed index and update only the preview and its affected draw buckets."""

    def __init__(self, base: MarkerProjection, key_id: int, x: float | None):
        self._base, self._moved, self._x = base, key_id, x
        self.positions = _MovedPositions(base.positions, key_id, x)

    def hit(self, x: float, radius: float, *, excluded: int | None = None) -> int:
        original = self._base.hit(x, radius, excluded=self._moved)
        candidates = []
        if original >= 0:
            candidates.append((abs(self._base.positions[original] - x), original))
        if self._x is not None and abs(self._x - x) <= radius:
            candidates.append((abs(self._x - x), self._moved))
        return min(candidates, default=(math.inf, -1))[1]

    def _position_order(self, key_id):
        if key_id == self._moved and key_id not in self._base.positions:
            return math.inf
        return self._base._order[key_id]

    def draw_ids(self, lo, hi, spacing, priority_ids=()):
        spacing = max(float(spacing), 1.0)
        base = self._base
        first, last = bisect.bisect_left(base._xs, lo), bisect.bisect_right(base._xs, hi)
        old = base.positions.get(self._moved)
        visible = self._x is not None and lo <= self._x <= hi
        count = last - first - int(old is not None and lo <= old <= hi) + int(visible)
        capacity = max(1, int(max(0.0, hi - lo) // spacing) + 1)
        if count <= capacity:
            keys = [key for key in base._ids[first:last] if key != self._moved]
            if visible:
                keys.append(self._moved)
            return tuple(sorted(keys, key=self._position_order))

        priority = set(priority_ids)
        # Treat the old position as excluded. Each remaining bucket contributes at
        # most one candidate, so preview work depends on screen width, not data size.
        keys = [
            key
            for key in base.draw_ids(lo, hi, spacing, (*priority_ids, self._moved))
            if key != self._moved and key not in priority
        ]
        if visible and self._moved not in priority:
            keys.append(self._moved)
        buckets = {}
        for key in sorted(keys, key=self._position_order):
            buckets.setdefault(int((self.positions[key] - lo) // spacing), key)
        selected = sorted(
            (key for key in priority if key in self.positions and lo <= self.positions[key] <= hi),
            key=self._position_order,
        )
        return (*buckets.values(), *selected)


class TimelineMarkerIndex:
    """Index immutable marker metadata and retain only the most recent view projection.

    Rebuild on content revision. Visible-range lookup is logarithmic; projection work
    is proportional to visible markers, with a pixel margin for edge hit testing.
    """

    def __init__(self, markers: Sequence[tuple[int, float]]):
        self._markers = tuple(sorted(markers, key=lambda marker: (marker[1], marker[0])))
        self._times = tuple(time for _, time in self._markers)
        self._order = {key: i for i, (key, _) in enumerate(self._markers)}
        self._view_key = None
        self._projection = MarkerProjection({})
        self._drag_key = None
        self._drag_projection = None

    def ids_between(self, start: float, end: float) -> tuple[int, ...]:
        """Return exact inclusive selection identities using the sorted time index."""
        first = bisect.bisect_left(self._times, start)
        last = bisect.bisect_right(self._times, end)
        return tuple(key for key, _ in self._markers[first:last])

    def project(
        self,
        start: float,
        end: float,
        lo: float,
        hi: float,
        margin: float = 0.0,
        *,
        moved: tuple[int, float] | None = None,
    ) -> MarkerProjection:
        """Project a visible slice, optionally substituting a transient drag preview."""
        key = (start, end, lo, hi, margin)
        if key == self._view_key:
            return self._with_preview(key, moved)
        factor = (hi - lo) / max(end - start, MIN_TIMELINE_SPAN)
        if factor <= 0.0:
            projection = MarkerProjection({})
        else:
            padding = max(0.0, margin) / factor
            first = bisect.bisect_left(self._times, start - padding)
            last = bisect.bisect_right(self._times, end + padding)
            positions = {
                key_id: timeline_time_to_x(time, start, end, lo, hi)
                for key_id, time in self._markers[first:last]
            }
            projection = MarkerProjection(positions, order=self._order)
        self._view_key, self._projection = key, projection
        self._drag_key = None
        self._drag_projection = None
        return self._with_preview(key, moved)

    def _with_preview(self, view, moved):
        if moved is None:
            return self._projection
        key = (view, moved)
        if key != self._drag_key:
            start, end, lo, hi, margin = view
            factor = (hi - lo) / max(end - start, MIN_TIMELINE_SPAN)
            padding = max(0.0, margin) / factor if factor > 0 else 0.0
            x = (
                timeline_time_to_x(moved[1], start, end, lo, hi)
                if factor > 0 and start - padding <= moved[1] <= end + padding
                else None
            )
            self._drag_projection = _DraggedProjection(self._projection, moved[0], x)
            self._drag_key = key
        return self._drag_projection
