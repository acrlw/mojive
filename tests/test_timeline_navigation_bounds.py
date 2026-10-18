"""Timeline navigation remains finite and useful without narrowing stored content."""

import math
from types import SimpleNamespace

import pytest

from mojive.adapters.base import KeyframeInfo
from mojive.interaction.timeline import (
    clamp_timeline_range,
    nice_timeline_step,
    pan_timeline_range,
    timeline_navigation_limits,
    timeline_range_format,
    valid_timeline_range,
    zoom_timeline_range,
)
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.keyframe_editor.track import TimelineInput, TimelineLayout, _update_timeline_view
from mojive.ui.pointer_bindings import PointerFrame


@pytest.mark.parametrize("wheel", (-1e300, -1.0, 1.0, 1e300))
def test_repeated_zoom_saturates_between_milliseconds_and_content_context(wheel):
    limits = timeline_navigation_limits((0.0, 0.15))
    start, end = 0.0, 0.15
    for _ in range(500):
        anchor = start + (end - start) * 0.35
        start, end = zoom_timeline_range(
            start, end, anchor, wheel, max_span=limits.max_span, min_span=limits.min_span
        )
        assert valid_timeline_range(start, end)
        assert limits.min_span - 1e-12 <= end - start <= limits.max_span + 1e-12
        assert (anchor - start) / (end - start) == pytest.approx(0.35)
    assert end - start == pytest.approx(60 if wheel < 0 else 0.001)
    format = timeline_range_format(start, end)
    assert format % start != format % end


@pytest.mark.parametrize("value", (math.inf, -math.inf, math.nan))
def test_nonfinite_zoom_and_pan_deltas_do_not_poison_a_valid_view(value):
    limits = timeline_navigation_limits((0, 10))
    assert zoom_timeline_range(0, 10, 5, value) == (0, 10)
    assert zoom_timeline_range(0, 10, value, 1) == (0, 10)
    assert pan_timeline_range(0, 10, value, limits) == (0, 10)
    assert math.isfinite(nice_timeline_step(value, 800))


def test_old_runaway_view_recovers_near_content_and_pan_is_reversible_within_bounds():
    limits = timeline_navigation_limits((0, 0.15))
    start, end = clamp_timeline_range(-4.99e7, 9.63e7, limits)
    assert end - start == pytest.approx(60)
    assert limits.lower <= start < end <= limits.upper
    assert start <= 0 <= 0.15 <= end
    recovered = clamp_timeline_range(math.nan, math.inf, limits)
    assert valid_timeline_range(*recovered)
    assert recovered[0] <= 0 <= 0.15 <= recovered[1]
    assert recovered[1] - recovered[0] == pytest.approx(60)
    assert pan_timeline_range(0, 1, 2.5, limits) == (2.5, 3.5)
    assert pan_timeline_range(2.5, 3.5, -2.5, limits) == (0, 1)
    for direction in (-1, 1):
        a, b = pan_timeline_range(0, 1, direction * 1e300, limits)
        assert limits.lower <= a < b <= limits.upper
        assert b - a == pytest.approx(1)


@pytest.mark.parametrize("times", ((0, 7200), (1e8, 1e8 + 120)))
def test_long_authored_ranges_and_large_offsets_keep_their_timestamps(times):
    limits = timeline_navigation_limits(times)
    assert limits.max_span >= times[1] - times[0]
    assert clamp_timeline_range(*times, limits) == times
    a, b = zoom_timeline_range(*times, sum(times) / 2, -1e300, max_span=limits.max_span)
    assert a < times[0] < times[1] < b
    assert all(math.isfinite(value) for value in (a, b))


def test_manual_long_and_fine_ranges_remain_available_until_fit_or_model_change():
    editor = TimelineEditor()
    assert editor.set_view_range(0, 7200)
    limits = timeline_navigation_limits((0, 0.15), editor.manual_view_range)
    assert clamp_timeline_range(editor.view_start, editor.view_end, limits) == (0, 7200)
    for bad in ((math.nan, 1), (0, math.inf), (0, -1), (0, 1e-9), (-1e308, 1e308)):
        assert not editor.set_view_range(*bad)
        assert editor.manual_view_range == (0, 7200)
    assert editor.set_view_range(0, 0.00001)
    limits = timeline_navigation_limits((0,), editor.manual_view_range)
    assert limits.min_span == pytest.approx(0.00001)
    start, end = zoom_timeline_range(0, 0.00001, 0.000005, 100, min_span=limits.min_span)
    assert end - start == pytest.approx(0.00001)
    format = timeline_range_format(start, end, suffix=" s")
    assert format % start != format % end
    editor.set_model(1)
    assert editor.manual_view_range is None


@pytest.mark.parametrize("anchor", (0.25, 0.5, 1.2501, 10.25))
def test_normal_wheel_floor_keeps_three_decimals_despite_subtraction_rounding(anchor):
    start, end = anchor - 0.5, anchor + 0.5
    for _ in range(40):
        start, end = zoom_timeline_range(start, end, anchor, 100000)
    format = timeline_range_format(start, end, suffix=" s")
    assert end - start == pytest.approx(0.001)
    assert format == "%.3f s"
    assert format % start != format % end


def test_wheel_floor_increases_precision_when_millisecond_captions_really_coincide():
    start, end = 0.75, 1.75
    for _ in range(40):
        start, end = zoom_timeline_range(start, end, 1.25, 100000)
    assert f"{start:.3f}" == f"{end:.3f}"
    format = timeline_range_format(start, end)
    assert format == "%.4f"
    assert format % start != format % end


@pytest.mark.parametrize("offset", (0, 0.25, 0.5, 1.25, 10.25, 1000))
@pytest.mark.parametrize("span,precision", ((0.0001, 4), (0.00001, 5)))
def test_manual_decimal_spans_do_not_gain_an_extra_digit_from_rounding(offset, span, precision):
    end = offset + span
    assert valid_timeline_range(offset, end)
    format = timeline_range_format(offset, end)
    assert format == f"%.{precision}f"
    assert format % offset != format % end


@pytest.mark.parametrize("span,precision", ((0.000999, 4), (0.000009999, 6)))
def test_genuinely_finer_spans_keep_the_precision_below_the_decimal_boundary(span, precision):
    start, end = 1.25, 1.25 + span
    format = timeline_range_format(start, end)
    assert format == f"%.{precision}f"
    assert format % start != format % end


def test_endpoint_ulp_tolerance_cannot_make_two_valid_values_display_identically():
    offset = 1e12
    start = offset + 5 * math.ulp(offset)
    end = start + 7 * math.ulp(offset)
    assert f"{start:.3f}" == f"{end:.3f}"
    format = timeline_range_format(start, end)
    assert format % start != format % end


def test_fine_range_straddling_zero_does_not_rely_on_the_sign_of_rounded_zero():
    start, end = -0.0000005, 0.0000005
    format = timeline_range_format(start, end)
    assert format == "%.7f"
    assert float(format % start) != float(format % end)


def test_zoom_at_large_offset_stays_above_floating_point_resolution():
    start, end = 1e16, 1e16 + 128
    for _ in range(200):
        start, end = zoom_timeline_range(start, end, (start + end) / 2, 4, max_span=1024)
        assert valid_timeline_range(start, end)
        assert end - start >= 8 * math.ulp(start)


@pytest.mark.parametrize("offset", (-1e308, 1e308))
def test_an_extreme_finite_content_offset_still_has_a_representable_view(offset):
    limits = timeline_navigation_limits((offset,))
    start, end = clamp_timeline_range(offset, offset, limits)
    assert valid_timeline_range(start, end)
    assert start <= offset <= end
    assert math.isfinite(nice_timeline_step(end - start, 1))


def test_production_view_update_bounds_wheel_recovers_legacy_state_and_uses_take_endpoints():
    class Times:
        reads = 0

        def __len__(self):
            return 100_000

        def __getitem__(self, index):
            assert index in (0, -1)
            self.reads += 1
            return 0.0 if index == 0 else 0.15

    times = Times()
    editor = TimelineEditor()
    editor.view_start, editor.view_end, editor.view_needs_fit = -4.99e7, 9.63e7, False
    editor.view_model_id = editor.model_id
    editor.follow_mode = "off"
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            scene_snapshots=(), state_take_playing=False, state_take_recording=False
        )
    )
    layout = TimelineLayout((0, 0), 1000, 200, 30, 150, 7)
    inputs = TimelineInput(
        (500, 50),
        True,
        True,
        1,
        PointerFrame(wheel=-1e300),
        DEFAULT_INPUT_BINDINGS,
        None,
        None,
        None,
        None,
        False,
        False,
        -1e300,
        0,
        6,
    )
    for _ in range(200):
        _update_timeline_view(editor, ctx, layout, inputs, (KeyframeInfo(0, "key", 0),), times)
        limits = timeline_navigation_limits((0, 0.15))
        assert limits.lower <= editor.view_start < editor.view_end <= limits.upper
        assert editor.view_end - editor.view_start <= 60 + 1e-9
    assert times.reads == 400


@pytest.mark.parametrize(
    "mode,fraction", (("locked", 0.05), ("locked", 0.15), ("locked", 0.95), ("page", 0.08))
)
def test_follow_then_wheel_keeps_the_playhead_fraction_at_the_zoom_limit(mode, fraction):
    editor = TimelineEditor()
    editor.view_needs_fit = False
    editor.view_model_id = editor.model_id
    editor.view_start, editor.view_end = (50, 110) if mode == "page" else (0, 60)
    editor.follow_mode, editor.locked_fraction = mode, fraction
    editor.playhead = 0.1
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            scene_snapshots=(), state_take_playing=True, state_take_recording=False
        )
    )
    layout = TimelineLayout((0, 0), 1000, 200, 30, 150, 7)

    def inputs(wheel):
        return TimelineInput(
            (
                layout.time_bounds[0] + (layout.time_bounds[1] - layout.time_bounds[0]) * fraction,
                50,
            ),
            True,
            True,
            1,
            PointerFrame(wheel=wheel),
            DEFAULT_INPUT_BINDINGS,
            None,
            None,
            None,
            None,
            False,
            False,
            wheel,
            0,
            6,
        )

    keys = (KeyframeInfo(0, "start", 0), KeyframeInfo(1, "end", 0.15))
    _update_timeline_view(editor, ctx, layout, inputs(0), keys, ())
    assert (editor.playhead - editor.view_start) / (
        editor.view_end - editor.view_start
    ) == pytest.approx(fraction)
    for wheel in (-100000, 1, -1, 0, -100000):
        before = editor.view_end - editor.view_start
        _update_timeline_view(editor, ctx, layout, inputs(wheel), keys, ())
        span = editor.view_end - editor.view_start
        assert (editor.playhead - editor.view_start) / span == pytest.approx(fraction)
        assert span <= 60 + 1e-9
        if wheel > 0:
            assert span < before
