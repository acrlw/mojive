"""Authored gaps remain open after the camera/delete masters' visible stroke."""

import math
from itertools import pairwise

import pytest
from examples.ui_design.native.compile_icons import ROOT, compile_svg, library_masters


def _master(name):
    return compile_svg((ROOT / "svg" / "mojive" / f"{name}.svg").read_text())


def _point_distance(point, start, end):
    delta = tuple(b - a for a, b in zip(start, end, strict=True))
    length2 = sum(v * v for v in delta)
    t = sum((p - a) * d for p, a, d in zip(point, start, delta, strict=True)) / length2
    t = max(0, min(1, t))
    return math.dist(point, tuple(a + t * d for a, d in zip(start, delta, strict=True)))


def _segments_distance(a, b, c, d):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    ab, cd, ac = ((q[0] - p[0], q[1] - p[1]) for p, q in ((a, b), (c, d), (a, c)))
    denominator = cross(ab, cd)
    if abs(denominator) > 1e-12:
        t, u = cross(ac, cd) / denominator, cross(ac, ab) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0
    return min(
        _point_distance(a, c, d),
        _point_distance(b, c, d),
        _point_distance(c, a, b),
        _point_distance(d, a, b),
    )


def _visible_gap(first, second):
    distance = min(
        _segments_distance(a, b, c, d)
        for a, b in pairwise(first["points"])
        for c, d in pairwise(second["points"])
    )
    return distance - (first["width"] + second["width"]) / 2


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_isometric_arrow_keeps_clearance_from_both_open_frame_ends(size):
    frame, shaft, head = _master("view-iso")["contours"]
    for arrow in (shaft, head):
        # The old five-unit arrow arms left only 0.5 grid units of visible air.
        # This budget includes both strokes and scales with the whole glyph.
        assert _visible_gap(frame, arrow) * size / 24 >= 2.5 * size / 24 - 1e-6


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_delete_handle_lid_and_body_have_no_intersecting_end_caps(size):
    lid, body, first_slot, second_slot = _master("trash")["contours"]
    # The handle is part of the lid's single continuous path. Its only open
    # endpoints are outside the can, so no handle cap can protrude below the lid.
    assert lid["points"][0] == (3.5, 6.5)
    assert lid["points"][-1] == (20.5, 6.5)
    assert min(point[1] for point in lid["points"]) == 3
    assert max(point[1] for point in lid["points"]) == 6.5
    assert _visible_gap(lid, body) * size / 24 >= 2 * size / 24 - 1e-6
    for slot in (first_slot, second_slot):
        assert _visible_gap(body, slot) * size / 24 >= 1.75 * size / 24 - 1e-6
    assert _visible_gap(first_slot, second_slot) >= 2.5


@pytest.mark.parametrize("name", ("view-iso", "trash"))
def test_local_camera_and_delete_masters_keep_the_family_weight_and_safe_area(name):
    glyph = _master(name)
    assert glyph["viewBox"] == [0, 0, 24, 24]
    assert min(glyph["bounds"]) >= 2
    assert max(glyph["bounds"]) <= 22
    for contour in glyph["contours"]:
        assert (contour["width"], contour["cap"], contour["join"]) == (1.5, "round", "round")


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_gauge_pointer_has_visible_clearance_from_its_frame(size):
    frame, mark, _hub = _master("performance")["contours"]
    assert _visible_gap(frame, mark) * size / 24 >= 1.5 * size / 24 - 1e-6


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_pose_object_keeps_a_clear_gap_below_the_rotation_arrow(size):
    glyph = _master("pose")
    body, arc, head = glyph["contours"]
    assert body["closed"]
    assert not arc["closed"] and not head["closed"]
    for contour in glyph["contours"]:
        assert (contour["width"], contour["cap"], contour["join"]) == (1.5, "round", "round")
    # Include the closing segment when measuring the whole expanded object.
    closed_body = dict(body, points=[*body["points"], body["points"][0]])
    for contour in (arc, head):
        assert _visible_gap(closed_body, contour) * size / 24 >= 2.5 * size / 24 - 1e-6
    # The arc terminates at the arrow tip without overshooting its round join.
    assert arc["points"][-1] == pytest.approx(head["points"][1])
    sources = library_masters()
    for name in ("shape-box", "view-iso", "rotate", "joint-free"):
        assert glyph != compile_svg(sources[name])


@pytest.mark.parametrize("size", (14, 16, 18, 24, 56, 112))
def test_sensor_reception_arcs_do_not_merge_with_the_sensing_point(size):
    left, right, point = _master("sensor")["contours"]
    assert point["fill"] == "currentColor" and point["stroke"] == "none"
    for arc in (left, right):
        assert _visible_gap(arc, point) * size / 24 >= 5.5 * size / 24 - 1e-6
