"""Compact joint summaries preserve selection, precision and local expansion."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import relative_luminance
from mojive.ui.workspace.widgets import MUTED, TEXT
from tests.test_ui_design_pose import editor as editor
from tests.test_ui_design_pose import session as session

pytestmark = pytest.mark.physics


def _summary(editor, kind="free"):
    editor.kind = kind
    editor.collapsible_poses = editor.show_names = True
    editor.frame(3)
    return f"pose-joint-{editor.joint.joint_id}-expand"


def _click(editor, key):
    editor.event("add_mouse_pos_event", *editor.point(key))
    editor.event("add_mouse_button_event", 0, True)
    editor.event("add_mouse_button_event", 0, False)
    editor.frame(2)


@pytest.mark.parametrize("width", (160, 280, 640))
@pytest.mark.parametrize("kind", ("ball", "free"))
def test_folded_pose_retains_stable_contained_numeric_columns(editor, width, kind):
    editor.width = width
    _summary(editor, kind)
    joint = editor.joint
    prefix = f"pose-summary-{joint.joint_id}-"
    first = {key: rect for key, rect in editor.ui.hits.items() if key.startswith(prefix)}
    assert len(first) == (6 if kind == "free" else 3)
    initial = editor.session.adapter.primary.data.qpos.copy()
    if kind == "free":
        editor.session.adapter.primary.data.qpos[joint.qpos_adr : joint.qpos_adr + 3] = (
            1e9,
            -1e-9,
            0.123456789,
        )
    else:
        editor.session.adapter.primary.data.qpos[joint.qpos_adr : joint.qpos_adr + 4] = (
            0.5,
            0.5,
            0.5,
            0.5,
        )
    editor.frame(2)
    assert {key: rect for key, rect in editor.ui.hits.items() if key.startswith(prefix)} == first
    x, y, w, h = editor.ui.hits[f"pose-fields-{joint.joint_id}"]
    for cx, cy, cw, ch in first.values():
        assert x <= cx < cx + cw <= x + w + 1e-5
        assert y <= cy < cy + ch <= y + h + 1e-5
    assert not any(key.startswith(f"##pose-joint-{joint.joint_id}-") for key in editor.ui.hits)
    editor.session.adapter.primary.data.qpos[:] = initial


def test_folded_pose_can_expand_and_edit_without_losing_precision_or_history(editor):
    key = _summary(editor)
    collapsed = editor.bottom
    _click(editor, key)
    assert editor.bottom > collapsed
    field = f"##pose-joint-{editor.joint.joint_id}-positionX"
    value = 0.123456789
    before = editor.session.adapter.primary.data.qpos.copy()
    editor.text(field, value)
    assert editor.session.adapter.primary.data.qpos[editor.joint.qpos_adr] == pytest.approx(
        value, abs=1e-8
    )
    assert editor.session.submit(cmd.Undo()).ok
    np.testing.assert_array_equal(editor.session.adapter.primary.data.qpos, before)
    assert not editor.session.can_undo
    _click(editor, key)
    assert field not in editor.ui.hits
    _click(editor, key)
    assert field in editor.ui.hits


def test_selecting_a_folded_joint_expands_it_but_explicit_collapse_remains_local(editor):
    key = _summary(editor, "ball")
    joint = editor.joint
    field = f"##pose-joint-{joint.joint_id}-rotationX"
    assert field not in editor.ui.hits
    node = next(node for node in editor.session.nodes if node.joint_index == joint.joint_id)
    assert editor.session.submit(cmd.SelectNode(node.node_id)).ok
    editor.frame(2)
    assert field in editor.ui.hits
    _click(editor, key)
    editor.frame(3)
    assert field not in editor.ui.hits
    # Inspector still renders its direct controls and does not inherit the list fold.
    editor.duplicate = True
    editor.frame(2)
    assert field not in editor.primary_hits
    assert field in editor.ui.hits


def test_pose_expansion_remains_usable_without_write_capability(editor):
    editor.session.adapter.caps = replace(
        editor.session.adapter.caps, write_qpos=False, edit_history=False
    )
    key = _summary(editor)
    before = editor.session.adapter.primary.data.qpos.copy()
    _click(editor, key)
    assert f"##pose-joint-{editor.joint.joint_id}-positionX" in editor.ui.hits
    _click(editor, key)
    assert f"pose-summary-{editor.joint.joint_id}-position-X" in editor.ui.hits
    np.testing.assert_array_equal(editor.session.adapter.primary.data.qpos, before)
    assert not editor.session.editing


@pytest.mark.parametrize("kind", ("hinge", "slide"))
def test_scalar_joint_controls_do_not_gain_a_disclosure_or_summary(editor, kind):
    key = _summary(editor, kind)
    assert key not in editor.ui.hits
    assert f"##pose-joint-{editor.joint.joint_id}-value" in editor.ui.hits


def test_pose_summary_group_overrides_do_not_change_sibling_joint_groups(editor, tmp_path):
    editor.ui.design = DesignPresentation(
        editor.ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    )
    _summary(editor)
    prefix = f"pose-summary-{editor.joint.joint_id}-"
    with group_scope("Pose"):
        editor.frame(2)
        before = {key: rect for key, rect in editor.ui.hits.items() if key.startswith(prefix)}
    with group_scope("Inspector"):
        editor.frame(2)
        peer = {key: rect for key, rect in editor.ui.hits.items() if key.startswith(prefix)}
    spec = deepcopy(editor.ui.design.runtime.spec)
    spec["parts"]["pose_joint"] = {"fontSize": 20, "paddingLeft": 18, "rowGap": 12}
    editor.ui.design.runtime.update_spec(spec)
    with group_scope("Pose"):
        editor.frame(2)
        after = {key: rect for key, rect in editor.ui.hits.items() if key.startswith(prefix)}
    with group_scope("Inspector"):
        editor.frame(2)
        assert {key: rect for key, rect in editor.ui.hits.items() if key.startswith(prefix)} == peer
    assert after != before
    outer_x = editor.ui.hits[f"pose-fields-{editor.joint.joint_id}"][0]
    assert next(iter(after.values()))[0] >= outer_x + 18 * editor.ui.s
    assert next(iter(after.values()))[3] > next(iter(before.values()))[3]


def test_heading_gap_is_pickable_adjustable_and_local_to_pose(editor, tmp_path):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    runtime.observe_components = True
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    _summary(editor)
    key = f"pose-joint-{editor.joint.joint_id}"

    def bounds(owner):
        runtime.observed.clear()
        with group_scope(owner):
            editor.frame(1)
        return editor.ui.hits[key], editor.ui.hits[key + "-expand"]

    before_name, before_fold = bounds("Pose")
    before_gap = before_fold[0] - before_name[0] - before_name[2]
    assert before_gap >= 8 * editor.ui.s
    peer = bounds("Inspector")
    spec = deepcopy(runtime.spec)
    spec["parts"]["pose_joint"] = {"columnGap": 18}
    runtime.update_spec(spec)
    name, fold = bounds("Pose")
    assert fold == before_fold
    gap = fold[0] - name[0] - name[2]
    assert gap == pytest.approx(18 * editor.ui.s)
    record = next(row for row in runtime.observed if row["key"].startswith("joint-card-"))
    gx, gy, gw, gh = record["fieldRegions"]["columnGap"]
    assert gx == pytest.approx(name[0] + name[2])
    assert gy == pytest.approx(name[1])
    assert gw == pytest.approx(gap)
    assert gh == pytest.approx(name[3])
    assert bounds("Inspector") == peer

    bounds("Pose")
    summary = f"pose-summary-{editor.joint.joint_id}-position-X"
    first_y = editor.ui.hits[summary][1]
    spec["parts"]["pose_joint"]["rowGap"] = 18
    runtime.update_spec(spec)
    name, fold = bounds("Pose")
    assert fold[0] - name[0] - name[2] == pytest.approx(gap)
    assert editor.ui.hits[summary][1] - first_y == pytest.approx(10 * editor.ui.s)
    record = next(row for row in runtime.observed if row["key"].startswith("joint-card-"))
    assert record["fieldRegions"]["rowGap"][3] == pytest.approx(18 * editor.ui.s)
    _click(editor, key + "-expand")
    assert f"##pose-joint-{editor.joint.joint_id}-positionX" in editor.ui.hits


@pytest.mark.parametrize("width", (160, 280, 640))
def test_folded_axis_markers_keep_labels_and_neutral_readings(editor, monkeypatch, width):
    editor.width = width
    _summary(editor)
    markers, captions = [], []
    draw_rect, draw_text = editor.ui.rect, editor.ui.component_text

    def rect(x, y, width, height, color, *args, **kwargs):
        markers.append((x, y, width, height, color))
        return draw_rect(x, y, width, height, color, *args, **kwargs)

    def text(part, bounds, caption, color, **kwargs):
        captions.append((bounds, caption, color))
        return draw_text(part, bounds, caption, color, **kwargs)

    monkeypatch.setattr(editor.ui, "rect", rect)
    monkeypatch.setattr(editor.ui, "component_text", text)
    editor.frame(1)
    value_colors = []
    cells = []
    for index, axis in enumerate("XYZ"):
        x, y, width, height = editor.ui.hits[
            f"pose-summary-{editor.joint.joint_id}-position-{axis}"
        ]
        cells.append((x, y))
        marker = next(
            item
            for item in markers
            if item[0] == x and y <= item[1] < y + height and item[2] == item[3]
        )
        mx, my, mw, mh, color = marker
        assert mx + mw <= x + width
        assert my + mh <= y + height
        assert color[index] > max(color[channel] for channel in range(3) if channel != index)
        assert any(caption == axis and bounds[0] == x for bounds, caption, _ in captions)
        reading = next(
            item
            for item in captions
            if item[0][0] > x and item[0][1] == y and item[0][0] < x + width
        )
        assert reading[1]
        value_colors.append(reading[2])
    assert value_colors[0] == value_colors[1] == value_colors[2]
    assert value_colors[0] == MUTED and value_colors[0] != TEXT
    foreground, background = (
        relative_luminance(tuple(channel / 255 for channel in color))
        for color in (value_colors[0], rgba8("background"))
    )
    assert (foreground + 0.05) / (background + 0.05) >= 4.5
    assert (len({x for x, _ in cells}), len({y for _, y in cells})) in ((3, 1), (1, 3))


@pytest.mark.parametrize(
    "designed,width,columns",
    (
        (False, 160, 1),
        (False, 360, 3),
        (False, 640, 3),
        (True, 160, 1),
        (True, 320, 3),
        (True, 640, 3),
    ),
)
@pytest.mark.parametrize("kind", ("ball", "free"))
def test_expanded_pose_uses_a_complete_xyz_row_when_its_fixed_precision_fits(
    editor, tmp_path, designed, width, columns, kind
):
    if designed:
        editor.ui.design = DesignPresentation(
            editor.ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
        )
    editor.width, editor.kind = width, kind
    editor.frame(3)
    names = ("position", "rotation") if kind == "free" else ("rotation",)
    for name in names:
        fields = [
            editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-{name}{axis}-frame"]
            for axis in "XYZ"
        ]
        assert len({round(x, 3) for x, _y, _w, _h in fields}) == columns
        assert len({round(y, 3) for _x, y, _w, _h in fields}) == 3 // columns
        x, y, width, height = editor.ui.hits[f"pose-fields-{editor.joint.joint_id}"]
        for fx, fy, fw, fh in fields:
            assert x <= fx < fx + fw <= x + width + 1
            assert y <= fy < fy + fh <= y + height + 1
            if columns == 3:
                assert fw >= editor.ui.numeric_minimum(format="%.3f") - 1


@pytest.mark.parametrize("columns,rows", ((1, 3), (2, 3), (3, 1)))
def test_expanded_pose_respects_explicit_xyz_columns(editor, tmp_path, columns, rows):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    spec = deepcopy(runtime.spec)
    spec["parts"]["vector_group"] = {"columns": columns}
    runtime.update_spec(spec)
    editor.width, editor.kind = 640, "free"

    editor.frame(2)
    fields = [
        editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-position{axis}-frame"]
        for axis in "XYZ"
    ]
    assert len({round(field[1], 3) for field in fields}) == rows


def test_expanded_xyz_keeps_signed_decimal_domain_and_geometry_stable(
    editor, tmp_path, monkeypatch
):
    editor.ui.design = DesignPresentation(
        editor.ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    )
    editor.width, editor.kind = 320, "free"
    readings = []
    draw_text = ImguiDraw2D.text

    def text(draw, position, color, caption, **kwargs):
        readings.append(caption)
        return draw_text(draw, position, color, caption, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    editor.frame(2)
    start = editor.joint.qpos_adr
    values = (-360.0, 0, 0.123456789)
    editor.session.adapter.primary.data.qpos[start : start + 3] = values
    readings.clear()
    editor.frame(1)
    assert all(shown in readings for shown in ("-360.000", "0.000", "0.123"))
    fields = [
        editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-position{axis}-frame"]
        for axis in "XYZ"
    ]
    assert len({round(field[1], 3) for field in fields}) == 1
    assert editor.session.adapter.primary.data.qpos[start : start + 3] == pytest.approx(
        values, abs=1e-12
    )
    editor.session.adapter.primary.data.qpos[start : start + 3] = (1e9, -1e-9, 123.456789)
    editor.frame(1)
    assert [
        editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-position{axis}-frame"]
        for axis in "XYZ"
    ] == fields


def test_folded_readings_keep_three_decimals_without_rounding_the_pose(editor, monkeypatch):
    editor.width = 640
    _summary(editor)
    readings = []
    draw_text = editor.ui.component_text

    def text(part, bounds, caption, color, **kwargs):
        readings.append((bounds, caption))
        return draw_text(part, bounds, caption, color, **kwargs)

    monkeypatch.setattr(editor.ui, "component_text", text)
    start = editor.joint.qpos_adr
    before = None
    for values, shown in (
        ((-3, 0, 0.123456789), ("-3.000", "0.000", "0.123")),
        ((-3.25, 12.5, -0.00001), ("-3.250", "12.500", "0.000")),
    ):
        editor.session.adapter.primary.data.qpos[start : start + 3] = values
        readings.clear()
        editor.frame(1)
        cells = []
        for axis, expected in zip("XYZ", shown, strict=True):
            x, y, width, height = editor.ui.hits[
                f"pose-summary-{editor.joint.joint_id}-position-{axis}"
            ]
            cells.append((x, y, width, height))
            assert any(
                caption == expected and x < bounds[0] < x + width and bounds[1] == y
                for bounds, caption in readings
            )
        assert editor.session.adapter.primary.data.qpos[start : start + 3] == pytest.approx(
            values, abs=1e-12
        )
        if before is not None:
            assert cells == before
        before = cells


def test_selected_joint_filtered_out_then_revealed_opens_its_editors(editor):
    _summary(editor, "ball")
    target = next(j for j in editor.session.joints if j.type == "free")
    node = next(node for node in editor.session.nodes if node.joint_index == target.joint_id)
    assert editor.session.submit(cmd.SelectNode(node.node_id)).ok
    editor.frame(2)
    editor.kind = "free"
    editor.frame(2)
    assert f"##pose-joint-{target.joint_id}-positionX" in editor.ui.hits


def test_missing_frame_values_do_not_leave_stale_summary_numbers(editor, monkeypatch):
    key = _summary(editor)
    monkeypatch.setattr(editor.session, "tick", lambda *args, **kwargs: None)
    editor.session._frame = replace(editor.session.frame, qpos=None)
    editor.frame(2)
    assert not any(key.startswith("pose-summary-") for key in editor.ui.hits)
    _click(editor, key)
    assert not any(key.startswith("##pose-joint-") for key in editor.ui.hits)
    assert key in editor.ui.hits


@pytest.mark.parametrize("width", (160, 280, 640))
@pytest.mark.parametrize("kind", ("ball", "free"))
@pytest.mark.parametrize("label_gap", (4, 6))
def test_collapsed_pose_reserves_caption_to_value_gap(editor, tmp_path, width, kind, label_gap):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    spec = deepcopy(runtime.spec)
    spec["parts"]["pose_joint"] = {"summaryLabelGap": label_gap}
    runtime.update_spec(spec)
    runtime.observe_components = True
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    editor.width = width
    with group_scope("Pose"):
        _summary(editor, kind)
    prefix = f"pose-summary-{editor.joint.joint_id}-"
    cells = editor.ui.hits
    for name in ("position", "rotation") if kind == "free" else ("rotation",):
        label = next(
            row["labelRect"]
            for row in reversed(runtime.observed)
            if row["key"] == f"pose-summary-label-{editor.joint.joint_id}-{name}"
        )
        value = cells[prefix + name + "-X"]
        if width < 640:
            assert value[1] - label[1] - label[3] == pytest.approx(label_gap * editor.ui.s)
        else:
            assert value[1] == pytest.approx(label[1])
        field = cells[f"pose-fields-{editor.joint.joint_id}"]
        assert value[1] + value[3] <= field[1] + field[3] + 1e-5


@pytest.mark.parametrize("width", (160, 280))
def test_collapsed_pose_spacing_expresses_group_ownership(editor, tmp_path, monkeypatch, width):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    runtime.observe_components = True
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    editor.width = width
    free = next(joint for joint in editor.session.joints if joint.type == "free")
    ball = next(joint for joint in editor.session.joints if joint.type == "ball")
    draw = editor.panel.draw

    def draw_neighbors(*args, **kwargs):
        return draw(*args, **(kwargs | {"joints": (free, ball)}))

    monkeypatch.setattr(editor.panel, "draw", draw_neighbors)
    with group_scope("Pose"):
        _summary(editor)
    labels = {
        name: next(
            row["labelRect"]
            for row in reversed(runtime.observed)
            if row["key"] == f"pose-summary-label-{free.joint_id}-{name}"
        )
        for name in ("position", "rotation")
    }
    cells = editor.ui.hits
    prefix = f"pose-summary-{free.joint_id}-"

    def last_value_bottom(name):
        return max(
            cells[prefix + name + "-" + axis][1] + cells[prefix + name + "-" + axis][3]
            for axis in "XYZ"
        )

    within = [cells[prefix + name + "-X"][1] - rect[1] - rect[3] for name, rect in labels.items()]
    between_fields = labels["rotation"][1] - last_value_bottom("position")
    next_heading = cells[f"pose-joint-{ball.joint_id}"]
    between_joints = next_heading[1] - last_value_bottom("rotation")
    heading = cells[f"pose-joint-{free.joint_id}"]
    heading_gap = labels["position"][1] - heading[1] - heading[3]
    assert within[0] == pytest.approx(within[1])
    assert 0 < within[0] < between_fields < between_joints
    assert heading_gap > within[0]
