"""Keep audit evidence independent of instrumentation shape and instance names."""

import ctypes
from types import SimpleNamespace

import numpy as np
from imgui_bundle import imgui

from mojive.tools.design_audit import (
    GEOMETRY_FIELDS,
    ParameterAudit,
    changed_value,
    child_bounds_changes,
    rectangle_issues,
)
from mojive.tools.design_audit_draw import (
    VERTEX_DTYPE,
    NativeDrawSamples,
    _FontTexture,
    triangle_overlap,
)
from mojive.tools.design_audit_evidence import (
    field_effect_status,
    inactive_reason,
    instance_effects,
)
from mojive.tools.design_audit_viewport import partial_keys, translate_observation


def snapshot(*records):
    return {"records": list(records), "vertices": np.empty((0, 3))}


def test_added_observation_fields_are_not_a_parameter_effect():
    before = {"key": "apply", "windowId": 1, "hit": [10, 10, 80, 28], "kind": "button"}
    after = {**before, "content": [16, 14, 68, 20], "radius": 6}
    result = ParameterAudit.differences(snapshot(before), snapshot(after))
    assert result["changedGeometryKeys"] == []
    assert result["drawChanged"] is False


def test_changed_style_values_without_used_geometry_are_not_an_effect():
    before = {
        "key": "apply",
        "windowId": 1,
        "hit": [10, 10, 80, 28],
        "padding": [4, 4, 4, 4],
        "margin": [0, 0, 0, 0],
        "fieldValues": {"paddingX": 4},
        "kind": "button",
    }
    after = {
        **before,
        "padding": [8, 4, 8, 4],
        "margin": [4, 0, 0, 0],
        "fieldValues": {"paddingX": 8},
    }
    result = ParameterAudit.differences(snapshot(before), snapshot(after))
    assert result["changedGeometryKeys"] == []
    assert result["drawChanged"] is False


def test_same_control_key_in_two_windows_keeps_both_instances():
    first = {"key": "apply", "windowId": 1, "hit": [10, 10, 80, 28]}
    second = {"key": "apply", "windowId": 2, "hit": [100, 10, 80, 28]}
    changed = {**second, "hit": [100, 10, 80, 32]}
    result = ParameterAudit.differences(snapshot(first, second), snapshot(first, changed))
    assert result["instancesBefore"] == result["instancesAfter"] == 2
    assert result["changedGeometryKeys"] == ["2:apply"]


def test_draw_changes_are_observed_even_when_hit_geometry_is_stable():
    before = snapshot({"key": "icon", "hit": [0, 0, 28, 28]})
    after = snapshot({"key": "icon", "hit": [0, 0, 28, 28]})
    before["vertices"] = np.array([[4, 4, 255], [20, 20, 255]])
    after["vertices"] = np.array([[2, 2, 255], [22, 22, 255]])
    result = ParameterAudit.differences(before, after)
    assert result["changedGeometryKeys"] == []
    assert result["drawChanged"] is True


def test_duplicate_instance_identity_is_an_observation_error():
    records = [
        {"key": "component/form_row", "windowId": 1, "hit": [0, 0, 80, 28]},
        {"key": "component/form_row", "windowId": 1, "hit": [0, 40, 80, 28]},
    ]
    assert rectangle_issues(records) == [
        {
            "key": "component/form_row",
            "windowId": 1,
            "kind": "duplicate-observation-key",
            "count": 2,
        }
    ]


def test_clipped_away_numeric_text_is_not_only_a_draw_change():
    record = {"key": "mass", "windowId": 1, "hit": [0, 0, 80, 28], "kind": "input_number"}
    before, after = snapshot(record), snapshot(record)
    before["instanceVertices"] = {"1:mass": np.array([[10, 10, 255, 1], [0, 0, 10, 0]])}
    after["instanceVertices"] = {"1:mass": np.array([[0, 0, 10, 0]])}
    result = ParameterAudit.differences(before, after)
    assert result["changedDrawKeys"] == ["1:mass"]
    assert result["missingTextKeys"] == ["1:mass"]


def test_nonfinite_and_out_of_bounds_content_remain_failures():
    records = [
        {"key": "bad", "hit": [0, 0, float("nan"), 28]},
        {"key": "wide", "hit": [0, 0, 80, 28], "content": [0, 0, 100, 28]},
    ]
    assert [issue["kind"] for issue in rectangle_issues(records)] == [
        "invalid-frame",
        "content-outside-frame",
    ]


def test_scroll_children_are_checked_inside_their_actual_clip():
    record = {
        "key": "scroll-body",
        "hit": [0, 0, 100, 50],
        "clip": [0, 0, 100, 50],
        "fields": [[0, 40, 100, 20], [0, 70, 100, 20]],
    }
    assert rectangle_issues([record]) == []
    record["clip"] = [0, 0, 200, 200]
    assert len(rectangle_issues([record])) == 2


def test_bounded_change_at_numeric_limit_uses_another_valid_value():
    rule = {"type": "number", "min": 0, "max": 80, "step": 1}
    assert changed_value(rule, 80) == 76
    assert changed_value(rule, 0) == 4


def test_baseline_waits_for_measured_geometry_to_stabilize():
    audit = object.__new__(ParameterAudit)
    counter = 0

    def next_snapshot():
        nonlocal counter
        counter += 1
        y = min(counter, 3) * 10
        return snapshot({"key": "header", "hit": [0, y, 100, 28]})

    audit.snapshot = next_snapshot
    state, stable, _ = audit.settled_snapshot()
    assert stable and counter == 4
    assert state["records"][0]["hit"][1] == 30


def test_one_changed_instance_does_not_pass_an_unchanged_peer():
    first = {"key": "apply", "windowId": 1, "hit": [10, 10, 80, 28], "kind": "button"}
    second = {**first, "key": "reset", "hit": [10, 50, 80, 28]}
    before = snapshot(first, second)
    after = snapshot({**first, "hit": [10, 10, 80, 32]}, second)
    effects = instance_effects(
        before,
        after,
        "button",
        "height",
        {},
        1,
        GEOMETRY_FIELDS,
        ParameterAudit.differences(before, after),
    )
    assert [row["status"] for row in effects] == ["changed", "unchanged"]
    assert field_effect_status(effects)[0] == "partial-effect"


def test_specific_root_and_empty_lane_conditions_do_not_hide_applicable_peers():
    tree = {"depth": 0, "hit": [0, 0, 100, 28]}
    assert inactive_reason(tree, tree, "tree_row", "indent", {}, 1)
    assert not inactive_reason({**tree, "depth": 1}, tree, "tree_row", "indent", {}, 1)
    lane = {"hasLaneText": False, "hit": [0, 0, 100, 28]}
    assert inactive_reason(lane, lane, "timeline_lane", "fontSize", {}, 1)
    assert not inactive_reason(
        {**lane, "hasLaneText": True}, lane, "timeline_lane", "fontSize", {}, 1
    )


def test_total_instance_loss_never_passes_as_a_draw_change():
    record = {"key": "apply", "windowId": 1, "hit": [10, 10, 80, 28], "kind": "button"}
    before, after = snapshot(record), snapshot()
    before["vertices"] = np.array([[10, 10, 255]])
    effects = instance_effects(
        before,
        after,
        "button",
        "height",
        {},
        1,
        GEOMETRY_FIELDS,
        ParameterAudit.differences(before, after),
    )
    assert effects[0]["status"] == "removed"
    assert field_effect_status(effects)[0] == "visibility-loss"


def test_submitted_but_clipped_instance_retains_its_missing_paint_state():
    record = {"key": "apply", "windowId": 1, "hit": [10, 10, 80, 28], "kind": "button"}
    before, after = snapshot(record), snapshot()
    after["rawRecords"] = [{**record, "hit": [10, 200, 80, 28]}]
    effects = instance_effects(
        before,
        after,
        "button",
        "height",
        {},
        1,
        GEOMETRY_FIELDS,
        ParameterAudit.differences(before, after),
    )
    assert effects[0]["status"] == "not-visible"
    assert effects[0]["geometryChanged"] is True
    assert field_effect_status(effects)[0] == "visibility-loss"


def test_raw_field_values_and_style_origins_cannot_be_physical_effects():
    record = {"key": "apply", "windowId": 1, "hit": [10, 10, 80, 28], "kind": "button"}
    before, after = (
        snapshot(record),
        snapshot({**record, "fieldValues": {"height": 32}, "styleOrigins": {"height": "button"}}),
    )
    result = ParameterAudit.differences(before, after)
    assert result["changedGeometryKeys"] == []
    assert result["drawChanged"] is False


def test_missing_child_is_reported_instead_of_skipped():
    parent = {"key": "row", "windowId": 1, "kind": "container", "content": [0, 0, 100, 40]}
    child = {
        "key": "value",
        "windowId": 1,
        "kind": "input_number",
        "hit": [10, 10, 80, 28],
        "parentKey": "row",
    }
    before = {"records": [parent], "allRecords": [parent, child]}
    after = {"records": [parent], "allRecords": [parent]}
    assert child_bounds_changes(before, after) == [
        {"kind": "child-not-visible", "parent": "row", "child": "value"}
    ]


def test_triangle_bbox_overlap_is_not_sufficient_for_visible_paint():
    points = np.array(
        [
            [[0, 0], [10, 0], [0, 10]],
            [[8, 8], [10, 8], [8, 10]],
            [[5, 5], [6, 5], [5, 6]],
        ],
        dtype=float,
    )
    assert triangle_overlap(points, (8, 8, 9, 9)).tolist() == [False, True, False]


def test_scroll_normalization_preserves_empty_icons_and_immutable_native_rectangles():
    record = {"hit": (1, 2, 80, 28), "iconRect": None, "fields": [(1, 2, 80, 28)]}
    normalized = translate_observation(record, (0, 100))
    assert normalized["hit"] == [1, 102, 80, 28]
    assert normalized["fields"] == [[1, 102, 80, 28]]
    assert normalized["iconRect"] is None
    assert record["hit"] == (1, 2, 80, 28)


def test_row_container_background_sliver_requires_real_text_scroll_recovery():
    row = {
        "key": "stats-counts-3",
        "windowId": 1,
        "kind": "container",
        "parts": ["stat_row"],
        "content": [0, 39, 80, 20],
        "clip": [0, 0, 100, 40],
    }
    assert partial_keys(snapshot(row)) == {"1:stats-counts-3"}
    assert partial_keys(snapshot({**row, "parts": ["data_list"]})) == set()


def test_track_without_a_content_record_still_requires_full_visible_drawing():
    track = {
        "key": "joint-track",
        "windowId": 1,
        "kind": "track",
        "hit": [0, 39, 80, 28],
        "clip": [0, 0, 100, 40],
    }
    assert partial_keys(snapshot(track)) == {"1:joint-track"}
    workspace = {**track, "kind": "workspace", "parts": ["workspace"]}
    assert partial_keys(snapshot(workspace)) == set()


def test_role_consumption_metadata_cannot_replace_actual_effect():
    row = {
        "key": "status",
        "windowId": 1,
        "kind": "container",
        "parts": ["application"],
        "hit": [0, 0, 80, 28],
        "consumedFields": {"application": ["fontSize", "statusHeight"]},
    }
    state = snapshot(row)
    effect = ParameterAudit.differences(state, state)
    inactive = instance_effects(
        state, state, "application", "titleHeight", {}, 1, GEOMETRY_FIELDS, effect
    )
    assert inactive[0]["status"] == "inactive"
    unchanged = instance_effects(
        state, state, "application", "fontSize", {}, 1, GEOMETRY_FIELDS, effect
    )
    assert unchanged[0]["status"] == "unchanged"


def test_overlapping_workbench_draw_list_cannot_change_production_sample(monkeypatch):
    class Buffer:
        def __init__(self, array):
            self.array = array
            self.bytes = (ctypes.c_ubyte * array.nbytes).from_buffer(array)

        def __len__(self):
            return len(self.array)

        def data_address(self):
            return ctypes.addressof(self.bytes)

    def batch(color):
        vertices = np.zeros(3, dtype=VERTEX_DTYPE)
        vertices["position"] = [[1, 1], [9, 1], [1, 9]]
        vertices["color"] = color
        command = SimpleNamespace(
            elem_count=3,
            idx_offset=0,
            vtx_offset=0,
            clip_rect=SimpleNamespace(x=0, y=0, z=10, w=10),
            get_tex_id=lambda: 7,
        )
        return SimpleNamespace(
            vtx_buffer=Buffer(vertices),
            idx_buffer=Buffer(np.array([0, 1, 2], dtype=np.uint32)),
            cmd_buffer=[command],
        )

    production, editor = batch(0xFF000064), batch(0xFF000384)
    windows = [
        SimpleNamespace(id_=1, draw_list=production, parent_window=None, flags=0),
        SimpleNamespace(id_=2, draw_list=editor, parent_window=None, flags=0),
    ]
    monkeypatch.setattr(imgui, "get_current_context", lambda: SimpleNamespace(windows=windows))
    monkeypatch.setattr(imgui, "get_platform_io", lambda: SimpleNamespace(viewports=[]))
    monkeypatch.setattr(
        imgui, "get_draw_data", lambda: SimpleNamespace(cmd_lists=[production, editor])
    )
    monkeypatch.setattr(
        imgui,
        "get_io",
        lambda: SimpleNamespace(
            fonts=SimpleNamespace(
                tex_list=[
                    SimpleNamespace(
                        get_tex_id=lambda: 7,
                        width=512,
                        height=512,
                        bytes_per_pixel=1,
                        get_pixels_array=lambda: np.zeros((512, 512), np.uint8),
                    )
                ],
                tex_uv_white_pixel=SimpleNamespace(x=0, y=0),
            )
        ),
    )
    sampler = NativeDrawSamples(SimpleNamespace(_editor_window=lambda window: window.id_ == 2))
    rows, source = sampler.sample(
        {"windowId": 1, "drawListSource": "window", "kind": "button", "visibleHit": [0, 0, 10, 10]}
    )
    assert source["triangleCount"] == 1
    assert rows[:, 2].tolist() == [0xFF000064] * 3
    editor.vtx_buffer.array["color"] = 12345
    again = NativeDrawSamples(
        SimpleNamespace(_editor_window=lambda window: window.id_ == 2)
    ).sample(
        {"windowId": 1, "drawListSource": "window", "kind": "button", "visibleHit": [0, 0, 10, 10]}
    )[0]
    np.testing.assert_array_equal(rows, again)
    content = {
        "windowId": 1,
        "drawListSource": "window",
        "kind": "container",
        "visibleHit": [8, 8, 1, 1],
        "parts": ["panel"],
        "docked": False,
        "windowRect": [0, 0, 10, 10],
        "nativeBackgroundClip": [0, 0, 10, 10],
    }
    corners, source = sampler.sample(content)
    np.testing.assert_array_equal(corners, rows)
    assert source["triangleCount"] == 1
    docked, source = sampler.sample({**content, "docked": True})
    assert len(docked) == source["triangleCount"] == 0
    production.vtx_buffer.array["color"] = 100
    transparent, source = NativeDrawSamples(
        SimpleNamespace(_editor_window=lambda window: window.id_ == 2)
    ).sample({**content, "visibleHit": [0, 0, 10, 10]})
    assert len(transparent) == source["triangleCount"] == 0


def test_font_texture_pixels_are_evidence_when_glyph_quads_and_ink_are_identical(monkeypatch):
    vertices = np.zeros(3, dtype=VERTEX_DTYPE)
    vertices["position"] = [[1, 1], [9, 1], [1, 9]]
    vertices["uv"] = [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]
    vertices["color"] = 0xFFFFFFFF
    triangles = np.array([[0, 1, 2]])
    positions = vertices["position"][triangles]
    texture = _FontTexture(
        SimpleNamespace(
            width=512,
            height=512,
            bytes_per_pixel=1,
            get_pixels_array=lambda: np.arange(512 * 512, dtype=np.uint8),
        )
    )
    commands = [
        (
            triangles,
            positions,
            positions.min(axis=1),
            positions.max(axis=1),
            (0, 0, 10, 10),
            texture,
        )
    ]
    monkeypatch.setattr(
        imgui,
        "get_io",
        lambda: SimpleNamespace(
            fonts=SimpleNamespace(tex_uv_white_pixel=SimpleNamespace(x=0, y=0))
        ),
    )
    record = {"key": "caption", "windowId": 1, "kind": "button", "visibleHit": [0, 0, 10, 10]}
    initial = NativeDrawSamples.sample_batch((vertices, commands), record)[0]
    vertices["uv"] += 0.25
    changed = NativeDrawSamples.sample_batch((vertices, commands), record)[0]
    np.testing.assert_array_equal(initial[:, :4], changed[:, :4])
    assert not np.array_equal(initial[:, 4:], changed[:, 4:])
    before, after = snapshot(record), snapshot(record)
    before["instanceVertices"] = {"1:caption": initial}
    after["instanceVertices"] = {"1:caption": changed}
    assert ParameterAudit.differences(before, after)["changedDrawKeys"] == ["1:caption"]


def test_identical_glyph_bitmap_survives_atlas_growth_and_repacking():
    glyph = np.arange(16, dtype=np.uint8).reshape(4, 4)
    original = np.zeros((16, 16), np.uint8)
    original[1:5, 2:6] = glyph
    repacked = np.zeros((32, 32), np.uint8)
    repacked[10:14, 20:24] = glyph

    def texture(pixels):
        return _FontTexture(
            SimpleNamespace(
                width=pixels.shape[1],
                height=pixels.shape[0],
                bytes_per_pixel=1,
                get_pixels_array=lambda: pixels,
            )
        )

    before = texture(original).signature(np.array([[2, 1], [6, 1], [2, 5]]) / 16)
    uv_after = np.array([[20, 10], [24, 10], [20, 14]]) / 32
    assert texture(repacked).signature(uv_after) == before
    repacked[11, 21] += 1
    assert texture(repacked).signature(uv_after) != before
