"""Warm native style requests retain live geometry, ownership and field evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mojive.design.groups import group_part, group_scope
from mojive.design.native import controls
from mojive.design.native.controls import NativePrimitives
from mojive.design.native.geometry import (
    _classify_button,
    classify_button,
    corner_radius_scope,
    part_scope,
)
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.typography import TextRole, Typography
from tests.test_component_appearance import appearance as appearance


@pytest.fixture
def styles(tmp_path):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui = SimpleNamespace(s=1.2, typography=Typography())
    return NativePrimitives(ui, runtime, imgui=SimpleNamespace()), runtime, ui


def test_warm_requests_resolve_group_and_type_once(styles, monkeypatch):
    painter, _runtime, _ui = styles
    calls = []
    scoped = controls.scoped_parts

    def resolve(parts):
        calls.append(parts)
        return scoped(parts)

    monkeypatch.setattr(controls, "scoped_parts", resolve)
    with group_scope("Inspector"), part_scope("section_body"):
        original = painter.style("input_number")
        for parts in ("input_number", ("input_number",), ["input_number"]) * 30:
            assert painter.style(parts) is original
    assert calls == [("input_number",)]


def test_parent_group_family_and_authored_modes_keep_distinct_requests(styles):
    painter, runtime, _ui = styles
    spec = deepcopy(DEFAULT)
    spec["parts"]["vector_row"] = {"fontSize": 23}
    spec["parts"][group_part("Scene", "tree_row")] = {"fontSize": 19}
    spec["components"]["settings"] = {"body_font_size": 22}
    runtime.update_spec(spec)
    baseline = painter.style("input_number")
    with part_scope("vector_row"):
        assert painter.style("input_number").get("fontSize") == 23
    assert painter.style("input_number") is baseline
    shared = painter.style("tree_row")
    with group_scope("Scene"):
        assert painter.style("tree_row").get("fontSize") == 19
    assert painter.style("tree_row") is shared
    painter.family = "settings"
    assert painter.style("tree_row").get("fontSize") == 22
    painter.family = None
    assert painter.style("input_number") is baseline
    runtime.authored_defaults = False
    assert not painter.style("input_number").authored_defaults
    assert painter.style("input_number", authored_defaults=True) is baseline


def test_typography_replacement_and_live_runtime_edits_invalidate_warm_requests(styles):
    painter, runtime, ui = styles
    original = painter.style("input_number")
    ui.typography = Typography(numeric=TextRole("mono", 18))
    assert painter.style("input_number").get("fontSize") == 18
    spec = deepcopy(runtime.spec)
    spec["parts"]["input_number"] = {"fontSize": 16, "radius": 9}
    runtime.update_spec(spec)
    edited = painter.style("input_number")
    assert edited.get("fontSize") == 16 and edited.get("radius") == 9
    runtime.update_spec(deepcopy(DEFAULT))
    ui.typography = Typography()
    assert dict(painter.style("input_number").values) == dict(original.values)


def test_store_poll_updates_requests_and_unrelated_edits_keep_prepared_styles(styles):
    painter, runtime, _ui = styles
    original = painter.style("input_number")
    spec = deepcopy(runtime.spec)
    spec["parts"].setdefault("camera_preview", {})["radius"] = 19
    runtime.update_spec(spec)
    assert painter.style("input_number") is original
    spec["parts"]["input_number"] = {"fontSize": 20}
    runtime.store.write(spec, save=True)
    assert runtime.poll(force=True)
    assert painter.style("input_number").get("fontSize") == 20
    runtime.store.write(DEFAULT, save=True)
    assert runtime.poll(force=True)
    assert dict(painter.style("input_number").values) == dict(original.values)


def test_first_constrained_request_caches_only_the_base_and_applies_live_native_radii(styles):
    painter, _runtime, ui = styles
    with corner_radius_scope(5.2, "input_number"):
        outer = painter.style("input_number")
        assert outer.get("radius") == 5.2 / ui.s
        with corner_radius_scope(0, "input_number"):
            assert painter.style("input_number").get("radius") == 0
        assert painter.style("input_number") is outer
    baseline = painter.style("input_number")
    assert baseline.origins["radius"] != "parent_corner"
    assert all(
        style.origins["radius"] != "parent_corner" for style in painter._style_requests.values()
    )
    ui.s = 1.25
    with corner_radius_scope(5.2, "input_number"):
        assert painter.style("input_number").get("radius") == 5.2 / ui.s
    assert painter.style("input_number") is baseline


def test_warm_styles_publish_field_reads_each_observed_frame(styles):
    painter, runtime, _ui = styles
    painter.style("input_number")
    runtime.track_field_reads = True
    for _ in range(2):
        runtime.begin_frame()
        painter.style("input_number").get("radius")
        assert "radius" in runtime.used_fields["input_number"]


def test_request_cache_is_bounded_and_repopulates_after_eviction(styles):
    painter, _runtime, _ui = styles
    style = painter.style("input_number")
    for index in range(5000):
        painter._remember_style_request((index,), style)
    assert len(painter._style_requests) <= 4096
    assert painter.style("input_number") is style


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("legacy_radius", (0, 7.25, 80))
def test_warm_search_requests_keep_real_capsule_and_ordinary_fields_keep_dynamic_parent_corners(
    appearance, monkeypatch, legacy_radius
):
    ui, _runtime, update, frame = appearance
    update({"search": {"height": 40, "radius": legacy_radius}})
    painted, radius = [], 0
    rectangle = ui.rect

    def rect(x, y, width, height, tint, r=None, **options):
        painted.append(((x, y, width, height), (r or 0) * ui.s))
        return rectangle(x, y, width, height, tint, r, **options)

    monkeypatch.setattr(ui, "rect", rect)

    def draw():
        painted.clear()
        with ui.corner_scope(radius, ("input_text", "search")):
            ui.component_style(("input_text", "search"))
            ui.search("cached-search", 30 * ui.s, 40 * ui.s, 300 * ui.s, "Filter")
            ui.input_text("cached-field", 30 * ui.s, 110 * ui.s, 300 * ui.s, "Value")

    for radius in (0, 5.2, 9.1):
        for _ in range(3):
            frame(draw)
        bounds = ui.hits["cached-search"]
        capsule = [r for box, r in painted if box == pytest.approx(bounds)]
        ordinary = [r for box, r in painted if box == pytest.approx(ui.hits["cached-field"])]
        assert capsule and all(r == pytest.approx(bounds[3] / 2) for r in capsule)
        assert ordinary and all(r == pytest.approx(radius) for r in ordinary)


def test_button_classification_caches_only_semantics_with_bounded_keys():
    _classify_button.cache_clear()
    assert classify_button("expand-3", "Label") == ("tree_expander",)
    assert classify_button("expand-3", "Another label", []) == ("tree_expander",)
    assert _classify_button.cache_info().hits == 1
    assert classify_button("plain", "") == ("icon_button",)
    assert classify_button("plain", "Caption") == ("button",)
    assert classify_button("plain", "Caption", "tab") == ("tab",)
    assert classify_button("plain", "Caption", ["material"]) == ("material",)
    assert classify_button("receipt-open", "Open", "tab") == ("link_button",)
    assert classify_button("material-color", "Color", "tab") == ("color_input",)
    assert classify_button("tab-properties", "Properties") == ("tab",)
    assert classify_button("entity-7", "Entity") == ("tree_label",)
    assert classify_button("top-menu-File", "File") == ("menu_trigger",)
    for index in range(5000):
        classify_button("dynamic-" + str(index), "")
    assert _classify_button.cache_info().currsize <= 4096
