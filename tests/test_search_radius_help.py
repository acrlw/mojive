"""Search capsule metadata preserves old documents without promising radius edits."""

from copy import deepcopy

import pytest

from mojive.design import reference_contracts
from mojive.design.editor import SpecificationEditor
from mojive.design.help import condition_help, get_field_help, get_part_help
from mojive.design.specification import DEFAULT, component_catalog, validate_spec
from mojive.design.store import SpecificationStore, read_spec
from mojive.design.support import native_consumers, native_support


@pytest.mark.parametrize("radius", (0, 7.25, 80))
def test_legacy_search_radius_round_trips_but_its_editor_control_stays_inactive(tmp_path, radius):
    spec = deepcopy(DEFAULT)
    spec["parts"].setdefault("search", {})["radius"] = radius
    assert validate_spec(spec) == spec
    store = SpecificationStore(tmp_path / "design.json")
    store.write(spec, save=True)
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    assert editor.effective("search")["radius"] == radius
    condition = condition_help("search", "radius", editor.effective("search"))
    assert not condition["enabled"]
    assert "实际高度" in condition["reason"] and "旧值" in condition["reason"]
    assert "高度" in condition["enable"]
    editor.change(("parts", "search", "height"), 40)
    assert not condition_help("search", "radius", editor.effective("search"))["enabled"]
    editor.save()
    saved = read_spec(store.saved_path)
    assert saved["parts"]["search"]["radius"] == radius
    assert saved["parts"]["search"]["height"] == 40


def test_search_support_and_help_identify_derived_capsule_instead_of_independent_radius():
    catalog = component_catalog()
    help = get_field_help("search", "radius")
    assert "radius" in catalog["search"]["fields"]
    assert "兼容旧值" in catalog["search"]["fields"]["radius"]["label"]
    assert help == get_part_help("search")["fields"]["radius"]
    assert "不采用父容器" in help["reference"]
    assert "不改变固定胶囊形" in help["defaultHelp"]
    support = native_support()["search"]["fields"]["radius"]
    assert support["status"] == "derived"
    assert "实际绘制高度" in support["description"]
    assert support["consumers"] == []
    assert "radius" not in native_consumers()["search"]
    assert native_support()["search"]["fields"]["height"]["status"] == "applied"
    assert condition_help("input_text", "radius", {})["enabled"]


def test_reference_manifest_counts_derived_fields_without_claiming_an_authored_consumer(
    monkeypatch,
):
    catalog = component_catalog()
    monkeypatch.setattr(
        reference_contracts,
        "component_catalog",
        lambda: {part: catalog[part] for part in ("search", "input_text")},
    )
    monkeypatch.setattr(
        reference_contracts,
        "CONTRACTS",
        (reference_contracts.contract("Text", "search input_text", "Default", "Text", "Edit"),),
    )
    manifest = reference_contracts.reference_manifest()
    counts = manifest["counts"]
    assert counts["derived"] == 1
    assert counts["production"] + counts["reference"] + counts["derived"] == counts["fields"]
    radius = manifest["parts"]["search"]["fields"]["radius"]
    assert radius["status"] == "derived" and radius["consumers"] == []
