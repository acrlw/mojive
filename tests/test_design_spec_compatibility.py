"""Known additive v1 surface tokens load without weakening persisted-spec validation."""

import json
import os
from copy import deepcopy

import pytest

from mojive.design.groups import group_part
from mojive.design.specification import DEFAULT, validate_spec
from mojive.design.store import SpecificationStore, read_spec


def legacy_spec():
    value = deepcopy(DEFAULT)
    value["tokens"]["body_font_size"] = 17
    value["components"]["navigation"] = {"body_font_size": 19}
    value["parts"]["switch"] = {"switchWidth": 56}
    value["parts"][group_part("Scene", "panel_body")] = {"paddingLeft": 17, "rowGap": 9}
    value["elements"][".custom"] = {"styles": {"marginLeft": 13}}
    validate_spec(value)
    del value["tokens"]["floating_color"]
    del value["tokens"]["popup_color"]
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def with_surface_defaults(value):
    return {
        **value,
        "tokens": {
            **value["tokens"],
            **{name: DEFAULT["tokens"][name] for name in ("floating_color", "popup_color")},
        },
    }


@pytest.mark.parametrize("newest", ("saved", "live"))
def test_saved_and_live_legacy_specs_load_without_rewriting_or_losing_overrides(tmp_path, newest):
    store = SpecificationStore(tmp_path / "current-spec.json")
    saved, live = legacy_spec(), legacy_spec()
    live["components"]["navigation"]["body_font_size"] = 21
    for name, path, value in (
        ("saved", store.saved_path, saved),
        ("live", store.live_path, live),
    ):
        write_json(path, value)
        stamp = 2_000_000_000 if name == newest else 1_000_000_000
        os.utime(path, ns=(stamp, stamp))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
        for path in (store.saved_path, store.live_path)
    }
    assert read_spec(store.saved_path) == with_surface_defaults(saved)
    assert read_spec(store.live_path) == with_surface_defaults(live)
    latest = store.latest()
    assert latest["source"] == (store.saved_path if newest == "saved" else store.live_path).name
    assert latest["savedSpec"] == with_surface_defaults(saved)
    assert latest["spec"] == with_surface_defaults(saved if newest == "saved" else live)
    assert latest["errors"] == []
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) for path in before
    }
    with pytest.raises(ValueError, match="全部全局参数"):
        validate_spec(saved)


@pytest.mark.parametrize(
    "damage",
    ("other_missing", "only_one_surface_missing", "unknown_token", "invalid_color", "version"),
)
def test_compatibility_reader_rejects_unknown_or_damaged_shapes(tmp_path, damage):
    value = legacy_spec()
    if damage == "other_missing":
        del value["tokens"]["body_font_size"]
    elif damage == "only_one_surface_missing":
        value["tokens"]["floating_color"] = DEFAULT["tokens"]["floating_color"]
    elif damage == "unknown_token":
        value["tokens"]["future_color"] = "#123456"
    elif damage == "invalid_color":
        value["tokens"]["accent_color"] = "invalid"
    else:
        value["version"] = 2
    store = SpecificationStore(tmp_path / "current-spec.json")
    write_json(store.saved_path, value)
    original = store.saved_path.read_bytes()
    with pytest.raises(ValueError):
        read_spec(store.saved_path)
    latest = store.latest()
    assert latest["source"] is None and latest["errors"]
    assert latest["savedSpec"] is None
    assert store.saved_path.read_bytes() == original


def test_editing_a_loaded_legacy_spec_persists_new_tokens_and_existing_overrides(tmp_path):
    store = SpecificationStore(tmp_path / "current-spec.json")
    original = legacy_spec()
    write_json(store.saved_path, original)
    before = store.latest()["spec"]
    edited = deepcopy(before)
    edited["parts"]["button"]["height"] = 40
    latest = store.merge(before, edited, save=True)
    assert latest["spec"] == edited
    assert read_spec(store.saved_path) == edited
    assert latest["spec"]["components"] == original["components"]
    assert latest["spec"]["elements"] == original["elements"]
    for part in ("switch", group_part("Scene", "panel_body")):
        assert latest["spec"]["parts"][part] == original["parts"][part]
    validate_spec(json.loads(store.saved_path.read_text()))
