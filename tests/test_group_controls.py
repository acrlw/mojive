"""Group-owned control geometry never changes a sibling group's class defaults."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mojive.design.groups import group_scope
from mojive.design.native.controls import NativePrimitives
from mojive.design.native.geometry import part_scope
from mojive.design.runtime import DesignRuntime
from mojive.design.specification import DEFAULT, component_catalog
from mojive.design.store import SpecificationStore
from mojive.design.support import native_support

CONTROLS = [
    (name, entry["controlGroup"], entry["basePart"])
    for name, entry in component_catalog().items()
    if "controlGroup" in entry
]


@pytest.mark.parametrize("name,parent,base", CONTROLS)
def test_group_controls_own_complete_geometry_without_changing_peers(tmp_path, name, parent, base):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    painter = NativePrimitives(SimpleNamespace(s=1), runtime, imgui=SimpleNamespace())
    fields = component_catalog()[name]["fields"]
    changes = {
        "widthMode": "fixed",
        "width": 143,
        "height": 42,
        "paddingLeft": 13,
        "paddingRight": 17,
        "paddingTop": 5,
        "paddingBottom": 7,
        "gap": 4,
        "fontSize": 18,
        "textColor": "#f4b860",
        "iconSize": 20,
    }
    changes = {field: value for field, value in changes.items() if field in fields}
    global_before = dict(painter.style(base).values)
    peer_scope = "unrelated"
    for _peer, scope, kind in CONTROLS:
        if kind == base and scope != parent:
            with part_scope(scope, inherit=False):
                if name not in painter.style(base)._dependencies():
                    peer_scope = scope
                    break
    with part_scope(peer_scope, inherit=False):
        peer_before = dict(painter.style(base).values)
    candidate = deepcopy(DEFAULT)
    candidate["parts"][name] = changes
    runtime.update_spec(candidate)
    with part_scope(parent, inherit=False):
        local = painter.style(base)
        assert local.part_ids[-1] == name
        for field, value in changes.items():
            assert local.get(field) == value
            assert native_support()[name]["fields"][field]["status"] == "applied"
        box = local.box(0, 0, 100, 28, intrinsic=(0, 25))
        assert box.frame.width == (143 if "width" in fields else 100)
        assert box.frame.height == (42 if "height" in fields else 25 + 5 + 7)
        # Axis fields own horizontal padding and an explicit control height;
        # their schema does not offer independent top/bottom padding.
        assert box.padding == (
            13,
            5 if "paddingTop" in fields else 0,
            17,
            7 if "paddingBottom" in fields else 0,
        )
        assert painter.style(base) is local
    assert dict(painter.style(base).values) == global_before
    with part_scope(peer_scope, inherit=False):
        assert dict(painter.style(base).values) == peer_before
    assert candidate["parts"].get(base) == DEFAULT["parts"].get(base)


def test_closest_control_group_and_explicit_override_precedence(tmp_path):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    painter = NativePrimitives(SimpleNamespace(s=1), runtime, imgui=SimpleNamespace())
    candidate = deepcopy(DEFAULT)
    candidate["parts"].update(
        {
            "button_row_button": {"height": 36, "paddingLeft": 9},
            "control_button_row_button": {"height": 44},
            "settings_nav_button": {"height": 40, "paddingLeft": 12},
        }
    )
    runtime.update_spec(candidate)
    with group_scope("Control"), part_scope("button_row"):
        style = painter.style("button")
        assert style.get("height") == 44
        assert style.get("paddingLeft") == 9
    with group_scope("Cameras"), part_scope("button_row"):
        assert painter.style("button").get("height") == 36
    with group_scope("Settings/Camera"), part_scope("settings_page"):
        assert painter.style("button").part_ids[-1] == "settings_camera_settings_page_button"
        with part_scope("settings_nav", inherit=False):
            assert painter.style("button").part_ids[-1] == "settings_nav_button"
            assert painter.style("button").get("paddingLeft") == 12


def test_shared_defaults_are_distinct_from_owned_groups_in_the_workbench():
    catalog = component_catalog()
    for name, parent, _base in CONTROLS:
        assert catalog[name]["selectionScope"] == catalog[parent].get("selectionScope", "class")
    assert catalog["button_row_button"]["selectionScope"] == "class"
    assert catalog["control_button_row_button"]["selectionScope"] == "group"
    assert catalog["cameras_button_row_button"]["selectionScope"] == "group"
    assert catalog["settings_nav_button"]["selectionScope"] == "group"


def test_sensor_icon_actions_do_not_add_control_classes_to_other_button_rows():
    catalog = component_catalog()
    assert catalog["sensors_button_row"]["controlParts"] == {
        "icon_button": "sensors_button_row_icon_button"
    }
    assert catalog["sensors_button_row_icon_button"]["controlDefaults"] == "icon_button"
    assert catalog["button_row"]["controlParts"] == {"button": "button_row_button"}
    assert catalog["control_button_row"]["controlParts"] == {"button": "control_button_row_button"}
