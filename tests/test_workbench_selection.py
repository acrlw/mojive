"""Inspection edits a local semantic group without hiding shared class controls."""

from mojive.design.native.selection import selection_parts


def record(parts, width, height):
    return {"parts": parts, "hit": (0, 0, width, height)}


def test_local_navigation_wins_over_its_shared_button_and_layout():
    catalog = {
        "button": {},
        "settings_nav": {"selectionScope": "group"},
        "settings_layout": {},
    }
    records = [
        record(("settings_layout",), 600, 500),
        record(("button",), 100, 28),
        record(("settings_nav",), 130, 300),
    ]
    assert selection_parts(records, catalog) == ["settings_nav", "button", "settings_layout"]


def test_nested_groups_choose_the_nearest_region_and_keep_all_ancestors():
    catalog = {
        "statusbar": {"selectionScope": "group"},
        "status_context": {"selectionScope": "group"},
        "status_hints": {"selectionScope": "group"},
        "button": {},
        "icon_button": {},
    }
    records = [
        record(("statusbar", "status_context"), 500, 28),
        record(("button", "icon_button"), 20, 20),
        record(("statusbar", "status_context", "status_hints"), 100, 28),
    ]
    assert selection_parts(records, catalog) == [
        "status_hints",
        "status_context",
        "statusbar",
        "icon_button",
        "button",
    ]


def test_classes_remain_selectable_without_an_enclosing_semantic_group():
    catalog = {"input_number": {}, "axis_input": {}, "vector_row": {}}
    records = [
        record(("vector_row",), 300, 28),
        record(("input_number", "axis_input"), 60, 28),
        record(("input_number", "axis_input"), 60, 28),
        record(("unregistered",), 1, 1),
    ]
    assert selection_parts(records, catalog) == ["axis_input", "input_number", "vector_row"]
