"""Capture real panel action menus and validate their painted row budgets."""

import json
import math
from copy import deepcopy

from imgui_bundle import imgui as g

PANEL_PAGES = (
    ("Layers", "Layers"),
    ("Sensors", "Sensors"),
    ("Pose", "Pose"),
    ("Cameras", "Cameras"),
    ("Scene", "Scene"),
    ("Control", "Control"),
    ("Assets", "Assets"),
    ("Output", "Output"),
    ("Statistics", "Statistics"),
    ("Plot", "Curves"),
    ("Inspector", "Inspector"),
)


def _contains(clip, rect, epsilon):
    x, y, width, height = rect
    return (
        clip[0] - epsilon <= x
        and clip[1] - epsilon <= y
        and x + width <= clip[2] + epsilon
        and y + height <= clip[3] + epsilon
    )


def _text_budget(row, scale):
    """Use production measurements, allowing only floating-point arithmetic noise."""
    epsilon = row["nativeEpsilon"]
    assert math.isfinite(epsilon) and 0 < epsilon < scale
    label_width = row["labelMeasuredWidth"]
    assert label_width <= row["labelRect"][2] + epsilon, row["key"] + " clips its title"
    assert not row["clippedLabel"], row["key"] + " elides its title"
    assert not row["clippedShortcut"], row["key"] + " elides its shortcut"
    assert _contains(row["clip"], row["labelRect"], epsilon), row["key"] + " clips its title box"
    if row["shortcutRect"] is not None:
        assert row["shortcutMeasuredWidth"] <= row["shortcutRect"][2] + epsilon
        assert _contains(row["clip"], row["shortcutRect"], epsilon)


def _popup_geometry(v, popup_key, hovered_key):
    menu = v.sample.app.workspace.menus
    popup = deepcopy(menu.observed[popup_key])
    container = next(row for row in v.runtime.observed if row["key"] == "local-menu-" + popup_key)
    rows = sorted(
        (
            deepcopy(row)
            for row in v.runtime.observed
            if row.get("windowId") == container["windowId"] and "checkRect" in row
        ),
        key=lambda row: row["hit"][1],
    )
    assert rows, "No production rows were recorded for " + popup_key
    inventory = menu._popup_rows[popup_key]
    checkable = any(item.checkable or item.checked for item in inventory)
    for row in rows:
        _text_budget(row, v.s)
        assert row["columnCapabilities"]["check"] == checkable
    first, last = rows[0]["hit"], rows[-1]["hit"]
    epsilon = max(row["nativeEpsilon"] for row in rows)
    for divider in popup.get("dividers", ()):
        assert first[1] + first[3] <= divider[1] + epsilon, "Leading menu divider"
        assert divider[1] + divider[3] <= last[1] + epsilon, "Trailing menu divider"
    hovered = next(row for row in rows if row["key"] == hovered_key)
    assert hovered["hovered"] and hovered["backgroundRect"] is not None
    assert hovered["backgroundCornerFlags"] == "all"
    assert _contains(hovered["clip"], hovered["backgroundRect"], epsilon)
    assert all(
        abs(a - b) <= epsilon
        for a, b in zip(hovered["backgroundRect"], hovered["hit"], strict=True)
    )
    x, y, width, height = container["windowRect"]
    envelope = [
        min(row["hit"][0] for row in rows),
        first[1],
        max(row["hit"][0] + row["hit"][2] for row in rows),
        last[1] + last[3],
    ]
    insets = [envelope[0] - x, envelope[1] - y, x + width - envelope[2], y + height - envelope[3]]
    assert min(insets) >= -epsilon
    assert all(abs(inset - insets[0]) <= epsilon for inset in insets)
    outer_radius = popup["radius"] * v.s
    inner_radius = hovered["backgroundRadius"]
    assert abs(outer_radius - inner_radius - insets[0]) <= epsilon
    fx, fy, fw, fh = container["hit"]
    cx, cy, cw, ch = container["content"]
    return {
        "popup": popup,
        "container": deepcopy(container),
        "rows": rows,
        "hovered_key": hovered_key,
        "native_epsilon": epsilon,
        "native_window": container["windowRect"],
        "actual_row_envelope": envelope,
        "inventory": [vars(item) for item in inventory],
        "intrinsic_width_native": menu._popup_widths[popup_key],
        "outer_radius_logical": popup["radius"],
        "inner_radius_logical": hovered["backgroundRadius"] / v.s,
        "insets_logical": [value / v.s for value in insets],
        "insets_native": insets,
        "content_insets_logical": [
            (cx - fx) / v.s,
            (cy - fy) / v.s,
            (fx + fw - cx - cw) / v.s,
            (fy + fh - cy - ch) / v.s,
        ],
        "concentric_radius_passed": True,
        "no_leading_or_trailing_divider": True,
        "strict_text_budget_passed": True,
        "check_column_required_by_group": checkable,
    }


def capture_more_menus(v, page, click, save, event):
    """Visit existing production entry points without activating their commands."""
    menu = v.sample.app.workspace.menus
    previous = v.page, deepcopy(v.values), deepcopy(v.selected)
    menu_state = menu.requested_path, menu.reset_pending
    observed = v.runtime.observe_components
    report = {"scale": v.s, "panels": {}, "geometry_file": "more-menu-geometry.json"}

    def hover(key):
        x, y, width, height = v.ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)

    def visit(name):
        if len(g.get_current_context().open_popup_stack):
            g.internal.close_popup_to_level(0, True)
        menu.requested_path = ()
        page(name)

    v.runtime.observe_components = True
    menu.requested_path = ()
    try:
        for name, guide_page in PANEL_PAGES:
            visit(guide_page)
            trigger = "output-more" if name == "Output" else "header-" + name + "-more"
            if trigger not in v.ui.hits:
                report["panels"][name] = {
                    "status": "No More entry at this production layout",
                    "reason": "Toolbar actions fit inline"
                    if name == "Output"
                    else "The panel retains its dedicated native header controls",
                    "page": guide_page,
                }
                continue
            popup_key = "##output-actions" if name == "Output" else "Panel actions###" + name
            click(trigger)
            for _ in range(4):
                v.frame()
            container = next(
                row for row in v.runtime.observed if row["key"] == "local-menu-" + popup_key
            )
            rows = sorted(
                (
                    row
                    for row in v.runtime.observed
                    if row.get("windowId") == container["windowId"] and "checkRect" in row
                ),
                key=lambda row: row["hit"][1],
            )
            hide_key = "header-menu-" + name + "-hide"
            enabled = {row["key"] for row in menu.observed[popup_key]["rows"] if row["enabled"]}
            ordinary = next(
                (
                    row
                    for row in rows
                    if row["key"] != hide_key and row["key"] in enabled and row["arrowRect"] is None
                ),
                None,
            )
            captures = {}
            for state, key in (
                ("action", ordinary["key"] if ordinary is not None else None),
                ("hide", hide_key if hide_key in v.ui.hits else None),
            ):
                if key is None:
                    continue
                hover(key)
                image = "more-" + name.lower() + "-" + state + "-hover"
                save(image)
                captures[state] = {"image": image + ".png", **_popup_geometry(v, popup_key, key)}
            if name == "Cameras":
                child = "header-menu-Cameras-actions"
                action = "header-action-Cameras-frame"
                hover(child)
                for _ in range(60):
                    if action in v.ui.hits:
                        break
                    v.frame()
                assert action in v.ui.hits, "The real Cameras Header actions submenu did not open"
                # The first child frame measures its intrinsic size; keep the
                # parent active until native constraints and placement settle.
                for _ in range(4):
                    hover(child)
                hover(action)
                save("more-cameras-header-actions")
                geometry = _popup_geometry(v, child, action)
                assert [row["label"] for row in geometry["inventory"]] == ["Frame all"]
                captures["header_actions"] = {
                    "image": "more-cameras-header-actions.png",
                    "parent_popup": deepcopy(menu.observed[popup_key]),
                    **geometry,
                }
            assert captures, "The production popup has no capturable action: " + name
            report["panels"][name] = {
                "trigger": trigger,
                "page": guide_page,
                "captures": captures,
                "single_hide_row": len(rows) == 1 and rows[0]["key"] == hide_key,
            }

        visit("Menus")
        click("menu-Edit")
        key = "menu-item-Edit/Redo"
        hover(key)
        save("menu-edit-redo")
        row = next(row for row in v.runtime.observed if row["key"] == key)
        _text_budget(row, v.s)
        shortcut = next(item.shortcut for item in menu.items("Edit") if item.label == "Redo")
        assert shortcut == ("Cmd" if g.get_io().config_mac_osx_behaviors else "Ctrl") + "+Shift+Z"
        checkable = any(item.checkable or item.checked for item in menu.items("Edit") if item)
        assert row["columnCapabilities"]["check"] == checkable
        report["edit_redo"] = {
            "image": "menu-edit-redo.png",
            "shortcut": shortcut,
            "row": deepcopy(row),
            "popup": deepcopy(menu.observed["Edit"]),
            "strict_text_budget_passed": True,
        }
        (v.output / report["geometry_file"]).write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if len(g.get_current_context().open_popup_stack):
            g.internal.close_popup_to_level(0, True)
        v.page, v.values, v.selected = previous
        menu.requested_path, menu.reset_pending = menu_state
        v.runtime.observe_components = observed
        event("add_mouse_pos_event", -100, -100)
