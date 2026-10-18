"""Entity and bookmark popup measurements have bounded, shared lifetimes."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.workspace import menus as menus_module
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menus import Menus
from tests.test_panel_action_menus import action_surface as action_surface


def test_many_dynamic_popup_keys_evict_their_widths_and_rows_together():
    menus = Menus(SimpleNamespace(ui=None, s=1), ())
    count = menus_module._POPUP_CACHE_LIMIT * 8
    for index in range(count):
        key = f"entity-context-{index}"
        rows = (Item(f"Entity {index}", shortcut="Ctrl+F"),)
        menus._remember_popup(key, rows, 200 + index)
    expected = tuple(f"entity-context-{index}" for index in range(count - 128, count))
    assert tuple(menus._popup_widths) == tuple(menus._popup_rows) == expected
    assert menus._popup_widths[expected[-1]] == 200 + count - 1
    assert menus._popup_rows[expected[-1]][0].label == f"Entity {count - 1}"


def test_repainting_current_popup_refreshes_its_rows_width_and_cache_lifetime(monkeypatch):
    monkeypatch.setattr(menus_module, "_POPUP_CACHE_LIMIT", 2)
    menus = Menus(SimpleNamespace(ui=None, s=1), ())
    menus._remember_popup("current", (Item("Old caption"),), 120)
    menus._remember_popup("previous", (Item("Previous"),), 140)
    current = (Item("A newly expanded caption", shortcut="Ctrl+Shift+F"),)
    menus._remember_popup("current", current, 340)
    menus._remember_popup("new", (Item("New"),), 180)
    assert tuple(menus._popup_rows) == tuple(menus._popup_widths) == ("current", "new")
    assert menus._popup_rows["current"] is current
    assert menus._popup_widths["current"] == 340


def test_evicted_popup_remeasures_to_the_same_actual_row_height_and_width(
    action_surface, monkeypatch
):
    monkeypatch.setattr(menus_module, "_POPUP_CACHE_LIMIT", 3)
    state, target = action_surface, "camera-bookmark-actions/a-long-view-name"

    def popup(key, detailed=False):
        requested = True

        def draw():
            nonlocal requested
            if requested:
                imgui.open_popup(key)
                requested = False
            with state.menus.local_popup(key) as opened:
                if opened:
                    state.menus.local_item(
                        "Update from current view" if detailed else "Restore view",
                        "Ctrl+Shift+F" if detailed else "",
                        checked=False if detailed else None,
                        key="bookmark-update",
                        icon="save",
                    )
                    state.menus.local_item("Delete", key="bookmark-delete", icon="delete")

        state.draw = draw
        state.frame(3)
        record = state.menus.observed[key]
        assert all(not row["clippedLabel"] for row in record["rows"])
        return (
            state.menus._popup_widths[key],
            record["rect"][2:],
            [row["frame_rect"][2:] for row in record["rows"]],
        )

    before = popup(target, detailed=True)
    for index in range(8):
        popup("entity-context-" + str(index))
    assert target not in state.menus._popup_widths
    assert target not in state.menus._popup_rows
    after = popup(target, detailed=True)
    assert after[0] == before[0]
    assert after[1] == pytest.approx(before[1])
    assert after[2] == before[2]
    assert len(state.menus._popup_rows) == len(state.menus._popup_widths) == 3
