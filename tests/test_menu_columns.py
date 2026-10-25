"""Native menus reserve separate state, icon, title, shortcut and arrow columns."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
from examples.ui_design.native.reference_menus import Menus as ReferenceMenus
from imgui_bundle import imgui

from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.choice_popup import ChoicePopup
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit, text_budget
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import MenuColumns, menu_row_size, paint_menu_row
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.popup_style import menu_popup_content, menu_popup_style, prepare_menu_popup
from mojive.ui.workspace.widgets import DIM, TEXT
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize(
    "appearance", (1.0, 1.1, 1.2, 1.25, 1.3, 1.5, 2.0), indirect=True
)


@pytest.fixture
def columns(appearance, monkeypatch):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    state = SimpleNamespace(
        ui=ui, row=Item("Command", icon="camera", shortcut="Ctrl+K", checkable=True)
    )
    icons, captions = [], []
    icon, text = ui.icon, ui.component_text

    def draw_icon(name, x, y, size, color, *args, **kwargs):
        icons.append((name, x, y, size * ui.s, color))
        return icon(name, x, y, size, color, *args, **kwargs)

    def draw_text(part, bounds, caption, color, **kwargs):
        captions.append((caption, bounds, color))
        return text(part, bounds, caption, color, **kwargs)

    monkeypatch.setattr(ui, "icon", draw_icon)
    monkeypatch.setattr(ui, "component_text", draw_text)

    def draw():
        runtime.observed.clear()
        ui.design.begin_frame()
        icons.clear()
        captions.clear()
        imgui.set_cursor_screen_pos((40 * ui.s, 50 * ui.s))
        state.result = paint_menu_row(
            ui,
            state.row,
            "command",
            content_right=460 * ui.s,
            part=state.part,
            columns=(
                MenuColumns(
                    True,
                    True,
                    True,
                    ui.measure(state.row.label, 14, font_resolved=True),
                    ui.measure(state.row.shortcut, 14, translate=False, font_resolved=True),
                )
                if state.part == "menu_item"
                else None
            ),
        )
        state.item_id = imgui.get_item_id()
        state.record = next(row for row in runtime.observed if row["key"] == "command")
        state.icons, state.captions = list(icons), list(captions)
        return deepcopy(state.record)

    def event(method, *args):
        getattr(imgui.get_io(), method)(*args)
        return frame(draw)

    state.frame, state.event, state.part = lambda: frame(draw), event, "menu_item"
    state.frame()
    return state


def test_command_columns_are_fixed_for_all_check_icon_shortcut_submenu_combinations(columns):
    state, ui = columns, columns.ui
    baseline, shortcut_bounds = None, None
    for checked, has_icon, has_shortcut, submenu, enabled in product((False, True), repeat=5):
        state.row = Item(
            "Command",
            icon="camera" if has_icon else "",
            shortcut="Ctrl+K" if has_shortcut else "",
            checked=checked,
            submenu="Child" if submenu else "",
            enabled=enabled,
            checkable=True,
        )
        row = state.frame()
        fixed = (
            row["checkRect"],
            row["iconSlotRect"],
            row["labelRect"][0],
            row["shortcutAnchor"],
            row["arrowSlotRect"],
        )
        if baseline is None:
            baseline = fixed
        assert fixed == baseline
        check, icon, title, arrow = (
            row["checkRect"],
            row["iconSlotRect"],
            row["labelRect"],
            row["arrowSlotRect"],
        )
        assert check[0] + check[2] < icon[0]
        assert icon[0] + icon[2] < title[0]
        assert title[0] + title[2] < arrow[0]
        if has_shortcut:
            if shortcut_bounds is None:
                shortcut_bounds = row["shortcutRect"]
            assert row["shortcutRect"] == shortcut_bounds
            sx, _sy, sw, _sh = row["shortcutRect"]
            assert title[0] + title[2] < sx < sx + sw < arrow[0]
            assert sx + sw == row["shortcutAnchor"]
            caption = next(item for item in state.captions if item[0] == "Ctrl+K")
            assert caption[2] == DIM if enabled else caption[2] != TEXT
        names = [name for name, *_ in state.icons]
        assert names.count("check") == int(checked)
        assert names.count("right") == int(submenu)
        assert names.count("camera") == int(has_icon)
        for name, x, _y, extent, _color in state.icons:
            slot = check if name == "check" else arrow if name == "right" else icon
            assert x + extent / 2 == pytest.approx(slot[0] + slot[2] / 2)
            assert extent == pytest.approx(16 * ui.s)
        assert not row["clippedLabel"]
        assert state.result.visible
        assert row["hit"][2] == native_unit(420 * ui.s)


def test_checked_state_does_not_resize_the_menu_or_move_its_shortcut(columns):
    state, ui = columns, columns.ui
    for icon, shortcut, submenu in product(("", "camera"), ("", "Ctrl+Shift+K"), ("", "Child")):
        state.row = Item("Command", icon=icon, shortcut=shortcut, submenu=submenu, checkable=True)
        unchecked_size = menu_row_size(ui, state.row)
        unchecked = state.frame()
        state.row = replace(state.row, checked=True)
        assert menu_row_size(ui, state.row) == unchecked_size
        checked = state.frame()
        for field in ("hit", "labelRect", "iconSlotRect", "shortcutRect", "arrowSlotRect"):
            assert checked[field] == unchecked[field]


@pytest.mark.parametrize("has_icon", (False, True))
def test_native_choices_keep_a_leading_check_and_stable_title_shortcut_columns(columns, has_icon):
    state, ui = columns, columns.ui
    state.part = "select_option"
    baseline = None
    for checked, shortcut, enabled in product((False, True), ("", "Ctrl+K"), (False, True)):
        state.row = Item(
            "Command",
            icon="camera" if has_icon else "",
            checked=checked,
            shortcut=shortcut,
            enabled=enabled,
        )
        actual = state.frame()
        fixed = actual["checkRect"], actual["labelRect"][0], actual["shortcutAnchor"]
        if baseline is None:
            baseline = fixed
        assert fixed == baseline
        check, title = actual["checkRect"], actual["labelRect"]
        assert check[0] == actual["content"][0]
        assert check[0] + check[2] < title[0]
        if has_icon:
            icon = actual["iconRect"]
            assert check[0] + check[2] < icon[0]
            assert icon[0] + icon[2] < title[0]
        else:
            assert actual["iconRect"] is None
        if shortcut:
            tail = actual["shortcutRect"]
            assert title[0] + title[2] < tail[0]
            assert tail[0] + tail[2] == actual["content"][0] + actual["content"][2]
            caption = next(item for item in state.captions if item[0] == shortcut)
            assert caption[2] == DIM if enabled else caption[2] != TEXT
        assert [name for name, *_ in state.icons].count("check") == int(checked)
        size = menu_row_size(ui, state.row, part=state.part)
        assert size == menu_row_size(ui, replace(state.row, checked=not checked), part=state.part)


def test_choice_check_and_icon_spacing_keep_their_independent_style_fields(columns):
    state, ui = columns, columns.ui
    spec = deepcopy(ui.design.runtime.spec)
    spec["parts"]["select_option"] = {"checkGap": 9, "gap": 5}
    ui.design.runtime.update_spec(spec)
    state.part, state.row = "select_option", Item("Command", icon="camera", checked=True)
    row = state.frame()
    check, icon, title = row["checkRect"], row["iconSlotRect"], row["labelRect"]
    assert icon[0] - check[0] - check[2] == native_unit(9 * ui.s)
    assert title[0] - icon[0] - icon[2] == native_unit(5 * ui.s)
    assert row["fieldRegions"]["checkGap"] == pytest.approx(
        (check[0] + check[2], check[1], native_unit(9 * ui.s), check[3])
    )


@pytest.mark.parametrize("renderer", ("choice-popup", "shared-combo"))
def test_actual_choice_popup_selection_retains_leading_column_geometry(appearance, renderer):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    choices = ChoicePopup(ui)
    state = SimpleNamespace(selected=0)
    labels = ("First choice", "Second choice", "Third choice")

    def draw():
        runtime.observed.clear()
        ui.design.begin_frame()
        if renderer == "shared-combo":
            _, state.selected = ui.combo(
                "choices", 40 * ui.s, 50 * ui.s, 240 * ui.s, state.selected, labels
            )
        elif ui.begin_combo("choices", 40 * ui.s, 50 * ui.s, 240 * ui.s, labels[state.selected]):
            choices.begin("choices")
            try:
                for index, label in enumerate(labels):
                    clicked, _ = choices.choice(label, selected=index == state.selected)
                    if clicked:
                        state.selected = index
            finally:
                choices.end()
        return [deepcopy(row) for row in runtime.observed if "/choice/" in row["key"]]

    def event(method, *args):
        getattr(imgui.get_io(), method)(*args)
        return frame(draw)

    def click(key):
        x, y, width, height = ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        return event("add_mouse_button_event", 0, False)

    frame(draw)
    frame(draw)
    click("choices")
    before = frame(draw)
    assert len(before) == 3
    assert len({row["labelRect"][0] for row in before}) == 1
    assert all(row["checkRect"][0] + row["checkRect"][2] < row["labelRect"][0] for row in before)
    click("choices/choice/1")
    assert state.selected == 1
    assert not frame(draw)
    click("choices")
    after = frame(draw)
    assert len(after) == 3
    for old, new in zip(before, after, strict=True):
        assert old["checkRect"] == new["checkRect"]
        assert old["labelRect"] == new["labelRect"]
    event("add_key_event", imgui.Key.escape, True)
    event("add_key_event", imgui.Key.escape, False)
    assert not frame(draw)


def test_shortcuts_have_a_shared_trailing_anchor_and_titles_keep_their_leading_edge(columns):
    state = columns
    records = []
    for shortcut in ("K", "Ctrl+K", "Ctrl+Shift+K"):
        state.row = replace(state.row, shortcut=shortcut)
        records.append(state.frame())
    assert len({row["labelRect"][0] for row in records}) == 1
    assert len({row["shortcutAnchor"] for row in records}) == 1
    assert len({row["shortcutRect"][2] for row in records}) == 3


@pytest.mark.parametrize("part", ("menu_item", "select_option"))
def test_native_choice_and_command_activation_keep_bounds_while_the_check_changes(columns, part):
    state = columns
    state.part = part
    first = state.frame()
    x, y, width, height = first["hit"]
    state.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    assert state.result.activated
    state.row = replace(state.row, checked=True)
    checked = state.frame()
    assert checked["labelRect"] == first["labelRect"]
    assert checked["shortcutRect"] == first["shortcutRect"]
    assert checked["hit"] == first["hit"]
    state.row = replace(state.row, enabled=False)
    state.frame()
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    assert not state.result.activated


def test_native_menu_keyboard_activation_keeps_focus_in_the_fixed_row(columns):
    state = columns
    x, y, width, height = state.frame()["hit"]
    state.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_button_event", 0, False)
    state.event("add_mouse_pos_event", -100, -100)
    state.event("add_key_event", imgui.Key.tab, True)
    state.event("add_key_event", imgui.Key.tab, False)
    state.frame()
    focused_id = imgui.get_current_context().nav_id
    assert focused_id == state.item_id
    state.event("add_key_event", imgui.Key.space, True)
    activated = state.result.activated
    state.event("add_key_event", imgui.Key.space, False)
    activated |= state.result.activated
    assert activated
    assert imgui.get_current_context().nav_id == focused_id


@pytest.mark.parametrize("surface", (Menus, ReferenceMenus), ids=("workspace", "reference"))
def test_both_native_menu_surfaces_use_the_same_production_columns(appearance, surface):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    menus = surface(SimpleNamespace(ui=ui, s=ui.s), ())
    row = Item("Command", icon="camera", shortcut="Ctrl+K", checked=True)
    record = {"rows": []}

    def draw():
        runtime.observed.clear()
        ui.design.begin_frame()
        imgui.set_cursor_screen_pos((40 * ui.s, 50 * ui.s))
        menus.item(row, ("Example",), record)
        imgui.dummy((0, 0))
        return next(item for item in runtime.observed if item["key"] == "menu-item-Example/Command")

    actual = frame(draw)
    check, icon, title, shortcut = (
        actual["checkRect"],
        actual["iconRect"],
        actual["labelRect"],
        actual["shortcutRect"],
    )
    assert check[0] + check[2] < icon[0] < title[0] < shortcut[0]
    assert actual["labelRect"][0] + actual["labelRect"][2] < shortcut[0]
    assert record["rows"][-1]["label_rect"] == actual["labelRect"]


@pytest.fixture
def compact_menu(appearance, monkeypatch):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    state = SimpleNamespace(ui=ui, rows=(), painted=[])
    state.menus = Menus(SimpleNamespace(ui=ui, s=ui.s), ())
    state.menus.items = lambda _name: state.rows
    state.menus.preview(("Commands",))
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        state.painted.append((value, imgui.get_font().font_id, imgui.get_font_size(), color))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        state.painted.clear()
        state.menus.begin_frame()
        state.menus.popup("Commands", (30 * ui.s, 30 * ui.s, 70 * ui.s, 28 * ui.s))
        return [deepcopy(row) for row in runtime.observed if "checkRect" in row]

    def settle():
        for _ in range(5):
            state.records = frame(draw)
        return state.records

    state.frame, state.update = settle, update
    return state


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("family", ("system", "portable"))
def test_compact_group_budgets_independent_longest_title_and_shortcut_without_elision(
    compact_menu, language, family
):
    state, ui = compact_menu, compact_menu.ui
    ui.translate = lambda value: (
        {
            "Undo": "撤销",
            "Redo": "重做",
            "Restore visibility": "恢复全部对象可见性",
        }.get(value, value)
        if language == "zh_CN"
        else value
    )
    state.update({"menu_item": {"fontFamily": family, "fontSize": 14}})
    state.rows = (
        Item("Undo", icon="undo", shortcut="Cmd+Z", enabled=False),
        Item("Redo", icon="redo", shortcut="Cmd+Shift+Z", enabled=False),
        Item("Restore visibility", icon="eye", shortcut="K", checkable=True),
    )
    records = state.frame()
    assert len(records) == 3
    assert all(not row["clippedLabel"] and not row["clippedShortcut"] for row in records)
    assert len({tuple(row["labelRect"])[:1] for row in records}) == 1
    assert len({row["labelRect"][2] for row in records}) == 1
    assert len({row["shortcutAnchor"] for row in records}) == 1
    assert len({row["shortcutSlotRect"][2] for row in records}) == 1
    assert all(
        row["columnCapabilities"] == {"check": True, "icon": True, "arrow": False}
        for row in records
    )
    for row in records:
        assert row["labelRect"][2] >= row["labelMeasuredWidth"]
        assert row["shortcutSlotRect"][2] >= row["shortcutMeasuredWidth"]
    texts = {value for value, *_ in state.painted}
    assert {ui.translate(row.label) for row in state.rows} <= texts
    assert {row.shortcut for row in state.rows} <= texts
    assert not any("…" in value for value in texts)
    assert len({(font, size) for _value, font, size, _ink in state.painted}) == 1
    assert next(ink for value, _font, _size, ink in state.painted if value == "K") != next(
        ink
        for value, _font, _size, ink in state.painted
        if value == ui.translate("Restore visibility")
    )
    popup = state.menus.observed["Commands"]["rect"]
    assert popup[2] == round(popup[2])


@pytest.mark.parametrize("icon", ("", "camera"))
def test_noncheckable_menu_omits_empty_slots_and_intrinsic_width_can_be_below_224(
    compact_menu, icon
):
    state, ui = compact_menu, compact_menu.ui
    state.rows = (Item("One", icon=icon), Item("Two"))
    records = state.frame()
    assert all(row["checkRect"][2] == 0 and row["arrowSlotRect"][2] == 0 for row in records)
    assert all(bool(row["iconSlotRect"][2]) == bool(icon) for row in records)
    assert len({row["labelRect"][0] for row in records}) == 1
    assert state.menus.observed["Commands"]["rect"][2] < 224 * ui.s
    inset = native_unit(10 * ui.s)
    assert all(row["content"][0] - row["hit"][0] == inset for row in records)
    if not icon:
        assert all(row["labelRect"][0] - row["hit"][0] == inset for row in records)


def test_checkable_menu_keeps_columns_when_all_options_are_off(compact_menu):
    state = compact_menu
    state.rows = (Item("One", checkable=True), Item("Two", checkable=True))
    before = state.frame()
    state.rows = tuple(replace(row, checked=True) for row in state.rows)
    checked = state.frame()
    state.rows = tuple(replace(row, checked=False) for row in state.rows)
    after = state.frame()
    for records in (before, checked, after):
        assert all(row["checkRect"][2] > 0 for row in records)
    assert [row["labelRect"] for row in before] == [row["labelRect"] for row in after]
    assert [row["labelRect"] for row in before] == [row["labelRect"] for row in checked]


def test_legacy_checked_items_infer_their_checkable_capability(appearance):
    assert Item("Legacy", checked=True).checkable
    assert not Item("Command").checkable


def test_named_submenu_does_not_pollute_its_parent_local_menu_inventory(appearance):
    ui, _runtime, _update, frame = appearance
    menus = Menus(SimpleNamespace(ui=ui, s=ui.s), ())
    menus.items = lambda _name: (Item("A much longer child command", shortcut="Cmd+Shift+Z"),)
    menus.preview(("Nested",))

    def draw():
        menus.begin_frame()
        if not imgui.is_popup_open("Outer"):
            imgui.open_popup("Outer")
        with menus.local_popup("Outer") as opened:
            if opened:
                menus.local_item("Plain", key="plain")
                menus.reveal(("Nested",))
                menus.constrain("Nested")
                with menus.local_submenu("Nested", key="Nested", content=False) as child:
                    if child:
                        menus.content("Nested", ("Nested",))

    for _ in range(12):
        frame(draw)
    assert "Nested" in menus.observed
    assert [row.label for row in menus._popup_rows["Outer"]] == ["Plain", "Nested"]
    assert menus._popup_widths["Outer"] < menus.observed["Nested"]["rect"][2]


def test_fractional_native_popup_auto_height_contains_the_entire_final_row(appearance):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    menus = Menus(SimpleNamespace(ui=ui, s=ui.s), ())

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        menus.begin_frame()
        if not imgui.is_popup_open("Fractional rows"):
            imgui.open_popup("Fractional rows")
        with menus.local_popup("Fractional rows") as opened:
            if opened:
                menus.local_item(
                    "Transform handles", checked=False, checkable=True, key="handles", icon="move"
                )
                menus.local_separator()
                menus.local_item("Hide panel", key="hide", icon="close")

    for _ in range(14):
        frame(draw)
    row = next(row for row in runtime.observed if row["key"] == "hide")
    x, y, width, height = row["hit"]
    clip = row["clip"]
    assert x >= clip[0] and x + width <= clip[2]
    assert y >= clip[1] and y + height <= clip[3]


@pytest.mark.parametrize("part,inset", (("menu", 6), ("select_popup", 8)))
def test_native_popup_without_design_runtime_retains_its_class_insets_and_radius(
    appearance, part, inset
):
    ui, _runtime, _update, frame = appearance
    ui.design = None

    def draw():
        if not imgui.is_popup_open("Native baseline"):
            imgui.open_popup("Native baseline")
        with menu_popup_style(ui, part=part):
            prepare_menu_popup(ui, auto_minimum_width=180 * ui.s, part=part)
            if imgui.begin_popup("Native baseline"):
                with menu_popup_content(ui, "Native baseline", part=part) as box:
                    imgui.dummy((80 * ui.s, 30 * ui.s))
                    actual = box, imgui.internal.get_current_window().window_rounding
                imgui.end_popup()
                return actual
        return None

    for _ in range(5):
        box, rounding = frame(draw)
    assert box.padding == (native_unit(inset * ui.s),) * 4
    expected = 6 * ui.s + native_unit(inset * ui.s)
    assert rounding == pytest.approx(expected)


def test_native_text_budget_only_recovers_float32_coordinate_loss(appearance):
    epsilon = native_epsilon(529.2, 760.2)
    assert epsilon < 0.001
    assert text_budget(35.99998779296872, 36, epsilon) == 36
    assert text_budget(35.5, 36, epsilon) == 35.5
