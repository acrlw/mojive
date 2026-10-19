"""Switch fitting measures real ink and controls, not optional shared columns."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.workspace.property_style import (
    NativePropertyPresentation,
    property_controls,
    property_toggle,
    switch_form_layout,
)
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("hint", (None, 300))
def test_short_caption_fits_even_when_preferred_column_does_not(appearance, language, hint):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    ui.translate = Localizer(Language(language)).text
    label = "A" if language == "en" else "开"

    def draw():
        ui.design.begin_frame()
        imgui.set_cursor_screen_pos((40 * ui.s, 40 * ui.s))
        layout = switch_form_layout(
            ui, 92 * ui.s, label, label_width=None if hint is None else hint * ui.s
        )
        property_toggle(
            ui,
            "short",
            label,
            False,
            width=92 * ui.s,
            label_width=None if hint is None else hint * ui.s,
        )
        return layout, ui.design.primitives.observed[-1]

    for _ in range(3):
        layout, painted = frame(draw)
    assert not layout.label_above
    assert painted["textPresent"]
    assert painted["label"] == label
    assert painted["labelRect"][0] + painted["labelRect"][2] <= painted["switchRect"][0]
    assert frame(lambda: switch_form_layout(ui, 400 * ui.s, label, label_above=True)).label_above


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_raw_display_labels_fit_using_the_final_localized_caption(appearance, language):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    ui.translate = Localizer(Language(language)).text

    def draw():
        ui.design.begin_frame()
        width = 118 * ui.s
        hint = switch_form_layout(ui, width, "Transform handles").label_width
        for index, label in enumerate(("Visible", "Transform handles")):
            imgui.set_cursor_screen_pos((40 * ui.s, (40 + index * 90) * ui.s))
            property_toggle(ui, str(index), label, False, width=width, label_width=hint)
        return {r["key"]: r for r in ui.design.primitives.observed}

    for _ in range(3):
        rows = frame(draw)
    assert rows["0"]["textPresent"] and rows["0"]["label"] == ui.translate("Visible")
    assert not rows["1"]["textPresent"]
    assert "1-label-text" in rows


@pytest.mark.parametrize("family", ("inherit", "mono"))
@pytest.mark.parametrize("width", (80, 158))
def test_authored_switch_caption_font_is_used_for_fit_and_wrapped_paint(
    appearance, monkeypatch, family, width
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"switch": {"fontFamily": family}})
    label = "WWWWWWWWWW"
    fonts = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        fonts.append(imgui.get_font().font_id)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def draw():
        ui.design.begin_frame()
        fonts.clear()
        imgui.set_cursor_screen_pos((40 * ui.s, 40 * ui.s))
        property_toggle(ui, "font", label, False, width=width * ui.s)
        expected_font = ui.resolve_font(ui.component_style(("switch", "form_label"))).font_id
        return ui.design.primitives.observed[-1], expected_font

    for _ in range(3):
        painted, expected_font = frame(draw)
    assert painted["textPresent"] is (family == "mono" and width == 158)
    assert fonts and set(fonts) == {expected_font}


@pytest.mark.parametrize("custom", (False, True))
def test_switch_grid_reserves_complete_padding_and_row_margins(appearance, custom):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    if custom:
        update(
            {
                "switch": {
                    "paddingLeft": 20,
                    "paddingRight": 20,
                    "marginLeft": 5,
                    "marginRight": 7,
                },
                "form_row": {"marginLeft": 9, "marginRight": 11},
            }
        )
    presentation = NativePropertyPresentation(ui)
    width = (250 if custom else 180) * ui.s

    def draw():
        ui.design.begin_frame()
        imgui.set_cursor_screen_pos((40 * ui.s, 40 * ui.s))
        imgui.begin_child("switch grid", (width, 300 * ui.s))
        try:
            presentation.toggles((("first", "A", False), ("second", "B", False)))
            return tuple(ui.design.primitives.observed)
        finally:
            imgui.end_child()

    for _ in range(3):
        records = frame(draw)
    rows = [row for row in records if row["key"] in ("first", "second")]
    assert len(rows) == 2 and all(row["textPresent"] for row in rows)
    assert rows[1]["hit"][1] > rows[0]["hit"][1]


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("size", (18, 26))
def test_settings_checkbox_row_uses_square_budget_instead_of_numeric_minimum(
    appearance, monkeypatch, language, size
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    ui.translate = Localizer(Language(language)).text
    update({"checkbox": {"size": size, "marginLeft": 3, "marginRight": 5}})
    fields = NativePropertyPresentation(ui, prefix="settings", fit_labels=True)
    settings = WorkspaceSettings.__new__(WorkspaceSettings)
    settings.workspace = SimpleNamespace(ui=ui, s=ui.s)
    settings.fields = fields
    settings._checkbox_labels = frozenset()
    shown, painted = ui.translate("Copy to clipboard"), []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append(value)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def draw():
        runtime.begin_frame()
        ui.design.begin_frame()
        painted.clear()
        imgui.set_cursor_screen_pos((40 * ui.s, 40 * ui.s))
        imgui.begin_child("settings checkbox", (180 * ui.s, 150 * ui.s))
        try:
            with property_controls(ui):
                assert fields.begin_table("settings-recording")
                settings._property(shown, control_part="checkbox")
                settings._checkbox("##capture_clipboard", False)
                fields.end_table()
            return next(
                r for r in runtime.observed if r["key"] == "settings-recording/row/" + shown
            )
        finally:
            imgui.end_child()

    for _ in range(3):
        row = frame(draw)
    assert row["layout"] == "left"
    assert shown in painted
    label = ui.hits["settings-label-" + shown]
    control = ui.hits["##capture_clipboard"]
    assert control[0] >= label[0] + label[2] + 12 * ui.s - 0.5
