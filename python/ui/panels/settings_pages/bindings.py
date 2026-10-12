"""Bindings settings; apply changes through PanelContext callbacks."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui.input_bindings import InputAction, input_action_name, key_choices
from mojive.ui.panels import (
    PanelContext,
)
from mojive.ui.pointer_bindings import NAVIGATION_PRESETS, POINTER_ACTION_NAMES, PointerAction


def draw(panel, ctx: PanelContext) -> None:
    """Draw conflict-free viewport key remapping from the shared map."""

    t = ctx.tr
    choices = key_choices()
    labels = tuple(t(choice.label) for choice in choices)
    identifiers = tuple(choice.identifier for choice in choices)
    panel._group_heading(t("Shortcuts"))
    if not panel._begin_properties("settings_interaction_shortcuts"):
        return
    for action in InputAction:
        if action is InputAction.PERTURB and not ctx.session.adapter.caps.perturb:
            continue
        panel._property(t(input_action_name(action)))
        current = ctx.input_bindings.key_id(action)
        index = identifiers.index(current)
        changed, index = imgui.combo(
            f"##shortcut_{action.value}",
            index,
            labels,
        )
        imgui.set_item_tooltip(t("A key already in use swaps the two actions"))
        if changed and ctx.set_input_binding is not None:
            try:
                ctx.apply_setting(ctx.set_input_binding, action, identifiers[index])
                panel._pointer_error = ""
            except ValueError as error:
                panel._pointer_error = str(error)
    panel._property("")
    if imgui.button(t("Reset shortcuts")) and ctx.reset_input_bindings is not None:
        ctx.apply_setting(
            ctx.reset_input_bindings,
        )
    imgui.end_table()
    mouse_shortcuts(panel, ctx)


def mouse_shortcuts(panel, ctx: PanelContext) -> None:
    t = ctx.tr
    panel._group_heading(t("Mouse gestures"))
    if ctx.input_bindings is not panel._pointer_bindings_seen:
        panel._pointer_bindings_seen = ctx.input_bindings
        panel._pointer_text = {
            action: "; ".join(
                chord.identifier() for chord in ctx.input_bindings.pointer_chords(action)
            )
            for action in PointerAction
        }
    if panel._begin_properties("settings_navigation_preset"):
        panel._property(t("Navigation preset"))
        if imgui.begin_combo("##navigation_preset", t("Choose preset...")):
            for name in NAVIGATION_PRESETS:
                if imgui.selectable(name, False)[0] and ctx.set_navigation_preset is not None:
                    try:
                        ctx.apply_setting(ctx.set_navigation_preset, name)
                        panel._pointer_error = ""
                    except ValueError as error:
                        panel._pointer_error = str(error)
            imgui.end_combo()
        imgui.end_table()
    imgui.text_wrapped(
        t(
            "Separate alternatives with ;. Example: alt+left; middle. Add :double for a double-click. Press Enter to apply; empty unbinds."
        )
    )
    if panel._begin_properties("settings_mouse_gestures"):
        for action in PointerAction:
            if action.value.startswith("perturb.") and not ctx.session.adapter.caps.perturb:
                continue
            panel._property(t(POINTER_ACTION_NAMES[action]))
            submitted, value = imgui.input_text(
                f"##mouse_{action.value}",
                panel._pointer_text[action],
                imgui.InputTextFlags_.enter_returns_true,
            )
            panel._pointer_text[action] = value
            if submitted and ctx.set_pointer_binding is not None:
                try:
                    ctx.apply_setting(
                        ctx.set_pointer_binding,
                        action,
                        tuple(part.strip() for part in value.split(";") if part.strip()),
                    )
                    panel._pointer_error = ""
                except ValueError as error:
                    panel._pointer_error = str(error)
        imgui.end_table()
    if panel._pointer_error:
        imgui.text_wrapped(panel._pointer_error)
