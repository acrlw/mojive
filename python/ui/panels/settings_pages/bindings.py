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
    actions = tuple(
        action
        for action in InputAction
        if action is not InputAction.PERTURB or ctx.session.adapter.caps.perturb
    )
    for row in panel._property_rows(tuple(t(input_action_name(action)) for action in actions)):
        action = actions[row]
        panel._property(t(input_action_name(action)), control_part="select")
        current = ctx.input_bindings.key_id(action)
        index = identifiers.index(current)
        changed, index = panel._combo(
            f"##shortcut_{action.value}",
            index,
            labels,
        )
        panel._tooltip(t("A key already in use swaps the two actions"))
        if changed and ctx.set_input_binding is not None:
            try:
                ctx.apply_setting(ctx.set_input_binding, action, identifiers[index])
                panel._pointer_error = ""
            except ValueError as error:
                panel._pointer_error = str(error)
    panel._property("", control_part="button")
    if panel._button(t("Reset shortcuts")) and ctx.reset_input_bindings is not None:
        ctx.apply_setting(
            ctx.reset_input_bindings,
        )
    panel._end_properties()
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
        panel._property(t("Navigation preset"), control_part="select")
        names = tuple(NAVIGATION_PRESETS)
        changed, selected = panel._choice_list(
            "##navigation_preset", -1, names, preview=t("Choose preset...")
        )
        if changed and ctx.set_navigation_preset is not None:
            try:
                ctx.apply_setting(ctx.set_navigation_preset, names[selected])
                panel._pointer_error = ""
            except ValueError as error:
                panel._pointer_error = str(error)
        panel._end_properties()
    if panel._begin_properties("settings_mouse_gestures"):
        actions = tuple(
            action
            for action in PointerAction
            if not action.value.startswith("perturb.") or ctx.session.adapter.caps.perturb
        )
        for row in panel._property_rows(tuple(t(POINTER_ACTION_NAMES[a]) for a in actions)):
            action = actions[row]
            panel._property(t(POINTER_ACTION_NAMES[action]), control_part="input_text")
            submitted, value = panel._input_text(
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
        panel._end_properties()
    if panel._pointer_error:
        imgui.text_wrapped(ctx.tr(panel._pointer_error))
