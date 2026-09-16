"""Environment Inspector controls for adapter-owned physics options."""

from __future__ import annotations

from itertools import groupby

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.controls import begin_property_table, property_row
from mojive.ui.panels import PanelContext


def _format(value) -> str:
    if isinstance(value, tuple):
        return " ".join(f"{item:.12g}" for item in value)
    return str(value) if isinstance(value, int) else f"{value:.12g}"


class _Options:
    """Present one physics world; do not store engine options in UI preferences."""

    def show_environment_physics(self) -> None:
        self._environment_tab = 1

    def _physics_options(self, ctx: PanelContext) -> None:
        options = ctx.session.physics_options
        if not options:
            imgui.text_wrapped(
                ctx.tr(
                    "Physics options are controlled by the simulation owner."
                    if ctx.session.adapter.caps.external_clock
                    else "This adapter does not expose physics options."
                )
            )
            return
        identity = (
            ctx.session.document_id,
            ctx.session.document_revision,
            ctx.session.structure_generation,
        )
        if self._physics_document != identity:
            self._physics_document = identity
            self._physics_text.clear()
            self._physics_error = ""
        if self._physics_error:
            imgui.text_wrapped(self._physics_error)
        changes = {}
        invalid = False
        for index, (group, fields) in enumerate(groupby(options, key=lambda option: option.group)):
            fields = tuple(fields)
            flags = imgui.TreeNodeFlags_.default_open if index < 2 else 0
            opened = imgui.collapsing_header(group, flags)
            if fields[0].group_description and imgui.is_item_hovered():
                imgui.set_tooltip(ctx.tr(fields[0].group_description))
            if not opened:
                continue
            labels = tuple(field.label for field in fields)
            if not begin_property_table(f"physics-{group}", labels=labels):
                continue
            for field in fields:
                property_row(
                    field.label,
                    tooltip=ctx.tr(field.description) if field.description else field.key,
                )
                item_id = f"##physics-{field.key}"
                if field.kind == "bool":
                    changed, value = imgui.checkbox(item_id, field.value)
                    if changed:
                        changes[field.key] = value
                elif field.kind == "choice":
                    preview = next(
                        (label for label, value in field.choices if value == field.value),
                        str(field.value),
                    )
                    if imgui.begin_combo(item_id, preview):
                        for label, value in field.choices:
                            if (
                                imgui.selectable(label, field.value == value)[0]
                                and value != field.value
                            ):
                                changes[field.key] = value
                        imgui.end_combo()
                else:
                    text = self._physics_text.get(field.key, _format(field.value))
                    entered, text = imgui.input_text(
                        item_id, text, imgui.InputTextFlags_.enter_returns_true
                    )
                    self._physics_text[field.key] = text
                    if entered or imgui.is_item_deactivated_after_edit():
                        try:
                            if field.kind == "vector":
                                value = tuple(float(part) for part in text.split())
                            else:
                                value = int(text) if field.kind == "int" else float(text)
                            if text != _format(field.value) and value != field.value:
                                changes[field.key] = value
                            else:
                                self._physics_error = ""
                        except ValueError:
                            self._physics_error = (
                                f"{field.label}: {ctx.tr('Enter a valid number.')}"
                            )
                            invalid = True
                    elif not imgui.is_item_active():
                        self._physics_text.pop(field.key, None)
            imgui.end_table()
        if changes and not invalid:
            result = ctx.submit(cmd.SetPhysicsOptions(changes))
            self._physics_error = "" if result.ok else result.message
            if result.ok:
                self._physics_text.clear()
