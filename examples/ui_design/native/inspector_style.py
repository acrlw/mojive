"""Native presentation for the registered Inspector's real component controls."""

from __future__ import annotations

import re

import numpy as np
from imgui_bundle import imgui

from mojive.ui.panels import pointer_hint, pointer_pressed
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.pointer_bindings import PointerAction

from .color_editor import ColorEditor
from .widgets import MUTED, SWITCH_WIDTH, TEXT

# Related scalar/vector sections reserve the same XYZ budget for every row.
XYZ_COLUMN_TABLES = frozenset(
    (
        "insp_light_transform",
        "body_inertial_properties",
        "insp_geometry_contact",
        "insp_geometry_advanced",
        "joint_properties_table",
        "joint_advanced_properties_table",
    )
)


def _label(value, unit=""):
    visible = str(value).split("##", 1)[0]
    return visible[:1].upper() + visible[1:] + (f" ({unit})" if unit else "")


def _number_format(fmt, unit):
    # Source metadata supplies the unit; retain the source's numeric precision.
    number = re.fullmatch(r"(%[-+ #0]*\d*(?:\.\d+)?[deEfFgGi])(?:\s*.*)", fmt)
    return number.group(1) if unit and number else fmt


class NativeInspectorPresentation:
    """Present fields without owning Inspector drafts or Session transactions."""

    def __init__(self, ui):
        self.ui = ui
        self.ctx = None
        self.canceled = False
        self._row = None
        self._row_unit = ""
        self._color_identity = None
        self._color_defaults = {}
        self._color_editors = {}
        self._vector_group = False
        self._control_minimum = 96
        self._table_id = ""
        self.color_edit_open = False
        self.color_edit_started = False
        self.color_edit_action = None
        self.color_edit_owner = None

    def begin_table(self, table_id):
        self._table_id = table_id
        self._control_minimum = 164 if table_id == "inspector_camera_projection" else 96
        self._vector_group = table_id in XYZ_COLUMN_TABLES
        flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        return imgui.begin_table(table_id, 1, flags)

    def _draw_label(self, key, text, x, y, width, *, above, row_height=28):
        s = self.ui.s
        height = 13 * s
        top = y + (3 * s if above else max(0, (row_height * s - height) / 2))
        self.ui.text(x, top, text, MUTED, 13, max_width=width, translate=False)
        hit_width = width
        self.ui.hits[key] = (x, top, hit_width, height)
        return hit_width, height

    def row(self, label, tooltip="", *, unit="", toggle_unit=None, toggle=False, components=1):
        imgui.table_next_row()
        imgui.table_next_column()
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        s = self.ui.s
        caption = "Armature" if label == "armature" else label
        shown = _label(self.ctx.tr(caption), unit)
        color_channels = (
            4
            if label in ("base color", "instance color")
            else 3
            if label == "color"
            or (
                self._table_id in ("insp_light_properties", "environment_headlight")
                and label in ("specular", "ambient")
            )
            else 0
        )
        components = color_channels or components
        group_vector = self._vector_group
        layout = self.ui.form_layout(
            width,
            max(3, components) if group_vector else components,
            axis=group_vector or bool(color_channels),
            minimum=0 if group_vector else self._control_minimum * s,
        )
        above = not toggle and layout.label_above
        label_width = max(s, width - (SWITCH_WIDTH + 12) * s) if toggle else layout.label_width
        row_height = 28
        hit_width, text_height = self._draw_label(
            "property-label-" + label,
            shown,
            p.x,
            p.y,
            label_width,
            above=above,
            row_height=row_height,
        )
        imgui.set_cursor_screen_pos((p.x, p.y))
        label_height = text_height + 11 * s if above else row_height * s
        if (
            imgui.invisible_button("##property-label-" + label, (hit_width, label_height))
            and toggle_unit
        ):
            toggle_unit()
        if imgui.is_item_hovered():
            if toggle_unit:
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
            hint = self.ctx.tr("Switch to degrees" if unit == "rad" else "Switch to radians")
            self.ui.tooltip(
                shown + ("\n" + tooltip if tooltip else "") + ("\n" + hint if toggle_unit else "")
            )
        control_x, control_y = (
            (p.x, p.y + (row_height - 28) * s / 2)
            if toggle
            else (p.x, p.y + label_height)
            if above
            else (
                round(p.x + layout.fields[0][0]),
                p.y + (row_height - 28) * s / 2,
            )
        )
        self._row = (p.x, control_y, width)
        self._row_unit = unit
        imgui.set_cursor_screen_pos((control_x, control_y))
        imgui.set_next_item_width(max(s, width - (control_x - p.x)))

    def section(self, label, flags):
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        storage = imgui.get_state_storage()
        key = imgui.get_id("native-section-" + label)
        default_open = bool(flags & imgui.TreeNodeFlags_.default_open)
        if label == "advanced joint properties":
            default_open = False
        opened = bool(storage.get_int(key, int(default_open)))
        next_item = imgui.get_current_context().next_item_data
        if next_item.has_flags & imgui.internal.NextItemDataFlags_.has_open and (
            next_item.open_cond == imgui.Cond_.always or storage.get_int(key, -1) == -1
        ):
            opened = next_item.open_val
            storage.set_int(key, int(opened))
        if self.ui.section(_label(self.ctx.tr(label)).upper(), p.x, p.y, width, opened):
            opened = not opened
            storage.set_int(key, int(opened))
        self._advance(p.x, p.y + (38 if opened else 34) * self.ui.s)
        self._row = None
        self._row_unit = ""
        return opened

    def joint_metadata(self, joint):
        p = imgui.get_cursor_screen_pos()
        s = self.ui.s
        width = imgui.get_content_region_avail().x
        kind = self.ctx.tr(
            {
                "hinge": "Hinge joint",
                "slide": "Slide joint",
                "ball": "Ball joint",
                "free": "Free joint",
            }.get(joint.type, joint.type)
        )
        detail = f"Qpos {joint.qpos_adr}  ·  DOF {joint.dof}"
        kind_width = self.ui.measure(kind, 13, translate=False)
        detail_width = self.ui.measure(detail, 13, translate=False)
        inline = kind_width + detail_width + 16 * s <= width
        self.ui.text(p.x, p.y + 3 * s, kind, TEXT, 13, translate=False)
        self.ui.text(
            p.x + width - detail_width if inline else p.x,
            p.y + (3 if inline else 24) * s,
            detail,
            MUTED,
            13,
            translate=False,
        )
        imgui.invisible_button("##joint-metadata", (width, (24 if inline else 45) * s))
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits["joint-metadata"] = (a.x, a.y, b.x - a.x, b.y - a.y)
        if imgui.is_item_hovered():
            self.ui.tooltip(self.ctx.tr("qpos address") + f": {joint.qpos_adr}\nDOF: {joint.dof}")

    def _advance(self, x, y):
        imgui.set_cursor_screen_pos((x, y))
        spacing = imgui.get_style().item_spacing.x
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (spacing, 0))
        imgui.dummy((0, 0))
        imgui.pop_style_var()

    def _field(self, item_id, value, x, y, width, *, axis=None, speed, lo, hi, fmt, reset=0.0):
        changed, edited = self.ui.input(
            item_id,
            x,
            y,
            width,
            value,
            axis=axis,
            speed=speed,
            lo=lo,
            hi=hi,
            format=fmt,
            reset_value=reset,
        )
        self.canceled |= self.ui.numeric_edits[item_id].canceled
        if imgui.is_item_hovered():
            if pointer_pressed(self.ctx, PointerAction.PROPERTY_COPY):
                imgui.set_clipboard_text(fmt % value)
            self.ui.tooltip(
                str(float(edited if changed else value))
                + "\n"
                + pointer_hint(self.ctx, PointerAction.PROPERTY_COPY, self.ctx.tr("Copy value"))
            )
        return changed, edited

    def numeric(self, item_id, value, speed, lo, hi, fmt, flags, *, integer=False):
        fmt = _number_format(fmt, self._row_unit)
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        if integer:
            imgui.push_font(self.ui.mono, 13 * self.ui.s)
            imgui.push_style_var(imgui.StyleVar_.frame_padding, (5 * self.ui.s, 7.5 * self.ui.s))
            imgui.set_next_item_width(max(self.ui.s, width))
            result = imgui.drag_int(item_id, int(value), speed, int(lo), int(hi), fmt, flags)
            a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.ui.hits[item_id] = (a.x, a.y, b.x - a.x, b.y - a.y)
            imgui.pop_style_var()
            imgui.pop_font()
            return result
        return self._field(item_id, value, p.x, p.y, width, speed=speed, lo=lo, hi=hi, fmt=fmt)

    def numbers(self, item_id, values, speed, lo, hi, fmt, flags):
        fmt = _number_format(fmt, self._row_unit)
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        count = len(values)
        # Reserve precision and sign from the declared format and bounds rather
        # than resizing fields while the user edits a value.
        samples = (0.123456789, 1000.0, -0.123456789) if lo < 0 else (0.123456789, 1000.0)
        imgui.push_font(self.ui.mono, 13 * self.ui.s)
        minimum = max(
            50 * self.ui.s, max(imgui.calc_text_size(fmt % v).x for v in samples) + 16 * self.ui.s
        )
        imgui.pop_font()
        gap = 4 * self.ui.s
        columns = count if count * minimum + (count - 1) * gap <= width else 1
        field_width = (width - (columns - 1) * gap) / columns
        changed = False
        out = list(values)
        imgui.begin_group()
        for index, value in enumerate(values):
            dx, dy = (index % columns) * (field_width + gap), (index // columns) * 36 * self.ui.s
            edit, out[index] = self._field(
                f"{item_id}-{index}",
                value,
                p.x + dx,
                p.y + dy,
                field_width,
                speed=speed,
                lo=lo,
                hi=hi,
                fmt=fmt,
            )
            changed |= edit
        imgui.end_group()
        return changed, out

    def toggle(self, item_id, value):
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        if self._row is not None and self._row[2] >= 88 * self.ui.s:
            x, y, width = self._row
        else:
            x, y = p.x, p.y
        updated = self.ui.switch(item_id, x + max(0, width - SWITCH_WIDTH * self.ui.s), y, value)
        return updated != value, updated

    def value_rail(self, item_id, value, bounds, **options):
        imgui.push_font(self.ui.mono, 13 * self.ui.s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (6 * self.ui.s, 7.5 * self.ui.s))
        active = imgui.is_any_item_active()
        imgui.begin_group()
        result = value_rail(self.ctx, item_id, value, bounds, show_unit=False, **options)
        imgui.end_group()
        self.canceled |= (active or imgui.is_any_item_active()) and imgui.is_key_pressed(
            imgui.Key.escape, False
        )
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits[item_id] = (a.x, a.y, b.x - a.x, b.y - a.y)
        imgui.pop_style_var()
        imgui.pop_font()
        return result

    def text(self, item_id, value, flags):
        imgui.push_font(self.ui.body, 13 * self.ui.s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (5 * self.ui.s, 7.5 * self.ui.s))
        result = imgui.input_text(item_id, value, flags)
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits[item_id] = (a.x, a.y, b.x - a.x, b.y - a.y)
        imgui.pop_style_var()
        imgui.pop_font()
        return result

    def color(self, item_id, values):
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        s = self.ui.s
        rgba = tuple(values) if len(values) == 4 else (*values, 1.0)
        pressed = imgui.color_button(
            item_id + "-picker", rgba, imgui.ColorEditFlags_.no_tooltip, (28 * s, 28 * s)
        )
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits[item_id + "-picker"] = (a.x, a.y, b.x - a.x, b.y - a.y)
        changed, out = False, list(values)
        identity = (self.ctx.session.document_id, self.ctx.session.structure_generation)
        if self._color_identity != identity:
            self._color_identity = identity
            self._color_defaults.clear()
            self._color_editors.clear()
        resets = self._color_defaults.setdefault((self.node_id, item_id), tuple(values))
        owner = (self.node_id, item_id)
        editor = self._color_editors.get(owner)
        if editor is None:
            editor = self._color_editors[owner] = ColorEditor()
        popup = item_id + f"-color-editor-{self.node_id}"
        if pressed:
            editor.begin(values)
            imgui.open_popup(popup)
            self.color_edit_owner = owner
            self.color_edit_started = True
        owns_popup = self.color_edit_owner == owner
        was_open = self.color_edit_open if owns_popup else False
        if imgui.is_item_hovered():
            self.ui.tooltip(self.ctx.tr("Edit color"))
        viewport = imgui.get_main_viewport()
        popup_width = min(320 * s, max(160 * s, viewport.work_size.x - 16 * s))
        imgui.set_next_window_size_constraints(
            (popup_width, 0), (popup_width, max(80 * s, viewport.work_size.y - 16 * s))
        )
        if imgui.begin_popup(popup):
            changed, out = editor.draw(self.ui, popup, out, imgui.get_content_region_avail().x)
            self.canceled |= any(
                self.ui.numeric_edits.get(popup + axis)
                and self.ui.numeric_edits[popup + axis].canceled
                for axis in "RGBA"
            )
            if self.canceled or imgui.is_key_pressed(imgui.Key.escape, False):
                self.color_edit_action = "cancel"
                changed = False
                imgui.close_current_popup()
            else:
                footer = imgui.get_cursor_screen_pos()
                available = imgui.get_content_region_avail().x
                half = (available - 6 * s) / 2
                if self.ui.button(popup + "-cancel", footer.x, footer.y, half / s, label="Cancel"):
                    self.color_edit_action = "cancel"
                    changed = False
                    imgui.close_current_popup()
                if self.ui.button(
                    popup + "-done",
                    footer.x + half + 6 * s,
                    footer.y,
                    half / s,
                    label="Done",
                    active=True,
                    enabled=not editor.error,
                ):
                    self.color_edit_action = "apply"
                    imgui.close_current_popup()
            imgui.end_popup()
        if owns_popup:
            self.color_edit_open = imgui.is_popup_open(popup)
            if was_open and not self.color_edit_open and self.color_edit_action is None:
                self.color_edit_action = (
                    "cancel" if imgui.is_key_pressed(imgui.Key.escape, False) else "apply"
                )
        hex_color = "#" + "".join(f"{round(np.clip(value, 0, 1) * 255):02X}" for value in out[:3])
        self.ui.text(b.x + 8 * self.ui.s, p.y + 7.5 * self.ui.s, hex_color, TEXT, 13, mono=True)
        x = p.x
        y = b.y + 6 * self.ui.s
        layout = self.ui.numeric_layout(width, len(out), column_options=(4, 2, 1))
        imgui.begin_group()
        for index, axis in enumerate("RGBA"[: len(out)]):
            dx, dy, field_width, _ = layout.fields[index]
            edit, out[index] = self._field(
                item_id + axis,
                out[index],
                x + dx,
                y + dy,
                field_width,
                axis=axis,
                speed=0.005,
                lo=0,
                hi=1,
                fmt="%.3g",
                reset=resets[index],
            )
            changed |= edit
        imgui.end_group()
        return changed, out

    def _vector(
        self,
        node,
        name,
        label,
        values,
        *,
        editable,
        speed,
        lo=0,
        hi=0,
        fmt,
        reset_values=None,
        unit="",
        layout=None,
    ):
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        s = self.ui.s
        shown = _label(self.ctx.tr(label), unit)
        layout = layout or self.ui.vector_layout(width)
        above, label_width = layout.label_above, layout.label_width
        imgui.begin_group()
        hit_width, label_height = self._draw_label(
            "vector-label-" + name, shown, p.x, p.y, label_width, above=above
        )
        extra_height = max(0, label_height - 13 * s)
        imgui.set_cursor_screen_pos((p.x, p.y))
        imgui.invisible_button(
            "##vector-label-" + name, (hit_width, (24 if above else 28) * s + extra_height)
        )
        if imgui.is_item_hovered():
            if pointer_pressed(self.ctx, PointerAction.PROPERTY_COPY):
                imgui.set_clipboard_text(", ".join(f"{float(value):.6g}" for value in values))
            self.ui.tooltip(
                shown
                + "\n"
                + pointer_hint(self.ctx, PointerAction.PROPERTY_COPY, self.ctx.tr("Copy XYZ"))
            )
        fmt = _number_format(fmt, unit)
        resets = (0.0, 0.0, 0.0) if reset_values is None else reset_values
        changed = False
        out = np.asarray(values, np.float64).copy()
        imgui.begin_disabled(not editable)
        for index, axis in enumerate("XYZ"):
            dx, dy, field_width, _ = layout.fields[index]
            edit, out[index] = self._field(
                f"##{name}_{index}_{node.node_id}",
                float(out[index]),
                p.x + dx,
                p.y + dy + extra_height,
                field_width,
                axis=axis,
                speed=speed,
                lo=lo,
                hi=hi,
                fmt=fmt,
                reset=float(resets[index]),
            )
            changed |= edit
        imgui.end_group()
        imgui.end_disabled()
        return changed, out

    def vectors(self, node, table_id, rows, editable, *, units=()):
        result = []
        layout = self.ui.vector_layout(imgui.get_content_region_avail().x)
        for index, (label, values, speed, fmt, resets) in enumerate(rows):
            result.append(
                self._vector(
                    node,
                    table_id + label,
                    label,
                    values,
                    editable=editable if isinstance(editable, bool) else editable[index],
                    speed=speed,
                    fmt=fmt,
                    reset_values=resets,
                    unit=units[index] if units else "",
                    layout=layout,
                )
            )
        return tuple(result)

    def vector_row(self, node, label, name, values, **options):
        imgui.table_next_row()
        imgui.table_next_column()
        tooltip = options.pop("tooltip", "")
        result = self._vector(node, name, label, values, **options)
        if tooltip and imgui.is_item_hovered():
            self.ui.tooltip(tooltip)
        return result
