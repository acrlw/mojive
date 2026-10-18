"""Native presentation for the registered Inspector's real component controls."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.panels import pointer_pressed
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.workspace.button_rows import draw_button_row, measure_button_row
from mojive.ui.workspace.choice_popup import ChoicePopup
from mojive.ui.workspace.color_editor import (
    ColorEditor,
    color_popup,
    color_popup_width,
    place_color_popup,
)
from mojive.ui.workspace.component_layout import control_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN, MONO_SIZE
from mojive.ui.workspace.property_groups import PropertyGroups
from mojive.ui.workspace.property_style import (
    advance,
    property_label,
    property_section,
    switch_form_layout,
)
from mojive.ui.workspace.widgets import MUTED, TEXT

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
        self.groups = PropertyGroups(ui)
        self.choices = ChoicePopup(ui)
        self.ctx = None
        self.canceled = False
        self._row = None
        self._control_width = None
        self._row_unit = ""
        self._toggle_row = None
        self._color_identity = None
        self._color_defaults = {}
        self._color_editors = {}
        self._vector_group = False
        self._control_minimum = 96
        self._table_id = ""
        self._switch_labels = {}
        self._current_switch_labels = []
        self.color_edit_open = False
        self.color_edit_started = False
        self.color_edit_action = None
        self.color_edit_owner = None

    def _field_width(self):
        return imgui.calc_item_width() if self._control_width is None else self._control_width

    def tooltip(self, text, *, hovered=False):
        if hovered or imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
            self.ui.tooltip(text, translate=False)

    def begin_table(self, table_id):
        self._control_width = None
        self._table_id = table_id
        self._current_switch_labels = []
        self._control_minimum = 164 if table_id == "inspector_camera_projection" else 96
        self._vector_group = table_id in XYZ_COLUMN_TABLES
        flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        opened = imgui.begin_table(table_id, 1, flags)
        if opened:
            self.groups.begin(table_id)
        return opened

    def end_table(self):
        self._switch_labels[self._table_id] = tuple(self._current_switch_labels)
        self.groups.end()
        self._control_width = None

    def component_actions(self, part, key, label, labels, enabled):
        from mojive.ui.workspace.model_components import component_actions

        return component_actions(self.ui, part, key, label, labels, enabled)

    def _draw_label(
        self, key, text, x, y, width, *, above, row_height=DESIGN.control_height, parent="form_row"
    ):
        scoped_key = self._table_id + "/" + key
        result = property_label(
            self.ui,
            scoped_key,
            text,
            x,
            y,
            width,
            above=above,
            row_height=row_height,
            parent=parent,
        )
        self.ui.hits[key] = self.ui.hits[scoped_key]
        return result

    def row(
        self,
        label,
        tooltip="",
        *,
        unit="",
        toggle_unit=None,
        toggle=False,
        components=1,
        control_part="input_number",
        control_labels=(),
    ):
        imgui.table_next_row()
        imgui.table_next_column()
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        gx, gy, width = self.groups.row(p.x, p.y, width)
        p = imgui.ImVec2(gx, gy)
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
        # Color channels wrap inside the same scalar control column. Reserving
        # three/four full fields here would shift only color labels to the left.
        components = 1 if color_channels else components
        group_vector = self._vector_group
        minimum_height = None
        if control_part == "segment":
            minimum_height = self.ui.segment_height()
        elif control_part == "button_row" and control_labels:
            first = self.ui.form_layout(width, 1, field_part="button_row")
            available = max(s, width - first.fields[0][0])
            minimum_height = measure_button_row(self.ui, available, control_labels).outer_height
        if toggle:
            self._current_switch_labels.append(shown)
        shared_label_width = None
        shared_above = None
        captions = (*self._switch_labels.get(self._table_id, ()), *self._current_switch_labels)
        if captions and not group_vector:
            # Scalar fields and switches share one measured caption column.
            # Cache captions, never width decisions, so resizing is reversible.
            ui = self.ui
            label_style = ui.component_style(("switch", "form_label"))
            label_margin = label_style.edges("margin", s)
            row_style = ui.component_style("form_row")
            row_margin = row_style.edges("margin", s)
            available = max(0, width - row_margin[0] - row_margin[2])
            shared_label_width = max(
                min(row_style.get("labelWidth", DESIGN.label_width) * s, available * 0.28),
                max(
                    ui.measure(
                        text,
                        label_style.get("fontSize", BODY_SIZE),
                        translate=False,
                        font_resolved=True,
                    )
                    for text in captions
                )
                + label_margin[0]
                + label_margin[2],
            )
            gap = row_style.get("columnGap", row_style.get("labelGap", DESIGN.label_gap)) * s
            shared_above = shared_label_width + gap + self._control_minimum * s > available
        layout = self.ui.form_layout(
            width,
            1 if toggle else max(3, components) if group_vector else components,
            axis=not toggle and (group_vector or bool(color_channels)),
            minimum=0 if group_vector else self._control_minimum * s,
            label_above=shared_above if shared_above is not None else False if toggle else None,
            label_width=shared_label_width,
            field_part="switch" if toggle else control_part,
            minimum_height=minimum_height,
        )
        if components > 1 and not toggle:
            # The caption shares the group's column, but numbers() owns the
            # actual plain-field grid. An XYZ estimate can reserve extra rows
            # even when all of the formatted plain numbers fit inline.
            layout = self.ui.form_layout(
                width,
                1,
                axis=False,
                minimum=0 if group_vector else self._control_minimum * s,
                label_above=layout.label_above,
                label_width=layout.label_width,
                field_part=control_part,
                minimum_height=minimum_height,
            )
        if toggle and group_vector:
            anchor = self.ui.form_layout(width, 3, axis=True, minimum=0)
            label_style = self.ui.component_style(("switch", "form_label"))
            label_margin = label_style.edges("margin", s)
            caption_width = self.ui.measure(
                shown, label_style.get("fontSize", BODY_SIZE), translate=False, font_resolved=True
            )
            shared_label_width = anchor.label_width
            shared_above = (
                anchor.label_above
                or caption_width + label_margin[0] + label_margin[2] > anchor.label_width
            )
        if toggle:
            layout = switch_form_layout(
                self.ui,
                width,
                shown,
                label_width=shared_label_width,
                label_above=shared_above,
            )
        above = layout.label_above
        switch_style = self.ui.component_style("switch")
        switch_margin = switch_style.edges("margin", s)
        if toggle and not above and self.ui.design is not None:
            switch_height = self.ui.design.primitives.field_height("switch") * s
            row_style = self.ui.component_style("form_row")
            row_margin = row_style.edges("margin", s)
            cross = max(
                layout.height - row_margin[1] - row_margin[3],
                switch_height + switch_margin[1] + switch_margin[3],
            )
            align = row_style.get("alignItems", "center")
            offset = max(0, cross - switch_height - switch_margin[1] - switch_margin[3]) * (
                0.5 if align == "center" else 1 if align == "end" else 0
            )
            layout = replace(
                layout,
                fields=tuple(
                    (dx, row_margin[1] + offset, field_width, switch_height)
                    for dx, _, field_width, _ in layout.fields
                ),
                height=cross + row_margin[1] + row_margin[3],
                label_height=cross,
                minimum_content_height=max(
                    layout.label_line_height, switch_height + switch_margin[1] + switch_margin[3]
                ),
            )
        row_margin = layout.row_margin
        label_width = layout.label_width
        row_height = DESIGN.control_height
        hit_width, _ = self._draw_label(
            "property-label-" + label,
            shown,
            p.x + layout.label_x,
            p.y + layout.label_y,
            label_width,
            above=above,
            row_height=layout.label_height / s,
            parent="switch" if toggle else "form_row",
        )
        published_height = layout.height
        if group_vector and components == 1 and not toggle:
            margin = self.ui.component_style("form_row").edges("margin", s)
            published_height = (
                max(
                    layout.fields[0][1] + layout.fields[0][3],
                    layout.label_y + layout.label_height,
                )
                + margin[3]
            )
        if self.ui.design is not None:
            minimum_content_height = layout.minimum_content_height
            if group_vector and components == 1 and not toggle:
                child_margin = self.ui.component_style(control_part).edges("margin", s)
                minimum_content_height = max(
                    layout.label_line_height,
                    layout.fields[0][3] + child_margin[1] + child_margin[3],
                )
            self.ui.design.record(
                self._table_id + "/row/" + label,
                ("form_row",),
                [p.x, p.y, width, published_height],
                "form_row",
                layout="top" if above else "left",
                minContentHeight=minimum_content_height,
                outer=[p.x, p.y, width, published_height],
                frameRect=[
                    p.x + layout.row_margin[0],
                    p.y + layout.row_margin[1],
                    max(0, width - layout.row_margin[0] - layout.row_margin[2]),
                    max(0, published_height - layout.row_margin[1] - layout.row_margin[3]),
                ],
                **(
                    {
                        "role": "left-aligned-switch",
                        "ownedSpacingPart": "switch",
                        "labelTypographyPart": "switch",
                        "fontSizeRole": "minimum-row-height",
                        "consumedFields": {
                            "form_row": [
                                "height",
                                "labelWidth",
                                "labelGap",
                                "columnGap",
                                "rowGap",
                                "fontSize",
                                "lineHeightMode",
                                "lineHeight",
                                "alignItems",
                                "marginLeft",
                                "marginTop",
                                "marginRight",
                                "marginBottom",
                            ]
                        },
                    }
                    if toggle
                    else {}
                ),
                fields=[
                    [
                        p.x + layout.label_x,
                        p.y + layout.label_y,
                        layout.label_width,
                        layout.label_height,
                    ],
                    *[[p.x + dx, p.y + dy, fw, fh] for dx, dy, fw, fh in layout.fields],
                ],
                labelRect=[
                    p.x + layout.label_x,
                    p.y + layout.label_y,
                    label_width,
                    layout.label_height,
                ],
            )
        self._toggle_row = (
            (
                [p.x, p.y, width, published_height],
                [p.x + layout.label_x, p.y + layout.label_y, label_width, layout.label_height],
            )
            if toggle
            else None
        )
        advance(self.ui, p.x, p.y + published_height)
        imgui.set_cursor_screen_pos((p.x, p.y))
        label_height = layout.fields[0][1] if above else row_height * s
        imgui.begin_disabled(hit_width <= 0)
        clicked = imgui.invisible_button(
            "##property-label-" + label, (max(1, hit_width), max(1, label_height))
        )
        imgui.end_disabled()
        if clicked and toggle_unit:
            toggle_unit()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            if toggle_unit:
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
            if self.ui.measure(shown, BODY_SIZE, translate=False) > label_width:
                self.ui.tooltip(shown, translate=False)
        dx, dy, _, _ = layout.fields[0]
        control_x, control_y = round(p.x + dx), p.y + dy
        self._row = (control_x, control_y, max(0, width - dx - row_margin[2]))
        self._row_unit = unit
        imgui.set_cursor_screen_pos((control_x, control_y))
        self._control_width = max(0, width - (control_x - p.x) - row_margin[2])
        imgui.set_next_item_width(max(1, self._control_width))

    def section(self, label, flags):
        if label == "advanced joint properties":
            flags &= ~imgui.TreeNodeFlags_.default_open
        opened = property_section(
            self.ui, _label(self.ctx.tr(label)).upper(), flags, item_id="native-section-" + label
        )
        self._row = None
        self._control_width = None
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
        kind_width = self.ui.measure(kind, BODY_SIZE, translate=False)
        detail_width = self.ui.measure(detail, BODY_SIZE, translate=False)
        inline = kind_width + detail_width + 16 * s <= width
        self.ui.text(p.x, p.y + 3 * s, kind, TEXT, BODY_SIZE, translate=False)
        self.ui.text(
            p.x + width - detail_width if inline else p.x,
            p.y + (3 if inline else 24) * s,
            detail,
            MUTED,
            BODY_SIZE,
            translate=False,
        )
        imgui.invisible_button("##joint-metadata", (width, (24 if inline else 45) * s))
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits["joint-metadata"] = (a.x, a.y, b.x - a.x, b.y - a.y)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            self.ui.tooltip(self.ctx.tr("qpos address") + f": {joint.qpos_adr}\nDOF: {joint.dof}")

    def _advance(self, x, y):
        advance(self.ui, x, y)

    def _field(
        self,
        item_id,
        value,
        x,
        y,
        width,
        *,
        axis=None,
        speed,
        lo,
        hi,
        fmt,
        reset=0.0,
        integer=False,
        flags=0,
    ):
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
            integer=integer,
            flags=flags,
        )
        self.canceled |= self.ui.numeric_edits[item_id].canceled
        if imgui.is_item_hovered() and pointer_pressed(self.ctx, PointerAction.PROPERTY_COPY):
            imgui.set_clipboard_text(fmt % value)
        return changed, edited

    def button(self, label, size=None):
        pos, ui = imgui.get_cursor_screen_pos(), self.ui
        caption = label.split("##", 1)[0]
        style = ui.component_style("button")
        padding = style.edges("padding", ui.s, (8, 0, 8, 0))
        width = (
            ui.measure(caption, style.get("fontSize", DESIGN.body_size), font_resolved=True)
            + padding[0]
            + padding[2]
        )
        if size is not None and size[0] != 0:
            width = imgui.get_content_region_avail().x if size[0] < 0 else size[0]
        width = min(max(ui.s, width), imgui.get_content_region_avail().x)
        height = DESIGN.control_height if size is None or not size[1] else size[1] / ui.s
        margin = style.edges("margin", ui.s)
        limit = max(0, imgui.get_content_region_avail().x - margin[0] - margin[2])
        return ui.button(
            label,
            pos.x,
            pos.y,
            width / ui.s,
            height,
            label=caption,
            translate=False,
            width_limit=limit,
        )

    def button_row(self, key, labels, enabled=None):
        return draw_button_row(self.ui, key, labels, enabled, available=self._field_width())

    def combo(self, key, index, labels, flags=0):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.combo(key, pos.x, pos.y, self._field_width(), index, labels, flags=flags)

    def begin_combo(self, key, preview, flags=0):
        pos = imgui.get_cursor_screen_pos()
        opened = self.ui.begin_combo(
            key,
            pos.x,
            pos.y,
            self._field_width(),
            preview,
            flags=flags,
            popup_width=self.choices.minimum_width(key),
        )
        if opened:
            self.choices.begin(key)
        return opened

    def choice(self, label, selected=False, *, enabled=True):
        return self.choices.choice(label, selected, enabled=enabled)

    def end_combo(self):
        self.choices.end()

    def numeric(self, item_id, value, speed, lo, hi, fmt, flags, *, integer=False):
        fmt = _number_format(fmt, self._row_unit)
        p = imgui.get_cursor_screen_pos()
        width = self._field_width()
        return self._field(
            item_id,
            value,
            p.x,
            p.y,
            width,
            speed=speed,
            lo=lo,
            hi=hi,
            fmt=fmt,
            integer=integer,
            flags=flags,
        )

    def numbers(self, item_id, values, speed, lo, hi, fmt, flags):
        fmt = _number_format(fmt, self._row_unit)
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        count = len(values)
        width = self._field_width()
        layout = self.ui.numeric_layout(
            width,
            count,
            axis=False,
            minimum=self.ui.numeric_minimum(axis=False, format=fmt),
        )
        changed = False
        out = list(values)
        imgui.begin_group()
        for index, value in enumerate(values):
            dx, dy, field_width, _ = layout.fields[index]
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
        advance(self.ui, p.x, p.y + layout.height)
        imgui.end_group()
        self.ui.record_numeric_group(item_id + "/controls", p.x, p.y, width, layout)
        return changed, out

    def toggle(self, item_id, value):
        p = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        if self._row is not None:
            x, y, width = self._row
        else:
            x, y = p.x, p.y
        updated = self.ui.switch(item_id, x, y, value, width=width)
        if self.ui.design is not None and self._toggle_row is not None:
            row_rect, label_rect = self._toggle_row
            self.ui.design.record(
                item_id + "/label-row",
                ("switch",),
                row_rect,
                "switch_row",
                labelRect=label_rect,
                switchRect=list(self.ui.hits[item_id]),
                consumedFields={
                    "switch": [
                        "height",
                        "fontSize",
                        "lineHeightMode",
                        "lineHeight",
                        "gap",
                        "columnGap",
                        "switchWidth",
                        "switchHeight",
                        "switchThumb",
                        "marginLeft",
                        "marginTop",
                        "marginBottom",
                    ]
                },
            )
        return updated != value, updated

    def value_rail(self, item_id, value, bounds, **options):
        active = imgui.is_any_item_active()
        p = imgui.get_cursor_screen_pos()
        result = self.ui.rail(
            item_id, p.x, p.y, self._field_width(), value, bounds, show_unit=False, **options
        )
        self.canceled |= (active or imgui.is_any_item_active()) and imgui.is_key_pressed(
            imgui.Key.escape, False
        )
        return result

    def text(self, item_id, value, flags, *, mono=False):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.input_text(
            item_id, pos.x, pos.y, self._field_width(), value, flags=flags, mono=mono
        )

    def input_text_multiline(self, item_id, value, size, flags=0):
        """Keep multiline editing native inside the authored padding and margins."""
        pos, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
        s, ui = self.ui.s, self.ui
        width = max(s, size.x if size.x > 0 else available.x + size.x + 1)
        height = max(s, size.y if size.y > 0 else available.y + size.y)
        if ui.design is None:
            return imgui.input_text_multiline(item_id, value, (width, height), flags)
        style = ui.component_style("textarea")
        margin, padding = style.edges("margin", s), style.edges("padding", s, (10, 10, 10, 10))
        minimum = max(
            style.get("minHeight", 90) * s,
            style.get("fontSize", BODY_SIZE) * s + padding[1] + padding[3],
        )
        if size.y > 0:
            # A table cell reports its current row height before the child is
            # submitted. Its intrinsic multiline height must grow that row.
            height = max(height, minimum + margin[1] + margin[3])
        painter = ui.design.primitives
        painter.family = ui.design.family()
        result = painter.input_text(
            item_id,
            Rect(pos.x, pos.y, width, max(s, height)),
            value,
            multiline=True,
            parts="textarea",
            flags=flags,
        )
        advance(ui, pos.x, pos.y + result.box.outer.height)
        return result.changed, result.value

    def segmented(self, item_id, labels, selected, *, icons=None):
        pos = imgui.get_cursor_screen_pos()
        if icons is not None:
            names = {"persp": "perspective", "ortho": "orthographic"}
            icons = tuple(names.get(name, name) for name in icons)
        return self.ui.segmented(
            item_id,
            pos.x,
            pos.y,
            self._field_width() / self.ui.s,
            labels,
            selected,
            icons=icons,
            layout="content",
        )

    def color(self, item_id, values):
        p = imgui.get_cursor_screen_pos()
        width = self._field_width()
        s = self.ui.s
        rgba = tuple(values) if len(values) == 4 else (*values, 1.0)
        swatch_width = min(DESIGN.control_height * s, max(0, width))
        self.ui.draw.push_clip_rect(
            (p.x, p.y), (p.x + max(0, width), p.y + DESIGN.control_height * s), True
        )
        imgui.begin_disabled(swatch_width <= 0)
        pressed = imgui.color_button(
            item_id + "-picker",
            rgba,
            imgui.ColorEditFlags_.no_tooltip,
            (max(1, swatch_width), DESIGN.control_height * s),
        )
        imgui.end_disabled()
        self.ui.draw.pop_clip_rect()
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.ui.hits[item_id + "-picker"] = (a.x, a.y, swatch_width, b.y - a.y)
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
        popup_width = color_popup_width(self.ui, len(out))
        imgui.set_next_window_size_constraints(
            (popup_width, 0), (popup_width, max(80 * s, viewport.work_size.y - 16 * s))
        )
        place_color_popup(self.ui, popup)
        with color_popup(self.ui, popup) as opened:
            if opened:
                changed, out = editor.draw(
                    self.ui,
                    popup,
                    out,
                    imgui.get_content_region_avail().x,
                    footer_height=control_size(
                        self.ui, popup + "-done", 80, DESIGN.control_height, label="Apply"
                    )[1],
                )
                self.canceled |= any(
                    self.ui.numeric_edits.get(popup + axis)
                    and self.ui.numeric_edits[popup + axis].canceled
                    for axis in "RGBA"
                )
                if self.canceled or imgui.is_key_pressed(imgui.Key.escape, False):
                    advance(self.ui, *imgui.get_cursor_screen_pos())
                    self.color_edit_action = "cancel"
                    changed = False
                    imgui.close_current_popup()
                else:
                    footer = imgui.get_cursor_screen_pos()
                    available = editor.content_width
                    gap = self.ui.metric("color_editor", "columnGap", 6) * s
                    half = (available - gap) / 2
                    if self.ui.button(
                        popup + "-cancel", footer.x, footer.y, half / s, label="Cancel"
                    ):
                        self.color_edit_action = "cancel"
                        changed = False
                        imgui.close_current_popup()
                    if self.ui.button(
                        popup + "-done",
                        footer.x + half + gap,
                        footer.y,
                        half / s,
                        label="Done",
                        active=True,
                        enabled=not editor.error,
                    ):
                        self.color_edit_action = "apply"
                        imgui.close_current_popup()
        if owns_popup:
            self.color_edit_open = imgui.is_popup_open(popup)
            if was_open and not self.color_edit_open and self.color_edit_action is None:
                self.color_edit_action = (
                    "cancel" if imgui.is_key_pressed(imgui.Key.escape, False) else "apply"
                )
        hex_color = "#" + "".join(f"{round(np.clip(value, 0, 1) * 255):02X}" for value in out[:3])
        self.ui.text(
            b.x + 8 * self.ui.s,
            p.y + DESIGN.control_padding_y(MONO_SIZE) * self.ui.s,
            hex_color,
            TEXT,
            MONO_SIZE,
            mono=True,
            max_width=max(0, width - (b.x - p.x) - 8 * s),
        )
        x = p.x
        y = b.y + DESIGN.row_gap * self.ui.s
        with self.ui.component_scope("rgba"):
            number_format = "%.3f"
            layout = self.ui.numeric_layout(
                width,
                len(out),
                column_options=(len(out), 1),
                minimum=self.ui.color_channel_minimum(format=number_format),
                parts="rgba",
            )
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
                    fmt=number_format,
                    reset=resets[index],
                )
                changed |= edit
            advance(self.ui, x, y + layout.height)
            imgui.end_group()
            self.ui.record_numeric_group(item_id + "-channels", x, y, width, layout, parts="rgba")
        return changed, out

    @part_scope("vector_row")
    def _vector_caption_width(self, captions):
        return max(self.ui.vector_caption_width(text) for text in captions)

    @part_scope("vector_row")
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
        fmt = _number_format(fmt, unit)
        layout = layout or self.ui.vector_layout(
            width,
            format=fmt,
            label_minimum=self._vector_caption_width((shown,)),
            key=f"{node.node_id}-{name}",
        )
        above, label_width = layout.label_above, layout.label_width
        imgui.begin_group()
        hit_width, label_height = self._draw_label(
            "vector-label-" + name,
            shown,
            p.x + layout.label_x,
            p.y + layout.label_y,
            label_width,
            above=above,
            row_height=layout.label_height / s,
            parent="vector_row",
        )
        extra_height = max(0, label_height - BODY_SIZE * s)
        imgui.set_cursor_screen_pos((p.x, p.y))
        label_bounds_height = layout.fields[0][1] if above else DESIGN.control_height * s
        imgui.begin_disabled(hit_width <= 0)
        imgui.invisible_button(
            "##vector-label-" + name,
            (max(1, hit_width), max(1, label_bounds_height + extra_height)),
        )
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            if pointer_pressed(self.ctx, PointerAction.PROPERTY_COPY):
                imgui.set_clipboard_text(", ".join(f"{float(value):.6g}" for value in values))
            if self.ui.measure(shown, BODY_SIZE, translate=False) > label_width:
                self.ui.tooltip(shown, translate=False)
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
        advance(self.ui, p.x, p.y + layout.height + extra_height)
        imgui.end_group()
        imgui.end_disabled()
        self.ui.record_form_group(f"{node.node_id}/{name}", p.x, p.y + extra_height, layout)
        return changed, out

    @part_scope("vector_row")
    def vectors(self, node, table_id, rows, editable, *, units=()):
        result = []
        minimum = max(
            self.ui.numeric_minimum(format=_number_format(row[3], units[index] if units else ""))
            for index, row in enumerate(rows)
        )
        layout = self.ui.vector_layout(
            imgui.get_content_region_avail().x,
            minimum=minimum,
            label_minimum=self._vector_caption_width(
                _label(self.ctx.tr(row[0]), units[index] if units else "")
                for index, row in enumerate(rows)
            ),
            key=f"{node.node_id}-{table_id}",
        )
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

    @part_scope("vector_row")
    def vector_row(self, node, label, name, values, **options):
        imgui.table_next_row()
        imgui.table_next_column()
        pos = imgui.get_cursor_screen_pos()
        gx, gy, body_width = self.groups.row(pos.x, pos.y, imgui.get_content_region_avail().x)
        imgui.set_cursor_screen_pos((gx, gy))
        options.pop("tooltip", None)
        if self._vector_group:
            # Related rows use the table's XYZ budget, including scalar label columns.
            options["layout"] = self.ui.form_layout(body_width, 3, axis=True, vector=True)
        if "layout" not in options:
            options["layout"] = self.ui.vector_layout(
                body_width,
                format=options.get("fmt"),
                label_minimum=self._vector_caption_width(
                    (_label(self.ctx.tr(label), options.get("unit", "")),)
                ),
                key=f"{node.node_id}-{name}",
            )
        result = self._vector(node, name, label, values, **options)
        return result
