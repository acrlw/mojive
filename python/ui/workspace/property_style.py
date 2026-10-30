"""Shared native property headings and label geometry."""

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace

from imgui_bundle import imgui

from mojive.design.groups import scoped_parts
from mojive.design.native.geometry import Rect
from mojive.ui.controls import clearable_combo, search_presentation
from mojive.ui.workspace.choice_popup import ChoicePopup
from mojive.ui.workspace.design import BODY_SIZE, DESIGN, rail_style
from mojive.ui.workspace.property_groups import PropertyGroups
from mojive.ui.workspace.widgets import MUTED, TEXT


@contextmanager
def property_controls(ui):
    """Match field height, font and table row spacing across native panels."""
    style = ui.component_style("section_body")
    imgui.push_font(ui.resolve_font(style), style.get("fontSize", DESIGN.body_size) * ui.s)
    ink = ui.style_color(
        style,
        "textColor",
        tuple(round(channel * 255) for channel in imgui.get_style_color_vec4(imgui.Col_.text)),
    )
    imgui.push_style_color(imgui.Col_.text, tuple(channel / 255 for channel in ink))
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding,
        (DESIGN.frame_padding_x * ui.s, DESIGN.control_padding_y(DESIGN.body_size) * ui.s),
    )
    # Tables separate rows through cell padding, rather than item_spacing.
    imgui.push_style_var(
        imgui.StyleVar_.cell_padding,
        (
            0,
            ui.metric("section_body", "rowGap", ui.metric("form_row", "rowGap", DESIGN.row_gap))
            * ui.s
            / 2,
        ),
    )
    imgui.push_style_var(
        imgui.StyleVar_.item_spacing, (DESIGN.item_spacing_x * ui.s, DESIGN.row_gap * ui.s)
    )
    try:
        with ui.component_scope("section_body"), search_presentation(ui):
            yield
    finally:
        imgui.pop_style_var(3)
        imgui.pop_style_color()
        imgui.pop_font()


def advance(ui, x, y):
    """Publish the extent of manually positioned controls to ImGui."""
    # ImGui truncates CursorMaxPos when calculating the scrollable content size.
    # Round the published endpoint outward so the final fractional row remains reachable.
    imgui.set_cursor_screen_pos((x, math.ceil(y)))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, 0))
    imgui.dummy((0, 0))
    imgui.pop_style_var()


def property_label(ui, key, text, x, y, width, *, above, row_height=None, parent="form_row"):
    scale = ui.s
    row_height = DESIGN.control_height if row_height is None else row_height
    font_size = ui.metric("form_label", "fontSize", ui.metric(parent, "fontSize", DESIGN.body_size))
    height = font_size * scale
    top = y + (3 * scale if above else max(0, (row_height * scale - height) / 2))
    if ui.design is not None:
        painter = ui.design.primitives
        painter.family = ui.design.family()
        inherit = {}
        for field in ("fontSize", "fontFamily", "textColor", "lineHeightMode", "lineHeight"):
            if ui.design.runtime.edited(parent, field):
                inherit[field] = (ui.design.runtime.resolved_part(parent, field), parent)
        box = painter.label(
            key,
            Rect(x, y, width, row_height * scale),
            text,
            color=MUTED,
            inherit=inherit,
        )
        return box.outer.width, font_size * scale + box.margin[1] + box.margin[3]
    ui.text(x, top, text, MUTED, font_size, max_width=width, translate=False)
    ui.hits[key] = (x, top, width, height)
    return width, height


def property_section(ui, label, flags, *, item_id=None, divider=True):
    """Use one section painter and native persistent open state in every panel."""
    pos = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    storage = imgui.get_state_storage()
    key = imgui.get_id(item_id or "native-section-" + label)
    opened = bool(storage.get_int(key, int(bool(flags & imgui.TreeNodeFlags_.default_open))))
    next_item = imgui.get_current_context().next_item_data
    if next_item.has_flags & imgui.internal.NextItemDataFlags_.has_open and (
        next_item.open_cond == imgui.Cond_.always or storage.get_int(key, -1) == -1
    ):
        opened = next_item.open_val
        storage.set_int(key, int(opened))
    if ui.section(label, pos.x, pos.y, width, opened, divider=divider):
        opened = not opened
        storage.set_int(key, int(opened))
    advance(ui, pos.x, pos.y + ui.section_offset(38 if opened else 34) * ui.s)
    return opened


def toggle_row_height(ui, label_style=None):
    """Measure a switch row from the same label and glyph height used for paint."""
    if ui.design is None:
        return DESIGN.control_height * ui.s
    painter = ui.design.primitives
    painter.family = ui.design.family()
    style = painter.style("switch")
    label_style = label_style or painter.style(("switch", "form_label"))
    margin, label_margin = style.edges("margin"), label_style.edges("margin")
    height = max(
        painter.field_height("switch"),
        label_style.line_height() + label_margin[1] + label_margin[3],
    )
    height = math.ceil(height / 2) * 2
    return (height + margin[1] + margin[3]) * ui.s


@dataclass(frozen=True, slots=True)
class _SwitchWidths:
    caption: float
    preferred_label: float
    gap: float
    field: float
    field_inset: float
    row_inset: float

    @property
    def minimum(self):
        return self.caption + self.gap + self.field + self.field_inset + self.row_inset


def _switch_widths(ui, width, label):
    """Measure the painted caption and complete switch slot in native units."""
    scale = ui.s
    row, switch = ui.component_style("form_row"), ui.component_style("switch")
    label_style = ui.component_style(("switch", "form_label"))
    margins, label_margins = row.edges("margin", scale), label_style.edges("margin", scale)
    switch_margins = switch.edges("margin", scale)
    available = max(0, width - margins[0] - margins[2])
    text_width = ui.measure(
        ui.translate(label),
        label_style.get("fontSize", BODY_SIZE),
        translate=False,
        font_resolved=True,
        style=label_style,
    )
    caption = text_width + label_margins[0] + label_margins[2]
    baseline = min(row.get("labelWidth", DESIGN.label_width) * scale, available * 0.28)
    gap = (
        switch.get(
            "columnGap",
            switch.get("gap", row.get("columnGap", row.get("labelGap", DESIGN.label_gap))),
        )
        * scale
    )
    padding = switch.edges("padding", scale)
    minimum = (switch.get("switchWidth", DESIGN.switch_track_width) + 4) * scale
    minimum += padding[0] + padding[2]
    return _SwitchWidths(
        caption,
        baseline,
        gap,
        minimum,
        switch_margins[0] + switch_margins[2],
        margins[0] + margins[2],
    )


def switch_form_layout(ui, width, label, *, label_width=None, label_above=None):
    """Fit the real caption and switch; shared label columns are preferences."""
    measured = _switch_widths(ui, width, label)
    desired = max(
        measured.caption, measured.preferred_label if label_width is None else label_width
    )
    above = measured.minimum > width if label_above is None else label_above
    layout = ui.form_layout(
        width,
        label_width=desired,
        label_gap=measured.gap,
        label_above=above,
        field_part="switch",
        field_minimum=measured.field,
        minimum=measured.field + measured.field_inset,
    )
    _, field_y, field_width, field_height = layout.fields[0]
    field_width = min(field_width, measured.field + measured.field_inset)
    return replace(
        layout,
        fields=((width - layout.row_margin[2] - field_width, field_y, field_width, field_height),),
    )


def property_toggle(ui, key, label, value, *, width=None, tooltip="", label_width=None):
    """Draw a complete switch row with shared label bounds and row spacing."""
    pos = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x if width is None else width
    layout = switch_form_layout(ui, width, label, label_width=label_width)
    control_x = pos.x + layout.fields[0][0]
    if layout.label_above:
        shown = ui.translate(label)
        property_label(
            ui,
            key + "-label-text",
            shown,
            pos.x + layout.label_x,
            pos.y + layout.label_y,
            layout.label_width,
            above=True,
            row_height=layout.label_height / ui.s,
            parent="switch",
        )
        imgui.set_cursor_screen_pos((pos.x + layout.label_x, pos.y + layout.label_y))
        clicked = imgui.invisible_button(
            key + "-label", (max(1, layout.label_width), layout.label_height)
        )
        if tooltip and imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        ):
            ui.tooltip(tooltip, translate=False)
        value = not value if clicked else value
        if ui.design is not None:
            painter = ui.design.primitives
            painter.family = ui.design.family()
            style = painter.style("switch")
            margin = style.edges("margin", ui.s)
            _, field_y, field_width, field_height = layout.fields[0]
            box = painter.field_box(
                style,
                Rect(control_x, pos.y + field_y, field_width, field_height + margin[1] + margin[3]),
                (0, 0, 0, 0),
            )
            value = painter.switch(
                key,
                box.frame,
                value,
                layout_box=box,
                tooltip=tooltip,
                control_x=box.content.right
                - style.get("switchWidth", DESIGN.switch_track_width) * ui.s,
            ).value
        else:
            value = ui.switch(key, control_x, pos.y + layout.fields[0][1], value)
        advance(ui, pos.x, pos.y + layout.height + DESIGN.row_gap * ui.s)
        return value
    if ui.design is not None:
        painter = ui.design.primitives
        painter.family = ui.design.family()
        style = painter.style("switch")
        label_style = painter.style(("switch", "form_label"))
        margin = style.edges("margin", ui.s)
        row_height = toggle_row_height(ui, label_style)
        height = row_height - margin[1] - margin[3]
        box = painter.field_box(
            style,
            Rect(
                pos.x + layout.row_margin[0],
                pos.y + layout.row_margin[1],
                max(0, width - layout.row_margin[0] - layout.row_margin[2]),
                row_height,
            ),
            (0, 0, 0, 0),
            minimum_height=height / ui.s,
        )
        result = painter.switch(
            key,
            box.frame,
            value,
            label=label,
            label_style=label_style,
            layout_box=box,
            tooltip=tooltip,
            control_x=box.content.right
            - style.get("switchWidth", DESIGN.switch_track_width) * ui.s,
        )
        advance(
            ui,
            pos.x,
            box.outer.bottom + layout.row_margin[3] + DESIGN.row_gap * ui.s,
        )
        return result.value
    shown = ui.translate(label)
    clicked = ui.button(
        key + "-label",
        pos.x,
        pos.y,
        max(1, layout.label_width / ui.s),
        DESIGN.control_height,
        tooltip=tooltip
        or (shown if ui.measure(shown, translate=False) > layout.label_width else ""),
    )
    property_label(
        ui,
        key + "-label-text",
        shown,
        pos.x,
        pos.y,
        layout.label_width,
        above=False,
        parent="switch",
    )
    edited = ui.switch(key, control_x, pos.y, value)
    advance(ui, pos.x, pos.y + (DESIGN.control_height + DESIGN.row_gap) * ui.s)
    return not value if clicked else edited


class NativePropertyPresentation:
    def __init__(self, ui, *, prefix="property", fit_labels=False):
        self.ui = ui
        self.prefix = prefix
        self._colors = {}
        self.color_committed = False
        self.groups = PropertyGroups(ui)
        self._control_width = None
        self.fit_labels = fit_labels
        self._table_labels = {}
        self._label_groups = {}
        self._label_group = None
        self._shared_label_width = None
        self._label_width = None
        self._labels = []
        self._active_rows = {}
        self.choices = ChoicePopup(ui)

    def _field_width(self):
        return imgui.calc_item_width() if self._control_width is None else self._control_width

    def tooltip(self, text, *, hovered=False):
        if hovered or imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
            self.ui.tooltip(text, translate=False)

    def section(self, label, flags=0):
        self._control_width = None
        shown = label.split("##", 1)[0]
        return property_section(self.ui, shown, flags, item_id=self.prefix + "-section-" + label)

    @contextmanager
    def label_column(self, key):
        """Share a measured label column across independent sections of one page."""
        previous = self._label_group, self._shared_label_width
        tables = self._label_groups.setdefault(key, set())
        size = self.ui.metric("form_label", "fontSize", BODY_SIZE)
        self._label_group = tables
        self._shared_label_width = max(
            (
                self.ui.measure(label, size, translate=False)
                for table in tables
                for label in self._table_labels.get(table, ())
            ),
            default=DESIGN.label_width * self.ui.s,
        )
        try:
            yield
        finally:
            self._label_group, self._shared_label_width = previous

    def begin_table(self, table_id):
        self._control_width = None
        self._labels = []
        self._label_width = None
        if self._label_group is not None:
            self._label_group.add(table_id)
        if self.fit_labels and not self.ui.component_style("form_row").explicit("labelWidth"):
            size = self.ui.metric("form_label", "fontSize", BODY_SIZE)
            self._label_width = max(
                (
                    self.ui.measure(label, size, translate=False)
                    for label in self._table_labels.get(table_id, ())
                ),
                default=DESIGN.label_width * self.ui.s,
            )
            if self._shared_label_width is not None:
                self._label_width = self._shared_label_width
        opened = imgui.begin_table(
            table_id,
            1,
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x,
        )
        if opened:
            self.groups.begin(table_id)
        return opened

    def end_table(self):
        if self.fit_labels and self.groups.tables:
            self._table_labels[self.groups.tables[-1].key] = tuple(self._labels)
        self.groups.end()
        self._control_width = None

    def clipped_rows(self, labels):
        """Clip uniform property rows while retaining label width and active edits."""
        labels = tuple(labels)
        if not labels:
            return
        ui = self.ui
        owner = self.groups.tables[-1].key
        if (
            self.fit_labels
            and self._shared_label_width is None
            and not ui.component_style("form_row").explicit("labelWidth")
        ):
            size = ui.metric("form_label", "fontSize", BODY_SIZE)
            self._label_width = max(ui.measure(label, size, translate=False) for label in labels)
        active = self._active_rows.pop(owner, None)

        def remember(index):
            item_id = imgui.get_item_id()
            if item_id and item_id == imgui.get_current_context().active_id:
                self._active_rows[owner] = (index, item_id)

        # Section top padding belongs only to the first row. Measure the uniform
        # remainder separately so an authored section inset never changes its stride.
        yield 0
        remember(0)
        clipper = imgui.ListClipper()
        clipper.begin(len(labels) - 1)
        if active and active[0] > 0 and active[1] == imgui.get_current_context().active_id:
            clipper.include_item_by_index(active[0] - 1)
        try:
            while clipper.step():
                for index in range(clipper.display_start + 1, clipper.display_end + 1):
                    yield index
                    remember(index)
        finally:
            clipper.end()
            self._labels[:] = labels

    def row(
        self,
        label,
        *,
        minimum=96,
        rail=False,
        control_part="input_number",
        label_width=None,
        readonly_text=None,
    ):
        ui, scale = self.ui, self.ui.s
        imgui.table_next_row()
        imgui.table_next_column()
        pos = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        gx, gy, width = self.groups.row(pos.x, pos.y, width)
        pos = imgui.ImVec2(gx, gy)
        self._labels.append(label)
        if label_width is None and self._label_width is not None:
            label_width = max(
                self._label_width,
                ui.measure(label, ui.metric("form_label", "fontSize", BODY_SIZE), translate=False),
            )
        style = rail_style()
        minimum = (
            style.entry_width + style.minimum_track_width + DESIGN.row_gap if rail else minimum
        ) * scale
        field_minimum = label_above = None
        if control_part == "checkbox":
            control = ui.component_style("checkbox")
            control_margin = control.edges("margin", scale)
            field_minimum = control.get("size", DESIGN.checkbox_size) * scale
            minimum = field_minimum + control_margin[0] + control_margin[2]
            row, caption_style = ui.component_style("form_row"), ui.component_style("form_label")
            row_margin = row.edges("margin", scale)
            available = max(0, width - row_margin[0] - row_margin[2])
            caption_margin = caption_style.edges("margin", scale)
            caption = (
                ui.measure(
                    label,
                    caption_style.get("fontSize", BODY_SIZE),
                    translate=False,
                    font_resolved=True,
                    style=caption_style,
                )
                + caption_margin[0]
                + caption_margin[2]
            )
            preferred = min(row.get("labelWidth", DESIGN.label_width) * scale, available * 0.28)
            label_width = max(caption, preferred if label_width is None else label_width)
            gap = row.get("columnGap", row.get("labelGap", DESIGN.label_gap)) * scale
            label_above = caption + gap + minimum > available
        layout = ui.form_layout(
            width,
            1,
            axis=False,
            label_width=label_width,
            label_above=label_above,
            minimum=minimum,
            field_minimum=field_minimum,
            minimum_height=ui.rail_height()
            if rail
            else ui.segment_height()
            if control_part == "segment"
            else None,
            field_part=control_part,
        )
        if readonly_text is not None:
            value_style = ui.component_style("form_label")
            pad, margin = value_style.edges("padding", scale), value_style.edges("margin", scale)
            value_width = max(0, layout.fields[0][2] - pad[0] - pad[2] - margin[0] - margin[2])
            self._readonly_lines = self._wrap_value(readonly_text, value_width, value_style)
            content_height = len(self._readonly_lines) * value_style.line_height() * scale
            layout = ui.form_layout(
                width,
                1,
                axis=False,
                label_width=label_width,
                minimum=minimum,
                minimum_height=content_height + pad[1] + pad[3] + margin[1] + margin[3],
                field_part=control_part,
            )
        key = self.prefix + "-label-" + label
        if ui.design is not None and ui.design.runtime.observes(scoped_parts("form_row")):
            owner = self.groups.tables[-1].key if self.groups.tables else self.prefix
            ui.design.record(
                owner + "/row/" + label,
                ("form_row",),
                [pos.x, pos.y, width, layout.height],
                "form_row",
                layout="top" if layout.label_above else "left",
                minContentHeight=layout.minimum_content_height,
                outer=[pos.x, pos.y, width, layout.height],
                frameRect=[
                    pos.x + layout.row_margin[0],
                    pos.y + layout.row_margin[1],
                    max(0, width - layout.row_margin[0] - layout.row_margin[2]),
                    max(0, layout.height - layout.row_margin[1] - layout.row_margin[3]),
                ],
                fields=[
                    [
                        pos.x + layout.label_x,
                        pos.y + layout.label_y,
                        layout.label_width,
                        layout.label_height,
                    ],
                    *[[pos.x + dx, pos.y + dy, fw, fh] for dx, dy, fw, fh in layout.fields],
                ],
                labelRect=[
                    pos.x + layout.label_x,
                    pos.y + layout.label_y,
                    layout.label_width,
                    layout.label_height,
                ],
            )
        property_label(
            ui,
            key,
            label,
            pos.x + layout.label_x,
            pos.y + layout.label_y,
            layout.label_width,
            above=layout.label_above,
            row_height=layout.label_height / scale,
        )
        advance(ui, pos.x, pos.y + layout.height)
        imgui.set_cursor_screen_pos((pos.x, pos.y))
        label_height = layout.fields[0][1] if layout.label_above else layout.label_height
        imgui.begin_disabled(layout.label_width <= 0)
        imgui.invisible_button("##" + key, (max(1, layout.label_width), max(1, label_height)))
        imgui.end_disabled()
        if (
            imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            and ui.measure(label, BODY_SIZE, translate=False) > layout.label_width
        ):
            ui.tooltip(label, translate=False)
        dx, dy, fw, fh = layout.fields[0]
        self._field_rect = Rect(pos.x + dx, pos.y + dy, fw, fh)
        imgui.set_cursor_screen_pos((pos.x + dx, pos.y + dy))
        self._control_width = max(0, width - dx - layout.row_margin[2])
        imgui.set_next_item_width(max(1, self._control_width))

    def value_rail(self, item_id, value, bounds, **options):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.rail(item_id, pos.x, pos.y, self._field_width(), value, bounds, **options)

    def segmented(self, item_id, labels, selected, *, icons=None, theme=None):
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

    def combo(self, key, index, labels, flags=0, *, preview=None):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.combo(
            key, pos.x, pos.y, self._field_width(), index, labels, flags=flags, preview=preview
        )

    def begin_combo(self, key, preview, flags=0, *, popup_width=0):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.begin_combo(
            key, pos.x, pos.y, self._field_width(), preview, flags=flags, popup_width=popup_width
        )

    @contextmanager
    def clearable_combo(self, key, preview, *, choices=(), **options):
        """Keep the clear action ahead of the shared select field's hit target."""
        from mojive.ui.workspace.popup_style import menu_choice_width

        ui = self.ui
        pos, width = imgui.get_cursor_screen_pos(), self._field_width()
        style = ui.component_style("select")
        margin = style.edges("margin", ui.s)
        height = (
            ui.design.primitives.field_height("select", DESIGN.control_height) * ui.s
            if ui.design is not None
            else imgui.get_frame_height()
        )
        rect = Rect(pos.x, pos.y, width, height + margin[1] + margin[3])
        if ui.design is not None:
            frame = ui.design.primitives.field_box(style, rect).frame
        else:
            frame = rect.inset(*margin)
        popup_width = menu_choice_width(ui, choices or (preview,), -1)

        def begin(shown):
            return ui.begin_combo(key, pos.x, pos.y, width, shown, popup_width=popup_width)

        imgui.push_font(ui.resolve_font(style), style.get("fontSize", BODY_SIZE) * ui.s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (style.edges("padding", ui.s)[0], 0))
        try:
            with clearable_combo(
                key, preview, begin=begin, frame_rect=frame.tuple(), **options
            ) as result:
                if result[0]:
                    self.choices.begin(key)
                try:
                    yield result
                finally:
                    if result[0]:
                        self.choices.end(close_combo=False)
        finally:
            imgui.pop_style_var()
            imgui.pop_font()

    def choice(self, label, selected=False, *, enabled=True):
        return self.choices.choice(label, selected, enabled=enabled)

    def choice_filter(self, key, value, *, hint, clear_tooltip):
        return self.choices.filter(key, value, hint=hint, clear_tooltip=clear_tooltip)

    def input_text(self, key, value, flags=0):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.input_text(key, pos.x, pos.y, self._field_width(), value, flags=flags)

    def button(self, label, size=None):
        pos = imgui.get_cursor_screen_pos()
        width = self._field_width() if size is None else size[0]
        height = DESIGN.control_height if size is None or not size[1] else size[1] / self.ui.s
        margin = self.ui.component_style("button").edges("margin", self.ui.s)
        limit = max(0, imgui.get_content_region_avail().x - margin[0] - margin[2])
        return self.ui.button(
            "property-button/" + label,
            pos.x,
            pos.y,
            width / self.ui.s,
            height,
            label=label,
            width_limit=limit,
            filled=True,
        )

    def numeric(self, item_id, value, speed, low, high, fmt, *, integer=False):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.input(
            item_id,
            pos.x,
            pos.y,
            self._field_width(),
            value,
            speed=speed,
            lo=low,
            hi=high,
            format=fmt,
            integer=integer,
        )

    def numbers(self, item_id, values, speed, low, high, fmt):
        pos = imgui.get_cursor_screen_pos()
        layout = self.ui.numeric_layout(
            self._field_width(),
            len(values),
            axis=False,
            minimum=self.ui.numeric_minimum(axis=False, format=fmt),
        )
        changed, out = False, list(values)
        imgui.begin_group()
        for index, value in enumerate(values):
            dx, dy, width, _ = layout.fields[index]
            edited, out[index] = self.ui.input(
                f"{item_id}-{index}",
                pos.x + dx,
                pos.y + dy,
                width,
                value,
                speed=speed,
                lo=low,
                hi=high,
                format=fmt,
            )
            changed |= edited
        advance(self.ui, pos.x, pos.y + layout.height)
        imgui.end_group()
        self.ui.record_numeric_group(
            item_id + "/controls", pos.x, pos.y, self._field_width(), layout
        )
        return changed, out

    def color(self, item_id, values):
        from mojive.ui.workspace.color_editor import ColorField

        field = self._colors.get(item_id)
        if field is None:
            if len(self._colors) >= 32:
                self._colors.clear()
            field = self._colors[item_id] = ColorField()
        result = field.draw(self.ui, item_id, values, width=self._field_width())
        self.color_committed = field.committed
        return result

    def toggle(self, key, label, value, *, width=None):
        if width is None and self._control_width is not None:
            width = self._control_width
        return property_toggle(self.ui, key, label, value, width=width)

    def toggles(self, items):
        """Fit complete switch/label pairs into equal columns, then wrap rows."""
        pos = imgui.get_cursor_screen_pos()
        width, scale = imgui.get_content_region_avail().x, self.ui.s
        if not items:
            return ()
        measurements = tuple(_switch_widths(self.ui, width, label) for _, label, _ in items)
        shared_label_width = max(item.caption for item in measurements)
        label_gap = measurements[0].gap
        minimum = max(item.minimum for item in measurements) + label_gap
        columns = max(1, min(len(items), int(width / minimum)))
        column_width = width / columns
        row_height = max(
            toggle_row_height(self.ui),
            *(
                switch_form_layout(
                    self.ui, column_width - label_gap, label, label_width=shared_label_width
                ).height
                for _, label, _ in items
            ),
        )
        row_gap = self.ui.metric("section_body", "rowGap", DESIGN.row_gap) * scale
        result = []
        imgui.begin_group()
        for index, (key, label, value) in enumerate(items):
            imgui.set_cursor_screen_pos(
                (
                    pos.x + index % columns * column_width,
                    pos.y + index // columns * (row_height + row_gap),
                )
            )
            result.append(
                property_toggle(
                    self.ui,
                    key,
                    label,
                    value,
                    width=column_width - label_gap,
                    label_width=shared_label_width,
                )
            )
        imgui.end_group()
        rows = (len(items) + columns - 1) // columns
        advance(self.ui, pos.x, pos.y + rows * (row_height + row_gap))
        return result

    def plot_height(self, count):
        available = imgui.get_content_region_avail().y
        return min(
            90 * self.ui.s,
            max(
                50 * self.ui.s,
                (available - DESIGN.row_gap * self.ui.s * max(0, count - 1)) / max(1, count),
            ),
        )

    @contextmanager
    def numeric_text_style(self, size_scale=1.0):
        """Keep numeric readouts proportional to their owning group's caption size."""
        imgui.push_font(self.ui.numeric_font, imgui.get_font_size() * size_scale)
        try:
            yield
        finally:
            imgui.pop_font()

    def multi_select(self, key, labels, values):
        """Use the shared choice rows for compact visibility controls."""
        from mojive.ui.workspace.popup_style import menu_choice_width

        selected = list(values)
        width = menu_choice_width(self.ui, labels, -1)
        if self.begin_combo(key, self.ui.translate("Channels"), popup_width=width):
            self.choices.begin(key)
            try:
                for index, label in enumerate(labels):
                    changed, value = self.choices.choice(label, selected[index])
                    if changed:
                        selected[index] = value
            finally:
                self.choices.end()
        return tuple(selected)

    def action_menu(self, key, label, actions):
        """Present enabled actions in the shared measured menu without a selected value."""
        from mojive.ui.workspace.popup_style import menu_choice_width

        chosen = None
        width = menu_choice_width(self.ui, tuple(label for label, _ in actions), -1)
        imgui.begin_disabled(not any(enabled for _, enabled in actions))
        opened = self.begin_combo(key, label, popup_width=width)
        if opened:
            self.choices.begin(key)
            try:
                for index, (caption, enabled) in enumerate(actions):
                    if self.choices.choice(caption, enabled=enabled)[0]:
                        chosen = index
            finally:
                self.choices.end()
        imgui.end_disabled()
        return chosen

    def choice_height(self):
        style = self.ui.component_style("select")
        margin = style.edges("margin", self.ui.s)
        height = (
            self.ui.design.primitives.field_height("select", DESIGN.control_height) * self.ui.s
            if self.ui.design is not None
            else imgui.get_frame_height()
        )
        return height + margin[1] + margin[3]

    def labeled(self, label, value):
        self.readonly_value(label, value, muted=False)

    def _wrap_value(self, value, width, style):
        """Use the active body font's native word boundaries for property values."""
        size = style.get("fontSize", BODY_SIZE) * self.ui.s
        lines = []
        for paragraph in str(value).split("\n"):
            if not paragraph:
                lines.append("")
                continue
            while paragraph:
                boundary = self.ui.body.calc_word_wrap_position_python(
                    size, paragraph, max(1, width)
                )
                # The binding returns a UTF-8 byte offset from ImGui's native string.
                count = max(1, len(paragraph.encode("utf-8")[:boundary].decode("utf-8", "ignore")))
                lines.append(paragraph[:count].rstrip())
                paragraph = paragraph[count:].lstrip()
        return tuple(lines)

    def readonly_value(self, label, value, *, muted=True):
        """Measure wrapped text before allocating its row and share the label baseline."""
        self.row(label, control_part="form_label", readonly_text=str(value))
        ui, scale = self.ui, self.ui.s
        style = ui.component_style("form_label")
        content = self._field_rect.inset(*style.edges("margin", scale)).inset(
            *style.edges("padding", scale)
        )
        font_size = style.get("fontSize", BODY_SIZE)
        line_height = style.line_height() * scale
        text_height = line_height * len(self._readonly_lines)
        top = content.y + max(0, (content.height - text_height) / 2)
        observe = ui.design is not None and ui.design.runtime.observes(scoped_parts("form_label"))
        lines = [] if observe else None
        ui.draw.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
        try:
            for index, text in enumerate(self._readonly_lines):
                y = top + index * line_height
                ui.text(
                    content.x,
                    y + (line_height - font_size * scale) / 2,
                    text,
                    MUTED if muted else TEXT,
                    font_size,
                    translate=False,
                    font_resolved=True,
                )
                if observe:
                    lines.append(
                        (
                            content.x,
                            y,
                            ui.measure(text, font_size, translate=False, font_resolved=True),
                            line_height,
                        )
                    )
        finally:
            ui.draw.pop_clip_rect()
        if observe:
            ui.design.record(
                self.prefix + "-value-" + label,
                ("form_label",),
                self._field_rect.tuple(),
                "readonly_value",
                content=content.tuple(),
                text=str(value),
                textRects=lines,
                fontSize=font_size,
                lineHeight=line_height / scale,
                textAlign="left",
            )
        advance(ui, self._field_rect.x, self._field_rect.bottom)

    def checkbox(self, key, value, theme=None):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.checkbox(
            key, pos.x, pos.y, value, label=key.split("#", 1)[0], width=self._field_width()
        )
