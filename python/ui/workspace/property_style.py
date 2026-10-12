"""Shared native property headings and label geometry."""

from contextlib import contextmanager

from imgui_bundle import imgui

from .design import BODY_SIZE, DESIGN, RAIL_STYLE
from .widgets import MUTED, SWITCH_WIDTH


@contextmanager
def property_controls(ui):
    """Match field height, font and table row spacing across native panels."""
    imgui.push_font(ui.body, BODY_SIZE * ui.s)
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding,
        (DESIGN.frame_padding_x * ui.s, DESIGN.control_padding_y(BODY_SIZE) * ui.s),
    )
    # Tables separate rows through cell padding, rather than item_spacing.
    imgui.push_style_var(imgui.StyleVar_.cell_padding, (0, DESIGN.row_gap * ui.s / 2))
    imgui.push_style_var(
        imgui.StyleVar_.item_spacing, (DESIGN.item_spacing_x * ui.s, DESIGN.row_gap * ui.s)
    )
    try:
        yield
    finally:
        imgui.pop_style_var(3)
        imgui.pop_font()


def advance(ui, x, y):
    """Publish the extent of manually positioned controls to ImGui."""
    imgui.set_cursor_screen_pos((x, y))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, 0))
    imgui.dummy((0, 0))
    imgui.pop_style_var()


def property_label(ui, key, text, x, y, width, *, above, row_height=DESIGN.control_height):
    scale = ui.s
    height = BODY_SIZE * scale
    top = y + (3 * scale if above else max(0, (row_height * scale - height) / 2))
    ui.text(x, top, text, MUTED, BODY_SIZE, max_width=width, translate=False)
    ui.hits[key] = (x, top, width, height)
    return width, height


def property_section(ui, label, flags, *, item_id=None):
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
    if ui.section(label, pos.x, pos.y, width, opened):
        opened = not opened
        storage.set_int(key, int(opened))
    advance(ui, pos.x, pos.y + (38 if opened else 34) * ui.s)
    return opened


class NativePropertyPresentation:
    def __init__(self, ui, *, prefix="property"):
        self.ui = ui
        self.prefix = prefix
        self._colors = {}
        self.color_committed = False

    def section(self, label, flags=0):
        shown = label.split("##", 1)[0]
        return property_section(self.ui, shown, flags, item_id=self.prefix + "-section-" + label)

    def begin_table(self, table_id):
        return imgui.begin_table(
            table_id,
            1,
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x,
        )

    def row(self, label, *, minimum=96, rail=False):
        ui, scale = self.ui, self.ui.s
        imgui.table_next_row()
        imgui.table_next_column()
        pos = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        minimum = (
            RAIL_STYLE.entry_width
            + RAIL_STYLE.unit_width
            + RAIL_STYLE.minimum_track_width
            + DESIGN.row_gap
            if rail
            else minimum
        ) * scale
        layout = ui.form_layout(width, 1, axis=False, minimum=minimum)
        key = self.prefix + "-label-" + label
        property_label(ui, key, label, pos.x, pos.y, layout.label_width, above=layout.label_above)
        imgui.set_cursor_screen_pos((pos.x, pos.y))
        label_height = 24 * scale if layout.label_above else DESIGN.control_height * scale
        imgui.invisible_button("##" + key, (layout.label_width, label_height))
        if (
            imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            and ui.measure(label, BODY_SIZE, translate=False) > layout.label_width
        ):
            ui.tooltip(label, translate=False)
        dx, dy, _, _ = layout.fields[0]
        imgui.set_cursor_screen_pos((pos.x + dx, pos.y + dy))
        imgui.set_next_item_width(max(scale, width - dx))

    def value_rail(self, item_id, value, bounds, **options):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.rail(
            item_id, pos.x, pos.y, imgui.calc_item_width(), value, bounds, **options
        )

    def segmented(self, item_id, labels, selected, *, icons=None):
        pos = imgui.get_cursor_screen_pos()
        if icons is not None:
            names = {"persp": "perspective", "ortho": "orthographic"}
            icons = tuple(names[name] for name in icons)
        return self.ui.segmented(
            item_id,
            pos.x,
            pos.y,
            imgui.calc_item_width() / self.ui.s,
            labels,
            selected,
            icons=icons,
            layout="content",
        )

    def numeric(self, item_id, value, speed, low, high, fmt):
        pos = imgui.get_cursor_screen_pos()
        return self.ui.input(
            item_id,
            pos.x,
            pos.y,
            imgui.calc_item_width(),
            value,
            speed=speed,
            lo=low,
            hi=high,
            format=fmt,
        )

    def numbers(self, item_id, values, speed, low, high, fmt):
        pos = imgui.get_cursor_screen_pos()
        layout = self.ui.numeric_layout(
            imgui.calc_item_width(),
            len(values),
            axis=False,
            minimum=self.ui.numeric_minimum(axis=False, format=fmt, values=values),
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
        imgui.end_group()
        return changed, out

    def color(self, item_id, values):
        from .color_editor import ColorField

        field = self._colors.get(item_id)
        if field is None:
            if len(self._colors) >= 32:
                self._colors.clear()
            field = self._colors[item_id] = ColorField()
        result = field.draw(self.ui, item_id, values)
        self.color_committed = field.committed
        return result

    def toggle(self, key, label, value, *, width=None):
        pos = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x if width is None else width
        clicked = self.ui.button(
            key + "-label",
            pos.x,
            pos.y,
            max(1, width / self.ui.s - SWITCH_WIDTH - DESIGN.label_gap),
            label=label,
            align="left",
            text_color=MUTED,
        )
        edited = self.ui.switch(key, pos.x + width - SWITCH_WIDTH * self.ui.s, pos.y, value)
        advance(self.ui, pos.x, pos.y + (DESIGN.control_height + DESIGN.row_gap) * self.ui.s)
        return not value if clicked else edited

    def toggles(self, items):
        """Fit complete switch/label pairs into equal columns, then wrap rows."""
        pos = imgui.get_cursor_screen_pos()
        width, scale = imgui.get_content_region_avail().x, self.ui.s
        minimum = max(self.ui.measure(label, translate=False) for _, label, _ in items)
        minimum += (SWITCH_WIDTH + 2 * DESIGN.label_gap) * scale
        columns = max(1, min(len(items), int(width / minimum)))
        column_width = width / columns
        result = []
        imgui.begin_group()
        for index, (key, label, value) in enumerate(items):
            imgui.set_cursor_screen_pos(
                (
                    pos.x + index % columns * column_width,
                    pos.y + index // columns * (DESIGN.control_height + DESIGN.row_gap) * scale,
                )
            )
            result.append(
                self.toggle(key, label, value, width=column_width - DESIGN.label_gap * scale)
            )
        imgui.end_group()
        rows = (len(items) + columns - 1) // columns
        advance(self.ui, pos.x, pos.y + rows * (DESIGN.control_height + DESIGN.row_gap) * scale)
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

    def labeled(self, label, value):
        self.row(label)
        imgui.text_wrapped(str(value))

    def checkbox(self, key, value):
        pos = imgui.get_cursor_screen_pos()
        result = self.ui.switch(key, pos.x, pos.y, value)
        return result != value, result
