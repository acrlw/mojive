"""Native numeric editing, responsive field layout and value rails."""

from collections import OrderedDict
from dataclasses import dataclass, replace

from imgui_bundle import imgui

from mojive.ui.compound_fields import (
    borderless_numeric_input,
    draw_focus_frame,
    draw_joined_field_frame,
)
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.theme import THEME, mix_color

from .design import BODY_SIZE, CONTROL_RADIUS, DESIGN, MONO_SIZE
from .widget_style import (
    FORM_LABEL_COLUMN_GAP,
    FORM_LABEL_COLUMN_MINIMUM,
    FORM_LABEL_COLUMN_WIDTH,
    MUTED,
    TEXT,
)


@dataclass(frozen=True, slots=True)
class NumericEditState:
    """Native item signals for the caller's command transaction."""

    activated: bool = False
    active: bool = False
    deactivated: bool = False
    committed: bool = False
    canceled: bool = False
    reset: bool = False


@dataclass(frozen=True, slots=True)
class NumericGroupLayout:
    """Shared field rectangles and complete group height in screen coordinates."""

    fields: tuple[tuple[float, float, float, float], ...]
    columns: int
    height: float
    label_above: bool = False
    label_width: float = 0.0


class _NumericFields:
    """Numeric implementation of UI; the UI instance owns gesture and hit state."""

    def input(
        self,
        key,
        x,
        y,
        w,
        value,
        *,
        axis=None,
        speed=0.01,
        lo=0.0,
        hi=0.0,
        format="%.3f",
        reset_value=0.0,
    ):
        """Draw a native drag field; callers own command transactions and cancellation."""
        s = self.s
        badge = min(18 * s, max(0, w * 0.35)) if axis else 0
        draw_list = self.draw
        splitter = None
        reset = False
        badge_active = badge_hovered = badge_activated = badge_deactivated = False
        badge_id = None
        imgui.push_font(self.mono, MONO_SIZE * s)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding, (8 * s, DESIGN.control_padding_y(MONO_SIZE) * s)
        )
        if axis:
            splitter = imgui.ImDrawListSplitter()
            splitter.split(draw_list, 2)
            splitter.set_current_channel(draw_list, 1)
            imgui.set_cursor_screen_pos((x, y))
            for slot in (
                imgui.Col_.button,
                imgui.Col_.button_hovered,
                imgui.Col_.button_active,
                imgui.Col_.nav_cursor,
            ):
                imgui.push_style_color(slot, (0, 0, 0, 0))
            reset = imgui.button("##" + key + "-reset", (badge, imgui.get_frame_height()))
            badge_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            badge_active = imgui.is_item_active()
            badge_activated = imgui.is_item_activated()
            badge_deactivated = imgui.is_item_deactivated()
            badge_lo, badge_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            badge_id = imgui.get_item_id()
            disabled = bool(imgui.get_item_flags() & imgui.ItemFlags_.disabled)
            imgui.pop_style_color(4)
            for slot in (
                imgui.Col_.frame_bg,
                imgui.Col_.frame_bg_hovered,
                imgui.Col_.frame_bg_active,
            ):
                imgui.push_style_color(slot, (0, 0, 0, 0))
        imgui.set_cursor_screen_pos((x + badge, y))
        imgui.set_next_item_width(max(s, w - badge))
        # This painter owns all visible text. A bare ID would add an ImGui label
        # after the field and enlarge both its item bounds and popup content.
        item_label = key if key.startswith("##") else "##" + key
        item_id = imgui.get_id(item_label)
        context, io = imgui.get_current_context(), imgui.get_io()
        hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
            (x + badge, y), (x + w, y + imgui.get_frame_height())
        )
        # Preserve native text on its activation frame. DragFloat still owns
        # activation and editing; these requests only select its text visibility.
        native_text = not bool(context.current_item_flags & imgui.ItemFlags_.disabled) and (
            imgui.internal.temp_input_is_active(item_id)
            or (
                hovered
                and (
                    (imgui.is_mouse_clicked(0) and io.key_ctrl)
                    or imgui.is_mouse_double_clicked(0)
                    or (
                        io.config_drag_click_to_input_text
                        and context.active_id == item_id
                        and imgui.is_mouse_released(0)
                    )
                )
            )
            or (
                context.nav_activate_id == item_id
                and bool(context.nav_activate_flags & imgui.internal.ActivateFlags_.prefer_input)
            )
        )
        if not native_text:
            imgui.push_style_color(imgui.Col_.text, (0, 0, 0, 0))
        draw_list.push_clip_rect((x + badge, y), (x + w, y + imgui.get_frame_height()), True)
        with borderless_numeric_input():
            changed, edited = imgui.drag_float(
                item_label,
                float(value),
                speed,
                lo,
                hi,
                format,
                imgui.SliderFlags_.always_clamp if lo < hi else 0,
            )
        draw_list.pop_clip_rect()
        if not native_text:
            imgui.pop_style_color()
        field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        item_id = imgui.get_item_id()
        text_editing = imgui.internal.temp_input_is_active(item_id)
        field_active = imgui.is_item_active()
        field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        active = field_active or badge_active
        deactivated = imgui.is_item_deactivated() or badge_deactivated
        state = NumericEditState(
            activated=imgui.is_item_activated() or badge_activated,
            active=active,
            deactivated=deactivated,
            committed=imgui.is_item_deactivated_after_edit() or reset,
            canceled=(active or deactivated) and imgui.is_key_pressed(imgui.Key.escape, False),
            reset=reset,
        )
        self.numeric_edits[key] = state
        if axis:
            imgui.pop_style_color(3)
            hovered = imgui.is_item_hovered()
            axis_index = {"X": 0, "R": 0, "Y": 1, "G": 1, "Z": 2, "B": 2}.get(axis)
            axis_color = (
                THEME.axis_color(axis_index) if axis_index is not None else THEME.text_disabled
            )
            field_color = (
                THEME.bg_frame_active
                if field_active
                else THEME.bg_frame_hovered
                if hovered
                else THEME.bg_frame
            )
            splitter.set_current_channel(draw_list, 0)
            draw_joined_field_frame(
                draw_list,
                badge_lo,
                badge_hi,
                field_lo,
                field_hi,
                badge_color=mix_color(
                    THEME.bg_frame_active
                    if badge_active
                    else THEME.bg_frame_hovered
                    if badge_hovered and not disabled
                    else THEME.bg_frame,
                    axis_color,
                    0.88 if badge_active else 0.72 if badge_hovered and not disabled else 0.56,
                ),
                field_color=field_color,
                rounding=CONTROL_RADIUS * s,
                badge_opacity=imgui.get_style().alpha,
                field_opacity=imgui.get_style().alpha,
            )
            splitter.set_current_channel(draw_list, 1)
            draw_list.push_clip_rect((x, y), (x + badge, field_hi.y), True)
            self.text(x + (badge - self.measure(axis, 12)) / 2, y + 8 * s, axis, TEXT, 12)
            draw_list.pop_clip_rect()
            draw_focus_frame(
                badge_lo,
                badge_hi,
                rounding=CONTROL_RADIUS * s,
                corners=imgui.ImDrawFlags_.round_corners_left,
                item_id=badge_id,
            )
            self.hits[key + "-reset"] = (
                badge_lo.x,
                badge_lo.y,
                badge_hi.x - badge_lo.x,
                badge_hi.y - badge_lo.y,
            )
        current = reset_value if reset else edited if changed else value
        truncated = False
        if not native_text and not text_editing:
            shown_value = format % current
            fixed_number = format in ("%f", "%F") or (
                format.startswith("%.") and format.endswith(("f", "F")) and format[2:-1].isdigit()
            )
            if fixed_number and "." in shown_value:
                shown_value = shown_value.rstrip("0").rstrip(".")
            padding = min(8 * s, max(0, (field_hi.x - field_lo.x) / 2))
            text_left, text_right = field_lo.x + padding, field_hi.x - padding
            content_width = max(0, text_right - text_left)
            text_width = imgui.calc_text_size(self.translate(shown_value)).x
            truncated = text_width > content_width
            text_x = text_left + max(0, (content_width - text_width) / 2)
            draw_list.push_clip_rect((text_left, field_lo.y), (text_right, field_hi.y), True)
            self.text(
                text_x,
                field_lo.y + imgui.get_style().frame_padding.y,
                shown_value,
                TEXT,
                MONO_SIZE,
                mono=True,
                max_width=content_width,
                native_line=True,
            )
            draw_list.pop_clip_rect()
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + w, field_hi.y),
            rounding=CONTROL_RADIUS * s,
            item_id=item_id,
        )
        if splitter is not None:
            splitter.merge(draw_list)
        imgui.pop_style_var()
        imgui.pop_font()
        self.hits[key] = (
            field_lo.x,
            field_lo.y,
            field_hi.x - field_lo.x,
            field_hi.y - field_lo.y,
        )
        self.hits[key + "-frame"] = (x, y, w, field_hi.y - y)
        if field_hovered and truncated:
            self.tooltip(str(float(current)))
        if axis and badge_hovered:
            self.tooltip(
                self.translate("Read only")
                if disabled
                else f"{self.translate('Reset')}: {reset_value:g}"
            )
        return (True, reset_value) if reset else (changed, edited if changed else value)

    def numeric_minimum(self, *, axis=True, format=None, values=()):
        """Reserve the sign, common precision, padding and optional axis badge."""
        s = self.s
        samples = (
            tuple(format % value for value in (0.123456789, 123.456789, -0.123456789, *values))
            if format
            else ("0.0041472", "0.00288568", "-0.000123")
        )
        font_key = (id(self.mono), s, imgui.get_io().display_framebuffer_scale.x, samples)
        measured = getattr(self, "_numeric_measurements", None)
        if measured is None:
            measured = self._numeric_measurements = OrderedDict()
        text_width = measured.get(font_key)
        if text_width is None:
            imgui.push_font(self.mono, MONO_SIZE * s)
            text_width = max(imgui.calc_text_size(value).x for value in samples)
            imgui.pop_font()
            if len(measured) >= 64:
                measured.popitem(last=False)
            measured[font_key] = text_width
        else:
            measured.move_to_end(font_key)
        return text_width + (34 if axis else 16) * s

    def numeric_layout(self, width, count, *, axis=True, column_options=None, minimum=None):
        """Keep common precision readable before placing equal-width numeric fields."""
        if not count:
            return NumericGroupLayout((), 0, 0)
        s = self.s
        minimum = self.numeric_minimum(axis=axis) if minimum is None else minimum
        gap = DESIGN.numeric_gap * s
        field_height = DESIGN.control_height * s
        row_stride = (DESIGN.control_height + DESIGN.row_gap) * s
        if column_options is None:
            column_options = (count, 2, 1) if count == 4 else (count, 1)
        column_options = tuple(min(count, columns) for columns in column_options)
        columns = next(
            (
                columns
                for columns in column_options
                if width >= columns * minimum + (columns - 1) * gap
            ),
            1,
        )
        field_width = (width - (columns - 1) * gap) / columns
        fields = tuple(
            (
                (index % columns) * (field_width + gap),
                (index // columns) * row_stride,
                field_width,
                field_height,
            )
            for index in range(count)
        )
        height = fields[-1][1] + field_height
        return NumericGroupLayout(fields, columns, height)

    def form_layout(
        self,
        width,
        count=1,
        *,
        axis=False,
        minimum=0,
        label_above=None,
        label_width=None,
        field_minimum=None,
    ):
        """Give controls their minimum width before reserving a label column."""
        s = self.s
        if label_width is None:
            label_width = min(FORM_LABEL_COLUMN_WIDTH * s, width * 0.28)
        field_minimum = self.numeric_minimum(axis=axis) if field_minimum is None else field_minimum
        required = count * max(minimum, field_minimum) + (count - 1) * DESIGN.numeric_gap * s
        label_width = min(label_width, max(0, width - required - FORM_LABEL_COLUMN_GAP * s))
        above = label_width < FORM_LABEL_COLUMN_MINIMUM * s if label_above is None else label_above
        heading_width = width if above else label_width
        field_x, field_y = (0, 24 * s) if above else (heading_width + FORM_LABEL_COLUMN_GAP * s, 0)
        layout = self.numeric_layout(width - field_x, count, axis=axis, minimum=field_minimum)
        return replace(
            layout,
            fields=tuple((x + field_x, y + field_y, w, h) for x, y, w, h in layout.fields),
            height=layout.height + field_y,
            label_above=above,
            label_width=heading_width,
        )

    def vector_layout(
        self,
        width,
        *,
        label_above=None,
        label_width=None,
        format=None,
        minimum=None,
        values=(),
        key=None,
    ):
        """Share the control-first label column and XYZ grid across each group."""
        cache = self.__dict__.setdefault("_vector_layouts", {})
        geometry = (width, self.s, id(self.mono), label_above, label_width, format)
        previous = cache.get(key) if key is not None else None
        # The active drag owns stable field rectangles until its gesture finishes.
        if previous is not None and previous[0] == geometry and imgui.is_any_item_active():
            return previous[1]
        layout = self.form_layout(
            width,
            3,
            axis=True,
            label_above=label_above,
            label_width=label_width,
            field_minimum=self.numeric_minimum(format=format, values=values)
            if minimum is None
            else minimum,
        )
        if key is not None:
            if len(cache) >= 128:
                cache.clear()
            cache[key] = (geometry, layout)
        return layout

    def vector(
        self,
        key,
        x,
        y,
        w,
        label,
        values,
        unit="",
        *,
        speed=0.01,
        lo=0.0,
        hi=0.0,
        format="%.3f",
        reset_values=(0.0, 0.0, 0.0),
        layout=None,
    ):
        s = self.s
        heading = self.translate(label) + (f" ({unit})" if unit else "")
        layout = (
            self.vector_layout(w, format=format, values=values, key=key)
            if layout is None
            else layout
        )
        self.text(
            x,
            y + (3 if layout.label_above else DESIGN.control_padding_y(BODY_SIZE)) * s,
            heading,
            MUTED,
            max_width=layout.label_width,
        )
        self.hits[key + "-label"] = (x, y, layout.label_width, 24 * s)
        if (
            self.measure(heading) > layout.label_width
            and imgui.is_window_hovered()
            and imgui.is_mouse_hovering_rect((x, y), (x + layout.label_width, y + 24 * s))
        ):
            self.tooltip(heading, translate=False)
        changed = False
        out = list(values)
        for n, a in enumerate("XYZ"):
            field_x, field_y, field_width, _ = layout.fields[n]
            edit, out[n] = self.input(
                key + a,
                x + field_x,
                y + field_y,
                field_width,
                out[n],
                axis=a,
                speed=speed,
                lo=lo,
                hi=hi,
                format=format,
                reset_value=reset_values[n],
            )
            changed |= edit
        states = [self.numeric_edits[key + axis] for axis in "XYZ"]
        self.numeric_edits[key] = NumericEditState(
            activated=any(state.activated for state in states),
            active=any(state.active for state in states),
            deactivated=any(state.deactivated for state in states),
            committed=any(state.committed for state in states),
            canceled=any(state.canceled for state in states),
            reset=any(state.reset for state in states),
        )
        return changed, out

    def rail(self, key, x, y, w, value, bounds, **options):
        """One native slider, numeric entry and optional attached unit control."""
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_font(self.mono, MONO_SIZE * s)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (DESIGN.frame_padding_x * s, DESIGN.control_padding_y(MONO_SIZE) * s),
        )
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (DESIGN.row_gap * s, DESIGN.row_gap * s))
        for part in ("track", "value", "unit"):
            self.hits.pop(key + "-" + part, None)
        item_id = key if key.startswith("##") else "##" + key
        imgui.begin_group()
        imgui.set_next_item_width(max(s, w))
        try:
            edit = value_rail(
                self.value_context,
                item_id,
                value,
                bounds,
                hit_bounds=self.hits,
                unit_font=self.body,
                **options,
            )
        finally:
            imgui.end_group()
            imgui.pop_style_var(2)
            imgui.pop_font()
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        if item_id != key:
            for part in ("track", "value", "unit"):
                bounds = self.hits.get(item_id + "-" + part)
                if bounds is not None:
                    self.hits[key + "-" + part] = bounds
        self.hits[key] = (a.x, a.y, b.x - a.x, b.y - a.y)
        return edit

    def slider(self, key, x, y, w, value, lo, hi, format="%.2f"):
        edit = self.rail(
            key,
            x,
            y,
            w,
            value,
            (lo, hi) if lo < hi else None,
            initial=None,
            fmt=format,
            show_reset=False,
        )
        return edit.changed, edit.value
