"""Native numeric editing, responsive field layout and value rails."""

import math
from collections import OrderedDict
from dataclasses import dataclass, replace

from imgui_bundle import imgui

from mojive.design.native.geometry import NativePartStyle, Rect, budget_margins, part_scope
from mojive.ui.compound_fields import (
    borderless_numeric_input,
    draw_focus_frame,
    draw_joined_field_frame,
)
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.theme import THEME, mix_color
from mojive.ui.workspace.design import BODY_SIZE, CONTROL_RADIUS, DESIGN, MONO_SIZE, rail_style
from mojive.ui.workspace.widget_style import (
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
    label_x: float = 0.0
    label_y: float = 0.0
    label_height: float = 0.0
    label_line_height: float = 0.0
    minimum_content_height: float = 0.0
    width: float = 0.0
    group_margin: tuple[float, float, float, float] = (0, 0, 0, 0)
    row_margin: tuple[float, float, float, float] = (0, 0, 0, 0)


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
        integer=False,
        flags=0,
    ):
        """Draw a native drag field; callers own command transactions and cancellation."""
        s = self.s
        if self.design is not None:
            result = self.design.number(
                key,
                x,
                y,
                w,
                value,
                axis=axis,
                speed=speed,
                lo=lo,
                hi=hi,
                format=format,
                reset_value=reset_value,
                integer=integer,
                flags=flags,
            )
            return result.changed, result.value
        badge = min(18 * s, max(0, w * 0.35)) if axis else 0
        draw_list = self.draw
        splitter = None
        reset = False
        badge_active = badge_hovered = badge_activated = badge_deactivated = False
        badge_id = None
        imgui.push_font(self.numeric_font, (self.typography.numeric.size or DESIGN.mono_size) * s)
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
            drag = imgui.drag_int if integer else imgui.drag_float
            numeric = int if integer else float
            changed, edited = drag(
                item_label,
                numeric(value),
                speed,
                numeric(lo),
                numeric(hi),
                format,
                flags | (imgui.SliderFlags_.always_clamp if lo < hi else 0),
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
        style = self.component_style(("input_number", "axis_input") if axis else "input_number")
        font_size = style.get("fontSize", DESIGN.mono_size)
        font_key = (
            id(self.numeric_font),
            font_size,
            s,
            imgui.get_io().display_framebuffer_scale.x,
            samples,
        )
        measured = getattr(self, "_numeric_measurements", None)
        if measured is None:
            measured = self._numeric_measurements = OrderedDict()
        text_width = measured.get(font_key)
        if text_width is None:
            imgui.push_font(self.numeric_font, font_size * s)
            text_width = max(imgui.calc_text_size(value).x for value in samples)
            imgui.pop_font()
            if len(measured) >= 64:
                measured.popitem(last=False)
            measured[font_key] = text_width
        else:
            measured.move_to_end(font_key)
        padding = style.edges(
            "padding", s, (3, 0, 3, 0) if axis and self.design is not None else (8, 0, 8, 0)
        )
        return (
            text_width + padding[0] + padding[2] + (style.get("axisWidth", 18) * s if axis else 0)
        )

    def color_channel_minimum(self):
        """Keep normalized color channels compact while preserving axis and padding."""
        style = self.component_style(("input_number", "axis_input"))
        font_size = style.get("fontSize", DESIGN.mono_size)
        imgui.push_font(self.numeric_font, font_size * self.s)
        try:
            text_width = imgui.calc_text_size("0.5").x
        finally:
            imgui.pop_font()
        pad = style.edges("padding", self.s, (3, 0, 3, 0))
        return text_width + pad[0] + pad[2] + style.get("axisWidth", 18) * self.s

    def numeric_layout(
        self,
        width,
        count,
        *,
        axis=True,
        column_options=None,
        minimum=None,
        parts="vector_group",
        minimum_height=None,
        field_part=None,
    ):
        """Budget group margins and every child frame before placing equal columns."""
        if not count:
            return NumericGroupLayout((), 0, 0)
        s = self.s
        automatic_minimum = minimum is None
        minimum = self.numeric_minimum(axis=axis) if automatic_minimum else minimum
        if axis and automatic_minimum:
            # Axis rows keep equal compact columns. Full precision remains
            # available in native editing and the clipped-value tooltip.
            minimum = min(minimum, self.numeric_minimum(axis=True, format="%.1f"))
        runtime = self.design.runtime if self.design is not None else None
        group = NativePartStyle(runtime, parts, authored_defaults=True) if runtime else None
        field = (
            NativePartStyle(
                runtime,
                field_part or (("input_number", "axis_input") if axis else "input_number"),
                authored_defaults=True,
            )
            if runtime
            else None
        )
        margin = (
            budget_margins(group.edges("margin", s), width, math.inf) if group else (0, 0, 0, 0)
        )
        field_margin = field.edges("margin", s) if field else (0, 0, 0, 0)
        content_width = max(0, width - margin[0] - margin[2])
        gap = group.get("columnGap", DESIGN.numeric_gap) * s if group else DESIGN.numeric_gap * s
        row_gap = group.get("rowGap", DESIGN.row_gap) * s if group else DESIGN.row_gap * s
        field_height = (
            self.design.primitives.field_height(field.part_ids, DESIGN.control_height) * s
            if field
            else DESIGN.control_height * s
        )
        if minimum_height is not None:
            field_height = max(field_height, minimum_height - field_margin[1] - field_margin[3])
        field_minimum = minimum + field_margin[0] + field_margin[2]
        if field and field.get("widthMode", "auto") == "fixed":
            field_minimum = field.get("width", minimum / s) * s + field_margin[0] + field_margin[2]
        if column_options is None:
            column_options = (count, 2, 1) if count == 4 else (count, 1)
        requested = group.get("columns", count) if group else count
        if requested != count:
            column_options = (min(count, max(1, int(requested))), 1)
        column_options = tuple(min(count, columns) for columns in column_options)
        columns = next(
            (
                columns
                for columns in column_options
                if content_width >= columns * field_minimum + (columns - 1) * gap
            ),
            1,
        )
        cell_width = max(0, (content_width - (columns - 1) * gap) / columns)
        field_width = cell_width
        row_height = field_height + field_margin[1] + field_margin[3]
        fields = tuple(
            (
                margin[0] + (index % columns) * (cell_width + gap),
                margin[1] + (index // columns) * (row_height + row_gap),
                field_width,
                field_height,
            )
            for index in range(count)
        )
        height = fields[-1][1] + row_height + margin[3]
        return NumericGroupLayout(fields, columns, height, width=width, group_margin=margin)

    def record_numeric_group(self, key, x, y, width, layout, *, parts="vector_group"):
        """Expose the same complete group bounds used by rows and color popups."""
        self.hits[key] = (x, y, width, layout.height)
        if self.design is not None:
            self.design.record(
                key,
                (parts,),
                self.hits[key],
                "numeric_group",
                columns=layout.columns,
                fields=[[x + dx, y + dy, w, h] for dx, dy, w, h in layout.fields],
            )

    def record_form_group(self, key, x, y, layout):
        if not layout.fields:
            return
        margin = layout.group_margin
        field = self.component_style(("input_number", "axis_input"))
        field_margin = field.edges("margin", self.s)
        left = min(item[0] for item in layout.fields) - margin[0]
        top = min(item[1] for item in layout.fields) - margin[1]
        width = max(item[0] + item[2] for item in layout.fields) + margin[2] - left
        height = (
            max(item[1] + item[3] for item in layout.fields)
            + field_margin[1]
            + field_margin[3]
            + margin[3]
            - top
        )
        relative = replace(
            layout,
            fields=tuple((dx - left, dy - top, w, h) for dx, dy, w, h in layout.fields),
            height=height,
        )
        self.record_numeric_group(key + "/controls", x + left, y + top, width, relative)
        if self.design is not None:
            row_margin = layout.row_margin
            right = (
                max(
                    layout.label_x + layout.label_width,
                    max(dx + field_width for dx, _, field_width, _ in layout.fields),
                )
                + row_margin[2]
            )
            self.design.record(
                key + "/row",
                ("form_row", "vector_row"),
                [x, y, right, layout.height],
                "vector_row",
                layout="top" if layout.label_above else "left",
                minContentHeight=layout.minimum_content_height,
                fields=[
                    [
                        x + layout.label_x,
                        y + layout.label_y,
                        layout.label_width,
                        layout.label_height,
                    ],
                    *[[x + dx, y + dy, fw, fh] for dx, dy, fw, fh in layout.fields],
                ],
                labelRect=[
                    x + layout.label_x,
                    y + layout.label_y,
                    layout.label_width,
                    layout.label_height,
                ],
            )

    def form_layout(
        self,
        width,
        count=1,
        *,
        axis=False,
        minimum=0,
        label_above=None,
        label_width=None,
        label_gap=None,
        field_minimum=None,
        vector=False,
        minimum_height=None,
        field_part=None,
    ):
        """Reserve one label column and a measured child grid inside row margins."""
        s = self.s
        runtime = self.design.runtime if self.design is not None else None
        row = (
            NativePartStyle(
                runtime,
                ("form_row", "vector_row") if vector else "form_row",
                authored_defaults=True,
            )
            if runtime
            else None
        )
        margins = budget_margins(row.edges("margin", s), width, math.inf) if row else (0, 0, 0, 0)
        available = max(0, width - margins[0] - margins[2])
        layout_mode = (
            row.get("layout", "row") if vector and row and row.explicit("layout") else None
        )
        force_left = layout_mode in ("left", "row", "inline")
        if label_width is None:
            label_width = (
                row.get("labelWidth", DESIGN.label_width) if row else DESIGN.label_width
            ) * s
            if not (force_left and row.explicit("labelWidth")):
                label_width = min(label_width, available * 0.28)
        if label_gap is None:
            label_gap = (
                row.get("columnGap", row.get("labelGap", DESIGN.label_gap))
                if row
                else DESIGN.label_gap
            ) * s
        field_minimum = self.numeric_minimum(axis=axis) if field_minimum is None else field_minimum
        # A left label shares the row with at least one field. The numeric grid
        # owns XYZ reflow; reserving every axis before that pass erases the label.
        required_count = 1 if force_left else count
        required = (
            required_count * max(minimum, field_minimum)
            + (required_count - 1) * DESIGN.numeric_gap * s
        )
        label_width = min(label_width, max(0, available - required - label_gap))
        above = label_width < DESIGN.label_minimum * s if label_above is None else label_above
        if layout_mode is not None:
            above = layout_mode in ("top", "column", "stack") or label_width < 24 * s
        heading_width = available if above else label_width
        row_height = (
            row.get("height", DESIGN.control_height) if row else DESIGN.control_height
        ) * s
        label_line = (
            row.line_height(DESIGN.body_size * 1.4) * s if row else DESIGN.body_size * 1.4 * s
        )
        if runtime is not None:
            label_style = NativePartStyle(runtime, "form_label", authored_defaults=True)
            label_margin = label_style.edges("margin", s)
            label_line = max(
                label_line, label_style.line_height() * s + label_margin[1] + label_margin[3]
            )
        row_height = max(row_height, label_line)
        label_height = max(24 * s, label_line)
        if above and row and row.explicit("rowGap"):
            label_height = (
                row.line_height(DESIGN.body_size * 1.4) * s
                + row.get("rowGap", DESIGN.label_gap) * s
            )
        field_x, field_y = (
            (0, label_height) if above else (min(available, heading_width + label_gap), 0)
        )
        layout = self.numeric_layout(
            available - field_x,
            count,
            axis=axis,
            minimum=field_minimum,
            minimum_height=minimum_height,
            field_part=field_part,
            parts=() if field_part == "switch" else "vector_group",
        )
        valign = row.get("alignItems", "center") if row else "center"
        cross = max(row_height, layout.height)
        align_y = max(0, cross - layout.height) * (
            0.5 if valign == "center" else 1 if valign == "end" else 0
        )
        return replace(
            layout,
            fields=tuple(
                (x + field_x + margins[0], y + field_y + margins[1] + align_y, w, h)
                for x, y, w, h in layout.fields
            ),
            height=cross + field_y + margins[1] + margins[3],
            label_above=above,
            label_width=heading_width,
            label_x=margins[0],
            label_y=margins[1],
            label_height=label_height if above else cross,
            label_line_height=label_line,
            minimum_content_height=max(label_line, layout.height),
            width=width,
            row_margin=margins,
        )

    @part_scope("vector_row")
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
        geometry = (
            width,
            self.s,
            id(self.numeric_font),
            label_above,
            label_width,
            format,
            self.design.runtime.revision if self.design else 0,
        )
        previous = cache.get(key) if key is not None else None
        # The active drag owns stable field rectangles until its gesture finishes.
        if previous is not None and previous[0] == geometry and imgui.is_any_item_active():
            return previous[1]
        layout = self.form_layout(
            width,
            3,
            axis=True,
            vector=True,
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

    @part_scope("vector_row")
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
        rect = Rect(x + layout.label_x, y + layout.label_y, layout.label_width, layout.label_height)
        if self.design is not None:
            with self.component_scope("vector_row"):
                self.design.primitives.family = self.design.family()
                self.design.primitives.label(key + "-label", rect, label, unit=unit, color=MUTED)
        else:
            self.text(
                rect.x,
                rect.y + max(0, (rect.height - BODY_SIZE * s) / 2),
                heading,
                MUTED,
                max_width=layout.label_width,
                translate=False,
            )
        self.hits[key + "-label"] = rect.tuple()
        if (
            self.measure(heading) > layout.label_width
            and imgui.is_window_hovered()
            and imgui.is_mouse_hovering_rect((rect.x, rect.y), (rect.right, rect.bottom))
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
        self.record_form_group(key, x, y, layout)
        return changed, out

    def rail_height(self, *, unit=True):
        if self.design is None:
            return DESIGN.control_height * self.s
        painter = self.design.primitives
        painter.family = self.design.family()
        return max(painter.shell_height(unit=unit), self.metric("range", "height", 28)) * self.s

    def rail_extent(self, width, *, bounded=True, unit=True, show_reset=False):
        """Measure the same inline/stacked budget used by value_rail."""
        scale = self.s
        style = self.component_style("range_number")
        margin = style.edges("margin", scale)
        inner = max(scale, width - margin[0] - margin[2])
        gap = style.get("columnGap", style.get("gap", DESIGN.row_gap)) * scale
        entry = self.rail_height(unit=unit)
        track = (
            max(
                self.metric("input_number", "fontSize", DESIGN.mono_size),
                self.metric("range", "height", DESIGN.control_height),
            )
            * scale
        )
        rail = rail_style()
        reset_space = entry + gap if show_reset else 0
        number_width = min(rail.entry_width * scale, max(scale, inner - reset_space))
        stacked = (
            bounded and inner < number_width + reset_space + gap + rail.minimum_track_width * scale
        )
        height = (
            track + DESIGN.row_gap * scale + entry
            if stacked
            else max(track, entry)
            if bounded
            else entry
        )
        return height + margin[1] + margin[3]

    def rail(self, key, x, y, w, value, bounds, **options):
        """One native slider, numeric entry and optional attached unit control."""
        s = self.s
        outer_x, outer_y, outer_width = x, y, w
        wrapper = None
        if self.design is not None:
            wrapper = self.design.box("range_number", (x, y, w, DESIGN.control_height * s))
            x, y = wrapper.content.x, wrapper.content.y
            w = max(s, w - wrapper.margin[0] - wrapper.margin[2])
        imgui.set_cursor_screen_pos((x, y))
        self.value_context.rail_style = rail_style()
        font_size = self.metric("input_number", "fontSize", DESIGN.mono_size)
        height = (
            self.rail_height(unit=bool(options.get("unit")) and options.get("show_unit", True)) / s
        )
        imgui.push_font(self.numeric_font, font_size * s)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (
                self.metric("input_number", "paddingX", DESIGN.frame_padding_x) * s,
                max(0, (height - font_size) / 2) * s,
            ),
        )
        gap = self.metric("range_number", "columnGap", DESIGN.row_gap)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (gap * s, DESIGN.row_gap * s))
        entry_painter = None
        if self.design is not None:
            from mojive.ui.panels import value_edit

            def entry_painter(
                ctx, label, value, bounds, initial, fmt, width, unit, toggle, hits, _font, _frame
            ):
                pos = imgui.get_cursor_screen_pos()
                painter = self.design.primitives
                painter.family = self.design.family()
                low, high = bounds if bounds is not None else (0.0, 0.0)
                result = painter.number_shell(
                    label + "-value",
                    Rect(pos.x, pos.y, width, height * s),
                    value,
                    unit,
                    toggle_unit=toggle,
                    format=fmt,
                    lo=low,
                    hi=high,
                    reset_value=initial if initial is not None else 0.0,
                )
                hits[label + "-value"] = result.box.frame.tuple()
                if unit:
                    hits[label + "-unit"] = self.hits[label + "-value-unit"]
                edit = value_edit(
                    label,
                    value,
                    result.changed,
                    result.value,
                    initial=initial,
                    fmt=fmt,
                    bindings=ctx.input_bindings,
                )
                edit.activated |= result.activated
                return edit

        for part in ("track", "value", "unit"):
            self.hits.pop(key + "-" + part, None)
        item_id = key if key.startswith("##") else "##" + key
        imgui.set_cursor_screen_pos((outer_x, outer_y))
        imgui.begin_group()
        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(max(s, w))
        try:
            edit = value_rail(
                self.value_context,
                item_id,
                value,
                bounds,
                hit_bounds=self.hits,
                unit_font=self.body,
                entry_painter=entry_painter,
                track_height=self.metric("range", "height", 28) * s,
                align_items=self.metric("range_number", "alignItems", "center"),
                frame_draw=lambda lo, hi, fill: self.rect(
                    lo.x,
                    lo.y,
                    hi.x - lo.x,
                    hi.y - lo.y,
                    tuple(round(channel * 255) for channel in fill),
                    r=DESIGN.control_radius,
                ),
                **options,
            )
            if wrapper is not None and any(wrapper.margin):
                bottom = max(imgui.get_item_rect_max().y, y + height * s) + wrapper.margin[3]
                imgui.set_cursor_screen_pos((outer_x, outer_y))
                imgui.dummy((outer_width, max(s, bottom - outer_y)))
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
        if self.design is not None:
            self.design.record(
                key,
                ("range_number",),
                self.hits[key],
                "range_number",
                fields=[
                    self.hits[key + "-" + suffix]
                    for suffix in ("track", "value")
                    if key + "-" + suffix in self.hits
                ],
            )
            for suffix, parts in (("track", ("range",)), ("value", ("input_number",))):
                if key + "-" + suffix in self.hits:
                    self.design.record(
                        key + "-" + suffix, parts, self.hits[key + "-" + suffix], suffix
                    )
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
