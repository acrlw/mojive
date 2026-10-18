"""Color entry and clipboard formats shared by native Inspector popups."""

from __future__ import annotations

import math
import re
from contextlib import contextmanager

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, independent_corner_scope, part_scope
from mojive.ui.workspace.component_layout import control_frame_size, control_size, flow_layout
from mojive.ui.workspace.corner_geometry import nested_corners
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.menu_geometry import native_edges, native_epsilon, native_unit
from mojive.ui.workspace.panel_geometry import corner_inset


@independent_corner_scope()
def color_popup_geometry(ui):
    style = ui.component_style("color_editor")
    padding = native_edges(style, "padding", ui.s, (8, 8, 8, 8))
    heights = [28 * ui.s]
    for key, label, icon in (("color-popup-done", "Done", None), ("color-popup-copy", "", "copy")):
        size = control_size(ui, key, 80 if label else 28, 28, label=label, icon=icon)
        heights.append(control_frame_size(ui, key, *size, label=label, icon=icon)[1] * ui.s)
    if ui.design is not None:
        heights.append(ui.design.primitives.field_height(("input_number", "axis_input")) * ui.s)
    limit = max(0, min(heights) / 2 - 1)
    inner = min(ui.component_style("button").get("radius", DESIGN.control_radius) * ui.s, limit)
    corners = nested_corners(style, ui.s, padding, inner)
    # Preserve compact controls and an authored outer curve by growing the
    # actual uniform inset before native painters would clamp a child radius.
    padding = (corner_inset(corners.radius, max(padding), limit),) * 4
    return nested_corners(style, ui.s, padding, inner)


@contextmanager
def color_popup(ui, key):
    """Let the color editor own popup insets once, including the footer's bottom edge."""
    style = ui.component_style("color_editor")
    corners = color_popup_geometry(ui)
    padding, inner_radius, radius = corners.padding, corners.inner_radius / ui.s, corners.radius
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.push_style_var(imgui.StyleVar_.popup_rounding, radius)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, corners.inner_radius)
    opened = imgui.begin_popup(key)
    try:
        if opened and ui.design is not None:
            window = imgui.internal.get_current_window()
            ui.design.record(
                key + "-popup-container",
                ("color_editor",),
                (window.pos.x, window.pos.y, window.size.x, window.size.y),
                "color_popup",
                radius=window.window_rounding / ui.s,
                padding=list(padding),
                innerRadius=inner_radius,
                radiusDerived=not corners.authored_radius,
                radiusAuthored=style.explicit("radius"),
                cornerOffsetDegenerate=radius < padding[0],
                nativeEpsilon=native_epsilon(window.pos.x, window.pos.y, *window.size),
            )
        with ui.corner_scope(
            corners.inner_radius, ("button", "icon_button", "input_number", "axis_input")
        ):
            yield opened
    finally:
        if opened:
            last = imgui.get_item_rect_max()
            # The footer's visible endpoint already lies on the native grid. Publish
            # its matching inset without ItemSize truncating fractional item spacing.
            imgui.set_cursor_screen_pos(
                (native_unit(last.x + padding[2]), native_unit(last.y + padding[3]))
            )
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
            imgui.dummy((0, 0))
            imgui.pop_style_var()
            imgui.end_popup()
        imgui.pop_style_var(3)


def place_color_popup(ui, key, *, position=None):
    """Keep format-dependent popup growth inside the current viewport."""
    popup_id = imgui.get_id(key)
    window = next(
        (
            item.window
            for item in imgui.get_current_context().open_popup_stack
            if item.popup_id == popup_id
        ),
        None,
    )
    if window is None:
        return
    # ImGui preserves an open popup's position when auto-fit changes its height.
    # Measure the next size before Begin so its clipping and controls move together.
    size = imgui.internal.calc_window_next_auto_fit_size(window)
    x, y = (window.pos.x, window.pos.y) if position is None else position
    viewport, inset = imgui.get_main_viewport(), 8 * ui.s
    left, top = viewport.work_pos.x + inset, viewport.work_pos.y + inset
    right = viewport.work_pos.x + viewport.work_size.x - inset
    bottom = viewport.work_pos.y + viewport.work_size.y - inset
    location = (max(left, min(x, right - size.x)), max(top, min(y, bottom - size.y)))
    if location != (x, y):
        imgui.set_next_window_pos(location)


@part_scope("color_editor")
def color_popup_width(ui, count):
    """Fit a complete bounded channel row inside a compact native popup."""
    style, scale = ui.component_style("color_editor"), ui.s
    available = imgui.get_main_viewport().work_size.x - 16 * scale
    requested = style.get("popupWidth", 280) * scale
    if not style.explicit("popupWidth"):
        with ui.component_scope("rgba"):
            group = ui.component_style("rgba")
            field = ui.component_style(("input_number", "axis_input"))
            margin = group.edges("margin", scale)
            field_margin = field.edges("margin", scale)
            minimum = max(
                ui.color_channel_minimum(format="%.3f"),
                ui.color_channel_minimum(format="%.0f"),
            )
            columns = count if group.get("columns", count) >= count else 1
            minimum = columns * (minimum + field_margin[0] + field_margin[2])
            minimum += (columns - 1) * group.get("columnGap", DESIGN.numeric_gap) * scale
            minimum += margin[0] + margin[2]
        padding, margin = (
            color_popup_geometry(ui).padding,
            native_edges(style, "margin", scale),
        )
        minimum += padding[0] + padding[2] + margin[0] + margin[2]
        minimum += 2 * imgui.get_style().window_border_size
        requested = max(requested, minimum)
    return max(scale, min(available, requested))


class ColorField:
    """Compact swatch and the shared native picker; callers apply edited values."""

    def __init__(self):
        self.editor = ColorEditor()
        self.before = ()
        self.open = False
        self.committed = False

    def draw(self, ui, key, values, *, width=None):
        self.committed = False
        pos = imgui.get_cursor_screen_pos()
        width = imgui.calc_item_width() if width is None else width
        rgba = tuple(values) if len(values) == 4 else (*values, 1)
        swatch_width = min(DESIGN.control_height * ui.s, max(0, width))
        ui.draw.push_clip_rect(
            (pos.x, pos.y), (pos.x + max(0, width), pos.y + DESIGN.control_height * ui.s), True
        )
        imgui.begin_disabled(swatch_width <= 0)
        clicked = imgui.color_button(
            key + "-swatch",
            rgba,
            imgui.ColorEditFlags_.no_tooltip,
            (max(1, swatch_width), DESIGN.control_height * ui.s),
        )
        imgui.end_disabled()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits[key + "-swatch"] = (lo.x, lo.y, swatch_width, hi.y - lo.y)
        ui.text(
            hi.x + DESIGN.numeric_gap * ui.s,
            pos.y + DESIGN.control_padding_y(BODY_SIZE) * ui.s,
            format_color(values),
            size=BODY_SIZE,
            max_width=max(0, width - hi.x + pos.x),
            translate=False,
        )
        ui.draw.pop_clip_rect()
        popup = key + "-color"
        if clicked:
            self.before = tuple(values)
            self.editor.begin(values)
            imgui.open_popup(popup)
        changed, out = False, list(values)
        if self.open and not imgui.is_popup_open(popup):
            self.committed = True
            if imgui.is_key_pressed(imgui.Key.escape, False):
                changed, out = tuple(values) != self.before, list(self.before)
        viewport = imgui.get_main_viewport()
        popup_width = color_popup_width(ui, len(values))
        imgui.set_next_window_size_constraints(
            (popup_width, 0), (popup_width, viewport.work_size.y - 16 * ui.s)
        )
        place_color_popup(ui, popup)
        with color_popup(ui, popup) as opened:
            if opened:
                changed, out = self.editor.draw(
                    ui,
                    key,
                    values,
                    imgui.get_content_region_avail().x,
                    footer_height=control_size(
                        ui, key + "-done", 80, DESIGN.control_height, label="Done"
                    )[1],
                )
                footer = imgui.get_cursor_screen_pos()
                gap = ui.metric("color_editor", "columnGap", 6) * ui.s
                half = (self.editor.content_width - gap) / 2
                canceled = imgui.is_key_pressed(imgui.Key.escape, False)
                canceled |= ui.button(
                    key + "-cancel", footer.x, footer.y, half / ui.s, label="Cancel"
                )
                if canceled:
                    changed, out = tuple(values) != self.before, list(self.before)
                    self.committed = True
                    imgui.close_current_popup()
                elif ui.button(
                    key + "-done",
                    footer.x + half + gap,
                    footer.y,
                    half / ui.s,
                    label="Done",
                    active=True,
                    enabled=not self.editor.error,
                ):
                    self.committed = True
                    imgui.close_current_popup()
        self.open = imgui.is_popup_open(popup)
        return changed, out


def format_color(values, mode=0):
    rgba = tuple(values) if len(values) == 4 else (*values, 1.0)
    if mode == 0:
        channels = rgba if rgba[3] < 1 else rgba[:3]
        return "#" + "".join(f"{round(value * 255):02X}" for value in channels)
    if mode == 1:
        return "rgba255(" + ", ".join(str(round(value * 255)) for value in rgba) + ")"
    return "rgba1(" + ", ".join(f"{value:.6g}" for value in rgba) + ")"


def parse_color(text, *, normalized=None):
    """Accept explicit hex or three/four finite, bounded color channels."""
    text = text.strip()
    if text.startswith("#") or re.fullmatch(r"[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", text):
        digits = text.removeprefix("#")
        if len(digits) in (3, 4):
            digits = "".join(character * 2 for character in digits)
        if len(digits) not in (6, 8) or re.fullmatch(r"[0-9a-fA-F]+", digits) is None:
            raise ValueError("Use #RGB, #RGBA, #RRGGBB or #RRGGBBAA.")
        return [int(digits[index : index + 2], 16) / 255 for index in range(0, len(digits), 2)]
    function = re.fullmatch(r"(rgba?(?:255|1)?)\s*\(\s*([0-9eE+.,\s\-]+)\s*\)", text)
    channels = function[2] if function else text
    if re.fullmatch(r"[0-9eE+.,\s\-]+", channels) is None:
        raise ValueError("Paste a hex color or RGB/RGBA values.")
    try:
        values = [float(value) for value in re.split(r"[,\s]+", channels.strip())]
    except ValueError as error:
        raise ValueError("Color channels must be numbers.") from error
    if len(values) not in (3, 4) or not all(math.isfinite(value) for value in values):
        raise ValueError("Use three or four finite color channels.")
    if function:
        if len(values) != (4 if function[1].startswith("rgba") else 3):
            raise ValueError("Use three or four finite color channels.")
        normalized = (
            True if function[1].endswith("1") else False if function[1].endswith("255") else None
        )
    byte_values = normalized is False or (normalized is None and any(value > 1 for value in values))
    limit = 255 if byte_values else 1
    if any(value < 0 or value > limit for value in values):
        raise ValueError(f"Color channels must be between 0 and {limit}.")
    out = [value / limit for value in values]
    if function and function[1] == "rgba" and byte_values and values[3] <= 1:
        out[3] = values[3]
    return out


class ColorEditor:
    def __init__(self):
        self.mode = 0
        self.hex_text = ""
        self.error = ""
        self.defaults = ()
        self.content_origin = (0.0, 0.0)
        self.content_width = 0.0

    def begin(self, values):
        self.defaults = tuple(values)
        self.hex_text = format_color(values)
        self.error = ""

    @part_scope("color_editor")
    def draw(self, ui, key, values, width, *, footer_height=0):
        """Draw editing controls; the caller retains its existing edit transaction."""
        s = ui.s
        out, changed = list(values), False
        if not self.defaults or len(self.defaults) != len(out):
            self.begin(out)
        origin = imgui.get_cursor_screen_pos()
        editor_style = ui.component_style("color_editor")
        margins = native_edges(editor_style, "margin", s)
        padding = color_popup_geometry(ui).padding
        width = max(s, width - margins[0] - margins[2] - padding[0] - padding[2])
        self.content_width = width
        origin = imgui.ImVec2(
            origin.x + margins[0] + padding[0], origin.y + margins[1] + padding[1]
        )
        self.content_origin = (origin.x, origin.y)
        gap = editor_style.get("columnGap", 6) * s
        row_gap = editor_style.get("rowGap", 8) * s
        swatch_size = 28 * s
        captions = (ui.translate("Original"), ui.translate("Current"))
        comparison_sizes = tuple(
            (swatch_size + gap + ui.measure(label, BODY_SIZE, translate=False), 28 * s)
            for label in captions
        )
        action_sizes = tuple(
            control_size(ui, key + "-" + name, 28, 28, icon=name) for name in ("copy", "paste")
        )
        header = flow_layout(
            Rect(origin.x, origin.y, width, 0),
            (
                *comparison_sizes,
                (
                    sum(size[0] for size in action_sizes) + gap,
                    max(size[1] for size in action_sizes),
                ),
            ),
            column_gap=gap,
            row_gap=row_gap,
        )
        boxes = list(header.rectangles[:2])
        actions = header.rectangles[2]
        action_x = actions.x
        if header.height == max(size[1] for size in (*comparison_sizes, *action_sizes)):
            # Keep the comparison at the leading edge and the two actions together.
            action_x = origin.x + width - sum(size[0] for size in action_sizes) - gap
        for size in action_sizes:
            boxes.append(Rect(action_x, actions.y + (actions.height - size[1]) / 2, *size))
            action_x += size[0] + gap
        ui.hits[key + "-header"] = (origin.x, origin.y, width, header.height)
        for index, name in enumerate(("copy", "paste")):
            box = boxes[index + 2]
            frame = control_frame_size(ui, key + "-" + name, box.width, box.height, icon=name)
            pressed = ui.button(
                key + "-" + name,
                box.x,
                box.y,
                *frame,
                icon=name,
                tooltip="Copy" if name == "copy" else "Paste",
            )
            if not pressed:
                continue
            if name == "copy":
                imgui.set_clipboard_text(format_color(out, self.mode))
            else:
                try:
                    pasted = parse_color(
                        imgui.get_clipboard_text() or "",
                        normalized=(self.mode == 2) if self.mode else None,
                    )
                    out[: len(out)] = (pasted + out[3:])[: len(out)]
                    changed, self.error = out != list(values), ""
                    self.hex_text = format_color(out)
                except ValueError as error:
                    self.error = str(error)
        for index, name in enumerate(("original", "current")):
            box = boxes[index]
            imgui.set_cursor_screen_pos((box.x, box.y + (box.height - swatch_size) / 2))
            values_to_show = self.defaults if name == "original" else out
            rgba = tuple(values_to_show)
            if len(rgba) == 3:
                rgba = (*rgba, 1.0)
            pressed = imgui.color_button(
                key + "-" + name,
                rgba,
                imgui.ColorEditFlags_.no_tooltip | imgui.ColorEditFlags_.no_drag_drop,
                (min(swatch_size, box.width), swatch_size),
            )
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            ui.hits[key + "-" + name] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            ui.text(
                hi.x + gap,
                box.y + (box.height - BODY_SIZE * s) / 2,
                captions[index],
                size=BODY_SIZE,
                max_width=max(0, box.right - hi.x - gap),
                translate=False,
            )
            if name == "original":
                if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                    ui.tooltip("Restore original color")
                if pressed:
                    changed, out = list(values) != list(self.defaults), list(self.defaults)
                    self.hex_text, self.error = format_color(out), ""
        picker_y = origin.y + header.height + row_gap
        channel_gap = ui.metric("color_channels", "rowGap", 8) * s
        with ui.component_scope("rgba"):
            channel_layout = ui.numeric_layout(
                width,
                len(out),
                column_options=(len(out), 1),
                parts="rgba",
                minimum=max(
                    ui.color_channel_minimum(format="%.3f"), ui.color_channel_minimum(format="%.0f")
                ),
            )
        number_style = ui.component_style("input_number")
        number_margin = number_style.edges("margin", s)
        number_height = (
            (
                ui.design.primitives.field_height("input_number") * s
                if ui.design is not None
                else DESIGN.control_height * s
            )
            + number_margin[1]
            + number_margin[3]
        )
        format_height = max(
            control_size(
                ui,
                key + "-format-" + str(index),
                width / (3 * s),
                DESIGN.control_height,
                label=label,
            )[1]
            for index, label in enumerate(("Hex", "RGBA 255", "RGBA 1"))
        )
        segment = ui.component_style("segment")
        segment_padding, segment_margin = segment.edges("padding", s), segment.edges("margin", s)
        format_height += (
            segment_padding[1] + segment_padding[3] + segment_margin[1] + segment_margin[3]
        )
        error_height = (
            imgui.calc_text_size(ui.translate(self.error), wrap_width=width).y + channel_gap
            if self.error
            else 0
        )
        reserved = (
            format_height
            + max(number_height, channel_layout.height)
            + footer_height
            + row_gap
            + 2 * channel_gap
            + padding[3]
            + margins[3]
            + error_height
        )
        picker_style = ui.component_style("color_picker")
        picker_padding = picker_style.edges("padding", s)
        picker_margin = picker_style.edges("margin", s)
        font_size = picker_style.get("fontSize", BODY_SIZE) * s
        bar_width = max(font_size, picker_style.get("barWidth", 14) * s)
        bar_gap = picker_style.get("columnGap", 6) * s
        bars_extent = (2 if len(out) == 4 else 1) * (bar_width + bar_gap)
        maximum_sv = max(
            s,
            width
            - picker_padding[0]
            - picker_padding[2]
            - picker_margin[0]
            - picker_margin[2]
            - bars_extent,
        )
        viewport = imgui.get_main_viewport()
        remaining = viewport.work_size.y - 16 * s - (picker_y - origin.y)
        remaining -= padding[1] + margins[1] + reserved
        remaining -= picker_padding[1] + picker_padding[3] + picker_margin[1] + picker_margin[3]
        sv_size = min(maximum_sv, max(picker_style.get("minimumHeight", 160) * s, remaining))
        sv_size = min(sv_size, editor_style.get("pickerHeight", 160) * s)
        if editor_style.explicit("pickerShare"):
            sv_size = min(sv_size, width * editor_style.get("pickerShare", 100) / 100)
        if picker_style.explicit("height"):
            sv_size = min(sv_size, picker_style.get("height", 260) * s)
        picker_width = sv_size + bars_extent + picker_padding[0] + picker_padding[2]
        picker_width += picker_margin[0] + picker_margin[2]
        picker_height = sv_size + picker_padding[1] + picker_padding[3]
        picker_height += picker_margin[1] + picker_margin[3]
        picker_box = ui.component_box(
            "color_picker",
            origin.x + (width - picker_width) / 2,
            picker_y,
            picker_width,
            picker_height,
            key=key + "-picker-container",
        )
        imgui.set_cursor_screen_pos((picker_box.content.x, picker_box.content.y))
        imgui.set_next_item_width(picker_box.content.width)
        flags = (
            imgui.ColorEditFlags_.no_side_preview
            | imgui.ColorEditFlags_.no_small_preview
            | imgui.ColorEditFlags_.no_inputs
            | imgui.ColorEditFlags_.no_options
            | imgui.ColorEditFlags_.no_label
            | imgui.ColorEditFlags_.picker_hue_bar
            | imgui.ColorEditFlags_.alpha_bar
        )
        picker = imgui.color_picker4 if len(out) == 4 else imgui.color_picker3
        imgui.push_font(ui.resolve_font(picker_style), font_size)
        # Native fonts quantize fractional sizes. Match the actual frame height
        # used for each bar so the remaining SV square keeps its measured size.
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding, (0, (bar_width - imgui.get_font_size()) / 2)
        )
        imgui.push_style_var(
            imgui.StyleVar_.item_inner_spacing, (bar_gap, imgui.get_style().item_inner_spacing.y)
        )
        try:
            edited, picked = picker(key + "-picker", out, flags)
        finally:
            imgui.pop_style_var(2)
            imgui.pop_font()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits[key + "-picker"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        ui.hits[key + "-sv"] = (lo.x, lo.y, hi.y - lo.y, hi.y - lo.y)
        picker_bottom = hi.y + picker_box.margin[3] + picker_box.padding[3]
        if edited:
            changed, out, self.error = True, list(picked), ""
            self.hex_text = format_color(out)
        imgui.set_cursor_screen_pos((origin.x, picker_bottom + row_gap))
        p = imgui.get_cursor_screen_pos()
        channels_name = "RGBA" if len(out) == 4 else "RGB"
        mode = ui.segmented(
            key + "-format",
            origin.x,
            p.y,
            width / s,
            ("Hex", channels_name + " 255", channels_name + " 1"),
            self.mode,
        )
        if mode != self.mode:
            self.mode, self.error = mode, ""
            self.hex_text = format_color(out)
        _, format_y, _, format_height = ui.hits[key + "-format"]
        channel_gap = ui.metric("color_channels", "rowGap", 8) * s
        y = format_y + format_height + channel_gap
        if self.mode == 0:
            submitted, self.hex_text = ui.input_text(
                key + "-hex",
                origin.x,
                y,
                width,
                self.hex_text,
                flags=imgui.InputTextFlags_.enter_returns_true
                | imgui.InputTextFlags_.auto_select_all,
                mono=True,
            )
            submitted |= imgui.is_item_deactivated_after_edit()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            ui.hits[key + "-hex"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            active = imgui.is_item_active()
            if submitted:
                try:
                    parsed = parse_color(self.hex_text)
                    out[: len(out)] = (parsed + out[3:])[: len(out)]
                    changed, self.error = out != list(values), ""
                except ValueError as error:
                    self.error = str(error)
            elif not active and not self.error:
                self.hex_text = format_color(out)
            bottom = hi.y
        else:
            with ui.component_scope("rgba"):
                factor = 255 if self.mode == 1 else 1
                number_format = "%.0f" if factor == 255 else "%.3f"
                layout = ui.numeric_layout(
                    width,
                    len(out),
                    column_options=(len(out), 1),
                    parts="rgba",
                    minimum=ui.color_channel_minimum(format="%.0f" if factor == 255 else "%.3f"),
                )
                for index, axis in enumerate("RGBA"[: len(out)]):
                    dx, dy, field_width, _ = layout.fields[index]
                    edited, value = ui.input(
                        key + axis,
                        origin.x + dx,
                        y + dy,
                        field_width,
                        out[index] * factor,
                        axis=axis,
                        speed=1 if factor == 255 else 0.005,
                        lo=0,
                        hi=factor,
                        format=number_format,
                        reset_value=self.defaults[index] * factor,
                    )
                    if edited and math.isfinite(value):
                        out[index], changed, self.error = value / factor, True, ""
                bottom = y + layout.height
                ui.record_numeric_group(key + "-channels", origin.x, y, width, layout, parts="rgba")
            if ui.design is not None:
                ui.design.record(
                    key + "-channel-list",
                    ("color_channels",),
                    (origin.x, format_y + format_height, width, channel_gap + layout.height),
                    "color_channels",
                    rowGap=channel_gap / s,
                    fieldRegions={
                        name: [origin.x, format_y + format_height, width, channel_gap]
                        for name in ("gap", "rowGap")
                    },
                )
        imgui.set_cursor_screen_pos((origin.x, bottom + channel_gap))
        if self.error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + width)
            imgui.text_colored((0.94, 0.49, 0.49, 1), ui.translate(self.error))
            imgui.pop_text_wrap_pos()
        bottom = max(imgui.get_cursor_screen_pos().y, imgui.get_item_rect_max().y) + margins[3]
        if ui.design is not None:
            ui.design.record(
                key + "-editor",
                ("color_editor",),
                (
                    origin.x - margins[0] - padding[0],
                    origin.y - margins[1] - padding[1],
                    width + margins[0] + margins[2] + padding[0] + padding[2],
                    bottom - origin.y + margins[1] + padding[1] + padding[3],
                ),
                "color_editor",
                fieldRegions={
                    "columnGap": [boxes[2].right, boxes[2].y, gap, boxes[2].height],
                    "rowGap": [
                        [origin.x, picker_y - row_gap, width, row_gap],
                        [origin.x, picker_bottom, width, row_gap],
                    ],
                    "gap": [
                        [boxes[2].right, boxes[2].y, gap, boxes[2].height],
                        [origin.x, picker_y - row_gap, width, row_gap],
                    ],
                },
                rowGap=row_gap / s,
                columnGap=gap / s,
                padding=list(padding),
                content=[origin.x, origin.y, width, bottom - origin.y],
                pickerWidth=picker_width / s,
                pickerHeight=ui.hits[key + "-sv"][3] / s,
                svRect=list(ui.hits[key + "-sv"]),
                barWidth=bar_width / s,
                originalColor=list(self.defaults),
                currentColor=list(out),
                originalColorRect=list(ui.hits[key + "-original"]),
                currentColorRect=list(ui.hits[key + "-current"]),
            )
        # Keep the footer's authored height, but align its allocated row's trailing
        # edge to the native popup grid. Rounding only the popup would leave the
        # fractional remainder in its bottom inset.
        footer_y = math.ceil(bottom + footer_height) - footer_height
        imgui.set_cursor_screen_pos((origin.x, footer_y))
        return changed, out
