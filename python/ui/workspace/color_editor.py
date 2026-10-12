"""Color entry and clipboard formats shared by native Inspector popups."""

from __future__ import annotations

import math
import re

from imgui_bundle import imgui

from .design import BODY_SIZE, DESIGN, MONO_SIZE
from .widgets import DIM


class ColorField:
    """Compact swatch and the shared native picker; callers apply edited values."""

    def __init__(self):
        self.editor = ColorEditor()
        self.before = ()
        self.open = False
        self.committed = False

    def draw(self, ui, key, values):
        self.committed = False
        pos = imgui.get_cursor_screen_pos()
        width = imgui.calc_item_width()
        rgba = tuple(values) if len(values) == 4 else (*values, 1)
        clicked = imgui.color_button(
            key + "-swatch",
            rgba,
            imgui.ColorEditFlags_.no_tooltip,
            (DESIGN.control_height * ui.s, DESIGN.control_height * ui.s),
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits[key + "-swatch"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        ui.text(
            hi.x + DESIGN.numeric_gap * ui.s,
            pos.y + DESIGN.control_padding_y(BODY_SIZE) * ui.s,
            format_color(values),
            size=BODY_SIZE,
            max_width=max(0, width - hi.x + pos.x),
            translate=False,
        )
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
        popup_width = min(320 * ui.s, viewport.work_size.x - 2 * DESIGN.panel_padding_x * ui.s)
        imgui.set_next_window_size_constraints(
            (popup_width, 0), (popup_width, viewport.work_size.y - 16 * ui.s)
        )
        if imgui.begin_popup(popup):
            changed, out = self.editor.draw(ui, key, values, imgui.get_content_region_avail().x)
            footer = imgui.get_cursor_screen_pos()
            half = (imgui.get_content_region_avail().x - DESIGN.item_spacing_x * ui.s) / 2
            canceled = imgui.is_key_pressed(imgui.Key.escape, False)
            canceled |= ui.button(key + "-cancel", footer.x, footer.y, half / ui.s, label="Cancel")
            if canceled:
                changed, out = tuple(values) != self.before, list(self.before)
                self.committed = True
                imgui.close_current_popup()
            elif ui.button(
                key + "-done",
                footer.x + half + DESIGN.item_spacing_x * ui.s,
                footer.y,
                half / ui.s,
                label="Done",
                active=True,
                enabled=not self.editor.error,
            ):
                self.committed = True
                imgui.close_current_popup()
            imgui.end_popup()
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

    def begin(self, values):
        self.defaults = tuple(values)
        self.hex_text = format_color(values)
        self.error = ""

    def draw(self, ui, key, values, width):
        """Draw editing controls; the caller retains its existing edit transaction."""
        s = ui.s
        out, changed = list(values), False
        if not self.defaults or len(self.defaults) != len(out):
            self.begin(out)
        origin = imgui.get_cursor_screen_pos()
        gap = 6 * s
        half = (width - gap) / 2
        if ui.button(key + "-copy", origin.x, origin.y, half / s, icon="copy", label="Copy"):
            imgui.set_clipboard_text(format_color(out, self.mode))
        if ui.button(
            key + "-paste", origin.x + half + gap, origin.y, half / s, icon="paste", label="Paste"
        ):
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
        imgui.set_cursor_screen_pos((origin.x, origin.y + 36 * s))
        # Keep room for explicit entry and actions on short, high-DPI viewports.
        picker_width = min(width, max(110 * s, imgui.get_main_viewport().work_size.y - 254 * s))
        imgui.set_cursor_screen_pos((origin.x + (width - picker_width) / 2, origin.y + 36 * s))
        imgui.set_next_item_width(picker_width)
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
        edited, picked = picker(key + "-picker", out, flags)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        ui.hits[key + "-picker"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        if edited:
            changed, out, self.error = True, list(picked), ""
            self.hex_text = format_color(out)
        p = imgui.get_cursor_screen_pos()
        mode = ui.segmented(
            key + "-format", origin.x, p.y, width / s, ("Hex", "RGBA 255", "RGBA 1"), self.mode
        )
        if mode != self.mode:
            self.mode, self.error = mode, ""
            self.hex_text = format_color(out)
        _, format_y, _, format_height = ui.hits[key + "-format"]
        y = format_y + format_height + 8 * s
        if self.mode == 0:
            imgui.set_cursor_screen_pos((origin.x, y))
            imgui.set_next_item_width(width)
            imgui.push_font(ui.mono, MONO_SIZE * s)
            imgui.push_style_var(
                imgui.StyleVar_.frame_padding, (8 * s, DESIGN.control_padding_y(MONO_SIZE) * s)
            )
            submitted, self.hex_text = imgui.input_text(
                "##" + key + "-hex",
                self.hex_text,
                imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
            )
            submitted |= imgui.is_item_deactivated_after_edit()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            ui.hits[key + "-hex"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            active = imgui.is_item_active()
            imgui.pop_style_var()
            imgui.pop_font()
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
            layout = ui.numeric_layout(width, len(out), column_options=(len(out), 2, 1))
            factor = 255 if self.mode == 1 else 1
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
                    format="%.0f" if factor == 255 else "%.6g",
                    reset_value=self.defaults[index] * factor,
                )
                if edited and math.isfinite(value):
                    out[index], changed, self.error = value / factor, True, ""
            bottom = y + layout.height
        imgui.set_cursor_screen_pos((origin.x, bottom + 8 * s))
        if self.error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + width)
            imgui.text_colored((0.94, 0.49, 0.49, 1), ui.translate(self.error))
            imgui.pop_text_wrap_pos()
        else:
            ui.text(
                origin.x,
                bottom + 8 * s,
                "0–255" if self.mode == 1 else "0–1" if self.mode == 2 else "#RRGGBB / #RRGGBBAA",
                DIM,
                11,
                translate=False,
                max_width=width,
            )
            imgui.dummy((width, 14 * s))
        return changed, out
