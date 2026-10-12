"""Native workspace controls sharing painters, hit bounds and gesture state."""

from types import SimpleNamespace

from imgui_bundle import imgui

from mojive.ui.compound_fields import draw_focus_frame
from mojive.ui.controls import segmented_control
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.text_layout import fit_text, text_line_y
from mojive.ui.theme import THEME

from .design import BODY_SIZE, CONTROL_RADIUS, RAIL_STYLE
from .numeric_fields import NumericEditState as NumericEditState
from .numeric_fields import NumericGroupLayout as NumericGroupLayout
from .numeric_fields import _NumericFields
from .widget_icons import GLYPHS as GLYPHS
from .widget_icons import ICON_GRID as ICON_GRID
from .widget_icons import ICON_ROLES, IconRole, draw_icon, icon_role_canvas
from .widget_icons import ICON_STROKE as ICON_STROKE
from .widget_icons import IconRoleSpec as IconRoleSpec
from .widget_icons import icon_geometry as icon_geometry
from .widget_style import AMBER as AMBER
from .widget_style import (
    CONTROL_THEME,
    DIM,
    INPUT,
    LINE,
    MUTED,
    PANEL,
    SAGE,
    SWITCH_HEIGHT,
    SWITCH_WIDTH,
    TEXT,
    install_fonts,
    install_style,
)
from .widget_style import font_config as font_config


def color(c):
    return imgui.get_color_u32(imgui.ImVec4(*(v / 255 for v in c)))


class UI(_NumericFields):
    def __init__(self, window, *, portable_fonts=False):
        self.window = window
        self.s = window.style_scale
        self.translate = lambda value: value
        self.font_sources, fonts = install_fonts(window, portable=portable_fonts)
        self.body, self.bold, self.mono = fonts
        install_style(self.s)
        self.hits = {}
        self.numeric_edits = {}
        self.value_context = SimpleNamespace(
            style_scale=self.s,
            rail_style=RAIL_STYLE,
            theme=THEME,
            tr=lambda value: self.translate(value),
            input_bindings=DEFAULT_INPUT_BINDINGS,
        )

    @property
    def draw(self):
        return imgui.get_window_draw_list()

    def text(
        self,
        x,
        y,
        text,
        c=TEXT,
        size=BODY_SIZE,
        bold=False,
        mono=False,
        draw=None,
        *,
        max_width=None,
        translate=True,
        native_line=False,
    ):
        """Center cap height in the requested line box, retaining a common text baseline."""
        font = self.mono if mono else self.bold if bold else self.body
        draw_list = draw or self.draw
        imgui.push_font(font, size * self.s)
        painter = ImguiDraw2D(draw_list)
        line_y = y if native_line else text_line_y(painter, y + size * self.s / 2)
        shown = self.translate(str(text)) if translate else str(text)
        if max_width is not None:
            shown = fit_text(painter, shown, max(0, max_width))
        painter.text(
            (x, line_y),
            (*[channel / 255 for channel in c[:3]], c[3] / 255 * imgui.get_style().alpha),
            shown,
            pixel_snap=native_line,
        )
        imgui.pop_font()

    def measure(self, text, size=BODY_SIZE, *, translate=True):
        imgui.push_font(self.body, size * self.s)
        width = imgui.calc_text_size(self.translate(text) if translate else text).x
        imgui.pop_font()
        return width

    def line(self, x, y, w):
        self.draw.add_line((x, y), (x + w, y), color(LINE), self.s)

    def rect(self, x, y, w, h, c, r=CONTROL_RADIUS):
        self.draw.add_rect_filled((x, y), (x + w, y + h), color(c), r * self.s)

    def icon(self, name, x, y, size=16, c=MUTED, draw=None):
        draw_icon(draw or self.draw, name, x, y, size, c, self.s)

    def role_icon(
        self, name, x, y, w=None, h=None, c=MUTED, *, role: IconRole = "toolbar", draw=None
    ):
        """Center the master anchor in a physical slot without equalizing glyph bounds."""
        draw = draw or self.draw
        spec = ICON_ROLES[role]
        w = spec.slot_size * self.s if w is None else w
        h = spec.slot_size * self.s if h is None else h
        fit = min(1.0, max(0.0, min(w, h)) / (spec.canvas_size * self.s))
        if fit <= 0:
            return
        size = icon_role_canvas(role) * fit
        canvas = size * self.s
        left, top = x + (w - canvas) / 2, y + (h - canvas) / 2
        self.icon(name, left, top, size, c, draw)

    def tooltip(self, text, *, translate=True):
        imgui.push_style_var(imgui.StyleVar_.window_padding, (9 * self.s, 7 * self.s))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, CONTROL_RADIUS * self.s)
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 260 * self.s)
        imgui.text_unformatted(self.translate(text) if translate else text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
        imgui.pop_style_var(3)

    def button(
        self,
        key,
        x,
        y,
        w=28,
        h=28,
        *,
        icon=None,
        label="",
        active=False,
        filled=False,
        enabled=True,
        tooltip="",
        text_color=None,
        icon_color=None,
        solid=False,
        icon_size=None,
        icon_role: IconRole | None = None,
        align="center",
        background=True,
        translate=True,
    ):
        s = self.s
        icon_role = icon_role or ("inline" if label else "toolbar")
        w *= s
        h *= s
        imgui.set_cursor_screen_pos((x, y))
        original_alpha = imgui.get_style().alpha
        imgui.begin_disabled(not enabled)
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(key, (w, h), imgui.ButtonFlags_.enable_nav)
        disabled = bool(imgui.get_item_flags() & imgui.ItemFlags_.disabled)
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
        hovered = imgui.is_item_hovered()
        held = imgui.is_item_active()
        self.hits[key] = (x, y, w, h)
        if background and held:
            self.rect(x, y, w, h, (130, 164, 116, 255) if active and solid else (156, 191, 141, 55))
        elif background and active:
            self.rect(x, y, w, h, SAGE if solid else (156, 191, 141, 42 if hovered else 30))
        elif background and hovered:
            self.rect(x, y, w, h, (255, 255, 255, 12))
        elif background and filled:
            self.rect(x, y, w, h, INPUT)
        c = text_color or (
            (30, 40, 27, 255) if active and solid else SAGE if active else TEXT if label else MUTED
        )
        shown = ""
        translated_label = self.translate(label) if translate else label
        label_width = 0
        icon_extent = ICON_ROLES[icon_role].canvas_size if icon_size is None else icon_size
        if label:
            horizontal_padding = icon_extent + 21 if icon else 12 if align == "center" else 20
            imgui.push_font(self.body, BODY_SIZE * s)
            shown = fit_text(
                ImguiDraw2D(),
                translated_label,
                max(0, w - horizontal_padding * s),
            )
            label_width = imgui.calc_text_size(shown).x
            imgui.pop_font()
        content_width = label_width + ((icon_extent + 5) * s if icon and shown else 0)
        start = (
            x + max((8 if icon else 6) * s, (w - content_width) * 0.5)
            if align == "center"
            else x + 8 * s
        )
        glyph_color = icon_color or c
        if disabled:
            # Opaque muted ink keeps intersecting strokes uniform when disabled.
            original_alpha = imgui.get_current_context().disabled_alpha_backup
            opacity = imgui.get_style().alpha / max(original_alpha, 1e-9)

            def muted_ink(tint):
                return (
                    *(
                        round(a * opacity + b * (1 - opacity))
                        for a, b in zip(tint[:3], PANEL[:3], strict=True)
                    ),
                    tint[3],
                )

            c, glyph_color = muted_ink(c), muted_ink(glyph_color)
            imgui.push_style_var(imgui.StyleVar_.alpha, original_alpha)
        # The native hit region and clipping share the same control rectangle.
        self.draw.push_clip_rect((x + s, y), (x + w - s, y + h), True)
        if icon:
            icon_x = start if shown else x + (w - icon_extent * s) / 2
            icon_y = y + (h - icon_extent * s) / 2
            if icon_size is None:
                self.role_icon(
                    icon,
                    icon_x,
                    icon_y,
                    icon_extent * s,
                    icon_extent * s,
                    glyph_color,
                    role=icon_role,
                )
            else:
                self.icon(icon, icon_x, icon_y, icon_size, glyph_color)
        if shown:
            text_x = (
                start + (icon_extent + 5) * s
                if icon
                else start + (0 if align == "center" else 2 * s)
            )
            self.text(text_x, y + (h - BODY_SIZE * s) / 2, shown, c, translate=False)
        self.draw.pop_clip_rect()
        if disabled:
            imgui.pop_style_var()
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + w, y + h),
            rounding=CONTROL_RADIUS * s,
            item_id=item_id,
        )
        hint = tooltip or (translated_label if shown != translated_label else "")
        if hint and imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        ):
            self.tooltip(hint, translate=translate)
        imgui.end_disabled()
        return hit

    def segmented(
        self,
        key,
        x,
        y,
        w,
        labels,
        selected,
        *,
        keys=None,
        icons=None,
        enabled=True,
        layout="equal",
        show_labels=True,
        tooltips=None,
    ):
        """Choose one value in the shared contiguous shell and retain native hit bounds."""
        if keys is not None and len(keys) != len(labels):
            raise ValueError("Each segment requires one hit key")
        if tooltips is not None and len(tooltips) != len(labels):
            raise ValueError("Each segment requires one tooltip")
        if not labels:
            self.hits[key] = (x, y, w * self.s, 0)
            return 0

        def record(index):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            item_key = keys[index] if keys is not None else f"{key}-{index}"
            self.hits[item_key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            if tooltips is not None and imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                self.tooltip(tooltips[index])

        imgui.push_font(self.body, BODY_SIZE * self.s)
        padding_y = max(0, (28 * self.s - imgui.get_font_size()) / 2)
        padding_x = (6 if layout == "content" else 8) * self.s
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (padding_x, padding_y))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, CONTROL_RADIUS * self.s)
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_disabled(not enabled)
        chosen = segmented_control(
            key,
            tuple(self.translate(label) for label in labels),
            selected,
            width=w * self.s,
            theme=CONTROL_THEME,
            icons=icons,
            icon_label_drawer=self._segment_label,
            on_item=record,
            layout=layout,
            show_labels=show_labels,
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        imgui.end_disabled()
        imgui.pop_style_var(2)
        imgui.pop_font()
        return chosen

    def _segment_label(self, draw, lo, hi, c, scale, icon, label):
        """Place native glyphs and captions on the shared segment's visual center."""
        alpha = imgui.get_style().alpha
        rgba = (
            *[round(channel * 255) for channel in c[:3]],
            round(c[3] * 255 / alpha) if alpha else 0,
        )
        role: IconRole = "inline" if label else "toolbar"
        icon_width = ICON_ROLES[role].canvas_size * self.s
        caption_width = imgui.calc_text_size(label).x if label else 0
        content_width = icon_width + (5 * self.s + caption_width if label else 0)
        x, center_y = lo[0] + (hi[0] - lo[0] - content_width) / 2, (lo[1] + hi[1]) / 2
        self.role_icon(icon, x, center_y - icon_width / 2, icon_width, icon_width, rgba, role=role)
        if label:
            self.text(
                x + icon_width + 5 * self.s,
                center_y - BODY_SIZE * self.s / 2,
                label,
                rgba,
                translate=False,
            )

    def section(self, label, x, y, w, open=None):
        self.line(x, y, w)
        hit = False
        s = self.s
        item_id = None
        if open is not None:
            imgui.set_cursor_screen_pos((x, y + 4 * s))
            imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
            hit = imgui.invisible_button(
                "section-" + label, (w, 30 * s), imgui.ButtonFlags_.enable_nav
            )
            imgui.pop_style_color()
            item_id = imgui.get_item_id()
            self.hits["section-" + label] = (x, y + 4 * s, w, 30 * s)
            if imgui.is_item_hovered():
                self.rect(x, y + 4 * s, w, 30 * s, (255, 255, 255, 7))
            self.role_icon(
                "down" if open else "right",
                x + w - 16 * s,
                y + 13 * s,
                12 * s,
                12 * s,
                DIM,
                role="status",
            )
        self.draw.push_clip_rect((x, y + 4 * s), (x + max(0, w - 22 * s), y + 34 * s), True)
        self.text(
            x + 2 * s,
            y + (4 + (30 - BODY_SIZE) / 2) * s,
            label,
            MUTED,
            BODY_SIZE,
            bold=True,
            max_width=max(0, w - 24 * s),
        )
        self.draw.pop_clip_rect()
        if item_id is not None:
            draw_focus_frame(
                imgui.ImVec2(x, y + 4 * s),
                imgui.ImVec2(x + w, y + 34 * s),
                rounding=CONTROL_RADIUS * s,
                item_id=item_id,
            )
        return hit

    def switch(self, key, x, y, on):
        s = self.s
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(
            key, (SWITCH_WIDTH * s, SWITCH_HEIGHT * s), imgui.ButtonFlags_.enable_nav
        )
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
        hovered, held = imgui.is_item_hovered(), imgui.is_item_active()
        tint = (
            (105, 128, 93, 255)
            if on and hovered
            else (88, 108, 78, 255)
            if on
            else ((70, 78, 86, 255) if hovered else (55, 62, 68, 255))
        )
        self.rect(x + 2 * s, y + 5 * s, 32 * s, 18 * s, tint, 9)
        self.draw.add_circle_filled(
            (x + (25 if on else 11) * s, y + 14 * s),
            (6 if held else 7) * s,
            color(SAGE if on else MUTED),
            num_segments=32,
        )
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + SWITCH_WIDTH * s, y + SWITCH_HEIGHT * s),
            rounding=9 * s,
            item_id=item_id,
        )
        self.hits[key] = (x, y, SWITCH_WIDTH * s, SWITCH_HEIGHT * s)
        return not on if hit else on
