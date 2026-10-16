"""Native workspace controls sharing painters, hit bounds and gesture state."""

from types import SimpleNamespace

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.compound_fields import draw_focus_frame
from mojive.ui.controls import segmented_control
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.interaction_paint import ItemPaint, draw_switch
from mojive.ui.text_layout import fit_text, text_line_y
from mojive.ui.theme import THEME
from mojive.ui.workspace.component_style import ComponentStyles
from mojive.ui.workspace.design import (
    BODY_SIZE,
    CONTROL_RADIUS,
    DESIGN,
    MONO_SIZE,
    RAIL_STYLE,
    TITLE_SIZE,
)
from mojive.ui.workspace.numeric_fields import NumericEditState as NumericEditState
from mojive.ui.workspace.numeric_fields import NumericGroupLayout as NumericGroupLayout
from mojive.ui.workspace.numeric_fields import _NumericFields
from mojive.ui.workspace.typography import Typography
from mojive.ui.workspace.widget_icons import GLYPHS as GLYPHS
from mojive.ui.workspace.widget_icons import ICON_GRID as ICON_GRID
from mojive.ui.workspace.widget_icons import ICON_ROLES, IconRole, draw_icon, icon_role_canvas
from mojive.ui.workspace.widget_icons import ICON_STROKE as ICON_STROKE
from mojive.ui.workspace.widget_icons import IconRoleSpec as IconRoleSpec
from mojive.ui.workspace.widget_icons import icon_geometry as icon_geometry
from mojive.ui.workspace.widget_style import (
    ACCENT,
    CONTROL_THEME,
    DIM,
    INPUT,
    LINE,
    MUTED,
    PANEL,
    SWITCH_HEIGHT,
    SWITCH_WIDTH,
    TEXT,
    install_fonts,
    install_style,
)
from mojive.ui.workspace.widget_style import AMBER as AMBER
from mojive.ui.workspace.widget_style import SAGE as SAGE
from mojive.ui.workspace.widget_style import font_config as font_config


def color(c):
    return imgui.get_color_u32(imgui.ImVec4(*(v / 255 for v in c)))


class UI(ComponentStyles, _NumericFields):
    design = None
    typography = Typography()

    def __init__(self, window, *, portable_fonts=False):
        self.window = window
        self.s = window.style_scale
        self.translate = lambda value: value
        self.font_sources, fonts = install_fonts(window, portable=portable_fonts)
        self.body, self.bold, self.mono = fonts
        install_style(self.s)
        self.hits = {}
        self.numeric_edits = {}
        self.design = None
        self.value_context = SimpleNamespace(
            style_scale=self.s,
            rail_style=RAIL_STYLE,
            theme=THEME,
            tr=lambda value: self.translate(value),
            input_bindings=DEFAULT_INPUT_BINDINGS,
        )

    @property
    def numeric_font(self):
        return self.mono if self.typography.numeric.family == "mono" else self.body

    @property
    def output_font(self):
        return self.mono if self.typography.output.family == "mono" else self.body

    @property
    def draw(self):
        return imgui.get_window_draw_list()

    def metric(self, part, field, fallback):
        if field == "fontSize":
            size = self.typography.size_for((part,))
            if size is not None:
                return size
        if self.design is None:
            return fallback
        return self.design.runtime.metric(part, field, fallback, family=self.design.family())

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
        font_resolved=False,
    ):
        """Center cap height in the requested line box, retaining a common text baseline."""
        if self.design is not None:
            if not font_resolved and size == BODY_SIZE:
                size = self.design.text_size(
                    self.design.runtime.value(
                        "mono_font_size" if mono else "body_font_size", self.design.family()
                    )
                )
            elif not font_resolved and size == TITLE_SIZE:
                size = self.design.runtime.value("title_font_size", self.design.family())
            c = self.design.runtime.ink(c)
        font = self.numeric_font if mono else self.bold if bold else self.body
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

    def measure(
        self, text, size=BODY_SIZE, *, translate=True, font_resolved=False, mono=False, bold=False
    ):
        if not font_resolved and self.design is not None and size == BODY_SIZE:
            size = self.design.runtime.value(
                "mono_font_size" if mono else "body_font_size", self.design.family()
            )
        elif not font_resolved and self.design is not None and size == TITLE_SIZE:
            size = self.design.runtime.value("title_font_size", self.design.family())
        imgui.push_font(
            self.numeric_font if mono else self.bold if bold else self.body, size * self.s
        )
        width = imgui.calc_text_size(self.translate(text) if translate else text).x
        imgui.pop_font()
        return width

    def line(self, x, y, w):
        thickness = self.metric("separator", "thickness", 1) * self.s
        inset = min(w / 2, self.metric("separator", "margin", 0) * self.s)
        self.draw.add_line((x + inset, y), (x + w - inset, y), color(LINE), thickness)
        if self.design is not None:
            self.design.record(
                f"separator/{x:g}/{y:g}",
                ("separator",),
                (x + inset, y, max(0, w - 2 * inset), thickness),
                "separator",
            )

    def rect(self, x, y, w, h, c, r=CONTROL_RADIUS, *, draw=None):
        if w <= 0 or h <= 0:
            return
        draw = self.draw if draw is None else draw
        if self.design is not None:
            c = self.design.runtime.ink(c)
            if r == CONTROL_RADIUS and not self.design.runtime.paint_context.paint_depth:
                r = DESIGN.control_radius
            if self.design.runtime.spec["native"]["g3"] and r > 0:
                rgba = (
                    *tuple(channel / 255 for channel in c[:3]),
                    c[3] / 255 * imgui.get_style().alpha,
                )
                if r * self.s >= min(w, h) / 2 - 1e-6:
                    from mojive.geometry2d.curves import smooth_capsule_points

                    ImguiDraw2D(draw).convex_fill(
                        smooth_capsule_points(
                            x, y, w, h, self.design.runtime.spec["native"]["g3_capsule_smoothing"]
                        ),
                        rgba,
                    )
                    return
                ImguiDraw2D(draw).rect_filled(
                    (x, y),
                    (x + w, y + h),
                    rgba,
                    rounding=r * self.s,
                    smoothing=self.design.runtime.spec["native"]["g3_smoothing"],
                )
                return
        draw.add_rect_filled((x, y), (x + w, y + h), color(c), r * self.s)

    def icon(self, name, x, y, size=16, c=MUTED, draw=None):
        if self.design is not None:
            c = self.design.runtime.ink(c)
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
        width_limit=None,
    ):
        s = self.s
        icon_role = icon_role or ("inline" if label else "toolbar")
        if self.design is not None:
            return self.design.button(
                key,
                x,
                y,
                w,
                h,
                icon=icon,
                label=label,
                active=active,
                filled=filled,
                enabled=enabled,
                tooltip=tooltip,
                text_color=text_color,
                icon_color=icon_color,
                solid=solid,
                icon_size=icon_size,
                icon_role=icon_role,
                align=align,
                background=background,
                translate=translate,
                width_limit=width_limit,
            )
        w *= s
        if width_limit is not None:
            w = min(w, max(0, width_limit))
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
            self.rect(
                x, y, w, h, rgba8("primary_dim") if active and solid else rgba8("primary", 55)
            )
        elif background and active:
            self.rect(x, y, w, h, ACCENT if solid else rgba8("primary", 42 if hovered else 30))
        elif background and hovered:
            self.rect(x, y, w, h, (255, 255, 255, 12))
        elif background and filled:
            self.rect(x, y, w, h, INPUT)
        c = text_color or (
            rgba8("child") if active and solid else ACCENT if active else TEXT if label else MUTED
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

    def input_text(
        self,
        key,
        x,
        y,
        w,
        value,
        *,
        flags=0,
        hint="",
        enabled=True,
        parts="input_text",
        icon=None,
        mono=False,
    ):
        """Use the same field painter for settings, inspectors and source text."""
        if self.design is not None:
            painter = self.design.primitives
            painter.family = self.design.family()
            style = self.component_style(parts)
            margin = style.edges("margin", self.s)
            height = (
                painter.field_height(parts, DESIGN.control_height) * self.s + margin[1] + margin[3]
            )
            result = painter.input_text(
                key,
                Rect(x, y, w, height),
                value,
                flags=flags,
                hint=hint,
                enabled=enabled,
                parts=parts,
                icon=icon,
                mono=mono,
            )
            return result.changed, result.value
        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(max(1, w))
        if mono:
            imgui.push_font(self.numeric_font, (self.typography.numeric.size or MONO_SIZE) * self.s)
        changed, out = imgui.input_text(key, value, flags)
        if mono:
            imgui.pop_font()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return changed, out

    def search(self, key, x, y, w, value, *, hint=""):
        if self.design is not None:
            return self.input_text(
                key, x, y, w, value, hint=hint, parts=("input_text", "search"), icon="search"
            )
        from mojive.ui.controls import search_input

        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(w)
        result = search_input(key, value, hint=hint)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return result

    def begin_combo(self, key, x, y, w, preview, *, flags=0, enabled=True, popup_width=0):
        if self.design is not None:
            painter = self.design.primitives
            painter.family = self.design.family()
            style = self.component_style("select")
            margin = style.edges("margin", self.s)
            height = (
                painter.field_height("select", DESIGN.control_height) * self.s
                + margin[1]
                + margin[3]
            )
            opened, _ = painter.begin_combo(
                key,
                Rect(x, y, w, height),
                preview,
                flags=flags,
                enabled=enabled,
                popup_width=popup_width,
            )
            return opened
        imgui.set_cursor_screen_pos((x, y))
        imgui.set_next_item_width(max(1, w))
        if not enabled:
            imgui.begin_disabled()
        opened = imgui.begin_combo(key if key.startswith("##") else "##" + key, preview, flags)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        if not enabled:
            if opened:
                imgui.close_current_popup()
                imgui.end_combo()
                opened = False
            imgui.end_disabled()
        return opened

    def combo(self, key, x, y, w, index, labels, *, flags=0, preview=None):
        selected = max(0, min(index, len(labels) - 1)) if labels else 0
        popup_width = 0
        if self.design is not None:
            from mojive.ui.workspace.popup_style import menu_choice_width

            popup_width = menu_choice_width(self, labels, index)
        opened = self.begin_combo(
            key,
            x,
            y,
            w,
            (labels[selected] if labels else "") if preview is None else preview,
            flags=flags,
            popup_width=popup_width,
        )
        chosen = index
        if opened:
            if self.design is not None:
                from mojive.ui.workspace.menu_item import Item
                from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row
                from mojive.ui.workspace.popup_style import menu_popup_content

                with menu_popup_content(self, key, part="select_popup") as content:
                    for n, label in enumerate(labels):
                        row = paint_menu_row(
                            self,
                            Item(str(label), checked=n == index),
                            f"{key}/choice/{n}",
                            content_right=content.content.right,
                            part="select_option",
                        )
                        if row.activated:
                            chosen = n
                        advance_menu_row(self, row, part="select_popup")
            else:
                for n, label in enumerate(labels):
                    if imgui.selectable(self.translate(str(label)), n == index)[0]:
                        chosen = n
            imgui.end_combo()
        return chosen != index, chosen

    def checkbox(self, key, x, y, value, *, enabled=True, label="", width=None):
        if self.design is not None:
            painter = self.design.primitives
            painter.family = self.design.family()
            result = painter.checkbox(key, x, y, value, enabled=enabled, label=label, width=width)
            return result.changed, result.value
        imgui.set_cursor_screen_pos((x, y))
        padding_y = max(0, (DESIGN.checkbox_size * self.s - imgui.get_font_size()) / 2)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (2 * self.s, padding_y))
        imgui.begin_disabled(not enabled)
        changed, value = imgui.checkbox(label + "###" + key if label else key, value)
        imgui.end_disabled()
        imgui.pop_style_var()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return changed, value

    def segment_height(self):
        style = self.component_style("segment")
        margin, padding = style.edges("margin", self.s), style.edges("padding", self.s)
        return (
            max(DESIGN.control_height, DESIGN.body_size) * self.s
            + margin[1]
            + margin[3]
            + padding[1]
            + padding[3]
        )

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

        style = self.component_style("segment")
        font_size = DESIGN.body_size
        height = max(DESIGN.control_height, font_size)
        radius = DESIGN.control_radius
        margin, padding = style.edges("margin", self.s), style.edges("padding", self.s)
        box = self.component_box(
            "segment",
            x,
            y,
            w * self.s,
            height * self.s + margin[1] + margin[3] + padding[1] + padding[3],
            key=key + "-container",
            observe=False,
        )
        imgui.push_font(self.body, font_size * self.s)
        padding_y = max(0, (height * self.s - imgui.get_font_size()) / 2)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding, ((6 if layout == "content" else 8) * self.s, padding_y)
        )
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, radius * self.s)
        content_width = box.content.width
        left = box.content.x
        if layout == "content" and style.explicit("align"):
            from mojive.ui.controls import segmented_control_width

            natural = segmented_control_width(
                tuple(self.translate(label) for label in labels),
                icons=icons,
                show_labels=show_labels,
                layout=layout,
            )
            if natural < content_width:
                factor = (
                    0.5
                    if style.get("align") == "center"
                    else 1
                    if style.get("align") == "end"
                    else 0
                )
                left += (content_width - natural) * factor
                content_width = natural
        imgui.set_cursor_screen_pos((box.outer.x, box.outer.y))
        imgui.begin_group()
        imgui.set_cursor_screen_pos((left, box.content.y))
        imgui.begin_disabled(not enabled)
        chosen = segmented_control(
            key,
            tuple(self.translate(label) for label in labels),
            selected,
            width=content_width,
            theme=CONTROL_THEME,
            icons=icons,
            icon_label_drawer=self._segment_label,
            on_item=record,
            layout=layout,
            show_labels=show_labels,
            gap=style.get("columnGap", 0) * self.s,
        )
        self.record_component_box(
            "segment",
            box,
            key=key + "-container",
            layout=layout,
            fields=[
                list(self.hits[keys[i] if keys is not None else f"{key}-{i}"])
                for i in range(len(labels))
            ],
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        imgui.end_disabled()
        bottom = hi.y + box.padding[3] + box.margin[3]
        imgui.set_cursor_screen_pos((box.outer.x, box.outer.y))
        imgui.dummy((box.outer.width, max(self.s, bottom - box.outer.y)))
        imgui.end_group()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        if self.design is not None:
            self.design.record(
                key,
                ("segment",),
                self.hits[key],
                "segment",
                layout=layout,
                content=list(box.content.tuple()),
                padding=list(box.padding),
                margin=list(box.margin),
            )
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
                center_y - imgui.get_font_size() / 2,
                label,
                rgba,
                imgui.get_font_size() / self.s,
                translate=False,
                font_resolved=True,
            )

    def section_offset(self, baseline):
        style = self.component_style("section_header")
        margin = style.edges("margin")
        padding = style.edges("padding")
        height = max(style.get("height", 30), style.line_height() + padding[1] + padding[3])
        return baseline + height - 30 + margin[1] + margin[3]

    def section(self, label, x, y, w, open=None):
        """Paint every section from the same measured header and text box."""
        s, style = self.s, self.component_style("section_header")
        padding = style.edges("padding")
        height = max(style.get("height", 30), style.line_height() + padding[1] + padding[3])
        margin = style.edges("margin", s)
        box = (
            self.design.primitives.field_box(
                style,
                Rect(x, y + 4 * s, w, height * s + margin[1] + margin[3]),
                (2, 0, 2, 0),
                baseline=30,
            )
            if self.design is not None
            else self.component_box(
                "section_header",
                x,
                y + 4 * s,
                w,
                height * s + margin[1] + margin[3],
                key="section-" + label + "-container",
                padding=(2 * s, 0, 2 * s, 0),
            )
        )
        self.line(box.frame.x, y, box.frame.width)
        hit, item_id = False, None
        if open is not None:
            imgui.set_cursor_screen_pos((box.frame.x, box.frame.y))
            imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
            hit = imgui.invisible_button(
                "section-" + label,
                (max(s, box.frame.width), max(s, box.frame.height)),
                imgui.ButtonFlags_.enable_nav,
            )
            imgui.pop_style_color()
            item_id = imgui.get_item_id()
            self.hits["section-" + label] = box.frame.tuple()
            if imgui.is_item_hovered():
                self.rect(*box.frame.tuple(), (255, 255, 255, 7))
        icon_size = (
            min(box.content.height / s, style.get("iconSize", 12)) if open is not None else 0
        )
        gap = style.get("columnGap", 8) * s if open is not None else 0
        label_rect = box.content.inset(0, 0, icon_size * s + gap, 0)
        font = style.get("fontSize", DESIGN.body_size)
        self.component_text(
            "section_header",
            label_rect,
            label,
            MUTED,
            size=font,
            bold=True,
        )
        if open is not None:
            icon_rect = style.align_rect(
                box.content,
                icon_size * s,
                icon_size * s,
                horizontal="end",
                vertical=style.get("alignItems", "center"),
            )
            self.role_icon(
                "down" if open else "right",
                icon_rect.x,
                icon_rect.y,
                icon_rect.width,
                icon_rect.height,
                DIM,
                role="status",
            )
        if self.design is not None:
            self.design.record(
                "section-" + label,
                ("section_header",),
                box.frame.tuple(),
                "section_header",
                content=list(box.content.tuple()),
                outer=list(box.outer.tuple()),
                labelRect=list(label_rect.tuple()),
                iconRect=list(icon_rect.tuple()) if open is not None else None,
                gap=gap,
                fontSize=font,
                iconSize=icon_size,
                margin=list(margin),
                padding=list(box.padding),
            )
        if item_id is not None:
            draw_focus_frame(
                imgui.ImVec2(box.frame.x, box.frame.y),
                imgui.ImVec2(box.frame.right, box.frame.bottom),
                rounding=CONTROL_RADIUS * s,
                item_id=item_id,
            )
        return hit

    def switch(self, key, x, y, on, *, width=None):
        s = self.s
        if self.design is not None:
            return self.design.switch(key, x, y, on, width=width)
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        hit = imgui.invisible_button(
            key, (SWITCH_WIDTH * s, SWITCH_HEIGHT * s), imgui.ButtonFlags_.enable_nav
        )
        imgui.pop_style_color()
        item_id = imgui.get_item_id()
        paint = ItemPaint.capture(imgui)
        track_width, track_height = DESIGN.switch_track_width, DESIGN.switch_track_height
        draw_switch(
            self,
            x,
            y + (SWITCH_HEIGHT - track_height) * s / 2,
            track_width * s,
            track_height * s,
            DESIGN.switch_thumb_size * s,
            not on if hit else on,
            paint,
        )
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + SWITCH_WIDTH * s, y + SWITCH_HEIGHT * s),
            rounding=track_height * s / 2,
            item_id=item_id,
        )
        self.hits[key] = (x, y, SWITCH_WIDTH * s, SWITCH_HEIGHT * s)
        return not on if hit else on
