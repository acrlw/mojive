"""Draw and operate controls from the same boxes used by the native inspector."""

import math
from contextlib import contextmanager

from mojive.design.groups import CURRENT_GROUPS, control_parts, scoped_parts
from mojive.design.native.geometry import (
    CURRENT_NATIVE_PARTS,
    INHERITED_TEXT_FIELDS,
    PRECISE_TEXT_PARENTS,
    Box,
    LayoutChild,
    NativePartStyle,
    PrimitiveResult,
    Rect,
    SliderGeometry,
    budget_margins,
    equal_spacing_layout,
    scoped_corner_radius,
)
from mojive.design.native.style_cache import prepared_geometry
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.interaction_paint import FieldLayer, ItemPaint, draw_switch
from mojive.ui.numeric_text import fit_numeric_text, numeric_edit_format, numeric_text
from mojive.ui.text_layout import fit_text
from mojive.ui.workspace.design import DESIGN


class NativePrimitives:
    """Use actual ImGui controls for editing and catalogue geometry for every frame."""

    def __init__(self, ui, runtime, *, family=None, imgui=None):
        if imgui is None:
            from imgui_bundle import imgui
        self.imgui = imgui
        self.ui = ui
        self.runtime = runtime
        self.family = family
        self.scale = ui.s
        self.observed = []
        self._styles = {}
        self._style_requests = {}
        self._request_typography = None
        self._style_revision = -1
        self._field_splitters = []

    def style(self, parts, *, authored_defaults=False):
        if self._style_revision != self.runtime.revision:
            self._styles = {
                key: style
                for key, style in self._styles.items()
                if style.compatible(self.runtime.spec)
            }
            self._style_requests.clear()
            self._style_revision = self.runtime.revision
        parts = (parts,) if isinstance(parts, str) else tuple(parts)
        parents = CURRENT_NATIVE_PARTS.get()
        typography = getattr(self.ui, "typography", None)
        if typography is not self._request_typography:
            self._style_requests.clear()
            self._request_typography = typography
        authored_defaults = authored_defaults or self.runtime.authored_defaults
        request = (parts, parents, CURRENT_GROUPS.get(), self.family, authored_defaults)
        cached = self._style_requests.get(request)
        if cached is not None:
            return self._corner_style(cached)
        parts = control_parts(scoped_parts(parts), parents)
        font_size = typography.size_for(parts) if typography is not None else None
        key = (font_size, parts, self.family, authored_defaults, parents)
        cached = self._styles.get(key)
        if cached is not None:
            self._remember_style_request(request, cached)
            return self._corner_style(cached)
        style = NativePartStyle(
            self.runtime, parts, self.family, authored_defaults=authored_defaults
        )
        # Parent boxes own their padding/margins and dimensions. Only inherited
        # typography crosses a parent boundary into an unmodified child.
        generic = {"button", "icon_button"}
        font_origin = style.origins.get("fontSize")
        font_rule = style.catalog.get(font_origin, {}).get("fields", {}).get("fontSize", {})
        inherits_font = font_origin in generic or isinstance(font_rule.get("default"), dict)
        local_fields = {
            field
            for field in INHERITED_TEXT_FIELDS
            if style.explicit(field) and style.origins.get(field) not in generic
        }
        for parent in parents:
            if parent in style.part_ids:
                continue
            overrides = style.spec.get("parts", {}).get(parent, {})
            if not overrides:
                continue
            for field in INHERITED_TEXT_FIELDS:
                value = overrides.get(field)
                if value is None or value == "inherit" or field in local_fields:
                    continue
                if field == "fontSize" and not self._precise_parent(parent) and not inherits_font:
                    continue
                style.values[field], style.origins[field] = value, parent
        layout_engine = getattr(self.runtime, "paint_context", None)
        if layout_engine is not None:
            for field in INHERITED_TEXT_FIELDS:
                rule = (
                    style.catalog.get(style.origins.get(field), {}).get("fields", {}).get(field, {})
                )
                if (
                    style.has(field)
                    and field not in local_fields
                    and (
                        field != "fontSize"
                        or isinstance(rule.get("default"), dict)
                        or any(self._precise_parent(parent) for parent in parents)
                    )
                ):
                    style.values[field] = layout_engine.inherited(field, style.values[field])
        size_origin = style.origins.get("fontSize")
        if font_size is not None and "fontSize" not in style.spec.get("parts", {}).get(
            size_origin, {}
        ):
            style.values["fontSize"] = font_size
            style.origins["fontSize"] = "preferences"
            style._explicit_fields = style._explicit_fields | {"fontSize"}
        self._styles[key] = style.freeze()
        self._remember_style_request(request, style)
        return self._corner_style(style)

    def _remember_style_request(self, request, style):
        if len(self._style_requests) >= 4096:
            self._style_requests.clear()
        self._style_requests[request] = style

    def _corner_style(self, style):
        if "search" in style.part_ids:
            return style
        radius = scoped_corner_radius(style.part_ids)
        return (
            style
            if radius is None
            else style.with_overrides({"radius": (radius / self.ui.s, "parent_corner")})
        )

    def _precise_parent(self, part):
        return part in PRECISE_TEXT_PARENTS or "groupOwner" in self.runtime.catalog.get(part, {})

    def record(self, key, style, box, kind, **geometry):
        self.ui.hits[key] = box.frame.tuple()
        if not self.runtime.observes(style.part_ids):
            return
        get_context = getattr(self.imgui, "get_current_context", None)
        context = get_context() if get_context is not None else None
        window = (
            context.current_window if context is not None and context.within_frame_scope else None
        )
        if window is not None:
            # A submitted control can be below a scroll clip or behind a smaller
            # child. Keep its authored hit geometry and the actual visibility
            # boundary together for the workbench's instance count and picking.
            clip_min, clip_max = self.ui.draw.get_clip_rect_min(), self.ui.draw.get_clip_rect_max()
            geometry.update(
                clip=(clip_min.x, clip_min.y, clip_max.x, clip_max.y),
                windowId=window.id_,
                drawListSource="window",
                submittedFrame=self.imgui.get_frame_count(),
            )
        geometry = {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in geometry.items()
        }
        record = {
            "key": key,
            "parts": style.part_ids,
            "kind": kind,
            "styleFamily": style.family,
            "marginBox": box.outer.tuple(),
            "hit": box.frame.tuple(),
            "content": box.content.tuple(),
            "styleOrigins": style.record_origins,
            "fieldValues": style.record_values,
            **geometry,
        }
        self.observed.append(record)

    @contextmanager
    def native_field(self, style, box, *, mono=False, multiline=False):
        """Native editing gets the same font, line box, padding and clip as display."""
        g, s = self.imgui, self.scale
        font_size = style.get("fontSize", 14.0)
        # The actual native item fills its visible frame. Custom inactive text
        # uses the four independent edges; native editing uses the same left
        # edge and preserves its caret, selection and IME behavior.
        content = box.content
        native_rect = content if multiline else box.frame
        g.set_cursor_screen_pos((native_rect.x, native_rect.y))
        g.set_next_item_width(max(1.0, native_rect.width))
        g.push_font(self.ui.resolve_font(style, mono=mono), font_size * s)
        ink = self.ui.style_color(style, "textColor", self.ui.ink(rgba8("text"), style))
        g.push_style_color(g.Col_.text, tuple(channel / 255 for channel in ink))
        g.push_style_var(
            g.StyleVar_.frame_padding,
            (0, 0)
            if multiline
            else (box.padding[0], max(0.0, (box.frame.height - g.get_font_size()) / 2)),
        )
        g.push_style_var(
            g.StyleVar_.frame_rounding,
            self._field_radius(style, box.frame) * s,
        )
        g.push_style_var(g.StyleVar_.frame_border_size, 0.0)
        for color in (
            g.Col_.frame_bg,
            g.Col_.frame_bg_hovered,
            g.Col_.frame_bg_active,
            g.Col_.nav_cursor,
        ):
            g.push_style_color(color, (0.0, 0.0, 0.0, 0.0))
        draw = self.ui.draw
        draw.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
        try:
            yield content
        finally:
            draw.pop_clip_rect()
            g.pop_style_color(5)
            g.pop_style_var(3)
            g.pop_font()

    def field_height(self, parts, baseline=28.0):
        """Measure the native line and padding before allocating its parent row."""
        style = self.style(parts)
        return self._field_height(style, baseline)

    @staticmethod
    @prepared_geometry
    def _field_height(style, baseline=28.0):
        if style.part_ids == ("checkbox",):
            return style.get("size", 18)
        font = style.get("fontSize", 14.0)
        pad = style.edges("padding")
        line = max(font, style.line_height(font * 1.4))
        if "switch" in style.part_ids:
            line = max(
                line,
                style.get("switchHeight", DESIGN.switch_track_height),
                style.get("switchThumb", DESIGN.switch_thumb_size),
            )
        minimum = math.ceil((line + pad[1] + pad[3]) / 2) * 2
        return max(style.get("height", baseline), minimum)

    def field_box(self, style, rect, padding=(8, 0, 8, 0), *, baseline=28, minimum_height=0):
        """Allocate a readable native line inside the measured physical parent budget."""
        scale = self.scale
        margin = budget_margins(style.edges("margin", scale), rect.width, rect.height)
        available_width = max(0.0, rect.width - margin[0] - margin[2])
        available_height = max(0.0, rect.height - margin[1] - margin[3])
        width = min(available_width, style.dimension("width", available_width / scale) * scale)
        height = min(
            available_height,
            max(minimum_height, self._field_height(style, baseline)) * scale,
        )
        pad = style.edges("padding", scale, padding)
        frame = Rect(rect.x + margin[0], rect.y + margin[1], width, height)
        outer = Rect(rect.x, rect.y, width + margin[0] + margin[2], height + margin[1] + margin[3])
        return Box(outer, frame, frame.inset(*pad), pad, margin)

    def _field_radius(self, style, frame):
        # Search is a fixed shape exception to nested panel corners. Resolve
        # its capsule from the painted frame, including an authored height.
        if "search" in style.part_ids:
            return min(frame.width, frame.height) / (2 * self.scale)
        return style.get("radius", 3.0 if "checkbox" in style.part_ids else 6.0)

    def field_background(self, style, box, color=None, *, draw=None):
        self.ui.rect(
            *box.frame.tuple(),
            color or rgba8("field"),
            r=self._field_radius(style, box.frame),
            draw=draw,
        )

    def focus_frame(self, style, frame, *, item_id=None):
        from mojive.ui.compound_fields import draw_focus_frame

        g = self.imgui
        draw_focus_frame(
            g.ImVec2(frame.x, frame.y),
            g.ImVec2(frame.right, frame.bottom),
            rounding=self._field_radius(style, frame) * self.scale,
            item_id=g.get_item_id() if item_id is None else item_id,
        )

    def _field_paint_visible(self, frame):
        if frame.width <= 0 or frame.height <= 0:
            return False
        # Keep AA fringes and the inset focus stroke near a clip edge. This is
        # native paint clearance, independent of the authored UI scale/radius.
        clearance = 2.0
        draw = self.ui.draw
        lo, hi = draw.get_clip_rect_min(), draw.get_clip_rect_max()
        return (
            frame.right + clearance > lo.x
            and frame.bottom + clearance > lo.y
            and frame.x - clearance < hi.x
            and frame.y - clearance < hi.y
        )

    def slider_geometry(self, rect):
        """Measure the complete range/numeric compound before paint or reflow."""
        s = self.scale
        style = self.style("range_number")
        number = self.style("input_number")
        rail = self.style("range")
        margin = style.edges("margin", s)
        padding = style.edges("padding", s)
        gap = style.get("columnGap", style.get("gap", 8.0)) * s
        frame_width = max(0.0, rect.width - margin[0] - margin[2])
        number_width = min(frame_width, style.get("numberWidth", 70.0) * s)
        number_box = self.field_box(
            number, Rect(0.0, 0.0, number_width, self.field_height("input_number") * s)
        )
        rail_box = rail.box(0.0, 0.0, 0.0, rail.dimension("height", 28.0) * s, scale=s)
        height = max(number_box.outer.height, rail_box.outer.height) + padding[1] + padding[3]
        box = style.box(rect.x, rect.y, frame_width, height, scale=s)
        rail_width = max(
            0.0,
            box.content.width
            - number_box.outer.width
            - gap
            - rail_box.margin[0]
            - rail_box.margin[2],
        )
        children = (
            LayoutChild(rail_width, rail_box.frame.height, margin=rail_box.margin),
            LayoutChild(number_box.frame.width, number_box.frame.height, margin=number_box.margin),
        )
        placements = equal_spacing_layout(
            box.content, children, gap, align_items=style.get("alignItems", "center")
        )
        source_rects = tuple(
            Rect(
                rectangle.x - child_margin[0],
                rectangle.y - child_margin[1],
                rectangle.width,
                rectangle.height,
            )
            for rectangle, child_margin in zip(
                placements, (rail_box.margin, number_box.margin), strict=True
            )
        )
        return SliderGeometry(style, box, placements, source_rects, gap)

    def button(
        self,
        key,
        rect,
        label="",
        icon=None,
        *,
        parts="button",
        active=False,
        filled=False,
        enabled=True,
        tooltip="",
        text_color=None,
        icon_color=None,
        solid=False,
        background=True,
        translate=True,
        icon_size=None,
        icon_role=None,
        radius=None,
        authored_defaults=False,
        padding=None,
        inherit=None,
        align="center",
        width_limit=None,
    ):
        g, ui, s = self.imgui, self.ui, self.scale
        style = self.style(
            parts, authored_defaults=authored_defaults or self.runtime.authored_defaults
        )
        overrides = {
            field: value for field, value in (inherit or {}).items() if not style.explicit(field)
        }
        # Chrome owns the authored slot. A glyph or padding edit must not resize
        # its rail/toolbox/camera slot to the generic toolbar default.
        if "icon_button" in style.part_ids:
            for dimension, fallback in (("width", rect.width / s), ("height", rect.height / s)):
                if not style.explicit(dimension) and not (
                    style.explicit(dimension + "Mode") and style.get(dimension + "Mode") == "fixed"
                ):
                    overrides[dimension] = (fallback, style.origins[dimension])
        style = style.with_overrides(overrides)
        from mojive.design.native.button_layout import button_layout, button_metrics
        from mojive.ui.workspace.widget_icons import icon_role_canvas

        icon_fallback = icon_role_canvas(icon_role or ("inline" if label else "toolbar"))
        corner_radius, font_size, icon_size, gap, line_height, horizontal = button_metrics(
            style, bool(icon), bool(label), icon_fallback, icon_size, radius, align
        )
        layout_engine = getattr(self.runtime, "paint_context", None)
        if icon and not style.explicit("iconSize") and layout_engine is not None:
            icon_size = layout_engine.inherited("iconSize", icon_size)
        gap *= s
        line_height *= s
        shown = ui.translate(str(label)) if translate else str(label)
        text_width = (
            ui.measure(shown, font_size, translate=False, font_resolved=True, style=style)
            if label
            else 0.0
        )
        get_context = getattr(g, "get_current_context", None)
        context = get_context() if get_context is not None else None
        window = (
            context.current_window if context is not None and context.within_frame_scope else None
        )
        limit = max(0.0, width_limit) if width_limit is not None else math.inf
        if window is not None:
            margin = style.edges("margin", s)
            limit = min(limit, max(0.0, window.work_rect.max.x - rect.x - margin[0] - margin[2]))
        box, icon_x, icon_y, text_x, text_y, visible_text_width, icon_extent = button_layout(
            style,
            rect,
            s,
            padding or (0.0, 0.0, 0.0, 0.0),
            text_width,
            line_height,
            icon_size,
            gap,
            horizontal,
            limit,
        )
        text_y += (line_height - font_size * s) / 2
        content = box.content
        g.set_cursor_screen_pos((box.frame.x, box.frame.y))
        if not enabled:
            g.begin_disabled(True)
        # Mouse-driven frames have no navigation cursor to suppress. Avoid a
        # native style-stack round trip for every tree row and toolbar button.
        nav_visible = context is None or context.nav_cursor_visible
        if nav_visible:
            g.push_style_color(g.Col_.nav_cursor, (0.0, 0.0, 0.0, 0.0))
        hit = g.invisible_button(
            key, (max(1.0, box.frame.width), max(1.0, box.frame.height)), g.ButtonFlags_.enable_nav
        )
        if nav_visible:
            g.pop_style_color()
        paint = ItemPaint.capture(g)
        hovered, held = paint.hovered, paint.active
        if background:
            if active:
                self.ui.rect(
                    *box.frame.tuple(),
                    rgba8("primary_bright" if held else "primary_dim" if hovered else "primary")
                    if solid
                    else rgba8("primary", 75 if held else 55 if hovered else 30),
                    r=corner_radius,
                )
            elif held or hovered or filled:
                tint = (
                    paint.color()
                    if filled
                    else rgba8("primary", 55)
                    if held
                    else (255, 255, 255, 12)
                )
                self.ui.rect(*box.frame.tuple(), tint, r=corner_radius)
        c = text_color or (
            rgba8("child")
            if active and solid
            else rgba8("primary")
            if active
            else rgba8("text")
            if label
            else rgba8("secondary")
        )
        glyph = ui.style_color(style, "iconColor", ui.ink(icon_color or c, style))
        c = ui.style_color(style, "textColor", ui.ink(c, style))
        disabled = bool(context.last_item_data.item_flags & g.ItemFlags_.disabled.value)
        if disabled:
            original_alpha = context.disabled_alpha_backup
            opacity = g.get_style().alpha / max(original_alpha, 1e-9)
            panel = ui.ink(rgba8("background"), style)

            def muted_ink(tint):
                return (
                    *(
                        round(a * opacity + b * (1 - opacity))
                        for a, b in zip(tint[:3], panel[:3], strict=True)
                    ),
                    tint[3],
                )

            c, glyph = muted_ink(c), muted_ink(glyph)
            g.push_style_var(g.StyleVar_.alpha, original_alpha)
        ui.draw.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
        if icon:
            ui.icon(icon, icon_x, icon_y, icon_size, glyph, style=style, color_resolved=True)
        if label:
            ui.text(
                text_x,
                text_y,
                shown,
                c,
                font_size,
                translate=False,
                font_resolved=True,
                style=style,
                color_resolved=True,
                max_width=max(0.0, min(visible_text_width, content.right - text_x)),
            )
        ui.draw.pop_clip_rect()
        if disabled:
            g.pop_style_var()
        clipped_label = bool(label) and (
            visible_text_width < text_width
            or text_x < content.x
            or text_x + text_width > content.right
        )
        hint = tooltip or (shown if clipped_label else "")
        if hint and g.is_item_hovered(
            g.HoveredFlags_.delay_normal.value | g.HoveredFlags_.allow_when_disabled.value
        ):
            ui.tooltip(hint, translate=translate if tooltip else False)
        if not enabled:
            g.end_disabled()
        from mojive.ui.compound_fields import draw_focus_frame

        if nav_visible:
            draw_focus_frame(
                g.ImVec2(box.frame.x, box.frame.y),
                g.ImVec2(box.frame.right, box.frame.bottom),
                rounding=corner_radius * s,
                item_id=g.get_item_id(),
            )
        self.ui.hits[key] = box.frame.tuple()
        if self.runtime.observes(style.part_ids):
            self.record(
                key,
                style,
                box,
                "button",
                background=background,
                active=active,
                filled=filled,
                hovered=hovered,
                enabled=enabled,
                fontSize=font_size,
                iconSize=icon_size,
                iconPosition=[icon_x, icon_y],
                textPosition=[text_x, text_y],
                gap=gap,
                iconRect=[icon_x, icon_y, icon_extent, icon_extent] if icon else None,
                textRect=[text_x, text_y, visible_text_width, font_size * s] if label else None,
                label=shown,
                icon=icon,
                clippedLabel=clipped_label,
            )
        return PrimitiveResult(bool(hit), bool(hit), box)

    def input_text(
        self,
        key,
        rect,
        value,
        *,
        multiline=False,
        enabled=True,
        parts=None,
        native_call=None,
        hint="",
        flags=0,
        icon=None,
        mono=False,
        search_tooltip="Search",
        clear_tooltip="Clear search",
    ):
        style = self.style(parts or ("textarea" if multiline else "input_text"))
        if multiline:
            margin = budget_margins(style.edges("margin", self.scale), rect.width, rect.height)
            width = max(0, rect.width - margin[0] - margin[2])
            frame = Rect(
                rect.x + margin[0],
                rect.y + margin[1],
                min(width, style.dimension("width", width / self.scale) * self.scale),
                max(0, rect.height - margin[1] - margin[3]),
            )
            padding = style.edges("padding", self.scale, (10, 10, 10, 10))
            outer = Rect(
                rect.x,
                rect.y,
                frame.width + margin[0] + margin[2],
                frame.height + margin[1] + margin[3],
            )
            box = Box(outer, frame, frame.inset(*padding), padding, margin)
        else:
            box = self.field_box(style, rect)
        with FieldLayer(self.imgui, self.ui.draw, self._field_splitters) as layer:
            g, scale = self.imgui, self.scale
            glyph_size = min(style.get("iconSize", 14) * scale, box.content.height) if icon else 0
            icon_gap = style.get("columnGap", style.get("gap", 8)) * scale if icon else 0
            search = "search" in style.part_ids
            text_box = box
            glyph = None
            if icon:
                glyph = style.align_rect(box.content, glyph_size, glyph_size, horizontal="start")
                text_box = Box(
                    box.outer,
                    box.frame,
                    box.content.inset(glyph_size + icon_gap, 0, 0, 0),
                    (box.padding[0] + glyph_size + icon_gap, *box.padding[1:]),
                    box.margin,
                )
            if search:
                # End-circle centers and a reserved trailing slot stay stable while typing.
                slot = min(box.frame.height, box.frame.width / 2)
                inset = min(box.frame.width / 2, slot / 2 + glyph_size / 2 + icon_gap)
                glyph = Rect(
                    box.frame.x + (slot - glyph_size) / 2,
                    box.frame.y + (box.frame.height - glyph_size) / 2,
                    glyph_size,
                    glyph_size,
                )
                text_box = Box(
                    box.outer,
                    box.frame,
                    box.content.inset(
                        max(0, inset - box.padding[0]), 0, max(0, inset - box.padding[2]), 0
                    ),
                    (inset, box.padding[1], inset, box.padding[3]),
                    box.margin,
                )
            empty_content = text_box.content.width <= 0 or text_box.content.height <= 0
            g.begin_disabled(not enabled or empty_content)
            editor_box = text_box
            if search:
                # Native text input must not own the clear button's hit region.
                editor_box = Box(
                    text_box.outer,
                    Rect(text_box.content.x, box.frame.y, text_box.content.width, box.frame.height),
                    text_box.content,
                    (0, text_box.padding[1], 0, text_box.padding[3]),
                    text_box.margin,
                )
            with self.native_field(style, editor_box, mono=mono, multiline=multiline):
                if native_call is not None:
                    changed, out = native_call(box)
                elif multiline:
                    changed, out = g.input_text_multiline(
                        "##" + key,
                        str(value),
                        (max(1.0, box.content.width), max(1.0, box.content.height)),
                        flags,
                    )
                else:
                    changed, out = g.input_text("##" + key, str(value), flags)
                input_id = g.get_item_id()
                disabled = bool(g.get_item_flags() & g.ItemFlags_.disabled)
                paint = ItemPaint.capture(g)
                active = g.is_item_active()
                result = self._result(out, changed, box)
                native_child = None
                if multiline and self.runtime.observes(style.part_ids):
                    parent, item_id = g.internal.get_current_window(), g.get_item_id()
                    native_child = next(
                        (
                            window.id_
                            for window in g.get_current_context().windows
                            if window.active
                            and window.parent_window == parent
                            and window.child_id == item_id
                        ),
                        None,
                    )
            g.end_disabled()
            # Standard text editing remains native. An inactive display follows the
            # requested alignment; editing always retains caret/selection behavior.
            if not active and not multiline:
                self.field_background(style, box, paint.color())
                self.aligned_text(
                    style,
                    text_box.content,
                    str(out) if out else hint,
                    mono=mono,
                    color=rgba8("text") if out else rgba8("disabled"),
                )
            if glyph is not None:
                self.ui.draw.push_clip_rect(
                    (box.content.x, box.content.y), (box.content.right, box.content.bottom), True
                )
                self.ui.icon(
                    icon, glyph.x, glyph.y, glyph_size / scale, rgba8("secondary"), style=style
                )
                self.ui.draw.pop_clip_rect()
            self.ui.hits[key] = box.frame.tuple()
            if self.runtime.observes(style.part_ids):
                self.record(
                    key,
                    style,
                    box,
                    "textarea" if multiline else "search" if icon else "input_text",
                    fontSize=style.get("fontSize", 14.0),
                    textAlign=style.get("textAlign", "left"),
                    emptyContent=empty_content,
                    nativeMinimumHit=1
                    if (multiline and empty_content)
                    or box.frame.width <= 0
                    or box.frame.height <= 0
                    else None,
                    **({"constraint": "parent budget exhausted"} if empty_content else {}),
                    **({"nativeChildWindowId": native_child} if native_child is not None else {}),
                    **(
                        {
                            "iconSize": glyph_size / scale,
                            "iconRect": list(glyph.tuple()),
                            "columnGap": icon_gap / scale,
                            "textRect": list(text_box.content.tuple()),
                        }
                        if icon
                        else {}
                    ),
                )
            self.focus_frame(style, box.frame, item_id=input_id)
            if search:
                g.begin_disabled(disabled)
                focus_requested = False
                if (
                    not disabled
                    and g.is_window_hovered()
                    and g.is_mouse_hovering_rect(
                        (box.frame.x, box.frame.y), (box.frame.x + slot, box.frame.bottom)
                    )
                ):
                    self.ui.tooltip(search_tooltip)
                    focus_requested = g.is_mouse_clicked(g.MouseButton_.left)
                if out:
                    clear_x = box.frame.right - slot
                    g.set_cursor_screen_pos((clear_x, box.frame.y))
                    clicked = g.invisible_button(
                        "##clear-" + key, (slot, box.frame.height), g.ButtonFlags_.enable_nav
                    )
                    self.ui.icon(
                        "close",
                        clear_x + (slot - glyph_size) / 2,
                        glyph.y,
                        glyph_size / scale,
                        rgba8("secondary"),
                        style=style,
                    )
                    if g.is_item_hovered():
                        self.ui.tooltip(clear_tooltip)
                    if clicked:
                        result.value, result.changed = "", True
                        focus_requested = True
                    self.focus_frame(style, Rect(clear_x, box.frame.y, slot, box.frame.height))
                    self.ui.hits[key + "-clear"] = (clear_x, box.frame.y, slot, box.frame.height)
                if focus_requested:
                    g.internal.activate_item_by_id(input_id)
                    g.get_current_context().nav_next_activate_flags = (
                        g.internal.ActivateFlags_.prefer_input
                    )
                g.set_cursor_screen_pos((box.outer.x, box.outer.y))
                g.dummy((box.outer.width, box.outer.height))
                g.end_disabled()
            layer.background()
            self.field_background(style, box, paint.color(), draw=layer.draw)
            return result

    def _result(self, value, changed, box, *, reset=False):
        g = self.imgui
        activated, active = g.is_item_activated(), g.is_item_active()
        deactivated = g.is_item_deactivated()
        return PrimitiveResult(
            value,
            changed or reset,
            box,
            activated,
            active,
            deactivated,
            g.is_item_deactivated_after_edit() or reset,
            (active or deactivated) and g.is_key_pressed(g.Key.escape, False),
            reset,
        )

    def aligned_text(
        self,
        style,
        content,
        text,
        *,
        mono=False,
        color=rgba8("text"),
        draw=None,
        horizontal=None,
        fit_line=False,
    ):
        s = self.scale
        font_size = style.get("fontSize", 14.0)
        width = self.ui.measure(
            str(text), font_size, translate=False, font_resolved=True, mono=mono, style=style
        )
        line_height = style.line_height(font_size * 1.4) * s
        if fit_line:
            line_height = min(line_height, max(font_size * s, content.height))
        placed = style.align_rect(
            content, width, line_height, horizontal=horizontal or style.get("textAlign", "left")
        )
        draw = draw or self.ui.draw
        draw.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
        layout_engine = getattr(self.runtime, "paint_context", None)
        if layout_engine is not None:
            layout_engine.paint_depth += 1
        try:
            self.ui.text(
                placed.x,
                placed.y + (line_height - font_size * s) / 2,
                text,
                color,
                font_size,
                mono=mono,
                translate=False,
                font_resolved=True,
                max_width=max(0.0, content.right - placed.x),
                draw=draw,
                style=style,
            )
        finally:
            if layout_engine is not None:
                layout_engine.paint_depth -= 1
            draw.pop_clip_rect()
        return placed

    def input_number(
        self,
        key,
        rect,
        value,
        *,
        axis=None,
        speed=0.01,
        lo=0.0,
        hi=0.0,
        format="%.3f",
        reset_value=0.0,
        enabled=True,
        parts=None,
        background=True,
        draw_text=True,
        integer=False,
        flags=0,
        minimum_height=0,
    ):
        style = self.style(parts or (("input_number", "axis_input") if axis else "input_number"))
        box = self.field_box(
            style, rect, (3, 0, 3, 0) if axis else (8, 0, 8, 0), minimum_height=minimum_height
        )
        paint_visible = self._field_paint_visible(box.frame)
        with FieldLayer(self.imgui, self.ui.draw, self._field_splitters) as layer:
            g, ui, s = self.imgui, self.ui, self.scale
            axis_size = style.get("axisWidth", 18.0) * s if axis else 0.0
            badge = min(axis_size, max(0.0, box.frame.width - 1.0))
            reset = False
            empty_frame = box.frame.width <= 0 or box.frame.height <= 0
            g.begin_disabled(not enabled or empty_frame)
            if axis:
                axis_rect = Rect(box.frame.x, box.frame.y, badge, box.frame.height)
                g.set_cursor_screen_pos((axis_rect.x, axis_rect.y))
                reset = g.invisible_button(
                    "##" + key + "-reset",
                    (max(1.0, axis_rect.width), max(1.0, axis_rect.height)),
                    g.ButtonFlags_.enable_nav,
                )
                badge_signals = (g.is_item_activated(), g.is_item_active(), g.is_item_deactivated())
                badge_item_id = g.get_item_id()
                badge_paint = ItemPaint.capture(g)
                if paint_visible:
                    self.aligned_text(style, axis_rect, axis, mono=False, horizontal="center")
                    self.focus_frame(style, axis_rect, item_id=badge_item_id)
                ui.hits[key + "-reset"] = axis_rect.tuple()
            numeric_frame = box.frame.inset(badge, 0, 0, 0)
            content = numeric_frame.inset(*box.padding)
            numeric_box = Box(box.outer, numeric_frame, content, box.padding, box.margin)
            with self.native_field(style, numeric_box, mono=True):
                # DragFloat supplies activation, keyboard editing, cancellation and
                # command-transaction signals, while inactive ink is painted below.
                item_id = g.get_id("###" + key)
                context, io = g.get_current_context(), g.get_io()
                hovered = g.is_window_hovered() and g.is_mouse_hovering_rect(
                    (content.x, content.y), (content.right, content.bottom)
                )
                native_text = (
                    enabled
                    and not bool(context.current_item_flags & g.ItemFlags_.disabled)
                    and (
                        g.internal.temp_input_is_active(item_id)
                        or (
                            hovered
                            and (
                                (g.is_mouse_clicked(0) and io.key_ctrl)
                                or g.is_mouse_double_clicked(0)
                                or (
                                    io.config_drag_click_to_input_text
                                    and context.active_id == item_id
                                    and g.is_mouse_released(0)
                                )
                            )
                        )
                        or (
                            context.nav_activate_id == item_id
                            and bool(
                                context.nav_activate_flags & g.internal.ActivateFlags_.prefer_input
                            )
                        )
                    )
                )
                if not native_text:
                    g.push_style_color(g.Col_.text, (0.0, 0.0, 0.0, 0.0))
                drag = g.drag_int if integer else g.drag_float
                numeric = int if integer else float
                changed, out = drag(
                    "###" + key,
                    numeric(value),
                    speed,
                    numeric(lo),
                    numeric(hi),
                    numeric_edit_format(format) if native_text else format,
                    flags
                    | g.SliderFlags_.no_round_to_format
                    | (g.SliderFlags_.always_clamp if lo < hi else 0),
                )
                if not native_text:
                    g.pop_style_color()
                paint = ItemPaint.capture(g)
                result = self._result(
                    reset_value if reset else out if changed else value, changed, box, reset=reset
                )
                editing = g.internal.temp_input_is_active(g.get_item_id())
                if axis:
                    result.activated |= badge_signals[0]
                    result.active |= badge_signals[1]
                    result.deactivated |= badge_signals[2]
            g.end_disabled()
            shown = None
            if paint_visible and draw_text and not editing and not native_text:
                g.push_font(ui.resolve_font(style, mono=True), style.get("fontSize", 14.0) * s)
                shown = fit_numeric_text(result.value, format, content.width, g.calc_text_size)
                g.pop_font()
                self.aligned_text(
                    style,
                    content,
                    shown,
                    mono=True,
                    horizontal=style.get("textAlign", "center"),
                    fit_line=True,
                )
            if (
                not empty_frame
                and g.is_item_hovered(
                    g.HoveredFlags_.delay_normal | g.HoveredFlags_.allow_when_disabled
                )
                and not editing
            ):
                g.push_font(ui.resolve_font(style, mono=True), style.get("fontSize", 14.0) * s)
                clipped = (
                    (
                        (shown is not None and shown != numeric_text(result.value, format))
                        or g.calc_text_size(shown or numeric_text(result.value, format)).x
                        > content.width
                    )
                    if draw_text
                    else g.calc_text_size(format % result.value).x > content.width
                )
                g.pop_font()
                if clipped:
                    ui.tooltip(format % result.value, translate=False)
            self.ui.hits[key] = box.frame.tuple()
            if self.runtime.observes(style.part_ids):
                self.record(
                    key,
                    style,
                    box,
                    "input_number",
                    fontSize=style.get("fontSize", 14.0),
                    textAlign=style.get("textAlign", "center"),
                    axisWidth=badge,
                    numericRect=numeric_frame.tuple(),
                    emptyContent=content.width <= 0 or content.height <= 0,
                    nativeMinimumHit=1 if empty_frame else None,
                    constraint="parent budget exhausted" if empty_frame else None,
                )
            ui.hits[key + "-frame"] = box.frame.tuple()
            if paint_visible:
                self.focus_frame(style, box.frame)
            if hasattr(ui, "numeric_edits"):
                # Use the existing type so callers continue their atomic command flow.
                from mojive.ui.workspace.widgets import NumericEditState

                ui.numeric_edits[key] = NumericEditState(
                    result.activated,
                    result.active,
                    result.deactivated,
                    result.committed,
                    result.canceled,
                    result.reset,
                )
            layer.background()
            if paint_visible and background:
                self.field_background(style, box, paint.color(), draw=layer.draw)
            if paint_visible and axis:
                from mojive.ui.compound_fields import draw_joined_field_frame
                from mojive.ui.theme import THEME, mix_color

                axis_index = {"X": 0, "R": 0, "Y": 1, "G": 1, "Z": 2, "B": 2}.get(axis)
                tint = (
                    THEME.axis_color(axis_index) if axis_index is not None else THEME.text_disabled
                )
                draw_joined_field_frame(
                    layer.draw,
                    g.ImVec2(axis_rect.x, axis_rect.y),
                    g.ImVec2(axis_rect.right, axis_rect.bottom),
                    g.ImVec2(axis_rect.right, axis_rect.y),
                    g.ImVec2(box.frame.right, box.frame.bottom),
                    badge_color=mix_color(
                        tuple(v / 255 for v in badge_paint.color()),
                        tint,
                        0.88 if badge_paint.active else 0.72 if badge_paint.hovered else 0.56,
                    ),
                    field_color=tuple(v / 255 for v in paint.color()),
                    rounding=style.get("radius", 6.0) * s,
                )
            return result

    def begin_combo(
        self, key, rect, preview, *, enabled=True, parts="select", flags=0, popup_width=0
    ):
        """Open native choices after drawing the catalogue-controlled select field."""
        style = self.style(parts)
        box = self.field_box(style, rect)
        with FieldLayer(self.imgui, self.ui.draw, self._field_splitters) as layer:
            g, ui, s = self.imgui, self.ui, self.scale
            self.aligned_text(
                style, box.content.inset(0, 0, box.frame.height, 0), ui.translate(str(preview))
            )
            glyph = style.align_rect(
                box.content,
                min(style.get("iconSize", 12) * s, box.content.width),
                min(style.get("iconSize", 12) * s, box.content.height),
                horizontal="end",
                vertical="center",
            )
            ui.role_icon(
                "down",
                glyph.x,
                glyph.y,
                glyph.width,
                glyph.height,
                rgba8("secondary"),
                role="inline",
                style=style,
            )
            empty_content = box.content.width <= 0 or box.content.height <= 0
            self.ui.hits[key] = box.frame.tuple()
            if self.runtime.observes(style.part_ids):
                self.record(
                    key,
                    style,
                    box,
                    "select",
                    iconRect=glyph.tuple(),
                    textAlign=style.get("textAlign", "left"),
                    preview=str(preview),
                    emptyContent=empty_content,
                    nativeMinimumHit=1 if box.frame.width <= 0 or box.frame.height <= 0 else None,
                    **({"constraint": "parent budget exhausted"} if empty_content else {}),
                )
            disabled = not enabled or empty_content
            if disabled:
                g.begin_disabled()
            from mojive.ui.workspace.popup_style import (
                menu_choice_width,
                menu_popup_style,
                prepare_menu_popup,
            )

            identifier = key if key.startswith("##") else "##" + key
            item_id = g.get_id(identifier)
            context = g.get_current_context()
            # Match BeginCombo's popup identity. Measure before Begin while its
            # draw list can still accept font changes, and only for an open or
            # immediately reachable mouse/keyboard target.
            popup_id = g.internal.im_hash_str("##ComboPopup", 0, item_id)
            needs_width = not disabled and (
                g.internal.is_popup_open(popup_id, 0)
                or context.nav_activate_id == item_id
                or (
                    (g.is_mouse_clicked(0) or g.is_mouse_released(0))
                    and g.is_mouse_hovering_rect(
                        (box.frame.x, box.frame.y), (box.frame.right, box.frame.bottom)
                    )
                )
            )
            measured_width = (
                (
                    (popup_width() if callable(popup_width) else popup_width)
                    or menu_choice_width(ui, (preview,), 0)
                )
                if needs_width
                else 0
            )
            prepare_menu_popup(
                ui,
                minimum_width=box.frame.width,
                auto_minimum_width=max(box.frame.width, measured_width),
                part="select_popup",
            )
            with menu_popup_style(ui, part="select_popup"), self.native_field(style, box):
                # BeginCombo paints its arrow button after our shared indicator.
                # Keep native hit/navigation behavior without covering authored paint.
                for color in (g.Col_.text, g.Col_.button, g.Col_.button_hovered):
                    g.push_style_color(color, (0, 0, 0, 0))
                # BeginCombo copies FramePadding.x into popup WindowPadding.
                # The popup's authored content box owns that inset independently.
                g.push_style_var(g.StyleVar_.frame_padding, (0, g.get_style().frame_padding.y))
                try:
                    opened = g.begin_combo(identifier, str(preview), flags)
                finally:
                    g.pop_style_var()
                paint = ItemPaint.capture(g, opened=opened)
                g.pop_style_color(3)
                if disabled and opened:
                    g.close_current_popup()
                    g.end_combo()
                    opened = False
            if disabled:
                g.end_disabled()
            layer.background()
            self.field_background(style, box, paint.color(), draw=layer.draw)
            return opened, box

    def select(self, key, rect, value, choices, *, enabled=True, parts="select"):
        from mojive.ui.workspace.popup_style import menu_choice_width

        choices = tuple(choices)
        selected = choices.index(value) if value in choices else -1
        opened, box = self.begin_combo(
            key,
            rect,
            value,
            enabled=enabled,
            parts=parts,
            popup_width=lambda: menu_choice_width(self.ui, choices, selected),
        )
        g, changed, out = self.imgui, False, value
        if opened:
            from mojive.ui.workspace.menu_item import Item
            from mojive.ui.workspace.menu_rows import advance_menu_row, paint_menu_row
            from mojive.ui.workspace.popup_style import menu_popup_content

            with menu_popup_content(self.ui, key, part="select_popup") as content:
                for index, choice in enumerate(choices):
                    row = paint_menu_row(
                        self.ui,
                        Item(str(choice), checked=choice == value),
                        f"{key}/choice/{index}",
                        content_right=content.content.right,
                        part="select_option",
                    )
                    if row.activated:
                        changed, out = True, choice
                    advance_menu_row(self.ui, row, part="select_popup")
            g.end_combo()
        return PrimitiveResult(out, changed, box)

    def checkbox(self, key, x, y, value, *, enabled=True, label="", width=None):
        g, ui, s = self.imgui, self.ui, self.scale
        style = self.style("checkbox")
        requested_extent = style.get("size", 18.0) * s
        extent = min(requested_extent, max(0, width)) if width is not None else requested_extent
        label_style = self.style("form_label") if label else None
        shown = ui.translate(label) if label else ""
        font = label_style.get("fontSize", 14) if label else 14
        label_margin = label_style.edges("margin", s) if label else (0, 0, 0, 0)
        label_width = (
            ui.measure(shown, font, translate=False, font_resolved=True, style=label_style)
            if label
            else 0
        )
        label_offset = extent + 6 * s + label_margin[0]
        if width is not None:
            label_offset = min(max(0, width), label_offset)
            label_width = min(label_width, max(0, width - label_offset - label_margin[2]))
        row_height = (
            max(
                extent,
                ui.metric("form_row", "height", 28) * s,
                max(font, label_style.line_height(font * 1.4)) * s
                + label_margin[1]
                + label_margin[3],
            )
            if label
            else extent
        )
        if label:
            row_height = math.ceil(row_height / (2 * s)) * 2 * s
        box = style.box(x, y + (row_height - extent) / 2, extent, extent, scale=s)
        g.set_cursor_screen_pos((x, y))
        g.begin_disabled(not enabled or extent <= 0)
        hit_width = extent + (
            6 * s + label_width + label_margin[0] + label_margin[2] if label else 0
        )
        if width is not None:
            hit_width = min(hit_width, max(0, width))
        hit = g.invisible_button(
            key,
            (
                max(1.0, hit_width),
                max(1.0, row_height),
            ),
            g.ButtonFlags_.enable_nav,
        )
        out = not value if hit else value
        paint = ItemPaint.capture(g)
        tint = paint.color("primary", "primary_bright", "primary_dim") if out else paint.color()
        ui.rect(*box.frame.tuple(), tint, r=style.get("radius", 3.0))
        if out:
            ui.icon("check", box.frame.x, box.frame.y, extent / s, rgba8("child"))
        result = self._result(out, bool(hit), box)
        g.end_disabled()
        self.focus_frame(style, box.frame)
        label_rect = None
        if label:
            label_rect = Rect(
                x + label_offset,
                y + label_margin[1],
                label_width,
                row_height - label_margin[1] - label_margin[3],
            )
            self.aligned_text(label_style, label_rect, shown, color=rgba8("text"))
        self.ui.hits[key] = box.frame.tuple()
        if self.runtime.observes(style.part_ids):
            self.record(
                key,
                style,
                box,
                "checkbox",
                size=extent,
                requestedSize=requested_extent,
                emptyContent=extent <= 0,
                radius=style.get("radius", 3.0),
                **({"labelRect": list(label_rect.tuple()), "label": shown} if label_rect else {}),
            )
        return result

    def switch(
        self,
        key,
        rect,
        value,
        *,
        label="",
        enabled=True,
        label_style=None,
        layout_box=None,
        tooltip="",
        control_x=None,
    ):
        g, ui, s = self.imgui, self.ui, self.scale
        style = self.style("switch")
        box = layout_box or self.field_box(style, rect, (0, 0, 0, 0))
        width, height = (
            style.get("switchWidth", DESIGN.switch_track_width) * s,
            style.get("switchHeight", DESIGN.switch_track_height) * s,
        )
        thumb = style.get("switchThumb", DESIGN.switch_thumb_size) * s
        label_style = label_style or style
        gap = style.get("columnGap", style.get("gap", 8.0)) * s if label else 0.0
        font = label_style.get("fontSize", 14.0)
        shown = ui.translate(label)
        label_width = (
            ui.measure(shown, font, translate=False, font_resolved=True, style=label_style)
            if label
            else 0.0
        )
        g.set_cursor_screen_pos((box.frame.x, box.frame.y))
        empty_frame = box.frame.width <= 0 or box.frame.height <= 0
        g.begin_disabled(not enabled or empty_frame)
        hit = g.invisible_button(
            key, (max(1.0, box.frame.width), max(1.0, box.frame.height)), g.ButtonFlags_.enable_nav
        )
        out = not value if hit else value
        paint = ItemPaint.capture(g)
        sx, sy = (
            min(max(box.content.x, control_x), max(box.content.x, box.content.right - width))
            if control_x is not None
            else max(box.content.x, box.content.right - width)
            if label
            else box.content.x,
            box.content.y + (box.content.height - height) / 2,
        )
        ui.draw.push_clip_rect(
            (box.content.x, box.content.y), (box.content.right, box.content.bottom), True
        )
        draw_switch(ui, sx, sy, width, height, thumb, out, paint)
        if label:
            label_rect = Rect(
                box.content.x, box.content.y, max(0.0, sx - gap - box.content.x), box.content.height
            )
            label_content = label_rect.inset(*label_style.edges("margin", s))
            self.aligned_text(label_style, label_content, shown)
        if g.is_item_hovered(
            g.HoveredFlags_.delay_normal | g.HoveredFlags_.allow_when_disabled
        ) and (tooltip or (label and label_width > label_content.width)):
            ui.tooltip(tooltip or shown, translate=False)
        ui.draw.pop_clip_rect()
        result = self._result(out, bool(hit), box)
        g.end_disabled()
        self.ui.hits[key] = box.frame.tuple()
        if self.runtime.observes(style.part_ids):
            self.record(
                key,
                style,
                box,
                "switch",
                switchRect=[sx, sy, width, height],
                switchWidth=width,
                switchHeight=height,
                switchThumb=thumb,
                textPresent=bool(label),
                emptyContent=empty_frame,
                constraint="parent budget exhausted" if empty_frame else None,
                gap=gap,
                **(
                    {
                        "labelRect": list(label_rect.tuple()),
                        "label": shown,
                        "fontSize": font,
                        "lineHeight": label_style.line_height(),
                        "labelOrigins": {
                            field: label_style.origins.get(field) for field in INHERITED_TEXT_FIELDS
                        },
                    }
                    if label
                    else {}
                ),
            )
        self.focus_frame(style, box.frame)
        return result

    def range(self, key, rect, value, lo, hi, *, enabled=True):
        g, ui, s = self.imgui, self.ui, self.scale
        style = self.style("range")
        box = style.box(*rect.tuple(), scale=s)
        g.set_cursor_screen_pos((box.frame.x, box.frame.y))
        g.begin_disabled(not enabled)
        g.invisible_button(
            key, (max(1.0, box.frame.width), max(1.0, box.frame.height)), g.ButtonFlags_.enable_nav
        )
        paint = ItemPaint.capture(g)
        active = g.is_item_active()
        out = value
        if active and hi > lo:
            mouse = g.get_io().mouse_pos
            out = lo + min(
                1.0, max(0.0, (mouse.x - box.content.x) / max(1.0, box.content.width))
            ) * (hi - lo)
        if g.is_item_focused():
            step = (hi - lo) / 100
            if g.is_key_pressed(g.Key.left_arrow, True):
                out = max(lo, out - step)
            if g.is_key_pressed(g.Key.right_arrow, True):
                out = min(hi, out + step)
        track, thumb = style.get("trackSize", 3.0) * s, style.get("thumbSize", 12.0) * s
        cy = box.content.y + box.content.height / 2
        ratio = min(1.0, max(0.0, (out - lo) / (hi - lo))) if hi > lo else 0.0
        ui.rect(
            box.content.x,
            cy - track / 2,
            box.content.width,
            track,
            rgba8("field_hovered"),
            r=track / (2 * s),
        )
        ui.rect(
            box.content.x,
            cy - track / 2,
            box.content.width * ratio,
            track,
            rgba8("primary"),
            r=track / (2 * s),
        )
        from mojive.ui.workspace.widgets import color

        ui.draw.add_circle_filled(
            (box.content.x + box.content.width * ratio, cy),
            thumb / 2,
            color(paint.color("primary", "primary_bright", "text")),
            num_segments=32,
        )
        result = self._result(out, out != value, box)
        g.end_disabled()
        self.ui.hits[key] = box.frame.tuple()
        if self.runtime.observes(style.part_ids):
            self.record(key, style, box, "range", trackSize=track, thumbSize=thumb)
        self.focus_frame(style, box.frame)
        return result

    def label(
        self,
        key,
        rect,
        text,
        *,
        parts="form_label",
        color=rgba8("secondary"),
        inherit=None,
        compound_parts=(),
        owner=None,
        label_gap=None,
        gap_origin=None,
        unit="",
    ):
        style = self.style(parts)
        style = style.with_overrides(
            {field: value for field, value in (inherit or {}).items() if not style.explicit(field)}
        )
        box = self.ui.component_box(style.part_ids, *rect.tuple(), key=key + "-container")
        shown = self.ui.translate(str(text))
        if unit:
            heading = self.style("vector_heading")
            gap = heading.get("columnGap", heading.get("gap", 6)) * self.scale
            font = style.get("fontSize", 14)
            label_width = self.ui.measure(
                shown, font, translate=False, font_resolved=True, style=style
            )
            suffix = f"({unit})"
            unit_width = self.ui.measure(
                suffix, font, translate=False, font_resolved=True, style=style
            )
            group = style.align_rect(
                box.content,
                min(box.content.width, label_width + gap + unit_width),
                box.content.height,
                horizontal=style.get("textAlign", "left"),
            )
            self.aligned_text(
                style,
                Rect(group.x, group.y, min(group.width, label_width), group.height),
                shown,
                color=color,
                horizontal="start",
                fit_line=True,
            )
            left = min(group.right, group.x + label_width + gap)
            self.aligned_text(
                style,
                Rect(left, group.y, max(0, group.right - left), group.height),
                suffix,
                color=color,
                horizontal="start",
                fit_line=True,
            )
            if self.runtime.observes(scoped_parts(("vector_heading",))):
                self.ui.design.record(
                    key + "/unit",
                    ("vector_heading",),
                    box.content.tuple(),
                    "vector_heading",
                    gap=gap,
                    fieldRegions={
                        name: [left - gap, group.y, gap, group.height]
                        for name in ("gap", "columnGap")
                    },
                    unitRect=[left, group.y, max(0, group.right - left), group.height],
                )
        else:
            self.aligned_text(style, box.content, shown, color=color, fit_line=True)
        self.ui.hits[key] = box.frame.tuple()
        if self.runtime.observes(style.part_ids):
            self.record(
                key,
                style,
                box,
                "label",
                fontSize=style.get("fontSize", 14.0),
                lineHeight=style.line_height(),
                textAlign=style.get("textAlign", "left"),
                **(
                    {"compoundParts": list(compound_parts), "owner": owner}
                    if compound_parts
                    else {}
                ),
                **(
                    {"labelGap": label_gap, "gapOrigin": gap_origin}
                    if label_gap is not None
                    else {}
                ),
            )
        return box

    def shell_numeric_height(self, parts="input_number", *, unit=True):
        child, shell = self.style(parts), self.style("input_shell", authored_defaults=True)
        height = self._field_height(child)
        if unit:
            font = child.get("fontSize", 14)
            padding = child.edges("padding")
            unit_line = shell.get("unitSize", font) * 1.4
            height = max(height, math.ceil((unit_line + padding[1] + padding[3]) / 2) * 2)
        return height

    def shell_height(self, parts="input_number", *, unit=True):
        shell, child = self.style("input_shell", authored_defaults=True), self.style(parts)
        pad, margin, child_margin = (
            shell.edges("padding"),
            shell.edges("margin"),
            child.edges("margin"),
        )
        minimum = self.shell_numeric_height(parts, unit=unit) + sum(
            (child_margin[1], child_margin[3], pad[1], pad[3])
        )
        return max(shell.get("height", 28), minimum) + margin[1] + margin[3]

    def number_shell_box(self, rect, unit="", *, parts="input_number"):
        """Measure a single field silhouette including shell and numeric margins."""
        s = self.scale
        style = self.style("input_shell", authored_defaults=True)
        margin = budget_margins(style.edges("margin", s), rect.width, math.inf)
        height = max(
            rect.height - margin[1] - margin[3],
            self.shell_height(parts, unit=bool(unit)) * s - margin[1] - margin[3],
        )
        width = max(0, rect.width - margin[0] - margin[2])
        proposed = style.box(rect.x, rect.y, width, height, scale=s)
        frame = Rect(
            rect.x + margin[0],
            rect.y + margin[1],
            min(width, proposed.frame.width),
            max(height, proposed.frame.height),
        )
        outer = Rect(
            rect.x,
            rect.y,
            frame.width + margin[0] + margin[2],
            frame.height + margin[1] + margin[3],
        )
        return Box(outer, frame, frame.inset(*proposed.padding), proposed.padding, margin)

    def number_shell(
        self, key, rect, value, unit="", *, toggle_unit=None, compact_caption=False, **kwargs
    ):
        """Lay out one value/unit caption without changing the native edit transaction."""
        with FieldLayer(self.imgui, self.ui.draw, self._field_splitters) as layer:
            s, g = self.scale, self.imgui
            style = self.style("input_shell", authored_defaults=True)
            parts = kwargs.get("parts") or (
                ("input_number", "axis_input") if kwargs.get("axis") else "input_number"
            )
            numeric = self.style(parts)
            box = self.number_shell_box(rect, unit, parts=parts)
            numeric_rect = Rect(
                box.content.x,
                box.content.y,
                box.content.width,
                box.content.height,
            )
            minimum_height = self.shell_numeric_height(parts, unit=bool(unit))
            numeric_box = self.field_box(numeric, numeric_rect, minimum_height=minimum_height)
            content = numeric_box.content
            font_size = numeric.get("fontSize", 14.0)
            unit_size = style.get("unitSize", font_size)
            number_format = kwargs.get("format", "%.3f")
            suffix = self.ui.translate(unit)
            g.push_font(self.ui.resolve_font(numeric, mono=True), unit_size * s)
            gap = (
                (
                    style.get("columnGap", 0) * s
                    if style.explicit("columnGap") or style.explicit("gap")
                    else g.calc_text_size(" ").x
                )
                if unit
                else 0.0
            )
            suffix_width = g.calc_text_size(suffix).x if unit else 0.0
            g.pop_font()
            value_width = max(0, content.width - gap - suffix_width)
            line_height = min(
                max(numeric.line_height(font_size * 1.4), unit_size * 1.4) * s,
                max(font_size * s, content.height),
            )

            def caption_layout(number):
                g.push_font(self.ui.resolve_font(numeric, mono=True), font_size * s)
                shown = fit_numeric_text(number, number_format, value_width, g.calc_text_size)
                shown = fit_text(ImguiDraw2D(self.ui.draw), shown, value_width)
                text_width = g.calc_text_size(shown).x
                g.pop_font()
                number_width = text_width if compact_caption else value_width
                caption = numeric.align_rect(
                    content,
                    number_width + gap + suffix_width,
                    line_height,
                    horizontal=numeric.get("textAlign", "center"),
                    vertical=style.get("alignItems", "center"),
                )
                suffix_rect = Rect(
                    caption.x + number_width + gap, caption.y, suffix_width, line_height
                )
                return shown, text_width, caption, suffix_rect

            shown, text_width, caption, suffix_rect = caption_layout(value)
            unit_paint = ItemPaint()
            if unit and toggle_unit:
                g.set_cursor_screen_pos((suffix_rect.x - 4 * s, numeric_box.frame.y))
                if g.invisible_button(
                    "##" + key + "-unit", (suffix_width + 8 * s, numeric_box.frame.height)
                ):
                    toggle_unit()
                unit_paint = ItemPaint.capture(g)
                if g.is_item_hovered():
                    self.ui.tooltip("° / rad", translate=False)
            result = self.input_number(
                key,
                numeric_rect,
                value,
                background=False,
                draw_text=False,
                minimum_height=minimum_height,
                **kwargs,
            )
            paint = ItemPaint.capture(g).combine(unit_paint)
            if result.value != value:
                shown, text_width, caption, suffix_rect = caption_layout(result.value)
            value_x = caption.x + (0 if compact_caption else max(0, value_width - text_width))
            # Native text editing owns its caret and selection. The inactive caption
            # shares one baseline and one alignment budget across value and unit.
            if not g.internal.temp_input_is_active(g.get_item_id()):
                content = result.box.content
                self.ui.draw.push_clip_rect(
                    (content.x, content.y), (content.right, content.bottom), True
                )
                self.ui.text(
                    value_x,
                    caption.y + (line_height - font_size * s) / 2,
                    shown,
                    rgba8("text"),
                    font_size,
                    mono=True,
                    translate=False,
                    font_resolved=True,
                    max_width=value_width,
                    style=numeric,
                )
                if unit:
                    self.ui.text(
                        suffix_rect.x,
                        caption.y + (line_height - unit_size * s) / 2,
                        suffix,
                        rgba8("text"),
                        unit_size,
                        mono=True,
                        translate=False,
                        font_resolved=True,
                        max_width=max(0, content.right - suffix_rect.x),
                        style=numeric,
                    )
                self.ui.draw.pop_clip_rect()
            self.ui.hits[key + "-unit"] = (
                (
                    suffix_rect.x - 4 * s,
                    result.box.frame.y,
                    suffix_width + 8 * s,
                    result.box.frame.height,
                )
                if unit
                else (result.box.frame.right, result.box.frame.y, 0, result.box.frame.height)
            )
            self.ui.hits[key + "-shell"] = box.frame.tuple()
            if self.runtime.observes(style.part_ids):
                self.record(
                    key + "-shell",
                    style,
                    box,
                    "input_shell",
                    unitSize=unit_size,
                    gap=gap,
                    columnGap=gap / s,
                    unit=unit,
                    numericFormat=kwargs.get("format", "%.3f"),
                    numericRect=list(result.box.frame.tuple()),
                    unitRect=list(suffix_rect.tuple()),
                    textRect=[value_x, caption.y, text_width, line_height],
                    textPosition=[value_x, caption.y],
                    valueTextPosition=[value_x, caption.y + (line_height - font_size * s) / 2],
                    unitTextPosition=[suffix_rect.x, caption.y + (line_height - unit_size * s) / 2],
                    fontSize=font_size,
                    readonly=False,
                )
            self.imgui.set_cursor_screen_pos((box.outer.x, box.outer.y))
            self.imgui.internal.item_size((box.outer.width, box.outer.height))
            layer.background()
            self.field_background(style, box, paint.color(), draw=layer.draw)
            return result

    def form_layout(self, rect, count=1, *, vector=False, axis=False):
        """Resolve label/control rows and columns from one catalogue-driven grid."""
        s = self.scale
        row = self.style(("form_row", "vector_row") if vector else "form_row")
        box = row.box(*rect.tuple(), scale=s)
        group = self.style("vector_group")
        field_style = self.style(("input_number", "axis_input") if axis else "input_number")
        label_width = min(row.get("labelWidth", 120.0) * s, box.content.width)
        label_gap = row.get("columnGap", row.get("labelGap", 12.0)) * s
        above = vector and row.get("layout", "row") in ("top", "column", "stack")
        if above:
            label_gap = row.get("rowGap", row.get("labelGap", 12.0)) * s
        label_height = self.style("form_label").line_height() * s
        if above:
            label = Rect(box.content.x, box.content.y, box.content.width, label_height)
            fields_rect = box.content.inset(0.0, label_height + label_gap, 0.0, 0.0)
        else:
            label = Rect(box.content.x, box.content.y, label_width, box.content.height)
            fields_rect = box.content.inset(label_width + label_gap, 0.0, 0.0, 0.0)
        columns = max(1, min(count, int(group.get("columns", count))))
        gap = group.get("columnGap", group.get("gap", 4.0)) * s
        row_gap = group.get("rowGap", group.get("gap", 4.0)) * s
        group_margin = group.edges("margin", s)
        fields_rect = fields_rect.inset(*group_margin)
        width = max(0.0, (fields_rect.width - (columns - 1) * gap) / columns)
        field_height = self.field_height(field_style.part_ids) * s
        rows = math.ceil(count / columns)
        height = rows * field_height + (rows - 1) * row_gap
        available_cross = max(height, fields_rect.height)
        valign = row.get("alignItems", "center")
        top = fields_rect.y + max(0.0, available_cross - height) * (
            0.5 if valign == "center" else 1.0 if valign == "end" else 0.0
        )
        fields = tuple(
            Rect(
                fields_rect.x + n % columns * (width + gap),
                top + n // columns * (field_height + row_gap),
                width,
                field_height,
            )
            for n in range(count)
        )
        return {
            "box": box,
            "label": label,
            "fields": fields,
            "columns": columns,
            "height": max(box.outer.height, height + (label_height + label_gap if above else 0.0)),
            "labelAbove": above,
            "gap": gap,
        }
