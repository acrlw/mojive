"""Explicit component boxes, typography and sibling placement for native panels."""

from contextlib import nullcontext

from imgui_bundle import imgui

from mojive.design.native.geometry import (
    CURRENT_NATIVE_PARTS,
    Box,
    LayoutChild,
    NativePartStyle,
    Rect,
    budget_margins,
    equal_spacing_layout,
    part_scope,
)


class _BaselineStyle:
    """Retain caller geometry when the workspace has no design runtime."""

    def __init__(self, font_size=None):
        self.font_size = font_size

    def get(self, field, fallback=None):
        return self.font_size if field == "fontSize" and self.font_size is not None else fallback

    def explicit(self, field):
        return False

    def has(self, field):
        return False

    def line_height(self, fallback=18):
        return fallback

    def edges(self, prefix, scale=1, fallback=(0, 0, 0, 0)):
        return tuple(v * scale for v in fallback)

    def dimension(self, field, fallback):
        return fallback

    def box(self, x, y, width, height, *, scale=1, padding=(0, 0, 0, 0), intrinsic=None):
        frame = Rect(x, y, max(0, width), max(0, height))
        return Box(frame, frame, frame.inset(*padding), padding, (0, 0, 0, 0))

    def align_rect(self, content, width, height, *, horizontal=None, vertical=None):
        return NativePartStyle.align_rect(
            self, content, width, height, horizontal=horizontal, vertical=vertical
        )


_BASELINE = _BaselineStyle()


class ComponentStyles:
    def component_style(self, part):
        if self.design is not None:
            return self.design.style(part)
        size = self.typography.size_for((part,) if isinstance(part, str) else part)
        return _BASELINE if size is None else _BaselineStyle(size)

    def component_scope(self, *parts):
        return (
            part_scope(*CURRENT_NATIVE_PARTS.get(), *parts)
            if self.design is not None
            else nullcontext()
        )

    def component_box(
        self,
        part,
        x,
        y,
        w,
        h,
        *,
        key=None,
        padding=(0, 0, 0, 0),
        minimum_height=0,
        draw=None,
        background=None,
        observe=True,
    ):
        """Consume a parent's physical budget, including this component's margins."""
        style = self.component_style(part)
        requested_margin = style.edges("margin", self.s)
        margin = budget_margins(requested_margin, w, h)
        available_w = max(0, w - margin[0] - margin[2])
        available_h = max(0, h - margin[1] - margin[3])
        if self.design is None:
            frame = Rect(x, y, max(0, w), max(0, h))
            return Box(frame, frame, frame.inset(*padding), padding, margin)
        proposed = style.box(x, y, available_w, available_h, scale=self.s, padding=padding)
        if style.has("fontSize"):
            font = style.get("fontSize", 14)
            minimum_height = max(
                minimum_height,
                style.line_height(font * 1.4) * self.s + proposed.padding[1] + proposed.padding[3],
            )
        frame = Rect(
            x + margin[0],
            y + margin[1],
            min(available_w, proposed.frame.width),
            min(available_h, max(proposed.frame.height, minimum_height)),
        )
        outer = Rect(
            x, y, frame.width + margin[0] + margin[2], frame.height + margin[1] + margin[3]
        )
        box = Box(outer, frame, frame.inset(*proposed.padding), proposed.padding, margin)
        if observe:
            self.record_component_box(
                part,
                box,
                key=key,
                draw=draw,
                background=background,
                requested_margin=requested_margin,
            )
        return box

    def record_component_box(
        self, part, box, *, key=None, draw=None, background=None, requested_margin=None, **geometry
    ):
        """Observe a measured box inside the clip scope used for its actual painting."""
        if self.design is None:
            return
        if key is None:
            name = part if isinstance(part, str) else "/".join(part)
            key = f"component/{name}/{imgui.get_id(name)}"
        self.design.record(
            key,
            (part,) if isinstance(part, str) else tuple(part),
            box.frame.tuple(),
            "container",
            draw=draw,
            **({"background": background} if background is not None else {}),
            content=list(box.content.tuple()),
            outer=list(box.outer.tuple()),
            padding=list(box.padding),
            margin=list(box.margin),
            requestedMargin=list(
                self.component_style(part).edges("margin", self.s)
                if requested_margin is None
                else requested_margin
            ),
            **geometry,
        )

    def component_text(
        self,
        part,
        rect,
        text,
        c,
        *,
        size=None,
        align=None,
        translate=True,
        draw=None,
        bold=False,
        mono=False,
        font_field="fontSize",
    ):
        """Align one text line within the component's already allocated content box."""
        from mojive.ui.workspace.design import DESIGN

        rect = rect if isinstance(rect, Rect) else Rect(*rect)
        style = self.component_style(part)
        font_size = style.get(font_field, DESIGN.body_size if size is None else size)
        line = min(rect.height, max(font_size, style.line_height(font_size * 1.4)) * self.s)
        horizontal = style.get("textAlign", style.get("align", align or "start"))
        vertical = style.get("alignItems", "center")
        shown = self.translate(text) if translate else text
        text_width = min(
            rect.width,
            self.measure(
                shown, font_size, translate=False, font_resolved=True, mono=mono, bold=bold
            ),
        )
        factor_x = (
            0.5
            if horizontal == "center"
            else 1
            if horizontal in ("right", "end", "flex-end")
            else 0
        )
        factor_y = (
            0.5 if vertical == "center" else 1 if vertical in ("bottom", "end", "flex-end") else 0
        )
        x = rect.x + max(0, rect.width - text_width) * factor_x
        y = rect.y + max(0, rect.height - line) * factor_y + max(0, line - font_size * self.s) / 2
        target = draw or self.draw
        target.push_clip_rect((rect.x, rect.y), (rect.right, rect.bottom), True)
        try:
            with self.component_scope(part):
                self.text(
                    x,
                    y,
                    shown,
                    c,
                    font_size,
                    bold=bold,
                    mono=mono,
                    draw=target,
                    max_width=max(0, rect.right - x),
                    translate=False,
                    font_resolved=True,
                )
        finally:
            target.pop_clip_rect()

    def component_items(
        self,
        part,
        rect,
        sizes,
        *,
        direction="row",
        gap=0,
        align="start",
        align_items="center",
        wrap=False,
    ):
        """Lay out complete child extents with the same measured and painted gap."""
        style = self.component_style(part)
        field = "columnGap" if direction == "row" else "rowGap"
        actual_gap = style.get(field, style.get("gap", gap)) * self.s
        return equal_spacing_layout(
            rect,
            tuple(LayoutChild(*size) for size in sizes),
            actual_gap,
            direction=direction,
            align=style.get("align", align),
            align_items=style.get("alignItems", align_items),
            wrap=wrap,
        )
