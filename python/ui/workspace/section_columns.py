"""Keep section headings and their body in one owned native content column."""

from contextlib import contextmanager
from dataclasses import dataclass
from math import floor

from imgui_bundle import imgui

from mojive.design.native.geometry import Box, Rect, budget_margins
from mojive.ui.workspace.menu_geometry import native_unit


def _native_horizontal_budget(edges, width, height):
    bounded = budget_margins(edges, width, height)
    if edges[0] + edges[2] <= max(0, width):
        return bounded
    # Exhausted widths still need an integer leading edge: proportional fitting
    # alone reintroduces fractions that native ItemSize subsequently truncates.
    left = min(native_unit(bounded[0]), floor(max(0, width)))
    return left, bounded[1], max(0, width) - left, bounded[3]


def section_box(ui, x, y, width, *, native_horizontal=False):
    """Measure the same header box for its painter and its parent content column."""
    style, scale = ui.component_style("section_header"), ui.s
    padding, margin = style.edges("padding"), style.edges("margin", scale)
    height = max(style.get("height", 30), style.line_height() + padding[1] + padding[3])
    rect = Rect(x, y + 4 * scale, width, height * scale + margin[1] + margin[3])
    box = (
        ui.design.primitives.field_box(style, rect, (6, 0, 6, 0), baseline=30)
        if ui.design is not None
        else ui.component_box(
            "section_header", *rect.tuple(), padding=(6 * scale, 0, 6 * scale, 0), observe=False
        )
    )
    if not native_horizontal:
        return box
    # Native tables and ItemSize truncate the next cursor X. Resolve the common
    # column's horizontal spacing on that same grid before either painter runs.
    margins = _native_horizontal_budget(
        (native_unit(box.margin[0]), 0, native_unit(box.margin[2]), 0), width, box.frame.height
    )
    available = max(0, width - margins[0] - margins[2])
    frame = Rect(
        x + margins[0],
        box.frame.y,
        min(available, style.dimension("width", available / scale) * scale),
        box.frame.height,
    )
    padding = _native_horizontal_budget(
        (native_unit(box.padding[0]), box.padding[1], native_unit(box.padding[2]), box.padding[3]),
        frame.width,
        frame.height,
    )
    return Box(
        Rect(x, box.outer.y, frame.width + margins[0] + margins[2], box.outer.height),
        frame,
        frame.inset(*padding),
        padding,
        (margins[0], box.margin[1], margins[2], box.margin[3]),
    )


@dataclass(frozen=True, slots=True)
class _SectionColumn:
    window_id: int
    outer: Rect
    content: Rect
    left: float
    right: float
    style_key: tuple

    def header_bounds(self, x, width):
        return x - self.left, width + self.left + self.right


def current_section_column(ui):
    """A child window has its own column even while its parent's scope is active."""
    window_id = imgui.internal.get_current_window().id_
    return next(
        (column for column in reversed(ui._section_columns) if column.window_id == window_id),
        None,
    )


@contextmanager
def section_column(ui):
    """Inset one native window's body once and restore its horizontal layout on exit."""
    position, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
    style = ui.component_style("section_header")
    style_key = (
        style.get("width", None),
        style.get("widthMode", None),
        *(native_unit(value) for value in style.edges("margin", ui.s)[::2]),
        *(native_unit(value) for value in style.edges("padding", ui.s, (6, 0, 6, 0))[::2]),
    )
    existing = current_section_column(ui)
    if existing is not None and existing.style_key == style_key:
        yield Rect(position.x, position.y, available.x, available.y)
        return
    window = imgui.internal.get_current_window()
    outer = Rect(position.x, position.y, available.x, available.y)
    box = section_box(ui, outer.x, outer.y, outer.width, native_horizontal=True)
    content = Rect(box.content.x, outer.y, box.content.width, outer.height)
    left, right = content.x - outer.x, outer.right - content.right
    rects = (window.work_rect, window.parent_work_rect, window.content_region_rect)
    saved = [(rect.min.x, rect.max.x) for rect in rects]
    if left:
        imgui.indent(left)
    imgui.set_cursor_screen_pos((content.x, content.y))
    # Use the actual native cursor origin for both body layout and header paint.
    actual_x = imgui.get_cursor_screen_pos().x
    content = Rect(actual_x, content.y, max(0, content.right - actual_x), content.height)
    column = _SectionColumn(window.id_, outer, content, actual_x - outer.x, right, style_key)
    for rect in rects:
        rect.min.x = max(rect.min.x, content.x)
        rect.max.x = min(rect.max.x, content.right)
    ui._section_columns.append(column)
    if ui.design is not None:
        ui.design.record(
            "section-column/" + str(window.id_),
            ("section_body",),
            content.tuple(),
            "section_column",
            outer=list(outer.tuple()),
            content=list(content.tuple()),
            insets=[column.left, column.right],
        )
    try:
        yield content
    finally:
        ui._section_columns.pop()
        for rect, (low, high) in zip(rects, saved, strict=True):
            rect.min.x, rect.max.x = low, high
        if left:
            imgui.unindent(left)
        imgui.set_cursor_screen_pos((outer.x, imgui.get_cursor_screen_pos().y))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, 0))
        imgui.dummy((0, 0))
        imgui.pop_style_var()
